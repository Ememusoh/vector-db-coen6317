"""IVF-Flat: k-means (FAISS) for the coarse quantizer, our own list storage and search."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import kernels
from .base import NO_LABELS, SearchResult, as_query


class IVFIndex:
    def __init__(self, centroids, offsets, ids, vecs, build_info=None):
        self.centroids = np.ascontiguousarray(centroids, np.float32)  # (nlist, d)
        self.offsets = np.ascontiguousarray(offsets, np.int64)        # (nlist+1,)
        self.ids = np.ascontiguousarray(ids, np.int64)                # (N,) base id at each list position
        self.vecs = np.ascontiguousarray(vecs, np.float32)            # (N, d) vectors in list order
        self.build_info = build_info or {}

    @property
    def nlist(self):
        return self.centroids.shape[0]

    @classmethod
    def build(cls, base: np.ndarray, nlist: int, train_size: int | None = None,
              niter: int = 20, seed: int = 0) -> "IVFIndex":
        import faiss

        t0 = time.perf_counter()
        n, d = base.shape
        train_size = train_size or min(n, max(64 * nlist, 100_000))
        rng = np.random.default_rng(seed)
        sample = base[np.sort(rng.choice(n, min(train_size, n), replace=False))]
        km = faiss.Kmeans(d, nlist, niter=niter, seed=seed, verbose=False)
        km.train(np.ascontiguousarray(sample, np.float32))

        assign = np.empty(n, np.int64)
        for s in range(0, n, 200_000):
            _, a = km.index.search(np.ascontiguousarray(base[s:s + 200_000], np.float32), 1)
            assign[s:s + 200_000] = a[:, 0]
        ids = np.argsort(assign, kind="stable")
        offsets = np.zeros(nlist + 1, np.int64)
        offsets[1:] = np.cumsum(np.bincount(assign, minlength=nlist))
        info = {"type": "ivf", "nlist": nlist, "train_size": int(sample.shape[0]),
                "niter": niter, "seed": seed, "build_seconds": time.perf_counter() - t0}
        return cls(km.centroids, offsets, ids, base[ids], info)

    def list_labels(self, labels: np.ndarray) -> np.ndarray:
        """Labels in list order, for in-search filtering."""
        return np.ascontiguousarray(labels[self.ids], np.int32)

    def search(self, q, k: int, nprobe: int, label: int = -1, list_labels=None) -> SearchResult:
        if label >= 0 and list_labels is None:
            raise ValueError("filtered search needs list_labels (see IVFIndex.list_labels)")
        ids, d, ndis = kernels.ivf_search(as_query(q), self.centroids, self.offsets, self.vecs, self.ids,
                                          NO_LABELS if list_labels is None else list_labels,
                                          label, nprobe, k)
        return SearchResult(ids, d, int(ndis))

    def memory_bytes(self) -> int:
        return sum(a.nbytes for a in (self.centroids, self.offsets, self.ids, self.vecs))

    # Vectors are not stored on disk; they are re-gathered from base at load time.
    def save(self, path):
        path = Path(path)
        np.savez(path, centroids=self.centroids, offsets=self.offsets, ids=self.ids)
        path.with_suffix(".json").write_text(json.dumps(self.build_info, indent=2))

    @classmethod
    def load(cls, path, base: np.ndarray) -> "IVFIndex":
        path = Path(path)
        z = np.load(path)
        info = json.loads(path.with_suffix(".json").read_text())
        return cls(z["centroids"], z["offsets"], z["ids"], base[z["ids"]], info)
