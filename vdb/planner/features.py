"""Query features for the planner.

Global selectivity comes from the attribute index. Local selectivity comes from
a coarse histogram: the space is cut into a few hundred k-means cells and we
store how many vectors of each label fall in each cell. For a query we look at
its nearest cells and measure how common the label is there. If the label is
rare near the query but common overall (correlated labels, query outside the
label's region), index-based plans will struggle; the ratio exposes that.
"""
from __future__ import annotations

import numpy as np

from .fast import coarse_features

FEATURE_NAMES = ("log_sel", "log_nelig", "log_local4", "log_local16", "log_ratio4", "log_ratio16")


class CoarseStats:
    def __init__(self, centroids: np.ndarray, counts: np.ndarray):
        self.centroids = np.ascontiguousarray(centroids, np.float32)  # (C, d)
        self.cnorm = (self.centroids.astype(np.float64) ** 2).sum(1)
        self.counts = np.ascontiguousarray(counts, np.float64)        # (C, L)
        self.sizes = self.counts.sum(1)
        self.label_totals = self.counts.sum(0)
        self.n = float(self.sizes.sum())

    @staticmethod
    def build_centroids(base: np.ndarray, n_cells: int = 256, sample: int = 100_000, seed: int = 0):
        import faiss

        rng = np.random.default_rng(seed)
        x = base[np.sort(rng.choice(base.shape[0], min(sample, base.shape[0]), replace=False))]
        km = faiss.Kmeans(base.shape[1], n_cells, niter=20, seed=seed, verbose=False)
        km.train(np.ascontiguousarray(x, np.float32))
        return km.centroids

    @staticmethod
    def assign_cells(base: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        import faiss

        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(np.ascontiguousarray(centroids, np.float32))
        cell = np.empty(base.shape[0], np.int64)
        for s in range(0, base.shape[0], 200_000):
            _, a = index.search(np.ascontiguousarray(base[s:s + 200_000], np.float32), 1)
            cell[s:s + 200_000] = a[:, 0]
        return cell

    @classmethod
    def build(cls, base: np.ndarray, labels: np.ndarray, centroids: np.ndarray | None = None,
              n_cells: int = 256, seed: int = 0) -> "CoarseStats":
        if centroids is None:
            centroids = cls.build_centroids(base, n_cells, seed=seed)
        cell = cls.assign_cells(base, centroids)
        L = int(labels.max()) + 1
        counts = np.bincount(cell * L + labels, minlength=centroids.shape[0] * L).reshape(-1, L)
        return cls(centroids, counts)

    @classmethod
    def from_postings(cls, base: np.ndarray, order: np.ndarray, offsets: np.ndarray,
                      centroids: np.ndarray) -> "CoarseStats":
        """Multi-label attributes: per-cell counts of every predicate's posting list."""
        cell = cls.assign_cells(base, centroids)
        m = offsets.size - 1
        p_of = np.repeat(np.arange(m), np.diff(offsets))
        counts = np.bincount(cell[order] * m + p_of, minlength=centroids.shape[0] * m).reshape(-1, m)
        st = cls(centroids, counts)
        st.sizes = np.bincount(cell, minlength=centroids.shape[0]).astype(np.float64)  # all vectors per cell
        st.n = float(base.shape[0])
        return st

    @classmethod
    def for_scheme(cls, dataset: str, scheme: str, base: np.ndarray, centroids: np.ndarray) -> "CoarseStats":
        from ..data import datasets

        d = datasets.DATA_ROOT / dataset / scheme
        if (d / "postings.npz").exists():
            z = np.load(d / "postings.npz")
            return cls.from_postings(base, z["order"], z["offsets"], centroids)
        return cls.build(base, np.load(d / "labels.npy"), centroids)

    def features(self, q: np.ndarray, label: int) -> np.ndarray:
        return coarse_features(np.ascontiguousarray(q, np.float32), self.centroids, self.cnorm, self.counts,
                               self.sizes, self.label_totals, self.n, int(label))

    def features_batch(self, Q: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.stack([self.features(Q[i], labels[i]) for i in range(Q.shape[0])])
