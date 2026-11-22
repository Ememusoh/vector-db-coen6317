"""HNSW: graph built by FAISS, extracted to flat arrays, searched by our own kernel."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import kernels
from .base import NO_LABELS, SearchResult, as_query


class HNSWIndex:
    def __init__(self, base, neighbors, offsets, cum, levels, entry, max_level, build_info=None):
        self.base = np.ascontiguousarray(base, np.float32)
        self.neighbors = np.ascontiguousarray(neighbors, np.int32)  # all adjacency lists, -1 padded
        self.offsets = np.ascontiguousarray(offsets, np.int64)      # (N+1,) start of node i's block
        self.cum = np.ascontiguousarray(cum, np.int64)              # level l occupies [cum[l], cum[l+1]) of a block
        self.levels = np.ascontiguousarray(levels, np.int32)        # number of levels per node
        self.entry = int(entry)
        self.max_level = int(max_level)
        self.build_info = build_info or {}
        self._visited = np.zeros(self.base.shape[0], np.int32)
        self._tag = 0

    @classmethod
    def build(cls, base: np.ndarray, M: int = 32, ef_construction: int = 200) -> "HNSWIndex":
        """FAISS assigns node levels with a fixed internal seed, so builds are repeatable."""
        import faiss

        t0 = time.perf_counter()
        base = np.ascontiguousarray(base, np.float32)
        index = faiss.IndexHNSWFlat(base.shape[1], M)
        index.hnsw.efConstruction = ef_construction
        index.add(base)
        h = index.hnsw
        arrays = dict(
            neighbors=faiss.vector_to_array(h.neighbors).astype(np.int32),
            offsets=faiss.vector_to_array(h.offsets).astype(np.int64),
            cum=faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64),
            levels=faiss.vector_to_array(h.levels).astype(np.int32),
        )
        info = {"type": "hnsw", "M": M, "ef_construction": ef_construction,
                "build_seconds": time.perf_counter() - t0}
        return cls(base, entry=h.entry_point, max_level=h.max_level, build_info=info, **arrays)

    def _next_tag(self):
        self._tag += 1
        if self._tag == np.iinfo(np.int32).max:
            self._visited[:] = 0
            self._tag = 1
        return self._tag

    def search(self, q, k: int, ef: int, label: int = -1, labels=None, max_ndis: int = 0) -> SearchResult:
        if label >= 0 and labels is None:
            raise ValueError("filtered search needs the labels array")
        ef = max(ef, k)
        ids, d, ndis = kernels.hnsw_search(as_query(q), self.base, self.neighbors, self.offsets, self.cum,
                                           self.entry, self.max_level,
                                           NO_LABELS if labels is None else labels, label,
                                           ef, k, self._visited, self._next_tag(), max_ndis)
        return SearchResult(ids, d, int(ndis))

    def memory_bytes(self, include_vectors: bool = True) -> int:
        graph = sum(a.nbytes for a in (self.neighbors, self.offsets, self.cum, self.levels))
        return graph + (self.base.nbytes if include_vectors else 0)

    def save(self, path):
        path = Path(path)
        np.savez(path, neighbors=self.neighbors, offsets=self.offsets, cum=self.cum, levels=self.levels,
                 entry=self.entry, max_level=self.max_level)
        path.with_suffix(".json").write_text(json.dumps(self.build_info, indent=2))

    @classmethod
    def load(cls, path, base: np.ndarray) -> "HNSWIndex":
        path = Path(path)
        z = np.load(path)
        info = json.loads(path.with_suffix(".json").read_text())
        return cls(base, z["neighbors"], z["offsets"], z["cum"], z["levels"],
                   int(z["entry"]), int(z["max_level"]), info)
