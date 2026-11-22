"""Attribute-partitioned IVF: inside each inverted list, entries are grouped by
label, so a filtered search jumps directly to the matching slice."""
from __future__ import annotations

import time

import numpy as np

from . import kernels
from .base import SearchResult, as_query
from .ivf import IVFIndex


class PartitionedIVF:
    def __init__(self, centroids, part_offsets, ids, vecs, build_info=None):
        self.centroids = centroids
        self.part_offsets = np.ascontiguousarray(part_offsets, np.int64)  # (nlist, L+1)
        self.ids = np.ascontiguousarray(ids, np.int64)
        self.vecs = np.ascontiguousarray(vecs, np.float32)
        self.build_info = build_info or {}

    @classmethod
    def from_ivf(cls, ivf: IVFIndex, labels: np.ndarray) -> "PartitionedIVF":
        t0 = time.perf_counter()
        nlist = ivf.nlist
        L = int(labels.max()) + 1
        lab = labels[ivf.ids].astype(np.int64)
        list_of = np.repeat(np.arange(nlist), np.diff(ivf.offsets))
        order = np.lexsort((lab, list_of))  # by list, then label
        counts = np.bincount(list_of * L + lab, minlength=nlist * L).reshape(nlist, L)
        part = np.zeros((nlist, L + 1), np.int64)
        part[:, 1:] = np.cumsum(counts, axis=1)
        part += ivf.offsets[:-1, None]
        info = {**ivf.build_info, "type": "ivf_partitioned", "n_labels": L,
                "partition_seconds": time.perf_counter() - t0}
        return cls(ivf.centroids, part, ivf.ids[order], ivf.vecs[order], info)

    @property
    def nlist(self):
        return self.centroids.shape[0]

    def search(self, q, k: int, nprobe: int, label: int = -1) -> SearchResult:
        ids, d, ndis = kernels.ivf_partitioned_search(as_query(q), self.centroids, self.part_offsets,
                                                      self.vecs, self.ids, label, nprobe, k)
        return SearchResult(ids, d, int(ndis))

    def memory_bytes(self) -> int:
        return sum(a.nbytes for a in (self.centroids, self.part_offsets, self.ids, self.vecs))
