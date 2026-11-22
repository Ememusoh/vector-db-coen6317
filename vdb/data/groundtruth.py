"""Exact filtered k-NN ground truth.

For each workload entry (query q, label l) we compute the exact top-k among
base vectors whose label equals l. Entries with fewer than k eligible vectors
are padded with id -1 / distance +inf and should be reported separately.
"""
from __future__ import annotations

import numpy as np


def filtered_knn(base: np.ndarray, queries: np.ndarray, labels: np.ndarray,
                 query_ids: np.ndarray, query_labels: np.ndarray, k: int = 100,
                 chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ids, dists) of shape (W, k), squared L2 distances, ids into base."""
    import faiss

    W = query_ids.size
    out_ids = np.full((W, k), -1, dtype=np.int64)
    out_d = np.full((W, k), np.inf, dtype=np.float32)
    d = base.shape[1]

    for lab in np.unique(query_labels):
        rows = np.flatnonzero(query_labels == lab)
        members = np.flatnonzero(labels == lab)
        if members.size == 0:
            continue
        kk = min(k, members.size)
        index = faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(base[members], dtype=np.float32))
        for s in range(0, rows.size, chunk):
            r = rows[s:s + chunk]
            q = np.ascontiguousarray(queries[query_ids[r]], dtype=np.float32)
            dist, local = index.search(q, kk)
            out_ids[r, :kk] = members[local]
            out_d[r, :kk] = dist
    return out_ids, out_d


def brute_force_filtered(base, query, labels, label, k):
    """Slow reference implementation used in tests."""
    members = np.flatnonzero(labels == label)
    dist = ((base[members] - query) ** 2).sum(1)
    order = np.argsort(dist, kind="stable")[:k]
    return members[order], dist[order]
