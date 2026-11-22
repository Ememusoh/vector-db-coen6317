"""Synthetic categorical attributes with controlled selectivity.

One integer label per base vector. A few "target" labels hit exact
selectivities (50%, 10%, 1%, 0.1% by default); the rest of the rows get
filler labels so the column looks like a realistic category.

Two schemes:
  independent - labels assigned by random permutation, unrelated to vectors
  correlated  - rows ordered by vector region (k-means cluster, clusters
                ordered along the main principal direction), then labels
                assigned in contiguous chunks, so a label covers a compact
                region of the space
"""
from __future__ import annotations

import numpy as np

DEFAULT_SELECTIVITIES = (0.5, 0.1, 0.01, 0.001)
SCHEMES = ("independent", "correlated")


def label_sizes(n: int, selectivities=DEFAULT_SELECTIVITIES, filler_size: float = 0.01):
    """Return (sizes, target_label_ids). Target labels come first: label i has
    selectivity selectivities[i]. Remaining rows are split into filler labels of
    about filler_size each."""
    sizes = [max(1, round(s * n)) for s in selectivities]
    rest = n - sum(sizes)
    if rest < 0:
        raise ValueError("selectivities sum to more than 1")
    per = max(1, round(filler_size * n))
    while rest > 0:
        take = min(per, rest)
        sizes.append(take)
        rest -= take
    return np.array(sizes, dtype=np.int64), np.arange(len(selectivities))


def _labels_from_order(order: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    labels = np.empty(order.size, dtype=np.int32)
    labels[order] = np.repeat(np.arange(sizes.size, dtype=np.int32), sizes)
    return labels


def independent_labels(n: int, seed: int = 0, **kw) -> np.ndarray:
    sizes, _ = label_sizes(n, **kw)
    order = np.random.default_rng(seed).permutation(n)
    return _labels_from_order(order, sizes)


def correlated_labels(base: np.ndarray, seed: int = 0, n_clusters: int = 1024,
                      sample: int = 200_000, **kw) -> np.ndarray:
    import faiss

    n, d = base.shape
    rng = np.random.default_rng(seed)
    train = base[np.sort(rng.choice(n, min(sample, n), replace=False))]
    km = faiss.Kmeans(d, n_clusters, niter=20, seed=seed, verbose=False)
    km.train(np.ascontiguousarray(train, dtype=np.float32))
    assign = _assign_chunked(km.index, base)

    # Order clusters along the top principal direction of the centroids so
    # neighbouring chunks of labels are also neighbours in space.
    c = km.centroids - km.centroids.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    cluster_rank = np.argsort(np.argsort(c @ vt[0]))
    key = cluster_rank[assign].astype(np.float64) + rng.random(n) * 0.5  # shuffle inside cluster
    order = np.argsort(key, kind="stable")

    sizes, _ = label_sizes(n, **kw)
    return _labels_from_order(order, sizes)


def _assign_chunked(index, x: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.int64)
    for s in range(0, x.shape[0], chunk):
        _, a = index.search(np.ascontiguousarray(x[s:s + chunk], dtype=np.float32), 1)
        out[s:s + chunk] = a[:, 0]
    return out


def make_labels(scheme: str, base: np.ndarray, seed: int = 0, **kw) -> np.ndarray:
    if scheme == "independent":
        return independent_labels(base.shape[0], seed=seed, **kw)
    if scheme == "correlated":
        return correlated_labels(base, seed=seed, **kw)
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
