"""Numba kernels for the planner's per-query path (features + decision), so
planning costs a few microseconds instead of tens."""
from __future__ import annotations

import numpy as np
from numba import njit

from ..index.kernels import maxheap_push

N_FEATURES = 6


@njit(fastmath=True, cache=True)
def coarse_features(q, centroids, cnorm, counts, sizes, label_totals, n, label):
    C = centroids.shape[0]
    hd = np.empty(16, np.float64)
    hi = np.empty(16, np.int64)
    hs = 0
    for c in range(C):
        dot = 0.0
        for j in range(q.shape[0]):
            dot += centroids[c, j] * q[j]
        hs = maxheap_push(hd, hi, hs, 16, cnorm[c] - 2.0 * dot, c)
    order = np.argsort(hd[:hs])
    L = counts.shape[1]
    tot = label_totals[label] if 0 <= label < L else 0.0
    tot = max(tot, 0.5)
    sel = tot / n
    lc4 = 0.5
    sz4 = 1.0
    lc16 = 0.5
    sz16 = 1.0
    for t in range(hs):
        c = hi[order[t]]
        lc = counts[c, label] if 0 <= label < L else 0.0
        lc16 += lc
        sz16 += sizes[c]
        if t < 4:
            lc4 += lc
            sz4 += sizes[c]
    l4 = lc4 / sz4
    l16 = lc16 / sz16
    out = np.empty(N_FEATURES, np.float64)
    out[0] = np.log10(sel)
    out[1] = np.log10(tot)
    out[2] = np.log10(l4)
    out[3] = np.log10(l16)
    out[4] = np.log10(l4 / sel)
    out[5] = np.log10(l16 / sel)
    return out


@njit(fastmath=True, cache=True)
def decide(x, mu, sd, Wr, br, Wl, bl, bucket_edges, taus):
    """Index of the fastest config with P(high recall) >= tau[bucket(x)], else the most likely."""
    F = x.shape[0]
    z = np.empty(F)
    for f in range(F):
        z[f] = (x[f] - mu[f]) / sd[f]
    b = np.searchsorted(bucket_edges, x[0])
    tau = taus[b]
    best, best_lat, best_p, best_p_idx = -1, np.inf, -1.0, 0
    for c in range(Wr.shape[0]):
        sr = br[c]
        sl = bl[c]
        for f in range(F):
            sr += Wr[c, f] * z[f]
            sl += Wl[c, f] * z[f]
        p = 1.0 / (1.0 + np.exp(-sr))
        if p >= tau and sl < best_lat:
            best, best_lat = c, sl
        if p > best_p:
            best_p, best_p_idx = p, c
    return best if best >= 0 else best_p_idx


@njit(fastmath=True, cache=True)
def decide_lagrangian(x, mu, sd, Wr, br, Wl, bl, bucket_edges, lams):
    """Index of argmin_c latency_c(x) - lam[bucket(x)] * recall_c(x), from the linear models."""
    F = x.shape[0]
    z = np.empty(F)
    for f in range(F):
        z[f] = (x[f] - mu[f]) / sd[f]
    lam = lams[np.searchsorted(bucket_edges, x[0])]
    best, best_score = 0, np.inf
    for c in range(Wr.shape[0]):
        sr = br[c]
        sl = bl[c]
        for f in range(F):
            sr += Wr[c, f] * z[f]
            sl += Wl[c, f] * z[f]
        score = np.exp(sl) - lam / (1.0 + np.exp(-sr))
        if score < best_score:
            best, best_score = c, score
    return best
