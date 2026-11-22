"""Numba search kernels shared by all indexes.

Every kernel counts the distance computations it performs (`ndis`) so that
all plans are compared on the same footing.
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF = np.float32(np.inf)


@njit(inline="always", fastmath=True)
def l2sq(a, b):
    s = np.float32(0.0)
    for i in range(a.shape[0]):
        t = a[i] - b[i]
        s += t * t
    return s


# ---- bounded max-heap: keeps the `cap` smallest distances seen so far ----

@njit(inline="always")
def maxheap_push(hd, hi, size, cap, d, idx):
    if size < cap:
        j = size
        size += 1
        while j > 0:
            p = (j - 1) >> 1
            if hd[p] >= d:
                break
            hd[j] = hd[p]
            hi[j] = hi[p]
            j = p
        hd[j] = d
        hi[j] = idx
    elif d < hd[0]:
        j = 0
        while True:
            c = 2 * j + 1
            if c >= size:
                break
            if c + 1 < size and hd[c + 1] > hd[c]:
                c += 1
            if hd[c] <= d:
                break
            hd[j] = hd[c]
            hi[j] = hi[c]
            j = c
        hd[j] = d
        hi[j] = idx
    return size


# ---- growable min-heap: HNSW candidate queue ----

@njit(inline="always")
def minheap_push(hd, hi, size, d, idx):
    if size == hd.shape[0]:
        nd = np.empty(2 * size, hd.dtype)
        ni = np.empty(2 * size, hi.dtype)
        nd[:size] = hd
        ni[:size] = hi
        hd, hi = nd, ni
    j = size
    size += 1
    while j > 0:
        p = (j - 1) >> 1
        if hd[p] <= d:
            break
        hd[j] = hd[p]
        hi[j] = hi[p]
        j = p
    hd[j] = d
    hi[j] = idx
    return hd, hi, size


@njit(inline="always")
def minheap_pop(hd, hi, size):
    d0 = hd[0]
    i0 = hi[0]
    size -= 1
    d = hd[size]
    idx = hi[size]
    j = 0
    while True:
        c = 2 * j + 1
        if c >= size:
            break
        if c + 1 < size and hd[c + 1] < hd[c]:
            c += 1
        if hd[c] >= d:
            break
        hd[j] = hd[c]
        hi[j] = hi[c]
        j = c
    if size > 0:
        hd[j] = d
        hi[j] = idx
    return d0, i0, size


@njit
def heap_to_sorted(hd, hi, size, k, id_map):
    """Sort heap contents ascending, map positions through id_map (if non-empty), pad to k."""
    order = np.argsort(hd[:size])
    ids = np.full(k, -1, np.int64)
    dists = np.full(k, INF, np.float32)
    for t in range(min(size, k)):
        p = hi[order[t]]
        ids[t] = id_map[p] if id_map.shape[0] > 0 else p
        dists[t] = hd[order[t]]
    return ids, dists


# ---- exact search over an explicit candidate set (used by pre-filtering) ----

@njit(fastmath=True, cache=True)
def exact_subset_search(q, base, cand, k):
    hd = np.empty(k, np.float32)
    hi = np.empty(k, np.int64)
    hs = 0
    for t in range(cand.shape[0]):
        i = cand[t]
        hs = maxheap_push(hd, hi, hs, k, l2sq(q, base[i]), i)
    ids, dists = heap_to_sorted(hd, hi, hs, k, np.empty(0, np.int64))
    return ids, dists, cand.shape[0]


# ---- IVF ----

@njit(fastmath=True, cache=True)
def probe_lists(q, centroids, nprobe):
    """Return the nprobe closest list ids (ascending distance)."""
    nlist = centroids.shape[0]
    nprobe = min(nprobe, nlist)
    ph = np.empty(nprobe, np.float32)
    pi = np.empty(nprobe, np.int64)
    ps = 0
    for c in range(nlist):
        ps = maxheap_push(ph, pi, ps, nprobe, l2sq(q, centroids[c]), c)
    order = np.argsort(ph[:ps])
    return pi[order]


@njit(fastmath=True, cache=True)
def ivf_search(q, centroids, offsets, vecs, ids, list_labels, label, nprobe, k):
    """Scan the nprobe closest lists. If label >= 0, entries whose label differs
    are skipped before any distance is computed (in-search filtering)."""
    probes = probe_lists(q, centroids, nprobe)
    ndis = centroids.shape[0]
    hd = np.empty(k, np.float32)
    hi = np.empty(k, np.int64)
    hs = 0
    for t in range(probes.shape[0]):
        p = probes[t]
        for j in range(offsets[p], offsets[p + 1]):
            if label >= 0 and list_labels[j] != label:
                continue
            hs = maxheap_push(hd, hi, hs, k, l2sq(q, vecs[j]), j)
            ndis += 1
    rid, rd = heap_to_sorted(hd, hi, hs, k, ids)
    return rid, rd, ndis


@njit(fastmath=True, cache=True)
def ivf_partitioned_search(q, centroids, part_offsets, vecs, ids, label, nprobe, k):
    """Each list is grouped by label; jump straight to the label's slice."""
    probes = probe_lists(q, centroids, nprobe)
    ndis = centroids.shape[0]
    hd = np.empty(k, np.float32)
    hi = np.empty(k, np.int64)
    hs = 0
    L = part_offsets.shape[1] - 1
    for t in range(probes.shape[0]):
        p = probes[t]
        if label >= 0:
            if label >= L:
                continue
            b = part_offsets[p, label]
            e = part_offsets[p, label + 1]
        else:
            b = part_offsets[p, 0]
            e = part_offsets[p, L]
        for j in range(b, e):
            hs = maxheap_push(hd, hi, hs, k, l2sq(q, vecs[j]), j)
            ndis += 1
    rid, rd = heap_to_sorted(hd, hi, hs, k, ids)
    return rid, rd, ndis


# ---- HNSW ----

@njit(fastmath=True, cache=True)
def hnsw_search(q, base, neighbors, offsets, cum, entry, max_level,
                labels, label, ef, k, visited, tag, max_ndis):
    """Greedy descent on the upper layers, then beam search on layer 0.

    If label >= 0 (in-search filtering), every node can be traversed but only
    nodes with that label enter the result set. The search stops when the
    closest unexpanded candidate is farther than the worst of `ef` results,
    or after max_ndis distance computations (if max_ndis > 0)."""
    cur = entry
    dcur = l2sq(q, base[cur])
    ndis = 1
    for lev in range(max_level, 0, -1):
        changed = True
        while changed:
            changed = False
            b = offsets[cur] + cum[lev]
            e = offsets[cur] + cum[lev + 1]
            for t in range(b, e):
                n = neighbors[t]
                if n < 0:
                    break
                d = l2sq(q, base[n])
                ndis += 1
                if d < dcur:
                    dcur = d
                    cur = n
                    changed = True

    cd = np.empty(max(4 * ef, 64), np.float32)
    ci = np.empty(cd.shape[0], np.int64)
    cs = 0
    rd = np.empty(ef, np.float32)
    ri = np.empty(ef, np.int64)
    rs = 0
    visited[cur] = tag
    cd, ci, cs = minheap_push(cd, ci, cs, dcur, cur)
    if label < 0 or labels[cur] == label:
        rs = maxheap_push(rd, ri, rs, ef, dcur, cur)

    deg0 = cum[1] - cum[0]
    while cs > 0:
        dc, c, cs = minheap_pop(cd, ci, cs)
        if rs >= ef and dc > rd[0]:
            break
        b = offsets[c] + cum[0]
        for t in range(b, b + deg0):
            n = neighbors[t]
            if n < 0:
                break
            if visited[n] == tag:
                continue
            visited[n] = tag
            d = l2sq(q, base[n])
            ndis += 1
            if rs < ef or d < rd[0]:
                cd, ci, cs = minheap_push(cd, ci, cs, d, n)
                if label < 0 or labels[n] == label:
                    rs = maxheap_push(rd, ri, rs, ef, d, n)
        if max_ndis > 0 and ndis >= max_ndis:
            break

    rid, rdist = heap_to_sorted(rd, ri, rs, k, np.empty(0, np.int64))
    return rid, rdist, ndis
