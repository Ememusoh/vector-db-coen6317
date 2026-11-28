"""YFCC-10M with native tags (NeurIPS'23 big-ann filter track), as a 1M subset.

Every base vector carries a set of tags (4.8 on average, 200k distinct tags).
We use the public queries that filter on a single tag ("vector has tag t"),
which is the same predicate shape as the synthetic datasets, but with real,
naturally correlated, multi-label attributes. Two-tag (AND) queries are left
out. Queries are grouped into four selectivity bands by the measured
selectivity of their tag in the subset.

Layout (datasets/yfcc/):
    base.npy, queries.npy, meta.json        as for the other datasets
    native/postings.npz   order, offsets: base ids having predicate p are order[offsets[p]:offsets[p+1]]
    native/predicates.npy original YFCC tag id of predicate p
    native/workload.npz   labels = predicate id, selectivity = band (median selectivity of its queries)
    native/gt.npz, native/info.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
FILES = ("base.10M.u8bin", "base.metadata.10M.spmat", "query.public.100K.u8bin", "query.metadata.public.100K.spmat")
BANDS = ((1e-2, 1.0), (1e-3, 1e-2), (1e-4, 1e-3), (1e-5, 1e-4))  # selectivity ranges, high to low


def read_u8bin(path: Path, mmap: bool = True) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.uint32, count=2).astype(np.int64)
    if mmap:
        return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(n, d))
    return np.fromfile(path, dtype=np.uint8, offset=8).reshape(n, d)


def read_spmat(path: Path):
    """CSR matrix: int64 (nrow, ncol, nnz), int64 indptr[nrow+1], int32 indices[nnz], float32 data[nnz]."""
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype=np.int64, count=3)
        indptr = np.fromfile(f, dtype=np.int64, count=nrow + 1)
        indices = np.fromfile(f, dtype=np.int32, count=nnz)
    assert indptr[-1] == nnz
    return indptr, indices, int(ncol)


def download(raw: Path):
    import subprocess

    raw.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (raw / f).exists():
            tmp = raw / (f + ".part")
            subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(tmp), URL + f], check=True)
            tmp.rename(raw / f)


def prepare(out: Path, n: int = 1_000_000, n_queries: int = 2000, k: int = 100, val_fraction: float = 0.3,
            min_eligible: int = 10, seed: int = 0):
    from . import groundtruth
    from .workload import Workload

    raw = out / "raw"
    download(raw)
    rng = np.random.default_rng(seed)

    # base subset
    base_all = read_u8bin(raw / "base.10M.u8bin")
    sub = np.sort(rng.choice(base_all.shape[0], n, replace=False))
    base = np.ascontiguousarray(base_all[sub], dtype=np.float32)
    queries = read_u8bin(raw / "query.public.100K.u8bin", mmap=False).astype(np.float32)
    np.save(out / "base.npy", base)
    np.save(out / "queries.npy", queries)
    (out / "meta.json").write_text(json.dumps({
        "name": "yfcc", "metric": "l2", "distance": "l2", "n_base": n, "n_queries": int(queries.shape[0]),
        "dim": int(base.shape[1]), "source": URL, "subset": f"{n} of 10M, uniform, seed {seed}"}, indent=2))

    # tags of the subset, as (tag, base id) pairs
    bp, bi, n_tags = read_spmat(raw / "base.metadata.10M.spmat")
    lens = bp[sub + 1] - bp[sub]
    tag_of = bi[_ranges(bp[sub], lens)]
    id_of = np.repeat(np.arange(n, dtype=np.int64), lens)
    tag_count = np.bincount(tag_of, minlength=n_tags)

    # single-tag queries, banded by selectivity in the subset
    qp, qi, _ = read_spmat(raw / "query.metadata.public.100K.spmat")
    single = np.flatnonzero(np.diff(qp) == 1)
    qtag = qi[qp[single]]
    qsel = tag_count[qtag] / n
    ok = tag_count[qtag] >= min_eligible
    per_band = n_queries // len(BANDS)
    chosen, band_of = [], []
    for b, (lo, hi) in enumerate(BANDS):
        cand = single[ok & (qsel >= lo) & (qsel < hi)]
        pick = np.sort(rng.choice(cand, min(per_band, cand.size), replace=False))
        chosen.append(pick)
        band_of.append(np.full(pick.size, b))
    qids = np.concatenate(chosen)
    band_of = np.concatenate(band_of)
    tags = qi[qp[qids]]

    # predicates = distinct tags used by the workload, renumbered 0..m-1
    pred_tags, labels = np.unique(tags, return_inverse=True)
    keep = np.isin(tag_of, pred_tags)
    p_of = np.searchsorted(pred_tags, tag_of[keep])
    order = id_of[keep][np.lexsort((id_of[keep], p_of))]
    offsets = np.zeros(pred_tags.size + 1, np.int64)
    offsets[1:] = np.cumsum(np.bincount(p_of, minlength=pred_tags.size))

    sel_true = tag_count[tags] / n
    band_sel = np.array([np.median(sel_true[band_of == b]) for b in range(len(BANDS))])
    band_sel = np.array([float(f"{s:.2g}") for s in band_sel])
    split = np.ones(qids.size, np.uint8)
    split[rng.permutation(qids.size)[: round(val_fraction * qids.size)]] = 0
    wl = Workload(qids.astype(np.int32), labels.astype(np.int32), band_sel[band_of], split)

    sd = out / "native"
    sd.mkdir(exist_ok=True)
    np.savez(sd / "postings.npz", order=order, offsets=offsets)
    np.save(sd / "predicates.npy", pred_tags.astype(np.int32))
    wl.save(sd / "workload.npz")
    ids, dists = groundtruth.posting_knn(base, queries, order, offsets, wl.query_ids, wl.labels, k=k)
    np.savez(sd / "gt.npz", ids=ids, dists=dists)
    info = {"scheme": "native", "seed": seed, "k": k, "n_workload": len(wl), "n_predicates": int(pred_tags.size),
            "bands": [list(b) for b in BANDS], "band_selectivity": band_sel.tolist(),
            "queries_per_band": np.bincount(band_of).tolist(),
            "single_tag_queries": int(single.size), "two_tag_queries_excluded": int((np.diff(qp) == 2).sum()),
            "short_entries": int((ids[:, 9] < 0).sum())}
    (sd / "info.json").write_text(json.dumps(info, indent=2))
    return info


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Concatenation of arange(s, s+l) for every (s, l), vectorised (zero lengths allowed)."""
    before = np.cumsum(lens) - lens
    return np.arange(int(lens.sum()), dtype=np.int64) + np.repeat(starts - before, lens)
