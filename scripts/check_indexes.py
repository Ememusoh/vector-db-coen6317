"""Sanity check: our IVF / HNSW search vs FAISS's own search on the same structures.

Unfiltered recall@10 against exact ground truth, single-threaded latency, and
distance computations. Our numbers should track FAISS closely; this is what
lets us trust the instrumented kernels for the filtered experiments.

Usage: python scripts/check_indexes.py sift --queries 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdb.data import datasets  # noqa: E402
from vdb.index import HNSWIndex, IVFIndex  # noqa: E402


def recall_at_10(found, gt):
    return np.mean([len(set(f[:10]) & set(g[:10])) / 10 for f, g in zip(found, gt)])


def run_ours(fn, queries):
    fn(queries[0])  # JIT warm-up
    ids, ndis, t = [], [], []
    for q in queries:
        s = time.perf_counter()
        r = fn(q)
        t.append(time.perf_counter() - s)
        ids.append(r.ids)
        ndis.append(r.ndis)
    return np.array(ids), np.mean(ndis), np.array(t) * 1e3


def run_faiss(index, queries):
    ids, t = [], []
    for q in queries:
        s = time.perf_counter()
        _, i = index.search(q[None], 10)
        t.append(time.perf_counter() - s)
        ids.append(i[0])
    return np.array(ids), np.array(t) * 1e3


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--queries", type=int, default=1000)
    args = ap.parse_args()
    faiss.omp_set_num_threads(1)

    base, queries, _ = datasets.load(args.name, mmap=False)
    queries = queries[: args.queries]
    flat = faiss.IndexFlatL2(base.shape[1])
    flat.add(base)
    _, gt = flat.search(queries, 10)

    idx_dir = ROOT / "indexes" / args.name
    ivf = IVFIndex.load(next(idx_dir.glob("ivf_*.npz")), base)
    hnsw = HNSWIndex.load(next(idx_dir.glob("hnsw_*.npz")), base)

    # FAISS IVF on the same centroids and assignment
    fq = faiss.IndexFlatL2(base.shape[1])
    fq.add(ivf.centroids)
    fivf = faiss.IndexIVFFlat(fq, base.shape[1], ivf.nlist)
    fivf.is_trained = True
    fivf.add(base)
    fhnsw = faiss.IndexHNSWFlat(base.shape[1], hnsw.build_info["M"])
    fhnsw.hnsw.efConstruction = hnsw.build_info["ef_construction"]
    fhnsw.add(base)

    print(f"{'index':6} {'param':>6} | {'ours R@10':>9} {'p50 ms':>7} {'ndis':>7} | {'faiss R@10':>10} {'p50 ms':>7}")
    for nprobe in (8, 16, 32, 64, 128):
        ids, nd, t = run_ours(lambda q: ivf.search(q, 10, nprobe), queries)
        fivf.nprobe = nprobe
        fids, ft = run_faiss(fivf, queries)
        print(f"{'IVF':6} {nprobe:6} | {recall_at_10(ids, gt):9.3f} {np.median(t):7.3f} {nd:7.0f} | "
              f"{recall_at_10(fids, gt):10.3f} {np.median(ft):7.3f}")
    for ef in (16, 32, 64, 128, 256):
        ids, nd, t = run_ours(lambda q: hnsw.search(q, 10, ef), queries)
        fhnsw.hnsw.efSearch = ef
        fids, ft = run_faiss(fhnsw, queries)
        print(f"{'HNSW':6} {ef:6} | {recall_at_10(ids, gt):9.3f} {np.median(t):7.3f} {nd:7.0f} | "
              f"{recall_at_10(fids, gt):10.3f} {np.median(ft):7.3f}")


if __name__ == "__main__":
    main()
