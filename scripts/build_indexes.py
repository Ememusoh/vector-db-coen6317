"""Build and save IVF and HNSW indexes for a prepared dataset.

Usage:
    python scripts/build_indexes.py sift
    python scripts/build_indexes.py sift --nlist 4096 --M 32 --ef-construction 200

Output: indexes/<name>/ivf_nlist<nlist>.npz/.json, hnsw_M<M>_efc<efc>.npz/.json
(The attribute-partitioned IVF is derived from the IVF index at load time, per label scheme.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdb.data import datasets  # noqa: E402
from vdb.index import HNSWIndex, IVFIndex  # noqa: E402

INDEX_ROOT = ROOT / "indexes"


def default_nlist(n: int) -> int:
    return int(2 ** round(np.log2(4 * np.sqrt(n))))  # ~4 sqrt(N), power of two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--nlist", type=int, default=None)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--only", choices=["ivf", "hnsw"], default=None)
    args = ap.parse_args()

    base, _, meta = datasets.load(args.name, mmap=False)
    out = INDEX_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "ivf"):
        nlist = args.nlist or default_nlist(base.shape[0])
        ivf = IVFIndex.build(base, nlist)
        ivf.save(out / f"ivf_nlist{nlist}.npz")
        print(f"IVF nlist={nlist}: {ivf.build_info['build_seconds']:.1f}s, {ivf.memory_bytes() / 2**20:.0f} MiB")

    if args.only in (None, "hnsw"):
        h = HNSWIndex.build(base, M=args.M, ef_construction=args.ef_construction)
        h.save(out / f"hnsw_M{args.M}_efc{args.ef_construction}.npz")
        print(f"HNSW M={args.M} efc={args.ef_construction}: {h.build_info['build_seconds']:.1f}s, "
              f"graph {h.memory_bytes(include_vectors=False) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
