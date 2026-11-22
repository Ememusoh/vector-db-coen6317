"""Run every plan across its budget grid on a dataset's filtered workload.

Usage:
    python scripts/sweep_plans.py sift independent --per-selectivity 100 --split val

Writes:
    results/raw/<dataset>_<scheme>_<split>.parquet   per-query rows
    results/<dataset>_<scheme>_<split>_summary.csv  mean recall, p50/p95 latency, ndis per plan/budget/selectivity
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdb.data import Workload, datasets  # noqa: E402
from vdb.eval import sample_rows, summarize, sweep  # noqa: E402
from vdb.plans import FilterContext, make_plans  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("scheme")
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--per-selectivity", type=int, default=100, help="queries per selectivity level")
    ap.add_argument("--plans", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    ctx = FilterContext.load(args.dataset, args.scheme)
    _, queries, _ = datasets.load(args.dataset)
    d = datasets.DATA_ROOT / args.dataset / args.scheme
    wl = Workload.load(d / "workload.npz")
    gtz = np.load(d / "gt.npz")
    gt, gt_d = gtz["ids"], gtz["dists"]
    print(f"loaded in {time.time() - t0:.0f}s")

    rows = sample_rows(wl, args.split, args.per_selectivity, args.seed)

    df = sweep(make_plans(ctx, args.plans), queries, wl, gt, rows, k=args.k, gt_dists=gt_d)
    df.insert(0, "scheme", args.scheme)
    df.insert(0, "dataset", args.dataset)

    tag = f"{args.dataset}_{args.scheme}_{args.split}"
    (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "results" / "raw" / f"{tag}.parquet")
    summary = summarize(df)
    summary.to_csv(ROOT / "results" / f"{tag}_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s -> results/{tag}_summary.csv")


if __name__ == "__main__":
    main()
