"""Train the cost-model planner and baselines on validation sweeps, then run
them live on held-out test queries (end-to-end latency, planning included).

Prerequisite: python scripts/sweep_plans.py <dataset> <scheme> --split val   (for each scheme)

Usage:
    python scripts/evaluate_planner.py sift --target 0.95 --per-selectivity 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdb.data import Workload, datasets  # noqa: E402
from vdb.eval import run_plan  # noqa: E402
from vdb.index import HNSWIndex, IVFIndex  # noqa: E402
from vdb.planner import (CoarseStats, CostModelPlanner, FixedSearch, PlannedSearch,  # noqa: E402
                         ThresholdRule, ThresholdSearch, config_key)
from vdb.plans import FilterContext, make_plans  # noqa: E402


def load_training(dataset, schemes, queries, stats):
    feats, tables, sels, groups, offset = [], [], [], [], 0
    for s in schemes:
        df = pd.read_parquet(ROOT / "results" / "raw" / f"{dataset}_{s}_val.parquet")
        rows = np.sort(df.row.unique())
        first = df.drop_duplicates("row").set_index("row").loc[rows]
        feats.append(stats[s].features_batch(queries[first.query_id.to_numpy()], first.label.to_numpy()))
        sels.append(first.selectivity.to_numpy())
        groups.append(first.query_id.to_numpy())
        t = df.assign(qi=offset + np.searchsorted(rows, df.row),
                      config=[config_key(p, b) for p, b in zip(df.plan, df.budget)])
        tables.append(t[["qi", "config", "recall", "latency_ms"]])
        offset += rows.size
    return np.vstack(feats), pd.concat(tables, ignore_index=True), np.concatenate(sels), np.concatenate(groups)


def best_fixed(table, target):
    g = table.groupby("config").agg(recall=("recall", "mean"), lat=("latency_ms", "mean"))
    ok = g[g.recall >= target]
    return (ok if len(ok) else g.sort_values("recall").tail(1)).lat.idxmin()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--schemes", nargs="+", default=["independent", "correlated"])
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--per-selectivity", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    base, queries, _ = datasets.load(args.dataset, mmap=False)
    idx = ROOT / "indexes" / args.dataset
    ivf = IVFIndex.load(next(idx.glob("ivf_*.npz")), base)
    hnsw = HNSWIndex.load(next(idx.glob("hnsw_*.npz")), base)
    centroids = CoarseStats.build_centroids(base)
    ctxs, stats = {}, {}
    for s in args.schemes:
        labels = np.load(datasets.DATA_ROOT / args.dataset / s / "labels.npy")
        ctxs[s] = FilterContext.from_parts(base, labels, ivf, hnsw)
        stats[s] = CoarseStats.build(base, labels, centroids)
    print(f"loaded in {time.time() - t0:.0f}s")

    X, table, sel, groups = load_training(args.dataset, args.schemes, queries, stats)
    planner = CostModelPlanner(target=args.target).fit(X, table, groups)
    rule = ThresholdRule(target=args.target).fit(sel, table)
    fixed = best_fixed(table, args.target)
    print(f"trained on {X.shape[0]} validation queries x {table.config.nunique()} configs")
    print(f"  planner tau={planner.tau:.2f}; threshold rule: prefilter if sel < {rule.theta:g}, "
          f"else {rule.fallback}; best fixed config: {fixed}")

    frames = []
    rng = np.random.default_rng(args.seed)
    for s in args.schemes:
        ctx = ctxs[s]
        d = datasets.DATA_ROOT / args.dataset / s
        wl = Workload.load(d / "workload.npz")
        gt = np.load(d / "gt.npz")["ids"]
        rows = np.concatenate([np.sort(rng.choice(np.flatnonzero((wl.split == 1) & (wl.labels == lab)),
                                                  args.per_selectivity, replace=False))
                               for lab in np.unique(wl.labels)])
        plans = {p.name: p for p in make_plans(ctx)}
        for p in plans.values():  # compile every kernel before timing
            p.search(queries[0], 0, args.k, p.budgets[0])
        methods = [PlannedSearch(ctx, plans, planner, stats[s]), ThresholdSearch(ctx, plans, rule),
                   FixedSearch(ctx, plans, fixed)]
        for m in methods:
            df = run_plan(m, queries, wl, gt, rows, args.k, 0)
            df.insert(0, "scheme", s)
            frames.append(df)
            print(f"  {s:11} {m.name:9} recall={df.recall.mean():.3f} mean={df.latency_ms.mean():7.2f}ms "
                  f"p50={df.latency_ms.median():6.2f}ms overhead={df.overhead_ms.mean() * 1e3:5.0f}us", flush=True)

    res = pd.concat(frames, ignore_index=True)
    res.insert(0, "dataset", args.dataset)
    (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
    res.to_parquet(ROOT / "results" / "raw" / f"{args.dataset}_planner_test.parquet")
    summary = res.groupby(["scheme", "selectivity", "plan"]).agg(
        recall=("recall", "mean"), mean_ms=("latency_ms", "mean"), p50_ms=("latency_ms", "median"),
        p95_ms=("latency_ms", lambda x: np.percentile(x, 95))).reset_index()
    summary.to_csv(ROOT / "results" / f"{args.dataset}_planner_test_summary.csv", index=False)
    (ROOT / "results" / f"{args.dataset}_planner_config.json").write_text(json.dumps({
        "target": args.target, "tau": planner.tau, "threshold_theta": rule.theta,
        "threshold_fallback": rule.fallback, "best_fixed": fixed}, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\n" + summary.round(3).to_string(index=False))
    print("\nOverall (test):")
    print(res.groupby("plan").agg(recall=("recall", "mean"), mean_ms=("latency_ms", "mean"),
                                  p50_ms=("latency_ms", "median")).round(3).to_string())
    print("\nPlanner choices (scheme, selectivity -> most common configs):")
    pc = res[res.plan == "planner"].groupby(["scheme", "selectivity"]).chosen.agg(
        lambda x: ", ".join(f"{c} {n}" for c, n in x.value_counts().head(3).items()))
    print(pc.to_string())


if __name__ == "__main__":
    main()
