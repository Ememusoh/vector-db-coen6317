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
from vdb.eval import run_plan, sample_rows  # noqa: E402
from vdb.index import HNSWIndex, IVFIndex  # noqa: E402
from vdb.planner import (CoarseStats, CostModelPlanner, FixedSearch, LagrangianPlanner, PlannedSearch,  # noqa: E402
                         ThresholdRule, ThresholdSearch, config_key, oracle_choice)
from vdb.planner.data import load_split  # noqa: E402
from vdb.plans import FilterContext, MultiLabelContext, make_plans  # noqa: E402


def oracle_from_test_sweeps(dataset, schemes, target):
    """Best possible per-query choices with hindsight, on test-split sweeps (if they exist)."""
    frames = []
    for s in schemes:
        f = ROOT / "results" / "raw" / f"{dataset}_{s}_test.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f).assign(scheme=s)
        df["config"] = [config_key(p, b) for p, b in zip(df.plan, df.budget)]
        frames.append(df)
    df = pd.concat(frames)
    rec = df.pivot_table(index=["scheme", "row"], columns="config", values="recall")
    lat = df.pivot_table(index=["scheme", "row"], columns="config", values="latency_ms")
    idx = oracle_choice(rec.to_numpy(), lat.to_numpy(), target)
    q = np.arange(len(rec))
    sel = df.drop_duplicates(["scheme", "row"]).set_index(["scheme", "row"]).loc[rec.index, "selectivity"]
    return pd.DataFrame({"scheme": rec.index.get_level_values(0), "selectivity": sel.to_numpy(),
                         "config": rec.columns[idx], "recall": rec.to_numpy()[q, idx],
                         "latency_ms": lat.to_numpy()[q, idx]})


def best_fixed(table, target):
    g = table.groupby("config").agg(recall=("recall", "mean"), lat=("latency_ms", "mean"))
    ok = g[g.recall >= target]
    return (ok if len(ok) else g.sort_values("recall").tail(1)).lat.idxmin()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--schemes", nargs="+", default=None, help="default: the dataset's schemes")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--planner", choices=["lagrangian", "cutoff"], default="lagrangian",
                    help="lagrangian: argmin latency - lam*recall (default); cutoff: earlier per-query probability cutoff")
    ap.add_argument("--margin", type=float, default=None,
                    help="safety buffer added to the target when calibrating (default 0.005 lagrangian, 0.01 cutoff)")
    ap.add_argument("--per-bucket", action="store_true", help="lagrangian: one lam per selectivity level")
    ap.add_argument("--no-buckets", action="store_true", help="cutoff: one cutoff for all selectivities")
    ap.add_argument("--per-selectivity", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="must match the sweeps' seed for the oracle to line up")
    args = ap.parse_args()

    t0 = time.time()
    args.schemes = args.schemes or list(datasets.schemes(args.dataset))
    base, queries, _ = datasets.load(args.dataset, mmap=False)
    idx = ROOT / "indexes" / args.dataset
    ivf = IVFIndex.load(next(idx.glob("ivf_*.npz")), base)
    hnsw = HNSWIndex.load(next(idx.glob("hnsw_*.npz")), base)
    centroids = CoarseStats.build_centroids(base)
    ctxs, stats = {}, {}
    for s in args.schemes:
        d = datasets.DATA_ROOT / args.dataset / s
        if (d / "postings.npz").exists():
            z = np.load(d / "postings.npz")
            ctxs[s] = MultiLabelContext.from_postings(base, z["order"], z["offsets"], ivf, hnsw)
        else:
            ctxs[s] = FilterContext.from_parts(base, np.load(d / "labels.npy"), ivf, hnsw)
        stats[s] = CoarseStats.for_scheme(args.dataset, s, base, centroids)
    print(f"loaded in {time.time() - t0:.0f}s")

    X, table, meta = load_split(args.dataset, args.schemes, "val", queries, stats)
    sel, groups = 10 ** X[:, 0], meta.query_id.to_numpy()  # exact selectivity of each query's filter
    if args.margin is None:
        args.margin = 0.005 if args.planner == "lagrangian" else 0.01
    if args.planner == "lagrangian":
        planner = LagrangianPlanner(target=args.target, margin=args.margin, per_bucket=args.per_bucket)
    else:
        planner = CostModelPlanner(target=args.target, margin=args.margin, per_bucket=not args.no_buckets)
    planner.fit(X, table, groups)
    params = planner.lams if args.planner == "lagrangian" else planner.taus
    rule = ThresholdRule(target=args.target).fit(sel, table)
    fixed = best_fixed(table, args.target)
    print(f"trained on {X.shape[0]} validation queries x {table.config.nunique()} configs")
    taus = ", ".join(f"{t:.3g}" for t in params)
    print(f"  {args.planner} planner parameters: [{taus}]; threshold rule: prefilter if sel < {rule.theta:g}, "
          f"else {rule.fallback}; best fixed config: {fixed}")

    frames = []
    for s in args.schemes:
        ctx = ctxs[s]
        d = datasets.DATA_ROOT / args.dataset / s
        wl = Workload.load(d / "workload.npz")
        gtz = np.load(d / "gt.npz")
        gt, gt_d = gtz["ids"], gtz["dists"]
        rows = sample_rows(wl, "test", args.per_selectivity, args.seed)
        plans = {p.name: p for p in make_plans(ctx)}
        for p in plans.values():  # compile every kernel before timing
            ctx.prepare(int(wl.labels[rows[0]]))
            p.search(queries[0], int(wl.labels[rows[0]]), args.k, p.budgets[0])
        methods = [PlannedSearch(ctx, plans, planner, stats[s]), ThresholdSearch(ctx, plans, rule),
                   FixedSearch(ctx, plans, fixed)]
        # untimed pass over the same rows first: whichever method ran first otherwise paid for cold
        # memory and caches (up to 4x on YFCC), which biased the comparison against it
        for m in methods:
            run_plan(m, queries, wl, gt, rows, args.k, 0, gt_d)
        for m in methods:
            df = run_plan(m, queries, wl, gt, rows, args.k, 0, gt_d)
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
        "target": args.target, "planner": args.planner, "margin": args.margin,
        ("lams_ms" if args.planner == "lagrangian" else "taus"): params.tolist(),
        "bucket_edges_log10_sel": planner.bucket_edges.tolist(), "threshold_theta": rule.theta,
        "threshold_fallback": rule.fallback, "best_fixed": fixed}, indent=2, default=float))

    oracle = oracle_from_test_sweeps(args.dataset, args.schemes, args.target)
    if oracle is not None:
        print(f"\nOracle (hindsight, from test sweeps, {len(oracle)} queries): recall={oracle.recall.mean():.3f} "
              f"mean={oracle.latency_ms.mean():.2f}ms p50={oracle.latency_ms.median():.2f}ms")
        oracle.to_csv(ROOT / "results" / f"{args.dataset}_oracle_test.csv", index=False)

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
