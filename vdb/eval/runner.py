"""Run a plan over workload entries and record per-query measurements."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..plans.base import Plan
from .metrics import recall_at_k


def warm_up(plan: Plan, q, label, k, budget):
    plan.search(q, label, k, budget)  # trigger Numba compilation outside timing


def run_plan(plan: Plan, queries: np.ndarray, wl, gt_ids: np.ndarray, rows: np.ndarray,
             k: int, budget: int, gt_dists: np.ndarray | None = None) -> pd.DataFrame:
    """gt_dists (optional, same shape as gt_ids) enables tie-aware recall."""
    warm_up(plan, queries[wl.query_ids[rows[0]]], int(wl.labels[rows[0]]), k, budget)
    rec = []
    for r in rows:
        q = queries[wl.query_ids[r]]
        label = int(wl.labels[r])
        t0 = time.perf_counter()
        res = plan.search(q, label, k, budget)
        dt = time.perf_counter() - t0
        rec.append((int(r), int(wl.query_ids[r]), label, float(wl.selectivity[r]), int(wl.split[r]),
                    recall_at_k(res.ids, gt_ids[r], k, res.dists, None if gt_dists is None else gt_dists[r]), dt * 1e3, res.ndis, res.rounds,
                    int((res.ids >= 0).sum()), res.chosen, res.overhead_ms))
    df = pd.DataFrame(rec, columns=["row", "query_id", "label", "selectivity", "split", "recall",
                                    "latency_ms", "ndis", "rounds", "n_returned", "chosen", "overhead_ms"])
    df.insert(0, "budget", budget)
    df.insert(0, "strategy", plan.strategy)
    df.insert(0, "index", plan.index)
    df.insert(0, "plan", plan.name)
    return df


def sweep(plans, queries, wl, gt_ids, rows, k: int = 10, budgets: dict | None = None,
          progress: bool = True, gt_dists: np.ndarray | None = None) -> pd.DataFrame:
    """Run every plan at every budget in its grid (or budgets[plan.name])."""
    frames = []
    for plan in plans:
        grid = (budgets or {}).get(plan.name, plan.budgets)
        for b in grid:
            df = run_plan(plan, queries, wl, gt_ids, rows, k, b, gt_dists)
            frames.append(df)
            if progress:
                print(f"  {plan.name:11} budget={b:<5} recall={df.recall.mean():.3f} "
                      f"p50={df.latency_ms.median():8.3f}ms ndis={df.ndis.mean():9.0f}", flush=True)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["plan", "budget", "selectivity"])
    return g.agg(recall=("recall", "mean"), p50_ms=("latency_ms", "median"),
                 p95_ms=("latency_ms", lambda x: np.percentile(x, 95)),
                 ndis=("ndis", "mean"), rounds=("rounds", "mean"), n=("recall", "size")).reset_index()


def sample_rows(wl, split: str, per_selectivity: int, seed: int = 0) -> np.ndarray:
    """Workload rows used by sweeps and planner evaluation: up to `per_selectivity`
    rows per label from the given split ('val', 'test' or 'all'). Same arguments
    give the same rows, so sweeps, oracle and live planner runs line up."""
    rng = np.random.default_rng(seed)
    mask = np.ones(len(wl), bool) if split == "all" else wl.split == (0 if split == "val" else 1)
    rows = []
    for lab in np.unique(wl.labels):
        cand = np.flatnonzero(mask & (wl.labels == lab))
        rows.append(np.sort(rng.choice(cand, min(per_selectivity, cand.size), replace=False)))
    return np.concatenate(rows)
