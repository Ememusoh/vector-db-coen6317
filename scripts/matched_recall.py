"""Matched-recall comparison of planners across recall targets.

For each target T, every method is tuned on the validation sweeps and then
scored on the test sweeps by lookup: the method picks a (plan, budget) per test
query, and we read that query's measured recall and latency from the test
sweep. All methods are therefore timed on identical executions, so the only
difference is the choice. Planning overhead (measured in the live runs) is
added to the planner's latency.

Methods
    planner         Lagrangian cost-model planner (global + local selectivity features)
    planner_global  ablation: same model, global selectivity features only
    planner_cutoff  earlier planner: per-query probability cutoff, one per selectivity
    threshold       pre-filter below a tuned selectivity, else one tuned config
    fixed           single best config
    oracle          hindsight per-query optimum (lower bound on latency)

Usage: python scripts/matched_recall.py [--datasets sift glove deep]
Output: results/matched_recall.csv, results/figures/matched_recall.pdf/.png,
        results/tables/matched_recall_095.md/.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vdb.data import datasets  # noqa: E402
from vdb.planner import CoarseStats, CostModelPlanner, LagrangianPlanner, ThresholdRule, oracle_choice  # noqa: E402
from vdb.planner.data import load_split, matrices  # noqa: E402

TARGETS = (0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99)
GLOBAL_FEATURES = [0, 1]  # log_sel, log_nelig


def fixed_choice(rec, lat, target):
    ok = rec.mean(0) >= target
    j = np.where(ok, lat.mean(0), np.inf).argmin() if ok.any() else rec.mean(0).argmax()
    return np.full(rec.shape[0], j)


def planner_overhead_ms(ds):
    f = ROOT / "results" / "raw" / f"{ds}_planner_test.parquet"
    if not f.exists():
        return 0.0, 0.0
    p = pd.read_parquet(f)
    return (p[p.plan == "planner"].overhead_ms.mean(), p[p.plan == "threshold"].overhead_ms.mean())


def run_dataset(ds, targets):
    base, queries, _ = datasets.load(ds, mmap=False)
    centroids = CoarseStats.build_centroids(base)
    SCHEMES = datasets.schemes(ds)
    stats = {s: CoarseStats.for_scheme(ds, s, base, centroids) for s in SCHEMES}
    Xv, tv, mv = load_split(ds, SCHEMES, "val", queries, stats)
    Xt, tt, mt = load_split(ds, SCHEMES, "test", queries, stats)
    rec_t, lat_t, configs = matrices(tt, len(mt))
    col = {c: j for j, c in enumerate(configs)}
    groups = mv.query_id.to_numpy()
    sel_t = mt.selectivity.to_numpy()
    ov_planner, ov_rule = planner_overhead_ms(ds)
    rows = np.arange(len(mt))

    out = []
    for T in targets:
        picks = {}
        for name, cols in (("planner", None), ("planner_global", GLOBAL_FEATURES)):
            Xa, Xb = (Xv, Xt) if cols is None else (Xv[:, cols], Xt[:, cols])
            m = LagrangianPlanner(target=T, margin=0.0).fit(Xa, tv, groups)
            picks[name] = np.array([col[m.configs[i]] for i in m.choose_batch(Xb)])
        m = CostModelPlanner(target=T, margin=0.0).fit(Xv, tv, groups)
        picks["planner_cutoff"] = np.array([col[m.configs[i]] for i in m.choose_batch(Xt)])
        rule = ThresholdRule(target=T).fit(10 ** Xv[:, 0], tv)  # exact filter selectivity
        picks["threshold"] = np.array([col[rule.choose(s)] for s in 10 ** Xt[:, 0]])
        rv, lv, cv = matrices(tv, len(mv))
        fj = fixed_choice(rv, lv, T)[0]
        picks["fixed"] = np.full(len(mt), col[cv[fj]])
        picks["oracle"] = oracle_choice(rec_t, lat_t, T)
        overhead = {"planner": ov_planner, "planner_global": ov_planner, "planner_cutoff": ov_planner,
                    "threshold": ov_rule}
        for name, j in picks.items():
            r, t = rec_t[rows, j], lat_t[rows, j] + overhead.get(name, 0.0)
            scopes = [(s, (mt.scheme == s).to_numpy()) for s in SCHEMES] if len(SCHEMES) > 1 else []
            for scope, mask in [("all", np.ones(len(mt), bool))] + scopes:
                cells = pd.DataFrame({"sel": sel_t[mask], "r": r[mask]}).groupby("sel").r.mean()
                out.append(dict(dataset=ds, target=T, method=name, scope=scope, recall=r[mask].mean(),
                                mean_ms=t[mask].mean(), p50_ms=np.median(t[mask]),
                                p95_ms=np.percentile(t[mask], 95), worst_cell_recall=cells.min()))
        print(f"  {ds} T={T:<5} " + "  ".join(
            f"{n}: {rec_t[rows, j].mean():.3f}/{(lat_t[rows, j] + overhead.get(n, 0)).mean():.3f}ms"
            for n, j in picks.items()), flush=True)
    return pd.DataFrame(out)


def plot(df):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import make_figures as mf

    mf.setup_style()
    styles = {
        "planner":        ("Cost-model planner",            mf.METHODS["planner"][1], "o", "-", True),
        "planner_global": ("Planner, global selectivity only", mf.METHODS["planner"][1], "o", (0, (3, 2)), False),
        "planner_cutoff": ("Earlier planner (cutoff rule)",   mf.METHODS["planner"][1], "x", (0, (1, 1.5)), True),
        "threshold":      ("Threshold rule",                 mf.METHODS["threshold"][1], "s", "-", True),
        "fixed":          ("Best fixed config",              mf.METHODS["fixed"][1], "D", "-", True),
        "oracle":         ("Oracle (hindsight)",             mf.INK2, "o", (0, (1, 1.5)), False),
    }
    dsets = list(dict.fromkeys(df.dataset))
    fig, axes = plt.subplots(1, len(dsets), figsize=(mf.TEXT_W, 2.15), squeeze=False)
    d_all = df[df.scope == "all"]
    for ax, ds in zip(axes[0], dsets):
        d = d_all[d_all.dataset == ds]
        for m, (label, color, marker, ls, filled) in styles.items():
            g = d[d.method == m].sort_values("recall")
            ax.plot(g.recall, g.mean_ms, color=color, linestyle=ls, linewidth=1.1, marker=marker, markersize=3.6,
                    markerfacecolor=color if filled else mf.SURFACE, markeredgecolor=color, markeredgewidth=0.8)
        ax.set_yscale("log")
        lo, hi = d.mean_ms.min() / 1.3, d.mean_ms.max() * 1.3
        ax.set_ylim(lo, hi)
        mf.log_ms_axis(ax.yaxis, lo, hi)
        ax.set_xlim(0.84, 1.003)
        ax.set_title(mf.DATASET_NAMES.get(ds, ds), color=mf.INK)
        ax.set_xlabel("achieved recall@10 on test queries")
    axes[0, 0].set_ylabel("mean latency (ms)")
    handles = [Line2D([], [], color=c, linestyle=ls, marker=mk, markersize=4, linewidth=1.1,
                      markerfacecolor=c if f else mf.SURFACE, markeredgecolor=c, label=lab)
               for lab, c, mk, ls, f in styles.values()]
    fig.tight_layout(w_pad=0.8)
    top = max(ax.get_position().y1 for ax in axes.flat)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, top + 0.05),
               handlelength=2.2, columnspacing=1.2)
    fig.text(0.5, top + 0.25, "Matched recall: each point is one method tuned for a target in "
             f"{{{', '.join(f'{t:g}' for t in TARGETS)}}} on validation, scored on test (lower-right is better)",
             ha="center", fontsize=7.5, color=mf.INK)
    mf.save(fig, "matched_recall")


def table(df):
    import make_figures as mf

    d = df[(df.target == 0.95)].copy()
    names = {"planner": "Cost-model planner", "planner_global": "Planner (global sel. only)",
             "planner_cutoff": "Earlier planner (cutoff)",
             "threshold": "Threshold rule", "fixed": "Best fixed config", "oracle": "Oracle (hindsight)"}
    d["Dataset"] = d.dataset.map(lambda x: mf.DATASET_NAMES.get(x, x))
    d["Method"] = d.method.map(names)
    d["Scope"] = d.scope.map({"all": "both", "independent": "indep.", "correlated": "corr."})
    d = d[["Dataset", "Scope", "Method", "recall", "worst_cell_recall", "mean_ms", "p50_ms", "p95_ms"]]
    d.columns = ["Dataset", "Labels", "Method", "Recall@10", "Worst selectivity", "Mean ms", "p50 ms", "p95 ms"]
    mf.write_table(d, "matched_recall_095",
                   "Methods tuned for recall@10 = 0.95 on validation, scored on identical test executions. "
                   "'Worst selectivity' is the lowest per-selectivity mean recall.",
                   {c: ".3f" for c in ["Recall@10", "Worst selectivity", "Mean ms", "p50 ms", "p95 ms"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sift", "glove", "deep"])
    args = ap.parse_args()
    df = pd.concat([run_dataset(ds, TARGETS) for ds in args.datasets], ignore_index=True)
    df.to_csv(ROOT / "results" / "matched_recall.csv", index=False)
    plot(df)
    table(df)


if __name__ == "__main__":
    main()
