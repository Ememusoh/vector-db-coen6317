"""Figures and tables for the report, from the CSV/parquet files in results/.

Usage:
    python scripts/make_figures.py                      # every dataset with results
    python scripts/make_figures.py --datasets sift glove

Output:
    results/figures/recall_latency_<dataset>.pdf/.png   recall vs latency per plan (small multiples)
    results/figures/planner_comparison.pdf/.png         planner vs baselines vs oracle, all datasets
    results/figures/planner_choices_<dataset>.pdf/.png  which plan the planner picked, per selectivity
    results/tables/*.md / *.tex                         same numbers as tables (booktabs LaTeX)

Missing inputs are skipped with a note, so this can run at any stage.
PDFs embed TrueType fonts (ACM rejects Type 3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from vdb.data.datasets import schemes as schemes_of  # noqa: E402  (synthetic: independent/correlated; YFCC: native)
FIG = RES / "figures"
TAB = RES / "tables"

# ---- style: reference palette (validated light mode), thin marks, recessive axes ----
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#8a8984"

# Fixed colour + marker per plan (entity, never rank). Order = validated slot order.
PLANS = {
    "prefilter":  ("Pre-filter (exact)",       "#2a78d6", "s"),
    "ivf_post":   ("IVF post-filter",          "#eb6834", "v"),
    "ivf_in":     ("IVF in-search",            "#1baf7a", "o"),
    "ivfpart_in": ("Partitioned IVF in-search", "#eda100", "D"),
    "hnsw_post":  ("HNSW post-filter",         "#e87ba4", "^"),
    "hnsw_in":    ("HNSW in-search",           "#008300", "P"),
}
# Planner comparison: three methods take the first three slots (all-pairs safe); oracle is a neutral reference.
METHODS = {
    "planner":   ("Cost-model planner", "#2a78d6", "o", True),
    "threshold": ("Threshold rule",     "#eb6834", "s", True),
    "fixed":     ("Best fixed config",  "#1baf7a", "D", True),
    "oracle":    ("Oracle (hindsight)", INK2,      "o", False),
}
SCHEME_SHORT = {"independent": "indep.", "correlated": "corr.", "native": "native"}


DATASET_NAMES = {"sift": "SIFT1M", "glove": "GloVe-100", "deep": "Deep1M", "gist": "GIST-500K", "yfcc": "YFCC"}
TARGET = 0.95

COL_W, TEXT_W = 3.33, 7.0  # ACM single column / full text width (inches)


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": NEUTRAL, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})  # byte-stable reruns
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  figures/{name}.pdf")


def ink_on(hex_color):
    """Primary ink or white, whichever reads better on the fill (WCAG relative luminance)."""
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lum = sum(w * (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
              for w, c in zip((0.2126, 0.7152, 0.0722), rgb))
    return INK if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "#ffffff"


def sel_label(s):
    return f"{s * 100:g}%"


def ms_formatter():
    return FuncFormatter(lambda v, _: f"{v:g}")


def log_ms_axis(axis, lo, hi):
    """Readable log axis: ticks at 1-2-5 steps when the range is short, decades otherwise."""
    from matplotlib.ticker import FixedLocator, NullFormatter

    steps = (1, 2, 5) if hi / lo < 300 else (1,)
    ticks = [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
             for m in steps]
    ticks = [t for t in ticks if lo / 1.05 <= t <= hi * 1.05]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ms_formatter())
    axis.set_minor_formatter(NullFormatter())


def write_table(df: pd.DataFrame, name: str, caption: str, floatfmt: dict | None = None):
    TAB.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c, f in (floatfmt or {}).items():
        out[c] = out[c].map(lambda v, f=f: "–" if pd.isna(v) else format(v, f))
    md = ["| " + " | ".join(out.columns) + " |", "|" + "---|" * len(out.columns)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in out.itertuples(index=False)]
    (TAB / f"{name}.md").write_text(f"{caption}\n\n" + "\n".join(md) + "\n")
    cols = "l" * 2 + "r" * (len(out.columns) - 2)
    tex = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(_tex(c) for c in out.columns) + r" \\", r"\midrule"]
    tex += [" & ".join(_tex(str(v)) for v in row) + r" \\" for row in out.itertuples(index=False)]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"  tables/{name}.md/.tex")


def _tex(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


# ---------------- loaders ----------------

def load_sweep_summary(ds, scheme):
    for split in ("test", "val"):
        f = RES / f"{ds}_{scheme}_{split}_summary.csv"
        if f.exists():
            return pd.read_csv(f), split
    return None, None


def load_planner(ds):
    f = RES / "raw" / f"{ds}_planner_test.parquet"
    return pd.read_parquet(f) if f.exists() else None


def load_oracle(ds):
    f = RES / f"{ds}_oracle_test.csv"
    return pd.read_csv(f) if f.exists() else None


# ---------------- figure 1: recall vs latency ----------------

def fig_recall_latency(ds):
    sums = {s: load_sweep_summary(ds, s) for s in schemes_of(ds)}
    sums = {s: v for s, v in sums.items() if v[0] is not None}
    if not sums:
        print(f"  skip recall_latency_{ds}: no sweep summaries")
        return
    sels = sorted(next(iter(sums.values()))[0].selectivity.unique(), reverse=True)
    fig, axes = plt.subplots(len(sums), len(sels), figsize=(TEXT_W, 1.55 * len(sums) + 0.45),
                             sharex=True, squeeze=False)
    for i, (scheme, (df, split)) in enumerate(sums.items()):
        lo, hi = df.p50_ms[df.recall >= 0.5].min(), df.p50_ms.max()
        for j, sel in enumerate(sels):
            ax = axes[i, j]
            d = df[df.selectivity == sel]
            for plan, (_, color, marker) in PLANS.items():
                g = d[d.plan == plan].sort_values("budget")
                if g.empty:
                    continue
                kw = dict(color=color, marker=marker, markersize=3.6, markeredgecolor=SURFACE, markeredgewidth=0.4)
                if len(g) == 1:
                    ax.plot(g.recall, g.p50_ms, linestyle="none", zorder=4, **{**kw, "markersize": 5})
                else:
                    ax.plot(g.recall, g.p50_ms, linewidth=1.1, zorder=3, **kw)
            ax.axvline(TARGET, color=NEUTRAL, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(lo / 1.6, hi * 1.6)
            ax.set_xlim(0.5, 1.005)
            log_ms_axis(ax.yaxis, lo / 1.6, hi * 1.6)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(f"selectivity {sel_label(sel)}", color=INK)
            if j == 0:
                ax.set_ylabel(f"{scheme}\nmedian latency (ms)")
            if i == len(sums) - 1:
                ax.set_xlabel("recall@10")
    handles = [Line2D([], [], color=c, marker=m, markersize=4, linewidth=1.1, label=n) for n, c, m in PLANS.values()]
    handles.append(Line2D([], [], color=NEUTRAL, linewidth=0.7, linestyle=(0, (3, 2)), label=f"target {TARGET}"))
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    top = max(ax.get_position().y1 for ax in axes.flat)
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, top + 0.045),
               handlelength=1.8, columnspacing=1.2)
    fig.text(0.5, top + 0.13, f"{DATASET_NAMES.get(ds, ds)}: recall vs latency per plan "
             f"({'/'.join(sorted({v[1] for v in sums.values()}))} queries; each line sweeps the plan's budget)",
             ha="center", fontsize=7.5, color=INK)
    save(fig, f"recall_latency_{ds}")


# ---------------- figure 2: planner vs baselines (all datasets) ----------------

def planner_rows(datasets):
    rows = []
    for ds in datasets:
        p = load_planner(ds)
        if p is None:
            continue
        o = load_oracle(ds)
        for scheme in schemes_of(ds):
            for m in ("planner", "threshold", "fixed"):
                d = p[(p.scheme == scheme) & (p.plan == m)]
                if len(d):
                    rows.append(dict(dataset=ds, scheme=scheme, method=m, recall=d.recall.mean(),
                                     mean_ms=d.latency_ms.mean(), p50_ms=d.latency_ms.median(),
                                     p95_ms=np.percentile(d.latency_ms, 95),
                                     overhead_us=d.overhead_ms.mean() * 1e3))
            if o is not None:
                d = o[o.scheme == scheme]
                if len(d):
                    rows.append(dict(dataset=ds, scheme=scheme, method="oracle", recall=d.recall.mean(),
                                     mean_ms=d.latency_ms.mean(), p50_ms=d.latency_ms.median(),
                                     p95_ms=np.percentile(d.latency_ms, 95), overhead_us=np.nan))
    return pd.DataFrame(rows)


def fig_planner_comparison(datasets):
    df = planner_rows(datasets)
    if df.empty:
        print("  skip planner_comparison: no planner results")
        return df
    groups = list(df.groupby(["dataset", "scheme"], sort=False).groups)
    n = len(groups)
    fig, (axr, axl) = plt.subplots(1, 2, figsize=(TEXT_W, 0.32 * n + 0.75), sharey=True,
                                   gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.08))
    offsets = {"planner": -0.21, "threshold": -0.07, "fixed": 0.07, "oracle": 0.21}
    labels = []
    for gi, (ds, scheme) in enumerate(groups):
        y = n - 1 - gi
        labels.append((y, f"{DATASET_NAMES.get(ds, ds)} · {scheme}"))
        for m, (name, color, marker, filled) in METHODS.items():
            d = df[(df.dataset == ds) & (df.scheme == scheme) & (df.method == m)]
            if d.empty:
                continue
            kw = dict(marker=marker, s=22, linewidths=1.0, zorder=3,
                      facecolors=color if filled else "none", edgecolors=SURFACE if filled else color)
            axr.scatter(d.recall, [y + offsets[m]], **kw)
            axl.scatter(d.mean_ms, [y + offsets[m]], **kw)
    for ax in (axr, axl):
        ax.set_yticks([y for y, _ in labels], [t for _, t in labels])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        for y, _ in labels:
            ax.axhline(y, color=GRID, linewidth=0.5, zorder=0)
    axr.axvline(TARGET, color=NEUTRAL, linewidth=0.7, linestyle=(0, (3, 2)))
    lo = min(0.9, df.recall.min() - 0.01)
    axr.set_xlim(lo, 1.005)
    axr.set_xlabel("recall@10 (mean)")
    axl.set_xscale("log")
    lo, hi = df.mean_ms.min() / 1.4, df.mean_ms.max() * 1.4
    axl.set_xlim(lo, hi)
    log_ms_axis(axl.xaxis, lo, hi)
    axl.set_xlabel("mean latency per query (ms, log scale)")
    axl.spines["left"].set_visible(False)
    handles = [Line2D([], [], linestyle="none", marker=mk, markersize=5, markerfacecolor=c if f else "none",
                      markeredgecolor=SURFACE if f else c, label=nm) for nm, c, mk, f in METHODS.values()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.02))
    fig.subplots_adjust(top=1 - 0.42 / (0.32 * n + 0.75))
    save(fig, "planner_comparison")
    return df


# ---------------- figure 3: what the planner chose ----------------

def fig_planner_choices(ds):
    p = load_planner(ds)
    if p is None:
        print(f"  skip planner_choices_{ds}: no planner results")
        return
    p = p[p.plan == "planner"].copy()
    p["chosen_plan"] = p.chosen.str.rsplit("@", n=1).str[0]
    sels = sorted(p.selectivity.unique(), reverse=True)
    rows = [(s, sel) for s in schemes_of(ds) for sel in sels if ((p.scheme == s) & (p.selectivity == sel)).any()]
    fig, ax = plt.subplots(figsize=(COL_W, 0.2 * len(rows) + 0.75))
    for i, (s, sel) in enumerate(rows):
        y = len(rows) - 1 - i
        d = p[(p.scheme == s) & (p.selectivity == sel)]
        share = d.chosen_plan.value_counts(normalize=True)
        left = 0.0
        for plan, (_, color, _) in PLANS.items():
            w = share.get(plan, 0.0)
            if w > 0:
                ax.barh(y, w, left=left, height=0.72, color=color, edgecolor=SURFACE, linewidth=1.0)
                if w >= 0.12:
                    ax.text(left + w / 2, y, f"{w * 100:.0f}%", ha="center", va="center", fontsize=5.5,
                            color=ink_on(color))
                left += w
    ax.set_yticks(range(len(rows))[::-1], [f"{SCHEME_SHORT[s]} {sel_label(sel)}" for s, sel in rows])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel("share of test queries")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    used = [k for k in PLANS if k in set(p.chosen_plan)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=PLANS[k][1], label=PLANS[k][0]) for k in used]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False,
              handlelength=1.0, columnspacing=1.0)
    ax.set_title(f"{DATASET_NAMES.get(ds, ds)}: plan chosen by the planner", color=INK, loc="left")
    save(fig, f"planner_choices_{ds}")


# ---------------- tables ----------------

def table_planner(df):
    if df.empty:
        return
    t = df.copy()
    base = t[t.method == "threshold"].set_index(["dataset", "scheme"]).mean_ms
    t["speedup_vs_threshold"] = [base.get((d, s), np.nan) / m for d, s, m in zip(t.dataset, t.scheme, t.mean_ms)]
    t["dataset"] = t.dataset.map(lambda d: DATASET_NAMES.get(d, d))
    t["method"] = t.method.map(lambda m: METHODS[m][0])
    t = t[["dataset", "scheme", "method", "recall", "mean_ms", "p50_ms", "p95_ms", "speedup_vs_threshold"]]
    t.columns = ["Dataset", "Labels", "Method", "Recall@10", "Mean ms", "p50 ms", "p95 ms", "Speed-up"]
    write_table(t, "planner_results",
                "Planner vs baselines on held-out test queries (end-to-end latency, planning included). "
                "Speed-up is relative to the tuned threshold rule.",
                {"Recall@10": ".3f", "Mean ms": ".3f", "p50 ms": ".3f", "p95 ms": ".3f", "Speed-up": ".1f"})


def table_best_plan(datasets):
    rows = []
    for ds in datasets:
        for scheme in schemes_of(ds):
            df, split = load_sweep_summary(ds, scheme)
            if df is None:
                continue
            for sel, d in df.groupby("selectivity"):
                ok = d[d.recall >= TARGET].sort_values("p50_ms")
                best = ok.iloc[0] if len(ok) else None
                pre = d[d.plan == "prefilter"]
                rows.append(dict(Dataset=DATASET_NAMES.get(ds, ds), Labels=scheme, Selectivity=sel_label(sel),
                                 **{"Fastest plan at recall ≥ 0.95": PLANS[best.plan][0] if best is not None else "none",
                                    "Budget": int(best.budget) if best is not None and best.plan != "prefilter" else "–",
                                    "p50 ms": best.p50_ms if best is not None else np.nan,
                                    "Pre-filter p50 ms": pre.p50_ms.iloc[0] if len(pre) else np.nan}))
    if rows:
        t = pd.DataFrame(rows)
        t["_s"] = t.Selectivity.str.rstrip("%").astype(float)
        t = t.sort_values(["Dataset", "Labels", "_s"], ascending=[True, False, False]).drop(columns="_s")
        write_table(t, "best_plan", "Fastest single plan reaching recall@10 ≥ 0.95, per selectivity "
                    "(from the budget sweeps).", {"p50 ms": ".3f", "Pre-filter p50 ms": ".3f"})


def table_indexes(datasets):
    rows = []
    for ds in datasets:
        d = ROOT / "indexes" / ds
        meta_f = ROOT / "datasets" / ds / "meta.json"
        if not d.exists() or not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        n, dim = meta["n_base"], meta["dim"]
        ivf_f, hnsw_f = next(d.glob("ivf_*.json"), None), next(d.glob("hnsw_*.json"), None)
        ivf = json.loads(ivf_f.read_text()) if ivf_f else {}
        hnsw = json.loads(hnsw_f.read_text()) if hnsw_f else {}
        vec_mib = n * dim * 4 / 2**20
        ivf_mib = (vec_mib * 2**20 + n * 8 + ivf.get("nlist", 0) * (dim * 4 + 8)) / 2**20 if ivf else np.nan
        n_labels = _n_labels(ds)
        part_mib = ivf_mib + ivf.get("nlist", 0) * (n_labels + 1) * 8 / 2**20 if ivf and n_labels else np.nan
        graph_mib = np.nan
        if hnsw_f:
            z = np.load(hnsw_f.with_suffix(".npz"))
            graph_mib = sum(z[k].nbytes for k in ("neighbors", "offsets", "cum", "levels")) / 2**20
        rows.append({"Dataset": DATASET_NAMES.get(ds, ds), "Vectors": f"{n / 1e6:g}M × {dim}",
                     "Raw vectors MiB": vec_mib,
                     "IVF build s": ivf.get("build_seconds", np.nan), "IVF MiB": ivf_mib,
                     "Part. IVF extra MiB": part_mib - ivf_mib,
                     "HNSW build s": hnsw.get("build_seconds", np.nan), "HNSW graph MiB": graph_mib})
    if rows:
        write_table(pd.DataFrame(rows), "indexes",
                    "Index build time and memory (IVF nlist ≈ 4√N, HNSW M=32, efConstruction=200). "
                    "IVF stores its own copy of the vectors in list order; the HNSW graph is stored "
                    "in addition to the raw vectors.",
                    {"Raw vectors MiB": ".0f", "IVF build s": ".1f", "IVF MiB": ".0f", "Part. IVF extra MiB": ".2f",
                     "HNSW build s": ".1f", "HNSW graph MiB": ".0f"})


def _n_labels(ds):
    for s in schemes_of(ds):
        f = ROOT / "datasets" / ds / s / "info.json"
        if f.exists():
            return json.loads(f.read_text()).get("n_labels", 0)  # native multi-label: no partitioned IVF
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    datasets = args.datasets or [d for d in DATASET_NAMES if (RES / "raw" / f"{d}_planner_test.parquet").exists()]
    setup_style()
    print(f"datasets: {', '.join(datasets)}")
    for ds in datasets:
        fig_recall_latency(ds)
        fig_planner_choices(ds)
    df = fig_planner_comparison(datasets)
    table_planner(df)
    table_best_plan(datasets)
    table_indexes(datasets)


if __name__ == "__main__":
    sys.exit(main())
