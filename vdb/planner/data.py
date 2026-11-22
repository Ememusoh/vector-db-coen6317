"""Turn sweep results into planner training/evaluation matrices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cost_model import config_key

RESULTS = Path(__file__).resolve().parents[2] / "results"


def load_split(dataset: str, schemes, split: str, queries: np.ndarray, stats: dict, results_dir: Path = RESULTS):
    """Stack the per-query sweep rows of every scheme.

    Returns
        X     (Q, F) planner features per (scheme, workload row)
        table long format: qi, config, recall, latency_ms (one row per query x config)
        meta  per query: scheme, row, query_id, label, selectivity
    """
    feats, tables, metas, offset = [], [], [], 0
    for s in schemes:
        df = pd.read_parquet(results_dir / "raw" / f"{dataset}_{s}_{split}.parquet")
        rows = np.sort(df.row.unique())
        first = df.drop_duplicates("row").set_index("row").loc[rows]
        feats.append(stats[s].features_batch(queries[first.query_id.to_numpy()], first.label.to_numpy()))
        metas.append(pd.DataFrame({"scheme": s, "row": rows, "query_id": first.query_id.to_numpy(),
                                   "label": first.label.to_numpy(), "selectivity": first.selectivity.to_numpy()}))
        t = df.assign(qi=offset + np.searchsorted(rows, df.row.to_numpy()),
                      config=[config_key(p, b) for p, b in zip(df.plan, df.budget)])
        tables.append(t[["qi", "config", "recall", "latency_ms"]])
        offset += rows.size
    return np.vstack(feats), pd.concat(tables, ignore_index=True), pd.concat(metas, ignore_index=True)


def matrices(table: pd.DataFrame, n_queries: int, configs=None):
    """Wide (Q, C) recall and latency matrices, columns in `configs` order."""
    configs = sorted(table.config.unique()) if configs is None else list(configs)
    idx = np.arange(n_queries)
    rec = table.pivot(index="qi", columns="config", values="recall").reindex(index=idx, columns=configs)
    lat = table.pivot(index="qi", columns="config", values="latency_ms").reindex(index=idx, columns=configs)
    return rec.to_numpy(), lat.to_numpy(), configs
