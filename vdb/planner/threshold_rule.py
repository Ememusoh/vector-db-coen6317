"""Baseline: pre-filter when selectivity < theta, otherwise one fixed (plan, budget).

theta and the fallback configuration are chosen on the validation table to
minimise mean latency subject to mean recall >= target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ThresholdRule:
    def __init__(self, target: float = 0.95):
        self.target = target
        self.theta = 0.0
        self.fallback = None

    def fit(self, sel: np.ndarray, table: pd.DataFrame, prefilter: str = "prefilter@0"):
        """sel: (Q,) selectivity per training query; table: qi, config, recall, latency_ms."""
        Q = sel.size
        configs = sorted(table.config.unique())
        rec = table.pivot(index="qi", columns="config", values="recall").loc[np.arange(Q), configs]
        lat = table.pivot(index="qi", columns="config", values="latency_ms").loc[np.arange(Q), configs]
        levels = np.unique(sel)
        thetas = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2, [np.inf]])
        best = None
        for theta in thetas:
            pre = sel < theta
            for c in configs:
                r = np.where(pre, rec[prefilter], rec[c]).mean()
                t = np.where(pre, lat[prefilter], lat[c]).mean()
                if r >= self.target and (best is None or t < best[2]):
                    best = (theta, c, t, r)
        if best is None:  # nothing meets the target: always pre-filter (exact)
            best = (np.inf, prefilter, lat[prefilter].mean(), rec[prefilter].mean())
        self.theta, self.fallback = best[0], best[1]
        return self

    def choose(self, sel: float) -> str:
        return "prefilter@0" if sel < self.theta else self.fallback
