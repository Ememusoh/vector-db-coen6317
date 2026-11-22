"""Cost-model planner.

For every configuration c = (plan, budget) seen in the validation sweep we fit
two small models on query features x:
    P(recall_c(x) >= recall_hi)  logistic regression
    log latency_c(x)              ridge regression
At query time: pick the configuration with the lowest predicted latency among
those with predicted probability >= tau; if none qualifies, the most likely
one.

tau is calibrated on out-of-fold validation predictions. With per_bucket=True
(default) there is one tau per selectivity bucket, so every selectivity range
meets the recall target on its own rather than only on average. `margin` is
added to the target during calibration as a safety buffer for unseen queries.

All models are linear in standardized features; the per-query decision runs in
a Numba kernel (fast.decide).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fast import decide


def config_key(plan: str, budget: int) -> str:
    return f"{plan}@{int(budget)}"


class CostModelPlanner:
    def __init__(self, target: float = 0.95, margin: float = 0.0, per_bucket: bool = True,
                 recall_hi: float = 0.9, ridge: float = 1.0, C: float = 1.0):
        self.target = target
        self.margin = margin
        self.per_bucket = per_bucket
        self.recall_hi = recall_hi
        self.ridge = ridge
        self.C = C
        self.bucket_edges = np.empty(0)
        self.taus = np.array([0.5])

    # ---------- models ----------
    def _fit_models(self, X: np.ndarray, table: pd.DataFrame):
        """X: (Q, F) features per training query; table rows: qi, config, recall, latency_ms."""
        from sklearn.linear_model import LogisticRegression

        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        configs = sorted(table.config.unique())
        F = X.shape[1]
        n = len(configs)
        Wr, br, Wl, bl = np.zeros((n, F)), np.zeros(n), np.zeros((n, F)), np.zeros(n)
        for i, c in enumerate(configs):
            t = table[table.config == c]
            Zi = Z[t.qi.to_numpy()]
            y = (t.recall.to_numpy() >= self.recall_hi).astype(int)
            if y.min() == y.max():
                br[i] = 20.0 if y[0] == 1 else -20.0
            else:
                m = LogisticRegression(C=self.C, max_iter=1000).fit(Zi, y)
                Wr[i], br[i] = m.coef_[0], m.intercept_[0]
            ylat = np.log(t.latency_ms.to_numpy())
            A = np.hstack([Zi, np.ones((Zi.shape[0], 1))])
            reg = self.ridge * np.eye(F + 1)
            reg[-1, -1] = 0.0
            w = np.linalg.solve(A.T @ A + reg, A.T @ ylat)
            Wl[i], bl[i] = w[:-1], w[-1]
        self.configs = configs
        self.Wr, self.br, self.Wl, self.bl = Wr, br, Wl, bl

    def predict(self, X: np.ndarray):
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-(Z @ self.Wr.T + self.br)))
        lat = Z @ self.Wl.T + self.bl
        return p, lat  # (n, C) each

    @staticmethod
    def _choose_idx(p, lat, tau):
        tau = np.broadcast_to(np.asarray(tau, float), (p.shape[0],))[:, None]
        ok = p >= tau
        idx = np.where(ok, lat, np.inf).argmin(1)
        none = ~ok.any(1)
        idx[none] = p[none].argmax(1)
        return idx

    # ---------- buckets ----------
    def _make_buckets(self, log_sel: np.ndarray):
        if not self.per_bucket:
            return np.empty(0)
        levels = np.unique(np.round(log_sel, 6))
        if levels.size <= 10:
            return (levels[:-1] + levels[1:]) / 2
        return np.quantile(log_sel, [0.2, 0.4, 0.6, 0.8])

    def bucket_of(self, log_sel):
        return np.searchsorted(self.bucket_edges, log_sel)

    # ---------- decision ----------
    def choose_index(self, x: np.ndarray) -> int:
        return int(decide(np.asarray(x, np.float64), self.mu, self.sd, self.Wr, self.br, self.Wl, self.bl,
                          self.bucket_edges, self.taus))

    def choose(self, x: np.ndarray) -> str:
        return self.configs[self.choose_index(x)]

    def choose_batch(self, X: np.ndarray) -> np.ndarray:
        p, lat = self.predict(X)
        return self._choose_idx(p, lat, self.taus[self.bucket_of(X[:, 0])])

    # ---------- fitting ----------
    def fit(self, X: np.ndarray, table: pd.DataFrame, groups: np.ndarray | None = None,
            n_folds: int = 5, seed: int = 0, taus=None):
        """X: (Q, F), column 0 = log10 selectivity; table columns: qi, config, recall, latency_ms
        (one row per query x config). groups: (Q,) group id per query for cross-validation
        (e.g. the query vector id, so one vector never sits in both train and calibration folds)."""
        grid = np.linspace(0.05, 0.995, 64) if taus is None else np.asarray(taus)
        Q = X.shape[0]
        groups = np.arange(Q) if groups is None else groups
        ug = np.unique(groups)
        fold_of = dict(zip(np.random.default_rng(seed).permutation(ug), np.arange(ug.size) % n_folds))
        folds = np.array([fold_of[g] for g in groups])

        configs = sorted(table.config.unique())
        rec = table.pivot(index="qi", columns="config", values="recall").loc[np.arange(Q), configs].to_numpy()
        lat = table.pivot(index="qi", columns="config", values="latency_ms").loc[np.arange(Q), configs].to_numpy()

        # out-of-fold predictions for calibration
        P = np.zeros((Q, len(configs)))
        Lp = np.full((Q, len(configs)), 1e9)
        for f in range(n_folds):
            tr = np.flatnonzero(folds != f)
            te = np.flatnonzero(folds == f)
            sub = table[table.qi.isin(tr)]
            self._fit_models(X[tr], sub.assign(qi=np.searchsorted(tr, sub.qi.to_numpy())))
            p, l = self.predict(X[te])
            for j, c in enumerate(configs):
                if c in self.configs:
                    k = self.configs.index(c)
                    P[te, j], Lp[te, j] = p[:, k], l[:, k]

        goal = self.target + self.margin
        self.bucket_edges = self._make_buckets(X[:, 0])
        bucket = self.bucket_of(X[:, 0])
        self.taus = np.empty(self.bucket_edges.size + 1)
        self.calibration = []
        for b in range(self.taus.size):
            rows = np.flatnonzero(bucket == b)
            best = None
            for tau in grid:
                idx = self._choose_idx(P[rows], Lp[rows], tau)
                r = rec[rows, idx].mean()
                t = lat[rows, idx].mean()
                self.calibration.append((b, tau, r, t))
                if r >= goal and (best is None or t < best[1]):
                    best = (tau, t)
            self.taus[b] = best[0] if best else grid[-1]
        self._fit_models(X, table)
        return self
