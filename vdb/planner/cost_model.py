"""Cost-model planner.

For every configuration c = (plan, budget) seen in the validation sweep we fit
two small models on query features x:
    P(recall_c(x) >= recall_hi)  logistic regression
    log latency_c(x)              ridge regression
At query time: pick the configuration with the lowest predicted latency among
those with predicted probability >= tau; if none qualifies, the most likely
one. tau is calibrated on out-of-fold validation predictions so that the mean
recall of the chosen configurations meets the target.

All models are linear in standardized features, so inference for every
configuration is one small matrix-vector product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES


def config_key(plan: str, budget: int) -> str:
    return f"{plan}@{int(budget)}"


class CostModelPlanner:
    def __init__(self, target: float = 0.95, recall_hi: float = 0.9, ridge: float = 1.0, C: float = 1.0):
        self.target = target
        self.recall_hi = recall_hi
        self.ridge = ridge
        self.C = C
        self.tau = 0.5

    # ---------- fitting ----------
    def _fit_models(self, X: np.ndarray, table: pd.DataFrame, query_index: np.ndarray):
        """X: (Q, F) features per training query; table: rows (qi, config, recall, latency_ms)."""
        from sklearn.linear_model import LogisticRegression

        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        configs = sorted(table.config.unique())
        F = X.shape[1]
        Wr, br, Wl, bl = np.zeros((len(configs), F)), np.zeros(len(configs)), np.zeros((len(configs), F)), np.zeros(len(configs))
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

    def _choose_idx(self, p, lat, tau):
        ok = p >= tau
        masked = np.where(ok, lat, np.inf)
        idx = masked.argmin(1)
        none = ~ok.any(1)
        idx[none] = p[none].argmax(1)
        return idx

    def choose(self, x: np.ndarray) -> str:
        p, lat = self.predict(x)
        return self.configs[self._choose_idx(p, lat, self.tau)[0]]

    def fit(self, X: np.ndarray, table: pd.DataFrame, groups: np.ndarray | None = None,
            n_folds: int = 5, seed: int = 0, taus=None):
        """X: (Q, F); table columns: qi, config, recall, latency_ms (one row per query x config).
        groups: (Q,) group id per query for cross-validation (e.g. query vector id)."""
        taus = np.linspace(0.05, 0.99, 48) if taus is None else taus
        Q = X.shape[0]
        groups = np.arange(Q) if groups is None else groups
        ug = np.unique(groups)
        fold_of = dict(zip(np.random.default_rng(seed).permutation(ug), np.arange(ug.size) % n_folds))
        folds = np.array([fold_of[g] for g in groups])

        rec = table.pivot(index="qi", columns="config", values="recall")
        lat = table.pivot(index="qi", columns="config", values="latency_ms")
        configs = sorted(table.config.unique())
        rec, lat = rec.loc[np.arange(Q), configs].to_numpy(), lat.loc[np.arange(Q), configs].to_numpy()

        # out-of-fold predictions for tau calibration
        P = np.zeros((Q, len(configs)))
        Lp = np.zeros((Q, len(configs)))
        for f in range(n_folds):
            tr = np.flatnonzero(folds != f)
            te = np.flatnonzero(folds == f)
            self._fit_models(X[tr], table[table.qi.isin(tr)].assign(qi=lambda d: np.searchsorted(tr, d.qi)), tr)
            pos = [self.configs.index(c) if c in self.configs else None for c in configs]
            p, l = self.predict(X[te])
            for j, k in enumerate(pos):
                P[te, j] = p[:, k] if k is not None else 0.0
                Lp[te, j] = l[:, k] if k is not None else 1e9

        self.calibration = []
        best = None
        for tau in taus:
            idx = self._choose_idx(P, Lp, tau)
            r = rec[np.arange(Q), idx].mean()
            t = lat[np.arange(Q), idx].mean()
            self.calibration.append((tau, r, t))
            if r >= self.target and (best is None or t < best[2]):
                best = (tau, r, t)
        self.tau = best[0] if best else taus[-1]
        self._fit_models(X, table, np.arange(Q))
        return self
