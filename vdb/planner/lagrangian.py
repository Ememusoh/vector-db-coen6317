"""Lagrangian cost-model planner (default planner).

For every configuration c = (plan, budget) we fit two small linear models on
the query features x:
    E[recall_c | x]      fractional logistic regression (trained on the recall value itself)
    log latency_c(x)     ridge regression
At query time we pick
    argmin_c  latency_c(x) - lam * recall_c(x)
i.e. the same rule as the hindsight oracle, with predictions in place of the
measured values. lam is the price of recall in milliseconds: it is the
smallest value for which the mean recall of the out-of-fold validation choices
meets the target. Spending more on hard queries and less on easy ones is what
lets this rule meet the target with less latency than a per-query cutoff.

With per_bucket=True there is one lam per selectivity level, so every
selectivity meets the target on its own (stricter, slower).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cost_model import CostModelPlanner
from .fast import decide_lagrangian


class LagrangianPlanner(CostModelPlanner):
    def __init__(self, target: float = 0.95, margin: float = 0.0, per_bucket: bool = False,
                 ridge: float = 1.0, C: float = 1.0):
        super().__init__(target=target, margin=margin, per_bucket=per_bucket, ridge=ridge, C=C)
        self.lams = np.array([1.0])

    # ---------- models ----------
    def _fit_models(self, X: np.ndarray, table: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression

        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        configs = sorted(table.config.unique())
        F, n = X.shape[1], len(configs)
        Wr, br, Wl, bl = np.zeros((n, F)), np.zeros(n), np.zeros((n, F)), np.zeros(n)
        for i, c in enumerate(configs):
            t = table[table.config == c]
            Zi = Z[t.qi.to_numpy()]
            r = np.clip(t.recall.to_numpy(), 0.0, 1.0)
            if r.min() > 0.999 or r.max() < 0.001:
                br[i] = 20.0 if r.mean() > 0.5 else -20.0
            else:  # fractional logit: each query is a positive with weight r and a negative with weight 1 - r
                m = LogisticRegression(C=self.C, max_iter=2000).fit(
                    np.vstack([Zi, Zi]), np.r_[np.ones(len(r)), np.zeros(len(r))],
                    sample_weight=np.r_[r, 1.0 - r] + 1e-9)
                Wr[i], br[i] = m.coef_[0], m.intercept_[0]
            A = np.hstack([Zi, np.ones((Zi.shape[0], 1))])
            reg = self.ridge * np.eye(F + 1)
            reg[-1, -1] = 0.0
            w = np.linalg.solve(A.T @ A + reg, A.T @ np.log(t.latency_ms.to_numpy()))
            Wl[i], bl[i] = w[:-1], w[-1]
        self.configs = configs
        self.Wr, self.br, self.Wl, self.bl = Wr, br, Wl, bl

    def predict(self, X: np.ndarray):
        """Expected recall and latency in ms, (n, C) each."""
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(Z @ self.Wr.T + self.br))), np.exp(Z @ self.Wl.T + self.bl)

    @staticmethod
    def _pick(r, lat, lam):
        lam = np.broadcast_to(np.asarray(lam, float), (r.shape[0],))[:, None]
        return np.argmin(lat - lam * r, axis=1)

    # ---------- decision ----------
    def choose_index(self, x: np.ndarray) -> int:
        return int(decide_lagrangian(np.asarray(x, np.float64), self.mu, self.sd, self.Wr, self.br,
                                     self.Wl, self.bl, self.bucket_edges, self.lams))

    def choose_batch(self, X: np.ndarray) -> np.ndarray:
        r, lat = self.predict(X)
        return self._pick(r, lat, self.lams[self.bucket_of(X[:, 0])])

    # ---------- fitting ----------
    def fit(self, X: np.ndarray, table: pd.DataFrame, groups: np.ndarray | None = None,
            n_folds: int = 5, seed: int = 0):
        """Same inputs as CostModelPlanner.fit."""
        Q = X.shape[0]
        groups = np.arange(Q) if groups is None else groups
        ug = np.unique(groups)
        fold_of = dict(zip(np.random.default_rng(seed).permutation(ug), np.arange(ug.size) % n_folds))
        folds = np.array([fold_of[g] for g in groups])

        configs = sorted(table.config.unique())
        rec = table.pivot(index="qi", columns="config", values="recall").loc[np.arange(Q), configs].to_numpy()

        R = np.zeros((Q, len(configs)))           # out-of-fold predictions
        L = np.full((Q, len(configs)), 1e9)
        for f in range(n_folds):
            tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            sub = table[table.qi.isin(tr)]
            self._fit_models(X[tr], sub.assign(qi=np.searchsorted(tr, sub.qi.to_numpy())))
            r, lat = self.predict(X[te])
            for j, c in enumerate(configs):
                if c in self.configs:
                    k = self.configs.index(c)
                    R[te, j], L[te, j] = r[:, k], lat[:, k]

        goal = self.target + self.margin
        self.bucket_edges = self._make_buckets(X[:, 0])
        bucket = self.bucket_of(X[:, 0])
        self.lams = np.empty(self.bucket_edges.size + 1)
        for b in range(self.lams.size):
            rows = np.flatnonzero(bucket == b)
            if rows.size == 0:
                self.lams[b] = np.nan
                continue
            self.lams[b] = self._calibrate(R[rows], L[rows], rec[rows], goal)
        self.lams[np.isnan(self.lams)] = np.nanmax(self.lams)
        self._fit_models(X, table)
        return self

    def _calibrate(self, R, L, rec, goal, iters: int = 50):
        """Smallest lam whose out-of-fold choices reach mean recall >= goal."""
        rows = np.arange(R.shape[0])

        def ok(lam):
            return rec[rows, self._pick(R, L, lam)].mean() >= goal

        lo, hi = 0.0, 1.0
        while not ok(hi) and hi < 1e7:
            hi *= 2
        for _ in range(iters):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        return hi
