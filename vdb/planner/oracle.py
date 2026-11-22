"""Oracle: per-query best configuration with hindsight, as an upper bound.

Each query takes argmin(latency - lam * recall); lam is the smallest value
(found by bisection) for which mean recall meets the target. This is the
optimal per-query assignment for the Lagrangian relaxation of
"minimise mean latency s.t. mean recall >= target".
"""
from __future__ import annotations

import numpy as np


def oracle_choice(rec: np.ndarray, lat: np.ndarray, target: float, iters: int = 60) -> np.ndarray:
    """rec, lat: (Q, C). Returns chosen config index per query."""
    def pick(lam):
        return np.argmin(lat - lam * rec, axis=1)

    rows = np.arange(rec.shape[0])
    lo, hi = 0.0, 1.0
    while rec[rows, pick(hi)].mean() < target and hi < 1e9:
        hi *= 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        if rec[rows, pick(mid)].mean() >= target:
            hi = mid
        else:
            lo = mid
    return pick(hi)
