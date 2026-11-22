from __future__ import annotations

import numpy as np


def recall_at_k(found: np.ndarray, gt: np.ndarray, k: int) -> float:
    """|found[:k] ∩ gt[:k]| / |valid gt[:k]|. gt rows padded with -1 when fewer
    than k vectors are eligible; such queries are scored on what exists."""
    truth = gt[:k]
    truth = truth[truth >= 0]
    if truth.size == 0:
        return 1.0
    f = found[:k]
    return float(np.isin(truth, f[f >= 0]).sum()) / truth.size
