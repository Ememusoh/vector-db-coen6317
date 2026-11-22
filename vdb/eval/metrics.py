from __future__ import annotations

import numpy as np


def recall_at_k(found: np.ndarray, gt: np.ndarray, k: int,
                found_dists: np.ndarray | None = None, gt_dists: np.ndarray | None = None,
                rtol: float = 1e-5) -> float:
    """Recall@k. gt rows are padded with -1 when fewer than k vectors are
    eligible; such queries are scored on what exists.

    With distances given, recall is tie-aware (as in ann-benchmarks): a returned
    vector counts if its distance is within the k-th true distance, so a
    different vector at exactly the same distance as a true neighbor is not
    penalised. Without distances, it is plain id overlap."""
    truth = gt[:k]
    n_true = int((truth >= 0).sum())
    if n_true == 0:
        return 1.0
    f = found[:k]
    valid = f >= 0
    if found_dists is None or gt_dists is None:
        return float(np.isin(truth[truth >= 0], f[valid]).sum()) / n_true
    thr = gt_dists[n_true - 1]
    thr = thr + rtol * abs(thr) + 1e-6
    return min(int((found_dists[:k][valid] <= thr).sum()), n_true) / n_true
