from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    ids: np.ndarray    # (k,) int64 base ids, -1 = no result
    dists: np.ndarray  # (k,) float32 squared L2
    ndis: int          # distance computations performed


NO_LABELS = np.empty(0, np.int32)


def as_query(q) -> np.ndarray:
    return np.ascontiguousarray(q, dtype=np.float32)
