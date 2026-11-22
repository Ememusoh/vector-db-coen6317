from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .context import FilterContext


@dataclass
class PlanResult:
    ids: np.ndarray    # (k,) base ids satisfying the filter, -1 padded
    dists: np.ndarray  # (k,) squared L2, inf padded
    ndis: int          # total distance computations (all rounds)
    rounds: int = 1    # post-filter expansion rounds
    candidates: int = 0  # candidates examined after the ANN step (post-filter)
    chosen: str = ""     # plan@budget actually executed (set by planners)
    overhead_ms: float = 0.0  # planning time (features + decision), included in latency


def pad(ids, dists, k):
    out_i = np.full(k, -1, np.int64)
    out_d = np.full(k, np.inf, np.float32)
    n = min(k, ids.size)
    out_i[:n] = ids[:n]
    out_d[:n] = dists[:n]
    return out_i, out_d


class Plan(ABC):
    """A filtered k-NN strategy. `budget` is the knob that trades recall for
    work: nprobe for IVF plans, ef for HNSW plans, ignored for exact plans."""

    name: str = "plan"
    index: str = "none"         # which vector index it uses: exact | ivf | ivf_part | hnsw
    strategy: str = "none"      # pre | post | in
    budgets: tuple = (0,)

    def __init__(self, ctx: FilterContext):
        self.ctx = ctx

    @abstractmethod
    def search(self, q: np.ndarray, label: int, k: int, budget: int) -> PlanResult: ...

    def __repr__(self):
        return f"{type(self).__name__}()"
