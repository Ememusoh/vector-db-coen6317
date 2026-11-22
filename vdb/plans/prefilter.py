"""Pre-filtering: look up the eligible ids, then search them exactly."""
from __future__ import annotations

import numpy as np

from ..index import kernels
from .base import Plan, PlanResult


class PreFilter(Plan):
    name = "prefilter"
    index = "exact"
    strategy = "pre"
    budgets = (0,)

    def search(self, q, label, k, budget=0):
        cand = self.ctx.attr.eligible(label)
        ids, d, ndis = kernels.exact_subset_search(np.ascontiguousarray(q, np.float32), self.ctx.base, cand, k)
        return PlanResult(ids, d, int(ndis))
