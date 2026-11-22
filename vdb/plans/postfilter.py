"""Post-filtering: unfiltered ANN search, then drop non-matching results.

Round 0 asks the index for k' = ceil(oversample * k / selectivity) candidates
(selectivity is known exactly from the attribute index). If fewer than k
candidates pass the filter, both k' and the search budget double and the
search is re-run from scratch, as a client of an index without filter support
would. Distance computations are summed over all rounds.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Plan, PlanResult, pad


class _PostFilter(Plan):
    strategy = "post"

    def __init__(self, ctx, oversample: float = 1.0, max_rounds: int = 12):
        super().__init__(ctx)
        self.oversample = oversample
        self.max_rounds = max_rounds

    def _ann(self, q, kp, budget):  # -> SearchResult
        raise NotImplementedError

    def _budget_cap(self) -> int:
        raise NotImplementedError

    def search(self, q, label, k, budget):
        ctx = self.ctx
        n_elig = ctx.attr.count(label)
        if n_elig == 0:
            ids, d = pad(np.empty(0, np.int64), np.empty(0, np.float32), k)
            return PlanResult(ids, d, 0, rounds=0)
        sel = n_elig / ctx.n
        kp = min(ctx.n, max(k, math.ceil(self.oversample * k / sel)))
        b = min(budget, self._budget_cap())
        total, rounds = 0, 0
        while True:
            r = self._ann(q, kp, b)
            total += r.ndis
            rounds += 1
            valid = r.ids >= 0
            ids, dists = r.ids[valid], r.dists[valid]
            keep = ctx.labels[ids] == label
            enough = keep.sum() >= min(k, n_elig)
            exhausted = kp >= ctx.n or (b >= self._budget_cap() and valid.sum() < kp)
            if enough or exhausted or rounds >= self.max_rounds:
                break
            kp = min(ctx.n, 2 * kp)
            b = min(2 * b, self._budget_cap())
        out_i, out_d = pad(ids[keep], dists[keep], k)
        return PlanResult(out_i, out_d, total, rounds=rounds, candidates=int(ids.size))


class IVFPostFilter(_PostFilter):
    name = "ivf_post"
    index = "ivf"
    budgets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

    def _ann(self, q, kp, budget):
        return self.ctx.ivf.search(q, kp, nprobe=budget)

    def _budget_cap(self):
        return self.ctx.ivf.nlist


class HNSWPostFilter(_PostFilter):
    name = "hnsw_post"
    index = "hnsw"
    budgets = (10, 16, 32, 64, 128, 256, 512, 1024)

    def _ann(self, q, kp, budget):
        return self.ctx.hnsw.search(q, kp, ef=max(budget, kp))

    def _budget_cap(self):
        return self.ctx.n
