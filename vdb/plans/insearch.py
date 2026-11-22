"""In-search filtering: the filter is checked during index traversal.

IVF: non-matching entries are skipped before their distance is computed.
Partitioned IVF: each list is grouped by label, so only the matching slice is read.
HNSW: the search may walk through any node, but only matching nodes become results.
"""
from __future__ import annotations

from .base import Plan, PlanResult


class IVFInSearch(Plan):
    name = "ivf_in"
    index = "ivf"
    strategy = "in"
    budgets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

    def search(self, q, label, k, budget):
        r = self.ctx.ivf.search(q, k, nprobe=budget, label=label, list_labels=self.ctx.ivf_list_labels)
        return PlanResult(r.ids, r.dists, r.ndis)


class PartitionedIVFInSearch(Plan):
    name = "ivfpart_in"
    index = "ivf_part"
    strategy = "in"
    budgets = IVFInSearch.budgets

    def search(self, q, label, k, budget):
        r = self.ctx.pivf.search(q, k, nprobe=budget, label=label)
        return PlanResult(r.ids, r.dists, r.ndis)


class HNSWInSearch(Plan):
    name = "hnsw_in"
    index = "hnsw"
    strategy = "in"
    budgets = (10, 16, 32, 64, 128, 256, 512)

    def __init__(self, ctx, max_ndis: int = 0):
        super().__init__(ctx)
        self.max_ndis = max_ndis

    def search(self, q, label, k, budget):
        r = self.ctx.hnsw.search(q, k, ef=budget, label=label, labels=self.ctx.labels, max_ndis=self.max_ndis)
        return PlanResult(r.ids, r.dists, r.ndis)
