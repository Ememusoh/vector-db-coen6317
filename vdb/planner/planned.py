"""Plans that pick another plan per query (planner, threshold rule, fixed config)."""
from __future__ import annotations

import time

from ..plans.base import Plan, PlanResult
from .features import CoarseStats


def _split(config: str):
    name, b = config.rsplit("@", 1)
    return name, int(b)


class _Chooser(Plan):
    strategy = "auto"
    index = "auto"
    budgets = (0,)

    def __init__(self, ctx, plans: dict):
        super().__init__(ctx)
        self.plans = plans  # name -> Plan

    def _decide(self, q, label) -> str:
        raise NotImplementedError

    def search(self, q, label, k, budget=0):
        t0 = time.perf_counter()
        config = self._decide(q, label)
        overhead = (time.perf_counter() - t0) * 1e3
        name, b = _split(config)
        r = self.plans[name].search(q, label, k, b)
        r.chosen, r.overhead_ms = config, overhead
        return r


class PlannedSearch(_Chooser):
    name = "planner"

    def __init__(self, ctx, plans, model, stats: CoarseStats):
        super().__init__(ctx, plans)
        self.model, self.stats = model, stats

    def _decide(self, q, label):
        return self.model.choose(self.stats.features(q, label))


class ThresholdSearch(_Chooser):
    name = "threshold"

    def __init__(self, ctx, plans, rule):
        super().__init__(ctx, plans)
        self.rule = rule

    def _decide(self, q, label):
        return self.rule.choose(self.ctx.attr.selectivity(label))


class FixedSearch(_Chooser):
    name = "fixed"

    def __init__(self, ctx, plans, config: str):
        super().__init__(ctx, plans)
        self.config = config

    def _decide(self, q, label):
        return self.config
