import numpy as np
import pytest

from vdb.data import attributes, groundtruth, workload
from vdb.eval import recall_at_k, sweep
from vdb.index import HNSWIndex, IVFIndex
from vdb.plans import (FilterContext, HNSWPostFilter, IVFPostFilter, PreFilter, make_plans)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((50, 32)).astype(np.float32) * 4
    base = (centers[rng.integers(0, 50, 20_000)] + rng.standard_normal((20_000, 32))).astype(np.float32)
    queries = (centers[rng.integers(0, 50, 40)] + rng.standard_normal((40, 32))).astype(np.float32)
    labels = attributes.independent_labels(base.shape[0])
    ctx = FilterContext.from_parts(base, labels, IVFIndex.build(base, 64), HNSWIndex.build(base, 16, 100))
    wl = workload.build_workload(40, labels, [0, 1, 2, 3], max_queries=None)
    gt, _ = groundtruth.filtered_knn(base, queries, labels, wl.query_ids, wl.labels, k=10)
    return ctx, queries, wl, gt


def test_attribute_index(setup):
    ctx = setup[0]
    for lab in range(5):
        np.testing.assert_array_equal(ctx.attr.eligible(lab), np.flatnonzero(ctx.labels == lab))
    assert ctx.attr.count(10_000) == 0


def test_all_plans_share_interface_and_respect_filter(setup):
    ctx, queries, wl, gt = setup
    plans = make_plans(ctx)
    assert {p.name for p in plans} == {"prefilter", "ivf_post", "ivf_in", "ivfpart_in", "hnsw_post", "hnsw_in"}
    for p in plans:
        for r in range(0, len(wl), 7):
            res = p.search(queries[wl.query_ids[r]], int(wl.labels[r]), 10, p.budgets[-1])
            got = res.ids[res.ids >= 0]
            assert res.ids.shape == (10,)
            assert np.all(ctx.labels[got] == wl.labels[r]), p.name
            assert np.all(np.diff(res.dists[: got.size]) >= 0), p.name
            assert res.ndis > 0


def test_prefilter_is_exact(setup):
    ctx, queries, wl, gt = setup
    p = PreFilter(ctx)
    for r in range(len(wl)):
        res = p.search(queries[wl.query_ids[r]], int(wl.labels[r]), 10)
        assert recall_at_k(res.ids, gt[r], 10) == 1.0
        assert res.ndis == ctx.attr.count(int(wl.labels[r]))


@pytest.mark.parametrize("cls,budget", [(IVFPostFilter, 64), (HNSWPostFilter, 256)])
def test_postfilter_high_budget_recall(setup, cls, budget):
    ctx, queries, wl, gt = setup
    p = cls(ctx)
    recs = [recall_at_k(p.search(queries[wl.query_ids[r]], int(wl.labels[r]), 10, budget).ids, gt[r], 10)
            for r in range(len(wl))]
    assert np.mean(recs) > 0.95


def test_postfilter_expands_when_short(setup):
    ctx, queries, wl, gt = setup
    p = HNSWPostFilter(ctx, oversample=0.05)  # deliberately too few initial candidates
    r = np.flatnonzero(wl.labels == 2)[0]  # 1% selectivity
    res = p.search(queries[wl.query_ids[r]], 2, 10, 16)
    assert res.rounds > 1
    assert (res.ids >= 0).sum() == 10


def test_sweep_produces_rows(setup):
    ctx, queries, wl, gt = setup
    plans = make_plans(ctx, names=["prefilter", "ivf_in"])
    df = sweep(plans, queries, wl, gt, np.arange(len(wl)), budgets={"ivf_in": [4, 64]}, progress=False)
    assert len(df) == 3 * len(wl)
    assert df.groupby("plan").recall.mean()["prefilter"] == 1.0


def test_tie_aware_recall():
    gt = np.array([5, 6, 7])
    gd = np.array([1.0, 2.0, 2.0], np.float32)
    found = np.array([5, 6, 9])  # 9 ties with 7 at distance 2.0
    fd = np.array([1.0, 2.0, 2.0], np.float32)
    assert recall_at_k(found, gt, 3) == 2 / 3
    assert recall_at_k(found, gt, 3, fd, gd) == 1.0
    assert recall_at_k(found, gt, 3, np.array([1.0, 2.0, 2.5], np.float32), gd) == 2 / 3
