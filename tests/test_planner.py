import numpy as np
import pandas as pd

from vdb.data import attributes
from vdb.planner import CoarseStats, CostModelPlanner, ThresholdRule


def test_coarse_stats_local_selectivity():
    rng = np.random.default_rng(0)
    base = np.concatenate([rng.normal(-5, 1, (5000, 8)), rng.normal(5, 1, (5000, 8))]).astype(np.float32)
    labels = np.r_[np.zeros(5000, np.int32), np.ones(5000, np.int32)]  # label = which blob
    st = CoarseStats.build(base, labels, n_cells=32)
    inside = st.features(np.full(8, -5, np.float32), 0)
    outside = st.features(np.full(8, 5, np.float32), 0)
    assert abs(inside[0] - np.log10(0.5)) < 1e-6
    assert inside[4] > 0.2 and outside[4] < -1  # log ratio: label common near query vs absent


def synthetic_table(n=400, seed=0):
    """Two configs: 'fast' has good recall only when feature x > 0, 'slow' always good."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    X = np.stack([x, rng.standard_normal(n)], 1)
    rows = []
    for i in range(n):
        rows.append((i, "fast@1", 1.0 if x[i] > 0 else 0.3, 0.1))
        rows.append((i, "slow@1", 1.0, 1.0))
    return X, pd.DataFrame(rows, columns=["qi", "config", "recall", "latency_ms"])


def test_cost_model_learns_when_fast_is_safe():
    X, table = synthetic_table()
    m = CostModelPlanner(target=0.95).fit(X, table)
    assert m.choose(np.array([2.0, 0.0])) == "fast@1"
    assert m.choose(np.array([-2.0, 0.0])) == "slow@1"


def test_threshold_rule():
    sel = np.repeat([0.001, 0.5], 50)
    rows = []
    for i, s in enumerate(sel):
        rows.append((i, "prefilter@0", 1.0, 0.01 if s < 0.01 else 10.0))
        rows.append((i, "ann@8", 0.5 if s < 0.01 else 1.0, 0.1))
    rule = ThresholdRule(0.95).fit(sel, pd.DataFrame(rows, columns=["qi", "config", "recall", "latency_ms"]))
    assert rule.choose(0.001) == "prefilter@0"
    assert rule.choose(0.5) == "ann@8"


def test_fast_features_match_numpy_reference():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((5000, 16)).astype(np.float32)
    labels = attributes.independent_labels(5000)
    st = CoarseStats.build(base, labels, n_cells=32)
    q = rng.standard_normal(16).astype(np.float32)
    for lab in (0, 3, 7):
        d2 = ((st.centroids - q) ** 2).sum(1)
        top = np.argsort(d2)
        lc = st.counts[:, lab]
        sel = max(st.label_totals[lab], 0.5) / st.n
        l4 = (lc[top[:4]].sum() + 0.5) / (st.sizes[top[:4]].sum() + 1)
        l16 = (lc[top[:16]].sum() + 0.5) / (st.sizes[top[:16]].sum() + 1)
        ref = np.log10([sel, max(st.label_totals[lab], 0.5), l4, l16, l4 / sel, l16 / sel])
        np.testing.assert_allclose(st.features(q, lab), ref, rtol=1e-5, atol=1e-6)


def test_per_bucket_calibration_meets_target_in_every_bucket():
    """Bucket A (log_sel=-3): 'fast' fails half the time unless feature 1 > 0.
    Bucket B (log_sel=0): 'fast' always fine. Overall-mean calibration could
    sacrifice bucket A; per-bucket calibration must not."""
    rng = np.random.default_rng(2)
    n = 600
    log_sel = np.repeat([-3.0, 0.0], n // 2)
    f1 = rng.standard_normal(n)
    X = np.stack([log_sel, f1], 1)
    rows = []
    for i in range(n):
        good = log_sel[i] == 0.0 or f1[i] > 0
        rows.append((i, "fast@1", 1.0 if good else 0.5, 0.1))
        rows.append((i, "slow@1", 1.0, 1.0))
    table = pd.DataFrame(rows, columns=["qi", "config", "recall", "latency_ms"])
    m = CostModelPlanner(target=0.95).fit(X, table)
    idx = m.choose_batch(X)
    rec = np.where(np.array(m.configs)[idx] == "fast@1",
                   table[table.config == "fast@1"].sort_values("qi").recall.to_numpy(), 1.0)
    for v in (-3.0, 0.0):
        assert rec[log_sel == v].mean() >= 0.95
    # the numba path agrees with the vectorised path
    assert all(m.choose_index(X[i]) == idx[i] for i in range(0, n, 37))


def test_oracle_meets_target_at_min_cost():
    from vdb.planner import oracle_choice

    rec = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    lat = np.array([[10.0, 1.0], [10.0, 1.0], [5.0, 1.0], [10.0, 1.0]])
    idx = oracle_choice(rec, lat, target=0.75)
    r = rec[np.arange(4), idx]
    assert r.mean() >= 0.75
    assert lat[np.arange(4), idx].sum() == 1 + 1 + 5 + 1  # gives up only query 0, the costliest to fix
