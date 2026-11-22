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
