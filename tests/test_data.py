import numpy as np
import pytest

from vdb.data import attributes, groundtruth, io, workload


def test_fvecs_roundtrip(tmp_path):
    x = np.random.default_rng(0).standard_normal((50, 7)).astype(np.float32)
    io.write_fvecs(tmp_path / "x.fvecs", x)
    np.testing.assert_array_equal(io.read_fvecs(tmp_path / "x.fvecs"), x)
    np.testing.assert_array_equal(io.read_fvecs(tmp_path / "x.fvecs", count=10), x[:10])


def test_label_sizes_hit_targets():
    sizes, targets = attributes.label_sizes(100_000)
    assert sizes.sum() == 100_000
    np.testing.assert_array_equal(sizes[targets], [50_000, 10_000, 1_000, 100])


@pytest.mark.parametrize("scheme", attributes.SCHEMES)
def test_schemes_give_exact_selectivity(scheme):
    base = np.random.default_rng(1).standard_normal((20_000, 16)).astype(np.float32)
    labels = attributes.make_labels(scheme, base, seed=0, **({"n_clusters": 64} if scheme == "correlated" else {}))
    sel = workload.label_selectivity(labels)[:4]
    np.testing.assert_allclose(sel, [0.5, 0.1, 0.01, 0.001])


def test_correlated_labels_are_spatially_compact():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((20_000, 16)).astype(np.float32)
    ind = attributes.make_labels("independent", base)
    cor = attributes.make_labels("correlated", base, n_clusters=64)

    def spread(labels, lab):  # mean distance of members to their centroid
        m = base[labels == lab]
        return np.linalg.norm(m - m.mean(0), axis=1).mean()

    assert spread(cor, 2) < 0.9 * spread(ind, 2)


def test_groundtruth_matches_brute_force():
    rng = np.random.default_rng(3)
    base = rng.standard_normal((5_000, 12)).astype(np.float32)
    queries = rng.standard_normal((20, 12)).astype(np.float32)
    labels = attributes.independent_labels(base.shape[0])
    wl = workload.build_workload(20, labels, [0, 1, 2, 3], max_queries=None)
    ids, dists = groundtruth.filtered_knn(base, queries, labels, wl.query_ids, wl.labels, k=10)
    for i in range(len(wl)):
        ref_ids, ref_d = groundtruth.brute_force_filtered(base, queries[wl.query_ids[i]], labels, wl.labels[i], 10)
        n = ref_ids.size
        np.testing.assert_allclose(dists[i, :n], ref_d, rtol=1e-4, atol=1e-4)
        assert set(ids[i, :n]) == set(ref_ids)
        assert (ids[i, n:] == -1).all()  # label 3 has only 5 members -> padded
        assert np.all(labels[ids[i, :n]] == wl.labels[i])


def test_workload_split_is_by_query():
    labels = attributes.independent_labels(10_000)
    wl = workload.build_workload(500, labels, [0, 1, 2, 3], max_queries=200)
    val_q = set(wl.query_ids[wl.split == 0])
    test_q = set(wl.query_ids[wl.split == 1])
    assert not val_q & test_q
    assert len(val_q) == 60 and len(test_q) == 140
