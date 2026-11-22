import numpy as np
import pytest

from vdb.data import attributes, groundtruth
from vdb.index import HNSWIndex, IVFIndex, PartitionedIVF, kernels


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    # clustered data so IVF and HNSW behave like on real datasets
    centers = rng.standard_normal((50, 32)).astype(np.float32) * 4
    base = (centers[rng.integers(0, 50, 20_000)] + rng.standard_normal((20_000, 32))).astype(np.float32)
    queries = (centers[rng.integers(0, 50, 50)] + rng.standard_normal((50, 32))).astype(np.float32)
    labels = attributes.independent_labels(base.shape[0])
    return base, queries, labels


@pytest.fixture(scope="module")
def ivf(data):
    return IVFIndex.build(data[0], nlist=64)


@pytest.fixture(scope="module")
def hnsw(data):
    return HNSWIndex.build(data[0], M=16, ef_construction=100)


def exact(base, q, k, labels=None, label=-1):
    if label < 0:
        d = ((base - q) ** 2).sum(1)
        o = np.argsort(d)[:k]
        return o, d[o]
    return groundtruth.brute_force_filtered(base, q, labels, label, k)


def recall(found, truth):
    truth = truth[truth >= 0]
    return len(set(found[found >= 0]) & set(truth)) / max(1, len(truth))


def test_ivf_structure(data, ivf):
    base = data[0]
    assert ivf.offsets[-1] == base.shape[0]
    assert sorted(ivf.ids) == list(range(base.shape[0]))
    np.testing.assert_array_equal(ivf.vecs, base[ivf.ids])


def test_ivf_all_lists_is_exact(data, ivf):
    base, queries, labels = data
    ll = ivf.list_labels(labels)
    for q in queries[:10]:
        r = ivf.search(q, 10, nprobe=ivf.nlist)
        np.testing.assert_allclose(r.dists, exact(base, q, 10)[1], rtol=1e-4)
        assert r.ndis == ivf.nlist + base.shape[0]
        for lab in (0, 2, 3):
            r = ivf.search(q, 10, nprobe=ivf.nlist, label=lab, list_labels=ll)
            ref_ids, ref_d = exact(base, q, 10, labels, lab)
            np.testing.assert_allclose(r.dists[:ref_d.size], ref_d, rtol=1e-4)
            assert r.ndis == ivf.nlist + (labels == lab).sum()  # no distance for non-matching entries


def test_partitioned_matches_ivf_insearch(data, ivf):
    base, queries, labels = data
    pivf = PartitionedIVF.from_ivf(ivf, labels)
    ll = ivf.list_labels(labels)
    for q in queries[:10]:
        for lab in range(4):
            a = ivf.search(q, 10, nprobe=8, label=lab, list_labels=ll)
            b = pivf.search(q, 10, nprobe=8, label=lab)
            np.testing.assert_array_equal(np.sort(a.ids), np.sort(b.ids))
            assert a.ndis == b.ndis
        u = pivf.search(q, 10, nprobe=pivf.nlist)
        np.testing.assert_allclose(u.dists, exact(base, q, 10)[1], rtol=1e-4)


def test_hnsw_unfiltered_recall_matches_faiss(data, hnsw):
    import faiss

    base, queries, _ = data
    ref = faiss.IndexHNSWFlat(base.shape[1], 16)
    ref.hnsw.efConstruction = 100
    ref.add(base)
    ref.hnsw.efSearch = 64
    _, fi = ref.search(queries, 10)
    ours = np.mean([recall(hnsw.search(q, 10, ef=64).ids, exact(base, q, 10)[0]) for q in queries])
    theirs = np.mean([recall(fi[i], exact(base, q, 10)[0]) for i, q in enumerate(queries)])
    assert ours > 0.9
    assert abs(ours - theirs) < 0.05


def test_hnsw_filtered_returns_only_matching(data, hnsw):
    base, queries, labels = data
    recs = []
    for q in queries[:20]:
        r = hnsw.search(q, 10, ef=100, label=1, labels=labels)
        assert np.all(labels[r.ids[r.ids >= 0]] == 1)
        recs.append(recall(r.ids, exact(base, q, 10, labels, 1)[0]))
    assert np.mean(recs) > 0.9


def test_hnsw_max_ndis_caps_work(data, hnsw):
    _, queries, labels = data
    full = hnsw.search(queries[0], 10, ef=100, label=3, labels=labels)
    capped = hnsw.search(queries[0], 10, ef=100, label=3, labels=labels, max_ndis=500)
    assert capped.ndis < full.ndis
    assert capped.ndis < 500 + 64  # may overshoot by one adjacency list


def test_save_load_roundtrip(tmp_path, data, ivf, hnsw):
    base, queries, labels = data
    ivf.save(tmp_path / "ivf.npz")
    hnsw.save(tmp_path / "hnsw.npz")
    ivf2 = IVFIndex.load(tmp_path / "ivf.npz", base)
    hnsw2 = HNSWIndex.load(tmp_path / "hnsw.npz", base)
    q = queries[0]
    np.testing.assert_array_equal(ivf.search(q, 10, 8).ids, ivf2.search(q, 10, 8).ids)
    np.testing.assert_array_equal(hnsw.search(q, 10, 50).ids, hnsw2.search(q, 10, 50).ids)


def test_exact_subset_search(data):
    base, queries, labels = data
    cand = np.flatnonzero(labels == 2)
    ids, d, ndis = kernels.exact_subset_search(queries[0], base, cand, 10)
    ref_ids, ref_d = exact(base, queries[0], 10, labels, 2)
    np.testing.assert_allclose(d, ref_d, rtol=1e-4)
    assert ndis == cand.size
