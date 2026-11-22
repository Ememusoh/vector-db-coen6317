# Filtered Vector Search: Pre-, Post- and In-Search Filtering with a Cost-Based Planner

COEN 6317 course project.

We compare three ways of answering filtered nearest-neighbor queries ("the 10 closest vectors whose attribute = X") on IVF and HNSW indexes:

- **Pre-filter:** apply the filter first, then search exactly over the vectors that pass.
- **Post-filter:** run ANN search, then filter, widening the candidate set until enough results pass.
- **In-search filter:** check the filter during index traversal.

We also build an **attribute-partitioned IVF** and a **cost-model planner** that picks a plan for each query. The planner is compared against a simple selectivity-threshold rule.

## Layout

```
vdb/
  data/      dataset loaders, subsetting, synthetic attributes, ground truth
  index/     IVF, partitioned IVF, HNSW (FAISS build, own instrumented search)
  plans/     pre-filter, post-filter, in-search filter
  planner/   query features, cost model, threshold baseline
  eval/      metrics (recall@k, latency, distance counts), experiment runner
scripts/     dataset download and experiment and plotting scripts
configs/     experiment configs
tests/       unit tests
```

## Setup

Requires Python 3.10+ (we use 3.12).

```bash
python3.12 -m venv .venv        # or: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                          # run the tests
```

## Datasets

SIFT, GIST, Deep and GloVe (with synthetic labels), plus YFCC (with native labels, coming later). Data is stored under `datasets/` and is not committed.

```bash
python scripts/prepare_dataset.py sift     # also: gist, glove, deep
```

This downloads the dataset and, for each attribute scheme (`independent`, `correlated`), writes:

| File | Contents |
|------|----------|
| `base.npy`, `queries.npy` | float32 vectors (angular datasets are L2-normalised) |
| `<scheme>/labels.npy` | one integer label per base vector |
| `<scheme>/workload.npz` | (query, label) pairs at 50%, 10%, 1%, 0.1% selectivity, split into validation (30%) / test (70%) by query |
| `<scheme>/gt.npz` | exact filtered top-100 ids and squared L2 distances (-1 = fewer eligible vectors than k) |

Labels 0 to 3 hit those selectivities exactly; other labels fill the rest at about 1% each. In the `correlated` scheme, each label covers a compact region of the vector space.

## Indexes

```bash
python scripts/build_indexes.py sift      # IVF (nlist ~ 4 sqrt N) and HNSW (M=32, efC=200)
python scripts/check_indexes.py sift      # our search vs FAISS: recall, latency, distance counts
```

FAISS builds the structures (k-means centroids, HNSW graph). All searching runs through our own Numba kernels (`vdb/index/kernels.py`), which count every distance computation, so every plan is measured the same way. The attribute-partitioned IVF is derived from the IVF index per label scheme in a few seconds, so it isn't saved separately.

## Plans

Six plans share one interface, `plan.search(query, label, k, budget)` (`vdb/plans/`):

| Plan | Strategy | Budget knob |
|------|----------|-------------|
| `prefilter` | look up matching ids, exact search over them | none |
| `ivf_post`, `hnsw_post` | unfiltered ANN for ~k/selectivity candidates, drop non-matching, double and retry if short | nprobe / ef |
| `ivf_in` | skip non-matching list entries before computing distances | nprobe |
| `ivfpart_in` | jump to the label's slice of each partitioned list | nprobe |
| `hnsw_in` | traverse any node, only matching nodes become results | ef |

## Reproducing experiments

```bash
python scripts/sweep_plans.py sift independent --per-selectivity 100 --split val
```

Writes per-query rows to `results/raw/` and a per-plan/budget/selectivity summary to `results/`.
