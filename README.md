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

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

SIFT, GIST, Deep and GloVe (with synthetic labels), plus YFCC (with native labels). Download scripts will go in `scripts/`. Data is stored under `datasets/` and is not committed.

## Reproducing experiments

TBD.
