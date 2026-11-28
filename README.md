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

One command reproduces every result, figure and table in the report:

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt   # once
scripts/reproduce.sh                  # sift, glove, deep at full size
scripts/reproduce.sh --quick          # smoke run: 20 val / 10 test queries per selectivity
scripts/reproduce.sh --dry-run        # list the steps without running them
```

Per dataset it downloads the data and writes labels and ground truth, builds the indexes, runs the plan sweeps (validation for planner training, test for the oracle and matched-recall lookup) and runs the live planner evaluation. It then regenerates `results/figures/` and `results/tables/`. Steps whose outputs already exist are skipped (`--force` redoes them), logs go to `logs/`, and the software and hardware used are recorded in `results/environment.txt`. Our numbers come from an Apple M4 with 16 GB of memory. A full run needs about 10 GB of disk and takes several hours, mostly in the sweeps. Absolute latencies depend on the machine, but the rankings between methods should hold.

Individual steps can also be run by hand, for example:

```bash
python scripts/sweep_plans.py sift independent --per-selectivity 100 --split val
```

This writes per-query rows to `results/raw/` and a per-plan/budget/selectivity summary to `results/`.

### Planner

```bash
python scripts/evaluate_planner.py sift --target 0.95
```

Trains on the validation sweeps of every scheme and runs live on held-out test queries:

- `planner`: cost-model planner (`vdb/planner/lagrangian.py`). Features are global selectivity and *local* selectivity near the query, from a 256-cell histogram. Per (plan, budget) it predicts expected recall (fractional logistic regression) and log latency (ridge regression), then picks `argmin latency - λ·recall`, the same rule as the hindsight oracle. λ is the smallest value whose out-of-fold validation choices meet the target. The decision runs in Numba (~8 µs). `--planner cutoff` selects the earlier rule (fastest config whose P(high recall) passes a per-selectivity cutoff); `--per-bucket` calibrates one λ per selectivity level.

`python scripts/matched_recall.py` tunes every method for recall targets 0.85 to 0.99 and scores them on identical test executions (including the global-selectivity-only ablation and the earlier cutoff planner).
- `threshold`: pre-filter if selectivity < θ, else one fixed (plan, budget); θ and the fallback are tuned on validation.
- `fixed`: the single fastest (plan, budget) meeting the target on validation.

### Figures and tables

```bash
python scripts/make_figures.py          # every dataset with results
```

Writes PDF and PNG figures to `results/figures/` and Markdown and LaTeX (booktabs) tables to `results/tables/`:

- `recall_latency_<dataset>`: recall vs median latency for every plan across its budget sweep, per selectivity and label scheme
- `planner_comparison`: planner vs threshold rule vs best fixed config vs oracle, recall and mean latency
- `planner_choices_<dataset>`: share of test queries routed to each plan
- `tables/planner_results`, `tables/best_plan`, `tables/indexes`: the numbers behind the figures, plus build time and memory

PDFs embed TrueType fonts, as ACM requires. Missing inputs are skipped, so the script can run at any stage.

