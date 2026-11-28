#!/usr/bin/env bash
# Reproduce every result, figure and table in the report with one command.
#
#   scripts/reproduce.sh                       # all datasets (sift gist glove deep yfcc), full sizes
#   scripts/reproduce.sh --datasets "sift"     # one dataset
#   scripts/reproduce.sh --quick               # 20 val / 10 test queries per selectivity (smoke run)
#   scripts/reproduce.sh --force               # redo steps whose outputs already exist
#   scripts/reproduce.sh --dry-run             # print the steps without running them
#
# Datasets: sift gist glove deep (synthetic labels, independent + correlated) and
# yfcc (native tags). Steps per dataset: download + labels + ground truth, build indexes, plan sweeps
# (validation: planner training; test: oracle and matched-recall lookup), live
# planner evaluation. Then cross-dataset analysis, figures and tables.
# Existing outputs are reused unless --force is given. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="sift gist glove deep yfcc"
VAL=200
TEST=100
FORCE=0
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --quick) VAL=20; TEST=10; shift ;;
    --val) VAL="$2"; shift 2 ;;
    --test) TEST="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

PY=${PYTHON:-.venv/bin/python}
if [ ! -x "$PY" ] && [ "$DRY" = 0 ]; then
  echo "No virtualenv at .venv. Create it first:" >&2
  echo "  uv venv --python 3.12 .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi
mkdir -p logs results/raw

step() {  # step <description> <output-that-marks-done> <log> <command...>
  local desc=$1 done_file=$2 log=$3; shift 3
  if [ "$FORCE" = 0 ] && [ -n "$done_file" ] && [ -e "$done_file" ]; then
    echo "[$(date +%T)] skip   $desc (found $done_file)"
    return
  fi
  echo "[$(date +%T)] run    $desc"
  if [ "$DRY" = 1 ]; then echo "         $*  > $log"; return; fi
  "$@" > "$log" 2>&1 || { echo "FAILED: $desc (see $log)" >&2; tail -20 "$log" >&2; exit 1; }
}

START=$(date +%s)
if [ "$DRY" = 0 ]; then
  step "unit tests" "" logs/tests.log "$PY" -m pytest -q
  {
    echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "os: $(uname -srm)"
    echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
    echo "memory_gb: $( (sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal/{print $2*1024}' /proc/meminfo) | awk '{printf "%.0f", $1/2^30}')"
    "$PY" -c "import sys, numpy, numba, faiss, sklearn; print('python:', sys.version.split()[0]); print('numpy:', numpy.__version__); print('numba:', numba.__version__); print('faiss:', faiss.__version__); print('scikit-learn:', sklearn.__version__)"
    echo "datasets: $DATASETS   val/test queries per selectivity: $VAL/$TEST"
  } > results/environment.txt
fi

for DS in $DATASETS; do
  D=datasets/$DS
  SCHEMES=$(PYTHONPATH=. "$PY" -c "from vdb.data.datasets import schemes; print(' '.join(schemes('$DS')))")
  LAST=${SCHEMES##* }
  step "$DS: download, labels, ground truth" "$D/$LAST/gt.npz" "logs/${DS}_prepare.log" \
    "$PY" scripts/prepare_dataset.py "$DS"
  step "$DS: build IVF and HNSW" "$(ls indexes/$DS/hnsw_*.npz 2>/dev/null | head -1)" "logs/${DS}_build.log" \
    "$PY" scripts/build_indexes.py "$DS"
  for S in $SCHEMES; do
    step "$DS/$S: validation sweep ($VAL per selectivity)" "results/raw/${DS}_${S}_val.parquet" \
      "logs/${DS}_${S}_val_sweep.log" "$PY" scripts/sweep_plans.py "$DS" "$S" --split val --per-selectivity "$VAL"
    step "$DS/$S: test sweep ($TEST per selectivity)" "results/raw/${DS}_${S}_test.parquet" \
      "logs/${DS}_${S}_test_sweep.log" "$PY" scripts/sweep_plans.py "$DS" "$S" --split test --per-selectivity "$TEST"
  done
  step "$DS: live planner evaluation" "results/raw/${DS}_planner_test.parquet" "logs/${DS}_planner.log" \
    "$PY" scripts/evaluate_planner.py "$DS" --per-selectivity "$TEST"
done

# Analysis always reruns (cheap) over every dataset that has finished results,
# so running one new dataset does not drop the others from the figures.
FORCE=1
ALL=""
for DS in sift gist glove deep yfcc $DATASETS; do
  case " $ALL " in *" $DS "*) continue ;; esac
  if [ -e "results/raw/${DS}_planner_test.parquet" ] || [ "$DRY" = 1 ]; then ALL="$ALL $DS"; fi
done
step "matched-recall comparison ($ALL )" "" logs/matched_recall.log "$PY" scripts/matched_recall.py --datasets $ALL
step "figures and tables" "" logs/figures.log "$PY" scripts/make_figures.py --datasets $ALL

echo "[$(date +%T)] done in $(( ($(date +%s) - START) / 60 )) min. Figures: results/figures/  Tables: results/tables/"
