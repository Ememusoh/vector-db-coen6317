#!/usr/bin/env bash
# Sweeps for planner training (val) and the oracle (test), then live planner evaluation.
# Usage: scripts/run_evaluation.sh <dataset> [val-per-selectivity] [test-per-selectivity] [--skip-val]
set -euo pipefail
cd "$(dirname "$0")/.."
DS=$1; VAL=${2:-200}; TEST=${3:-100}; SKIP_VAL=${4:-}
PY=${PYTHON:-.venv/bin/python}
mkdir -p logs
for S in independent correlated; do
  if [ "$SKIP_VAL" != "--skip-val" ]; then
    echo "[$(date +%T)] $DS: sweep $S val ($VAL per selectivity)"
    $PY scripts/sweep_plans.py "$DS" "$S" --split val --per-selectivity "$VAL" > "logs/${DS}_${S}_val_sweep.log" 2>&1
  fi
  echo "[$(date +%T)] $DS: sweep $S test ($TEST per selectivity)"
  $PY scripts/sweep_plans.py "$DS" "$S" --split test --per-selectivity "$TEST" > "logs/${DS}_${S}_test_sweep.log" 2>&1
done
echo "[$(date +%T)] $DS: evaluate planner"
$PY scripts/evaluate_planner.py "$DS" --per-selectivity "$TEST" > "logs/${DS}_planner.log" 2>&1
tail -25 "logs/${DS}_planner.log"
