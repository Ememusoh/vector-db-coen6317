#!/usr/bin/env bash
# Full pipeline for one dataset: download + labels + ground truth, indexes, validation sweeps.
# Usage: scripts/run_pipeline.sh <dataset> [queries-per-selectivity]
set -euo pipefail
cd "$(dirname "$0")/.."
DS=$1
PER=${2:-200}
PY=${PYTHON:-.venv/bin/python}
mkdir -p logs
echo "[$(date +%T)] $DS: prepare"
$PY scripts/prepare_dataset.py "$DS" 2>&1 | grep -v '^\s*[0-9 ]*[0-9]\s' | tee "logs/${DS}_prepare.log"
echo "[$(date +%T)] $DS: build indexes"
$PY scripts/build_indexes.py "$DS" | tee "logs/${DS}_build.log"
for S in independent correlated; do
  echo "[$(date +%T)] $DS: sweep $S (val, $PER per selectivity)"
  $PY scripts/sweep_plans.py "$DS" "$S" --split val --per-selectivity "$PER" > "logs/${DS}_${S}_val_sweep.log" 2>&1
  tail -1 "logs/${DS}_${S}_val_sweep.log"
done
echo "[$(date +%T)] $DS: done"
