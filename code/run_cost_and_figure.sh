#!/bin/bash
# Short instance session: settle the cohort-batching cost question and redraw
# the baseline figure. No recalibration -- the config already holds the fit at
# the 2023 public-capital numbers.
#
#   1. cost ratio   TestCohortBatchedSurvival, which only means anything on a
#                   GPU: warns above 8x for 16x the cohorts, fails above 40x.
#                   This decides whether a cohort-batched SMM is affordable.
#   2. figure 1     one baseline transition at T_transition, filling the
#                   transition panels and the detrended-trend column.
#
# Usage (from code/):
#   nohup bash run_cost_and_figure.sh > output/cost_and_figure.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
CFG=${1:-calibration_input_GR.json}
NSIM=${NSIM:-2000}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

echo "=== 1/2 cohort-batching cost ratio ==="
# pytest is not part of the runtime dependency set, so a venv built for the
# chain alone will not have it.
python3 -c "import pytest" 2>/dev/null || pip install -q pytest
python3 -u -m pytest test_olg_transition.py::TestCohortBatchedSurvival \
  -q -rw -W "always::RuntimeWarning" 2>&1 | tail -20

echo
echo "=== 2/2 baseline transition and figure ==="
python3 -u reports/fill_report.py --config "$CFG" --backend jax \
  --run-baseline --implied --n-sim "$NSIM" \
  --outdir output/calibration_growth 2>&1 | tail -12

echo
echo "COST AND FIGURE DONE"
