#!/bin/bash
# Recalibrate at the base-year config, then run the baseline transition and
# the checks Step 0 is judged by. Each stage depends on the previous one
# having written the config, so the script aborts rather than continues on a
# stale configuration.
#
#   1. run_scale_loop.sh      SMM <-> A_tfp fixed point, then the closure
#   2. diag_ss_vs_transition  does the base-year equilibrium coincide with
#                             the transition's t=0?  This is the gap Step 0
#                             exists to close; it was 13.7% of output.
#   3. check_a0               A[0] predetermination, regression guard
#   4. fill_report            all tables, Figure 1 and the detrended-trend
#                             column, which need the baseline transition
#
# Usage (from code/):
#   nohup bash run_step0_baseline.sh > output/step0_baseline.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
CFG=${1:-calibration_input_GR.json}
NSIM=${NSIM:-2000}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

echo "=== 1/4 recalibration (scale loop) ==="
SMM_EXTRA=${SMM_EXTRA:---tol 1e-5} bash run_scale_loop.sh "$CFG" \
  2>&1 | tee /tmp/step0_scale.log
grep -q "SCALE LOOP DONE" /tmp/step0_scale.log \
  || { echo "ABORTED: scale loop did not finish"; exit 1; }
grep -q "OUTER LOOP CONVERGED" /tmp/step0_scale.log \
  || { echo "ABORTED: scale loop hit MAXROUND without convergence"; exit 1; }
python3 -c "
import json; c=json.load(open('$CFG'))
print('calibrated:', {k: round(v, 6) for k, v in c['_derived']['theta'].items()})
print('A_tfp', c['production']['A_tfp'], '| closure', c['fiscal']['other_net_spending_over_Y'])
print('K_g', c['production']['K_g'], '| delta_g', c['production']['delta_g'])
"

echo
echo "=== 2/4 base-year equilibrium vs transition t=0 ==="
python3 -u diag_ss_vs_transition.py jax "$NSIM" 2>&1 | tee /tmp/step0_diag.log \
  || { echo "ABORTED: diag_ss_vs_transition failed"; exit 1; }

echo
echo "=== 3/4 A[0] predetermination ==="
python3 -u check_a0_predetermination.py 2>&1 | tee /tmp/step0_a0.log
grep -q "FAIL" /tmp/step0_a0.log \
  && { echo "ABORTED: A[0] predetermination broken"; exit 1; }

echo
echo "=== 4/4 baseline transition and report ==="
python3 -u reports/fill_report.py --config "$CFG" --backend jax \
  --run-baseline --implied --n-sim "$NSIM" \
  --outdir output/calibration_growth 2>&1 | tee /tmp/step0_report.log \
  || { echo "ABORTED: fill_report failed"; exit 1; }

echo
echo "STEP 0 BASELINE DONE"
