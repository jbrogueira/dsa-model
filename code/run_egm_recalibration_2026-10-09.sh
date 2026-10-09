#!/bin/bash
# Recalibration and reruns with the minimum income benefit and the endogenous
# grid method at n_a = 100 (docs/EGM_PLAN.md section 6):
#   1. scale loop: SMM <-> joint (A_tfp, tau_y) pin until the fixed point
#   2. A[0] predetermination checks (grid search and EGM)
#   3. baseline for the report (fill_report.py --run-baseline), its figures
#   4. scenario 1: the 2023 output-tax rate throughout
#   5. baseline with the unemployment rate held at its 2023 level
#   6. policy exercises (I_g and health coverage, 2026; evaluator; figures)
#
# Usage (from code/, inside the venv, on the GPU instance):
#   nohup bash run_egm_recalibration_2026-10-09.sh > output/egm_recalibration.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CFG=calibration_input_GR.json
REPORT_OUT=${REPORT_OUT:-output/calibration_growth}
POLICY_OUT=${POLICY_OUT:-output/policy_egm_2026-10-09}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p output/calibration "$REPORT_OUT"
stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
TAG=egm_2026-10-09

if [ "${SKIP_CALIB:-0}" = "1" ]; then echo "[$(stamp)] step 1 skipped (SKIP_CALIB=1)"; else
echo "[$(stamp)] === 1. scale loop ==="
SMM_EXTRA="--tol 1e-5 --method least_squares" NORM_EXTRA="--tol 5e-4 --tol-pb 2e-4 --max-iter 20" \
  bash run_scale_loop.sh "$CFG" 2>&1 | tee output/scale_loop_$TAG.log
grep -q "SCALE LOOP DONE" output/scale_loop_$TAG.log \
  || { echo "[$(stamp)] RECALIBRATION FAILED: scale loop"; exit 1; }
grep -q "OUTER LOOP CONVERGED" output/scale_loop_$TAG.log \
  || echo "[$(stamp)] WARNING: scale loop did not converge within MAXROUND; continuing with the written values"
fi

echo "[$(stamp)] === 2. A[0] predetermination checks ==="
for S in grid egm; do
  python3 -u check_a0_predetermination.py --savings-solver $S 2>&1 | tee output/a0_check_${S}_$TAG.log
  grep -q "^DONE" output/a0_check_${S}_$TAG.log || echo "[$(stamp)] WARNING: A0 check ($S) did not finish"
  grep -q "FAIL" output/a0_check_${S}_$TAG.log && echo "[$(stamp)] WARNING: A0 predetermination FAIL ($S)"
done

echo "[$(stamp)] === 3. baseline for the report ==="
( cd reports && python3 -u fill_report.py --config "../$CFG" --outdir "../$REPORT_OUT" \
    --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee output/fill_report_$TAG.log
( cd reports && python3 -u baseline_figures.py --config "../$CFG" --outdir "../$REPORT_OUT" 2>&1 ) \
  | tee -a output/fill_report_$TAG.log

echo "[$(stamp)] === 4. scenario 1: the 2023 rate throughout ==="
python3 - "$CFG" calibration_input_GR_tau_pinned.json <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
fis = raw['fiscal']
for k in ('tau_y_debt_year', 'tau_y_first_year', 'tau_y_terminal_rule'):
    fis.pop(k, None)
fis['tau_y_mode'] = 'pinned_throughout'
json.dump(raw, open(sys.argv[2], 'w'), indent=2, ensure_ascii=False)
print('wrote', sys.argv[2], '(tau_y pinned throughout; theta, A_tfp, tau_y from', sys.argv[1] + ')')
PY
mkdir -p output/calibration_growth_tau_pinned
( cd reports && python3 -u fill_report.py --config ../calibration_input_GR_tau_pinned.json \
    --outdir ../output/calibration_growth_tau_pinned --run-baseline --n-sim 2000 --backend jax 2>&1 ) \
  | tee output/fill_report_tau_pinned_$TAG.log

echo "[$(stamp)] === 5. baseline with the unemployment rate held at its 2023 level ==="
python3 - "$CFG" calibration_input_GR_constant_u.json <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
raw['transition'].pop('unemployment_index_file', None)
json.dump(raw, open(sys.argv[2], 'w'), indent=2, ensure_ascii=False)
print('wrote', sys.argv[2], '(no unemployment index: the 2023 rates in every year)')
PY
mkdir -p output/calibration_growth_constant_u
( cd reports && python3 -u fill_report.py --config ../calibration_input_GR_constant_u.json \
    --outdir ../output/calibration_growth_constant_u --run-baseline --n-sim 2000 --backend jax 2>&1 ) \
  | tee output/fill_report_constant_u_$TAG.log
# The report's baseline figures carry scenario 1 as a dotted line.
( cd reports && python3 -u baseline_figures.py --config "../$CFG" --outdir "../$REPORT_OUT" 2>&1 ) \
  | tee -a output/fill_report_$TAG.log

echo "[$(stamp)] === 6. policy exercises ==="
OUT="$POLICY_OUT" bash run_policy_exercises.sh 2>&1 | tee output/policy_$TAG.log
grep -q "POLICY EXERCISES DONE" output/policy_$TAG.log \
  || { echo "[$(stamp)] RECALIBRATION CHAIN FAILED: policy exercises"; exit 1; }

echo "[$(stamp)] EGM RECALIBRATION CHAIN DONE"
