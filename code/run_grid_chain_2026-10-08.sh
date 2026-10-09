#!/bin/bash
# Full chain at a given asset-grid size: scale loop (SMM <-> A_tfp and tau_y
# pin), A[0] predetermination checks (shock in t = 0 and in t = 3), the
# baseline for the report, the constant-unemployment baseline, scenario 1
# (the 2023 output-tax rate throughout), the baseline figures (after scenario
# 1, which they draw dotted), then the policy exercises (I_g and health, shock
# in 2026; run_policy_exercises.sh).
#
# The grid size is the one in the configuration (lifecycle n_a). To run another
# grid without touching the adopted outputs, run this script in a copy of the
# repository whose configuration carries that n_a.
#
# Usage (from code/, inside the venv):
#   TAG=g100 nohup bash run_grid_chain_2026-10-08.sh > output/chain_g100.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CFG=calibration_input_GR.json
TAG=${TAG:-run}
POLICY_OUT=${POLICY_OUT:-output/policy_$TAG}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p output/calibration output/calibration_growth output/calibration_growth_constant_u \
  output/calibration_growth_tau_pinned "$POLICY_OUT"
stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
echo "[$(stamp)] chain $TAG, n_a = $(python3 -c "import json; print(json.load(open('$CFG'))['model']['n_a'])")"

if [ "${SKIP_CALIB:-0}" = "1" ]; then echo "[$(stamp)] steps 1-2 skipped (SKIP_CALIB=1)"; else
echo "[$(stamp)] === 1. scale loop ==="
SMM_EXTRA="--tol 1e-5 --method least_squares" NORM_EXTRA="--tol 5e-4 --tol-pb 2e-4 --max-iter 20" \
  bash run_scale_loop.sh "$CFG" 2>&1 | tee "output/scale_loop_$TAG.log"
grep -q "SCALE LOOP DONE" "output/scale_loop_$TAG.log" \
  || { echo "[$(stamp)] CHAIN FAILED: scale loop"; exit 1; }
grep -q "OUTER LOOP CONVERGED" "output/scale_loop_$TAG.log" \
  || echo "[$(stamp)] WARNING: scale loop did not converge within MAXROUND; continuing with the written values"

echo "[$(stamp)] === 2. A[0] predetermination checks ==="
for sp in 0 3; do
  python3 -u check_a0_predetermination.py --shock-period $sp 2>&1 | tee "output/a0_check_${TAG}_sp$sp.log"
  grep -q "FAIL" "output/a0_check_${TAG}_sp$sp.log" && echo "[$(stamp)] WARNING: A0 predetermination FAIL (shock period $sp)"
done
fi

echo "[$(stamp)] === 3. baseline for the report ==="
( cd reports && python3 -u fill_report.py --config "../$CFG" --outdir ../output/calibration_growth \
    --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee "output/fill_report_$TAG.log"

echo "[$(stamp)] === 4. baseline with the unemployment rate held at its 2023 level ==="
python3 - "$CFG" calibration_input_GR_constant_u.json <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
raw['transition'].pop('unemployment_index_file', None)
json.dump(raw, open(sys.argv[2], 'w'), indent=2, ensure_ascii=False)
print('wrote', sys.argv[2])
PY
( cd reports && python3 -u fill_report.py --config ../calibration_input_GR_constant_u.json \
    --outdir ../output/calibration_growth_constant_u --run-baseline --n-sim 2000 --backend jax 2>&1 ) \
  | tee "output/fill_report_constant_u_$TAG.log"

echo "[$(stamp)] === 5. scenario 1: the 2023 rate throughout ==="
python3 - "$CFG" calibration_input_GR_tau_pinned.json <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
fis = raw['fiscal']
for k in ('tau_y_debt_year', 'tau_y_first_year', 'tau_y_terminal_rule'):
    fis.pop(k, None)
fis['tau_y_mode'] = 'pinned_throughout'
json.dump(raw, open(sys.argv[2], 'w'), indent=2, ensure_ascii=False)
print('wrote', sys.argv[2])
PY
( cd reports && python3 -u fill_report.py --config ../calibration_input_GR_tau_pinned.json \
    --outdir ../output/calibration_growth_tau_pinned --run-baseline --n-sim 2000 --backend jax 2>&1 ) \
  | tee "output/fill_report_tau_pinned_$TAG.log"

echo "[$(stamp)] === 6. baseline figures ==="
( cd reports && python3 -u baseline_figures.py --config "../$CFG" --outdir ../output/calibration_growth 2>&1 ) \
  | tee -a "output/fill_report_$TAG.log"

echo "[$(stamp)] === 7. policy exercises ==="
OUT="$POLICY_OUT" bash run_policy_exercises.sh 2>&1 | tee "output/policy_$TAG.log"
grep -q "POLICY EXERCISES DONE" "output/policy_$TAG.log" || { echo "[$(stamp)] CHAIN FAILED: policy exercises"; exit 1; }
echo "[$(stamp)] CHAIN DONE"
