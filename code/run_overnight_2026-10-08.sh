#!/bin/bash
# Run of 2026-10-08 on the A100 after the demographic changes (ages 25-99,
# smoothed entering cohorts): recalibration, the baseline for the report, the
# policy experiments, and a baseline with the unemployment rate held at its
# 2023 level at the same parameters.
#
#   1. scale loop: SMM <-> joint (A_tfp, tau_y) pin until the fixed point
#   2. A[0] predetermination check
#   3. baseline for the report (fill_report.py --run-baseline), its figures
#   4. G + I_g experiment set (run_fiscal_figures.py), evaluator, report figures
#   5. constant-unemployment baseline (transition.unemployment_index_file removed)
#
# Usage (from code/, inside the venv):
#   nohup bash run_overnight_2026-10-08.sh > output/overnight_2026-10-08.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CFG=${CFG:-calibration_input_GR.json}
OUT=${OUT:-output/fiscal_2026-10-08b}
REPORT_OUT=${REPORT_OUT:-output/calibration_growth}
CF_CFG=${CF_CFG:-calibration_input_GR_constant_u.json}
CF_OUT=${CF_OUT:-output/calibration_growth_constant_u}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p output/calibration "$OUT" "$REPORT_OUT" "$CF_OUT"

stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

echo "[$(stamp)] === 1. scale loop ==="
SMM_EXTRA="--tol 1e-5 --method least_squares" NORM_EXTRA="--tol 1e-3 --tol-pb 2e-4 --max-iter 20" \
  bash run_scale_loop.sh "$CFG" 2>&1 | tee output/scale_loop_2026-10-08.log
grep -q "SCALE LOOP DONE" output/scale_loop_2026-10-08.log \
  || { echo "[$(stamp)] OVERNIGHT FAILED: scale loop"; exit 1; }
grep -q "OUTER LOOP CONVERGED" output/scale_loop_2026-10-08.log \
  || echo "[$(stamp)] WARNING: scale loop did not converge within MAXROUND; continuing with the written values"

echo "[$(stamp)] === 2. A[0] predetermination check ==="
python3 -u check_a0_predetermination.py 2>&1 | tee output/a0_check_2026-10-08.log
grep -q "^DONE" output/a0_check_2026-10-08.log || echo "[$(stamp)] WARNING: A0 check did not finish"
grep -q "FAIL" output/a0_check_2026-10-08.log && echo "[$(stamp)] WARNING: A0 predetermination FAIL (see log)"

echo "[$(stamp)] === 3. baseline for the report ==="
( cd reports && python3 -u fill_report.py --config "../$CFG" --outdir "../$REPORT_OUT" \
    --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee output/fill_report_2026-10-08.log
( cd reports && python3 -u baseline_figures.py --config "../$CFG" --outdir "../$REPORT_OUT" 2>&1 ) \
  | tee -a output/fill_report_2026-10-08.log

echo "[$(stamp)] === 4. policy experiments: G + I_g ==="
python3 -u run_fiscal_figures.py --config "$CFG" --shock both --backend jax --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
grep -q "Total run time" "$OUT/run.log" || { echo "[$(stamp)] OVERNIGHT FAILED: fiscal run"; exit 1; }
python3 -u eval_fiscal_results.py --input "$OUT/fiscal_results.json" --config "$CFG" \
  > "$OUT/eval.log" 2>&1; echo "evaluator exit $?" >> "$OUT/eval.log"
tail -4 "$OUT/eval.log"
( cd reports && python3 -u fiscal_figures.py --results "../$OUT/fiscal_results.json" \
    --baseline "../$REPORT_OUT/baseline_paths.npz" 2>&1 ) | tee -a "$OUT/run.log" \
  || echo "[$(stamp)] WARNING: report fiscal figures failed"

echo "[$(stamp)] === 5. baseline with the unemployment rate held at its 2023 level ==="
python3 - "$CFG" "$CF_CFG" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
raw['transition'].pop('unemployment_index_file', None)
json.dump(raw, open(sys.argv[2], 'w'), indent=2, ensure_ascii=False)
print('wrote', sys.argv[2], '(no unemployment index: the 2023 rates in every year)')
PY
( cd reports && python3 -u fill_report.py --config "../$CF_CFG" --outdir "../$CF_OUT" \
    --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee output/fill_report_constant_u_2026-10-08.log

echo "[$(stamp)] OVERNIGHT DONE"
