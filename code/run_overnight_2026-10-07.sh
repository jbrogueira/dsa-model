#!/bin/bash
# Overnight run of 2026-10-07/08 on the A100: recalibration under the budget
# restructure (output tax, lump sum, education, EU transfer, unemployment
# path, rate path), then the baseline for the report and the policy
# experiments, each followed by its checks.
#
#   1. scale loop: SMM <-> joint (A_tfp, tau_y) pin until the fixed point
#   2. A[0] predetermination check
#   3. baseline for the report (fill_report.py --run-baseline), its figures
#   4. G + I_g experiment set (run_fiscal_figures.py), evaluator, report figures
#
# Usage (from code/, inside the venv):
#   nohup bash run_overnight_2026-10-07.sh > output/overnight_2026-10-07.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CFG=${CFG:-calibration_input_GR.json}
OUT=${OUT:-output/fiscal_2026-10-08}
REPORT_OUT=${REPORT_OUT:-output/calibration_growth}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p output/calibration "$OUT" "$REPORT_OUT"

stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

echo "[$(stamp)] === 1. scale loop ==="
SMM_EXTRA="--tol 1e-5 --method least_squares" NORM_EXTRA="--tol 2e-4 --tol-pb 2e-4 --max-iter 20" \
  bash run_scale_loop.sh "$CFG" 2>&1 | tee output/scale_loop_2026-10-07.log
grep -q "SCALE LOOP DONE" output/scale_loop_2026-10-07.log \
  || { echo "[$(stamp)] OVERNIGHT FAILED: scale loop"; exit 1; }
grep -q "OUTER LOOP CONVERGED" output/scale_loop_2026-10-07.log \
  || echo "[$(stamp)] WARNING: scale loop did not converge within MAXROUND; continuing with the written values"

echo "[$(stamp)] === 2. A[0] predetermination check ==="
python3 -u check_a0_predetermination.py 2>&1 | tee output/a0_check_2026-10-07.log
grep -q "^DONE" output/a0_check_2026-10-07.log || echo "[$(stamp)] WARNING: A0 check did not finish"
grep -q "FAIL" output/a0_check_2026-10-07.log && echo "[$(stamp)] WARNING: A0 predetermination FAIL (see log)"

echo "[$(stamp)] === 3. baseline for the report ==="
( cd reports && python3 -u fill_report.py --config "../$CFG" --outdir "../$REPORT_OUT" \
    --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee output/fill_report_2026-10-07.log
( cd reports && python3 -u baseline_figures.py --config "../$CFG" --outdir "../$REPORT_OUT" 2>&1 ) \
  | tee -a output/fill_report_2026-10-07.log

echo "[$(stamp)] === 4. policy experiments: G + I_g ==="
python3 -u run_fiscal_figures.py --config "$CFG" --shock both --backend jax --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
grep -q "Total run time" "$OUT/run.log" || { echo "[$(stamp)] OVERNIGHT FAILED: fiscal run"; exit 1; }
python3 -u eval_fiscal_results.py --input "$OUT/fiscal_results.json" --config "$CFG" \
  > "$OUT/eval.log" 2>&1; echo "evaluator exit $?" >> "$OUT/eval.log"
tail -4 "$OUT/eval.log"
( cd reports && python3 -u fiscal_figures.py --input "../$OUT/fiscal_results.json" \
    --outdir "../$OUT" 2>&1 ) | tee -a "$OUT/run.log" || echo "[$(stamp)] WARNING: report fiscal figures failed"

echo "[$(stamp)] OVERNIGHT DONE"
