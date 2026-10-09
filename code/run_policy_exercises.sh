#!/bin/bash
# The public-investment and health-coverage exercises (docs/POLICY_EXERCISES_PLAN.md
# section 6.3): both shocks unanticipated in 2026, debt and labour-tax financing
# (and the labour tax over the years of the health cut), the two health
# decomposition runs, distributional and welfare outputs; then the evaluator
# and the report's figures and tables. Uses calibration_input_GR.json as it is.
#
# Usage (from code/, inside the venv):
#   OUT=output/policy_2026-10-XX nohup bash run_policy_exercises.sh > output/policy_run.log 2>&1 &
cd "$(dirname "$0")"
export MPLBACKEND=Agg
OUT=${OUT:-output/policy_$(date +%Y-%m-%d)}
mkdir -p "$OUT"
stamp() { echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] $*"; }

stamp "=== 1. experiments -> $OUT ==="
python3 -u run_fiscal_figures.py --config calibration_input_GR.json --backend jax \
    --shock Ig,health --scenarios debt,tau_l_debt,tau_l_window --shock-year 2026 \
    --output-dir "$OUT" 2>&1 | tee "$OUT/run.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { stamp "EXPERIMENTS FAILED"; exit 1; }

stamp "=== 2. evaluator ==="
python3 eval_fiscal_results.py --input "$OUT/fiscal_results.json" \
    --config calibration_input_GR.json 2>&1 | tee "$OUT/eval.log"

stamp "=== 3. report figures and tables ==="
# The report shows the health cut under debt and permanent labour-tax
# financing only; the labour tax over 2026-30 stays in the results file.
python3 reports/fiscal_figures.py --results "$OUT/fiscal_results.json" \
    --drop health:tax_financed_window 2>&1 | tee -a "$OUT/run.log"
stamp "POLICY EXERCISES DONE"
