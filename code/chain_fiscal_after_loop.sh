#!/bin/bash
# Chain after the scale loop: wait for run_scale_loop.sh to finish; if it
# converged and re-pinned the closure, run the A[0] predetermination check
# (tau_l + Ig cases, both backends), then the full G + Ig fiscal set.
# Aborts if the loop failed / hit MAXROUND, or if any A[0] check FAILs.
#
# Usage: nohup bash chain_fiscal_after_loop.sh > output/chain_fiscal.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CFG=${1:-calibration_input_GR.json}
LOOP_LOG=${LOOP_LOG:-output/scale_loop_20260710.log}
OUT_DIR=${OUT_DIR:-output/fiscal_test_kg}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "waiting for scale loop ($LOOP_LOG) ..."
while ! grep -qE "SCALE LOOP DONE|SCALE LOOP FAILED" "$LOOP_LOG" 2>/dev/null; do
  sleep 30
done
grep -q "SCALE LOOP DONE" "$LOOP_LOG" \
  || { echo "CHAIN ABORTED: scale loop failed"; exit 1; }
grep -q "OUTER LOOP CONVERGED" "$LOOP_LOG" \
  || { echo "CHAIN ABORTED: scale loop hit MAXROUND without convergence"; exit 1; }
echo "scale loop converged; proceeding"

echo "=== A[0] predetermination check ==="
python3 check_a0_predetermination.py 2>&1 | tee /tmp/a0_check.log
grep -q "^DONE" /tmp/a0_check.log \
  || { echo "CHAIN ABORTED: A0 check crashed"; exit 1; }
grep -q "FAIL" /tmp/a0_check.log \
  && { echo "CHAIN ABORTED: A0 predetermination FAIL"; exit 1; }

echo "=== full fiscal set: G + Ig shocks ==="
mkdir -p "$OUT_DIR"
python3 run_fiscal_figures.py --config "$CFG" --shock both --backend jax \
  --output-dir "$OUT_DIR" \
  || { echo "CHAIN FAILED: fiscal run"; exit 1; }
echo "CHAIN DONE"
