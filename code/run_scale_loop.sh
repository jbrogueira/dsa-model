#!/bin/bash
# Outer scale loop: SMM re-fit <-> A_tfp normalization (Y_ss = 1) until both
# hold, then re-pin the fiscal closure. Rationale: theta and A_tfp are coupled
# (moments depend on the output scale through level parameters; Y_ss depends on
# theta through hours/assets), so the two calibration steps iterate to a joint
# fixed point. See docs/PUBLIC_CAPITAL_KG_PLAN.md §4.
#
# Each round: (1) warm-start SMM initials from _derived.theta, (2) run SMM at
# the current A_tfp, (3) re-normalize A_tfp. Converged when the SMM round
# leaves |Y_ss - 1| < TOL before the A_tfp update (normalize iter-1 residual).
#
# Usage: bash run_scale_loop.sh [config.json]
set -uo pipefail
cd "$(dirname "$0")"
CFG=${1:-calibration_input_GR.json}
MAXROUND=${MAXROUND:-8}
TOL=${TOL:-5e-3}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p output/calibration

CONV=0
for r in $(seq 1 "$MAXROUND"); do
  echo "=== round $r: warm-start SMM initials from _derived.theta ==="
  python3 - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
th = c.get('_derived', {}).get('theta', {})
for prm in c['calibration']['params']:
    if prm['name'] in th:
        prm['initial'] = th[prm['name']]
json.dump(c, open(p, 'w'), indent=2)
print("initials:", {prm['name']: prm['initial'] for prm in c['calibration']['params']})
PY
  echo "=== round $r: SMM (calibrate.py) ==="
  python3 calibrate.py --config "$CFG" --backend jax 2>&1 | tee /tmp/smm_round.log \
    || { echo "SCALE LOOP FAILED: SMM round $r"; exit 1; }
  # calibrate.py writes _derived.theta ONLY on convergence; without it the rest
  # of the loop would silently reuse the stale theta.
  grep -q "Calibrated theta written" /tmp/smm_round.log \
    || { echo "SCALE LOOP FAILED: SMM round $r did not converge (no theta write-back)"; exit 1; }

  echo "=== round $r: A_tfp normalization ==="
  python3 normalize_A_tfp.py --backend jax --write --config "$CFG" \
    | tee /tmp/norm_round.log
  grep -q "^CONVERGED" /tmp/norm_round.log \
    || { echo "SCALE LOOP FAILED: normalize round $r"; exit 1; }

  resid1=$(grep "iter  1:" /tmp/norm_round.log \
           | sed -E 's/.*resid=([+-][0-9.eE+-]+).*/\1/')
  if python3 -c "import sys; sys.exit(0 if abs(float('$resid1')) < $TOL else 1)"; then
    echo "=== OUTER LOOP CONVERGED at round $r (|Y_ss-1| at fitted theta: $resid1) ==="
    CONV=1
    break
  fi
  echo "=== round $r done; scale still moving (iter-1 resid $resid1) ==="
done
[ "$CONV" = "1" ] || echo "WARNING: hit MAXROUND=$MAXROUND without scale convergence"

echo "=== re-pin fiscal closure ==="
python3 pin_baseline_closure.py --backend jax --config "$CFG" --write \
  || { echo "SCALE LOOP FAILED: pin_baseline_closure"; exit 1; }

echo "SCALE LOOP DONE"
