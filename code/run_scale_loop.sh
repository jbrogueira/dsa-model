#!/bin/bash
# Outer scale loop: SMM re-fit <-> joint normalisation of A_tfp (Y_ss = 1) and
# pin of the output tax tau_y (base-year primary balance = target) until all
# hold. Rationale: theta, A_tfp and tau_y are coupled (moments depend on the
# output scale and on the wage, which tau_y moves; Y_ss and the balance depend
# on theta through hours and assets), so the steps iterate to a joint fixed
# point. See docs/PUBLIC_CAPITAL_KG_PLAN.md §4 and docs/BUDGET_ALIGNMENT_PLAN.md §6.3.
#
# Each round: (1) warm-start SMM initials from _derived.theta, (2) run SMM at
# the current (A_tfp, tau_y), (3) re-solve (A_tfp, tau_y). Converged when the
# SMM round leaves |Y_ss - 1| < TOL and |pb - target| < TOL_PB before the
# update (normalize iter-1 residuals) and the moments at the written pair are
# within MOM_TOL.
#
# Usage: bash run_scale_loop.sh [config.json]
set -uo pipefail
cd "$(dirname "$0")"
CFG=${1:-calibration_input_GR.json}
MAXROUND=${MAXROUND:-8}
TOL=${TOL:-5e-3}
TOL_PB=${TOL_PB:-1e-3}
# Largest relative deviation of any targeted moment at the written (theta, A_tfp)
# pair that still counts as converged.
MOM_TOL=${MOM_TOL:-5e-3}
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
  # SMM_EXTRA passes through extra calibrate.py flags, e.g. --tol/--maxiter.
  # At n_sim=10000 the Monte Carlo noise floor sits above the default tol=1e-6,
  # so Nelder-Mead exhausts maxiter and the loop aborts; --tol 1e-5 avoids that.
  # -u: stdout is block-buffered through the pipe, so without it a long SMM
  # round shows no progress until it ends -- a stall looks like a run.
  python3 -u calibrate.py --config "$CFG" --backend jax ${SMM_EXTRA:-} 2>&1 | tee /tmp/smm_round.log \
    || { echo "SCALE LOOP FAILED: SMM round $r"; exit 1; }
  # calibrate.py writes _derived.theta ONLY on convergence; without it the rest
  # of the loop would silently reuse the stale theta.
  grep -q "Calibrated theta written" /tmp/smm_round.log \
    || { echo "SCALE LOOP FAILED: SMM round $r did not converge (no theta write-back)"; exit 1; }

  echo "=== round $r: A_tfp normalization and tau_y pin ==="
  # NORM_EXTRA passes through normalize_A_tfp.py flags, e.g. --tol. Y_ss is
  # not continuous in A_tfp at the 1e-4 scale: a change of 1e-6 in A_tfp can
  # move discrete asset choices on the grid and shift Y_ss by ~2e-4 (seen on
  # 2026-10-05), in which case no root lies within the default tolerance.
  # PIN_TAU_Y=0 keeps fiscal.tau_y as configured (A_tfp alone for Y = 1).
  PIN_FLAG=$([ "${PIN_TAU_Y:-1}" = "1" ] && echo --pin-tau-y)
  python3 -u normalize_A_tfp.py --backend jax --write $PIN_FLAG --config "$CFG" ${NORM_EXTRA:-} \
    | tee /tmp/norm_round.log
  grep -q "^CONVERGED" /tmp/norm_round.log \
    || { echo "SCALE LOOP FAILED: normalize round $r"; exit 1; }

  resid1=$(grep "iter  1:" /tmp/norm_round.log \
           | sed -E 's/.*  resid=([+-][0-9.eE+-]+).*/\1/')
  pbres1=$(grep "iter  1:" /tmp/norm_round.log | grep -q pb_resid \
           && grep "iter  1:" /tmp/norm_round.log | sed -E 's/.*pb_resid=([+-][0-9.eE+-]+).*/\1/' \
           || echo 0)
  # Two tests, both on what the config now holds. resid1 is |Y - 1| at the
  # fitted theta BEFORE this round's A_tfp update, so passing it alone left the
  # written pair (theta_r, A_r) unevaluated: on 2026-10-01 it passed by
  # 0.4e-3 while A/Y at the written pair was 1.1% off. normalize prints the
  # targeted moments at the solved A_tfp; the round converges only if every
  # one of them is within MOM_TOL (relative) as well.
  momdev=$(python3 - /tmp/norm_round.log <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
blk = txt.split('SMM target moments at solved A_tfp', 1)[-1]
devs = [abs(float(m)) for m in re.findall(r'^\s*\S+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+][0-9.]+)\s*$', blk, re.M)]
print(max(devs) / 100.0 if devs else 1.0)
PY
)
  if python3 -c "import sys; sys.exit(0 if abs(float('$resid1')) < $TOL and abs(float('$pbres1')) < $TOL_PB and float('$momdev') < $MOM_TOL else 1)"; then
    echo "=== OUTER LOOP CONVERGED at round $r (|Y_ss-1| at fitted theta: $resid1; |pb-target|: $pbres1; max moment deviation at the written pair: $momdev) ==="
    CONV=1
    break
  fi
  echo "=== round $r done; scale still moving (iter-1 resid $resid1, pb resid $pbres1, max moment deviation at written pair $momdev) ==="
done
[ "$CONV" = "1" ] || echo "WARNING: hit MAXROUND=$MAXROUND without scale convergence"

echo "=== pin check at the written (theta, A_tfp, tau_y) ==="
python3 pin_baseline_closure.py --backend jax --config "$CFG" \
  || { echo "SCALE LOOP FAILED: pin check"; exit 1; }

echo "SCALE LOOP DONE"
