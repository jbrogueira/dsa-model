#!/bin/bash
# The three output-tax rules with the transfer from abroad held at 1.0% of
# output (fiscal.foreign_transfer_over_Y):
#   1. scale loop on calibration_input_GR.json (tau_y pinned to the base-year
#      primary balance), shared by rules A and B
#   2. A: linear from the pin in 2023 to the rate of 2060 that puts the 2060
#      debt ratio at the projection's, constant after (tau_y_mode debt_ramp)
#      B: the pin in every year (tau_y_mode pinned_throughout)
#   3. C: one rate from 2023 for the 2060 debt ratio (tau_y_mode debt,
#      tau_y_first_year 2023); the base-year pin is dropped, so the
#      calibration (SMM and A_tfp, rate held) and the rate alternate until the
#      rate settles
#
# Usage (from code/, inside the venv):
#   nohup bash run_tau_rules_2026-10-09.sh > output/tau_rules_2026-10-09.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CFG=calibration_input_GR.json
stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
SMM="--tol 1e-5 --method least_squares"
NORM="--tol 5e-4 --tol-pb 2e-4 --max-iter 20"

variant() {   # variant OUT_CFG MODE [FIRST_YEAR]
  python3 - "$CFG" "$1" "$2" "${3:-}" <<'PY'
import json, sys
src, dst, mode, first = sys.argv[1:5]
raw = json.load(open(src))
fis = raw['fiscal']
for k in ('tau_y_debt_year', 'tau_y_first_year', 'tau_y_terminal_rule'):
    fis.pop(k, None)
fis['tau_y_mode'] = mode
if mode != 'pinned_throughout':
    fis['tau_y_debt_year'] = 2060
    fis['tau_y_terminal_rule'] = False
if first:
    fis['tau_y_first_year'] = int(first)
json.dump(raw, open(dst, 'w'), indent=2, ensure_ascii=False)
print('wrote', dst, 'tau_y_mode', mode, 'first year', first or '-')
PY
}

baseline() {  # baseline CFG OUTDIR LOG
  mkdir -p "$2"
  ( cd reports && python3 -u fill_report.py --config "../$1" --outdir "../$2" \
      --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee "$3"
  grep -q "wrote baseline_paths.npz" "$3" || { echo "[$(stamp)] TAU RULES FAILED: baseline $1"; exit 1; }
}

echo "[$(stamp)] === 1. scale loop with the constant transfer ==="
SMM_EXTRA="$SMM" NORM_EXTRA="$NORM" bash run_scale_loop.sh "$CFG" 2>&1 | tee output/scale_loop_tau_rules.log
grep -q "SCALE LOOP DONE" output/scale_loop_tau_rules.log || { echo "[$(stamp)] TAU RULES FAILED: scale loop"; exit 1; }

echo "[$(stamp)] === 2A. linear ramp to the 2060 rate ==="
variant calibration_input_GR_tau_ramp.json debt_ramp
baseline calibration_input_GR_tau_ramp.json output/calibration_growth_tau_ramp output/fill_report_tau_ramp.log

echo "[$(stamp)] === 2B. the pin throughout ==="
variant calibration_input_GR_tau_pinned.json pinned_throughout
baseline calibration_input_GR_tau_pinned.json output/calibration_growth_tau_pinned output/fill_report_tau_pinned.log
for d in tau_ramp tau_pinned; do
  ( cd reports && python3 -u baseline_figures.py --config "../calibration_input_GR_$d.json" \
      --outdir "../output/calibration_growth_$d" 2>&1 ) | tail -3
done

echo "[$(stamp)] === 3. one rate from 2023 ==="
C3=calibration_input_GR_tau_const.json
variant $C3 debt 2023
# Start from the mean of the ramp's rate over 2023-2060.
python3 - $C3 output/calibration_growth_tau_ramp/baseline_paths.npz <<'PY'
import json, sys, numpy as np
c = json.load(open(sys.argv[1])); tau = np.load(sys.argv[2])['tau_y_path'][:38]
c['fiscal']['tau_y'] = round(float(tau.mean()), 6)
json.dump(c, open(sys.argv[1], 'w'), indent=2, ensure_ascii=False)
print('initial tau_y', c['fiscal']['tau_y'])
PY
for k in 1 2 3 4 5; do
  echo "[$(stamp)] --- round $k: calibration at the rate held ---"
  PIN_TAU_Y=0 TOL_PB=1 SMM_EXTRA="$SMM" NORM_EXTRA="$NORM" bash run_scale_loop.sh $C3 2>&1 \
    | tee output/scale_loop_tau_const_$k.log
  grep -q "SCALE LOOP DONE" output/scale_loop_tau_const_$k.log || { echo "[$(stamp)] TAU RULES FAILED: rule C scale loop"; exit 1; }
  echo "[$(stamp)] --- round $k: baseline, rate for the 2060 debt ratio ---"
  baseline $C3 output/calibration_growth_tau_const output/fill_report_tau_const_$k.log
  python3 - $C3 output/calibration_growth_tau_const/baseline_paths.npz > /tmp/tau_c.txt <<'PY'
import json, sys, numpy as np
c = json.load(open(sys.argv[1])); old = float(c['fiscal']['tau_y'])
new = float(np.load(sys.argv[2])['tau_y_path'][0])
print(f'{old:.6f} {new:.6f} {abs(new - old):.6f}')
if abs(new - old) >= 5e-4:      # settled: keep the rate the calibration was made at
    c['fiscal']['tau_y'] = round(new, 6)
    json.dump(c, open(sys.argv[1], 'w'), indent=2, ensure_ascii=False)
PY
  read old new gap < /tmp/tau_c.txt
  echo "[$(stamp)] rule C round $k: rate held $old, rate solved $new, change $gap"
  python3 -c "import sys; sys.exit(0 if $gap < 5e-4 else 1)" && { echo "[$(stamp)] rule C settled"; break; }
done
( cd reports && python3 -u baseline_figures.py --config "../$C3" --outdir ../output/calibration_growth_tau_const 2>&1 ) | tail -3
echo "[$(stamp)] TAU RULES DONE"
