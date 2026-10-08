#!/bin/bash
# Full chain with UI eligibility at the start of a spell (p^ui = 0.332,
# docs/UI_ELIGIBILITY_PLAN.md) on top of the audit decisions of 2026-10-08
# (A/Y 2.96, capital income taxes 3.681% of output with tau_k in the SMM,
# delta_g 0.04316): scale loop, A[0] check, baseline (single-step rule),
# experiments, constant-unemployment baseline, then scenario 1 (the 2023
# output-tax rate throughout) at the same calibrated parameters.
#
# Usage (from code/, inside the venv):
#   nohup bash run_ui_eligibility_2026-10-08.sh > output/ui_eligibility_2026-10-08.log 2>&1 &
cd "$(dirname "$0")"
export MPLBACKEND=Agg
OUT=output/fiscal_2026-10-08f bash run_overnight_2026-10-08.sh
echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] === 6. scenario 1: the 2023 rate throughout ==="
# Scenario 1 is the calibrated configuration with the output tax pinned
# throughout: built from the main configuration after the scale loop, so it
# carries the same theta, A_tfp and tau_y.
python3 - calibration_input_GR.json calibration_input_GR_tau_pinned.json <<'PY'
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
( cd reports && python3 -u fill_report.py --config ../calibration_input_GR_tau_pinned.json --outdir ../output/calibration_growth_tau_pinned --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee output/fill_report_tau_pinned_2026-10-08.log
echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] SCENARIOS DONE"
