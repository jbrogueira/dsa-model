#!/bin/bash
# Full chain after the audit: scale loop (SMM with tau_k and the wealth target 2.96, delta_g 0.04316),
# A[0] check, baseline (single-step rule), experiments, constant-unemployment baseline, then scenario 1.
cd "$(dirname "$0")"
export MPLBACKEND=Agg
OUT=output/fiscal_2026-10-08e bash run_overnight_2026-10-08.sh
echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] === 6. scenario 1: the 2023 rate throughout ==="
mkdir -p output/calibration_growth_tau_pinned
( cd reports && python3 -u fill_report.py --config ../calibration_input_GR_tau_pinned.json --outdir ../output/calibration_growth_tau_pinned --run-baseline --n-sim 2000 --backend jax 2>&1 ) | tee output/fill_report_tau_pinned_2026-10-08.log
echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] SCENARIOS DONE"
