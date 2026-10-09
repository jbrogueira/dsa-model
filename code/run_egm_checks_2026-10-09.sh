#!/bin/bash
# Checks before the recalibration with the endogenous grid method
# (docs/EGM_PLAN.md section 5), at the current theta of calibration_input_GR.json:
# for EGM at n_a = 100 and 200 and grid search at n_a = 100, the targeted
# moments and the wealth above a = 20 in the base-year cross-section (with the
# cross-section's run time), then the baseline and the debt-financed I_g
# experiment (2026); then the table and the rule for the grid size.
#
# Usage (from code/, inside the venv, on the GPU instance):
#   OUT=output/egm_checks_2026-10-XX nohup bash run_egm_checks_2026-10-09.sh > output/egm_checks.log 2>&1 &
cd "$(dirname "$0")"
export MPLBACKEND=Agg
OUT=${OUT:-output/egm_checks_$(date +%Y-%m-%d)}
mkdir -p "$OUT"
stamp() { echo "[$(date -u "+%Y-%m-%d %H:%M:%S UTC")] $*"; }

for RUN in "egm 100" "egm 200" "grid 100"; do
    set -- $RUN
    SOLVER=$1; NA=$2
    D="$OUT/${SOLVER}_n${NA}"
    mkdir -p "$D"
    stamp "=== $SOLVER, n_a = $NA -> $D ==="
    python3 egm_grid_check.py config --n-a "$NA" --savings-solver "$SOLVER" --out "$D/config.json" \
        || { stamp "CONFIG FAILED"; exit 1; }
    python3 -u egm_grid_check.py moments --config "$D/config.json" --out "$D/moments.json" \
        2>&1 | tee "$D/moments.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { stamp "MOMENTS FAILED ($SOLVER $NA)"; exit 1; }
    START=$(date +%s)
    python3 -u run_fiscal_figures.py --config "$D/config.json" --backend jax \
        --shock Ig --scenarios debt --shock-year 2026 --no-distribution \
        --output-dir "$D" 2>&1 | tee "$D/run.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { stamp "EXPERIMENT FAILED ($SOLVER $NA)"; exit 1; }
    stamp "$SOLVER n_a=$NA: baseline + I_g run took $(( $(date +%s) - START )) s"
done

stamp "=== comparison ==="
python3 egm_grid_check.py compare --runs "$OUT/egm_n100" "$OUT/egm_n200" "$OUT/grid_n100" \
    --out "$OUT/egm_checks.md"
stamp "EGM CHECKS DONE"
