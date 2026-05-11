# Fiscal Experiments — Status Handoff

Last updated: 2026-05-11.

## Where we are

The fiscal-experiment code path through `run_fiscal_figures.py` is **functionally intact** after the Phase 8 σ_α merge (`4a9e4aa`, 2026-05-08) and the follow-up plot/income-moment fixes (`599454b`, 2026-05-11). Smoke tests confirmed:

- Minimal hardcoded run (`python run_fiscal_figures.py --shock G`): three scenarios complete, four figures saved, ~28 min on NumPy.
- FE-on Greek run (`python run_fiscal_figures.py --config calibration_input_GR.json --shock G --backend jax` with `JAX_PLATFORM_NAME=cpu`): three scenarios complete, bisection converged, four figures saved, **54.5 min** on JAX/CPU (n_alpha=5 multiplies the cost of every cohort solve in the transition).

**However, the economic results are not sensible.** This is a *separate* problem from anything Phase 8 touched.

## Headline symptoms (FE-on Greek run)

| Quantity | Value | Interpretation |
|---|---|---|
| Baseline `final B/Y` | 23,378 % (234× GDP) | Debt explodes — the baseline is not a fiscal steady state |
| Debt-financed G shock `final B/Y` | 24,492 % | Larger explosion than baseline |
| Tax-financed G shock `Δτ_l` | +109 pp | Bisection converges only by pushing τ_l from 0.10 to ~1.19 |
| Tax-financed G shock `final B/Y` | −92 % | Government becomes a net creditor |
| Cumulative fiscal multiplier | 0.000 | Almost certainly a units issue or baseline-noise division |

## Diagnosis (where the issue is *not* and where it *probably* is)

**Not Phase 8.** The same residual was visible in the no-FE Greek calibration (`c9b3f14`). It's pre-Phase-8.

**Probably the fiscal accounting in `olg_transition.py` plus the input ratios in `calibration_input_GR.json`.** The Phase 8.7 calibration report (`code/output/calibration/calibration_GR_20260508_145856.md`) shows the smoking gun in the "Fiscal Ratios" table:

| Ratio | Model | Greek data | Gap |
|---|---|---|---|
| `tax_revenue / Y` | 0.97 | 0.40 | model collects 2.4× the data |
| `pensions / Y` | 2.37 | 0.16 | model spends 15× the data |
| `UI / Y` | 0.04 | 0.01 | model 4× the data |
| `health / Y` | 0.019 | 0.054 | model is 0.35× the data |
| `interest / Y` | 0.07 | 0.03 | model 2× the data |
| `primary_balance / Y` | −1.42 (deficit) | — | implies wild divergence |

So in the **steady state**, the model already runs a primary deficit of ~140% of GDP. Once we start the transition, debt compounds at `(1+r)` and grows without bound.

The most likely arithmetic culprits, in order of suspicion:

1. **Pension level.** Pension formula is `PENS = ρ · w · κ_{J_R} · [λ z_last + (1-λ) z̄]`. With `ρ = 0.50`, `w ≈ 1.15`, `κ_{J_R} ≈ 1.06`, `z̄ ≈ 1`, this gives `PENS ≈ 0.61` per retiree-period. Aggregated across the retired share (≈35% of lifecycle), pensions are ~0.21 per-capita per period. Model `Y` is ~11.9 per period for the same denominator (population-weighted income). So `pensions / Y ≈ 0.21 / 11.9 ≈ 0.018` *per agent*. The reported 2.37 model number is ~130× higher — implies the model is **summing pensions across some aggregation that doesn't share the same denominator as `Y`**. Almost certainly an unweighted-sum-over-cohorts vs population-share normalization mismatch in `compute_fiscal_ratios()` or `_compute_ss_aggregates()` in `calibrate.py`, or in `compute_government_budget_path()` in `olg_transition.py`.

2. **Tax revenue normalization.** Same direction of mismatch — model collects 0.97 of Y when data shows 0.40. Either aggregating tax revenue without the right population weights, or dividing by an income aggregate that excludes some component (e.g., excludes pensions when tax is on total income).

3. **Units of `B_initial`.** `calibration_input_GR.json` sets `B/Y = 1.70`. The script computes `B_initial = B_over_Y * Y_path.mean()`. If `Y_path.mean()` is "model Y" (which is ~12) but the rest of the fiscal accounting operates in different units, `B_initial` is initialized in the wrong scale and accumulates incorrectly.

## What to do next session

Order of attack:

1. **Read** the steady-state aggregator: `_compute_ss_aggregates()` in `calibrate.py` (around line 524), and the per-period budget accumulator: `compute_government_budget_path()` in `olg_transition.py`.
2. **Add a units assertion**: build the per-period accounting via hand calculation for one period in one education stratum with `n_alpha=1` and tiny `n_sim`, and compare to what the function reports. Discrepancy should pinpoint the mis-weighting.
3. **Check the simulation tuple → aggregate mapping** in `_panel_to_age_means` and downstream: `effective_y_sim` includes wage and UI together; `pension_sim` is separate. If the aggregator sums them with different age-weights or different education-share multipliers, the levels diverge.
4. **Validate against a closed-economy sanity check**: build a tiny calibration where the agent's lifetime budget constraint is solvable by hand (constant wage, no health, no UI, no pension floor), simulate, and verify the reported `tax_revenue/Y` and `pensions/Y` match the analytical answer.
5. Once the unit/aggregation is right, the baseline B/Y path should be near-stationary at `1.70` and the bisection on τ_l should land at a reasonable Δτ_l (~1–5 pp for a 2%-of-Y G shock).

## Reproducer

The FE-on run that produced the symptoms:
```
source ~/venvs/jax-arm/bin/activate
JAX_PLATFORM_NAME=cpu python run_fiscal_figures.py \
    --config calibration_input_GR.json --shock G --backend jax
```

Outputs land in `code/output/fiscal_test/`. Full stdout log: `/tmp/fiscal_GR_FEon.log` (latest).

## Pointers

- Phase 8 work (where it landed, what changed): `code/docs/IMPLEMENTATION_PLAN.md` § Phase 8.
- Phase 9 plan (warm-glow bequest + initial wealth — separate from this fiscal issue): same file § Phase 9.
- Calibration report with the fiscal-ratio table: `code/output/calibration/calibration_GR_20260508_145856.md`.
- Hand calculation of expected pension/Y in the diagnosis section above.

## What is NOT broken (don't waste time re-checking)

- Phase 8 σ_α plumbing: smoke-tested, regression-tested. With `n_alpha=1` the new code path collapses to pre-Phase-8 bit-exact; with `n_alpha=5` simulated `Var(log y)` matches the LIS target.
- Plot generation: fixed in `599454b`.
- Income-moment double-counting: fixed in `599454b` (UI was being summed twice in `income_gini` and `p90_p10_income`).
- JAX backend on macOS: works with `JAX_PLATFORM_NAME=cpu`. The Metal float64 issue is documented; do not retry Metal.
