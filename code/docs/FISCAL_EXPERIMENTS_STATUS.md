# Fiscal Experiments — Status Handoff

Last updated: 2026-05-18. **STATUS: ROOT CAUSE FIXED.** See `## Resolution (2026-05-18)` at end.

Original handoff (2026-05-11) preserved below for context.

---

## Where we are (2026-05-11)

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

---

## Resolution (2026-05-18)

**Root cause was a single bug:** `mu_y` in `calibration_input_*.json` is the unconditional mean of log y per education stratum (per LIS estimation `data/lis/02_estimate_ar1.py:236`), but `lifecycle_perfect_foresight.py:_income_process()` was passing it as the AR(1) intercept to `tauchen()`. With ρ=0.95 this scaled the unconditional log-y mean by 20×. For Greek "high" edu (`mu_y=0.259`), the discretized stationary mean of log y was 5.18 instead of 0.259, producing y-grid levels around 110-290 instead of 0.8-2.1. Cross-section weighted mean y was 55 (should be 1.1).

This single error explained the entire fiscal-ratio blow-up:
- inflated `y_last` ⇒ inflated pensions
- inflated `effective_y` ⇒ inflated tax revenue
- inflated `Y` ⇒ understated `health/Y` (numerator was independent of y, denominator wasn't)
- inflated household income ⇒ `C > Y` (resource constraint violated)

The aggregator (`_compute_ss_aggregates`, `compute_government_budget`) was **correct** — numerator and denominator were always in the same per-living-person units. The bug was upstream, in the income-process discretization.

### Commits

1. **`8efa408`** — `mu_y` fix at `lifecycle_perfect_foresight.py:467`. Pass `(1-rho_y)*mu_y` as the intercept so `mu_y` is the unconditional mean of log y. Includes:
   - `test_income_process.py` (new) — data-driven regression test that auto-discovers `calibration_input_*.json` and asserts `mean(log y_grid) == mu_y` to 1e-12.
   - Updated default `edu_params` in `LifecycleConfig` so existing tests' y-grids stay bit-identical.
   - `lifecycle_jax.py` — auto-set `JAX_PLATFORMS=cpu` on macOS ARM (no more env-var ritual).
   - `IMPLEMENTATION_PLAN.md` — Phase 10 (test-suite audit) added.

2. **`99313fb`** — Class 1: data-target triage. Updates JSON fiscal targets to Greek 2023 data values; replaces single `health_over_Y` with `health_{gov,oop,total}_over_Y`; introduces `r_B` (sovereign rate) distinct from `r` (private K return). Code changes in `calibrate.py`, `olg_transition.py`, `eval_fiscal_results.py`, `run_fiscal_figures.py`.

3. **`93cec78`** — Class 1 fixup: Greek total health spending is ~8% of GDP (OECD), not 5.4% (data-sheet partial). Updated targets to match and rescaled `m_good` accordingly.

4. **`9c0bf90`** — Class 2: pension generosity. `pension_replacement_default 0.50 → 0.25`, `pension_min_floor 0.40 → 0.15`. Reason: model pension formula `ρ·w·y_last·α_mult` omits hours, so ρ represents replacement of "full-time potential earnings" not realised earnings. With Greek headline replacement 76% and realised aggregate ratio ~50%, ρ_model ≈ 0.25 calibrates the model concept to the data flow.

### Post-recalibration state

Re-calibrated Greek baseline (n_sim=10000, JAX/CPU, 33 min):

| Calibrated param | Value |
|---|---|
| ν | 36.57 |
| β | 1.019 (above 1; effective β·survival well below 1) |

Targeted moments:

| Moment | Target | Model | Dev |
|---|---|---|---|
| A_over_Y | 4.0 | 4.003 | exact |
| average_hours | 0.41 | 0.488 | +19% |

Fiscal ratios (untargeted):

| Ratio | Model | Data | Dev |
|---|---|---|---|
| `interest/Y` | 0.034 | 0.034 | exact |
| `K/Y` | 3.00 | (firm FOC) | exact |
| `health_total/Y` | 0.092 | 0.082 | +12% |
| `health_gov/Y` | 0.061 | 0.054 | +13% |
| `tax_revenue/Y` | 0.437 | 0.400 | +9% |
| `pensions/Y` | 0.221 | 0.160 | +38% |
| `ui/Y` | 0.011 | 0.006 | (small abs) |

Full-balance picture (including exogenous G/Ig/defense/transfers): primary balance ~ −6% of Y vs Greek 2023 +2%. The remaining gap is dominated by the pension overshoot.

Calibration report: `code/output/calibration/calibration_GR_20260518_184335.md`.

### Open items

- **Tighten pensions further.** Model 0.221 vs target 0.16. Likely needs another small reduction in `pension_replacement_default` (0.25 → ~0.18) followed by re-calibration.
- **Wealth-distribution residuals** (`wealth_gini = 0.38` vs 0.58, `zero_wealth_fraction = 2.9%` vs 1.1%) — Phase 9 (warm-glow bequest + initial wealth distribution).
- **Hours overshoots +19%.** Class 3 of `docs/CALIBRATION_FIX_CHECKLIST.md` proposes adding φ (Frisch curvature) as a third free parameter to close the trade-off with A/Y.
- **Fiscal experiments not yet re-validated.** The `run_fiscal_figures.py` smoke test should be re-run on the post-bug-fix baseline to confirm the debt path no longer explodes. Expected: debt path stable around target B/Y=1.64; bisection on τ_l for tax-financed shock should converge to plausible Δτ (~1-3 pp).
