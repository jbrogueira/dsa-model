# Greek-calibration follow-up after the `mu_y` fix

Tracking checklist for fiscal-ratio gaps surfaced by the 2026-05-18 recalibration (commit `8efa408`). Not a Phase plan — a working list. Each item: a concrete check or change, the source it depends on, the expected effect.

## Context

After the `mu_y` interpretation fix the model is in a sensible numerical regime (`C/Y = 0.68`, primary balance −7.7% of Y), but several fiscal-ratio targets in `calibration_input_GR.json` don't match the model output. Root cause is mixed: some targets are **mis-set** in the JSON (concept mismatch with the model's quantity); some targets are right but the model needs different parameters; some are downstream of Phase 9 (warm-glow bequest / initial wealth) and out of scope here.

Data source for verification: `data/DATA_GR.xlsx` sheet `DATA` (2023 values, Greek macro shares of GDP) and `data_inventory.md`. Fall back to Eurostat (`hlth_sha11_hf` for health, `gov_10a_main` for general government) only if the spreadsheet is silent.

## Class 1 — Data-definition triage (do first, cheap)

For each ratio: confirm the model's quantity is conceptually comparable to the data figure stored in the JSON. If not, either fix the JSON target or fix the model concept. **Don't re-calibrate against a mis-defined target.**

- [ ] **`pensions_over_Y`** — JSON 0.16, data (`Social benefits in cash Pensions / Y`, 2023) **0.120**. JSON is overstated by ~33%. Update JSON to 0.120 (or use a 2018–2023 average if you prefer cyclical smoothing — pre-COVID values are slightly higher).
- [ ] **`ui_over_Y`** — JSON 0.01, data (`Social benefits Unemployment / Y`, 2023) **0.006**. Update JSON to 0.006.
- [ ] **`health_over_Y`** — JSON **0.054 = 0.023 (public, `Social benefits Health`) + 0.031 (private, `Private Consumption in Health`)**. The model's `gov_health` aggregate covers *only* the government share (`kappa × m`). So the JSON target is comparing model-public-health to data-total-health. Two ways to fix:
  - **(A) Easy:** change JSON target to 0.023 (public only). Then `m_good` recalibrates downward to match the data, and the household OOP share (`(1-kappa) × m`) gets whatever level falls out — no longer pinned to the private-health data figure.
  - **(B) Right:** add a `health_total_over_Y` ratio that compares model `(kappa + (1-kappa)) × m = m` against data total 0.054, AND keep `health_gov_over_Y = kappa × m` against data 0.023. Two-equation constraint pins both `m_good` (level) and `kappa` (split). Currently `kappa = 0.662` from data_inventory.md ("Govt/compulsory share"); the implied private share would be 0.338.
  - **Recommendation:** (B). It's two lines of code change in `compute_fiscal_ratios()` and the right structural target.
- [ ] **`interest_over_Y`** — JSON 0.03, data **0.034** (2023). Both reasonable. But the model computes it as `r_path × B_over_Y × Y / Y = r × B/Y = 0.04 × 1.70 = 0.068`, which mechanically overshoots. The implied data interest rate is 0.034/1.64 ≈ 2.1% (real net effective rate on Greek sovereign debt, well below the model's 4%). Two ways to reconcile:
  - **(A)** Lower `r` in the JSON to ~0.02. Affects everything else through firm FOC (K/L jumps, w jumps). Heavy change.
  - **(B)** Keep `r = 0.04` (private return on capital) and parameterise the sovereign-debt rate `r_B` separately. Currently `compute_government_budget` uses `r_t = self.r_path[t_idx]` as the debt-service rate. Add a `r_B_path` (or scalar `r_B`) override that defaults to `r_path` for backward compatibility.
  - **Recommendation:** (B). Cleaner separation (private capital return ≠ sovereign yield), and matches the literature on Eurozone interest spreads.
- [ ] **`tax_revenue_over_Y`** — JSON 0.40, data **GG taxes only ≈ 0.42** (`tax_c + tax_l + tax_k + tax_p` = 0.171 + 0.059 + 0.027 + 0.130 = 0.387 for 2023), data total GG revenues 0.482. Model output `0.467` is between the two. The JSON target is closest to the tax-only definition (excludes "Other revenues" of 0.094 which is mostly transfers from EU and capital revenues, not modelled). Reasonable. Hold this target.

## Class 2 — Pension generosity (after Class 1)

Even after Class 1 fixes (`pensions_over_Y` target = 0.12 instead of 0.16), the model produces 0.384. That's 3× the data. Possible causes, in suspicion order:

- [ ] **Replacement rate is wrong.** JSON has `pension_replacement_default = 0.50`. `data_inventory.md` records the data figure as **76.275%**. The JSON value was deliberately set to half the data figure — almost certainly because someone realised early that the full 76% rate would blow up pension spending. With ρ=0.50, pension spending is ~38% of Y. With ρ=0.76, it would be ~58%. Greek data shows 12%. Something else is wrong.
- [ ] **Pension base is wrong.** The model formula: `pension = ρ × w × κ_{J_R} × y_last × α_mult`. The `y_last` is the income state at the moment of retirement, picked from a Tauchen grid. In real Greek pensions, the base is **career average** (with replacement rate applied to it), not last-year earnings. The model has a `pension_avg_weight` parameter (currently 1.0 = pure last-state) that can blend last-state with career average (`mean_y_employed`). Setting `pension_avg_weight` to a value below 1 reduces the base for the (small) tail of high-y_last agents. Worth experimenting.
- [ ] **Pension cap missing.** Greek pension system has an absolute cap (~€2,500/month for state pensions, more for supplementary). High-y_last agents in the model receive uncapped pensions. A `pension_max` parameter would limit the top tail.
- [ ] **Denominator mismatch.** Per-capita Y in the model isn't quite the same as aggregate GDP per capita in the data, because the model normalises population to 1 across all ages, while GDP per capita is over the working-age population in the standard definition. Compute the data ratio using the population-share of working-age + the right normalisation — might explain a 1.2-1.5× factor.

**Order of attack:** start with `pension_avg_weight = 0.5` (50% last-state + 50% career average) and see if pensions/Y falls. Then introduce a cap if still high.

## Class 3 — Hours / saving trade-off (after Class 1+2)

Calibration produced `nu = 20.05, beta = 0.994 (upper bound), average_hours = 0.574 vs target 0.41` — the optimizer can't reconcile `h̄ = 0.41` with `A/Y = 4` inside the (nu, beta) box.

- [ ] **Add `phi` (Frisch curvature) to free parameters.** Currently `phi = 2.0` is fixed. With three free params (β, ν, φ) and two targets, the system is under-determined but the SMM optimizer can pick a (β, ν, φ) combination that hits both targets. Add `{"name": "phi", "path": "phi", "lower": 1.5, "upper": 4.0, "initial": 2.0}` to the `calibration.params` block.
- [ ] **Relax `beta_upper`.** Currently 0.999, hit 0.994. Less relevant once `phi` is free, but worth raising to 0.9995 if the constraint still binds.
- [ ] **Sanity-check the target.** `h̄ = 0.41` was set from "Average weekly hours worked = 40.95 hours/week" in DATA_GR.xlsx, normalised by some weekly time endowment. If the endowment was 100 hours/week (waking time minus essentials), 41/100 = 0.41. If 168 hours/week, 41/168 = 0.24. The model's `l_sim` is dimensionless on [0, 1]. Verify which normalisation was intended.

## Class 4 — Phase 9 territory (out of scope)

- `wealth_gini = 0.366` vs target `0.580` → Phase 9 (warm-glow bequest).
- `zero_wealth_fraction = 4.7%` vs target `1.1%` → Phase 9 (initial wealth distribution).
- `income_gini = 0.300` vs target `0.318` — closer than before, may not need any action.
- `p90_p10_income = 2.95` vs target `3.9` — undershoots; depends on income process variance. Recheck after Phase 9.

## Acceptance criteria

After Class 1 + 2 + 3 fixes, the post-calibration report should show:
- `pensions/Y` within ±20% of target
- `health_total/Y` and `health_gov/Y` both within ±20% (Class 1B)
- `tax_revenue/Y` within ±10% (already close)
- `primary_balance/Y` within ±2% of zero (already at −7.7%)
- `average_hours` within ±10% of target
- `A_over_Y` within ±5% (currently exact)

Phase 9 targets (`wealth_gini`, `zero_wealth_fraction`) addressed separately.

## Computational cost

Each re-calibration with `--backend jax` on macOS ARM takes ~17 minutes. Plan for 4–6 iterations total: triage pass, post-Class-1 calibration, post-Class-2 calibration, final fine-tune. Wall time: ~2 hours of calibration runtime spread over the day.
