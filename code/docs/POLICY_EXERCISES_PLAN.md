# Plan: public-investment and health-coverage policy exercises

Written 2026-10-08. To be implemented and run in a separate session. Branch `trend-growth`.
Calibration: `calibration_input_GR.json` (the config behind `output/fiscal_2026-10-08d`).

## 0. Scope

Two exercises. Both use the same two financing schemes and report the same outputs.

| Exercise | Shock | Financing schemes run now |
|---|---|---|
| I_g | Permanent rise in public investment from 2026, a constant level equal to 2% of baseline output in 2025, the year before the shock | (i) debt; (ii) labour tax τ_l with a terminal debt-to-GDP target |
| Health | Five-year cut in government health spending, back to baseline in year 6 | same two |

The G shock and the τ_l scenario with a terminal NFA target stay in the code and stay selectable. They are not run in this round.

Outputs for every scenario:
1. Aggregates. This is the existing set, extended in §3.1.
2. Household means by age group.
3. Inequality: Gini, P90/P10 and P90/P50 for consumption, labour income and disposable income, plus the wealth Gini and the top-10% wealth share.
4. Welfare: consumption-equivalent variation by birth cohort, and by income quintile for the cohorts alive at the shock date.

Settled with the user:
- Both shocks are unanticipated and hit in 2026 (t_s = 3; t = 0 is 2023). Through 2025 the economy follows the baseline. In 2026 households learn the full counterfactual path and re-optimise from their 2026 state (§2.5).
- The health cut is flat over the five years 2026–2030 and reverts in 2031.
- The health targets are changes in shares of GDP.
- The τ_l scheme targets the baseline terminal B/Y (currently −6.45; accepted).
- The welfare decomposition into the κ and m components is run.
- The code allows both coverage κ and the level of medical spending m to vary over calendar time.
- For Greece, κ and m are moved jointly so that two targets are matched: government health spending falls 2.06 pp of GDP and out-of-pocket spending rises 0.23 pp of GDP.

## 1. Facts the plan rests on (checked 2026-10-08)

### Medical spending in the model
- Medical spending is a pure expenditure shock. Each period a household of age j pays out-of-pocket spending (1−κ)·m(j) and the government pays κ·m(j).
- m(j) = `m_age_profile[j]·m_good`. Health does not enter utility, survival or productivity (n_h = 1).
- Consequence: a fall in m frees resources at no cost to the household. Welfare gains from the m component are mechanical, and the plan reports them separately from the κ component (§3.4).

### Current code
- κ is a scalar: `LifecycleConfig.kappa` (lifecycle_perfect_foresight.py:127, :395).
- m is an age profile only: `m_grid` has shape (T, n_h) and is indexed by lifecycle age (:478).
- In the batched JAX kernels, `kappa` and `m_grid` are shared across cohorts (`in_axes=None`): lifecycle_jax.py:736/742, :1200/1205, :1409/1414. Neither is in `_solve_inputs_key` (olg_transition.py:686-699).
- The transition aggregates government health spending (panel index 10) but not out-of-pocket spending (index 8/9 is absent from `_PANEL_MEANS_IDX`, olg_transition.py:35).
- `eval_fiscal_results.chk_goods_market` recovers total medical spending as `gov_health / kappa` using the scalar κ (:300-326). `reports/fill_report.py:469` also reads a scalar κ.
- The transition keeps per-age means only. There is no household panel or distribution after a run.
  - Cohort models survive in `olg.birth_cohort_solutions[edu][bp]` and `olg.birth_cohort_later`, except on a household-cache hit, which sets them to None (olg_transition.py:2742-2747).
  - The distributions and value functions are recoverable from these models: `exact_age_means(return_dist=True)`, `exact_panel(rows)`, `V` / `V_alpha`.
  - MIT stitching does not overwrite V.
- No welfare or CEV code exists.
- Weighted `compute_gini` and `_quantile` exist (calibrate.py:113, :130). `compute_wealth_gini` and `compute_consumption_gini` are unweighted.
- `run_fiscal_figures.py` has no flag for selecting a subset of scenarios. Each call runs the baseline, debt financing, τ_l with a debt target and τ_l with an NFA target (:383-430).
- The last full GPU run (both shocks, four scenarios each) took 42.8 min on the A100. 18 min of that was model build plus the baseline fixed point.

### Health data (`code/data/health_flag_GR.csv`, Eurostat SHA, € million)

| | 2010 | 2014 | change |
|---|---|---|---|
| Government health / GDP | 6.63% | 4.57% | −2.06 pp |
| Out-of-pocket / GDP | 2.72% | 2.95% | +0.23 pp |
| Private (CHE − gov) / GDP | 2.99% | 3.37% | +0.38 pp |
| κ_data = gov/CHE | 0.689 | 0.576 | −0.113 |
| Government health, € | 14,821 | 8,050 | −46% |
| Out-of-pocket, € | 6,078 | 5,203 | −14% |

2010 is the pre-programme peak and 2014 the trough. Nominal GDP fell 21% over the window.

### Baseline model levels
- In the baseline, government health spending is 0.054 of Y on average over 2023–27 (`base_budget.gov_health / Y` in `fiscal_2026-10-08d`; recompute over 2026–30). κ = 0.631.
- This implies total medical spending M/Y ≈ 0.0856 and out-of-pocket spending ≈ 0.0316 of Y.

## 2. Model changes

### 2.1 Calendar-time paths for κ and medical spending

Add two optional calendar-time paths. Existing names and equations are unchanged, and both paths default to None:
- `kappa_path` (T,): the coverage rate in period t.
- `m_scale_path` (T,): a multiplier on the level of medical spending in period t, so that m_t(j) = `m_scale_path[t]`·`m_age_profile[j]`·m_good.

When both are None the model is bit-identical to the current code. The paths are defined over calendar time. Each cohort receives its age-indexed slice through the same stitching as `lump_sum_path`.

Template: thread them exactly as `lump_sum_path` is threaded. The subagent map lists every touch point; follow it step by step.
1. `LifecycleConfig`: add the fields and validation next to `lump_sum_path` (lifecycle_perfect_foresight.py:192, :253-259). Build an age-indexed `kappa_t` and an `m_grid_t` (T, n_h) per cohort.
   - NumPy budget: :834-837.
   - NumPy simulation: :1315-1318.
   - NumPy exact columns: :1472, :1508.
2. JAX:
   - Turn `kappa` into a per-age (T,) path, scanned with `m_grid_t`, in `compute_budget_jax`, `solve_lifecycle_jax`, `_state_outcomes_jax`, `simulate_lifecycle_jax` and `exact_age_means_jax`.
   - Switch `m_grid` and `kappa` to `in_axes=0` in all three batched axes tuples.
   - Stack them per cohort in the batched solve, simulate and exact routes (olg_transition.py:560-613, :785-967, :1081). Today these routes pass `ref.m_grid` and `ref.kappa` from the first cohort only.
3. `_solve_inputs_key` (olg_transition.py:686-699): include both paths. Without this, cohorts that differ only in the health path would be deduplicated, which is a silent bug.
4. `solve_cohort_problems`:
   - Baseline extension and the cohort slice use `_extract_cohort_path(..., pre_value=<baseline scalar>)`, as at :1556-1595.
   - MIT baseline config: the baseline cohorts keep scalar κ and scale 1 (:1662-1683).
5. `fiscal_experiments`:
   - Add both keys to `_PRE_TP_KEYS`.
   - Add `delta_kappa_path` and `m_scale_path` to `FiscalScenario`. `_apply_shock` builds the counterfactual paths. `_run_one_simulation` passes them on.
6. Household-cache key (olg_transition.py:2735): include both paths.

### 2.2 Out-of-pocket and total medical spending in the accounts

- Add out-of-pocket spending (panel index 9) to `_PANEL_MEANS_IDX`, so the transition aggregates it. Add total medical spending M_t = gov_health + oop_health as a booked line in `compute_government_budget` output (as a memo item, not a spending line). Update every consumer of the 12-tuple layout.
  - The tuple change invalidates `_cohort_panel_cache` keys. Bump `policy_version`, or add the tuple length to the key.
- `chk_goods_market`: use the booked M_t. Fall back to `gov_health / kappa` only when M_t is absent, for old JSON files.
- Stamp `kappa_path` and `m_scale_path` in `params`. `fill_report.py` reads `kappa_path[0]` when the path is present.

### 2.3 Health-shock calibration (Greece)

The cut lasts five periods, t_s … t_s+4 (2026–2030), and both paths return to baseline from t_s+5 (2031). Given baseline averages over the window, g0 = gov_health/Y and o0 = oop/Y, solve:

    κ1·μ1·(g0+o0) = g0 − 0.0206
    (1−κ1)·μ1·(g0+o0) = o0 + 0.0023

so that

    μ1 = (g0 + o0 − 0.0183)/(g0 + o0),   κ1 = (g0 − 0.0206)/(g0 + o0 − 0.0183).

With g0 ≈ 0.054 and o0 ≈ 0.0316, this gives μ1 ≈ 0.79 and κ1 ≈ 0.50 (indicative; recompute from the baseline). This is a static calibration on baseline Y.
- Report the realised changes in government health/GDP and out-of-pocket/GDP, which differ because Y responds.
- Optional (only if the realised gap exceeds 0.1 pp of GDP): one fixed-point pass that recomputes (κ1, μ1) on the counterfactual Y.

Implement the calibration as a function `health_cut_paths(base, d_gov_gdp, d_oop_gdp, t_s, n_years, T)` in fiscal_experiments.py. Targets and duration are arguments. Two special cases need no new code:
- d_oop_gdp = −d_gov_gdp gives a κ-only cut (μ1 = 1).
- A pure m cut is the case where κ1 = κ0.

Targets read from `code/data/health_flag_GR.csv`. The default window is 2010→2014 and is set by argument.

Start date: t_s = 3 (2026), the same date as the I_g shock.

### 2.4 Driver

`run_fiscal_figures.py`:
- `--shock {G,Ig,health,both,all}`. `both` keeps its current meaning, G and I_g.
- `--scenarios` takes a comma list from {debt, tau_l_debt, tau_l_nfa}, default all. This round runs `--shock Ig,health --scenarios debt,tau_l_debt`.
- I_g shock level: 0.02·Y^base_{t_s−1}, i.e. 2% of baseline output in the year before the shock (2025 for `--shock-year 2026`), constant from t_s on. This replaces 0.02·Y(0) (run_fiscal_figures.py:338-342). With `--shock-year 2023` it falls back to Y(0), so the current runs are reproduced.
- `--health-targets d_gov,d_oop`, `--health-years 5`, `--shock-year 2026` (common to all shocks; `--shock-year 2023` reproduces the current runs).
- The τ_l scheme: a uniform permanent Δτ_l from t_s (2026; τ_l is the baseline path before) that sets terminal B/Y equal to the baseline value (`terminal_debt_gdp`, same T_bal). The same rule applies to the health exercise. In that exercise Δτ_l is negative, because the five-year saving lowers the debt path.
- Write a JSON key `health` with the same structure as `Ig`.

### 2.5 Unanticipated shock in a year after the start of the transition

The current framework solves every counterfactual from t = 0 (2023) with perfect foresight, so a path that differs only from 2026 would be anticipated from 2023. An unanticipated shock at t_s needs the MIT stitching generalised from t = 0 to t_s:

1. Periods t < t_s: the counterfactual equals the baseline. Aggregates, budget lines, B and NFA are copied from the baseline run. Every counterfactual path (shock, Δτ_l, κ, m) is zero-deviation before t_s; `_apply_shock` and the τ_l adjustment profile ψ enforce this.
2. Households: every cohort alive at t_s is stitched at age t_s − bp, the way pre-transition cohorts are stitched at age −bp now.
   - For ages below t_s − bp, the policies (`a/c/l_policy` and `*_policy_alpha`) are the baseline cohort's.
   - From age t_s − bp on, they are solved under the counterfactual paths.
   - This covers the pre-transition cohorts (bp < 0) and the cohorts born 2023–2025 (0 ≤ bp < t_s). The second group is new, because those cohorts are now fully re-solved. Their baseline models come from the baseline run's `birth_cohort_solutions`. Extend `_mit_baseline_cache` with keys for 0 ≤ bp < t_s.
3. Cohorts born from t_s on are solved in full under the counterfactual, as now.
4. The asset distribution of each cohort in t_s is the baseline's, because the policies before t_s are the baseline's. Generalise `check_a0_predetermination.py` and the evaluator check `a0_predetermined` to test predetermination at t_s.
5. The debt path: B is the baseline's through t_s. `compute_debt_path` runs from B_{t_s} with the counterfactual primary deficit.
6. Multipliers and deviations are reported from t_s. The cumulative multiplier discounts to t_s.

Implementation: one parameter `shock_period` (default 0) on `FiscalScenario`, passed to `simulate_transition(shock_period=)` and `solve_cohort_problems`. Name the stitching age `pre = shock_period − bp` wherever it is `−bp` now. With `shock_period = 0` the code path is unchanged.

## 3. Outputs

### 3.1 Aggregates (extend `compare_scenarios` variable lists, run_fiscal_figures.py:460-501)
- Y, C, K_domestic, L (efficiency units), aggregate hours, w, I_g and K_g/Y.
- NFA/Y, B/Y, primary deficit/Y, τ_l, revenue/Y and spending/Y.
- Government health/Y, out-of-pocket/Y, and total medical/Y.
- Each variable as a level path under the baseline and each scheme, plus the percent (or pp) deviation from the baseline. Years 2023–2070 on the x-axis, as in the existing figures.
- Keep the existing `macro_overview`, `prices_sanity`, `fiscal_decomp` and `debt_fan_chart` figures.

### 3.2 Household means
For each reporting period t, compute from the exact distribution of each living cohort, weighted by cohort weights × education shares (× `later_share` where relevant):
- mean consumption, hours of the employed, labour income, disposable income and assets,
- for age groups 25–44, 45–64 and 65+.

Figure: percent deviation from the baseline by age group, one panel per variable.

### 3.3 Inequality
Weighted cross-section of all living households (model ages 25–84) in each period t. The income measures are:
- **Labour income:** w·κ_j·y·h·l·α_mult. Unemployment benefits are excluded, and the population is employed workers only.
- **Disposable income:** the calibration definition, `_disposable_income` (gross labour income + UI + pension − τ_l − τ_p).
- **Disposable income net of out-of-pocket health spending:** the previous measure minus (1−κ_t)·m_t(j). This is the measure on which the health shock acts directly.
- Consumption, and assets (wealth).

Statistics:
- Gini of each measure.
- P90/P10 and P90/P50 for consumption and the two disposable-income measures.
- Wealth Gini and the top-10% wealth share.

Implementation:
- New module `distribution_stats.py`. It builds the period-t cross-section from `exact_panel(rows)` of each cohort model, with row = age t − bp, and computes the statistics with the weighted `compute_gini` and `_quantile` from calibrate.py. Add a weighted top-share function.
- Do not use the unweighted `compute_wealth_gini` or `compute_consumption_gini`.
- Reporting periods: every year 2023–2040, then every 5 years to 2070. These cover the five-year window and the long run.

Figure: change from the baseline (Gini points; ratio points) by scheme.

### 3.4 Welfare
Consumption-equivalent variation λ: the permanent proportional change in baseline consumption from the shock date onward that equates expected lifetime utility with the counterfactual. With log utility and additively separable labour disutility, at state s and age j,

    λ(s, j) = exp[(V_cf(s, j) − V_base(s, j)) / D(j)] − 1,   D(j) = Σ_{k≥j} β^{k−j} Π_{i=j}^{k−1} s(i).

D(j) does not depend on the state, because survival depends on age only. Consumption is detrended. With log utility the trend enters V additively and cancels in the difference.

Groups:
- **Cohorts alive in 2026 (t_s):** λ at their age in 2026, averaged over the 2026 distribution (the baseline's, by §2.5), using cohort mass and education weights. Report by birth year, and by quintile of disposable income in 2026 for all cohorts alive in 2026 together.
- **Cohorts born from 2027 on:** λ at age 0, by birth year.

Requirements:
- Baseline V for the same cohort, from the baseline run's cohort models. Store the baseline V slices needed (age t_s − bp for living cohorts, age 0 for later ones) before running the counterfactual. Policy stitching does not supply V.
- Disable the household cache for the runs that produce welfare or distribution output (`household_cache_size = 0`). Alternatively, extract the outputs inside the run before the next scenario can hit the cache.
- Health exercise only: also report λ for the κ-only and m-only components. Run two extra debt-financed simulations, one with (κ1, μ = 1) and one with (κ0, μ1). Then the reader can separate the mechanical gain from the lower m from the transfer between government and households.

Check: λ ≡ 0 when the counterfactual equals the baseline. Add a test.

Figure: λ by birth year, one line per scheme, with a shaded region for cohorts alive in 2026. Bar chart by 2026 income quintile.

### 3.5 Tables
One table per exercise (LaTeX body, as `reports/fiscal_figures.py` writes it). Rows:
- Multiplier: cumulative and on impact.
- Δτ_l.
- B/Y in 2030, 2040, 2060 and 2070.
- Change in the Ginis in 2026, 2030, 2036 and the long run.
- Average λ for cohorts alive in 2026 and for newborns in 2027, 2036 and 2056.

## 4. Tests (add to `test_fiscal_experiments.py` or a new `test_policy_exercises.py`)
1. Paths set to the baseline constants give results bit-identical to paths set to None: NumPy and JAX, small economy, one education group, 30-point grid (as TestPhase6Features).
2. NumPy and JAX agree on a small economy with a nonconstant `kappa_path` and `m_scale_path`. Use exact aggregation, tolerance as in `validate_backends.py`.
3. Two cohorts differing only in the health path are not deduplicated (`_solve_inputs_key`).
4. The accounting identity gov_health + oop = M_t holds period by period, and M_t scales with `m_scale_path`.
5. `health_cut_paths` reproduces the two targets on baseline Y. The κ-only and m-only special cases come out as stated in §2.3.
6. The weighted Gini and quantiles on the exact cross-section equal those of the replicated sample (integer weights).
7. λ = 0 when the counterfactual equals the baseline. In a log-utility toy, scaling consumption by (1+x) in every period gives λ = x.
8. The goods-market check passes on a health-shock run with booked M_t.
9. With `shock_period = 0` the results are bit-identical to the current code. With `shock_period = 3`, aggregates, budget lines and every cohort's assets equal the baseline's through 2025, and the 2026 asset distribution of every living cohort is the baseline's.
10. With `shock_period = 3` and a zero shock, the counterfactual equals the baseline in every period.

Run the full suite on the GPU instance (15 float64 JAX tests are skipped on Metal).

## 5. Runs

1. Local smoke run: small economy, both exercises, both schemes. Confirm that all outputs are produced and that the eval checks pass.
2. A100 run, `calibration_input_GR.json`:
   - `--backend jax --shock Ig,health --scenarios debt,tau_l_debt`, n_sim 2000 (exact aggregation).
   - Add the two decomposition runs for the health exercise.
   - Estimated 40–60 min: 18 min build and baseline, about 6 min per scenario from the last run, plus extraction. The extraction cost is not measured yet.
   - Run in the background with a monitor on NaN, Traceback, FAIL and a non-converging bisection.
3. Output directory `output/policy_2026-10-XX/` containing:
   - `fiscal_results.json` with the keys `Ig` and `health`, and new subkeys `distribution` and `welfare` per scenario,
   - figures, table bodies, `run.log` and `eval.log`.
4. `/eval-fiscal` on the JSON. `/fiscal-note` for the one-page notes.

## 6. Remaining items
- **Existing driver issues to fix on the way:**
  - The summary line "tau_y 0.05950 to 2060" in run.log is mislabelled. The stamped path is 0.04412 from 2026.
  - The G multiplier prints 0.000.
- **Welfare caveat, for the write-up:** in the model the fall in m has no health cost. The κ-only and m-only runs (§3.4) measure that part of λ.
