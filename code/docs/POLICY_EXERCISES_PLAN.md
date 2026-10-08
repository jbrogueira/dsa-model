# Plan: public-investment and health-coverage policy exercises

Written 2026-10-08; revised the same day after three independent audits (implementation, economics, data). Branch `trend-growth`.

**Status (2026-10-08).** Sections 2-5 are implemented, with the tests of section 4 in `test_policy_exercises.py`. Section 6: the local smoke runs are done (the test economy of `policy_reference_case.py` through `--tiny`, and the hard-coded economy of `run_fiscal_figures.py` with `--shock-year 2026`). The A100 run (6.3) and the §1 numbers recomputed from the new baseline (6.1) are not done. Points where the implementation fills in or departs from the text:
- The reference arrays of test 11 are `tests_data/policy_reference_e80b119.npz`, written by the code of e80b119 on the small economy of `policy_reference_case.py`. The code of 59da648 writes the same arrays, and so does the new code with the new options off.
- The transition always uses the kernel variants with κ and `m_grid` per cohort (`_tr`, `_tr_pyc`), with or without health paths. The base tuples are unchanged.
- The fixed-point pass of §2.3 is in the driver. When the realised changes of the debt-financed run miss a target by more than 0.1 pp of output, (κ1, μ1) are recomputed on that run's output (`health_cut_paths(Y_eval=)`) and the health set is run again.
- The welfare measure needs log utility. At γ ≠ 1 (the hard-coded test economy) `distribution_stats.extract` skips it and says so.
- The cross-section check against A_t and C_t runs only under exact aggregation.
- The evaluator's `bisection_target` compared B/Y at T_bal, while the condition and the driver's target are dated T_bal − 1. It now uses T_bal − 1.
- The run time of the extraction at production size is not measured.

**Calibration.** The runs use `calibration_input_GR.json` after the recalibration that the user is doing first. The current HEAD config (b14f289) is not the one behind `output/fiscal_2026-10-08d`, which is 91a75e5. Since then δ_g has gone to 0.04316, the A/Y target to 2.96 and τ_k is calibrated, so θ in `_derived` is stale. The model numbers in §1 come from `fiscal_2026-10-08d`. They are indicative and are recomputed from the new baseline before the runs.

## 0. Scope

Two exercises. Both use the same financing schemes and report the same outputs.

| Exercise | Shock | Financing schemes run now |
|---|---|---|
| I_g | Permanent rise in public investment from 2026. A constant detrended level equal to 2% of baseline output in 2025, the year before the shock. | (i) debt; (ii) permanent Δτ_l from 2026 with a terminal debt-to-GDP target |
| Health | Five-year cut in coverage κ and in the level of medical spending m, 2026–2030, back to baseline in 2031 | (i) debt; (ii) permanent Δτ_l from 2026 with a terminal debt-to-GDP target; (iii) Δτ_l over 2026–30 only |

The G shock and the τ_l scenario with a terminal NFA target stay in the code and stay selectable. They are not run in this round.

Outputs for every scenario:
1. Aggregates (§3.1).
2. Household means by age group (§3.2).
3. Inequality (§3.3): Gini, P90/P10 and P90/P50 for consumption, labour income and disposable income, plus the wealth Gini and the top-10% wealth share.
4. Welfare (§3.4): consumption-equivalent variation by birth cohort, and by income quintile for the cohorts alive in 2026.

Settled with the user:
- **Timing.** Both shocks are unanticipated and hit in 2026 (t_s = 3; t = 0 is 2023). Through 2025 the economy follows the baseline. In 2026 households learn the full counterfactual path and re-optimise from their 2026 state (§2.5).
- **I_g shock size.** It is 2% of baseline Y_2025 (0.0194 at the current baseline), which is 1.87% of Y_2026, because Y jumps 6.8% in 2026 when the output tax steps down. The GDP share then drifts with Y_t, to about 2.25% in 2060. Plot ΔI_g/Y_t.
- **Health targets** (§2.3), as changes in shares of GDP on baseline output:
  - government health spending −1.29 pp;
  - household health spending +0.26 pp, measured as current health expenditure less government and compulsory schemes (CHE − HF.1);
  - both are the mean changes over 2011–15 relative to 2010. The code takes any targets.
- **κ and m in the code.** Both can vary over calendar time. For Greece they move jointly.
- **τ_l target.** The τ_l scheme targets the baseline terminal B/Y, currently −6.45.
- **Window-only variant.** The health exercise also runs a τ_l variant that moves only over 2026–30.
- **Welfare decomposition.** κ-only and m-only runs are made, and the interaction term is reported.
- **Disposable income.** The broad definition is used (§3.3), with the calibration definition reported alongside.
- **Rationale in the report.** The design rationale for the health experiment is in `reports/calibration_report.tex`, under "Policy experiments", in the block "Temporary cut in public health spending (design)".

## 1. Facts the plan rests on (checked 2026-10-08)

### Medical spending in the model
- Medical spending is a pure expenditure in the budget constraint. A household of age j pays (1−κ)·m(j) and the government pays κ·m(j). Here m(j) = `m_age_profile[j]`·m_good.
- Health enters neither utility, survival nor productivity (n_h = 1). A fall in m therefore frees resources at no cost to anyone in the model.
- The means-tested top-up is `transfer = max(0, floor − budget)`, and out-of-pocket spending is inside the budget (lifecycle_perfect_foresight.py:855-858; lifecycle_jax.py:892-896). The top-up therefore absorbs part of a κ cut for households at the floor.
- κ in the model is gov/(gov + household). Calibration: κ = 0.053/0.084 = 0.631.
  - 0.053 is public health spending from the general government accounts (ESA, purchases net of imputed contributions).
  - 0.084 is the total with household health consumption.

### Current code
- κ is a scalar: `LifecycleConfig.kappa` (lifecycle_perfect_foresight.py:127, :395). m varies by age only: `m_grid` (T, n_h), :478.
- In the batched JAX kernels, `kappa` and `m_grid` are shared across cohorts (`in_axes=None`: lifecycle_jax.py:736/742, :1200/1205, :1409/1414), and they are absent from `_solve_inputs_key` (olg_transition.py:686-699).
- The calibration cross-section kernels use the same base axes tuples: `_cross_section` (lifecycle_jax.py:1616-1700) and `_cross_section_exact` (:1464-1560).
- The transition aggregates government health spending (panel index 10) but not out-of-pocket spending or hours (`_PANEL_MEANS_IDX`, olg_transition.py:35).
- `eval_fiscal_results.chk_goods_market` recovers M = `gov_health / kappa` using the scalar κ (:300-326). `eval_fiscal_results.main` loops over `('G', 'Ig')` only (:825).
- The transition keeps per-age means only.
  - Cohort models survive in `olg.birth_cohort_solutions[edu][bp]` and `olg.birth_cohort_later`, but a household-cache hit sets them to None (:2742-2747).
  - The distributions and value functions are recoverable: `exact_age_means(return_dist=True)`, `exact_panel(rows)`, `V` / `V_alpha` of shape (n_alpha, T, n_a, n_y, n_h, n_y). V is dropped only with `jax_policies_on_device=True`.
  - MIT stitching does not overwrite V.
- No welfare code exists.
- Weighted `compute_gini` and `_quantile` exist (calibrate.py:113, :130). `compute_wealth_gini` and `compute_consumption_gini` are unweighted.
- `run_fiscal_figures.py` has no scenario-subset flag.
  - `run_experiment_set` returns a 4-tuple and maps every non-G shock to the I_g scenarios (:383-430).
  - The JSON always writes `nfa_constrained` (:645).
  - `reports/fiscal_figures.py` hard-codes `SHOCKS = {'G','Ig'}` and the three scenarios (:27-30).
- The last full GPU run took 42.8 min on the A100: 17.9 min build and baseline fixed point, then 20 transitions of which 14 were fresh (≈ 106 s each).
- A G multiplier of 0.000 under debt financing is the model's result. G enters neither utility nor production and the debt is held abroad. It is not a bug.

### Health data (`code/data/health_flag_GR.csv`; Eurostat `hlth_sha11_hf`, € million, current prices; GDP nominal)
Government = HF.1 (government and compulsory schemes). Household = CHE − HF.1. Out-of-pocket = HF.3.

| Share of GDP | 2010 | 2014 | 2010→2014 | mean 2011–15 − 2010 |
|---|---|---|---|---|
| Government health | 6.63% | 4.57% | −2.06 pp | −1.29 pp |
| Household (CHE − gov) | 2.99% | 3.37% | +0.38 pp | +0.26 pp |
| Out-of-pocket (HF.3) | 2.72% | 2.95% | +0.24 pp | +0.17 pp |
| κ = gov/(gov + HF.3) | 0.709 | 0.608 | −0.101 | |
| gov/CHE | 0.689 | 0.576 | −0.113 | |

- In euros, 2010→2014: government spending −46%, CHE − gov −11%, out-of-pocket −14%, CHE −35%. Nominal GDP fell 21%.
- The share of government health spending in GDP peaks in 2010. Its euro level peaks in 2009. 2010 is the first programme year (first MoU May 2010).
- A flat five-year cut equal to the 2010→2014 change gives −10.3 pp-years of GDP. The data over 2011–15 give −6.4 pp-years, so the mean change over 2011–15 is the one that matches the cumulative cut.

### Baseline model levels (fiscal_2026-10-08d; recompute after recalibration)
- 2026–30 averages: gov_health/Y = 0.0538 (0.0525 in 2026 rising to 0.0550 in 2030), M/Y = 0.0852, household health/Y = 0.0315.
- At the default targets: κ1 ≈ 0.546 (from 0.631) and μ1 ≈ 0.880 (m 12% lower).
- The government cut is 24% of the model's baseline share. The household level rises 8%.

### Comparison of model values with data, for the calibration report
This is an open item for the calibration report, not for this plan's runs. The 2023 data on each basis:
- **Public health spending:**
  - ESA: 5.3%.
  - SHA HF.1: 5.12%.
- **Total health spending:** CHE 8.41%.
- **Household health spending:**
  - CHE − ESA gov: 3.1%.
  - CHE − HF.1: 3.29%.
  - HF.3: 2.89%.

The model's 2026–30 values (5.38%, 8.52%, 3.15%) sit above the 2023 targets, because the baseline health share rises with ageing.

## 2. Model changes

### 2.1 Calendar-time paths for κ and medical spending

Add two optional calendar-time paths. Existing names and equations are unchanged, and both paths default to None:
- `kappa_path` (T,): coverage in period t.
- `m_scale_path` (T,): a multiplier on the level of medical spending in period t, so that m_t(j) = `m_scale_path[t]`·`m_age_profile[j]`·m_good.

Each cohort receives its age-indexed slice through the same stitching as `lump_sum_path`. When both paths are None the model is bit-identical to the current code. Thread them as `lump_sum_path` is threaded; every site is listed below.

1. **NumPy (`lifecycle_perfect_foresight.py`).**
   - `LifecycleConfig` fields and validation next to `lump_sum_path` (:192, :253-259).
   - An age-indexed `kappa_t` and `m_grid_t` per cohort.
   - Budget :834-837, simulation :1315-1318, exact columns :1472, :1508.
2. **JAX kernels (`lifecycle_jax.py`).**
   - `kappa` becomes a per-age (T,) path, scanned with `m_grid_t`, in `compute_budget_jax`, `solve_lifecycle_jax`, `_state_outcomes_jax`, `simulate_lifecycle_jax` and `exact_age_means_jax`.
   - The terminal period: `_solve_terminal_period_jax` receives `m_grid_path[T-1]` and the scalar κ (:629-634). κ is at position 7 of `model_params` (:618). Give κ a slot in the `period_params` tuple (:294, :316, :675-700) and pass the terminal-age values.
   - `LifecycleModelJAX` constructor (:1755, :1761) and its kernel calls (:1850, :2041, :2094, :2138, :2224, :2283).
3. **JAX axes.**
   - Do not change the base tuples `_SOLVE_IN_AXES`, `_SIMULATE_IN_AXES`, `_EXACT_IN_AXES`. The calibration cross-section kernels use them with a shared `m_grid` and a scalar κ, and axis 0 would break `cross_section_batched`/`cross_section_exact`, `normalize_A_tfp` and the scale loop.
   - Build transition-only variants with `_axes_override(..., m_grid=0, kappa=0)`, and the `_PYC` variants that combine this with the per-cohort income matrices (:765-773, :1233, :1435).
   - Alternative: broadcast κ and `m_grid` to (C, …) inside both cross-section kernels, as `ls_c = bc(...)` does, and switch the base tuples.
4. **Transition batching (`olg_transition.py`).**
   - Stack κ and `m_grid` per cohort in the batched solve (:560-613), simulate (:785-967) and exact (:1081) routes. Today these pass `ref.m_grid`, `ref.kappa` from the first cohort.
   - Fix the positional unpack `sliced[:20]` / `sliced[20]` in `_simulate_cohorts_jax_batched` (:936-939), which shifts when the stacks are added.
5. **Deduplication.** `_solve_inputs_key` (:686-699) includes both paths. This key also drives deduplication in `_exact_cohort_age_means` (:1007-1010).
6. **`OLGTransition` and `simulate_transition`.**
   - Constructor defaults, as for `lump_sum_path` (:298).
   - `simulate_transition` arguments, extended to full length like `lump_path_full` (:2625), and passed at both `solve_cohort_problems` call sites (bequest-loop branch :2685, normal branch :2752).
7. **`solve_cohort_problems`.**
   - Baseline extension and the cohort slice via `_extract_cohort_path(..., pre_value=...)` (:1556-1595).
   - The MIT baseline config takes the `pre_tp` baseline paths (:1662-1683).
8. **`fiscal_experiments`.**
   - Both keys in `_PRE_TP_KEYS`.
   - `run_baseline` copies the baseline paths into `pre_tp`, as for `lump_sum_path` (:1395-1396).
   - `FiscalScenario` gets `delta_kappa_path` and `delta_m_scale_path`, additive around the baseline paths, keeping the `delta_*` convention. `_apply_shock` builds the counterfactual paths. `_run_one_simulation` passes them on.
   - The driver puts `kappa_path = full(T, κ0)` and `m_scale_path = ones(T)` into `base_paths` before `run_baseline`. `_apply_shock` has no access to the config κ.
9. **Household-cache key** (`_household_inputs_key`, :2419-2452; :2735) includes both paths and `shock_period` (§2.5).

### 2.2 Out-of-pocket and total medical spending in the accounts

Do not change the 12-element layout of per-age means. At calendar t every living cohort faces κ_t, so
- oop_t = gov_health_t·(1−κ_t)/κ_t and M_t = gov_health_t/κ_t (κ_t > 0).

Book both as memo lines in the budget output, alongside `gov_health`. They are not spending lines.
- `chk_goods_market` uses the booked M_t. It falls back to `gov_health / kappa` only for old JSON files.
- Stamp `kappa_path`, `m_scale_path` and `shock_period` in `params`.
- `fill_report.py:469` reads the config κ for the baseline report, where there is no path, and needs no change.

### 2.3 Health-shock calibration

The cut lasts five periods, 2026–2030 (t_s … t_s+4), and both paths return to baseline in 2031. Let g0 and o0 be the baseline window averages of gov_health/Y and household health/Y (o0 = g0(1−κ0)/κ0), and Δg, Δo the targets. Solve

    κ1·μ1·(g0+o0) = g0 + Δg
    (1−κ1)·μ1·(g0+o0) = o0 + Δo

so that

    μ1 = (g0 + o0 + Δg + Δo)/(g0 + o0),   κ1 = (g0 + Δg)/(g0 + o0 + Δg + Δo).

- **Defaults:** Δg = −0.0129 and Δo = +0.0026 (mean 2011–15 relative to 2010; household = CHE − HF.1). Arguments select the window and the household concept (CHE − HF.1 or HF.3), read from `code/data/health_flag_GR.csv`.
- **Calibration on baseline Y.** The equations are linear in window averages, so a constant (κ1, μ1) hits the average targets exactly on baseline Y. Report the realised changes, which differ because Y responds. Do one fixed-point pass on counterfactual Y only if the gap exceeds 0.1 pp of GDP.
- **Function:** `health_cut_paths(base, d_gov_gdp, d_hh_gdp, t_s, n_years, T)` in fiscal_experiments.py. It returns `delta_kappa_path` and `delta_m_scale_path`.
- **Special cases:**
  - d_hh_gdp = −d_gov_gdp gives a κ-only cut (μ1 = 1).
  - κ1 = κ0 gives an m-only cut.
- **Reference values** at the current baseline:

  | Targets (Δg, Δo) | κ1 | μ1 |
  |---|---|---|
  | 2010→2014, CHE − HF.1 (−2.06, +0.38) | 0.485 | 0.803 |
  | 2010→2014, HF.3 (−2.06, +0.24) | 0.496 | 0.787 |
  | mean 2011–15, HF.3 (−1.29, +0.17) | 0.553 | 0.869 |

### 2.4 Driver

`run_fiscal_figures.py`:
- **Flags:**
  - `--shock` takes a comma list from {G, Ig, health}. `both` stays an alias for `G,Ig`.
  - `--scenarios` takes a comma list from {debt, tau_l_debt, tau_l_nfa, tau_l_window}, default `debt,tau_l_debt,tau_l_nfa`.
  - `--shock-year 2026`. With `--shock-year 2023` the current runs are reproduced.
  - `--health-targets d_gov,d_hh`, `--health-window 2011-2015`, `--health-household che_minus_gov|hf3`, `--health-years 5`.
  - This round: `--shock Ig,health --scenarios debt,tau_l_debt,tau_l_window`. `tau_l_window` applies only to health.
- **I_g shock level:** 0.02·Y^base_{t_s−1}, constant from t_s, replacing 0.02·Y(0) (:338-342). With `--shock-year 2023` it falls back to Y(0).
- **Permanent τ_l rule:** a uniform Δτ_l from t_s, profile ψ = `back_loaded(T_TR + n_post, t_s)` (fiscal_experiments.py:202), so it is zero before 2026. Δτ_l sets B/Y at T_bal equal to the baseline's (`terminal_debt_gdp`).
  - Report the extra term this target implies: the target is a ratio on a baseline debt path that turns strongly negative, so matching it needs government assets of 6.45·ΔY beyond ΔB = 0. For I_g that is about 3% of Δτ_l.
  - Report ΔB at T_bal as a check.
- **Window τ_l rule (health only):** ψ = 1 over 2026–30 and 0 elsewhere. Δτ_l is set by the same terminal debt-ratio target.
- **Return values and JSON:**
  - `run_experiment_set` returns a dict of the scenarios that ran, and the JSON writes only those.
  - JSON key `health`, with the structure of `Ig`.
- **Report consumers:**
  - `reports/fiscal_figures.py`: extend `SHOCKS`/`SCENARIOS` and make them presence-aware.
  - `regen_fiscal_figures_from_json.py` already skips absent scenarios (:217-224). Keep its MACRO/FISCAL lists in sync with the driver.
- **Driver fix:** the run.log summary prints the 2023 pin as "tau_y 0.05950 to 2060". The rate from 2026 is 0.04412 (:274-275). The `baseline_closure.py` docstring is equally out of date.

### 2.5 Unanticipated shock in a year after the start of the transition

The current framework solves every counterfactual from t = 0 (2023) with perfect foresight. A shock at t_s > 0 therefore needs the MIT stitching generalised from 0 to t_s.

1. **Household problem.** Every cohort alive at t_s is stitched at age t_s − bp, as pre-transition cohorts are stitched at −bp now.
   - For ages below t_s − bp, the policies (`a/c/l_policy` and `*_policy_alpha`) are the baseline cohort's.
   - From age t_s − bp on, they are solved under the counterfactual paths.
   - This covers cohorts with bp < 0 and the cohorts born 2023–25 (0 ≤ bp < t_s), which are fully re-solved today.
   - It covers both retirement parts of split cohorts (`birth_cohort_later`).
   - Cohorts born from t_s on are solved in full under the counterfactual.
2. **Sites that hard-code 0 or −bp:**
   - olg_transition.py:1641 (`birth_period < 0`) and :1642 (`pre = -birth_period`);
   - the JAX stitch list :1773-1781 (`range(min_birth_period, 0)`, the `later` parts with `bp < 0`, `pre = -bcs_key[1]`);
   - the cache fill in `run_baseline` (fiscal_experiments.py:1402-1414, `bp < 0`).

   `run_baseline` takes the largest `shock_period` of the scenarios and fills `_mit_baseline_cache` for (edu, bp) and (edu, bp, 'later') with bp < t_s. A stitch-list entry with no cache key is skipped silently (:1778 `continue`). Make that an error.
3. **No copying of aggregates.** With stitched baseline policies, zero-deviation paths before t_s and common random numbers by (edu, bp), the counterfactual reproduces the baseline through 2025 by construction. Wages are unchanged through t_s because K_g responds to I_g with a lag (:2584-2593). `compute_debt_path` from the existing `B_initial` reproduces B through t_s.
   - Assert `PD_cf[:t_s] == PD_base[:t_s]` and `A_cf[:t_s+1] == A_base[:t_s+1]`.
   - Copying aggregates would hide stitching errors.
4. **Predetermination at t_s.** Generalise `check_a0_predetermination.py` and the evaluator check `a0_predetermined` to test assets through t_s.
5. **Refusal.** Refuse `shock_period > 0` together with `recompute_bequests=True`. Bequests are unaffected only because τ_beq = 1 and the fiscal runs do not recompute bequests.
6. **Interface.** `shock_period` (default 0) on `FiscalScenario`, passed to `simulate_transition(shock_period=)` and `solve_cohort_problems`. With `shock_period = 0` the code path is unchanged. Include `shock_period` in the household-cache key.

## 3. Outputs

### 3.1 Aggregates (extend the variable lists, run_fiscal_figures.py:460-501)
- Y, C, K_domestic, L (efficiency units), aggregate hours of the employed, w, I_g/Y and K_g/Y. Hours are not in the transition means; take them from the distribution extraction (§3.2).
- NFA/Y, B/Y, primary deficit/Y, revenue/Y, spending/Y, and τ_l (from the base path plus Δτ_l·ψ; it is not in `cf_macro`).
- Government health/Y, household health/Y, total medical/Y, and means-tested transfers/Y.
- Each as a level under the baseline and each scheme, plus the deviation from the baseline. `compare_scenarios` plots periods 0…T−1; put calendar years on the axis (2023–2070), as `reports/fiscal_figures.py` does.
- Keep `macro_overview`, `prices_sanity`, `fiscal_decomp` and `debt_fan_chart`.

### 3.2 Household means
- **Weights.** In each reporting period t the cross-section weight of a state is
  - `_aggregation_weights(t)[t − bp]` × education share × retirement-part share (`later_share` or 1 − `later_share`) × state mass from `exact_panel`.
  - The exact mass already carries cumulative survival. Population-at-t weights would count it twice.
  - The weights sum to 1 in each period.
- **Statistics:** mean consumption, hours of the employed, labour income, disposable income and assets, for ages 25–44, 45–64 and 65+.
- **Figure:** percent deviation from the baseline by age group, one panel per variable.

### 3.3 Inequality
Cross-section of all living households, model ages 25–99 (T = 75), weighted as in §3.2. The measures are:
- **Labour income:** wage income of the employed, w·`wage_age_profile[j]`·y·h·l·α_mult. UI is excluded. The population is the employed.
- **Disposable income (broad, default):** labour income + UI + pension + (1−τ_k)·r·a + lump-sum transfer + means-tested top-up − labour-income tax − payroll tax.
- **Disposable income net of household health spending:** the broad measure minus (1−κ_t)·m_t(j).
- **Disposable income, calibration definition:** `_disposable_income` (calibrate.py:570-580), reported alongside for comparison with the calibration moments.
  - It excludes capital income, the lump sum and the top-up.
  - It is zero for the long-term unemployed.
  - It goes negative net of household health spending.
- **Consumption, and assets (wealth).**

Statistics:
- Gini of each measure.
- P90/P10 and P90/P50 for consumption and the disposable-income measures.
  - Report the share of the population with non-positive values.
  - For the ratios, restrict to positive values, stated in the table note.
- Wealth Gini and the top-10% wealth share.

Implementation:
- New module `distribution_stats.py`.
  - It builds the period-t cross-section from `exact_panel(rows)` of each cohort model (row = age t − bp) and computes the statistics with the weighted `compute_gini` and `_quantile` from calibrate.py, plus a weighted top-share function.
  - Do not use the unweighted `compute_wealth_gini` or `compute_consumption_gini`, and do not copy the calibration's age weighting.
- Cost:
  - About 730 models per scenario are alive over 2023–2070 (122 birth periods × 3 education groups × 2 retirement parts), with about 12,500 states each.
  - `exact_panel` jits with a static retirement age and retraces for each row length. Pad rows to a fixed length and batch by retirement group over the `_EXACT_IN_AXES + (0,)` route, as `_cross_section_exact` does.
  - Time this on the small economy before the A100 run. Keep the NumPy `_exact_columns` for tests only: it loops in Python over states when `transfer_floor > 0`.
- Reporting periods: every year 2023–2040, then 2045–2070 every five years (24 periods).

Figure: change from the baseline (Gini points; ratio points) by scheme.

### 3.4 Welfare
Consumption-equivalent variation: the permanent proportional change in baseline consumption from 2026 on that gives a cohort the same expected lifetime utility as the counterfactual.

**Formula.** Utility is log (γ = 1) with additively separable labour disutility, the trend enters V additively, and the dead get zero. The measure for cohort c at age j is

    λ_c = exp[(E_μ V_cf(·, j) − E_μ V_base(·, j)) / D_c(j)] − 1,   D_c(j) = Σ_{k=j}^{T−1} β^{k−j} Π_{i=j}^{k−1} s_c(i).

- μ is the distribution of the cohort at the start of age j:
  - for cohorts alive in 2026 it is the 2026 distribution, which equals the baseline's by §2.5;
  - for cohorts born from 2027 on it is `_initial_distribution()` at age 0.
- Mix across education groups and both retirement parts before taking the log.
- D_c(j) uses the cohort's own survival schedule (`model.survival_probs[:, 0]`), because survival is cohort-specific. It does not depend on the state, because n_h = 1.
- V_alpha[α, j, ·] is the value at the start of age j, after y_j is realised. Its C-order flattening must match `exact_panel`'s state order over (α, a, y, h, y_last); assert this.

**Groups:**
- **Cohorts alive in 2026:** λ_c by birth year.
- **Cohorts born 2027 on:** λ_c by birth year.
- **By income quintile:** state-level λ(s) = exp[(V_cf(s) − V_base(s))/D] − 1, averaged within quintiles of baseline 2026 broad disposable income across all cohorts alive in 2026. Ranking on baseline income is predetermined. This is a different statistic from λ_c and is labelled as such.

**Requirements:**
- Baseline V slices from the baseline run inside `run_experiment_set`: age t_s − bp for living cohorts, age 0 for later ones, both retirement parts. Do not take them from `run_baseline`, because `solve_baseline` reruns it.
- Only the slice at t_s − bp of a counterfactual model is valid. At younger ages the V of a stitched model is that of an anticipated shock.
- `household_cache_size = 0` and `jax_policies_on_device = False` in these runs.
- Extraction after `run_tax_financed` is valid for `terminal_debt_gdp`, where the last evaluation is at the root.

**Health decomposition.**
- Two extra debt-financed runs: κ-only (κ1, μ = 1) and m-only (κ0, μ1).
- The components do not add up. The interaction is (κ0 − κ1)(1 − μ1)·M in household spending, which at the defaults is about 0.09 pp of GDP.
- Report total, the two components and the residual (total − κ-only − m-only), for λ and for the main aggregates.
- The κ-only run does not hit the government-spending target.

Check: λ = 0 when the counterfactual equals the baseline.

Figures:
- λ_c by birth year, one line per scheme, with cohorts alive in 2026 shaded.
- A bar chart of state-level λ by 2026 income quintile.

### 3.5 Multipliers and tables
**Multipliers.**
- `fiscal_multiplier` re-trends by ΠΓ and does not discount (fiscal_experiments.py:1676-1690). Keep that definition, with sums from 2026.
- Per-period multipliers before 2026 are not defined.
- **Health multiplier:** ΔY over the net change in government spending, Δgov_health + Δtransfers (the top-up absorbs part of the cut). The sign convention is that a cut is a negative spending change, so a positive multiplier means output falls when spending falls.

**One table per exercise** (LaTeX body, written by `reports/fiscal_figures.py`). Rows:
- multipliers (cumulative 2026–35 and over the horizon; impact);
- Δτ_l;
- B/Y in 2030, 2040, 2060 and 2070;
- change in the Ginis in 2026, 2030, 2036 and 2070;
- λ_c for cohorts alive in 2026 (mean) and for newborns in 2027, 2036 and 2056.

## 4. Tests (new `test_policy_exercises.py`; small economy, one education group, 30-point grid; not inside `TestPhase6Features`, which hangs)
1. Health paths set to the baseline constants give results bit-identical to None, on NumPy and on JAX.
2. NumPy and JAX agree with a nonconstant `kappa_path` and `m_scale_path`, under exact aggregation, at `TestExactAggregation`'s tolerance.
3. The base-year cross-section (JAX, simulation and exact) is unchanged after the axes change.
4. Two cohorts differing only in the health path are not deduplicated, on the solve and the exact route.
5. Runs differing only in the health path, or only in `shock_period`, are not served from the household cache.
6. The booked M_t equals gov_health/κ_t, M_t scales with `m_scale_path`, and gov_health + household = M_t.
7. `health_cut_paths` reproduces the two targets on baseline Y, and the κ-only and m-only special cases come out as stated in §2.3.
8. The weighted Gini and quantiles on the exact cross-section equal those of the replicated sample (integer weights).
9. The §3.2 weights sum to 1, and the weighted means of assets and consumption reproduce A_t and C_t of the exact-aggregation transition.
10. λ = 0 when the counterfactual equals the baseline, on JAX with `shock_period = 3` and deduplicated models. In a log-utility toy, scaling consumption by (1+x) in every period gives λ = x.
11. With `shock_period = 0` the results equal stored reference arrays from e80b119. With `shock_period = 3`:
    - aggregates, budget lines and every cohort's assets equal the baseline's through 2025;
    - the 2026 asset distribution of every living cohort, including split cohorts and those born 2023–25, is the baseline's.
12. With `shock_period = 3` and a zero shock, the counterfactual equals the baseline in every period.
13. ψ is zero before t_s in a τ_l run, and zero outside 2026–30 in the window variant.
14. `shock_period > 0` with `recompute_bequests=True` raises.
15. The evaluator runs on a health JSON, and `reports/fiscal_figures.py` runs on a JSON without `nfa_constrained` or G.
16. The goods-market check passes on a health-shock run with booked M_t.

Run the full suite on the GPU instance (15 float64 JAX tests are skipped on Metal).

## 5. Evaluator (`eval_fiscal_results.py`)
- Add `health` to the shock loop (:825).
- Generalise `chk_a0_predetermined` (:474-487) to assets through t_s, using the stamped `shock_period`.
- Health shock line: Δgov_health = (κ1μ1 − κ0)·M_base inside the window and 0 outside. Government health spending does not depend on household choices, so this holds exactly.
- `chk_debt_financed_neutrality` (:490) will warn on the health debt run, where households respond. Exempt it.

## 6. Runs

1. **Prerequisite:** the recalibration (user). Recompute the §1 baseline levels and the reference (κ1, μ1).
2. **Local smoke run** on the small economy: both exercises, all schemes, extraction and welfare. Confirm that every output is produced and the eval checks pass. Time the extraction. The non-config small economy has `current_year = 2020` (run_fiscal_figures.py:161), so pass the shock year explicitly.
3. **A100 run:**
   - `--backend jax --shock Ig,health --scenarios debt,tau_l_debt,tau_l_window`, exact aggregation, plus the two health decomposition runs.
   - Estimate before extraction: 18 min build and baseline. Per exercise: baseline ≈ 1.8 min, debt ≈ 1.8 min, a τ_l search ≈ 7 min at 4 evaluations, with the cache off.
     - I_g ≈ 11 min.
     - Health ≈ 18 min with the window variant, plus 3.6 min for the decomposition.
     - Total ≈ 50 min plus extraction (unmeasured).
     - The health τ_l searches have a negative root and may take more evaluations.
   - Run in the background with a monitor on NaN, Traceback, FAIL and a non-converging root search.
4. **Output** `output/policy_2026-10-XX/`:
   - `fiscal_results.json` with keys `Ig` and `health` and subkeys `distribution` and `welfare` per scenario;
   - figures, table bodies, `run.log` and `eval.log`.
5. `/eval-fiscal` on the JSON. `/fiscal-note` for the notes.

## 7. Remaining items
- **Calibration report:** a paragraph comparing the model's health shares with the data on each basis (§1).
- **Results:** add the results to the policy-experiments section of `reports/calibration_report.tex`, next to the design block already there.
