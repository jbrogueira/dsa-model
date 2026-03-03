# Model vs Implementation: Feature Tracker

This document tracks all features from the theoretical model (DSA-LSA paper) and their implementation status. Features default to OFF for backward compatibility.

**Status key:** `Done` = fully implemented and tested. `Partial` = solve/config done, simulation/aggregation remaining. `Skipped` = code is more general than paper. `TODO` = not yet started.

---

## Household Side

### 1. Labor supply — `Done`

**Paper:** Endogenous labor hours `ℓ` with disutility of labor: `u(c,ℓ) = c^(1-σ)/(1-σ) - ν·ℓ^(1+φ)/(1+φ)`.
**Code:** FOC-based labor supply implemented in both NumPy and JAX backends. Config: `LifecycleConfig.labor_supply`, `nu`, `phi`. Labor hours recorded in simulation (`l_policy` arrays); OLG aggregation uses simulated labor supply. NumPy and JAX cross-validation tests pass.

### 2. Survival risk — `Done`

**Paper:** Stochastic survival `π(j,s)` depending on age and health. Enters the Bellman as `β·π_j(s)·E[V_{j+1}]`. Deceased agents' assets are taxed and redistributed.
**Code:** Survival probs `(T, n_h)` affect backward induction in both NumPy and JAX solves. Config: `LifecycleConfig.survival_probs`. Simulation mortality draws are performed in `_simulate_sequential()` via Uniform draws against `π(t, i_h)`; dead agents' rows are frozen and tracked via `alive_sim (T, n_sim)` and `bequest_sim (n_sim,)` in the 21-tuple output. OLG aggregation uses mortality-weighted cohort sizes via `_build_population_weights()` and `_cohort_survival_schedule()` in `OLGTransition`.

### 3. Human capital — `Done` (merged into #5)

**Paper:** Continuous human capital state `h` with `log h_{j+1} = log h_j + g_j + ε_j`.
**Code:** Absorbed into age/health-dependent productivity transitions (Feature #5). No separate state variable — human capital dynamics captured via age-varying `P_y` persistence and mean growth. Config: `LifecycleConfig.P_y_by_age_health`.

### 4. Schooling phase and children — `Done`

**Paper:** Children during first `S` years, child consumption costs `c_y(j)`, education subsidies `κ^school`.
**Code:** Config: `LifecycleConfig.schooling_years`, `child_cost_profile`, `education_subsidy_rate`. Child costs scale the consumption bill during schooling years.

### 5. Income process conditioning — `Done`

**Paper:** Productivity transition `P_z(z'|z,j,s)` depends on current productivity, age, and health state.
**Code:** `P_y` can be `(n_y, n_y)` (constant) or `(T, n_h, n_y, n_y)` (age/health-dependent). Config: `LifecycleConfig.P_y_by_age_health`. Both NumPy and JAX backends support 4D transitions.

### 6. Wage income structure — `Done` (part of #1)

**Paper:** `y^L = w_t·h_j·f(s)·z·ℓ`.
**Code:** `wage = w · y_grid[i_y] · h_grid[i_h] · ℓ` when labor supply is enabled.

### 7. Endogenous retirement — `Done`

**Paper:** Retirement is a choice within window `[J_R^min, J_R^max]`.
**Code:** Discrete choice between working and retired during retirement window. Config: `LifecycleConfig.retirement_window`. NumPy and JAX cross-validation tests pass (`test_retirement_window_*`, `test_jax_retirement_window_*`).

---

## Production Side

### 8. Public capital in production — `Done`

**Paper:** `Y = Z_t·(K^g)^η_g·K^α·L^(1-α)`.
**Code:** Production function includes public capital. Config: `OLGTransition` params `eta_g`, `K_g_initial`, `delta_g`. `eta_g=0` recovers standard Cobb-Douglas.

---

## Government and Fiscal Sector

### 9. Small open economy and sovereign debt — `Done`

**Paper:** SOE with sovereign bonds `B_t` at rate `r*`, debt service, net borrowing.
**Code:** Config: `OLGTransition` params `economy_type='soe'`, `r_star`, `B_path`. Computes NFA, debt service, fiscal deficit.

### 10. Public investment — `Done`

**Paper:** `K^g_{t+1} = (1-δ_g)·K^g_t + I^g_t`.
**Code:** Public capital accumulation in `simulate_transition`. Config: `OLGTransition` param `I_g_path`.

### 11. Pension formula — `Done`

**Paper:** `PENS_t = max{ρ·ȳ, b_min}` with minimum pension floor.
**Code:** `pension = max(replacement_rate × w_ret × y_grid[i_y_last], pension_min_floor)`. Config: `LifecycleConfig.pension_min_floor`.

### 12. Tax application to labor income — `Skipped`

**Paper:** Joint deduction `(1-τ^l-τ^p)·y^L`.
**Code:** Sequential application (payroll first, then labor tax). More general than the paper's joint deduction.

### 13. Capital income taxation — `Skipped`

**Paper:** No explicit `τ^k`.
**Code:** Explicit `tax_k = τ_k·r·a`. Setting `tau_k=0` recovers the paper's formulation. More general.

### 14. Progressive taxation — `Done`

**Paper:** `τ^l(y) = 1 - κ·y^{-η}` (HSV functional form).
**Code:** Config: `LifecycleConfig.tax_progressive`, `tax_kappa`, `tax_eta`. When `tax_progressive=False`, uses flat rate.

### 15. Means-tested transfers — `Done`

**Paper:** `T^W(·)` for workers, `T^R(·)` for retirees.
**Code:** Huggett-style consumption floor: `T(a, y) = max(0, c_floor - resources)`. Config: `LifecycleConfig.transfer_floor`.

### 16. Bequest taxation — `Done`

**Paper:** `τ^beq` on assets of deceased, redistributed lump-sum.
**Code:** Config field `LifecycleConfig.tau_beq` (default 0.0). `tau_beq` is wired into `compute_government_budget()`: `bequest_tax_revenue = tau_beq * total_bequests`, `bequest_transfers = (1 - tau_beq) * total_bequests`; both fields appear in the budget dict. The bequest redistribution fixed-point loop (`recompute_bequests=True` in `simulate_transition()`) closes the feedback circuit automatically. Tested in `TestBequestLoop` in `test_fiscal_experiments.py`.

### 17. Government spending on goods — `Done`

**Paper:** Explicit `G_t` in government budget and resource constraints.
**Code:** Config: `OLGTransition` param `govt_spending_path`.

### 18. Pension trust fund — `Done`

**Paper:** `S^pens_{t+1} = (1+r*)·S^pens_t + Rev^p_t - PENS^out_t`.
**Code:** Trust fund accumulation in `compute_government_budget_path`. Config: `OLGTransition` param `S_pens_initial`.

### 19. Defense spending — `Done` (simplified)

**Paper:** Defense expenditures, government labor, welfare capital `K^h`, production `H = F^W(K^h, N^h)`.
**Code:** Simplified — `defense_spending_path` as exogenous budget category. Full government production function not implemented.

---

## Health Sector

### 20. Medical expenditure age-dependence — `Done`

**Paper:** `m^need(j,s)` depends on age and health. Coverage `κ^health_t(j,s)` varies by age/health/time.
**Code:** `m(j, h) = m_age_profile[j] * m_grid[h]`. Config: `LifecycleConfig.m_age_profile` (array of length T, default ones).

---

## Demographics

### 21. Population aging (fertility decline + mortality improvement) — `Done`

**Paper:** Population aging driven by declining fertility (smaller entering cohorts) and improving longevity (higher survival at older ages). Both shift the cross-sectional age distribution toward older ages, affecting aggregate K, L, fiscal balance, and individual lifecycle decisions.

**Code:** `OLGTransition` parameters `fertility_path` (1D, relative entering cohort size by birth year) and `survival_improvement_rate` (scalar annual multiplicative improvement) are implemented. `_survival_schedule_at_year(cal_year)` applies improvement rate to base schedule; `_cohort_survival_schedule(birth_period)` builds a `(T, n_h)` per-cohort survival array; `_build_population_weights()` computes `population_weights[t, age]` from fertility × cumulative survival and populates `cohort_sizes_path` so existing `_cohort_weights(t)` dispatch works. Per-cohort survival probs are passed to each `LifecycleConfig` in `solve_cohort_problems()`.

---

## Summary

| # | Feature | Status |
|---|---------|--------|
| 1 | Labor supply | Done |
| 2 | Survival risk | Done |
| 3 | Human capital | Done (merged into #5) |
| 4 | Schooling/children | Done |
| 5 | Income process conditioning | Done |
| 6 | Wage income structure | Done (part of #1) |
| 7 | Endogenous retirement | Done |
| 8 | Public capital | Done |
| 9 | SOE / sovereign debt | Done |
| 10 | Public investment | Done |
| 11 | Pension formula | Done |
| 12 | Tax application | Skipped (code more general) |
| 13 | Capital income tax | Skipped (code more general) |
| 14 | Progressive taxation | Done |
| 15 | Means-tested transfers | Done |
| 16 | Bequest taxation | Done |
| 17 | Govt spending | Done |
| 18 | Pension trust fund | Done |
| 19 | Defense spending | Done (simplified) |
| 20 | Medical age-dependence | Done |
| 21 | Population aging (fertility + survival) | Done |

