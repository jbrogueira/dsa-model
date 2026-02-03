# Model vs Implementation: Feature Tracker

This document tracks all features from the theoretical model (DSA-LSA paper) and their implementation status. Features default to OFF for backward compatibility.

**Status key:** `Done` = fully implemented and tested. `Partial` = solve/config done, simulation/aggregation remaining. `Skipped` = code is more general than paper. `TODO` = not yet started.

---

## Household Side

### 1. Labor supply — `Partial`

**Paper:** Endogenous labor hours `ℓ` with disutility of labor: `u(c,ℓ) = c^(1-σ)/(1-σ) - ν·ℓ^(1+φ)/(1+φ)`.
**Code:** FOC-based labor supply implemented in NumPy solve (`labor_supply=True`, `nu`, `phi` params). Config: `LifecycleConfig.labor_supply`, `nu`, `phi`.

**Remaining:**
- [ ] JAX solve (FOC-based)
- [ ] Simulation: record labor hours in both backends
- [ ] OLG aggregation: aggregate labor supply from simulation
- [ ] Tests (NumPy + JAX cross-validation)

### 2. Survival risk — `Partial`

**Paper:** Stochastic survival `π(j,s)` depending on age and health. Enters the Bellman as `β·π_j(s)·E[V_{j+1}]`. Deceased agents' assets are taxed and redistributed.
**Code:** Survival probs `(T, n_h)` affect backward induction in both NumPy and JAX solves. Config: `LifecycleConfig.survival_probs`. Agents still survive all T periods in simulation.

**Remaining:**

- [ ] **Simulation mortality draws** — Agents die stochastically during Monte Carlo forward simulation.
  - `_simulate_sequential()`: draw `u ~ Uniform(0,1)` per agent per period; if `u > π(t, i_h)`, agent dies; freeze dead agents' rows; track `alive (T, n_sim)` and `death_age (n_sim,)`.
  - `_agent_step_jax()` / `simulate_lifecycle_jax()`: third uniform draw alongside `u_y`, `u_h`; use `jnp.where(alive, new_state, frozen_state)`; carry `alive` flag in scan state.
  - Output: `alive_sim (T, n_sim)`, `bequest_sim (n_sim,)` (assets at death).

- [ ] **Accidental bequests** — Track total assets left by deceased agents each period.
  - Compute `bequests_t = sum of a[i]` for agents who died in period t.
  - In `OLGTransition`, aggregate bequests across cohorts and education types.
  - Redistribute as lump-sum transfers to living agents (add to `_compute_budget()` as income), or add to government revenue via bequest tax (Feature #16).

- [ ] **OLG aggregation with mortality weights** — Cross-sectional aggregation accounts for mortality-thinned cohorts.
  - `_period_cross_section()`: replace `weight = cohort_sizes[age] * education_shares[edu]` with `weight *= cumulative_survival(age)`.
  - Analytical approach: `cum_survival[j] = cum_survival[j-1] * mean(survival_probs[j-1, :])` — avoids MC noise in weights.
  - Impact: older cohorts get smaller weight → less K, less L from old agents → changes aggregates.

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

### 7. Endogenous retirement — `Partial`

**Paper:** Retirement is a choice within window `[J_R^min, J_R^max]`.
**Code:** Discrete choice between working and retired during retirement window. Config: `LifecycleConfig.retirement_window`.

**Remaining:**
- [ ] Tests (NumPy + JAX cross-validation)

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

### 16. Bequest taxation — `Partial`

**Paper:** `τ^beq` on assets of deceased, redistributed lump-sum.
**Code:** Config field `LifecycleConfig.tau_beq` exists (default 0.0) but has no effect — requires simulation mortality (Feature #2) to generate bequests.

**Remaining:**
- [ ] Wire up `tau_beq` in `compute_government_budget()` after simulation mortality is implemented
  - `bequest_tax_revenue = tau_beq * total_bequests_t`
  - Net bequests after tax: `(1 - tau_beq) * total_bequests_t` redistributed to living agents
  - Budget dict gains `"bequest_tax"` revenue field, `"bequest_transfers"` spending field

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

### 21. Population aging (fertility decline + mortality improvement) — `TODO`

**Paper:** Population aging driven by declining fertility (smaller entering cohorts) and improving longevity (higher survival at older ages). Both shift the cross-sectional age distribution toward older ages, affecting aggregate K, L, fiscal balance, and individual lifecycle decisions.

**Code:** `set_cohort_sizes_path_from_pop_growth()` provides reduced-form time-varying age weights via a single growth rate per period. `survival_probs (T, n_h)` is shared across all birth cohorts. Neither mechanism separates fertility from survival structurally.

**Formulation:**

Population at calendar time `t`, age `j`: `N(t, j) = fertility(t − j) · Π_{k=0}^{j-1} π̄(t−j+k, k)` where `fertility(s)` is the relative entering cohort size at birth time `s` and `π̄(year, age) = Σ_h weight(h) · π(year, age, h)` is health-averaged survival.

Cohort born at time `s` faces lifecycle survival schedule: `π_s(j, h) = π(s + j, j, h)` — evaluated at the calendar year when they reach age `j`.

**Remaining:**

- [ ] **`OLGTransition` parameters**: `fertility_path` (1D, relative entering cohort size by birth year), `survival_probs_initial` and `survival_probs_final` (each `(T, n_h)`, interpolated for transition cohorts), or `survival_improvement_rate` (scalar annual multiplicative improvement).
- [ ] **`_build_population_weights()`**: Compute `population_weights[t, age]` from fertility × cumulative survival. Supersedes `set_cohort_sizes_path_from_pop_growth()`. Populates `cohort_sizes_path` so existing `_cohort_weights(t)` dispatch works.
- [ ] **`_survival_schedule_at_year(cal_year)`**: Interpolate between initial/final survival schedules, or apply improvement rate to base schedule. Returns `(T, n_h)`.
- [ ] **`_cohort_survival_schedule(birth_period)`**: Build `(T, n_h)` survival array for a specific birth cohort by evaluating `survival_schedule_at_year(birth_period + j)` at each lifecycle age `j`.
- [ ] **`solve_cohort_problems()` per-cohort survival**: Pass cohort-specific `survival_probs` to each `LifecycleConfig` (currently all cohorts get the same default — this is a prerequisite bug-fix).
- [ ] **JAX batched solve**: Stack `survival_probs` as `(n_cohorts, T, n_h)` and vmap over cohort axis (currently uses shared `ref.survival_probs`).
- [ ] **Tests**: constant fertility + no survival improvement matches baseline; declining fertility raises dependency ratio and lowers L; improving survival raises K and pension spending.

**Dependencies:** Does not require simulation mortality (Feature #2 remaining work). Uses analytical cumulative survival for aggregation weights.

---

## Summary

| # | Feature | Status |
|---|---------|--------|
| 1 | Labor supply | Partial — NumPy solve done, JAX + simulation + aggregation remaining |
| 2 | Survival risk | Partial — solve done, simulation mortality + bequests + OLG weights remaining |
| 3 | Human capital | Done (merged into #5) |
| 4 | Schooling/children | Done |
| 5 | Income process conditioning | Done |
| 6 | Wage income structure | Done (part of #1) |
| 7 | Endogenous retirement | Partial — implementation done, tests remaining |
| 8 | Public capital | Done |
| 9 | SOE / sovereign debt | Done |
| 10 | Public investment | Done |
| 11 | Pension formula | Done |
| 12 | Tax application | Skipped (code more general) |
| 13 | Capital income tax | Skipped (code more general) |
| 14 | Progressive taxation | Done |
| 15 | Means-tested transfers | Done |
| 16 | Bequest taxation | Partial — config exists, needs simulation mortality |
| 17 | Govt spending | Done |
| 18 | Pension trust fund | Done |
| 19 | Defense spending | Done (simplified) |
| 20 | Medical age-dependence | Done |
| 21 | Population aging (fertility + survival) | TODO |

**Testing notes for remaining features:**
- `survival_probs = 1.0` must match deterministic baseline exactly
- With low survival, fewer agents alive at old ages → lower aggregate K from retirees
- `tau_beq = 0.0` must be a no-op
- JAX cross-validation: mortality-weighted aggregates must match NumPy within tolerance
- Constant fertility + equal initial/final survival → must match current baseline
- Declining fertility → higher old-age dependency ratio, lower L
- Improving survival → higher K, more pension spending
