# Implementation Plan: Model Features

Covers 18 of the 20 features listed in `model_vs_implementation.md` — the gaps between the DSA-LSA paper's theoretical model and the current code. Two features (#12 tax application, #13 capital income tax) are skipped because the code's current implementation is more general than the paper's. Features are organized into 7 phases ordered by dependency, complexity, and risk.

**Key architectural constraints:**
- State space is currently `(T, n_a, n_y, n_h, n_y_last)` — no phase adds new state variables
- Both NumPy (`lifecycle_perfect_foresight.py`) and JAX (`lifecycle_jax.py`) backends must be updated in lockstep
- `OLGTransition` aggregation logic must be updated for any new output fields
- Tests must be added/updated for each feature; JAX cross-validation tests must continue to pass

---

## Progress

| Phase | Features | Status |
|-------|----------|--------|
| 1 | #11, #17, #20 | **Done** |
| 2 | #3+#5, #2, #16 | **Done** |
| 3 | #14, #15 | **Done** |
| 4 | #1, #6, #7 | **Done** |
| 5 | #4 | **Done** |
| 6 | #8, #10, #9 | **Done** |
| 7 | #18, #19 | **Done** |

**Skipped:** #12 (tax application — keep code version), #13 (capital income tax — keep code version)
**Merged:** #3 (human capital) absorbed into #5 (age/health-dependent productivity) — no new state variable

---

## Phase 1: Budget Constraint & Fiscal Fixes

Low-risk, independent changes to the budget constraint and fiscal accounting. No new state variables. Each can be implemented and tested independently.

### Feature #11 — Minimum pension floor

- [x] Implement
- [x] Test (NumPy)
- [x] Test (JAX cross-validation)

**Paper:** `PENS = max{ρ·ȳ, b_min}` with minimum pension floor `b_min`.
**Current:** `pension = replacement_rate × w_at_retirement × y_grid[i_y_last]`, no floor.

**Decision:** Keep the current `i_y_last`-based pension formula. Only add a minimum pension floor.

**Changes:**
- `LifecycleConfig`: Add `pension_min_floor` (float, default 0.0).
- Modify `_compute_budget()`: pension becomes `max(replacement_rate * w_ret * y_grid[i_y_last], pension_min_floor)`.
- Apply the same floor in both solve and simulation.
- Files: `lifecycle_perfect_foresight.py` (_compute_budget), `lifecycle_jax.py` (equivalent), `olg_transition.py` (pass-through)

### ~~Feature #12 — Tax application~~ SKIPPED

Keep code's current sequential tax application (payroll tax first, then labor tax). More general than the paper's joint deduction.

### ~~Feature #13 — Capital income taxation~~ SKIPPED

Keep code's current explicit `τ_k` capital income tax. More general than the paper (which has no explicit `τ_k`). Setting `tau_k = 0` recovers the paper's formulation.

### Feature #17 — Government spending on goods (G_t)

- [x] Implement
- [x] Test

**Paper:** Explicit `G_t` in government budget constraint and resource constraint.
**Current:** No `G_t`.

**Changes:**
- `LifecycleConfig`: Add `govt_spending_path` (array of length T or scalar, default 0.0).
- `OLGTransition`: Add `G_t` to government budget accounting. In the resource constraint: `Y = C + I + G`. This affects the GE price-finding loop — government spending absorbs resources.
- This is primarily an `olg_transition.py` change. Individual lifecycle solve is unaffected (agents don't choose G).
- Files: `olg_transition.py`

### Feature #20 — Medical expenditure age-dependence

- [x] Implement
- [x] Test (NumPy)
- [x] Test (JAX cross-validation)

**Paper:** `m^need(j, s)` depends on age `j` and health `s`.
**Current:** `m_grid[i_h]` depends on health only.

**Changes:**
- `LifecycleConfig`: Replace `m_good`, `m_moderate`, `m_poor` (scalars) with `m_grid_by_age` (array of shape `(T, n_h)`). Keep scalar params as convenience that broadcast to all ages.
- Add age-profile parameters: `m_age_profile` (array of length T, default all ones) that multiplies the base medical costs. So `m(j, h) = m_age_profile[j] * m_grid[h]`.
- Modify `_compute_budget()` to accept age-indexed medical costs.
- Modify `_solve_period()` to pass age-specific medical costs.
- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`

---

## Phase 2: Unified Productivity Process (Features #3 + #5)

### Features #3 & #5 — Age/health-dependent productivity transitions (replaces separate human capital state)

- [x] Implement config parameters
- [x] Implement `_income_process()` age/health variant
- [x] Implement solve changes (NumPy)
- [x] Implement solve changes (JAX einsum)
- [x] Implement simulation changes (both backends)
- [x] Test (NumPy)
- [x] Test (JAX cross-validation)

**Paper (Feature #5):** `P_z(z'|z, j, s)` depends on age and health.
**Paper (Feature #3):** Continuous human capital state `h` with `log h_{j+1} = log h_j + g_j + ε_j`.
**Current:** `P_y` is `(n_y, n_y)` — constant across ages and health states.

**Approach:** Instead of adding a separate human capital state variable (which would add a grid dimension), combine human capital dynamics and stochastic productivity into a single "labor market productivity" state with age-dependent and health-dependent transition matrices. The `y_grid` already represents productivity states; by making `P_y` vary by age and health, we capture:
- **Human capital accumulation:** Age-dependent mean/persistence (young workers have higher expected growth)
- **Health effects on productivity dynamics:** Health-dependent transitions (poor health → higher probability of downward productivity transitions)

**No new state variable** — state space stays `(T, n_a, n_y, n_h, n_y_last)`.

**Changes:**

1. **`LifecycleConfig`:** Add per-education-type age-group parameters:
   - `rho_y_by_age`: dict mapping age groups to persistence values (e.g., `{'young': 0.95, 'middle': 0.97, 'old': 0.99}`)
   - `sigma_y_by_age`: dict mapping age groups to shock variances
   - `mu_y_by_age`: dict mapping age groups to mean growth (captures human capital accumulation)
   - `health_productivity_adjustment`: array `(n_h,)` of multiplicative adjustments to transition probabilities by health state (e.g., poor health shifts probability mass toward lower states)
   - OR: direct specification via `P_y_by_age_health` of shape `(T, n_h, n_y, n_y)` for full flexibility
   - Default: `None` (fall back to constant `P_y` for backward compatibility)

2. **`_income_process()` in `lifecycle_perfect_foresight.py`:**
   - When age/health-dependent params provided: construct `P_y` of shape `(T, n_h, n_y, n_y)` using age-group Tauchen discretizations with health adjustments
   - Use 3 age groups (young < 40, middle 40-60, old 60+) matching existing health transition pattern
   - The `y_grid` stays the same (shared grid for all ages/health states) — only transition probabilities change
   - Store as `self.P_y` with shape `(T, n_h, n_y, n_y)` when age-dependent, `(n_y, n_y)` when constant

3. **`_solve_period()` in `lifecycle_perfect_foresight.py` (line ~583):**
   - Change: `self.P_y[i_y, i_y_next]` → `self.P_y[t, i_h, i_y, i_y_next]` (when 4D)
   - Already inside `i_h` loop, `t` is the period parameter — trivial indexing change

4. **`solve_period_jax()` in `lifecycle_jax.py` (line ~189):**
   - Change einsum from `'ij, iabj->iab'` (P_y is `(n_y, n_y)`) to `'bij, iabj->iab'` (P_y_t is `(n_h, n_y, n_y)`)
   - `P_y_t` is the per-period slice, passed alongside `P_h_t` in the scan loop
   - The `b` (health) dimension in P_y_t aligns with the `b` (health) dimension in EV_h_t — correct contraction

5. **`_simulate_sequential()` (line ~811) and `_agent_step_jax()` (line ~468):**
   - Change: `P_y[i_y, :]` → `P_y[t, i_h, i_y, :]`
   - Both `t` (lifecycle_age) and `i_h` are known at the sampling point

6. **`_simulate_sequential()` initial distribution (line ~706):**
   - Compute stationary distribution from `P_y[0, i_h_initial, :, :]` (age 0, initial health state) instead of constant `P_y`

7. **OLG batched calls in `olg_transition.py`:**
   - `P_y` shape change flows through — still shared across cohorts within education type, just larger array

- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`, `olg_transition.py`

### Feature #2 — Survival risk (stochastic mortality)

- [x] Implement config + solve
- [ ] Implement simulation (mortality draws, bequest tracking)
- [ ] Implement OLG aggregation (mortality-weighted cohort sizes)
- [x] Test (NumPy)
- [x] Test (JAX cross-validation)

**Paper:** `β · π(j, s) · E[V_{j+1}]` with age/health-dependent survival probabilities.
**Current:** Deterministic survival (all agents live T periods).

**Changes:**
- `LifecycleConfig`: Add `survival_probs` — array of shape `(T, n_h)` with `π(j, s) ∈ [0, 1]`. Default: all 1.0 (deterministic survival, backwards compatible).
- Modify backward induction: multiply continuation value by `π(j, s)`. In terminal period, unchanged.
- Modify simulation: at each period, draw survival shock. Dead agents exit the simulation. Track accidental bequests (assets of deceased).
- Modify aggregation in `OLGTransition`: account for age-varying cohort sizes due to mortality. Weight agents by survival probability in cross-sectional aggregation.
- **Important:** This changes the effective discount factor from `β` to `β · π(j, s)`, which will affect calibrated `β`.
- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`, `olg_transition.py`

### Feature #16 — Bequest taxation

- [x] Implement (config param `tau_beq` added)
- [ ] Test (requires bequest aggregation in OLG, deferred to when simulation mortality is added)

**Depends on:** Feature #2 (survival risk).

**Paper:** `τ^beq` tax on assets of deceased, redistributed lump-sum.
**Current:** No mortality, no bequests.

**Changes:**
- `LifecycleConfig`: Add `tau_beq` (float, default 0.0).
- `OLGTransition`: After simulation, compute total accidental bequests, apply `τ^beq`, redistribute as lump-sum transfer to living agents. Add to government revenue accounting.
- Files: `olg_transition.py` primarily

---

## Phase 3: Tax & Transfer System

### Feature #14 — Progressive taxation

- [x] Implement
- [x] Test (NumPy)
- [x] Test (JAX cross-validation)

**Paper:** `τ^l(y) = 1 - κ · y^{-η}` (HSV functional form).

**Changes:**
- `LifecycleConfig`: Add `tax_progressive` (bool, default False), `tax_kappa` (float), `tax_eta` (float).
- When `tax_progressive=True`, replace flat `tau_l * income` with `income - kappa * income^(1-eta)` (HSV schedule). When False, use existing flat rate.
- Modify `_compute_budget()` to use progressive schedule.
- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`

### Feature #15 — Means-tested transfers

- [x] Implement
- [x] Test (NumPy)
- [x] Test (JAX cross-validation)

**Paper:** `T^W(·)` for workers, `T^R(·)` for retirees — functions of individual state.

**Changes:**
- `LifecycleConfig`: Add `transfer_floor` (consumption floor, float, default 0.0). This is the standard Huggett-style consumption floor: `T(a, y) = max(0, c_floor - resources)`.
- Modify `_compute_budget()`: after computing disposable income, apply floor transfer.
- This is a simplified but widely-used formulation. Can be extended later to more complex means-testing.
- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`

---

## Phase 4: Labor Supply

This is the largest single feature. It changes the utility function, the solve algorithm, and the simulation.

### Feature #1 — Endogenous labor hours (FOC approach)

- [x] Implement config parameters
- [x] Implement FOC-based solve (NumPy)
- [ ] Implement FOC-based solve (JAX)
- [ ] Implement simulation (record labor hours)
- [ ] Implement OLG aggregation (aggregate labor supply)
- [ ] Test (NumPy)
- [ ] Test (JAX cross-validation)

**Paper:** `u(c, ℓ) = c^(1-σ)/(1-σ) - ν · ℓ^(1+φ)/(1+φ)`, labor hours `ℓ ≥ 0`.

**Changes:**
- `LifecycleConfig`: Add `labor_supply` (bool, default False), `nu` (labor disutility weight), `phi` (Frisch elasticity parameter = 1/φ).
- **Approach — FOC-based:** Given separable utility, the FOC for labor is `ν · ℓ^φ = λ · w · h · z · (1-τ_l-τ_p)`, where `λ = c^{-σ}`. For each candidate `(c, a')`, solve for optimal `ℓ` analytically: `ℓ* = max(0, (c^{-σ} · w·h·z·(1-τ_l-τ_p) / ν)^{1/φ})`. Only non-negativity constraint (no upper bound). This avoids adding a grid dimension.
- Modify `_solve_period()`: for each `(a, y, h)` state and each candidate `a'`, compute optimal `ℓ*` from FOC, then compute `c` from budget constraint with `ℓ*`, evaluate utility `u(c, ℓ*)`.
- Modify simulation to record labor hours.
- **State space impact:** No new state dimension if using FOC approach. Policy arrays gain a labor dimension in output: `l_policy(T, n_a, n_y, n_h, n_y_last)`.
- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`, `olg_transition.py` (aggregate labor supply)

### Feature #6 — Wage income structure

- [x] Implement (part of Feature #1)

**Depends on:** Feature #1 (labor supply).

**Paper:** `y^L = w · h_j · f(s) · z · ℓ`.
**Current:** `wage = w · y_grid[i_y] · h_grid[i_h]`.

**Changes:**
- Once labor supply is added, wage income naturally becomes `w · y_grid[i_y] · h_grid[i_h] · ℓ`.
- This is essentially already handled by the labor supply implementation.
- Files: same as Feature #1

### Feature #7 — Endogenous retirement

- [x] Implement
- [ ] Test (NumPy)
- [ ] Test (JAX cross-validation)

**Paper:** Retirement is a choice within window `[J_R^min, J_R^max]`.

**Changes:**
- `LifecycleConfig`: Add `retirement_window` tuple `(min_age, max_age)`, default `None` (fixed retirement).
- During the retirement window, agents choose whether to retire. This is a discrete choice: `V(a, y, h) = max(V_work(a, y, h), V_retire(a, y_last, h))`.
- In backward induction, for ages in the retirement window, solve both the working and retired value functions and take the max.
- **State space impact:** No new state variable, but the solve is more complex in the retirement window.
- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`

---

## Phase 5: Schooling & Children

### Feature #4 — Schooling phase and children

- [x] Implement
- [x] Test (NumPy)
- [x] Test (JAX cross-validation)

**Paper:** Children during first S working years, child consumption costs `c_y(j)`, education expenditures with subsidies `κ^school`.

**Changes:**
- `LifecycleConfig`: Add `schooling_years` (int, default 0), `child_cost_profile` (array of length T, default zeros), `education_subsidy_rate` (float, default 0.0).
- During schooling years (first S periods), budget constraint includes child costs that scale consumption: `(1 + τ_c)(1 + child_cost(j)) · c + a' = ...`.
- Education subsidies reduce the effective cost: net child cost = `(1 - education_subsidy_rate) * child_cost(j)`.
- This is a budget constraint modification — no new state variables.
- Files: `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`

**Note:** Feature #3 (human capital) is absorbed into Phase 2 via age/health-dependent productivity transitions. The initial human capital level `h_0 = A_0 · e_s^γ` from education expenditure can be captured by education-type-specific initial productivity distributions (already supported via `edu_params`).

---

## Phase 6: Production & Public Sector

### Feature #8 — Public capital in production

- [x] Implement
- [x] Test

**Paper:** `Y = Z · (K^g)^{η_g} · K^α · L^{1-α}`.

**Changes:**
- `OLGTransition`: Add `K_g` (public capital stock), `eta_g` (public capital elasticity), `delta_g` (public capital depreciation).
- Modify production function: `Y = A * K_g**eta_g * K**alpha * L**(1-alpha)`.
- Modify firm FOCs for `r` and `w` to account for public capital.
- Files: `olg_transition.py`

### Feature #10 — Public investment

- [x] Implement
- [x] Test

**Depends on:** Feature #8.

**Paper:** `K^g_{t+1} = (1 - δ_g) · K^g_t + I^g_t`.

**Changes:**
- `OLGTransition`: Add `I_g_path` (public investment path) or `I_g_share` (share of GDP).
- Add public capital accumulation equation to the transition dynamics.
- Add `I_g` to government spending in the budget constraint.
- Files: `olg_transition.py`

### Feature #9 — Small open economy and sovereign debt

- [x] Implement
- [x] Test

**Paper:** SOE with `B_t` sovereign bonds, external rate `r*`.

**Changes:**
- `OLGTransition`: Add `economy_type` ('closed' | 'soe'), `r_star` (world interest rate), `B_path` (sovereign debt path).
- In SOE mode: `r = r*` (exogenous), no capital market clearing. Instead, current account adjusts.
- Government budget: `G + transfers + (1+r*)·B_t = revenue + B_{t+1}`.
- Resource constraint: `Y + r*·NFA = C + I + G + NX`.
- This is a major change to the equilibrium concept in `OLGTransition` but does NOT affect the lifecycle solve (agents just face given prices).
- Files: `olg_transition.py`

---

## Phase 7: Optional / Low Priority

### Feature #18 — Pension trust fund

- [x] Implement
- [x] Test

**Paper:** `S^pens_{t+1} = (1+r*)·S^pens_t + Rev^p_t - PENS^out_t`.

**Changes:**
- `OLGTransition`: Add `S_pens` (trust fund balance), accumulation equation.
- Mostly accounting — feeds into government budget constraint.
- Files: `olg_transition.py`

### Feature #19 — Defense and welfare-state production

- [x] Implement (simplified: defense spending as budget category)
- [x] Test

**Paper:** Defense spending, government labor, welfare capital `K^h`, production `H = F^W(K^h, N^h)`.

**Implementation:** Simplified version — `defense_spending_path` as exogenous spending in the government budget. The full government production function (K^h, N^h) can be extended later if needed.

**Changes:**
- `OLGTransition`: Add `defense_spending_path` (array, default None).
- Add defense spending to total government spending in budget constraint.
- Files: `olg_transition.py`

---

## Implementation Order Summary

| Phase | Features | Key Changes | Risk |
|-------|----------|-------------|------|
| 1 | #11, #17, #20 | Pension floor, govt spending, age-dependent medical | Low |
| 2 | #3+#5, #2, #16 | Age/health-dependent productivity (unified), survival risk, bequest tax | Medium |
| 3 | #14, #15 | Progressive taxation, means-tested transfers | Low-Medium |
| 4 | #1, #6, #7 | Labor supply (FOC-based), wage structure, endo retirement | High |
| 5 | #4 | Schooling phase and children | Low-Medium |
| 6 | #8, #10, #9 | Public capital, public investment, SOE/sovereign debt | Medium-High |
| 7 | #18, #19 | Pension trust fund, govt production | Low priority |

---

## Testing Strategy

Each phase follows the same testing protocol:

1. **Backward compatibility:** All new features default to OFF (e.g., `pension_min_floor=0.0`, `survival_probs=None`, `labor_supply=False`). Existing tests must pass unchanged after each phase.

2. **Feature unit tests:** Each feature gets dedicated tests in `test_olg_transition.py`:
   - Verify the feature activates correctly when enabled
   - Check boundary cases (e.g., pension floor = 0 should match no-floor, survival_probs = 1.0 should match deterministic)
   - Validate against analytical solutions where available (e.g., FOC-based labor supply has closed-form)

3. **JAX cross-validation:** After each phase that touches the lifecycle solve or simulation:
   - Value function agreement: `|V_jax - V_numpy| < 1e-6` for all states
   - Policy agreement: identical optimal indices
   - Simulation distributional match: mean profiles within 3 standard errors (n_sim >= 5000)

4. **Regression testing:** Run the full OLG transition (`python olg_transition.py --test`) before and after each phase. Compare:
   - Aggregate K, L, Y levels
   - Government budget balance
   - Consumption and wealth profiles by age

5. **Feature-specific validation:** Compare model moments against known benchmarks:
   - Phase 2: Health-conditional income distributions, mortality-adjusted wealth profiles
   - Phase 3: Tax revenue under progressive schedule, transfer spending
   - Phase 4: Labor supply elasticity, hours profile by age
   - Phase 6: K/Y ratio with public capital, SOE current account

## JAX Backend Strategy

- **Phase 1-3:** Straightforward — same state space, just different computations. Update both backends in lockstep.
- **Phase 2 (unified productivity):** `P_y` shape changes from `(n_y, n_y)` to `(T, n_h, n_y, n_y)`. JAX einsum change is minimal (`'ij,...' → 'bij,...'`). P_y slices passed per-period in the `lax.scan` loop, same pattern as P_h.
- **Phase 4 (labor supply):** FOC-based approach avoids new grid dimension, keeping JAX compatibility simple. The FOC computation is element-wise and fully vectorizable.
- **Phase 5 (schooling):** Budget constraint change only — no state space impact.
- **Phase 6:** Changes are in `olg_transition.py` (NumPy/Numba), not in the lifecycle backends.

## Key Files

| File | Role | Phases Affected |
|------|------|----------------|
| `lifecycle_perfect_foresight.py` | NumPy lifecycle solve + simulate | 1, 2, 3, 4, 5 |
| `lifecycle_jax.py` | JAX lifecycle solve + simulate | 1, 2, 3, 4, 5 |
| `olg_transition.py` | OLG aggregation, GE, govt budget | 1, 2, 3, 4, 5, 6, 7 |
| `test_olg_transition.py` | Test suite | All phases |
