# Differences Between Theoretical Model and Implementation

This document lists features present in the theoretical model (DSA-LSA paper) that are not yet implemented in the code, organized by category. The code implements a simplified version of the full model that captures the core lifecycle/OLG structure with education heterogeneity, income and health risk, and a basic fiscal sector.

---

## Household Side

### 1. Labor supply

**Paper:** Endogenous labor hours `ℓ` with disutility of labor: `u(c,ℓ) = c^(1-σ)/(1-σ) - ν·ℓ^(1+φ)/(1+φ)`.
**Code:** No labor supply choice. Utility is CRRA over consumption only: `u(c) = c^(1-γ)/(1-γ)`. Agents are either employed (ℓ=1 implicitly) or not.

### 2. Survival risk

**Paper:** Stochastic survival `π(j,s)` depending on age and health. Enters the Bellman as `β·π_j(s)·E[V_{j+1}]`. Deceased agents' assets are taxed and redistributed.
**Code:** Deterministic survival. All agents live exactly `T` periods with certainty. No mortality, no bequests.

### 3. Human capital

**Paper:** Continuous human capital state `h` with three components: (i) initial level `h_0 = A_0·e_s^γ` from education expenditure, (ii) deterministic age growth `g_j`, (iii) stochastic shocks `ε_j ~ N(0,σ_ε²)`. Evolution: `log h_{j+1} = log h_j + g_j + ε_j`. This is a continuous state variable in the Bellman.
**Code:** No human capital state. Income risk is a discrete AR(1) process discretized via Tauchen, with states `{0, z_1, ..., z_N}` where 0 is unemployment. Education heterogeneity enters only through different `mu_y`, `sigma_y`, `rho_y` parameters per type.

### 4. Schooling phase and children

**Paper:** Working households have children during the first `S` years of working life. Child consumption costs `c_y(j)` scale up the consumption bill: `(1+τ^c)(1+c_y(j))·c`. Education expenditures `e_s = (1-κ^school)·e(j)` with government education subsidies `κ^school_t(j)`.
**Code:** No schooling phase, no children, no education expenditures, no education subsidies. Agents enter as workers at age 0.

### 5. Income process conditioning

**Paper:** Productivity transition `P_z(z'|z,j,s)` depends on current productivity, age, and health state.
**Code:** Income transition matrix `P_y` is constant — independent of age and health.

### 6. Wage income structure

**Paper:** `y^L = w_t·h_j·f(s)·z·ℓ` — wage rate × human capital × health productivity × stochastic productivity × labor hours.
**Code:** `wage = w_t·y_grid[i_y]·h_grid[i_h]` — wage rate × income state × health multiplier. No human capital state, no labor hours.

### 7. Endogenous retirement

**Paper:** Optionally, retirement is endogenous within a window `[J_R^min, J_R^max]`.
**Code:** Fixed, exogenous `retirement_age`.

---

## Production Side

### 8. Public capital in production

**Paper:** `Y = Z_t·(K^g)^η_g·K^α·L^(1-α)` with public capital `K^g`, TFP shocks `Z_t`, and a CES generalization allowing imperfect substitution between public and private capital.
**Code:** `Y = A·K^α·L^(1-α)`. No public capital, no TFP shocks (constant `A`), no CES aggregation.

---

## Government and Fiscal Sector

### 9. Small open economy and sovereign debt

**Paper:** SOE with government sovereign bonds `B_t` held by external lenders at rate `r*`. Debt service `(1+r*)·B_t` and net borrowing `ΔB_t` appear in the government budget constraint. Long-term bond option with decay parameter `δ`.
**Code:** The household and firm sides are compatible with an SOE (prices can be taken as exogenous), but the equilibrium concept uses closed-economy capital market clearing (iterating on `r` until household savings match capital demand). No government debt, no external borrowing, no current account.

### 10. Public investment

**Paper:** Government invests `I^g_t` in public capital: `K^g_{t+1} = (1-δ_g)·K^g_t + I^g_t`. Public capital enters the production function and has its own depreciation rate.
**Code:** No public capital, no public investment, no `δ_g`.

### 11. Pension formula

**Paper:** `PENS_t = max{ρ·ȳ, b_min}` — replacement rate `ρ` applied to lifetime average earnings `ȳ` with a minimum pension floor `b_min`. Indexation rules (prices or wages) are a policy instrument.
**Code:** `pension = replacement_rate × w_at_retirement × y_grid[i_y_last]` — based on the last working-period income state, not lifetime average earnings. No minimum pension floor. `avg_earnings` is tracked in simulation but not used for pension computation.

### 12. Tax application to labor income

**Paper:** Working budget has `(1-τ^l-τ^p)·y^L·ℓ` — labor and payroll taxes are applied jointly to gross labor income as a combined deduction.
**Code:** Taxes are applied separately. Payroll tax on wages: `tax_p = τ_p·wage_income`. Labor tax on `(wages + UI - payroll_tax)`: `tax_l = τ_l·effective_income`. Computed sequentially, not as a joint deduction.

### 13. Capital income taxation

**Paper:** The working budget shows `(1+r^d_t)·a` on the RHS (gross asset return). No explicit capital income tax `τ^k` in the household budget constraint.
**Code:** Has an explicit capital income tax: `tax_k = τ_k·r·a`, yielding after-tax capital income `(1-τ_k)·r·a`.

### 14. Progressive taxation

**Paper:** Mentions optional progressive labor tax schedule: `τ^l(y) = 1 - κ·y^{-η}`.
**Code:** Only proportional (flat) tax rates.

### 15. Means-tested transfers

**Paper:** Has `T^W(·)` for workers and `T^R(·)` for retirees — means-tested transfer functions that depend on individual state.
**Code:** No means-tested transfers. The only transfers are UI benefits (for unemployed) and pensions (for retired).

### 16. Bequest taxation

**Paper:** Assets of deceased are taxed at rate `τ^beq` and redistributed lump-sum. `BeqTax_t = τ^beq·∫(1-π(j,s))·a·dμ_t` is a government revenue source.
**Code:** No mortality, no bequest tax, no bequest redistribution.

### 17. Government spending on goods

**Paper:** Has explicit `G_t` (non-transfer government purchases of goods and services) as a separate spending category in both the government budget constraint and the resource constraint.
**Code:** No explicit `G_t`. Government expenditure consists only of pensions, UI, and health subsidies.

### 18. Pension trust fund

**Paper:** Optional pension trust fund `S^pens_t` with accumulation: `S^pens_{t+1} = (1+r*)·S^pens_t + Rev^p_t - PENS^out_t`.
**Code:** No pension trust fund. Payroll tax revenue and pension spending are separate line items with no accumulation.

### 19. Defense and welfare-state production

**Paper:** Mentions defense expenditures, government labor costs for welfare state functions `N^h_t`, welfare state capital `K^h_t`, production of government services `H_t = F^W(K^h, N^h)`.
**Code:** None of these are present.

---

## Health Sector

### 20. Medical expenditure age-dependence

**Paper:** Medical expenditure needs `m^need(j,s)` depend on both age `j` and health `s`. Health coverage `κ^health_t(j,s)` can also vary by age, health, and time.
**Code:** Medical costs `m_grid[i_h]` depend only on health state, not on age. Coverage `kappa` is a single scalar constant.
