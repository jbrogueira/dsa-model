# OLG Model Calibration Strategy

## Model Overview

This is an **Overlapping Generations (OLG) model with heterogeneous agents** featuring:

1. **Lifecycle structure**: Agents live for `T` periods, with mandatory retirement at `retirement_age` or endogenous retirement within a window `retirement_window`
2. **Idiosyncratic risk**: Stochastic income (`n_y` states), health (`n_h` states), employment shocks, and stochastic mortality
3. **Education heterogeneity**: Multiple education types (`low`, `medium`, `high`) with different income profiles
4. **Perfect foresight over aggregates**: Agents know the full path of interest rates and wages
5. **Government sector**: Consumption/labor/payroll/capital taxes (flat or HSV progressive), unemployment insurance, pensions (with minimum floor), means-tested transfers, bequest taxation, health spending, public capital, defense spending
6. **Demographics**: Time-varying population growth and cohort-specific survival schedules allowing aging experiments
7. **Endogenous labor supply**: FOC-based hours choice when `labor_supply=True`; agents are employed, unemployed, or retired — with hours endogenous in the working state
8. **Age/health-dependent productivity**: Optional `P_y_by_age_health` of shape `(T, n_h, n_y, n_y)` replaces constant transition matrix
9. **Schooling phase and child costs**: Optional child consumption costs during first `schooling_years` periods
10. **Small open economy or closed economy**: `economy_type` selects whether `r` is exogenous (`soe`) or determined by market clearing (`closed`)

---

## Step 1: Externally Calibrated Parameters

These parameters are set directly from data or existing literature estimates.

| Parameter | Category | Source | Typical Value |
|-----------|----------|--------|---------------|
| **Preferences** | | | |
| `T` | Lifecycle length | Life expectancy − entry age | 40–60 |
| `retirement_age` | Mandatory retirement period index | Statutory/effective retirement | 30–45 |
| `retirement_window` | `(min_age, max_age)` for endogenous retirement | Statutory early/late retirement ages | e.g., `(35, 45)` |
| `gamma` | Risk aversion (CRRA) | Chetty (2006), Attanasio & Weber | 1–4 |
| `phi` | Inverse Frisch elasticity | Micro labor supply estimates (Chetty 2012) | 2.0 (Frisch = 0.5) |
| **Asset Grid** | | | |
| `a_min` | Borrowing constraint | Institutional (zero = no borrowing) | 0.0 |
| `a_max` | Maximum assets on grid | Large enough to not bind | 50–200 |
| `n_a` | Asset grid points | Numerical accuracy | 80–200 |
| **Production** | | | |
| `alpha` | Capital share | National accounts | 0.33 |
| `delta` | Depreciation rate | Investment/capital ratio | 0.05–0.10 |
| `r_default` | Steady-state interest rate | World real interest rate / country risk premium | 0.03–0.05 |
| `beta` | Discount factor | Set consistent with `r`: `beta ≈ 1/(1 + r·(1-tau_k))`; standard macro value | 0.95–0.98 |
| `A` | TFP level | Normalize to target wage level = 1.0, or set `A=1` | 1.0 |
| **Tax Rates** | | | |
| `tau_c` | Consumption tax | Effective rate from fiscal data | 0.05–0.25 |
| `tau_l` | Labor income tax | Effective rate from fiscal data | 0.10–0.30 |
| `tau_p` | Payroll tax (wages only) | Social security contribution | 0.10–0.25 |
| `tau_k` | Capital income tax | Effective rate from fiscal data | 0.15–0.30 |
| `tax_progressive` | Enable HSV progressive tax | Policy choice (True/False) | False (flat) or True |
| `tax_kappa` | HSV κ: progressivity scale | Heathcote-Storesletten-Violante (2017) | ~0.8 |
| `tax_eta` | HSV η: progressivity curvature | HSV (2017) | ~0.15 |
| `tau_beq` | Bequest (estate) tax rate | Statutory estate tax schedule | 0.0–0.40 |
| **Transfers** | | | |
| `pension_replacement_default` | Pension generosity | SS replacement rate data | 0.40–0.80 |
| `pension_min_floor` | Minimum pension floor | Statutory minimum pension level | Country-specific |
| `ui_replacement_rate` | UI wage replacement | Statutory UI rules | 0.20–0.60 |
| `transfer_floor` | Means-tested consumption floor | Program rules / poverty line | 0.0–0.10 |
| **Health Expenditure** | | | |
| `kappa` | Govt coverage of medical costs | MEPS / CMS data | 0.50–0.90 |
| `m_good` | Medical cost in good health (share of income) | MEPS | 0.02–0.10 |
| `m_moderate` | Medical cost in moderate health | MEPS | 0.30–0.60 |
| `m_poor` | Medical cost in poor health | MEPS | 0.70–1.00 |
| `m_age_profile` | Age multiplier on base medical costs | MEPS age profiles | Array of length T; all 1.0 = flat |
| `h_good` | Productivity in good health | Normalized | 1.0 |
| `h_moderate` | Productivity in moderate health | MEPS / HRS | 0.5–0.8 |
| `h_poor` | Productivity in poor health | MEPS / HRS | 0.1–0.4 |
| **Mortality** | | | |
| `survival_probs` | Age/health-dependent survival π(j, s) | Life tables (HMD, SSA actuarial) | Shape `(T, n_h)` |
| `survival_improvement_rate` | Annual longevity improvement rate | Demographic projections | 0.0–0.01 |
| **Education** | | | |
| `education_shares` | Population shares by education | Census / ACS | Country-specific |
| `edu_params[type].mu_y` | Mean log income by education | CPS / ACS | low: 0.03–0.08, med: 0.08–0.12, high: 0.10–0.15 |
| `edu_params[type].unemployment_rate` | Unemployment rate by education | CPS / BLS | low: 0.08–0.12, med: 0.04–0.08, high: 0.02–0.05 |
| **Demographics** | | | |
| `pop_growth` | Population growth rate | Demographic projections | -0.01–0.02 |
| `fertility_path` | Relative entering cohort sizes over time | Demographic projections | Array, length T + T_transition |
| `birth_year`, `current_year` | Cohort timing | Calendar alignment | Scenario-specific |
| **Schooling and Children** | | | |
| `schooling_years` | Periods with child costs | Demographic data (avg child-rearing years) | 0–15 |
| `child_cost_profile` | Per-period child cost (share of income) | Consumer expenditure surveys | Array of length T |
| `education_subsidy_rate` | Subsidy reducing net child costs | Program rules | 0.0–0.50 |

**Notes:**
- Payroll tax (`tau_p`) applies to wages only, not to pensions or UI benefits.
- Pensions are computed as `max(pension_replacement * w_at_retirement * y_grid[i_y_last], pension_min_floor)`, where `i_y_last` is the last working-period income state. This is not an average-earnings formula.
- Tax rates support time-varying paths (`tau_c_path`, etc.) for transition experiments.
- Medical costs now support age-dependent profiles: `m(j, h) = m_age_profile[j] * m_base[h]`. Set `m_age_profile = None` (default) for flat costs.
- `beta` is not calibrated via moment matching. Instead it is set to be consistent with the exogenous interest rate: a standard choice is `beta = 1/(1 + r*(1-tau_k))`, which ensures that an agent with median wealth and no idiosyncratic shocks would be approximately indifferent between saving and consuming in steady state. Alternatively, `beta = 0.96` (implying a 4% annual subjective discount rate) is a widely-used benchmark. When `survival_probs < 1`, the effective discount factor is `beta * pi(j,s)`, which is already lower than `beta`; to preserve a consistent implied discount rate across specifications, slightly raise `beta` (e.g., by `1/pi_bar` where `pi_bar` is the average working-age survival probability).
- `survival_probs` are taken from life tables, not calibrated.
- `phi` (inverse Frisch elasticity) is set externally from micro estimates. Chetty (2012) meta-analysis puts the micro Frisch elasticity at ~0.5, implying `phi ≈ 2`. `nu` (labor disutility weight) is calibrated internally to match average hours.
- For the HSV progressive tax, set `tax_progressive=True` and calibrate `tax_kappa`/`tax_eta` to country-specific estimates. The tax function is `T(y) = y − κ · y^{1−η}`, implying average net-of-tax rate `κ · y^{-η}`.

---

## Step 2: Internally Calibrated Parameters

These parameters are calibrated via moment matching (Simulated Method of Moments).

| Parameter | Role | Identified By | Notes |
|-----------|------|---------------|-------|
| `edu_params[type].rho_y` | Income persistence (per education) | Consumption-income comovement | Can calibrate jointly (same for all types) or separately |
| `edu_params[type].sigma_y` | Income shock variance (per education) | Wealth Gini, fraction with zero wealth | Can calibrate jointly (same for all types) or separately |
| `job_finding_rate` | Probability of exiting unemployment | Unemployment duration | Single parameter, shared across education types |
| `max_job_separation_rate` | Cap on separation rate | Unemployment rate by age | Actual separation rate is derived: `min(u/(1-u) * jfr, max_sep_rate)` |
| `P_h_young`, `P_h_middle`, `P_h_old` | Health transition matrices by age group | Health status distribution by age | 3 matrices × `n_h²` entries; highly parametric; reduce by imposing structure |
| `nu` | Labor disutility weight | Average annual hours worked | Only relevant when `labor_supply=True`; scales hours to match data level |
| `initial_asset_distribution` | Initial wealth distribution at entry | Wealth distribution at youngest cohort | Array of samples drawn at simulation start; use when entry-age wealth dispersion is non-trivial |

**Notes:**
- `rho_y` and `sigma_y` are defined per education type in `edu_params`. Joint calibration (same for all types) reduces dimensionality; type-specific calibration requires sufficient identifying variation in education-group wealth distributions.
- `job_separation_rate` is not a free parameter — it's computed from `unemployment_rate` and `job_finding_rate`. The free parameters are `job_finding_rate`, `max_job_separation_rate`, and `edu_params[type].unemployment_rate`.
- `initial_asset_distribution` is now implemented — pass an array of samples to `LifecycleConfig`. When the entry-age wealth distribution is well-measured (e.g., from SCF for 25-year-olds), this is preferable to a scalar `initial_assets`.
- When `P_y_by_age_health` is used instead of the Tauchen-based `rho_y`/`sigma_y`, the transition matrices themselves are constructed from data (e.g., PSID earnings transitions by age group and health status). In this case, `rho_y`/`sigma_y` are no longer free calibration parameters for those dimensions.

---

## Step 3: Target Moments

These are the empirical moments used to identify the internally calibrated parameters.

| Moment | Data Source | Identifies |
|--------|-------------|------------|
| **Macro Aggregates** | | |
| Capital-output ratio K/Y | National accounts | Validation |
| Labor share of income | National accounts | `alpha` (validation) |
| **Wealth Distribution** | | |
| Wealth Gini coefficient | SCF | `sigma_y`, `a_min` |
| Fraction with zero/negative wealth | SCF | `a_min`, `sigma_y` |
| Median wealth-to-income ratio | SCF | `beta` |
| Wealth-to-income ratio by age | SCF / PSID | `beta`, income process |
| Wealth-to-income ratio by education | SCF | Education-specific `mu_y` (validation) |
| Wealth distribution at entry age (~25) | SCF | `initial_asset_distribution` |
| **Income Dynamics** | | |
| Consumption-income comovement | CEX / PSID | `rho_y` |
| Earnings variance by age | PSID | `sigma_y` |
| **Labor Market** | | |
| Unemployment rate (aggregate) | CPS / BLS | `unemployment_rate`, `job_finding_rate` |
| Average unemployment duration | CPS | `job_finding_rate` |
| Unemployment rate by education | CPS | `edu_params[type].unemployment_rate` |
| Average annual hours worked | CPS / ATUS | `nu` (when `labor_supply=True`) |
| Hours profile by age | CPS / ATUS | `nu`, `phi` (validation) |
| **Health** | | |
| Fraction in poor health by age | MEPS / HRS | Health transition matrices |
| Medical spending by age | MEPS | `m_good`, `m_moderate`, `m_poor`, `m_age_profile`, `kappa` (validation) |
| **Fiscal** | | |
| Government spending / GDP | NIPA | Pension/health generosity (validation) |
| Tax revenue / GDP | NIPA | Tax rates (validation) |
| Pension spending / GDP | NIPA | `pension_replacement_default`, `pension_min_floor` (validation) |

**Notes:**
- "Unemployment rate by age" is not a tight target — the model's separation rate varies only by education type, not age. This moment validates aggregate fit but cannot identify age-specific parameters without model extension.
- The borrowing constraint `a_min` strongly affects wealth distribution moments. If `a_min = 0` is imposed externally, then `sigma_y` bears more of the identification burden for wealth dispersion.
- Hours moments (`nu`) are only relevant when `labor_supply=True`. If labor supply is exogenous (default), skip these.
- When progressive taxation is active (`tax_progressive=True`), tax revenue by income decile can be used to validate `tax_kappa`/`tax_eta`.

---

## Step 4: Calibration Algorithm

### 4.1 Calibration Structure

With `r`, `beta`, and `A` set externally (Step 1), calibration is a single partial-equilibrium stage with fixed prices.

**Calibration stage — Partial equilibrium (fixed prices):**
Fix `r` and `w` at empirically reasonable values. Calibrate parameters that primarily affect individual behavior:

| Parameter | Target |
|-----------|--------|
| `rho_y`, `sigma_y` | Earnings variance by age, consumption-income comovement |
| `job_finding_rate` | Average unemployment duration |
| `edu_params[type].unemployment_rate` | Unemployment rate by education |
| `P_h` matrices | Fraction in poor health by age |
| `nu` | Average annual hours worked (when `labor_supply=True`) |

### 4.2 SMM Objective Function

```
Q(theta) = [m_data - m_model(theta)]' W [m_data - m_model(theta)]
```

Where:
- `m_data`: vector of empirical target moments
- `m_model(theta)`: simulated moments from the model at parameter vector `theta`
- `W`: weighting matrix

**Weighting matrix options:**
- **Identity matrix** (equal weights): simple, robust, good for initial exploration
- **Diagonal of inverse variances**: weights moments by precision of data estimates
- **Optimal (two-step)**: use first-stage identity estimates to compute optimal `W` from simulated moment covariance

Recommendation: start with diagonal weighting, move to two-step optimal only if needed.

### 4.3 Optimizer

Use **derivative-free methods** — the objective is noisy (simulation-based) and non-smooth (grid search in the inner loop):

1. **Nelder-Mead simplex** (`scipy.optimize.minimize(method='Nelder-Mead')`) for the calibration stage

The JAX backend (`backend='jax'`) significantly accelerates the solve (cohort solves are vectorized), making Nelder-Mead feasible even with `n_sim >= 10,000`.

### 4.4 Simulation Requirements

- Use `n_sim >= 10,000` agents per education type to reduce simulation noise
- Fix the random seed across objective function evaluations for smooth optimization landscape
- Discard the first few periods of simulation as burn-in if agents start from non-ergodic initial conditions
- When `survival_probs` are active, cohort sizes shrink over the lifecycle — the aggregation in `OLGTransition` already applies mortality-weighted cohort sizes; no additional adjustment needed in moment computation

### 4.5 Convergence Criteria

- Calibration stage: `Q(theta) < tol` or parameter changes `< 1e-5` across iterations
- Full convergence: all target moments within 5% of data values, or within 2 standard errors of data estimates

### 4.6 Practical Workflow

1. Set externally calibrated parameters (Step 1) from data
2. If using `survival_probs` from life tables, load these before calibration (they affect value function computation but not the PE moments targeted in the calibration stage)
3. Run the calibration stage with fixed prices; include `nu` if `labor_supply=True`
4. Set `r_default` and `beta` from data/literature (Step 1). Verify that the implied `w` from the production function is consistent with observed wages; adjust `A` if needed.
5. Run the single-stage calibration targeting income dynamics, unemployment, health distributions, and (if applicable) hours.
6. Validate: check that the simulated K/Y, tax revenue/GDP, and pension spending/GDP are within plausible ranges as untargeted predictions.
7. Sensitivity analysis: vary `gamma`, `a_min`, `phi`, `beta`, `r` to check robustness.
