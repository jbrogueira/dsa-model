# OLG Model Calibration Strategy

## Model Overview

This is an **Overlapping Generations (OLG) model with heterogeneous agents** featuring:

1. **Lifecycle structure**: Agents live for `T` periods, with exogenous retirement at `retirement_age`
2. **Idiosyncratic risk**: Stochastic income (`n_y` states), health (`n_h` states), and employment shocks
3. **Education heterogeneity**: Multiple education types (`low`, `medium`, `high`) with different income profiles
4. **Perfect foresight over aggregates**: Agents know the full path of interest rates and wages
5. **Government sector**: Consumption/labor/payroll/capital taxes, unemployment insurance, pensions, and health spending
6. **Demographics**: Time-varying population growth allowing for aging experiments
7. **No intensive-margin labor supply**: Agents are either employed, unemployed, or retired — no hours choice

---

## Step 1: Externally Calibrated Parameters

These parameters are set directly from data or existing literature estimates.

| Parameter | Category | Source | Typical Value |
|-----------|----------|--------|---------------|
| **Preferences** | | | |
| `T` | Lifecycle length | Life expectancy − entry age | 40–60 |
| `retirement_age` | Retirement timing | Statutory/effective retirement | 30–45 |
| `gamma` | Risk aversion (CRRA) | Chetty (2006), Attanasio & Weber | 1–4 |
| **Asset Grid** | | | |
| `a_min` | Borrowing constraint | Institutional (zero = no borrowing) | 0.0 |
| `a_max` | Maximum assets on grid | Large enough to not bind | 50–200 |
| `n_a` | Asset grid points | Numerical accuracy | 80–200 |
| **Production** | | | |
| `alpha` | Capital share | National accounts | 0.33 |
| `delta` | Depreciation rate | Investment/capital ratio | 0.05–0.10 |
| **Tax Rates** | | | |
| `tau_c` | Consumption tax | Effective rate from fiscal data | 0.05–0.25 |
| `tau_l` | Labor income tax | Effective rate from fiscal data | 0.10–0.30 |
| `tau_p` | Payroll tax (wages only) | Social security contribution | 0.10–0.25 |
| `tau_k` | Capital income tax | Effective rate from fiscal data | 0.15–0.30 |
| **Transfers** | | | |
| `pension_replacement_default` | Pension generosity | SS replacement rate data | 0.40–0.80 |
| `ui_replacement_rate` | UI wage replacement | Statutory UI rules | 0.20–0.60 |
| **Health Expenditure** | | | |
| `kappa` | Govt coverage of medical costs | MEPS / CMS data | 0.50–0.90 |
| `m_good` | Medical cost in good health (share of income) | MEPS | 0.02–0.10 |
| `m_moderate` | Medical cost in moderate health | MEPS | 0.30–0.60 |
| `m_poor` | Medical cost in poor health | MEPS | 0.70–1.00 |
| `h_good` | Productivity in good health | Normalized | 1.0 |
| `h_moderate` | Productivity in moderate health | MEPS / HRS | 0.5–0.8 |
| `h_poor` | Productivity in poor health | MEPS / HRS | 0.1–0.4 |
| **Education** | | | |
| `education_shares` | Population shares by education | Census / ACS | Country-specific |
| `edu_params[type].mu_y` | Mean log income by education | CPS / ACS | low: 0.03–0.08, med: 0.08–0.12, high: 0.10–0.15 |
| `edu_params[type].unemployment_rate` | Unemployment rate by education | CPS / BLS | low: 0.08–0.12, med: 0.04–0.08, high: 0.02–0.05 |
| **Demographics** | | | |
| `pop_growth` | Population growth rate | Demographic projections | -0.01–0.02 |
| `birth_year`, `current_year` | Cohort timing | Calendar alignment | Scenario-specific |

**Notes:**
- Payroll tax (`tau_p`) applies to wages only, not to pensions or UI benefits.
- Pensions are computed as `pension_replacement * w_at_retirement * y_grid[i_y_last]`, where `i_y_last` is the last working-period income state. This is not an average-earnings formula.
- Tax rates support time-varying paths (`tau_c_path`, etc.) for transition experiments.
- Medical costs (`m_good`, `m_moderate`, `m_poor`) are flat across age in the current model. If age-varying costs are needed, the model must be extended.

---

## Step 2: Internally Calibrated Parameters

These parameters are calibrated via moment matching (Simulated Method of Moments).

| Parameter | Role | Identified By | Notes |
|-----------|------|---------------|-------|
| `beta` | Discount factor | Capital-output ratio K/Y | GE parameter — requires market clearing |
| `A` | TFP level | Output/wage level normalization | Production-side param in `OLGTransition`, not `LifecycleConfig` |
| `edu_params[type].rho_y` | Income persistence (per education) | Consumption-income comovement | Currently same for all types (0.97); can differ |
| `edu_params[type].sigma_y` | Income shock variance (per education) | Wealth Gini, fraction with zero wealth | Currently same for all types (0.03); can differ |
| `job_finding_rate` | Probability of exiting unemployment | Unemployment duration | Single parameter, shared across types |
| `max_job_separation_rate` | Cap on separation rate | Unemployment rate by age | Actual separation rate is derived: `min(u/(1-u) * jfr, max_sep_rate)` |
| `P_h_young`, `P_h_middle`, `P_h_old` | Health transition matrices by age group | Health status distribution by age | 3 matrices × `n_h²` entries; highly parametric |
| `initial_assets` | Initial asset level at entry | Wealth distribution at entry age | Currently a scalar (all agents start at same level, default `a_min`) |

**Notes:**
- `rho_y` and `sigma_y` are defined per education type in `edu_params`. The plan can calibrate them jointly (same for all types) or separately. Joint calibration reduces dimensionality.
- `job_separation_rate` is not a free parameter — it's computed from `unemployment_rate` and `job_finding_rate`. The free parameters are `job_finding_rate`, `max_job_separation_rate`, and `edu_params[type].unemployment_rate`.
- Changing `A` or `beta` requires re-solving the full GE equilibrium (price iteration), not just a single lifecycle problem.
- To match a wealth *distribution* at entry (rather than a single point), the model would need an `initial_asset_distribution` feature (not yet implemented).

---

## Step 3: Target Moments

These are the empirical moments used to identify the internally calibrated parameters.

| Moment | Data Source | Identifies |
|--------|-------------|------------|
| **Macro Aggregates** | | |
| Capital-output ratio K/Y | National accounts | `beta`, `A` |
| Labor share of income | National accounts | `alpha` (validation) |
| **Wealth Distribution** | | |
| Wealth Gini coefficient | SCF | `sigma_y`, `a_min` |
| Fraction with zero/negative wealth | SCF | `a_min`, `sigma_y` |
| Median wealth-to-income ratio | SCF | `beta` |
| Wealth-to-income ratio by age | SCF / PSID | `beta`, income process |
| Wealth-to-income ratio by education | SCF | Education-specific `mu_y` (validation) |
| **Income Dynamics** | | |
| Consumption-income comovement | CEX / PSID | `rho_y` |
| Earnings variance by age | PSID | `sigma_y` |
| **Labor Market** | | |
| Unemployment rate (aggregate) | CPS / BLS | `unemployment_rate`, `job_finding_rate` |
| Average unemployment duration | CPS | `job_finding_rate` |
| Unemployment rate by education | CPS | `edu_params[type].unemployment_rate` |
| **Health** | | |
| Fraction in poor health by age | MEPS / HRS | Health transition matrices |
| Medical spending by age | MEPS | `m_good`, `m_moderate`, `m_poor`, `kappa` (validation) |
| **Fiscal** | | |
| Government spending / GDP | NIPA | Pension/health generosity (validation) |
| Tax revenue / GDP | NIPA | Tax rates (validation) |

**Notes:**
- "Unemployment rate by age" was listed as a target, but the model's separation rate is not age-varying — it depends only on education type. This moment can validate the aggregate fit but cannot identify age-specific parameters without model extension.
- The borrowing constraint `a_min` strongly affects wealth distribution moments. If `a_min = 0` is imposed externally, then `sigma_y` bears more of the identification burden for wealth dispersion.

---

## Step 4: Calibration Algorithm

### 4.1 Calibration Structure

Split calibration into two stages to reduce dimensionality and avoid unnecessary GE iterations.

**Stage A — Partial equilibrium (fixed prices):**
Fix `r` and `w` at empirically reasonable values. Calibrate parameters that primarily affect individual behavior:

| Parameter | Target |
|-----------|--------|
| `rho_y`, `sigma_y` | Earnings variance by age, consumption-income comovement |
| `job_finding_rate` | Average unemployment duration |
| `edu_params[type].unemployment_rate` | Unemployment rate by education |
| `P_h` matrices | Fraction in poor health by age |

**Stage B — General equilibrium:**
With Stage A parameters fixed, calibrate GE parameters by iterating on market clearing:

| Parameter | Target |
|-----------|--------|
| `beta` | Capital-output ratio K/Y |
| `A` | Wage level normalization (or set `A=1` and let `w` be the target) |

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

1. **Nelder-Mead simplex** (`scipy.optimize.minimize(method='Nelder-Mead')`) for Stage A
2. **Brent's method** or bisection for Stage B (low-dimensional, often 1–2 parameters)

For Stage B, the GE loop already provides a natural fixed-point iteration:
1. Guess `beta` → solve all lifecycle problems → simulate → aggregate K, L → compute new `r`, `w` → check K/Y
2. Bisect on `beta` until K/Y matches target

### 4.4 Simulation Requirements

- Use `n_sim >= 10,000` agents per education type to reduce simulation noise
- Fix the random seed across objective function evaluations for smooth optimization landscape
- Discard the first few periods of simulation as burn-in if agents start from non-ergodic initial conditions

### 4.5 Convergence Criteria

- Stage A: `Q(theta) < tol` or parameter changes `< 1e-5` across iterations
- Stage B: `|K/Y_model - K/Y_target| < 0.01` (1% tolerance on capital-output ratio)
- Full convergence: all target moments within 5% of data values, or within 2 standard errors of data estimates

### 4.6 Practical Workflow

1. Set externally calibrated parameters (Step 1) from data
2. Run Stage A calibration with fixed prices
3. Validate Stage A: check that simulated income dynamics, unemployment rates, and health distributions match data
4. Run Stage B GE calibration
5. Validate Stage B: check fiscal moments (tax revenue/GDP, spending/GDP) as untargeted predictions
6. Sensitivity analysis: vary `gamma`, `a_min`, key tax rates to check robustness
