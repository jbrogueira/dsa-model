# OLG Model Calibration Strategy

## Model Overview

This is an **Overlapping Generations (OLG) model with heterogeneous agents** featuring:

1. **Lifecycle structure**: Agents live for `T` periods, with mandatory retirement at `retirement_age` or endogenous retirement within a window `retirement_window`
2. **Idiosyncratic risk**: Stochastic income (`n_y` states), employment shocks, and stochastic mortality
3. **Education heterogeneity**: Multiple education types (`low`, `medium`, `high`) with different income profiles
4. **Perfect foresight over aggregates**: Agents know the full path of interest rates and wages
5. **Government sector**: Consumption/labor/payroll/capital taxes (flat or HSV progressive), unemployment insurance, pensions (with minimum floor), means-tested transfers, bequest taxation, health spending, public capital, defense spending
6. **Demographics**: Time-varying population growth and cohort-specific survival schedules allowing aging experiments
7. **Endogenous labor supply**: FOC-based hours choice when `labor_supply=True`; agents are employed, unemployed, or retired — with hours endogenous in the working state
8. **Health expenditure**: Deterministic age-dependent medical cost level, split between government (`kappa`) and household (`1 − kappa`); no stochastic health states (`n_h=1`)
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
| `n_h` | Number of health states | Fixed at 1 (no health process) | 1 |
| `kappa` | Govt coverage of medical costs | Health expenditure by financing scheme (Eurostat `hlth_sha11_hf`) | 0.50–0.90 |
| `m_good` | Base medical expenditure level | Aggregate health spending / GDP × mean income | Country-specific |
| `m_age_profile` | Age multiplier on base medical costs | Health spending by age (national health accounts) | Array of length T; normalized so mean = 1.0 |
| **Age-Earnings Profile** | | | |
| `wage_age_profile` | Deterministic age multiplier on wages | Mean earnings by age from household survey or LFS | Array of length T; normalized so working-age mean = 1.0 |
| `pension_avg_weight` | Weight on last income state in pension base | Derived: `(1 − ρ^N) / (N(1 − ρ))` where N = retirement_age | 0.1–0.5 (lower = more averaging) |
| **Mortality** | | | |
| `survival_probs` | Age-dependent survival π(j) | Life tables (Eurostat `demo_mlifetable`) | Shape `(T,)` or `(T, 1)` |
| `survival_improvement_rate` | Annual longevity improvement rate | Demographic projections | 0.0–0.01 |
| **Education** | | | |
| `education_shares` | Population shares by education | Census / ACS | Country-specific |
| `edu_params[type].mu_y` | Mean log income by education | CPS / ACS | low: 0.03–0.08, med: 0.08–0.12, high: 0.10–0.15 |
| `edu_params[type].unemployment_rate` | Unemployment rate by education | CPS / BLS | low: 0.08–0.12, med: 0.04–0.08, high: 0.02–0.05 |
| `job_finding_rate` | Probability of exiting unemployment | Unemployment duration data: `jfr ≈ 1 / avg_duration_years` | 0.3–0.7 |
| `max_job_separation_rate` | Cap on separation rate | Implied by unemployment rates and jfr | 0.02–0.10 |
| **Demographics** | | | |
| `pop_growth` | Population growth rate | Demographic projections | -0.01–0.02 |
| `fertility_path` | Relative entering cohort sizes over time | Demographic projections | Array, length T + T_transition |
| `birth_year`, `current_year` | Cohort timing | Calendar alignment | Scenario-specific |
| **Schooling and Children** | | | |
| `schooling_years` | Periods with child costs | Demographic data (avg child-rearing years) | 0–15 |
| `child_cost_profile` | Per-period child cost (share of income) | Consumer expenditure surveys | Array of length T |
| `education_subsidy_rate` | Subsidy reducing net child costs | Program rules | 0.0–0.50 |

**Notes:**
- Payroll tax (`tau_p`) applies to wages only, not to pensions or UI benefits. `tau_l` applies to both labor income and pension income.
- **Tax decomposition**: the EC implicit tax rate on labour (ITR) includes SSC. Do not set both `tau_l = ITR` and `tau_p = SSC rate` — this double-counts. Instead: `tau_l` = PIT rate on labor/pensions (typically 5–15%), `tau_p` = SSC rate on wages only. The total wedge on wages is `tau_p + tau_l·(1 − tau_p)`, which should equal the ITR.
- **Pension formula**: `pension = max(replacement · w_ret · pension_base, pension_min_floor)`, where `pension_base = λ · κ(ret) · y_grid[i_y_last] + (1−λ) · mean_κ · mean_y_employed`. The parameter `λ = pension_avg_weight` controls how much the last income state matters vs the career average. With `λ = 1` (default), pensions depend entirely on the last state. With `λ = (1−ρ^N)/(N·(1−ρ))`, pensions approximate career-average earnings without tracking individual histories. The `pension_min_floor` captures flat pension components (e.g., national pension).
- Tax rates support time-varying paths (`tau_c_path`, etc.) for transition experiments.
- Health states are shut down (`n_h=1`). Medical expenditure is a deterministic age-dependent level: `m(j) = m_age_profile[j] * m_good`. Government covers fraction `kappa`; household pays `(1 - kappa) * m(j)` out of pocket. Calibrate `m_good` so that aggregate health spending matches the GDP share from national accounts, and `m_age_profile` from age-specific spending data.
- **Income process**: `log(y_it) = κ(t) + z_it`, where `κ(t) = wage_age_profile[t]` is a deterministic age-earnings profile and `z_it` is a mean-zero AR(1) with persistence `rho_y` and innovation std `sigma_y`. The Tauchen grid discretizes `z` (not `y`). The parameter `mu_y` shifts the level across education types: effective income = `w · κ(t) · exp(mu_y) · y_grid[i_y]`. Calibrate `wage_age_profile` from cross-sectional mean earnings by age; `rho_y` and `sigma_y` from the variance of log earnings residuals (after removing the age profile).
- **Wage consistency**: in SOE mode, `w` is derived from the firm FOC given `r` (exogenous), `alpha`, `delta`, `A_tfp`, and public capital `K_g`. Do not set `w` directly — it is computed as `w = (1−α)·A·K_g^η·(K/L)^α` where `K/L = [(r+δ)/(α·A·K_g^η)]^{1/(α−1)}`.
- `beta` is not calibrated via moment matching. Instead it is set to be consistent with the exogenous interest rate: a standard choice is `beta = 1/(1 + r*(1-tau_k))`, which ensures that an agent with median wealth and no idiosyncratic shocks would be approximately indifferent between saving and consuming in steady state. Alternatively, `beta = 0.96` (implying a 4% annual subjective discount rate) is a widely-used benchmark. When `survival_probs < 1`, the effective discount factor is `beta * pi(j,s)`, which is already lower than `beta`; to preserve a consistent implied discount rate across specifications, slightly raise `beta` (e.g., by `1/pi_bar` where `pi_bar` is the average working-age survival probability).
- `survival_probs` are taken from life tables, not calibrated.
- `phi` (inverse Frisch elasticity) is set externally from micro estimates. Chetty (2012) meta-analysis puts the micro Frisch elasticity at ~0.5, implying `phi ≈ 2`. `nu` (labor disutility weight) is calibrated internally to match average hours.
- For the HSV progressive tax, set `tax_progressive=True` and calibrate `tax_kappa`/`tax_eta` to country-specific estimates. The tax function is `T(y) = y − κ · y^{1−η}`, implying average net-of-tax rate `κ · y^{-η}`.

---

## Step 2: Internally Calibrated Parameters

These parameters are calibrated via moment matching (Simulated Method of Moments).

| Parameter | Role | Identified By | Notes |
|-----------|------|---------------|-------|
| `edu_params[type].rho_y` | Income persistence (per education) | Variance of log earnings residuals by age (slope), wealth Gini | Can calibrate jointly (same for all types) or separately |
| `edu_params[type].sigma_y` | Income shock variance (per education) | Variance of log earnings residuals by age (level), wealth Gini | Can calibrate jointly (same for all types) or separately |
| `nu` | Labor disutility weight | Average annual hours worked | Only relevant when `labor_supply=True`; scales hours to match data level |
| `initial_asset_distribution` | Initial wealth distribution at entry | Wealth distribution at youngest cohort | Array of samples drawn at simulation start; use when entry-age wealth dispersion is non-trivial |

**Notes:**
- `rho_y` and `sigma_y` are defined per education type in `edu_params`. Joint calibration (same for all types) reduces dimensionality; type-specific calibration requires sufficient identifying variation in education-group wealth distributions.
- `rho_y` and `sigma_y` are identified from the cross-sectional variance of **log earnings residuals** by age — i.e., after removing the deterministic age profile `kappa(t)`. Under a mean-zero AR(1), `Var(z_j) = sigma_eps^2 * (1 - rho^{2j}) / (1 - rho^2)`: the slope of this age profile identifies `rho_y`, the level identifies `sigma_y`. Wealth distribution moments (Gini, zero-wealth fraction) provide additional identification. When microdata is unavailable, income Gini and wealth Gini can be used as SMM targets, but note that the Eurostat income Gini (`ilc_di12`) is equivalised household disposable income, not individual gross earnings — the model's income Gini includes individual pensions and is not directly comparable.
- `job_finding_rate` and `max_job_separation_rate` are set externally (Step 1) from unemployment duration data. The separation rate is derived: `sep = u/(1-u) * jfr`.
- `initial_asset_distribution` is now implemented — pass an array of samples to `LifecycleConfig`. When the entry-age wealth distribution is well-measured (e.g., from SCF for 25-year-olds), this is preferable to a scalar `initial_assets`.
- With `n_h=1`, health transition matrices and health-dependent productivity are inactive. Medical expenditure is set externally (Step 1) and does not require moment matching.

---

## Step 3: Target Moments

These are the empirical moments used to identify the internally calibrated parameters.

| Moment | Data Source | Identifies |
|--------|-------------|------------|
| **Macro Aggregates** | | |
| Capital-output ratio K/Y | National accounts | Validation |
| Labor share of income | National accounts | `alpha` (validation) |
| **Wealth Distribution** | | |
| Financial wealth Gini (excl. housing) | HFCS | `sigma_y`, `a_min` |
| Fraction with zero financial wealth | HFCS | `a_min`, `sigma_y` |
| Median wealth-to-income ratio | HFCS | `beta` |
| Wealth-to-income ratio by age | HFCS | `beta`, income process |
| Wealth-to-income ratio by education | HFCS | Education-specific `mu_y` (validation) |
| **Income Distribution** | | |
| Variance of log earnings residuals by age | Household survey (after removing `wage_age_profile`) | `rho_y` (slope), `sigma_y` (level) |
| Earnings Gini or P90/P10 ratio | Household survey (cross-section) | `sigma_y` (validation) |
| Consumption Gini or variance (if available) | Household expenditure survey | `rho_y` (validation) |
| **Labor Market** | | |
| Unemployment rate (aggregate) | CPS / BLS | `unemployment_rate`, `job_finding_rate` |
| Average unemployment duration | CPS | `job_finding_rate` |
| Unemployment rate by education | CPS | `edu_params[type].unemployment_rate` |
| Average annual hours worked | CPS / ATUS | `nu` (when `labor_supply=True`) |
| Hours profile by age | CPS / ATUS | `nu`, `phi` (validation) |
| **Health Expenditure** | | |
| Health spending / GDP | National health accounts (Eurostat `hlth_sha11_hf`) | `m_good` (validation) |
| Government share of health spending | National health accounts | `kappa` (validation) |
| **Fiscal** | | |
| Government spending / GDP | NIPA | Pension/health generosity (validation) |
| Tax revenue / GDP | NIPA | Tax rates (validation) |
| Pension spending / GDP | NIPA | `pension_replacement_default`, `pension_min_floor` (validation) |

**Notes:**
- "Unemployment rate by age" is not a tight target — the model's separation rate varies only by education type, not age. This moment validates aggregate fit but cannot identify age-specific parameters without model extension.
- The borrowing constraint `a_min` strongly affects wealth distribution moments. If `a_min = 0` is imposed externally, then `sigma_y` bears more of the identification burden for wealth dispersion.
- **Wealth Gini target**: the model has no housing asset. Use financial wealth Gini (excluding real estate) from HFCS, not net wealth Gini. Greek net wealth Gini (0.58) is depressed by high homeownership (72%); financial wealth Gini is substantially higher.
- **Income Gini caveat**: the Eurostat `ilc_di12` Gini is equivalised household disposable income — adjusted for household size, after taxes/transfers. The model computes individual total income (earnings + pensions + UI). These are not directly comparable. Use as a rough target only; prefer variance of log earnings by age from microdata when available.
- Hours moments (`nu`) are only relevant when `labor_supply=True`. If labor supply is exogenous (default), skip these.
- When progressive taxation is active (`tax_progressive=True`), tax revenue by income decile can be used to validate `tax_kappa`/`tax_eta`.

---

## Step 4: Calibration Algorithm

### 4.1 Calibration Structure

With `r`, `beta`, and `A` set externally (Step 1), calibration is a single partial-equilibrium stage. `w` is derived from the firm FOC (not set directly).

**Calibration stage — Partial equilibrium (derived w):**
Fix `r` (exogenous SOE rate). Derive `w` from firm FOC given `{r, alpha, delta, A_tfp, K_g, eta_g}`. Calibrate:

| Parameter | Target |
|-----------|--------|
| `rho_y`, `sigma_y` | Wealth Gini (financial, excl. housing), variance of log earnings residuals by age |
| `nu` | Average annual hours worked (when `labor_supply=True`) |

All other parameters (job_finding_rate, unemployment rates, tax rates, pension params, wage_age_profile) are set externally in the JSON input file.

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

1. Populate the country JSON input file with all externally calibrated parameters (Step 1), including `survival_probs`, `m_age_profile`, `wage_age_profile`, production params, and fiscal data ratios.
2. `load_config()` derives `w` from firm FOC, computes age weights `omega(t) = (1+g)^{-t} · S(t)`, and auto-computes `pension_avg_weight` from `rho_y` and `retirement_age`.
3. Run SMM calibration (`calibrate.py --config <country>.json --backend jax`). The optimizer targets `rho_y` and `sigma_y` against wealth Gini and income/earnings moments. All moments are computed with stationary age weights.
4. The calibration report includes: external parameters, calibrated parameters, targeted moments (model vs data), untargeted moments (with data where available), and fiscal ratios (tax revenue/Y, pensions/Y, health/Y, etc.) compared to data.
5. Validate: check that untargeted fiscal ratios, P90/P10, zero-wealth fraction, and consumption Gini are within plausible ranges.
6. Sensitivity analysis: vary `gamma`, `a_min`, `beta`, `r`, `pension_replacement_default` to check robustness.
7. When LIS microdata becomes available: re-estimate `wage_age_profile` and `rho_y`/`sigma_y` from Greek EU-SILC earnings residuals, update JSON, re-run calibration.
