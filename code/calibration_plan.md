# OLG Model Calibration Strategy

## Model Overview

This is an **Overlapping Generations (OLG) model with heterogeneous agents** featuring:

1. **Lifecycle structure**: Agents live for `T` periods, with exogenous retirement at `retirement_age`
2. **Idiosyncratic risk**: Stochastic income (`n_y` states), health (`n_h` states), and employment shocks
3. **Education heterogeneity**: Multiple education types (`low`, `medium`, `high`) with different income profiles
4. **Perfect foresight over aggregates**: Agents know the full path of interest rates and wages
5. **Government sector**: Consumption/labor/payroll/capital taxes, unemployment insurance, pensions, and health spending
6. **Demographics**: Time-varying population growth allowing for aging experiments

---

## Step 1: Externally Calibrated Parameters

These parameters are set directly from data or existing literature estimates.

| Parameter | Category | Source | Typical Value |
|-----------|----------|--------|---------------|
| **Preferences** | | | |
| `T` | Lifecycle length | Life expectancy − entry age | 40–60 |
| `retirement_age` | Retirement timing | Statutory/effective retirement | 30–45 |
| `gamma` | Risk aversion (CRRA) | Chetty (2006), Attanasio & Weber | 1–4 |
| `frisch_elasticity` | Labor supply elasticity | Chetty et al. (2011) | 0.5–1.0 |
| **Production** | | | |
| `alpha` | Capital share | National accounts | 0.33 |
| `delta` | Depreciation rate | Investment/capital ratio | 0.05–0.10 |
| **Tax Rates** | | | |
| `tau_c` | Consumption tax | Effective rate from fiscal data | 0.05–0.25 |
| `tau_l` | Labor income tax | Effective rate from fiscal data | 0.10–0.30 |
| `tau_p` | Payroll tax | Social security contribution | 0.10–0.25 |
| `tau_k` | Capital income tax | Effective rate from fiscal data | 0.15–0.30 |
| **Transfers** | | | |
| `pension_replacement` | Pension generosity | SS replacement rate data | 0.40–0.80 |
| `ui_replacement_rate` | UI wage replacement | Statutory UI rules | 0.40–0.60 |
| `ui_eligibility_rate` | UI take-up rate | DOL / SIPP data | 0.30–0.80 |
| **Health Expenditure** | | | |
| `gov_health_share` | Govt coverage of medical costs | MEPS / CMS data | 0.50–0.70 |
| `health_cost_base` | Base medical cost (share of avg income) | MEPS | 0.03–0.08 |
| `health_cost_age_slope` | Medical cost growth per year of age | MEPS | 0.02–0.04 |
| `health_cost_sick_multiplier` | Sick vs healthy cost ratio | MEPS | 1.5–3.0 |
| **Education** | | | |
| `education_shares` | Population shares by education | Census / ACS | Country-specific |
| `education_wage_premia` | Wage multipliers by education | CPS / ACS | {'low': 0.6–0.8, 'medium': 1.0, 'high': 1.3–1.8} |
| **Demographics** | | | |
| `pop_growth` | Population growth rate | Demographic projections | -0.01–0.02 |
| `birth_year`, `current_year` | Cohort timing | Calendar alignment | Scenario-specific |

---

## Step 2: Internally Calibrated Parameters

These parameters are calibrated via moment matching (Simulated Method of Moments).

| Parameter | Role | Identified By |
|-----------|------|---------------|
| `beta` | Discount factor | Capital-output ratio K/Y |
| `A` | TFP level | Output level / wage level |
| `rho_y` | Income persistence | Consumption-income comovement |
| `sigma_y` | Income variance | Wealth Gini, fraction with zero wealth |
| `job_separation_rate` | Employment dynamics | Unemployment rate by age |
| `job_finding_rate` | Employment dynamics | Unemployment duration |
| `health_transition_probs` | Health dynamics | Health status distribution by age |
| `initial_asset_dist_params` | Initial conditions | Wealth distribution at entry age |

---

## Step 3: Target Moments

These are the empirical moments used to identify the internally calibrated parameters.

| Moment | Data Source | Identifies |
|--------|-------------|------------|
| **Macro Aggregates** | | |
| Capital-output ratio K/Y | National accounts | `beta`, `A` |
| Labor share of income | National accounts | `alpha` (validation) |
| **Wealth Distribution** | | |
| Wealth Gini coefficient | SCF | `sigma_y`, borrowing constraint |
| Fraction with zero/negative wealth | SCF | Borrowing constraint, `sigma_y` |
| Median wealth-to-income ratio | SCF | `beta` |
| Wealth-to-income ratio by age | SCF / PSID | `beta`, income process |
| Wealth-to-income ratio by education | SCF | Education-specific income |
| **Income Dynamics** | | |
| Consumption-income comovement | CEX / PSID | `rho_y` |
| Earnings variance by age | PSID | `sigma_y` |
| **Labor Market** | | |
| Unemployment rate (aggregate) | CPS / BLS | Separation/finding rates |
| Unemployment rate by age | CPS | Age-varying separation rates |
| UI recipiency rate | BLS / admin data | `ui_eligibility_rate` (validation) |
| **Health** | | |
| Fraction in poor health by age | MEPS / HRS | Health transition probs |
| Medical spending by age | MEPS | Health cost parameters (validation) |
| **Fiscal** | | |
| Government spending / GDP | NIPA | Health/pension generosity (validation) |
| Tax revenue / GDP | NIPA | Tax rates (validation) |

---

## Step 4: Calibration Algorithm
