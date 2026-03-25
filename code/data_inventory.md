# Data Inventory for Calibration (Greece)

Source file: `data/DATA_GR.xlsx` (11 sheets).

Other data in `data/`: `IDF20222023_a.xlsx` is the Portuguese Household Budget Survey (INE, *Inquérito às Despesas das Famílias* 2022/2023). It contains pre-tabulated aggregate tables (means by demographic group), not microdata. Country is Portugal, not Greece. It provides mean annual expenditure by COICOP category cross-tabulated by age group, education, income quintile, employment status, and household composition — but no household-level observations and no distributional moments (variance, Gini, percentiles). Useful as a reference for expenditure structure comparisons, but not directly usable for Greek calibration targets.

---

## 1. Available Data

### 1.1 Parameters sheet

Pre-computed calibration values for Greece. Single row.

| Field | Value | Maps to | Used for |
|-------|-------|---------|----------|
| Effective retirement age | 58.6 | `retirement_age` | Step 1 (external) |
| Consumption tax (effective) | 18.18% | `tau_c` | Step 1 |
| Labour income tax (effective) | 40.58% | `tau_l` | Step 1 |
| SSC effective rate | 38.34% | `tau_p` | Step 1 |
| Capital income tax (effective) | 22.36% | `tau_k` | Step 1 |
| Pension replacement rate | 76.275% | `pension_replacement_default` | Step 1 |
| UI wage replacement rate | 9.45% | `ui_replacement_rate` | Step 1 |
| Pop. share: less than upper secondary | 23.43% | `education_shares['low']` | Step 1 |
| Pop. share: secondary | 47.05% | `education_shares['medium']` | Step 1 |
| Pop. share: tertiary | 29.52% | `education_shares['high']` | Step 1 |
| Unemployment rate (less than upper) | 16.45% | `edu_params['low'].unemployment_rate` | Step 1 |
| Unemployment rate (secondary) | 15.80% | `edu_params['medium'].unemployment_rate` | Step 1 |
| Unemployment rate (tertiary) | 10.05% | `edu_params['high'].unemployment_rate` | Step 1 |
| Population growth rate | -0.573% | `pop_growth` | Step 1 |
| Mean income (less than secondary) | 9,528 EUR | `edu_params['low'].mu_y` | Step 1 |
| Mean income (secondary) | 11,965 EUR | `edu_params['medium'].mu_y` | Step 1 |
| Mean income (tertiary) | 16,725 EUR | `edu_params['high'].mu_y` | Step 1 |
| Capital share | 0.503 | `alpha` | Step 1 |
| Depreciation rate | 0.285 | `delta` | Step 1 |
| Government coverage of medical costs | 66.20% | `kappa` | Step 1 |

**Notes:**
- Capital share (0.503) and depreciation (0.285) are unusually high. Verify source; may need adjustment (standard macro: alpha ~ 0.33, delta ~ 0.05-0.10).
- `mu_y` values are in EUR levels. Need to convert to log income or normalise relative to mean.

### 1.2 DATA sheet

Annual macro time series, 1995-2024. Values normalised to output (GDP shares).

| Variable | Code | Maps to | Used for |
|----------|------|---------|----------|
| Output | 11.1 | Normalisation | — |
| Private Consumption | 12 | C/Y | Validation |
| Public Consumption | 13 | G/Y | `govt_spending_path` calibration |
| Investment | 15 | I/Y | Validation (K/Y implied) |
| Private Investment | 65 | I_priv/Y | Validation |
| Public Investment | 45 | I_g/Y | `I_g_path` calibration |
| GG Revenues | 20 | Tax revenue / GDP | Validation |
| GG Taxes on Consumption | 23 | tau_c revenue check | Validation |
| GG Taxes on Labour | 25 | tau_l revenue check | Validation |
| Taxes on Profits | 26 | tau_k revenue check | Validation |
| GG Social Security Contributions | 28 | tau_p revenue check | Validation |
| Social Benefits — Pensions | 77 | Pension spending / GDP | Validation |
| Social Benefits — Unemployment | 79 | UI spending / GDP | Validation |
| Social Benefits — Means-tested | 80/78 | `transfer_floor` calibration | Validation |
| Social Benefits — Health | 81 | Health spending / GDP | `m_good` calibration, validation |
| Interest Payments | 40 | Debt service | Validation |
| GG Balance | 47 | Fiscal balance / GDP | Validation |
| Public Debt | 49 | B/Y | `B_path` calibration |
| Share of Employed | 108 | Employment rate | Validation |
| Share of Unemployed | 109 | Aggregate unemployment rate | Validation |
| Average Weekly Hours Worked | 102 | Average hours | `nu` target (Step 2) |
| Compensation of Employees | 105 | Labour share | `alpha` validation |

### 1.3 Survival rates sheet

Eurostat life table (`demo_mlifetable`): probability of surviving between exact ages `px`, Greece, total sex. Ages <1 through 85+. Years 1960-2023.

| Maps to | Used for |
|---------|----------|
| `survival_probs` | Step 1 (external). Extract latest year, slice to model age range (e.g., ages 25-84 for T=60). |
| `survival_improvement_rate` | Compute from trend in px over recent decades. |

### 1.4 Population by age sheet

Eurostat (`demo_pjan`): population on 1 January by single year of age, Greece. Ages 0-99+. Years 1960-2025.

| Maps to | Used for |
|---------|----------|
| `fertility_path` | Derive entering cohort sizes from age-0 or age-25 population over time. |
| `pop_growth` | Cross-check against Parameters sheet value. |
| Age distribution | Validation of model steady-state age distribution. |

### 1.5 Life expectancy sheet

Eurostat life table: life expectancy at exact age `ex`, Greece. Same age/year coverage as survival rates.

| Maps to | Used for |
|---------|----------|
| `T` | Set model horizon: T = life_expectancy_at_entry_age − entry_age. |

### 1.6 Input Tax rates sheet

European Commission (DG Taxation): implicit tax rates on consumption, labour, capital for EU countries, 2011-2022.

| Maps to | Used for |
|---------|----------|
| `tau_c`, `tau_l`, `tau_k` | Cross-check against Parameters sheet. Time series for trend analysis. |

### 1.7 Input income sheet

Eurostat (`ilc_di08`): mean equivalised net income by educational attainment (ISCED 0-2, 3-4, 5-8), ages 18-64, EUR. Multiple EU countries, 2003-2024.

| Maps to | Used for |
|---------|----------|
| `edu_params[type].mu_y` | Cross-check against Parameters sheet. Time series for trend. |

### 1.8 Input pensions sheet

Eurostat (`lfso_23pens03`): age at which person started receiving old-age pension, 2023 survey. Average and median retirement age.

| Maps to | Used for |
|---------|----------|
| `retirement_age` | Cross-check: Greece average 58.6, median 59. |

### 1.9 Input LFS sheet

Eurostat (`nama_10_lp_ulc`): labour productivity and unit labour costs for Greece, 1975-2023. Compensation per employee, hours worked per employee.

| Maps to | Used for |
|---------|----------|
| Hours worked per employee | `nu` target (when `labor_supply=True`). Cross-check against DATA sheet. |
| Compensation per employee | Wage level normalisation (`A` / TFP calibration). |

### 1.10 Input check sheet

Eurostat (`hlth_sha11_hf`): health care expenditure by financing scheme, Greece, million EUR. 1992-2022.

| Maps to | Used for |
|---------|----------|
| Govt/compulsory share | `kappa` cross-check (should match 66.2% from Parameters). |
| Total health spending | `m_good` calibration: total health spending / GDP × mean income → base medical cost level. |

---

## 2. Missing Data

### 2.1 Income distribution (cross-sectional, by age) — Critical

**Needed for:** `rho_y` (slope of variance-age profile), `sigma_y` (level of variance-age profile). These are the primary internally calibrated parameters (Step 2).

**What's required:** Cross-sectional variance of log earnings by age, from a household survey. Under AR(1): `Var(log y_j) = sigma_eps^2 * (1 - rho^{2j}) / (1 - rho^2)`. The slope identifies `rho`, the level identifies `sigma`.

**What we have:** Mean income by education (Eurostat `ilc_di08`). No distributional moments (variance, percentiles) and no age breakdown. The IDF (`data/IDF20222023_a.xlsx`) is Portuguese aggregate tables — not usable.

#### Literature estimates (no direct Greece estimates exist)

The best proxies are Italy and Spain. Key source: **Cooper, Haan & Zhu (BIS WP 1102 / NBER w25082)**, estimated from ECHP 1994-2001 using GMM on autocovariance structure. Model: `y = z + epsilon`, `z_t = rho * z_{t-1} + eta_t`.

| Country | Education | rho | σ²_eta (persistent innov.) | σ²_epsilon (transitory) |
|---------|-----------|-----|---------------------------|------------------------|
| **Italy** | No college | 0.944 | 0.020 | 0.072 |
| **Italy** | College | 0.921 | 0.022 | 0.029 |
| **Spain** | No college | 0.951 | 0.016 | 0.092 |
| **Spain** | College | 0.986 | 0.004 | 0.058 |
| Germany | No college | 0.895 | 0.016 | 0.022 |
| France | No college | 0.971 | 0.006 | 0.031 |

Other sources:
- **Hintermaier & Koeniger (2024, *Quantitative Economics* 15, 1249-1301)**: assumes rho=0.95, estimates σ_epsilon≈0.23 for Italy, ≈0.24 for Spain (from HFCS).
- **Carroll, Slacalek & Tokuoka (ECB WP 1648)**: permanent-transitory spec; uses σ²_psi=0.010, σ²_xi=0.010 as fallback for Greece (not a Greece estimate — same as US baseline).
- **Social Indicators Research (2025)**: 75% of Greek earners with >20% earnings shock still earned <80% of baseline in t+1 (higher persistence than Italy/Spain at 69%).

**Starting-point calibration for Greece** (pooling Italy/Spain non-college):
- `rho_y ≈ 0.95` (midpoint of Italy 0.944 and Spain 0.951)
- `sigma_eta ≈ 0.13` (√0.018, averaging Italy 0.020 and Spain 0.016)
- Transitory component: σ_epsilon ≈ 0.28 (√0.082). Our model has a single AR(1) without a separate transitory shock; SMM calibration against income variance by age will absorb this into `sigma_y`.

#### Eurostat public data (pulled 2026-03-24)

**`ilc_di12` — Gini of equivalised disposable income, Greece:**
- Total population: 31.8 (2024), 31.4 (2022), 32.4 (2021)
- Only age breakdowns available: total and under-18

**`ilc_di03` — Mean and median equivalised disposable income (EUR), Greece 2024:**

| Age group | Median | Mean |
|-----------|--------|------|
| 18-24 | 9,821 | 11,036 |
| 25-54 | 11,434 | 13,064 |
| 55-64 | 11,800 | 13,481 |
| 65+ | 10,133 | 11,506 |
| Total | 10,850 | 12,391 |

**`ilc_di01` — Income distribution, Greece 2024:**

| Decile threshold | EUR |
|-----------------|-----|
| P10 | 5,031 |
| P50 (median) | 10,850 |
| P90 | 19,616 |
| P95 | 23,917 |
| P99 | 47,020 |
| P90/P10 ratio | 3.9 |

Income shares: bottom 20% = 7.5%, top 20% = 39.5%.

#### Where to get Greece-specific microdata

| Source | Access | What it provides | How to access |
|--------|--------|-----------------|---------------|
| **LIS (Luxembourg Income Study)** | Register online, run code remotely via LISSY | Harmonized EU-SILC microdata for Greece. Can compute var(log earnings) by single-year age directly. | Register at `lisdatacenter.org`. **Fastest path to microdata — no lengthy application.** |
| EU-SILC microdata (Eurostat) | Restricted; apply via CIRCABC | Variables `PY010G` (gross employee income), `PX020` (age), `PE040` (education). Full flexibility. | `ec.europa.eu/eurostat/web/microdata/european-union-statistics-on-income-and-living-conditions`. Turnaround: weeks to months. |
| Eurostat `ilc_di12` | **Public, no application** | Gini of equivalised disposable income by broad age group for Greece. Not var(log earnings), but usable as rough validation. | `ec.europa.eu/eurostat/databrowser/view/ilc_di12` |
| Eurostat `ilc_di03` | **Public** | Mean and median income by age and sex for Greece. | `ec.europa.eu/eurostat/databrowser/view/ilc_di03` |

**Current approach (provisional):** Use Italy/Spain literature values as starting point (rho≈0.95, σ_eta≈0.13). Calibrate via SMM against Eurostat public moments (Gini=31.8, P90/P10=3.9, mean income by age). LIS access application submitted (2026-03-24) — once approved, compute var(log earnings) by age from Greek EU-SILC cross-section and re-calibrate.

### 2.2 Wealth distribution — High

**Needed for:** Validation of `sigma_y`, `beta`, `a_min`. Key moments: wealth Gini, zero/negative wealth fraction, median wealth-to-income ratio, wealth-to-income by age.

**Data obtained** from HFCS Wave 4 (reference year 2021), ECB Statistical Tables (July 2023).

Source: `ecb.europa.eu/home/pdf/research/hfcn/HFCS_Statistical_Tables_Wave_2021_July2023.pdf`

#### Headline statistics

| Statistic | Value |
|-----------|-------|
| **Wealth Gini** | **0.58** (SE 0.020) |
| Median net wealth | €84,600 (SE 2,200) |
| Mean net wealth | €132,700 (SE 8,200) |
| **Share with negative net wealth** | **1.1%** (SE 0.2) |
| p90/p50 ratio | 3.4 |
| p80/p20 ratio | 12.7 |
| Top 10% share | 41.3% |
| Homeownership rate | 72.0% |

#### Wealth percentiles (EUR thousands)

| p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 0.8 | 15.1 | 41.6 | 63.7 | 84.6 | 107.1 | 144.2 | 190.7 | 287.0 |

#### Net wealth by age of reference person

| Age group | Median (€k) | Mean (€k) | % negative NW |
|-----------|------------|-----------|---------------|
| 16-34 | 8.4 | 37.5 | 1.1% |
| 35-44 | 68.1 | 82.7 | 1.8% |
| 45-54 | 86.5 | 116.7 | 0.8% |
| 55-64 | 336.8 | 180.5 | 1.0% |
| 65-74 | 296.7 | 104.3 | 1.3% |
| 75+ | 85.9 | 163.7 | 0.3% |

#### Net wealth by quintile (EUR thousands, mean)

| Bottom 20% | 20-40% | 40-60% | 60-80% | 80-90% | 90-100% |
|------------|--------|--------|--------|--------|---------|
| -1.7 | 55.7 | 127.5 | 237.2 | 441.4 | 1,485.0 |

**Notes:**
- Greek wealth Gini (0.58) is low relative to euro area average (0.694), driven by high homeownership and low financial asset holdings.
- The 55-64 age group has the highest median wealth (€336.8k) — strong lifecycle hump.
- Very low negative-wealth share (1.1%) compared to Northern Europe, consistent with limited mortgage/consumer debt markets.

**Missing: financial wealth excluding housing.** The model has no housing asset, so net wealth targets should exclude real estate. The HFCS reports gross/net financial wealth separately from real assets. The published statistical tables (Table A3/A4) may have this breakdown; if not, HFCS microdata is needed. The financial wealth Gini will be substantially higher than net wealth Gini (0.58) because housing is the great equaliser in Greece (72% homeownership). Zero-financial-wealth fraction will also be much higher than 1.1%, making the model's 4.5% more plausible.

### 2.3 Unemployment duration — Medium

**Needed for:** `job_finding_rate` (Step 2). Average unemployment duration identifies `job_finding_rate` directly: `jfr ≈ 1 / avg_duration_in_years`.

**Data obtained** from Eurostat `lfsa_ugad` and `lfst_r_lfu2ltu`, pulled via Python `eurostat` package (2026-03-24).

#### Aggregate unemployment duration (Greece, 2020-2024)

Duration bracket midpoints: <1m→0.5, 1-2m→1.5, 3-5m→4, 6-11m→8.5, 12-17m→14.5, 18-23m→20.5, 24-47m→35.5, 48+m→60.

| Year | Total unemp. (k) | LTU share (12m+) | Mean duration (months) | **jfr (annual)** |
|------|------------------|-------------------|----------------------|-----------------|
| 2020 | 755 | 66.7% | 29.4 | **0.41** |
| 2021 | 659 | 64.8% | 28.5 | **0.42** |
| 2022 | 556 | 59.2% | 26.1 | **0.46** |
| 2023 | 504 | 56.5% | 24.6 | **0.49** |
| 2024 | 480 | 54.9% | 23.6 | **0.51** |

#### Job finding rate by education (2024, approximate)

Education-specific estimates combine LTU shares by education with aggregate within-group duration distributions. Small gradient because LTU shares are similar across education groups in Greece (53-55% in 2024).

| Education | LTU share | Implied jfr (annual) |
|-----------|-----------|---------------------|
| ISCED 0-2 (lower secondary or below) | ~55% | **0.505** |
| ISCED 3-4 (upper secondary) | ~54% | **0.518** |
| ISCED 5-8 (tertiary) | ~53% | **0.522** |

**Notes:**
- The 48+ month bracket contains ~24% of the unemployed; midpoint assumption of 60 months matters. If true conditional mean is 72 months, jfr would be ~0.05 lower.
- Unemployment *rates* differ substantially by education (ISCED 0-2: ~16%, ISCED 5-8: ~10%), but this reflects separation rates and labor force composition, not just finding rates.
- For calibration: use `job_finding_rate ≈ 0.50` (2024 value). The model derives separation rates from unemployment rates and finding rates: `sep_rate = u/(1-u) * jfr`.

### 2.4 Health expenditure by age — Medium

**Needed for:** `m_age_profile` (Step 1). An age profile of per-capita health spending, normalised so that the weighted average equals 1.0.

**Data obtained** from EU 2024 Ageing Report (DG ECFIN), Graph I.2.2, p.68. No Greece-specific age profile is publicly available; the EU14 aggregate (which includes Greece) is used as proxy. Eurostat SHA has no age dimension; OECD data for Greece is incomplete.

Source: `economy-finance.ec.europa.eu/publications/2024-ageing-report_en`

#### Per-capita public health spending by age (% of GDP per capita, EU14 aggregate)

| Age group | % GDP p.c. | Normalised (mean=1.0) |
|-----------|-----------|----------------------|
| 20-24 | 1.78 | 0.223 |
| 25-29 | 2.32 | 0.291 |
| 30-34 | 2.64 | 0.331 |
| 35-39 | 2.44 | 0.306 |
| 40-44 | 2.80 | 0.351 |
| 45-49 | 3.44 | 0.431 |
| 50-54 | 4.40 | 0.551 |
| 55-59 | 5.48 | 0.687 |
| 60-64 | 6.74 | 0.845 |
| 65-69 | 8.20 | 1.027 |
| 70-74 | 10.06 | 1.260 |
| 75-79 | 12.50 | 1.566 |
| 80-84 | 15.00 | 1.879 |
| 85-89 | 17.40 | 2.180 |
| 90-94 | 17.00 | 2.130 |
| 95-99 | 14.50 | 1.817 |

**Key ratios:** avg(65+)/avg(20-64) = 3.84×. Peak at ages 85-89. Decline at 90+ (rationing + proximity-to-death effects).

**Greece context:** Public health spending = 5.37% of GDP (2022); per capita = €2,191 (2023).

**For calibration:** Interpolate 5-year groups to single-year ages for `m_age_profile`. Normalise so population-weighted average = 1.0. Set `m_good` so that aggregate `kappa * sum(m_age_profile * pop_weights)` matches health spending/GDP.

**References:**
- EU 2024 Ageing Report, DG ECFIN — age profiles used in EU fiscal projections.
- Przywara (2010), European Economy Economic Papers 417.
- de la Maisonneuve & Oliveira Martins (2013), OECD WP 1048.

### 2.5 Consumption distribution — Low

**Needed for:** Validation of `rho_y`. Consumption Gini or variance of log consumption.

**What we have:** Nothing for Greece. The IDF in `data/` is Portuguese aggregate tables — not usable.

**Where to get it:** Same microdata sources as 2.1 (LIS or EU-SILC). HFCS has limited consumption data (food expenditure and rough total, not full COICOP breakdown). For a consumption Gini, the best path is EU-SILC via LIS.

### 2.6 Earnings Gini / P90-P10 — Low

**Needed for:** Validation of `sigma_y`.

**What we have:** Nothing at the distributional level.

**Where to get it:**

| Source | Access | What it provides |
|--------|--------|-----------------|
| Eurostat `ilc_di12` | **Public** | Gini of disposable income by age group |
| Eurostat `earn_ses18_20` | **Public** | Mean annual earnings by age group (Structure of Earnings Survey) |
| OECD Income Distribution Database | **Public** | Gini by age group for Greece |
| LIS / EU-SILC microdata | Register / restricted | P90/P10 of earnings, custom age bins |

---

## 3. Summary: Parameter-to-Data Mapping

### Fully covered (data available)

| Parameter | Data source (sheet) |
|-----------|-------------------|
| `tau_c`, `tau_l`, `tau_p`, `tau_k` | Parameters |
| `pension_replacement_default` | Parameters |
| `ui_replacement_rate` | Parameters |
| `education_shares` | Parameters |
| `edu_params[type].unemployment_rate` | Parameters |
| `edu_params[type].mu_y` | Parameters, Input income |
| `pop_growth` | Parameters |
| `alpha` | Parameters (verify) |
| `delta` | Parameters (verify) |
| `kappa` | Parameters, Input check |
| `retirement_age` | Parameters, Input pensions |
| `survival_probs` | Survival rates |
| `T` | Life expectancy |
| `m_good` | DATA (health spending/GDP), Input check |

### Partially covered (data available but requires processing)

| Parameter / Moment | Available data | Gap |
|-------------------|---------------|-----|
| `fertility_path` | Population by age (age-0 counts by year) | Need to extract and normalise |
| `survival_improvement_rate` | Survival rates (multiple years) | Need to compute trend |
| `nu` target (avg hours) | DATA (weekly hours), Input LFS | Available as aggregate; no age profile |

### Obtained (2026-03-24) — previously missing

| Parameter / Moment | Status | Values |
|-------------------|--------|--------|
| `rho_y`, `sigma_y` | **Provisional** (literature proxies; LIS pending) | rho≈0.95, σ_eta≈0.13 (Italy/Spain). Eurostat: Gini=31.8, P90/P10=3.9 |
| Wealth Gini, zero-wealth fraction | **Obtained** (HFCS Wave 4) | Gini=0.58, negative NW=1.1%, median=€84.6k, by-age profile available |
| `job_finding_rate` | **Obtained** (Eurostat `lfsa_ugad`) | jfr≈0.50/year (2024); by education: 0.505-0.522 |
| `m_age_profile` | **Obtained** (EU Ageing Report 2024) | EU14 aggregate profile; normalised 5-year group values available |
| Earnings P90/P10 | **Obtained** (Eurostat `ilc_di01`) | P90/P10=3.9 (2024) |

### Still missing

| Parameter / Moment | Required data | Priority | Path to resolution |
|-------------------|--------------|----------|--------------------|
| `rho_y`, `sigma_y` (Greece-specific) | Var(log earnings) by age | **Critical** | LIS application pending (2026-03-24) |
| Financial wealth Gini (excl. housing) | HFCS gross/net financial assets | **High** | HFCS published tables (check Table A3/A4) or microdata. Model has no housing — net wealth targets overstate asset equality. |
| Zero-financial-wealth fraction | HFCS financial assets = 0 | **High** | Same source. Expected to be much higher than 1.1% net-wealth figure. |
| Consumption Gini | Consumption microdata | **Low** | LIS (when approved) |

---

## 4. Next Steps

### Completed

1. ~~Download HFCS statistical tables~~ — **Done.** Wealth Gini=0.58, by-age profiles, percentiles all recorded above.
2. ~~Download Eurostat unemployment duration~~ — **Done.** jfr≈0.50/year, by-education breakdown recorded above.
3. ~~Download health spending age profile~~ — **Done.** EU14 normalised profile from Ageing Report recorded above.
4. ~~Obtain income process proxies~~ — **Done (provisional).** Literature values (rho≈0.95, σ_eta≈0.13) + Eurostat moments recorded above.

### Remaining

5. **Wire data into calibration code**: set externally calibrated parameters from DATA_GR.xlsx and the data above; configure SMM targets from Eurostat moments and HFCS wealth distribution.
6. **LIS access** (application submitted 2026-03-24): once approved, compute var(log earnings) by age from Greek EU-SILC cross-section for direct `rho_y`/`sigma_y` identification. Re-calibrate and replace provisional literature proxies. Also covers consumption Gini.
7. **Verify capital share and depreciation**: the Parameters sheet values (alpha=0.503, delta=0.285) are unusually high; cross-check methodology and source.
