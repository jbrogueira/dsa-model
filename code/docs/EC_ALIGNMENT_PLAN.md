# Aligning the baseline with the Commission's debt projection for Greece

Written 2026-10-05. Status on 2026-10-06:

- Implemented: items 1 (real sovereign rate), 2 (pension per pensioner),
  3 (minimum pension), 4 (effective retirement age), the levels of item 5
  (2023 unemployment rates and education shares), and items 7 and 8 as a
  calculation on the baseline's saved paths (`baseline_closure.py`).
- Implemented later on 2026-10-06: the O/Y path and the stock-flow term in
  the fiscal experiments (`run_fiscal_figures.py`, `compute_debt_path`,
  `eval_fiscal_results.py`); results in `output/fiscal_2026-10-06/` and the
  last section of the calibration report.
- Not implemented: the declining unemployment path of item 5, item 6 (health
  care unit costs) and item 9.

The model numbers in this document are from the baseline transition of
2026-10-05 (effective retirement age, `960d20e`), before any of these changes.
The calibration report has the numbers of the run that includes them.

## Object

`data/2026-09-28 GR DSA Spring Forecast 2026.xlsx` holds one projection of
Greek government debt and gross financing needs for 2025-2060. Its rows follow
the template of the Commission's Debt Sustainability Monitor, started from the
Spring 2026 Economic Forecast. The file states neither units nor source. Its
accounting identities hold to machine precision with flows and stocks in % of
GDP and nominal GDP in EUR bn:

- GFN = amortisation + interest - primary balance + stock-flow adjustment
  (financing needs);
- d_t = d_{t-1}(1 + i_t)/(1 + nominal growth_t) - pb_t + both stock-flow
  adjustments;
- nominal effective rate i_t = interest_t Y_t / (d_{t-1} Y_{t-1});
- real effective rate = (1 + i_t)/(1 + deflator growth_t) - 1.

| % of GDP | Debt, DSA file | Debt, baseline | Primary balance, DSA file | Primary balance, baseline |
|---|---|---|---|---|
| 2025 | 146.1 | 165.4 | 4.88 | 1.17 |
| 2030 | 124.5 | 171.4 | 2.30 | 0.35 |
| 2040 | 111.4 | 202.5 | 1.07 | -2.14 |
| 2050 | 111.1 | 264.0 | 0.39 | -3.52 |
| 2060 | 103.4 | 320.7 | 1.53 | -1.80 |

Baseline debt is B_t/Y_t from B_{t+1} = [(1 + r_B)B_t + PD_t]/Γ_t, the series
in the calibration report. It peaks at 334% of output in 2067.

## The Commission's method

Debt Sustainability Monitor 2025 (Institutional Paper 332, February 2026), box
on the baseline assumptions:

- Real growth is the forecast for the first two years. The output gap then
  closes and growth follows potential growth (10-year methodology of the
  Output Gap Working Group), and the 2024 Ageing Report after that.
- GDP deflator inflation converges to market expectations 10 years ahead and
  to 2% within 30 years.
- The structural primary balance before cost of ageing stays at its forecast
  value. The DSA file holds it at 2.297% of GDP from 2027. The cost of ageing
  (pensions, health care, long-term care, education) from the 2024 Ageing
  Report and property income are added to it. The cyclical component is the
  output gap times a semi-elasticity and is zero once the gap closes. One-offs
  are zero after the forecast years.
- Long-term rates on new debt converge to market forward rates within 10
  years and to 4% nominal (2% real) within 30 years. Short-term rates converge
  to 2% nominal. The implicit rate follows from these rates, the maturity
  structure and the financing needs.
- Stock-flow adjustments are zero after the forecast years except in specific
  cases. The DSA file has entries to 2060.

The Monitor's own table for Greece uses the Autumn 2025 forecast and ends in
2036 (structural primary balance 1.8% of GDP, debt 123.5% in 2036). No
Commission document found publishes the 2037-2060 part of the projection.

## Differences from the baseline

| Assumption | Commission / DSA file | Baseline |
|---|---|---|
| Sovereign rate | Nominal implicit rate 1.8% in 2026, 3.8% in 2041, 3.3% in 2060, with deflator growth of 2.9% falling to 2.0%. Real effective rate -1.1%, 1.7%, 1.25% | r_B = 1.9% constant, a nominal average, set against real growth (item 1) |
| Public pensions, % of GDP | 14.5 (2022), 12.7 (2030), 13.7 (2040), 14.0 (2050), 12.7 (2060), 12.0 (2070) | 16.0 (2023), 18.2, 21.3, 23.2, 20.6, 15.5 |
| Pensioners per employed person, base year = 1 | 1.05 (2030), 1.26 (2040), 1.48 (2050), 1.45 (2060) | Retirees per non-retired person: 1.13, 1.34, 1.48, 1.32 |
| Pension per pensioner relative to GDP per employed person, base year = 1 | 0.84 (2030), 0.75 (2040), 0.65 (2050), 0.60 (2060) | 1.00, 0.99, 0.98, 0.98 |
| Indexation of pensions in payment | min(CPI, 0.5 CPI + 0.5 GDP growth) | Benefits grow with g, 1.7% a year in real terms |
| Effective retirement age | 63.8 (2022), 65.5 (2030), 66.4 (2040), 66.6 (2050), 67.9 (2070) | 63.8, 64.7, 65.7, 66.6, 68.6 |
| Unemployment rate | 12.5% (2022), 10.2% (2032), 6.8% (2050), 6.6% (2060) | 14.6% in every year |
| Employment | Employment rate 20-64 from 66.1% (2022) to 73.9% (2050). Employment 18.5% lower in 2050 than in 2022 | Non-retired population 25% lower in 2050 than in 2023 |
| Public health care, % of GDP | 5.2 in 2025, +0.6 pp by 2040, +0.7 pp by 2070. Unit costs grow with GDP per capita | 5.5 in 2025, +1.2 pp by 2040, peak 7.4 in 2053. Unit costs grow with trend productivity |
| Education, long-term care | -0.3 pp and 0.0 pp by 2040 | No separate lines |
| Primary balance rule | Structural balance before cost of ageing fixed at 2.3% of GDP from 2027 | Tax rates, G/Y, defence/Y and O/Y fixed. Revenue and transfers come from the household block |
| Revenue, % of GDP | Constant under that rule | 35.7 in 2023, 39.5 in 2050 |
| Real growth | 0.97% (2026-30), 0.92% (2031-40), 1.17% (2041-50), 1.20% (2051-60). Hourly productivity in the Ageing Report: 1.1%, 1.6%, 2.1%, 1.8% | 0.73%, 0.62%, 0.51%, 1.05%. g = 1.7% constant, labour input -0.96% a year over 2026-60 |
| Stock-flow adjustments | 6.2% of GDP in 2025, 2.8% in 2026, 0.8-1.1% a year in 2027-32 | None |
| Starting stock | 154.2% of GDP in 2024, 146.1% in 2025 | 164.3% in 2023, 165.4% in 2025 |
| Population | EUROPOP2023, all ages | EUROPOP2023, ages 25-84 |

The Ageing Report rows are from the country fiche for Greece (December 2023):
Table 6 (pension spending), Table 10 (pensioners and employment), Table 4
(effective retirement age), Table 3 (employment rates). Pension per pensioner
relative to GDP per employed person is the index of Table 6 divided by the
index of pensioners per employed person from Table 10. The fiche's own
decomposition of the change in pension spending between 2022 and 2050 (Table
8) is: dependency ratio +9.2 pp of GDP, coverage -1.4, benefit ratio -5.6,
labour market -2.0, residual -0.7, total -0.5.

In the baseline, pensions/Y in year t equals its 2023 value times retirees
per non-retired person relative to 2023, within 0.6 pp of output in every
year to 2100. The count uses the EUROPOP2023 population by age and each
cohort's retirement age from `data/retirement_age_GR.npz`.

## Changes

Effects marked mechanical hold household decisions fixed.

### 1. Sovereign rate in real terms

**The rate in the code.** `prices.r_B = 0.019` is the 2012-2024 mean of
interest payments over debt (`DATA_GR.xlsx`, sheet `DATA`, codes 40 and 49),
1.93%. Both series are nominal, so this is a nominal rate. The model has no
price level. The debt recursion divides by Γ_t = (1 + g)(1 + n_t), which is
real growth, and interest paid abroad enters the resource constraint in units
of output. The rate consistent with the model is (1 + i)/(1 + π) - 1.

| | Interest / GDP | Debt / GDP | Implicit rate i | GDP deflator growth π | Real rate |
|---|---|---|---|---|---|
| 2012 | 5.42 | 164.1 | 3.30 | -0.33 | 3.65 |
| 2013 | 4.16 | 180.4 | 2.31 | -1.96 | 4.35 |
| 2014 | 4.02 | 182.7 | 2.20 | -1.92 | 4.20 |
| 2015 | 3.58 | 179.6 | 1.99 | -0.17 | 2.17 |
| 2016 | 3.22 | 183.1 | 1.76 | -0.49 | 2.26 |
| 2017 | 3.14 | 182.1 | 1.72 | 0.20 | 1.52 |
| 2018 | 3.37 | 189.0 | 1.78 | -0.23 | 2.02 |
| 2019 | 2.97 | 183.2 | 1.62 | 0.24 | 1.37 |
| 2020 | 2.96 | 209.4 | 1.41 | -0.36 | 1.78 |
| 2021 | 2.45 | 197.3 | 1.24 | 1.39 | -0.15 |
| 2022 | 2.50 | 177.8 | 1.40 | 6.29 | -4.59 |
| 2023 | 3.39 | 164.3 | 2.06 | 6.27 | -3.96 |
| 2024 | 3.48 | 154.2 | 2.26 | 3.21 | -0.92 |
| Mean 2012-24 | | | 1.93 | 0.93 | 1.05 |
| Mean 2012-19 | | | 2.09 | -0.58 | 2.69 |
| Mean 2015-24 | | | 1.73 | 1.63 | 0.15 |
| Mean 2019-24 | | | 1.67 | 2.84 | -1.08 |

All columns in %. The deflator is nominal over chain-linked GDP
(`data/nomgdp_GR.json`, `data/realgdp_GR.json`).

**The rate adopted.** r_B is the mean real effective interest rate of the DSA
projection over 2026-2060, 1.05%. The file computes the real effective rate
each year as (1 + i_t)/(1 + deflator growth_t) - 1, with i_t the nominal
effective rate, interest_t Y_t / (d_{t-1} Y_{t-1}). The rate is -1.1% in 2026,
0.86% in 2033, 1.69% in 2041 and 1.25% in 2060.

| Measure | Period | Nominal rate | Deflator growth | Real rate |
|---|---|---|---|---|
| DSA file, real effective interest rate | 2026-60 | 3.25 | 2.18 | 1.05 |
| Same, with the debt-increasing adjustment of 2026-32 counted as interest | 2026-60 | 3.40 | 2.18 | 1.19 |
| Debt Sustainability Monitor 2025, table for Greece (Autumn 2025 forecast) | 2026-36 | 3.02 | 2.32 | 0.68 |
| Interest payments over debt and the GDP deflator (table above) | 2012-24 | 1.93 | 0.93 | 1.05 |

All columns in %, period means. No published series of the real rate was
found. AMECO publishes the nominal implicit rate (series AYIGD, interest as a
percentage of gross public debt of the preceding year).

The file's interest row is below the forecast's interest in 2026-32. Interest
is 2.47% of GDP in 2026 in the file. The Spring 2026 forecast has a primary
surplus of 4.0% and a headline surplus of 0.8%, so interest of 3.2%. The
difference equals the file's debt-increasing stock-flow adjustment (0.77% of
GDP in 2026, 0.8-1.1% a year to 2032). The file does not label the
adjustment. Its loans row is labelled "including deferred interests", and
interest rises by 1.2% of GDP in 2033, when the adjustment ends. With these
adjustments counted as interest the mean real rate is 1.19%. Equivalently, the constant real rate
that reproduces the file's 2060 debt, given its growth and primary balance, is
0.98% with the file's adjustments in the recursion and 1.19% without them. At
r_B = 1.05% the deferred interest enters the baseline through the adjustment
rows of item 7.

| r_B | Debt 2030 | 2040 | 2050 | 2060 | Peak |
|---|---|---|---|---|---|
| 1.9% (nominal mean 2012-24) | 171.4 | 202.5 | 264.0 | 320.7 | 334.4 in 2067 |
| 1.19% | 163.1 | 180.5 | 223.8 | 259.5 | 261.4 in 2065 |
| 1.05% (adopted) | 161.5 | 176.4 | 216.7 | 249.0 | 250.3 in 2061 |
| 0.68% | 157.3 | 166.1 | 199.1 | 223.6 | 224.1 in 2061 |
| DSA real effective rate, year by year | 139.9 | 150.5 | 200.0 | 239.9 | 242.8 in 2065 |

Debt in % of output, recomputed from the saved baseline paths with nothing
else changed. The last row holds the file's rate at its end values outside
2026-2060.

**What depends on r_B.**

- Debt recursion: `fiscal_experiments.compute_debt_path`.
- Debt service line of the budget: `olg_transition.py:2148-2152`.
- Pension fund recursion: `olg_transition.py:2712-2724`.
- Rest point PD/Y = (Γ - 1 - r_B) b and the terminal conditions:
  `fiscal_experiments.py:350-352`, `:437`, `eval_fiscal_results.py:600-612`.
  At 1.9% the rate exceeds Γ_T - 1 = 1.7%, so a constant debt ratio requires
  a primary surplus. At 1.05% it is consistent with a primary deficit of
  (0.017 - 0.0105) b, 1.07% of output at b = 1.64.
- Net income from abroad r NFA + (r - r_B) B in the resource-constraint
  checks: `eval_fiscal_results.py:202-268`, `fill_report.goods_market_residual`.
  (r - r_B) B/Y is 3.45% at 1.9% and 4.85% at 1.05%.
- Base-year interest line r_B B/Y: `calibrate.py:1868-1902`. The primary
  balance adds it back, so the closure O/Y does not depend on r_B. The SMM
  targets do not involve r_B either, so θ, A_tfp and the closure are
  unchanged.
- Tax-financed experiments: the tax path that meets a debt target depends on
  r_B, so these have to be rerun.

**What to change.**

- `prices.r_B = 0.0105`.
- `fiscal.interest_over_Y = 0.0339` is nominal interest over GDP. At r_B =
  1.05% the model's line is 1.72% of output. The two are not comparable.
  Drop the entry from the validation table or state the difference there.
- Text: the parameter table label "implicit rate 2012--24"
  (`reports/fill_report.py:105`), the sources paragraph of
  `reports/calibration_report.tex`, `data_inventory.md:196-198`, and
  `TREND_GROWTH_PLAN.md:1066-1090`, which compares 1.9% with Γ - 1 = 1.09%.

The return r = 4% enters the firm's first-order condition and the household
budget. Its source was not examined here.

Needs: config and text only. No recalibration and no household re-solve for
the baseline.

Done 2026-10-06: `prices.r_B = 0.0105`; the report's parameter table and
sources paragraph describe it as a real rate.

### 2. Pension per pensioner

- Commission: falls to 0.84 of its 2022 value in 2030, 0.75 in 2040, 0.65 in
  2050, 0.60 in 2060, relative to GDP per employed person.
- Baseline: constant.
- Change: the replacement rate follows that index by calendar year. Every
  pension in payment in year t is scaled by the same factor. The path does
  not distinguish cohorts or the erosion of a pension during retirement.
- Households know the path from entry, as they know the retirement age and
  survival rates of their cohort. The cohorts alive in 2023 have saved
  against it, so the calibration's base-year cross-section carries the path.
- The 2023 target stays at 16.0% of GDP (national accounts). The fiche's
  14.5% in 2022 is on the Ageing Working Group definition.
- Where: the transition takes `pension_replacement_path` by calendar year and
  stacks it by cohort in the batched solve. `calibrate.py:1769` fills it with
  a constant. `base_year_cross_section` takes survival and retirement by
  cohort and no replacement-rate path.
- Effect (mechanical): pensions in 2050 fall from 23.2% to 15.2% of output.
- Done 2026-10-06. `build_pension_index_GR.py` writes
  `data/pension_index_GR.npz` (fiche Tables 6 and 10, linear between decades,
  one in 2023, constant after 2070: 0.853 in 2030, 0.766 in 2040, 0.668 in
  2050, 0.616 in 2060, 0.624 from 2070). `transition.pension_index_file`
  names it. `build_olg_transition` multiplies the replacement-rate path by it,
  and the calibration's base-year cross-section gives each cohort the path on
  its own calendar diagonal (`CalibrationSpec.cohort_pension_index`,
  `pension_stack` in the JAX cross-sections). Tests: `test_pension_index.py`.

### 3. Minimum pension

- Commission: the national pension is indexed by the same rule, at most CPI.
- Baseline: `pension_min_floor` is constant in detrended units, so it grows
  with g.
- Change: a path in line with item 2. Otherwise the floor binds on more
  retirees as the replacement rate falls.
- Done 2026-10-06: `external_params.pension_floor_indexed`. The floor equals
  `pension_min_floor` where the replacement rate equals the calibrated rate
  and moves in proportion to the rate, so it follows the pension index. The
  solvers receive it as minus the ratio of the floor to the calibrated rate
  (`encoded_pension_floor`), which needed no new argument in the batched
  solve. A policy experiment that changes the replacement rate moves the
  floor with it.

### 4. Effective retirement age

- Commission: fiche Table 4 (see the table above).
- Baseline: 63.8 plus the change in life expectancy at 65 at the three-yearly
  reviews.
- Change: use the Table 4 path in `build_retirement_age_GR.py`, interpolated
  between decades, and the current rule after 2070.
- Needs: the data file, then a recalibration.
- Done 2026-10-06: `build_retirement_age_GR.py` uses the fiche path from 2022
  (64.0 in 2023) and, after 2070, the change in life expectancy at 65 year by
  year (70.3 in 2100). Applying the three-yearly reviews literally moved the
  age in steps, and each step put a one-year jump into hours and output
  growth and, through the debt-stabilising rule, a sawtooth into the primary
  balance after 2070; the annual path removes it. Cohorts entering before
  2028 are unaffected, so the calibration is unchanged.
  A cohort entering in 2023 retires on average at 67.7 and one entering in
  2050 at 69.9. The previous path is kept in the file as `rule_path`.
- Effect (retiree count): retirees per non-retired person are 1.04, 1.25,
  1.51 times the 2023 value in 2030, 2040, 2050. The fiche has 1.05, 1.26,
  1.48 for pensioners per employed person.

### 5. Unemployment and education shares

- Commission: Ageing Report path (see the table above).
- Baseline run: 14.6% in every year. Its rates by education, 16.45%, 15.80%
  and 10.05% for low, medium and high, are the means of the quarterly rates
  for ages 15-64 over 2019Q1-2024Q4 (Eurostat `lfsq_urgaed`).
- Data for 2023 (Eurostat `lfsa_urgaed`, updated 2026-09-10):

  | Ages | Low | Medium | High | Total | Weighted by the model's education shares |
  |---|---|---|---|---|---|
  | 25-64 | 12.3 | 11.6 | 7.7 | 10.2 | 10.6 |
  | 15-74 | 12.5 | 12.9 | 8.2 | 11.1 | 11.4 |
  | Baseline run | 16.45 | 15.80 | 10.05 | | 14.25 |

  For ages 25-64 the rates are 11.5%, 11.0% and 6.9% in 2024, and the total
  is 8.3% in 2025.
- Level, done 2026-10-05: `edu_params[*].unemployment_rate` is 0.123, 0.116
  and 0.077, the 2023 rates for ages 25-64, the model's working ages.
  `untargeted.unemployment_rate` is 0.102. The implied separation rates are
  0.062, 0.058 and 0.037, below the cap of 0.1. `_derived.theta`, `A_tfp` and
  the closure in the config were fitted at the previous rates, so a
  recalibration is pending.
- Education shares, done 2026-10-05: `education_shares` is 0.2914, 0.4117
  and 0.2969, the 2023 shares for ages 25-84, the model's population. The
  labour force survey tables stop at age 74. Ages 25-74 are the 2023 survey
  shares (24.4%, 43.9%, 31.7%). Ages 75-84 come from the 2021 census by
  cohort (68.7%, 18.6%, 12.7%) and are 10.8% of the model's 2023 population.
  `data_inventory.md` has the construction. The baseline run used 0.2343,
  0.4705 and 0.2952, the 2019-2024 mean for ages 15-64.
- The shares differ by age: 18.9%, 46.7% and 34.3% for ages 25-64 in 2023.
  The model gives every cohort the same shares, so its working-age population
  has a low-education share 10 pp above the data and its retired population
  one below the data.
- Path, to do: a declining path. It needs transition matrices that differ by
  cohort in the batched solve, which now shares one matrix.
- Effect of the path: about 9% more employment in 2050 at given
  participation. The fiche attributes -2.0 pp of GDP of pension spending to
  the labour market by 2050.

### 6. Health care unit costs

- Commission: cost per person by age grows with GDP per capita. Half of the
  gains in life expectancy are spent in good health. The income elasticity is
  1.1, converging to 1.
- Baseline: cost per person by age is constant in detrended units.
- Change: scale the cost by detrended output per person.
- Needs: code (a time-varying cost in the batched solve) and the output path
  of a previous run.
- Effect (mechanical): the rise in health spending between 2025 and 2040
  falls from 1.2 pp to 0.7 pp of output.

### 7. Starting stock and stock-flow adjustments

- Commission: see the table above. The DSA file's adjustment rows run to
  2060.
- Baseline: debt starts from 164.3% in 2023 and the recursion has no
  adjustment term.
- Change: B_{t+1} = [(1 + r_B)B_t + PD_t + SFA_t]/Γ_t, an added term in
  `compute_debt_path`. Set it in 2023-24 to
  reproduce the 2024 and 2025 ratios, and to the file's rows afterwards. The
  rows for 2026-32 carry the deferred interest that the rate of item 1
  excludes.
- Effect on debt in 2060: about -30 pp from the stock and +9 pp from the
  adjustments after 2025.
- Done 2026-10-06 for the baseline: `build_dsa_projection_GR.py` writes
  `data/dsa_projection_GR.npz` (`fiscal.dsa_projection_file`) and
  `baseline_closure.closure_paths` carries the adjustment. Debt is the stock
  at the end of the year over the same year's output, 164.28% in 2023 as in
  the data. The adjustment is -6.1% of output in 2024 and -4.3% in 2025 on
  the earlier baseline; these residuals include the effect of inflation on
  the ratio in those years.

### 8. Other net spending set to the Commission's primary balance

**What O is.** Other net spending O is primary expenditure less revenue that
the model has no explicit line for. It enters the government budget as a
spending line, as a constant share of output. No household pays or receives
it, and it cancels in the resource constraint (it enters with -O and inside
the primary deficit with +O). It moves the primary balance one for one and,
through it, debt and the country's net foreign assets.

In 2023 the model's revenue is 35.7% of output (taxes on consumption, labour
income, payroll and capital income 33.3%, bequest tax 2.4%). Primary spending
without O is 42.0% (pensions, health, unemployment insurance and minimum
income, G, defence, public investment). The primary balance without O is
-6.3%. O/Y = -8.3% brings it to the 2023 target of 1.95%. O is negative: the
revenue the model does not have exceeds the primary spending it does not
have. The data target for tax revenue is 40% of GDP.

**Closure.** O/Y is the residual that makes the primary balance follow a
target path:

- 2023: 1.95% of GDP, the outturn. This is the scalar in the config.
- 2024-2060: the Commission's primary balance (4.7% in 2024 from the
  Monitor's table, the DSA file from 2025).
- From 2061: the primary balance that holds the debt ratio at its 2060 value,
  pb_t = b_2060 [(1 + r_B)/(1 + g_t) - 1], with g_t real output growth.

In the experiments O/Y keeps the baseline's path, as the scalar does now.

With the other lines of the current baseline, the 2025 stock, the file's
adjustments (item 7) and r_B = 1.05%:

| % of output | 2025 | 2030 | 2040 | 2050 | 2060 | 2061 | 2070 | 2100 | 2202 |
|---|---|---|---|---|---|---|---|---|---|
| Primary balance | 4.9 | 2.3 | 1.1 | 0.4 | 1.5 | -0.2 | -1.1 | -1.1 | -0.8 |
| O/Y | -12.0 | -10.2 | -11.5 | -12.2 | -11.6 | -9.7 | -5.5 | -2.2 | -2.5 |
| Debt, end of year | 146.1 | 138.8 | 130.4 | 131.8 | 122.4 | 122.4 | 122.4 | 122.4 | 122.4 |

- O/Y is -8.3% in 2023 and -11.4% in 2024. Over 2028-2060 it lies between
  -10.2% (2032) and -12.8% (2052).
- To 2060 the primary balance and the debt path do not depend on the model's
  pensions, health spending or revenue. Items 2-6 change what O absorbs.
  Debt differs from the DSA file's (124.5, 111.4, 111.1, 103.4 in 2030, 2040,
  2050, 2060) through the constant rate, where the file's real rate is
  negative until 2032, and the model's lower growth.
- After 2060 the debt-stabilising primary balance averages -0.8% of output
  and moves between -1.7% and -0.2% with yearly growth. It is 1.8 pp below
  the 2060 value in 2061. O has no effect on households, so the step affects
  the budget lines only. A linear move over 2061-2070 leaves debt at 113%.
- O/Y rises after 2060 because the model's primary balance without O rises
  from -10.1% of output in 2060 to -6.6% in 2070 and -3.3% in 2100, as
  pensions fall from 20.6% to 11.1% of output.
- O stays negative throughout. Its path is recomputed after each of items
  2-6. It follows from the run's own budget lines, with no household
  re-solve.
- Where: `simulate_transition` accepts O/Y as a path. The config and
  `pin_baseline_closure.py` hold a scalar, which stays as the 2023 value.
- Done 2026-10-06 for the baseline: `baseline_closure.closure_from_run`
  computes the path from the saved budget lines, `fill_report.py` writes it
  to `baseline_closure.npz`, and `baseline_figures.py` draws it.

**Other rules after 2060.** None of these gives a constant debt ratio:

| Rule after 2060 | Primary balance 2070 | 2100 | Debt 2070 | 2100 | 2202 |
|---|---|---|---|---|---|
| O/Y held at its 2060 value | 5.1 | 8.3 | 82 | -131 | -686 |
| O/Y back to the 2027 structural level | 3.2 | 6.4 | 100 | -64 | -513 |
| Primary balance held at 1.5% | 1.5 | 1.5 | 100 | 40 | -94 |

**O/Y held constant after 2027, for comparison.** O/Y matches the forecast
only in 2024-27 and is then held at a structural level O* = -9.7%, the value that gives 2.3% in 2027
net of the cyclical component, less the file's cyclical component (0.91% in
2028, 0.44% in 2029, -0.03% from 2030). The model's own pensions, health
spending and revenue then move the primary balance: 1.8% in 2030, -0.7% in
2040, -2.1% in 2050, -0.4% in 2060. Debt is 140.6%, 145.1%, 170.9% and
187.9% of GDP in those years.

### 9. Later

- Ages 85 and over. In 2060 the baseline has 8% fewer retirees than in 2023.
  The fiche has 11% more pensioners than in 2022.
- Productivity growth by decade. g is a scalar in the household problem.

## Model outcomes that also differ

- Revenue rises by 3.84 pp of output between 2023 and 2050 (consumption tax
  +2.05, labour income tax +0.72, capital income tax +0.73, bequest tax
  +0.35, payroll tax unchanged at 13.0%).
  With item 2 alone the primary balance in 2050 moves mechanically from -3.5%
  to +4.5% of output, against 0.4% in the DSA file. Lower pensions lower
  retirees' consumption and tax payments, so revenue has to be measured again
  after item 2.
- Inflation is needed only to report nominal objects.
- Gross financing needs, amortisation and maturity need an accounting layer
  on the debt path, with the file's amortisation profile, issuance shares and
  yield curve.

## Order of work

0. A script that reads the DSA file and `baseline_paths.npz` and writes the
   tables of this document, so each step can be measured. No script in the
   repository produces these tables yet.
1. Item 1 (config and text) and item 7 on the saved paths. Rerun the
   tax-financed experiments at the new r_B.
2. Item 4, one recalibration. It also brings θ, A_tfp and the closure in line
   with the 2023 unemployment rates and education shares.
3. Items 2 and 3, one recalibration.
4. Items 5 (path) and 6, one recalibration.
5. Item 8 after each of steps 2-4.

## Sources

In `lit-review/data-debt/`, each as `.pdf` and `.txt`:

- `pps_assessments_spring2026`: Post-Programme Surveillance Assessments,
  Spring 2026, C(2026) 4000, 2 June 2026.
- `dsm2025_ip332`: Debt Sustainability Monitor 2025, Institutional Paper 332,
  February 2026.
- `country_report_EL_2026`: 2026 Country Report Greece, SWD(2026) 208, 3 June
  2026. Annex 2 has the cost of ageing by item.
- `ageing2024_country_fiche_EL`: 2024 Ageing Report, country fiche Greece,
  December 2023.
- `ageing2024_projections_ip279`: 2024 Ageing Report, projections volume,
  April 2024.
- `ageing2024`: 2024 Ageing Report, assumptions volume, Institutional Paper
  257, November 2023.

Online: AMECO series AYIGD, implicit interest rate of general government,
https://db.nomics.world/AMECO/AYIGD.
