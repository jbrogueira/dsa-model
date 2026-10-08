# Aligning the baseline with the Commission's debt projection for Greece

Written 2026-10-05. Status on 2026-10-07:

- Implemented: item 1 as a rate path by calendar year (see item 1), items 2
  and 3, item 4 with the post-2070 rule of item 4, item 5 in level and path,
  and item 7 as modified on 2026-10-07.
- Not adopted: item 6 (health care unit costs) and the pension path by
  cohort and age (see items 6 and 2).
- Superseded: item 8, by `BUDGET_ALIGNMENT_PLAN.md`. The budget has no
  residual line. A tax on gross output paid by firms, at a rate pinned in
  2023, carries what O carried, and the Commission's primary balance is a
  comparison series.
- Item 9: the ages 85 and over are in the model since 2026-10-08 (ages
  25-99, `BUDGET_ALIGNMENT_PLAN.md` §3.17); productivity growth by decade
  remains open.

The model numbers in the tables of this document are from the baseline
transition of 2026-10-05 (effective retirement age, `960d20e`), before any of
these changes. The calibration report has the current numbers.

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
| Unemployment rate | 12.4% (2022), 9.9% (2030), 8.5% (2040), 6.6% (2050), 6.5% (2060-70), ages 20-64 (projections volume Table II.1.50; fiche Table 3 gives the same through the share of workers in the labour force) | 14.6% in every year |
| Employment | Employment rate 20-64 from 66.1% (2022) to 73.9% (2050). Employment 18.5% lower in 2050 than in 2022 | Non-retired population 25% lower in 2050 than in 2023 |
| Public health care, % of GDP | 5.2 in 2025, +0.6 pp by 2040, +0.7 pp by 2070. Unit costs grow with GDP per capita | 5.5 in 2025, +1.2 pp by 2040, peak 7.4 in 2053. Unit costs grow with trend productivity |
| Education, long-term care | -0.3 pp and 0.0 pp by 2040 | No separate lines |
| Primary balance rule | Structural balance before cost of ageing fixed at 2.3% of GDP from 2027 | Tax rates, G/Y, defence/Y and O/Y fixed. Revenue and transfers come from the household block |
| Revenue, % of GDP | Constant under that rule | 35.7 in 2023, 39.5 in 2050 |
| Real growth | 0.97% (2026-30), 0.92% (2031-40), 1.17% (2041-50), 1.20% (2051-60). Hourly productivity in the Ageing Report: 1.1%, 1.6%, 2.1%, 1.8% | 0.73%, 0.62%, 0.51%, 1.05%. g = 1.7% constant, labour input -0.96% a year over 2026-60 |
| Stock-flow adjustments | 6.2% of GDP in 2025, 2.8% in 2026, 0.8-1.1% a year in 2027-32 | None |
| Starting stock | 154.2% of GDP in 2024, 146.1% in 2025 | 164.3% in 2023, 165.4% in 2025 |
| Population | EUROPOP2023, all ages | EUROPOP2023, ages 25-99 |

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

**Decision of 2026-10-07: a rate path.** r_B is a path by calendar year
(`data/r_B_path_GR.npz`, `build_r_B_path_GR.py`). To 2025 it is the data's
real effective rate: -4.1% in 2023, -1.1% in 2024, and a proxy for 2025 of
-1.0%, the file's 2026 nominal rate deflated by 2025 deflator growth (the 2025
interest expenditure is not in the two Spring 2026 documents). Over 2026-2060
it is the projection's real effective rate year by year: -1.1% in 2026, -0.55%
in 2030, 1.67% in 2040, 1.59% in 2050, 1.25% in 2060. Over 2061-2070 it moves
linearly from 1.25% to 2.0%. After 2070 it is 2.0%, the Monitor's long-run
real rate on market debt, reached once the official loans are repaid.

The reason for the path: a constant 1.05% averages -0.65% over 2026-32, when
the official loans carry rates below deflator growth, with 1.47% over 2033-60,
and it puts the terminal rate below output growth (1.05 against 1.7), while
the Commission-consistent terminal rate is above it. With r_B at 2% and g at
1.7% the debt-stabilising primary balance at a ratio of 1.26 is a surplus of
about 0.4% of output.

The path enters the debt recursion, the budget's interest line, the
pension-fund recursion, the resource-constraint checks and the terminal rest
point. It does not enter the household problem. `prices.r_B` in the
configuration is the terminal value, 0.02.

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

**Alternative considered on 2026-10-07, not adopted.** The fiche's Table 9
gives a total benefit ratio of 0.76, 0.73, 0.65, 0.57, 0.52, 0.54 over
2022-2070 and a replacement rate at retirement of 0.76, 0.77, 0.70, 0.67,
0.66, 0.71 (public old-age earnings-related pensions: 0.76, 0.77, 0.70, 0.67,
0.65, 0.65). Relative to 2022 the benefit ratio is 0.96, 0.86, 0.75, 0.68 in
2030-2060 and the replacement rate at retirement 1.01, 0.92, 0.88, 0.87,
against the index of Tables 6 and 10 of 0.84, 0.75, 0.65, 0.60. The gap in
2030 is composition (Table 7): loadings on outstanding claims fall from 0.9 to
0.2% of GDP, disability pensions from 0.9 to 0.5 and survivors' pensions from
2.2 to 1.5 by 2070. The constructed index reproduces the fiche's aggregate
pension spending given the model's dependency ratio and attributes these
composition changes to the pension of every retiree.

A path by cohort and age, with new awards at the Table 9 replacement rate and
pensions in payment indexed to prices (a fall of 1.7% a year in detrended
units, the fiche's rule from 2028), would change the household's pension
formula. It was not approved on 2026-10-07 and the calendar-year index stays.
Under the index a new award in 2060 is 0.62 of the 2023 rate against the
fiche's 0.87, and a pension in payment loses 7% in detrended terms over
2050-70 against 29% under price indexation.

Every cohort alive in 2023 is assumed to have anticipated the path from entry,
as it anticipates its survival and retirement age. The alternative, a shock in
2023, was not adopted.

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
- Changed 2026-10-07: after 2070 the age rises by 0.75 of the change in life
  expectancy at 65 (69.7 in 2100). The factor is the ratio of the fiche's
  rise in the effective age (4.1 years, Table 4) to its rise in the statutory
  age (5.5 years, 67 to 72.5, Table 1a) over 2022-70. The fiche's
  labour-market exit age (63.8 to 67.5, Table 4) is up to 0.9 years below the
  effective retirement age in 2030; the model uses the latter for both.
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
- Education shares, done 2026-10-05, extended 2026-10-08: `education_shares`
  is 0.3141, 0.3990 and 0.2869, the 2023 shares for ages 25-99, the model's
  population. The labour force survey tables stop at age 74. Ages 25-74 are
  the 2023 survey shares (24.4%, 43.9%, 31.7%). Ages 75-99 come from the 2021
  census by cohort (71.4%, 17.2%, 11.4%) and are 15.0% of the model's 2023
  population.
  `data_inventory.md` has the construction. The baseline run used 0.2343,
  0.4705 and 0.2952, the 2019-2024 mean for ages 15-64.
- The shares differ by age: 18.9%, 46.7% and 34.3% for ages 25-64 in 2023.
  The model gives every cohort the same shares, so its working-age population
  has a low-education share 10 pp above the data and its retired population
  one below the data.
- Path, done 2026-10-07 (`BUDGET_ALIGNMENT_PLAN.md` §3.15): each group's
  rate is its 2023 rate times an index by calendar year
  (`data/unemployment_index_GR.npz`, `build_unemployment_path_GR.py`). The
  index is built from the 2024 and 2025 outturns for ages 25-64 (9.5% and
  8.3%, Eurostat `lfsa_urgaed`); the Spring 2026 forecast for 2026-27 (8.3%
  and 7.9% for ages 15-74, scaled by the 2023 ratio of the two age groups,
  0.919, to 7.6% and 7.3%); a linear path to the Ageing Report's 2050 rate of
  6.6% scaled by the 2023 ratio of ages 25-64 to the Report's 20-64 (0.844,
  to 5.6%); a linear path to 5.5% in 2055; and a constant after. The index is
  0.93 in 2024, 0.81 in 2025, 0.69 in 2030, 0.62 in 2040, 0.55 in 2050 and
  0.54 from 2055. The Report's own 2025-2045 values (10.9% to 7.5%) are above
  the outturns and are not used. The separation rate by calendar year gives
  each cohort its own income transition matrix by age; the base-year
  cross-section carries the path as it carries the pension index.
- Effect of the path: employment of the labour force in 2050 is about 4%
  above a flat path at the 2023 rate (from 10.2% to 5.6% unemployment).

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
- Not adopted on 2026-10-07. Unit costs stay constant in detrended units.
  The reasons recorded: the rule makes the unit cost depend on the run's own
  output per person, a fixed point with the household's out-of-pocket share;
  the Commission's rule also has an elasticity above one and a shift of the
  age profile by half of the gain in life expectancy, which the plan omitted;
  and the age range to 100 (`BUDGET_ALIGNMENT_PLAN.md` §3.17), where unit
  costs are highest, changes the health path first.

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
- Changed 2026-10-07. The two years before the projection are history. The
  2024 debt ratio is the Monitor's 154.2% and the 2025 ratio the projection's
  146.1%. The flow that reconciles each with the recursion at the model's own
  primary balance, real rate and growth is recorded as that year's stock-flow
  adjustment and labelled as such. The comparison with the projection starts
  in 2026 from the same stock. The reconciling flow is not an inflation
  effect alone: in 2024 the data's carry factor (1 + i)/(1 + nominal growth)
  is 0.969 and the model's (1 + r_B)/(1 + g) about 1.004, six points of the
  ratio, half from real growth (2.1% in the data, about 0.7% in the model)
  and half from the real rate. The file's 2025 adjustment cell of 6.24% of
  GDP is not used: from the Monitor's 2024 stock with the file's primary
  balance, nominal growth of 4.9% and an implicit rate near 2.1% it would put
  the 2025 ratio near 151, not 146.1. Over 2026-2060 the projection's
  adjustment rows enter the recursion; after 2060 the adjustment is zero.

### 8. Other net spending set to the Commission's primary balance

Superseded on 2026-10-07 by `BUDGET_ALIGNMENT_PLAN.md` §3.14 and §5: the
budget has no residual line. A tax on gross output paid by firms, at a rate
pinned in 2023 to the outturn primary balance and constant to 2060, carries
what O carried; from 2061 the rate moves linearly over ten years to the rate
that makes the debt ratio in 2080 equal to its 2070 value and stays there (a
ten-year window: at a ratio above one the one-year condition moves with the
growth rate of a single year). The Commission's primary balance and debt are
comparison series. The rest of this item
describes the closure in force from 2026-10-05 to 2026-10-07.

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
  The fiche has 11% more pensioners than in 2022. (Scheduled:
  `BUDGET_ALIGNMENT_PLAN.md` §3.17, age range to 100 before the next
  recalibration.)
- Productivity growth by decade. g is a scalar in the household problem.
  The Ageing Report's hourly productivity growth for Greece is 1.0% in 2025,
  1.3% in 2030, 2.1% in 2040-45, 2.0% in 2050, 1.6% in 2060 and 1.2% in 2070
  (projections volume Table II.1.17), against the model's 1.7% in every
  year.

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

1. Item 1 (config and text) and item 7 on the saved paths, with the
   tax-financed experiments rerun at the new r_B: done 2026-10-05/06.
2. Item 4, with θ, A_tfp and the closure brought in line with the 2023
   unemployment rates and education shares: done 2026-10-06.
3. Items 2 and 3: done 2026-10-06. Steps 1-3 were carried by the
   recalibrations of `88b44a7`, `420f67c` and `ad61203`.
4. Item 5 (path): done 2026-10-07 inside the budget restructure
   (`BUDGET_ALIGNMENT_PLAN.md` §3.15). Item 6: not adopted.
5. Item 8: replaced by the output tax of `BUDGET_ALIGNMENT_PLAN.md`.

The next recalibration is the budget plan's (`BUDGET_ALIGNMENT_PLAN.md` §4).

## Sources added 2026-10-07

- 2024 Ageing Report, projections volume, Tables II.1.17 (hourly
  productivity growth) and II.1.50 (unemployment rate, ages 20-64), in
  `lit-review/data-debt/ageing2024_projections_ip279.txt`.
- Eurostat `lfsa_urgaed`, unemployment rates by age and education, in
  `data/eurostat_raw/`.
- The Commission's Spring 2026 forecast page for Greece (unemployment
  2026-27).
- Eurostat `gov_10a_main`, `gov_10a_taxag`, `gov_10a_exp`, in
  `data/gov_accounts_GR.json`.
- The Commission's EU budget spending and revenue workbook 2000-2025, in
  `data/eu_transfers_GR.npz`.

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
