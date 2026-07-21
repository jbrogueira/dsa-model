# Health-expenditure flag decomposition

Spec for the exercise that uses the model's government health-spending line as a
device to detect deviations of observed spending from the model's calibrated
structure. Companion to the fiscal-experiment pipeline; no transition code is
modified.

## 1. The model object

With a single health state (`n_h = 1`) the model's government health spending
for an individual of age `j` is deterministic in the model's exogenous inputs:

    g^m(j) = kappa * m_good * a(j)

where `kappa` is the government coverage ratio, `m_good` the per-capita health
scale, and `a(j)` the normalised age profile (`m_age_profile`, mean 1 over ages).
It depends on neither the household's economic state (assets, income) nor prices.
The period-t aggregate, as a share of output, is

    g^m_t = kappa * m_good * sum_j w_{j,t} phi(j,t) a(j) / Y_t

with `w_{j,t}` the birth-cohort weights and `phi(j,t)` the fraction of the age-j
cohort alive at t (survival is applied inside the aggregation: dead agents
contribute zero). Writing the population stock share `s_{j,t} = w_{j,t} phi(j,t)`
(normalised to sum to one over ages), and the alive mass, the aggregate reduces
to the identity used below.

Because `kappa`, `m_good`, and `a(j)` are exogenous, the model's prediction for a
given year is a closed-form function of coverage, scale, age profile, and
demographic structure. This is what makes an exact model-vs-data decomposition
possible for this line, and is the reason health — not pensions, which are
endogenous — is the first application.

## 2. The identity

Everything is expressed as a share of GDP (the model's native unit; population
scale cancels, so the demographic factor is age composition, not head count).

Define, for both model and data:

| symbol | meaning | model value | data counterpart |
|---|---|---|---|
| `g`   | government health spending / GDP        | baseline path (calibrated) | Gov / GDP |
| `kappa` | coverage = gov share of health expenditure | `kappa` (config) | Gov / CHE (HF accounts) |
| `chi` | current health expenditure (CHE) / GDP   | `g^m / kappa^m` | CHE / GDP |
| `Abar` | age-cost index `sum_j s_j a(j)`         | model stock shares | population age shares |
| `psi` | residual scale `chi / Abar`             | `chi^m / Abar^m` | `chi^d / Abar^d` |

Two exact identities hold on each side:

    g = kappa * chi                      (coverage x total-health share)
    chi = Abar * psi                     (age composition x residual unit cost)
  =>  g = kappa * Abar * psi

`kappa`, `chi`, and the population needed for `Abar` come from data directly; the
same age profile `a(j)` is used on both sides, so `Abar` isolates age-composition
differences rather than a different profile. `psi` is defined residually to close
the identity: it collects the real per-capita health intensity, the relative
price of health services, arrears and clawbacks, classification changes, and any
model misspecification — everything not in coverage or age composition. `psi` is
the flag.

The model benchmark is the calibrated stationary structure (baseline transition
period `t=0`, calendar year 2020): `kappa^m`, `Abar^m`, `psi^m`. By calibration
`kappa^m * Abar^m * psi^m = g^m` matches the health target. If a data year's
coverage, age composition, and unit cost all equal the model's, its gap is zero;
a non-unit residual `psi^d / psi^m` flags a deviation.

## 3. Decomposition

The gap `g^d_X - g^m` for year X is split into additive contributions of the
three factors by a Shapley decomposition of the log-multiplicative form
`g = kappa * Abar * psi`. Each factor `F_i` moves from its model value `F_i^m` to
its data value `F_i^d`; the Shapley contribution of factor `i` is the average,
over all orderings of the factors, of the change in `g` from switching factor `i`
model→data with the others held at their position in that ordering. The
contributions are exactly additive (`sum_i c_i = g^d_X - g^m`) and independent of
ordering. Reported in percentage points of GDP.

Two reporting granularities, from the same numbers:

- two-way: `g = kappa * chi` — coverage vs total-health share;
- three-way: `g = kappa * Abar * psi` — coverage, demographics, residual.

The residual contribution (`psi`) is the quantity of interest: it is the part of
the model-data gap not accounted for by measured coverage or age composition.

### Levels variant (optional)

In euro levels the same identity separates a head-count factor:
`G^h = kappa * price * x_bar * N * [sum_j s_j a(j)]`, where `N` is population size
and `price` the relative price of health (=1 in the model). In shares of GDP,
`N` and one power of GDP cancel; the levels variant is available when a distinct
population-size factor is wanted, at the cost of mapping model units to nominal
euros (which pushes GDP-measurement error into the residual). The share-of-GDP
form is the default.

## 4. Data

Assembled by `build_health_flag_data.py` into `data/health_flag_GR.csv` (+ the
population age-share matrix in `data/health_flag_age_shares_GR.npz`), all from
`data/DATA_GR.xlsx`:

- **Coverage and CHE** — Eurostat `hlth_sha11_hf`, health care expenditure by
  financing scheme, Greece, million euro ("Input check" sheet). `kappa^d` =
  Government/compulsory schemes ÷ all financing schemes; `chi^d` = all schemes ÷
  nominal GDP. Available 2009-2023.
- **Nominal GDP** — Greece, bn euro ("Input" sheet), the /GDP denominator.
- **Population age shares** — Eurostat population on 1 January by single year of
  age, Greece ("Population by age" sheet), mapped to the model age range (real
  ages 25-84 = model ages 0..T-1); `Abar^d` uses the model's `a(j)`.

Window 2009-2022(3) spans the austerity coverage collapse (`kappa^d` falls from
0.69 in 2010 to 0.58 in 2014), the natural stress test for the residual.

## 5. Model side

`build_health_model_baseline.py` runs one baseline transition and reads the
per-(age) health cross-section the aggregation itself uses, recovering the exact
alive-fraction and stock shares (`kappa m_good a(j)` factored out). Output
`data/health_model_baseline_GR.npz` holds `kappa`, `m_good`, `a(j)`, the model
stock shares `s^m_{j,t}`, `Abar^m_t`, and the aggregate `g^m_t`. The extraction
is exact for any baseline paths because the health line is independent of the
economic solution; the run only supplies the survival-and-weight structure.

Cross-check: `kappa^m * Abar^m * psi^m` reproduces the live baseline
`gov_health/Y` to Monte-Carlo tolerance, and the calibration anchor year has a
near-zero gap and residual by construction.

## 6. Pipeline

    build_health_flag_data.py        -> data/health_flag_GR.csv (+ npz)     [data side]
    build_health_model_baseline.py   -> data/health_model_baseline_GR.npz   [model side]
    health_flag_decomposition.py     -> per-year Shapley contributions       [decomposition]
    test_health_flag_decomposition.py                                        [additivity, anchor]

The decomposition module loads the two artefacts and reports, per year, `g^d`,
`g^m`, the gap, and the coverage / demographics / residual contributions (pp of
GDP), with the two-way `kappa`/`chi` view as an alternative grouping.
