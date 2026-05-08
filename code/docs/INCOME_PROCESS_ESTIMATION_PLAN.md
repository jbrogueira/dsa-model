# Income Process Estimation Plan — Greek EU-SILC via LIS

Estimation strategy for the AR(1) labor productivity process by education type, using Greek EU-SILC microdata accessed through LISSY remote execution.

## Status

| Step | State | As of |
|------|-------|-------|
| LIS access granted | done | 2026-05-06 |
| Cross-sectional vs panel structure verified | cross-sectional only | 2026-05-06 |
| `01_extract_moments.do` written | done | 2026-05-06 |
| Local test on `data/it20ip.dta` | passed | 2026-05-06 |
| `LOCAL_RUN = 0` set for LISSY submission | done | 2026-05-06 |
| LISSY job submitted | pending | — |
| `02_estimate_ar1.py` (MD second stage) | not started — write after LISSY output returns | — |
| `calibration_input_GR.json` updated | not started | — |

## Objective

Replace the provisional Italy/Spain proxies (Cooper, Haan & Zhu, BIS WP 1102: ρ ≈ 0.95, σ_η ≈ 0.13) with Greece-specific estimates of the parameters used in `_income_process()` (Tauchen-discretized AR(1), `n_y` states, by education type):

- `ρ_z^(e)` — persistence
- `σ_η^(e)` — innovation standard deviation
- `μ_z^(e)` — mean log earnings
- `σ_α^(e)` — fixed-effect standard deviation (used to validate σ_z and to inform `μ_z` dispersion)

for `e ∈ {low, medium, high}` corresponding to ISCED 0-2, 3-4, 5+.

## Data

**Source.** LIS Database via LISSY remote execution. Access granted 2026-05-06. Greek datasets available: GR95, GR00, GR02, GR03-GR05, GR06-GR08, GR09-GR11, GR12-GR14, GR15-GR17, GR18-GR20, GR21.

**Critical constraint.** LIS Greece is **cross-sectional only** — there is no cross-wave personal identifier in the LIS standardized variable set. This is verified from the LIS variable list (only `pid`/`hid`/`did`/`year`/`wave` exist; `pid` is unique within `did` only). The longitudinal EU-SILC component is held by Eurostat under a separate research-access agreement, not by LIS.

**Implication.** Within-individual autocovariances are not observable. Persistence ρ cannot be identified from individual time variation. Identification relies on the cross-sectional age profile of `var(log earnings)`.

**Local test data.** Italian sample files (`data/it20ip.dta`, `data/it20ih.dta`) — public LIS samples, used to develop and debug the LISSY .do file before submission.

## Variables

From the LIS 2024 template:

| Code | Description | Use |
|------|-------------|-----|
| `pid`, `hid`, `did`, `year`, `wave` | IDs | Stratification |
| `pi11` | Wage income (employees) | **Primary earnings concept** |
| `pi12` | Self-employment income | Diagnostic only; excluded from baseline |
| `pilabour` | Total labor income | Robustness |
| `educ` | Education, 3-category recode | **Maps to model's low/medium/high** |
| `age` | Age in years | Primary age dimension |
| `wexptl` | Years of total work experience | Alternative age dimension |
| `status1` | Status in employment | Employee selection (`110`/`120`) |
| `weeks`, `weeksft`, `hours1`, `fyft` | Hours/weeks worked | Full-year-full-time selection |
| `pwgt` | Person weight | Survey weighting |
| `grossnet` | Gross/net income flag | Verify gross consistently across waves |

## Sample Selection

1. Working-age 25-60 (avoids schooling and early retirement margins).
2. Employees only: `status1 ∈ {110, 120}`. Self-employed excluded — labor/capital income mixing contaminates ρ.
3. `pi11 > 0` (positive wage income).
4. `fyft == 1` if available (full-year-full-time) for cleaner interpretation as a wage rate; else condition on `weeks ≥ 26` and `hours1 ≥ 30` as fallback.
5. Drop top/bottom 0.5% of `log pi11` within each (education × wave) cell.
6. Verify `grossnet` is constant (gross) across pooled waves; otherwise stratify or restrict.

## What Is Identifiable

Under a permanent + persistent decomposition `u_{ij} = α_i + z_{ij}` with `z_{ij} = ρ z_{i,j-1} + η_{ij}` and stationary initialization, the cross-sectional variance of residual log earnings as a function of age `j` is

```
Var(u_j) = σ_α² + σ_η² · (1 − ρ^{2j}) / (1 − ρ²)     (RIP, no fixed effect at entry)
```

- The **slope** of `Var(u_j)` in `j` at young ages identifies `σ_η²`.
- The **rate of slowdown** identifies `ρ`.
- The **level** at any age identifies `σ_α²` (given the other two).

**Identification strength.**
- `σ_η²` is well identified given a moderate ρ (slope at young ages dominates).
- `σ_α²` is well identified once σ_η² and ρ are pinned.
- `ρ` is **weakly identified** from cross-sections alone — it shapes the curvature of the variance profile, which is sensitive to cohort effects, selection on retirement, and measurement error.

## Estimation Strategy

**Stage 1 — moments from LISSY.** Within each education stratum, after residualizing on year fixed effects:
- Mean log `pi11` by education (→ `μ_z^(e)`).
- `Var(log pi11)` by single-year age and by 5-year age band, weighted by `pwgt`.
- Sample sizes per cell.
- Selected percentiles (p10, p50, p90) by age × education for distributional validation.
- Same moments using `wexptl` instead of `age` as a robustness check.

**Stage 2 — minimum-distance estimation (local).** For each education stratum, fit `Var(u_j) = σ_α² + σ_η² · (1 − ρ^{2j}) / (1 − ρ²)` to the age profile by weighted nonlinear least squares.

Two specifications, reported side-by-side:

1. **Joint** — estimate `(ρ, σ_η, σ_α)` jointly. Standard errors via bootstrap over LISSY-returned cell variances. Expected: large standard error on ρ.
2. **Fixed ρ** — set ρ = 0.95 (Cooper-Haan-Zhu Italy/Spain proxy), estimate `(σ_η, σ_α)`. Treat ρ as a sensitivity parameter — re-estimate at ρ ∈ {0.90, 0.95, 0.97, 0.99} and report the implied σ_η range.

**Choice for the model.** Use spec 2 with ρ = 0.95 as the baseline calibration. Report spec 1 in the calibration appendix.

## Pipeline

```
code/data/lis/
├── 01_extract_moments.do          # Stata: data prep + moments → text output
├── 02_estimate_ar1.py             # Python: takes .do output, runs MD, writes JSON [TODO]
└── output/
    ├── moments_it20.txt           # local test output (Italy 2020 sample)
    ├── moments_GR_pooled.txt      # LISSY return file [pending]
    └── ar1_estimates_GR.json      # final estimates → feeds calibration_input_GR.json [pending]
```

**`01_extract_moments.do`.** Stata script. Designed to run unchanged in two modes:

- **Local test** — `LOCAL_RUN = 1` at line 22; reads `data/it20ip.dta` directly. Used for development and debugging.
- **LISSY submission** — `LOCAL_RUN = 0`; uses LIS dataset macros (`${gr03p}`, `${gr04p}`, ...). Pools 19 Greek waves (GR03-GR21).

Three subsets reported per education stratum:

- `emp_fyft` — employees (`status1 ∈ {110,120}`), `fyft==1`, `pi11>0`. Primary specification.
- `emp_all` — employees, no `fyft` filter. Robustness.
- `selfemp` — self-employed (`status1 ∈ {200,210,220,240}`), `pi12>0`. Diagnostic only, not used in AR(1) calibration.

Each subset writes six numbered blocks: `[1]` wave-level metadata + sample sizes (used to confirm `grossnet` consistency across pooled waves), `[2]` mean log earnings by education, `[3]` `var(u)` by single-year age × education (n ≥ 10), `[4]` `var(u)` by 5-year age band × education (n ≥ 30), `[5]` log-earnings percentiles by age band, `[6]` `var(u)` by experience band as robustness. All output is aggregated cell statistics with cell-size thresholds — LISSY-compatible (no individual-level data, no graphics).

**`02_estimate_ar1.py`** [TODO — write after LISSY output returns]. Reads the moments file, runs WLS fit of the variance profile `Var(u_j) = σ_α² + σ_η²·(1−ρ^{2j})/(1−ρ²)` per education stratum, produces `(ρ, σ_η, σ_α, μ)` and writes `ar1_estimates_GR.json`. The values are then merged into `calibration_input_GR.json` under the existing `rho_y`, `sigma_y`, `mu_y` fields by education.

## Local Test Results (IT 2020 sample)

The Italy 2020 LIS public sample (`data/it20ip.dta`, 2,232 persons / 1,000 households) was used to validate the pipeline before LISSY submission. After common filters (working-age 25-60, valid education):

| Subset | N | Mean log y by educ (low/med/high) | Pooled var(log y) |
|--------|---|------------------------------------|-------------------|
| `emp_fyft` | 297 | 9.81 / 10.06 / 10.29 | 0.20 |
| `emp_all`  | 416 | 9.48 / 9.82  / 10.15 | 0.44 |
| `selfemp`  | 158 | 9.43 / 9.84  / 10.01 | 0.86 |

Three sanity checks pass:

- The `fyft` filter cleans the wage rate as expected: drops 119 part-year/part-time observations, mean log earnings rises 0.25-0.34 log points across education, and pooled variance falls 2× (0.44 → 0.20).
- Self-employed earnings dispersion is ~4× employees, consistent with mixed labor/capital income and reporting noise. Confirms `selfemp` should stay out of the AR(1) baseline.
- `grossnet = 120` (gross including non-cash) consistent across the (single) wave.

Age × education cells are empty for `emp_fyft` and `selfemp` because the IT public-use sample is too small (n=297 / 158 spread over 3 educ × 36 ages = 108 cells). This is expected for a public sample; Greek pool of 19 waves at much higher per-wave sample size will populate the profile.

## Workflow

1. Develop and test `01_extract_moments.do` against `data/it20ip.dta` (single-wave Italy 2020 LIS sample).
2. Confirm output format matches LIS aggregate-only rules.
3. Submit the .do file to LISSY targeting Greek datasets (pool GR03-GR21 to maximize sample per age × education cell).
4. Run `02_estimate_ar1.py` on the returned LISSY output.
5. Write the per-education estimates into `calibration_input_GR.json` under `edu_params.<e>.rho_y`, `edu_params.<e>.sigma_y`, `edu_params.<e>.mu_y`. These fields are not in `calibration.params` — they are inputs, not free SMM parameters.
6. Rerun calibration: `python calibrate.py --config calibration_input_GR.json --backend jax`. SMM operates only on the parameters listed in `calibration.params` (currently `nu`, `beta`).
7. Re-validate fiscal experiments: `python run_fiscal_figures.py --config calibration_input_GR.json --shock G`.

## Validation

- **Sample-size check.** Reject any age × education cell with `n < 30` raw observations. Smooth the profile by 5-year age bands when single-year cells are sparse.
- **Cross-wave consistency.** Plot `Var(log pi11)` by age separately by wave. If the profile shifts substantially across waves, suspect a measurement-protocol break and pool only consistent waves.
- **Italy benchmark.** Run the same pipeline on `data/it20ip.dta` and compare against the Cooper-Haan-Zhu Italian estimates (ρ ≈ 0.95, σ_η ≈ 0.13). Discrepancies indicate sample-selection or specification problems before going to Greece.
- **Internal consistency with SMM.** After updating `calibration_input_GR.json`, the model-implied `wealth_gini` and `income_gini` (now untargeted validation moments) should remain in a reasonable neighborhood of the data values (0.58 / 0.318). Large deviations relative to the current provisional calibration warrant a separate review of the income process specification before continuing.

## Output Deliverables

For each education stratum `e ∈ {low, medium, high}`:

| Parameter | Source | Notes |
|-----------|--------|-------|
| `μ_z^(e)` | Mean log `pi11` (most recent wave) | Pre-residualization |
| `σ_η^(e)` | MD on age profile, fixed ρ = 0.95 | Baseline |
| `σ_α^(e)` | MD on age profile, fixed ρ = 0.95 | Validation only — model has no fixed effect |
| `ρ_z^(e)` | Fixed at 0.95 | Sensitivity range: {0.90, 0.95, 0.97, 0.99} |

Written directly into `calibration_input_GR.json` under the per-education fields `edu_params.<e>.rho_y`, `edu_params.<e>.sigma_y`, `edu_params.<e>.mu_y`.

## Integration with SMM Calibration

Income process parameters are **pinned externally** from the LIS estimation, not calibrated internally. The SMM problem in `calibrate.py` does not include `rho_y` or `sigma_y` in `calibration.params`, and `wealth_gini` and `income_gini` are not SMM targets — they are reported as untargeted validation moments alongside the calibration output.

Rationale: ρ is weakly identified from cross-sectional moments alone, and σ_η is identified by the age profile of `Var(log earnings)`, not by wealth-distribution moments. Folding ρ and σ into the SMM loop would let wealth-Gini residuals contaminate parameters that have a clean external identification.

After the LIS values are written into the JSON, the SMM problem reduces to free parameters that are not pinned by external data — currently `nu` (Frisch disutility scale) and `beta` (discount factor in SOE), against `average_hours` and `A/Y` respectively.

## Open Issues

- **Self-employment.** Greek labor markets have substantial self-employment shares; excluding them may bias `μ_z` for the low-education stratum upward. Report self-employed earnings separately for the appendix; baseline estimation excludes them.
- **Cohort effects.** Pooling cross-sections from 2003-2021 over a period spanning the sovereign debt crisis introduces year-specific shocks that may load onto the cross-sectional variance profile. Year fixed effects in the residualization absorb the level but not interactions with age; consider an alternative specification with cohort fixed effects if the variance profile is non-monotone.
- **Top-coding.** EU-SILC in Greece may top-code high earnings. Verify in METIS documentation; if present, the high-education `σ_η` will be biased downward.
- **`grossnet` consistency.** Some Greek waves may report mixed gross/net. Verify `grossnet` is consistent across pooled waves before computing variance moments.
- **CPI deflation.** Pooled multi-wave estimation requires deflating wages to a common base year. Use Greek HICP (Eurostat) or restrict to single-wave estimation per moment.
