# Trend and population growth: baseline (g, n), comparisons g = 0 and n = 0

## Objective

Solve the G and I_g experiments in an economy on a balanced growth path with
productivity growing at g per year and population at n, and compare the
results with the same experiments at g = 0 (n unchanged) and at n = 0
(g unchanged). The baseline pair (g, n) is to be specified; the arithmetic
below uses the baseline pair g = 1.7%, n = −0.60% — the 2024 Ageing Report's
projections for Greece — with the sovereign rate at r_B = 1.9% (the implicit
rate on the stock, 2012–2024 average).
The model is written in per-capita detrended units — every aggregate divided
by Z_t·N_t, with Z_t = (1+g)^t the productivity trend and N_t the
population — so the solved equilibrium is stationary and the reported ratios
(B/Y, K_g/Y, Δτ_l) stay comparable across experiments and across growth
scenarios. Aggregate output grows at Γ − 1 with Γ = (1+g)(1+n) = 1.0109 at those
values; per-capita output at g.

There is no g = 0 baseline. The calibration (Steps 4–5) is done once, at
(g, n); the g = 0 and n = 0 economies are comparison scenarios derived from
it (Step 6), not calibrations of their own. The draft's r_B = 0 run, whose
recursions carried no population factor, is superseded.

The code sees only g and n. The write-up's level technology is
Y = A·K_g^η_g·K^α·(X_t·L)^(1−α) with X_t = Z_t^((1−α−η_g)/(1−α))·N_t^(−η_g/(1−α)),
a labour-augmenting rate of 1.62% per year at α = 0.33, η_g = 0.05; the
population term is the scale effect of aggregate public capital, which the
per-capita form absorbs.

## What growth changes

A unit saved today carries 1/(1+g) units into next period in detrended terms,
and utility from a given detrended consumption level shrinks by (1+g)^(1−γ) per
period. The household therefore faces an effective gross return of
(1+r)/(1+g) = 2.26% instead of 4% (1.38% after the capital-income tax at
τ_k = 0.2236), and an effective discount factor β(1+g)^(1−γ). With log
consumption (γ = 1, point 1) that factor is β itself, so growth reaches the
household only through the (1+g) on next-period assets, and the disutility
weight ν carries no trend. The household problem is an individual problem and
never sees n.

Aggregates are per capita: cohort weights are (1+n)^(birth year − base),
normalised to sum to one each period (`olg_transition.py:798`, `:926`), so the
population behind them grows by (1+n) per period. Stocks that accumulate
outside the household problem — sovereign debt, public capital, the pension
fund, net foreign assets — are per-capita detrended stocks and lose a factor
Γ = (1+g)(1+n) per period. Today the stock recursions carry no population
factor while `terminal_flow_balance` does; both move to Γ below, so the laws
hold for any n, including the n = −0.573% already in the config at g = 0.
The baseline n = −0.60% replaces that value (point on (g, n) below).

Detrending convention: a period-t flow is divided by Z_t·N_t; the stock carried
into t+1 is divided by Z_{t+1}·N_{t+1}. Every aggregate law of motion below is
the level law with both sides written in these units, so the whole right-hand
side (stock and flow) is divided by Γ. The household budget follows the same
rule with Z_t alone: c_t + (1+g)·a_{t+1} = (1+r̃)·a_t + y_t.

Factor prices are unaffected. The firm condition r + δ = αY/K is a flow
condition, so K/L and w are what they are now.

Scope: Steps 1–7 were written with n as the constant
`external_params.pop_growth`, and that is what the 2026-09-29 calibration
used. Step 0 below supersedes it: demography becomes a path, Γ becomes Γ_t,
and the calibration is re-anchored. Steps 1–7 are otherwise unchanged.

## Step 0 — demographic assumptions

Decided 2026-09-29, before any further calibration, so that the assumptions
are fixed once rather than revisited after results exist.

### The problem being fixed

Three places make demographic assumptions and they disagree. The calibration
uses a fixed survival vector (close to the 2023 table) and a stationary age
distribution with births growing at n = −0.60%. The transition has cohorts
walk the historical period tables 1961–2023 along their own calendar
diagonals, frozen at 2023 thereafter, with births still growing at −0.60%.
So the transition's t = 0 cross-section is not the population the SMM
matched, and the targeted moments hold in the stationary equilibrium but not
in the year the data describe. Measured on the 2026-09-29 calibration the
output gap between the two is **13.7%** (ŷ = 1.0296 against 0.8851).

The stationary population is also markedly older than Greece: at
n = −0.60% its old-age dependency ratio (65–84 over 25–64, the model's age
span) is **0.458** against **0.360** in the 2023 data. Reweighting the
calibrated panels onto the measured cross-section, with nothing else changed,
moves the moments by

| moment | stationary | data 2023 | change |
|---|---|---|---|
| average hours | 0.4099 | 0.4097 | −0.07% |
| A/Y | 3.9903 | 3.7483 | −6.3% |
| payroll revenue/Y | 0.1300 | 0.1300 | −0.01% |
| pensions/Y | 0.1601 | 0.1261 | −21.2% |
| public health/Y | 0.0541 | 0.0468 | −13.3% |

A/Y falls rather than rises because retirees hold assets but produce nothing,
so shifting weight towards working ages raises output more than wealth.

### What is assumed

1. **t = 0 is the measured 2023 cross-section.** Model ages 0–59 = real ages
   25–84, from `data/DATA_GR.xlsx`, sheet `Population by age`. An aggregate is
   Σ_j (cohort weight) × (mean over that cohort's simulated agents), and those
   means run over all agents with the dead holding zero — so mortality is
   already in the mean and the weight must be the cohort's size **at birth**.
   The cross-section counts the living, so each age is divided by its
   cumulative survival probability to recover the birth cohort; using the
   observed counts directly would apply survival twice.
2. **The calibration targets that same cross-section**, not a stationary
   population. `compute_age_weights` takes the measured vector in place of
   ω_j ∝ (1+n)^(−j)·S_j. Nothing else in the solve changes, because **with the
   tax rates held fixed** the age distribution enters no household's problem:
   r is exogenous, K/L is pinned by the firm FOC and w with it, so individual
   policies are invariant to it and each SMM evaluation stays a single
   stationary solve. The one other channel that could bite, the
   accidental-bequest transfer, is an open circuit in the calibration.
   This invariance holds in the calibration and in the baseline, where the
   instruments are fixed. It does **not** hold in the τ_l-financed
   experiments: ageing raises pension and health spending, τ_l rises to meet
   the debt or NFA target, and household decisions move with it. That is the
   channel through which demographics reach behaviour, and it is a result of
   the exercise rather than a nuisance.
3. **The closure is pinned on the same weights as the calibration.**
   `pin_baseline_closure` computes aggregate ratios; if the two use different
   weightings the primary balance will not equal its target.
4. **Survival and entering cohorts follow EUROPOP2023 through 2100**,
   identical across the baseline and every counterfactual. Downloaded
   2026-09-30 by `code/build_europop_GR.py` into `data/europop2023_GR.npz`,
   Eurostat baseline variant for Greece: population on 1 January by single
   year of age and sex, assumed age-specific mortality rates by sex, assumed
   net migration by age. No projected life table is published, so survival is
   built from the mortality rates — the two sexes combined at the projected
   population weights of that age and year, then q = m/(1 + m/2). That
   reproduces the observed 2023 `demo_mlifetable` survival, the source of
   `data/survival_GR.npz`, to 2.3e-3 at worst, so the projected table
   continues the historical one on one definition.
5. **Net migration is routed through the entry age.** The model creates
   cohorts only at real age 25, so it cannot receive a 50-year-old arrival.
   Each year's entering cohort is instead solved as the residual that makes
   the model's 25–84 population equal the projection's,
   B_y = P_y − Σ_{j≥1} B_{y−j}·S_j, with the cohorts already alive in 2023
   pinned by the measured cross-section divided by cumulative survival. The
   aggregate population then matches EUROPOP2023 exactly and every lifecycle
   history stays intact. What it costs is age composition: EUROPOP2023 has
   Greece losing about 6% of a cohort to net emigration in its first decade
   and 10% by age 55, and booking those losses at entry makes cohorts
   entering before 2050 smaller than the projection's 25-year-olds (−16.2% in
   2030) and later ones larger (+19.3% in 2083). The old-age dependency ratio,
   65–84 over 25–64, then runs above the published path where ageing peaks and
   below it afterwards: 0.685 against 0.657 in 2050, 0.638 against 0.589 in
   2060, 0.473 against 0.529 in 2083.
6. **The terminal state is a genuine balanced growth path**, reached by an
   assumed tail past the projection. `terminal_debt_gdp`, the NFA target, the
   `K_g_ss_gap` diagnostic and the rest point PD/Y = (Γ_T − 1 − r_B)·b all
   require demography to have converged, and EUROPOP2023 has not converged at
   its horizon — the dependency ratio still swings with the baby-boom cohorts
   (0.657 in 2050, 0.494 in 2070, 0.529 in 2083, 0.495 in 2100), and the
   projection's own 25-year-old counts grow at −0.75%/yr in the 2070s, −0.22%
   in the 2080s and +0.13% in the 2090s. So: mortality is held at the 2100
   schedule; the growth rate of entering cohorts is ramped linearly to
   **n_∞ = 0.00%** over 2100–2120 and held there. The ramp starts from the
   entering series of point 5, which absorbs net migration and so grows at
   +0.93%/yr over the 2090s, not from the projection's 25-year-olds; the population reaches its stable age
   distribution about a lifetime later, by 2180. That sets
   **T_transition ≈ 157** against 60 today, so cohort solves go from 120 to
   about 217 per education type.

   n_∞ = 0 is an assumption, not a reading: entering-cohort growth is still
   swinging at 2100. It is the most consequential number in Step 0, because
   r_B − (Γ_T − 1) multiplies debt in every terminal condition — +0.20pp at
   n_∞ = 0 against +0.81pp at the −0.60% previously assumed and +0.98pp at the
   25–84 band's own 2023–2070 rate of −0.77%.

### What follows

- **The calendar anchor moves to 2023.** `transition.current_year` was 2020
  while t = 0 is the 2023 cross-section. Transition period t is calendar year
  `current_year + t`, and `_survival_schedule_at_year` maps the internal clock
  to a true year through `current_year − birth_year` before selecting a period
  life table, so every cohort was reading the table of three years earlier —
  and the oldest cohort alive at t = 0 sat on the 1961 clamp. Set to 2023
  (`calibration_input_GR.json`). The population weights are unaffected, being
  births-only and renormalised each period; what changes is the survival each
  cohort faces, so the moments move and the fit is superseded. The three
  hardcoded synthetic branches (`olg_transition.py:2676`, `:2794`,
  `run_fiscal_figures.py:162`) keep their own anchors — they carry no survival
  data file and use the legacy improvement path.
- **Γ becomes Γ_t = (1+g)(1+n_t)** and every stock recursion takes the time
  index rather than the scalar `olg.growth_factor`: the debt, K_g, pension
  fund and current-account laws, `new_borrowing`, `K_g_ss`, and the
  terminal-rest-point condition, which uses **Γ_T = (1+g)(1+n_∞)**.
  n_t is the growth rate of the model's own population, ages 25–84, computed
  from the same weights the aggregation uses — not the total-population rate.
  The two differ materially: EUROPOP2023 has total population falling 0.60%
  a year to 2070 while the 25–84 band falls 0.774%. The config's `pop_growth`
  therefore stops being the population growth rate of the model and becomes
  n_∞ alone, the terminal value; between 2023 and 2120 n_t comes from the
  data.
- **"Steady state" is retired** in favour of *base-year equilibrium*: one
  lifecycle problem at constant detrended prices, aggregated over the 2023
  cross-section, with ŷ = 1 a units normalisation of base-year output. With
  growth no level was ever stationary, and with a non-stationary population
  not even the detrended aggregate is. The term goes from
  `normalize_A_tfp.py`, `pin_baseline_closure.py`, this plan and the report.
- **The baseline is a demographic transition from t = 0**, with no policy
  change. Detrended aggregates move while the 40–64 cohorts retire and settle
  only once demography has settled, so the flatness check in the report is
  computed over the years after that.
- **Recalibration is required.** ρ_pens and m_good move most — both rise, by
  roughly a quarter and a seventh — and ρ_pens moving up from 0.186 is a
  gain, since that value is low for Greece precisely because it compensates
  for a model population with 46% more retirees per worker than the data.
  β rises slightly; ν and τ_p barely move.

### What is knowingly approximated

- Lifecycle asset profiles come from a problem solved at constant prices, so
  a 60-year-old at t = 0 did not live through the crisis. Fixing that would
  mean solving pre-2023 cohorts against historical price paths.
- Migrants carry no history of their own. Routing net migration through the
  entry age keeps every lifecycle intact but books a departure at 40 as a
  smaller cohort at 25, so the model's age composition departs from the
  published one by the amounts in point 5. Representing arrivals and
  departures at the age they occur would need agents created after age 25,
  with their own initial wealth, income state and partial pension entitlement
  — cohorts are currently created only at model age 0 and
  `initial_asset_distribution` applies at entry.
- A stationary-population device would have matched the dependency ratio but
  not the age *shape*: Greece has 0.507 of its 25–84 population aged 40–64
  against 0.445 in a stationary population matched on that ratio. Taking the
  measured cross-section removes this approximation, which is the reason for
  preferring it.
- The initial condition's coherence rests on the small-open-economy closure
  *and* on the instruments being fixed at t = 0. In a closed economy K/L would
  respond to the age structure, so a non-stationary initial population would
  make the constant pre-transition prices behind the MIT stitching internally
  inconsistent. Worth stating in the write-up.

## Step 1 — config and wiring

Config (`calibration_input_GR.json` is the baseline; the comparison
scenarios are derived configs, Step 6; `calibration_input_GR_rB0.json` is
retired):

- Add `trend_growth: 0.017` to `external_params`, and set
  `external_params.pop_growth = -0.006`. `calibrate.py:1044` assigns
  `external_params` keys to `LifecycleConfig` fields by name, so a
  `trend_growth: float = 0.0` field on `LifecycleConfig` picks it up.
- Set `model.gamma = 1.0` (log consumption, decided in point 1). Both
  backends already branch to log at γ = 1; no other code change.
- Set `prices.r_B = 0.019` — the implicit rate on the stock averaged over
  2012–2024, the period of the official-sector composition (point 4).
- `delta_g` per Step 4.

One source for each rate: g lives on `LifecycleConfig.trend_growth`; n
already lives on `OLGTransition.pop_growth` (the weights' source). Γ is
computed once, `OLGTransition.growth_factor = (1 + trend_growth) * (1 + pop_growth)`,
and the fiscal layer reads it from the `olg` object, so a harness or test that
builds the objects directly cannot solve households at one g and stocks at
another.

Population factor: the transition's cohort sizes are built as
`np.exp(pop_growth * years_since_base)` in `_cohort_sizes_njit`
(`olg_transition.py:810`) and `set_cohort_sizes_path_from_pop_growth`
(`:922`), while the steady-state weights use `(1 + pop_growth) ** (-t)`
(`calibrate.py:940`). Both transition lines become
`(1.0 + pop_growth) ** years_since_base`, so the weights, the steady state
and the recursions share one factor. The change moves the transition's age
weights by (1+n)^t/e^(nt) − 1 ≈ −1.81e-5·t at n = −0.60%, i.e. 1.1e-3 at the
oldest model age (T = 60). Only `:810` is on the live path — the fiscal
pipeline never passes `pop_growth_path`, so `:922` is dormant — but both
change for consistency.

NumPy household solver
- `LifecycleModelPerfectForesight.__init__` (`lifecycle_perfect_foresight.py:297`
  block): `self.trend_growth = config.trend_growth`.
- `_solve_state_choice` (`:915`): uses it in the budget (Step 2).

JAX household solver
- `LifecycleModelJAX.__init__` (`lifecycle_jax.py:1017` block):
  `self.trend_growth`; the `solve()` call at `:1118` passes it.
- `solve_lifecycle_jax` signature (`:424`, default `trend_growth=0.0` next to
  `nu=1.0` at `:445`) and the `model_params` tuple it builds at `:498–503`.
- `solve_period_jax`: docstring and unpack at `:219`/`:239`, budget at Step 2.
- `_solve_lifecycle_jax_batched` `in_axes` list (`:605–624`): one more `None`
  entry (shared scalar), at the same position as in the signature.
- `_solve_cohorts_jax_batched` call site (`olg_transition.py:482–499`, 39
  positional arguments today): pass `ref.trend_growth` at the same position.
  This is the only JAX path `OLGTransition` uses; the SMM uses the
  single-cohort path.
- Index constraint: `lifecycle_jax.py:564` rebuilds the per-period params as
  `model_params[:3] + (m_grid_t,) + model_params[4:]`, an index-based slice.
  `trend_growth` must be inserted at index ≥ 4 — next to `nu` it is ~31, which
  is safe — or it silently displaces `m_grid`.

Per-cohort configs and the MIT-stitching baseline model are built with
`lifecycle_config._replace(...)` (`olg_transition.py:1155`, `:1205`), which is
`dataclasses.replace`, so the field reaches them without an entry in
`_feature_kwargs`. The SMM path (`run_model_moments`), `normalize_A_tfp.py` and
`pin_baseline_closure.py` go through `build_lifecycle_config` and need no
change.

OLG transition
- `OLGTransition.__init__` (`olg_transition.py:91`), after `self.lifecycle_config`
  is set (`:144–148`, which falls back to a default `LifecycleConfig()` when the
  argument is None — six test sites rely on that):
  `self.trend_growth = float(getattr(self.lifecycle_config, 'trend_growth', 0.0))`
  — read `self.lifecycle_config`, not the argument — and
  `self.growth_factor = (1 + self.trend_growth) * (1 + self.pop_growth)`.
  Used by the K_g and pension-fund recursions (Step 3). No new constructor
  argument, so `build_olg_transition` (`calibrate.py:1133`) is unchanged.
- `get_test_config` (`olg_transition.py:2612`) — production code imported by
  the test suite and used in ~25 `OLGTransition` constructions — sets
  `trend_growth` for the g > 0 tests. It already sets `gamma = 1.0`.

Fiscal block. `run_fiscal_scenario` (`fiscal_experiments.py:1125`) is only
the dispatcher — none of the calls below is inside it — so Γ is read
separately in each of the three experiment functions, all of which take `olg`:

- `run_debt_financed` (`:666`), `run_tax_financed` (`:734`) and
  `run_nfa_constrained` (`:937`): `Γ = olg.growth_factor` at the top, used by
  their own calls and closed over by their nested helpers
  (`_simulate_and_residual` `:773`, `_objective` `:802`, `_full_nfa_ca` `:970`,
  `_nfa_ok_at` `:1036`).
- `_correct_base_macro_nfa` (`:617`) calls `compute_debt_path` at `:630` and
  has **no `olg` in scope**: it takes a new `growth_factor` parameter, and its
  three callers (`:702`, `:902`, `:1093`) pass it.
- `_check_terminal_convergence` (`:338`) takes `olg`, so the `K_g_ss` fix at
  `:417` needs nothing extra.
- The six `compute_debt_path` calls (`:630`, `:690`, `:783`, `:964`, `:974`,
  `:1081`), the five `_nfa_ca_paths` calls (`:700`, `:906`, `:968`, `:978`,
  `:1091`) and the one `_balance_residual` call (`:787`) pass Γ.
- **All three helper signatures default `growth_factor = 1.0`**, so the
  existing direct-call tests (`test_fiscal_experiments.py:167–181`, `:269`,
  `:276`, `:287`, `:297`, `:306`) keep passing unchanged.
- `FiscalScenario.pop_growth` (`fiscal_experiments.py:120`) and its comment
  (`:111`) are removed, with the four `pop_growth=` kwargs in
  `run_fiscal_figures.py`; no `trend_growth` field is added.
- `run_fiscal_figures.py:518` (`params_out`): record `trend_growth`,
  `pop_growth` and `delta_g` in the JSON, next to `r_B`.
- `eval_fiscal_results.py:590–609`: read `trend_growth` and `pop_growth` from
  the JSON `params` first and from `--config` `external_params` as fallback
  (the same precedence as `r_B`; the `production` loop at `:603` does not see
  `external_params`); pass Γ to `chk_debt_accumulation` (`:149`).
- `eval_fiscal_results.py:492–504`: the `terminal_flow_balance` check tests
  `|PD[T-1]/Y[T-1]| < BISECT_TOL`, i.e. against zero. The rest point is now
  `(Γ − 1 − r_B)·b` = −1.33% of Y at the baseline, so the check reports a
  false violation for any run using that condition.
- `regen_fiscal_figures_from_json.py:166`: `_nfa_ca_paths` with Γ from the
  JSON `params` (or flags, as `--r-b` does for the interest line).
- `check_a0_predetermination.py:31`: `trend_growth` on the `LifecycleConfig`;
  `:50`: the Ig-case I_g level becomes `(0.05 + Γ − 1)·0.745` with the
  harness's own `pop_growth`.

## Step 2 — household budget constraint

Replace `budget - a_next` with `budget - (1 + g) * a_next` at:

- `lifecycle_perfect_foresight.py:949` and `:956`
- `lifecycle_jax.py:273` and `:294`

The terminal-age blocks (`lifecycle_perfect_foresight.py:929`,
`lifecycle_jax.py:397` and `:414`) hold no next-period assets and stay as they
are. `_solve_labor_newton` and `solve_labor_robust_jax` take `c_guess` from
the modified budget and adjust linearly in l, so they need no change. The
simulations read the `a_policy` grid index and set `a_sim = a_grid[idx]`
(`lifecycle_perfect_foresight.py:1146`, `lifecycle_jax.py:680`) and read
consumption from `c_policy`, so they need no change and apply no second factor.

The discount factor needs no code change: `beta` multiplies the continuation
value directly (`lifecycle_perfect_foresight.py:985`), so the calibrated β is
the effective discount factor β(1+g)^(1−γ) — equal to β at the chosen γ = 1 —
and SMM estimates it directly.

Utility is u(c, l) = log c − ν·l^(1+φ)/(1+φ) (point 1), so ν is a constant in
levels and in detrended units alike and needs no trend.

Level objects inside the household problem — medical costs, the minimum pension
floor, the means-tested transfer floor, child costs, the bequest lump sum — are
already stationary in detrended units and take no growth factor. This makes
each of them grow with the trend in levels, which is the intended reading for
all five. One consequence to state where the health-flag work is cited:
indexing `m_good` to the trend holds government health spending flat as a
share of Y along the baseline, which is the opposite of the drift that
decomposition measures. UI is proportional to w and needs nothing. The bequest pool at t is
paid to the newborns of the same period t (`olg_transition.py:1817–1830`), per
capita within the period, so it carries no growth or population factor either.
The HSV progressive schedule (`tax_progressive`, off in the GR config) is not
homogeneous; if it is ever switched on, `tax_kappa` must scale as Z_t^η.

## Step 3 — stock accumulation

Γ = (1+g)(1+n) throughout.

| Object | File | New law of motion |
|---|---|---|
| Sovereign debt | `compute_debt_path`, `fiscal_experiments.py:264` | `B[t+1] = ((1+r_B)·B[t] + PD[t]) / Γ` |
| Public capital | `olg_transition.py:1918` | `K_g[t] = ((1−δ_g)·K_g[t−1] + I_g[t−1]) / Γ` |
| Pension fund | `olg_transition.py:2241` | `S[t+1] = ((1+r)·S[t] + tax_p[t] − pension[t]) / Γ` |
| Current account | `_nfa_ca_paths`, `fiscal_experiments.py:613` | `CA[t] = Γ·NFA[t+1] − NFA[t]` |

Follow-ons:

- `olg_transition.py:1771`: `new_borrowing = Γ·B_next − B_t`, which equals
  `PD[t] + r_B·B[t]` under the debt law. The line is live only when
  `olg.B_path` is set — the fiscal pipeline never sets it (B is computed after
  the budget in `run_fiscal_scenario`), so `debt_service`, `new_borrowing` and
  `fiscal_deficit` are zero in every reported run; the identity is exercised
  by `test_olg_transition.py:1543–1557` and the CLI.
- `eval_fiscal_results.py:160` recomputes the debt recursion to check it.
  Mirror the new form or the check reports a false violation.
- `fiscal_experiments.py:613`: the appended terminal value of `CA` is
  `(Γ − 1)·NFA[-1]` (the last period repeats the last NFA level), not 0.
- `fiscal_experiments.py:322` (`terminal_flow_balance`) becomes
  `PD/Y = (Γ − 1 − r_B) · b`. The runs in `run_fiscal_figures.py` select the
  terminal debt and NFA targets, which are ratio targets and need no change.
- `fiscal_experiments.py:417` (`K_g_ss_gap` diagnostic):
  `K_g_ss = I_g_T / (delta_g + Γ − 1)`.
- Existing tests assert the recursions without a population factor:
  `test_fiscal_experiments.py:158–165` and `:318–329` (debt), `:497–504`
  (terminal CA = 0), `test_olg_transition.py:1493–1511` (K_g),
  `:1543–1557` (`new_borrowing`), `:1615–1632` (pension fund). Their fixtures
  (`_make_olg`, `test_public_capital_accumulation`) build `OLGTransition`
  without `pop_growth`, i.e. at the constructor default 0.01, so they fail
  after the change and are rewritten to the Γ form — they then exercise n ≠ 0
  at g = 0 for free.

The population factor alone already changes the recursions at the config's
n = −0.60%: with I_g = δ_g·K_g, per-capita K_g rises 0.60% per year instead
of staying flat, and per-capita debt accumulates by the same factor. The
δ_g in Step 4 restores a stationary baseline.

## Step 4 — parameters re-identified from the same data

`delta_g = 0.047383` in the config is the data ratio I_g/K_g = 0.0353/0.745.
Aggregate K_g grows at Γ − 1, so that ratio identifies δ_g + Γ − 1. At the
baseline (g, n) = (1.7%, −0.60%): Γ − 1 = 0.010898, `delta_g = 0.036485`.
δ_g is a structural parameter and is identified once, at the baseline; the
comparison scenarios keep it (Step 6).

Baseline I_g/Y then stays at 3.53% and the long-run public capital response
to a given ΔI_g is unchanged. Holding δ_g at 0.047383 would raise baseline
I_g/Y to 4.34% and cut the long-run K_g response by 19%.

Stock-flow timing: `fiscal.B_over_Y = 1.64` is end-2023 debt over 2023 GDP,
while `B_initial = B_over_Y·Y0` (`run_fiscal_figures.py:118`) reads it as a
start-of-period stock over the same period's output, and
`_balance_residual`'s `terminal_debt_gdp` (`fiscal_experiments.py:300–303`)
forms `B[T_bal]/Y[T_bal−1]`. At Γ ≠ 1 the reported terminal B/Y is therefore
about 1% below the level ratio. It does not bias the τ_l experiments — the
target is pinned off the baseline with the identical convention
(`run_fiscal_figures.py:324`) — but the write-up should say which convention
the reported ratios use. The same applies to K_g/Y = 0.745 and the NFA/Y
target.

Notation: `Y_ss` throughout means **detrended per-capita** output
ŷ = Y_t/(Z_t·N_t), the object the stationary solve returns. With growth there
is no stationary level of output — levels grow at Γ − 1 and per-capita terms
at g — so normalising ŷ = 1 is a units choice, and it is what makes K_g = 0.745
equal K_g/Y and I_g = 0.0353 equal I_g/Y.

These ratios are in steady-state units (ŷ = 1). The transition's output
level is ≈0.885·Y_ss (the SS-vs-transition normalisation gap on record), so
along the baseline transition I_g/Y ≈ 4.0% and K_g/Y ≈ 0.84, at g = 0 already.

The stationary public-investment level is `(delta_g + Γ − 1) · K_g`. It is
coded as `delta_g * K_g` in five places, each of which changes (Γ from
`economy.growth_factor` where an `OLGTransition` exists, else from the config's
`trend_growth` and `pop_growth`):

- `run_fiscal_figures.py:95` (`I_g_warmup`, which is also the baseline I_g
  level path with `eta_g != 0`; the print at `:125` too)
- `run_fiscal_figures.py:170` — the non-`--config` test branch writes the
  literal `np.full(T_TR, 0.05 * 1.0)` on an economy built at `pop_growth=0.02`,
  so it does not show in a `delta_g` grep and drifts under the new K_g law
- `validate_backends.py:92`
- `diag_ss_vs_transition.py:76`
- `diag_bequest_decomp.py:75`
- `check_a0_predetermination.py:50`

Without this, baseline I_g/Y falls to 2.72% and K_g drifts along the
baseline transition.

Private capital: `delta = 0.05` is a standard value, not identified from I/K
(`data_inventory.md:37` makes the "standard, not identified" point but quotes
0.07, not the config's 0.05), so it stays. In steady state private investment is
(δ + Γ − 1)·K; nothing in the pipeline computes it, but the write-up's
resource constraint carries it.

The steady-state interest line becomes r_B·B/Y = 3.12% of Y at r_B = 1.9%
against 3.39% in 2023. The config's B/Y = 1.64 is itself the 2023 ratio
(1.6428), so the whole 0.27pp gap is the rate: the 2023 implicit rate was
2.06% and the baseline uses the 2012–2024 average of 1.9%.
`fiscal.interest_over_Y = 0.034` is a validation entry carrying the 2023
value; set it to 0.0312 or note the gap.
Nothing else in the fiscal block moves — `primary_balance_over_Y` in `calibrate.py:1291`
nets interest out, so the closure residual is identified exactly as before;
its value moves with g through the household panels, which Step 5 re-pins.
With the closure pinned at the data primary surplus of 1.95% of Y and
r_B = 1.9%, baseline B/Y falls by ≈0.62pp per year along the transition,
reaching 1.17 after sixty periods: the 1.95% surplus exceeds the 1.33% that
holds the ratio constant. Both figures assume a primary surplus of exactly
1.95% at every t and a flat detrended Y; `pin_baseline_closure.py` pins the
*initial steady state* and the transition starts from Y(0) ≈ 0.885, so they
are arithmetic, not model output. The superseded r_B = 0 run, without the population
factor, fell 1.95pp per year.

## Step 5 — recalibration

Run `bash run_scale_loop.sh` on the V100 instance (needs ≥32GB RAM). Each
round warm-starts the SMM initials from `_derived.theta`, runs the SMM for
θ = (ν, β, τ_p, ρ_pens, m_good), and runs `normalize_A_tfp.py --write` to
restore Y_ss = 1; after the last round it runs `pin_baseline_closure.py
--write`. `calibrate.py` writes `_derived.theta` itself on convergence, and the
closure must be pinned after that write. The steady state is per capita and
carries no stock recursion, so n enters the calibration only through the age
weights, as today.

- Seed for β. Detrended consumption growth is β·s·(1+r̃)/(1+g) at γ = 1,
  against [β·s·(1+r̃)]^(1/2) at γ = 2, so β_old·(1+g) does **not** preserve the
  profile once γ moves: at β = 0.9579 it implies consumption falling 2.87% a
  year against 1.45% in the current fit. The slope-preserving value is
  **β = 0.972** — (1+g)·[β_old(1+r̃)]^(1/2)/(1+r̃) with r̃ = r(1−τ_k) = 3.106%
  — and flat detrended consumption at s = 1 would need 0.986, so the A/Y = 4
  root is plausibly in 0.97–1.00. Seed 0.972. At γ = 1 the code's β is the
  structural β, so no separate effective/structural figure is reported. Write
  the seed into `_derived.theta.beta` — `run_scale_loop.sh` copies the
  initials from there and `build_lifecycle_config` (`calibrate.py:1060–1075`)
  overrides `model.beta` with it; the `initial` values already in the config
  are dead. Bounds [0.9, 1.1] do not bind.
- Seed for ν. The warm start carries ν = 20.66, fitted at γ = 2. The FOC is
  ν·l^φ = c^(−γ)·MW/(1+τ_c), so at γ = 1 the right-hand side falls by a factor
  c < 1 in Y_ss = 1 units and holding hours at 0.41 needs ν down by roughly
  that factor — order 10–12. Seed it there rather than at 20.66; the bound
  [0.1, 200] is not at risk, the cost of not doing it is scale-loop rounds.
  (Inference from the FOC, not solved.)
- `calibrate.py` writes θ only on scipy convergence, and the loop aborts
  otherwise (`run_scale_loop.sh:39–42`). Nelder-Mead runs on logit-transformed
  parameters with maxiter 500 and xatol = fatol = 1e-6; from a moved seed on a
  jagged surface (100-point asset grid, discrete a′) exhausting maxiter and
  aborting at round 1 is the likely first outcome. Mitigations that exist:
  `--tol` and `--maxiter` (`calibrate.py:1537`, `:1540`) and
  `--method differential_evolution`; or write a validated θ by hand.
- The A_tfp leg should get *easier* at γ = 1, not harder. With log consumption
  and separable hours the intratemporal FOC is invariant to a proportional
  scaling of w and c, so Y_ss is near-homogeneous of degree 1/(1−α) in A_tfp
  and `normalize_A_tfp.py`'s elasticity step is near-exact — its docstring's
  "Y_ss is NOT proportional to A_tfp" is a γ = 2 statement. Exactness breaks
  only through the level objects that do not scale (`pension_min_floor` = 0.15,
  child costs, the bequest lump sum). Budget more rounds for the SMM leg, not
  for this one.
- Numerical resolution, not the grid top, is the binding constraint when β and
  the IES both move: `_create_asset_grid` puts **8 of 100 points below a = 4**
  (the A/Y target at Y_ss = 1) with local spacing 0.834, 21% of the level.
- No wall-clock estimate per round is on record; the July K_g activation is
  the only comparable run.
- The Aitken step for the SMM ↔ A_tfp sequence is manual: after three rounds,
  Δ²-extrapolate the per-round `A_tfp` values and write the result into
  `production.A_tfp` before the next round.
- K_g = 0.745 is a K_g/Y target only at Y_ss = 1; the normalisation restores
  that. δ_g + Γ − 1 is ratio-identified, so g adds no new coupling to the loop.

## Step 6 — experiments and comparison scenarios

Baseline (g, n): run G and I_g, debt-financed and τ_l-financed, into a fresh
output directory; then `eval_fiscal_results.py` and the figures.

Comparison scenarios, five derived configs: (0, n) and (g, 0); the
held-spread variant (iii) below; and the two alternative growth pairs —
(1.87%, −0.53%), the 2013–2024 data window, as high growth, and
(1.06%, −0.05%), the 1995–2024 per-capita trend, as low growth. Each
is a derived config — a copy of the baseline config with `trend_growth` or
`pop_growth` overridden and every structural parameter held: θ (ν̄, τ_p,
ρ_pens, m_good), δ_g, δ, A_tfp, K_g, `other_net_spending_over_Y`, r, r_B and
the tax rates. No SMM, A_tfp normalisation or closure re-pin is run for them;
they are the baseline's structural economy at a different growth rate, not
economies fitted to the data (point 5 on the alternatives). **What n = 0 means now that n is a data path.** The scenario holds the age
structure at its measured 2023 shape for the whole transition: the number of
people of each age is constant at its 2023 value, total population is
constant, and Γ_t = 1 + g at every t. Survival is held at the 2023 life table
as well, since the shape cannot stay fixed while mortality improves. In the
model's weights this is the t = 0 vector — the 2023 cross-section divided by
cumulative survival — repeated at every period, so it costs one line. Two
things to say about it in the write-up. It is not the stationary population
that the 2023 life table generates on its own, which is older than Greece's
actual 2023 shape, and holding the measured shape fixed while cohorts die at
the 2023 rates needs an age-specific inflow, so it is not produced by an
entering cohort and survival alone. And because the baseline's terminal n_∞ is
also zero, the baseline and this scenario share Γ_T and therefore the same
terminal rest point; they differ only along the path, which is what the
comparison isolates.

Two quantities
are not structural and follow from the scenario's g and n:

- The code's β is β_eff = β·(1+g)^(1−γ). With the structural β held, the
  derived config's `_derived.theta.beta` is
  β_eff,base·((1+g')/(1+g))^(1−γ) — nil at the chosen γ = 1, so the derived
  configs keep the baseline β; the Change-count table's note to the contrary
  is superseded.
  `build_lifecycle_config` overrides `model.beta` with `_derived.theta.beta`,
  so that is the field to set. ν̄ is the detrended constant and does not
  change.
- The stationary public-investment level (δ_g + Γ' − 1)·K_g from the
  scenario's Γ'; the five sites in Step 4 compute it from the config, so
  nothing is hand-edited except `check_a0_predetermination.py:50`, which
  reads no config. K_g stays at its level, so the stationary I_g *level*
  differs: 2.27% of Y_ss at (0, n), 3.98% at (g, 0), against 3.53% at the
  baseline (these are levels in Y_ss = 1 units, not ratios to each scenario's
  own Y).

Two confounds to report alongside the results rather than leave implicit:

- **The closure is held but not re-pinned.** `other_net_spending_over_Y` was
  pinned so the baseline SS primary surplus is 1.95% of Y against an I_g bill
  of 3.53%. Holding it while the I_g bill moves puts the scenarios' surplus at
  ≈3.21% at (0, n) and ≈1.50% at (g, 0). On the same flat-Y arithmetic as
  Step 4 that is +0.7pp/yr of B/Y at (0, n) and −1.0pp/yr at (g, 0) against
  −0.62pp/yr at the baseline — so a large part of the cross-scenario debt
  difference comes from the public-investment line, not from the growth
  denominator. Report each scenario's primary surplus.
- **K_g/Y is not held.** With r, A_tfp, α, δ and the K_g *level* fixed,
  `K_over_L` and ŵ are identical across scenarios, so ŷ is proportional to
  aggregate hours alone — the whole cross-scenario difference in Y_ss runs
  through the labour-supply response to g, not through savings (K_domestic is
  demand-determined). K_g/Y and MPK_g = η_g·ŷ/k̂_g therefore differ, and the
  I_g experiment is not like-for-like across scenarios.

Y_ss is not 1 in the comparison scenarios (A_tfp is held while savings and
hours move with g), so K_g/Y and B/Y at t = 0 are read off each scenario's
own Y; `B_initial = B_over_Y·Y0` already scales. Ratios are compared across
scenarios, levels are not.

Each scenario's JSON records its g, n and δ_g (Step 1), so
`eval_fiscal_results.py` checks each at its own Γ.

## Step 7 — checks and tests

Unit tests at g = 0.017 and at the fixtures' n (the existing suite runs at the
default g = 0 and must stay green after the Step 3 rewrites). The fixtures
`test_fiscal_experiments._make_olg` and `test_olg_transition.get_test_config`
set `trend_growth` on the `LifecycleConfig`.

- Household isomorphism (exact, both backends). The growth problem at
  (r, τ_k, grid G, g) has the same `a_policy` indices, `c_policy`, `l_policy`
  and V as the no-growth problem at trend_growth = 0 on the grid (1+g)·G
  (`a_min`, `a_max` both scaled; the grid is `a_min + (a_max − a_min)·x^1.5`,
  `lifecycle_perfect_foresight.py:410–422`) with a return r̃ solving
  1 + r̃(1−τ_k) = (1 + r(1−τ_k))/(1+g), i.e. r̃ = 0.017801 at r = 0.04,
  τ_k = 0.2236, g = 1.7%. Every level object is unchanged between the two, so the
  policies coincide to machine precision. Catches a missing factor, a factor on
  the wrong side, or a factor on current rather than next assets.
- Backend agreement at g = 0.017 through `OLGTransition.solve_cohort_problems`,
  comparing `c_policy_alpha` and V, **not** the policy indices: a dropped
  (1+g) moves the price of a′ by 0.064 at a′ = 4 against a local grid spacing
  of 0.834, so index equality survives at most states, while c moves ~0.054,
  8–10% of per-capita consumption. Or `validate_backends.py` at g > 0. This is the only test that reaches the batched call site
  `olg_transition.py:482–497`; a `trend_growth` dropped there leaves the SMM at
  g > 0 and every transition at g = 0, and nothing else below fires.
- `diag_ss_vs_transition.py` at g > 0: SS ratios and transition t = 0 ratios
  agree to the recorded ±0.6%. Same detector, at the aggregate level.
- K_g: with I_g = (δ_g + Γ − 1)·K_g_initial and n ≠ 0,
  `K_g[t] == K_g_initial` at machine precision for all t (level, not
  K_g/Y = 0.745 — see Step 4 on units). The test must build I_g from its own
  literal Γ, not from `olg.growth_factor`: if it reads the same object the
  recursion divides by, it passes for any Γ, right or wrong.
- Debt: `compute_debt_path` with PD = (Γ − 1 − r_B)·B[0] = −0.01329·B[0]
  returns a constant B at n ≠ 0. (The n = 0, r_B = g, PD = 0 variant is
  degenerate — it reduces to (1+g)B/(1+g) = B and cannot distinguish Γ from
  (1+g), nor a missing Γ from a missing r_B.)
- Current account: `_nfa_ca_paths` returns `Γ·NFA[t+1] − NFA[t]` and
  `(Γ − 1)·NFA[-1]` in the last period.
- Budget: `new_borrowing[t] = Γ·B[t+1] − B[t] = PD[t] + r_B·B[t]` with
  `olg.B_path` set.
- On the produced JSON: `Γ·B[t+1] − (1+r_B)·B[t] − PD[t] = 0` at every t
  (the r_B = 0 run verified its analogue at 5e-16). This is a round-trip on
  the written paths, not a check of the choice of Γ — as is the mirrored
  recomputation in `eval_fiscal_results.py:160`.
- `check_a0_predetermination.py`: A[0] identical across scenarios. Kept as a
  regression; it passes whether or not g is threaded (both runs at the same
  g), so it does not test growth.
- Asset grid: no script reports it. Add to `generate_report`
  (`calibrate.py:1328`, reading `result['panels']` and the education shares —
  5–8 lines, not 2) the share of agents at `a_grid[-1]`, `max(a_sim)/a_max`,
  **and the number of grid points below the mean** — the top of the grid is
  implausible at a_max = 200 against mean assets ≈4, while resolution near the
  mean is the real risk.
- Detrended baseline flatness: with the transition run at the baseline
  policy, Y, A, C and L must be constant over t to the simulation's Monte
  Carlo noise. This is the cheapest single detector of a growth factor on the
  wrong side, and the one test that couples the household (1+g) to the
  aggregate Γ at the aggregate level. Caveat: the data survival table
  (`transition.survival_data_file`) is active, so each cohort walks its own
  calendar diagonal and the first lifetime carries a genuine transient —
  compare against a run with the table off, or test flatness after t = T.
- Pension fund: `S_pens` with r_B = 0 and tax_p = pension returns
  `S[t+1] = S[t]/Γ`. No test covers this recursion today.
- Age weights: after the `exp(n·t)` → `(1+n)^t` change, the transition's
  period-0 weights must equal `calibrate.py:compute_age_weights` to machine
  precision at the same n (`diag_norm_check.py` is the existing harness).
- Aggregate identity, per capita detrended:
  y + r·nfa + (r − r_B)·b = c + m + [Γ·k_dom' − (1−δ)k_dom] + G + I_g
  + [Γ·nfa' − nfa]. Note the investment bracket is built on `K_domestic`, not
  on household wealth A, and the `other_net_spending` line does **not** appear
  — it is a net fiscal residual, not goods-market absorption. Two known wedges
  must be entered explicitly rather than treated as reasons the identity
  cannot hold: L carries UI/w, so Y and K are 1.887% above their model
  definitions, and the bequest circuit is open in the fiscal runs (3.77% of Y
  leaves per period). The (r − r_B)·b term is already on the left-hand side,
  so the r-versus-r_B spread is not a wedge. With those two entered it is a
  levels test, and it is the only instrument that couples the household (1+g)
  to the aggregate Γ.

## Change count

Lines touched by Steps 1–7, per file (added / modified / removed). Code counts
are line-exact from the sites cited above; test and doc counts are estimates.

| File | Added | Modified | Removed | Sites |
|---|---|---|---|---|
| `lifecycle_perfect_foresight.py` | 2 | 2 | 0 | `LifecycleConfig` field; `__init__`; budget `:949`, `:956` |
| `lifecycle_jax.py` | 4 | 5 | 0 | `__init__`; `solve()` `:1118`; signature `:445`; `in_axes`; `model_params` `:498`; docstring `:219`; unpack `:240`; budget `:273`, `:294` |
| `olg_transition.py` | 2 | 8 | 0 | `self.trend_growth`, `self.growth_factor`; cohort sizes `:810`, `:922`; batched call `:482–499`; `new_borrowing` `:1771`; K_g `:1918`; S_pens `:2241`; `get_test_config` `:2612` |
| `fiscal_experiments.py` | 8 | 20 | 2 | signatures of `compute_debt_path`, `_nfa_ca_paths`, `_balance_residual`, `_correct_base_macro_nfa` (all defaulting Γ = 1.0); Γ read in `run_debt_financed`, `run_tax_financed`, `run_nfa_constrained`; recursion `:264`; CA `:613`; `terminal_flow_balance` `:319`, `:322`; `K_g_ss` `:417`; 6 `compute_debt_path` callers; 5 `_nfa_ca_paths` callers; `_balance_residual` caller `:787`; 3 `_correct_base_macro_nfa` callers `:702`, `:902`, `:1093`; `pop_growth` field `:120` and its comment `:111` |
| `run_fiscal_figures.py` | 3 | 3 | 4 | `params_out` `:518` (`trend_growth`, `pop_growth`, `delta_g`); `I_g_warmup` `:95`; print `:125`; test-branch literal `:170`; `pop_growth=` kwargs `:241`, `:255`, `:283`, `:294` |
| `eval_fiscal_results.py` | 4 | 6 | 0 | params/config read `:590–609`; `chk_debt_accumulation` signature `:149`, recursion `:160`, message `:164`; call `:463`; `terminal_flow_balance` check `:492–504` |
| `regen_fiscal_figures_from_json.py` | 2 | 1 | 0 | Γ from `params`; `_nfa_ca_paths` call `:166` |
| `calibrate.py` | 7 | 0 | 0 | `generate_report` `:1328`: grid-top share, `max(a_sim)/a_max`, points below the mean |
| `validate_backends.py` | 0 | 1 | 0 | `:92` |
| `diag_ss_vs_transition.py` | 0 | 1 | 0 | `:76` |
| `diag_bequest_decomp.py` | 0 | 1 | 0 | `:75` |
| `check_a0_predetermination.py` | 0 | 2 | 0 | `:31`, `:50` |
| `calibration_input_GR.json` | 1 | 7 | 0 | `trend_growth` = 0.017; `pop_growth` = −0.006; `gamma` = 1.0; `r_B` = 0.019; `delta_g` = 0.036485; `interest_over_Y` = 0.0312; `_derived.theta.beta` = 0.972 and `.nu` ≈ 11 seeds |
| `calibration_input_GR_rB0.json` | — | — | file | retired |
| derived configs (Step 6) | 5 files | — | — | baseline copies with `trend_growth`, `pop_growth` or `r_B` overridden; β unchanged at γ = 1 |
| `test_fiscal_experiments.py` | ~40 | 4 | 0 | fixture `_make_olg` `:44`/`:55` (no `pop_growth` → constructor default 0.01); asserts `:165`, `:329`, `:497–504`; new: constant-B, CA, JSON identity. The direct-helper tests (`:167–181`, `:269`, `:276`, `:287`, `:297`, `:306`, `:396`) stay green only because the new Γ arguments default to 1.0 |
| `test_olg_transition.py` | ~110 | 3 | 0 | asserts `:1493–1511`, `:1543–1557`, `:1615–1632`; new: household isomorphism (both backends), backend agreement at g > 0 on `c_policy`/V, K_g flat at n ≠ 0, `new_borrowing` identity, `S_pens` recursion, age-weight equivalence, detrended baseline flatness (`get_test_config` is edited in `olg_transition.py`) |
| `CLAUDE.md` | 0 | 3 | 0 | the plan's one-line pointer `:34` (still says "r_B = g = 1%"); `delta_g` lines `:147`, `:180` |
| `docs/dsa_economic_problem.md`, `docs/model_vs_implementation.md`, `docs/dsa_implementation.md`, `docs/FISCAL_SCENARIOS.md` | prose | prose | — | per-capita detrending, laws of motion, ν trend, β, labour-augmenting rate |
| pension indexation parameter (point 2) | 4 | 5 | 0 | `LifecycleConfig` field; `_compute_budget` `:706`; `compute_budget_jax` `:144` plus a new `period_params` element (`:239`, built in `solve_lifecycle_jax`) and its terminal call `:645–671`; simulations `:1171`, `lifecycle_jax.py:693` |
| debt-elastic r_B (point 6) | 13 | 2 | 0 | `prices.r_B_debt_elasticity`; `compute_debt_path`; `run_fiscal_scenario`; `params_out`; eval; regen |
| **Code total (excl. tests, docs, configs)** | **49** | **57** | **6** | 12 files |

Unchanged: `run_scale_loop.sh`, `chain_fiscal_after_loop.sh`,
`normalize_A_tfp.py`, `pin_baseline_closure.py`, `build_olg_transition`, the
simulation code in both backends, `_feature_kwargs`.

## Points to settle before writing this up

Settled: preferences (log consumption + separable hours), pension
indexation (own parameter, default g), the comparison protocol (default, with
the held-spread scenario as a second comparison), and the debt-elastic r_B.
Also settled: the r vs r_B closure — (b), external official-sector debt —
r_B = 1.9%, and the baseline (g, n) = (1.7%, −0.60%) from the 2024 Ageing
Report. Nothing is open.

Audited 2026-09-24 by three context-free agents (detrending economics; code
references and implementation readiness; numbers and calibration). Their
findings are folded into the steps above: the Γ threading rewritten around
the three experiment functions and `_correct_base_macro_nfa`, the Γ = 1.0
defaults, `get_test_config`'s true location, the corrected r̃ and β/ν seeds,
three missed sites, the two Step-6 confounds, and four added checks.

**Labour disutility — settled: log consumption, separable hours.** With
separable preferences the labour FOC is ν_t·l^φ = c_t^(−γ)·MW_t, and c_t and
MW_t both scale with Z_t, so constant hours along a growth path require
ν_t ∝ Z_t^(1−γ). At the γ = 2 of the current calibration the disutility weight
would have to fall at rate g; at the chosen γ = 1 it is constant.

Joao: literature review to choose one functional form.

Claude: review done → `lit-review/olg-growth-preferences.md`; 43 of the 69
cited works downloaded to `lit-review/pdfs/`, title-checked and read at the
claim-bearing passages, with the record in
`lit-review/olg-growth-preferences-verification.md` (26 not obtainable without
library access, listed there). The field splits three ways: KPR/Cobb-Douglas composite
[c^θ(1−l)^(1−θ)]^(1−γ), γ = 2–5 (Braun–Joines 2015, Kitao 2015,
Glomm–Jung–Tran 2018 for Greece, Nishiyama–Smetters 2007); log consumption +
separable hours (Krueger–Ludwig 2007, Hansen–İmrohoroğlu 2016, McGrattan et
al. 2019); AK nested CES with a growing time endowment. Separable CRRA with
γ ≠ 1 and growth is used with the Z_t^(1−γ) scaling of the disutility weight
by Guvenen–Rendall 2015 ("the utility cost ξ is assumed to grow at the annual
rate of (1+g)^(1−γ)"), Peterman–Sager 2022 (`v_{t+1}(h) = (1+g_w)^(1−σ) v_t(h)`,
with the stationary Bellman and the (1+g_w)a' budget written out) and Boullot
et al. 2026; Kitao 2014 and Conesa–Kitao–Krueger 2009 keep the form and drop
growth, saying so in a footnote. The growth-adjusted discount factor is
explicit in Nishiyama–Smetters 2007 (β(1+μ)^(α(1−γ)), α the consumption share)
and Boppart–Krusell–Olsson 2023 (β·g^(1−σ) plus a gross cost of saving g in
the budget). So the current form + trend in ν has precedent; the CD composite
is the modal choice in the fiscal-OLG strand and would need a new labour FOC.
(Settled below: log consumption.)

Joao: log consumption +
separable hours

Claude: decided — u(c, l) = log c − ν·l^(1+φ)/(1+φ), i.e. γ = 1 with the
current separable form. Consequences, all already reflected in the steps
above: `"gamma": 1.0` in the config is the whole code change — both backends
branch to log at γ = 1 (`lifecycle_perfect_foresight.py:606`,
`lifecycle_jax.py:37`) and the labour FOC ν·l^φ = c^(−γ)·MW/(1+τ_c) carries γ
as a parameter. The trend in ν disappears: ν_t ∝ Z_t^(1−γ) = 1, so ν̄ = ν is
a constant in levels as well as in detrended units, and the write-up carries
no trend in the disutility weight. The effective and structural discount
factors coincide: β(1+g)^(1−γ) = β, so growth touches β only through the
budget's (1+g)a′. Note that this also changes the consumption-growth formula
from [β·s·(1+r̃)]^(1/γ) to β·s·(1+r̃)/(1+g), which is why the Step 5 seed is
0.972 and not β_old·(1+g). This is the BGP case named as
such in the literature read: Krueger–Ludwig 2007 set σ = 1 in the
Cobb-Douglas composite "such that utility is separable",
Cooley–Henriksen–Nusbaum 2024 "set σ = 1 in order to obtain balanced growth
preferences as in KPR", and Carranza-Ugarte–Díaz-Saavedra–Galdon-Sanchez 2023
(the nearest neighbour: SOE OLG with public capital) use
log(c) + χ(1−l)^(1−σ)/(1−σ).

Two consequences to note. IES rises from 0.5 to 1, so θ moves well beyond the
seeds — treat the SMM as a fresh fit, not a warm start, and re-seed ν as well
as β (Step 5). And
`utility_jax` evaluates both branches of its `jnp.where`, so the γ = 1 case
computes c^0/0 = inf in the discarded branch; harmless for the forward grid
search, but clip the exponent if the function is ever differentiated.

**Pension indexation.** A retiree's benefit is ρ · w_at_retirement · base,
with `w_at_retirement` the cohort's own detrended wage at the retirement date
(`lifecycle_perfect_foresight.py:319`, `:706`). The detrended benefit is
constant through retirement, so in levels it grows at g: benefits are indexed
to the trend, not to the current wage (a later move in w̄, as under the I_g
shock, does not reach existing retirees). Price indexation would need a factor
(1+g)^−(years since retirement) at four sites: `_compute_budget` (`:706`),
`compute_budget_jax` (`lifecycle_jax.py:144`), the NumPy simulation (`:1171`)
and the JAX simulation step (`lifecycle_jax.py:693`).

Joao: we will review the pension indexation rule in the Greek case. For now, keep pension indexation as a separate parameter, set to g as default, but to be calibrated later.

Claude: add `pension_index_growth` to `LifecycleConfig` (default =
`trend_growth`). Detrended benefit at age j ≥ J_R:
ρ·w_ret·base·((1+g_p)/(1+g))^(j−J_R); the factor is 1 at g_p = g, so the
default reproduces the current code. **Five sites, not four**: the two NumPy
ones are one-line (`_compute_budget` `:706` and the simulation `:1171`, both
with t and `retirement_age` in scope), but `compute_budget_jax` (`:144`)
receives `is_retired` as a bool and has no notion of j − J_R, so the factor
must be threaded as a new element of `period_params` (unpacked at `:239`,
built in the per-period `xs` of `solve_lifecycle_jax`), and the **terminal
call to `compute_budget_jax` inside `solve_lifecycle_jax` (`:645–671`) is a
fifth site** — at T−1 = 59 ≥ retirement_age = 39 the terminal agent is
retired. The JAX simulation step (`:693`) is the fifth NumPy-side analogue.
Note the floor interaction: `pension = max(pension, pension_min_floor)` is
applied after, so with g_p < g the detrended benefit declines through
retirement and the 0.15 floor starts binding at older ages — a margin the
default g_p = g never exercises.

**Interest on government debt — settled: external, official-sector debt (b).**
Households earn r on all wealth including government debt, while debt accrues
at r_B. At r = 4%, r_B = 1.9%, B/Y = 1.64,
that is (r − r_B)·B/Y = 3.44% of Y at the chosen r_B = 1.9% (6.6% at the
draft's r_B = 0).
Under the closure settled below the difference is traced: households' gross
foreign assets are F = nfa + b, so net interest from abroad is
r·nfa + (r − r_B)·b, and nothing is missing from the resource constraint.

Joao: we need to check what SOE models typically assume on r and r_B differentials;
also, clarify what is means that we have one single capital market (international), but
two different interest rates in the model.  This paper may be relevant:
https://www.econstor.eu/bitstream/10419/79190/1/339969415.pdf

Claude: as coded there is no portfolio — r and r_B are two exogenous prices
on the same stock: households earn r on all of A (B included), the
government pays r_B, and (r − r_B)·B is an inflow that appears nowhere. Two
consistent closures: (a) B is held abroad at r_B and households hold K + NFA
at r (then A = K + NFA and the NFA accounting changes); (b) households hold B
at r_B and the rest at r (fixed-share portfolio). The linked paper is
Schmitt-Grohé–Uribe, "Closing small open economy models" (JIE 2003): one
world rate with a debt-elastic premium r* + ψ(e^(d−d̄) − 1) on the country's
net foreign position — the template for the elasticity below, not a two-rate
model. What SOE fiscal OLG models assume on r vs r_B: to check (queued after
the preference review).

Joao: still unclear what to do with r and r_b, and tha assumptions on household portfolio. Clarify what are the reasonable and liteature grounded options.

Claude: the spread is an arbitrage only if households hold B and optimise the
portfolio. Three closures, each read in a source:

(a) **One rate**, r_B = r. No arbitrage, no portfolio. Standard in the
fiscal-OLG literature. Here it forces r_B = 4% against a Greek implicit rate
of 1.6–2.2%, so the interest line is wrong by 2–4% of Y.

(b) **B held abroad**, households hold K_domestic plus foreign assets at r.
Carranza-Ugarte et al. 2023: the government "issues one period real foreign
debt". This is what the code already does — `NFA = A − K_domestic − B` is the
country's position, households' interest income is r·A whether the foreign
assets are gross or net of B, and the government's r_B·B leaves the country;
no code change and no missing term. It has to be said in the write-up why the
sovereign borrows at 1–2% while residents lend abroad at 4%: for Greece the
answer is that the stock is largely official-sector (ESM/EFSF/GLF) at
concessional rates, which is also what makes the data implicit rate 1.6–2.2%.
Check the official-sector share of Greek debt before writing that sentence.

(c) **Convenience yield on bonds.** Boullot–Cahn–Challe–Matheron 2026,
following Krishnamurthy–Vissing-Jorgensen: bonds enter utility, so "the two
assets cease paying the same equilibrium rate of return … necessary to
reconcile the low fiscal cost of public debt compared to the relatively high
return on capital". Their specification gives r_B = (1−α)·r in equilibrium
with α read off the cross-section of returns, and households indifferent over
portfolio shares — so no portfolio state. Hansen–İmrohoroğlu 2016 use the
same device for Japan (bonds in utility, endogenous bond price, bond interest
taxed at its own rate τ_b). Code cost here: households earn the blended
return r̃_t = [r(A_t − B_t) + r_B·B_t]/A_t, i.e. `r_path` becomes endogenous
to B/A; α = 1 − r_B/r is 0.53 at the chosen r_B = 1.9%.

For contrast, İmrohoroğlu–Kitao–Yamada 2019 simply set r_b = 1%, r = 3% and a
pension-fund return of 2% side by side — available to them because their
households do not optimise a portfolio (the consumption profile is
calibrated). That route is closed here.

Recommendation: (b), stated in one sentence in the model section, with (c) as
the answer if a referee presses on why residents do not arbitrage. Neither
changes the code; (c) would.

Joao: go with (b).

Claude: adopted. The supporting data, from the Greek Ministry of Finance
*Sovereign Borrowing Outlook — Annual Debt Bulletin 2024* (saved at
`lit-review/data-debt/greece-annual-debt-bulletin-2024.pdf`, with the text
extraction beside it):

- Debt by type of creditor at end-2024: **official sector 74%, private sector
  26%** (the bulletin's chart; 76/24 in 2022, 75/25 in 2023, against 16/84 in
  2010).
- Central government debt, loans vis-à-vis non-residents: EFSF/ESM/IMF
  **€218.8bn of €403.9bn gross = 54.2%**; other non-residents €22.7bn; loans
  in total €298.6bn = 73.9% of the stock. Loans vis-à-vis residents are
  €0.14bn, and the Bank of Greece holds none.
- Weighted average cost of total new borrowing in 2024, **excluding repos**:
  **3.43%**; weighted average maturity of new **medium-term** borrowing 16.54
  years — against the 1.6–2.2% implicit rate on the whole stock computed from
  `DATA_GR.xlsx` above. The gap between the two is the concession.
- Debt/GDP 154% in 2024 (the config's B/Y = 1.64 is the earlier figure).
- The ESM puts the interest saving to Greece at €12bn in 2017, 6.7% of GDP,
  "at similar level every year". ⚠️ This came from the ESM's web explainer and
  is **not** in either PDF saved under `lit-review/data-debt/`; source it
  properly or drop it before it reaches the paper.

So the model's B is held abroad at a concessional rate and households hold
domestic capital plus foreign assets at r: r_B ≈ 2% with r = 4% is not an
unexploited arbitrage but the official-sector composition of the stock. Write-up
consequence, one paragraph in the model section and one sentence in the
calibration: state that B is external and official, cite the 74%/54.2% shares
and the 3.43%-versus-implicit-rate gap, and note the approximation — 26% of
the stock is private and part of that is resident, so the clean split is an
approximation of a stock that is mostly, not entirely, external.

Code: nothing changes. `NFA = A − K_domestic − B` is already the country's
position and household interest income is r·A under either reading; the
resource constraint in the write-up carries r·(A − K_domestic) − r_B·B as net
interest from abroad.

**The level of r_B — settled: 1.9%, the implicit rate on the stock averaged
over 2012–2024.** The debt-ratio rest point is PD/Y = (Γ − 1 − r_B)·b. The
implicit rate (interest payments ÷ debt, `DATA_GR.xlsx` codes 40 and 49)
averages 1.93% over 2012–2024 and 1.90% on the aggregate (Σinterest ÷ Σdebt);
other windows: 1.73% for 2015–2024, 1.67% for 2019–2024, 3.46% for the full
1999–2024 sample, 2.06% in 2023 alone. 2012–2024 is the window over which the
official-sector composition of the stock holds, which is the closure adopted
in the point above; the full sample mixes in the pre-crisis market-financed
regime. So r_B = 1.9%, replacing the config's 2.1% (the 2023 value).

At that rate, with Γ − 1 = 1.09%: r_B − g_Y = **+0.81pp**, holding B/Y at 1.64
needs a primary surplus of **1.33% of Y** (against the 1.95% in the closure),
so the baseline ratio declines. The draft's r_B − g_Y = 0 case is r_B = 1.09%,
below anything in the data; it survives only as a comparison scenario, not as
the baseline. The cost of new borrowing in 2024
was 3.43% (Annual Debt Bulletin), so the marginal and average costs differ by
1.5pp — that gap is what the debt elasticity ψ would represent if it is ever
switched on.

The choice changes `prices.r_B` and nothing else in the code.
`fiscal.interest_over_Y = 0.034` is a validation entry carrying the 2023
value; at r_B = 1.9% and B/Y = 1.64 the model's line is 3.12%, so update it
or note the 0.3pp discrepancy.

Joao: check data on Greek average iterest on government debt. we may have this data
in the project's data folder

Claude: `data/DATA_GR.xlsx`, sheet DATA, codes 40 (interest payments/GDP)
and 49 (debt/GDP). Implicit rate = interest/debt: 2010–23 mean 2.24%,
2015–23 mean 1.66%; 2019 1.62%, 2023 2.06%, 2024 2.26%. The config's
`r_B = 0.021` and `interest_over_Y = 0.034` are the 2023 values. Against
the baseline g_Y = 1.09%, the data spread r_B − g_Y is +0.6 to +1.2pp.

**Comparison-scenario protocol.** Step 6 holds every structural parameter at
the baseline calibration and changes only g or n. Three alternatives, each a
different question:
(i) re-fit θ, A_tfp and the closure in each scenario — then each scenario
matches the data moments, and the comparison mixes the growth effect with the
re-fit;
(ii) hold the I_g level at 3.53% of Y_ss instead of the K_g level — then each
scenario's stationary K_g is I_g/(δ_g + Γ' − 1) (1.158 at (0, n), 0.660 at
(g, 0)) and `K_g_initial` must be set to it, or K_g drifts along the
scenario's baseline;
(iv) hold K_g/Y and let the K_g level follow each scenario's Y — the only
option that makes the I_g experiment like-for-like across scenarios;
(iii) hold r_B − g_Y instead of r_B — r_B' = r_B − (Γ − Γ'), i.e. 0.21% at
(0, n) and 2.51% at (g, 0), against r_B = 1.9% held. With r_B held, the
rest-point condition PD/Y = (Γ' − 1 − r_B)·b moves with the scenario.

Joao: I need clarification on what this choice is about.

Claude: what "the same economy without growth" means. Default (Step 6): keep
every calibrated parameter and change only g (or n); the g = 0 economy then
misses the data moments (A/Y, hours — and the pinned closure is one of them,
see Step 6) and its stationary I_g level is 2.27% of Y_ss, not 3.53%, because
δ_g is held. Alternatives: (i) re-fit θ at g = 0 so both
economies hit the moments — the comparison then includes the re-fit; (ii)
hold I_g/Y instead of the K_g level, so K_g/Y differs; (iii) hold the spread
r_B − g_Y instead of r_B. Recommendation: the default, with (iii) as a second
comparison.

Joao: ok

**Interest rate elasticity** Joao: add a function that can represent an elasticity of r_B
w.r.t. B/Y. baseline is r_B constant, but we want a parameter that can have a non zero elasticity.

Claude: r_B,t = r̄_B + ψ·(B_t/Y_{t−1} − b̄), with b̄ = `fiscal.B_over_Y` and
ψ = `prices.r_B_debt_elasticity` (default 0 gives the constant rate).
Precedent, read in the source: Carranza-Ugarte–Díaz-Saavedra–Galdon-Sanchez
2023 (JCE, Peru, SOE OLG with public capital) eq. (4)
`r_t = r* + ϱ·B_t/Y_{t−1}` with ϱ = 0.05, r* = 1.88%, so B/Y = 0.60 gives
4.88%, plus a hard ceiling B̄ = 0.8 on B/Y; Laubach 2009 puts the elasticity at
3–4bp per pp of debt/GDP (ψ ≈ 0.03–0.04). Implementation: `compute_debt_path`
takes `Y_path` and evaluates r_B,t inside the recursion on lagged Y — so there
is no fixed point and the τ_l root-find is unchanged; `r_B_path` becomes an output, recorded in the JSON and used by the
eval debt check and the `interest_payments` line. Under a debt-financed shock
the rate rises with B/Y. ≈15 lines: `prices` field, `compute_debt_path` (+6),
`run_fiscal_scenario` (+2), `params_out` (+1), eval (+2), regen (+1).

Joao: ok 

**The baseline (g, n) — settled: g = 1.7%, n = −0.60% (2024 Ageing Report).** All from
`data/DATA_GR.xlsx` (Eurostat, refreshed 2025-10-16): real GDP
(`Input`, Mil.Chn.2020.EUR, 1995–2024), population (`Input`, persons), and
single-year-of-age population (`Population by age`, 1961–2025).

Output per capita — the BGP counterpart of g:

| window | real GDP | population | per capita |
|---|---|---|---|
| 1995–2024 | +1.00% | −0.05% | **+1.06%** |
| 1995–2007 | +3.89% | +0.39% | +3.49% |
| 2010–2024 | −0.49% | −0.49% | 0.00% |
| 2013–2024 | +1.33% | −0.53% | +1.87% |
| 2015–2024 | +1.56% | −0.50% | +2.07% |
| 2019–2024 | +1.65% | −0.66% | +2.33% |

Population — the model's n is the growth rate of the entering cohort
(`cohort_sizes = exp(n·(birth year − base))`, model age 0 = real age 25), so
the age-25 series is the right target, with total population as the
cross-check:

| window | total population | age-25 cohort |
|---|---|---|
| 1980–2023 | +0.18% | **−0.54%** |
| 1995–2023 | −0.04% | −1.72% |
| 2000–2023 | −0.15% | −2.36% |
| 2010–2024 | −0.49% | — |
| 2013–2024 | −0.53% | — |

(The age-25 series ends in 2023, so those windows stop there; total population
runs to 2024 and gives −0.05% for 1995–2024, −0.16% for 2000–2024.)

The config's n = −0.573% is squarely inside this: it matches the age-25 cohort
over 1980–2023 (−0.54%) and total population over 2010–2024 (−0.49%) and
2013–2024 (−0.53%). Recent entering cohorts shrink far faster (−1.7% to
−2.7%), which a stationary n cannot represent; the ageing devices that could
are out of scope here.

**The Commission's own assumptions** (2024 Ageing Report, *Underlying
Assumptions and Projection Methodologies*, IP 257, November 2023; saved at
`lit-review/data-debt/ageing2024.pdf`), for Greece over 2022–2070:

- Population (EUROPOP2023): 10.4m in 2022 → 7.8m in 2070, −25%, i.e.
  **−0.60% per year** — the config's n = −0.573% in all but rounding.
- Hourly labour productivity **+1.6% per year** (1.1, 1.6, 2.1, 1.8, 1.4 by
  decade, Table I.3.3), TFP +1.1%, total hours worked −0.6% (Table I.3.2),
  potential GDP **+1.1%** (Table I.3.1). Hours per capita are therefore flat.
  The Report's own line for **real GDP per capita is 1.7%** (Table 5) — that
  is the model's g, and the 0.1pp against hourly productivity is one-decimal
  rounding in the tables. At g = 1.7% with n = −0.60%, Γ − 1 = 1.09% against
  the Report's GDP growth of 1.1%.

That is the natural benchmark for a debt-sustainability paper, since the
Commission's own DSA runs on these numbers.

For reference, the project data alone would support **g = 1.06%** (per-capita
output, 1995–2024, the one window spanning a full boom–bust–recovery rather
than a recovery phase) with **n ≈ −0.5%**. The choice moves the debt
arithmetic a long way at r_B = 1.9%:

| (g, n) | Γ − 1 | r_B − g_Y | flat-B/Y surplus | baseline ΔB/Y | B/Y at t = 60 | δ_g |
|---|---|---|---|---|---|---|
| 1%, −0.573% (old placeholder) | 0.42% | +1.48pp | 2.43% | +0.47pp/yr | 2.09 | 0.043170 |
| **1.7%, −0.60% (Ageing Report)** | 1.09% | +0.81pp | 1.33% | −0.62pp/yr | 1.17 | 0.036485 |
| 1.87%, −0.53% (2013–2024 data) | 1.33% | +0.57pp | 0.93% | −1.00pp/yr | 0.93 | 0.034082 |
| 1.06%, −0.05% (1995–2024 data) | 1.01% | +0.89pp | 1.46% | −0.48pp/yr | 1.26 | 0.037288 |

(ΔB/Y and B/Y(60) are the baseline path under a primary surplus held at
1.95% with detrended Y flat, starting from B/Y = 1.64 — arithmetic, not model
output.) Against that closure the placeholder has
the ratio rising and every other row has it falling; at the Ageing Report
pair it reaches 1.28 after sixty periods, at the 2013–2024 pair 0.93.

**Settled: the Ageing Report pair, g = 1.7%, n = −0.60%** — the Report's own
GDP-per-capita line — and the plan's arithmetic above is written at those
values. It is the Commission's own
projection for the country and the same input its DSA uses, it sits between
the historical windows rather than extrapolating either the crisis or the
recovery, and its n is the config's existing value to two decimals. The
2013–2024 pair (g = 1.87%) serves as the high-growth comparison and the
1995–2024 per-capita trend (g = 1.06%) as the low-growth one — both on the
Step 6 protocol, structural parameters held.

## Documentation

`docs/dsa_economic_problem.md`, `docs/model_vs_implementation.md`,
`docs/dsa_implementation.md`, `docs/FISCAL_SCENARIOS.md` and `CLAUDE.md`
(the `delta_g = 0.04738255` line and the `I_g = delta_g·K_g` convention lines)
carry the per-capita detrended aggregates, the household problem in Z_t
units, the four laws of motion above with Γ, the log-consumption preferences
with a constant ν, and the labour-augmenting rate implied by g, n and η_g.
`docs/dsa_economic_problem.md` also carries the external-official-debt reading
of B with the shares cited in the points above.
