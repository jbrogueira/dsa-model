# Implementation of the DSA-LSA pipeline

This document maps the economic problem in `dsa_economic_problem.md` to the code in `/Users/joaob.sousa/Work/research/dsa/code`. `(eq. N)` references point to that document. Conventions: policies are indexed by lifecycle age, not calendar time; the policy array shape is `(T, n_a, n_y, n_h, n_y_last)`; aggregate labor `L` is in efficiency units (wage-valued income divided by `w`); two interchangeable backends (`backend='numpy'|'jax'`) produce the same objects, NumPy being the reference and JAX the accelerated path. All new model features default OFF.

## 1. The live pipeline at a glance

```
calibrate.py                  SMM driver: load config → solve SS → match 5 moments → write _derived.theta
  └─ load_config / build_lifecycle_config / compute_equilibrium_prices   config → CalibrationSpec, w from firm FOC (11)
       └─ LifecycleModelPerfectForesight   household lifecycle solve+simulate (NumPy/Numba)   (1),(4)-(9)
            └─ LifecycleModelJAX            same interface, vmap/scan (JAX)

run_fiscal_figures.py         experiment driver: build economy → baseline → G/Ig counterfactual → figures
  └─ build_olg_transition (calibrate.py)    config → (OLGTransition, transition paths, T_TR)
       ├─ OLGTransition.simulate_transition  cohort solve → simulate → aggregate (10)-(14)
       │    ├─ solve_cohort_problems          per-cohort lifecycle solve + MIT stitching (§7)
       │    └─ _aggregate_capital_labor_njit  K, C, L aggregation (13)
       ├─ OLGTransition.compute_government_budget_path  budget, primary deficit, debt (15)-(16); called by the driver, not inside simulate_transition
       └─ run_fiscal_scenario (fiscal_experiments.py)  baseline+counterfactual, financing bisection (§6)

olg_transition.py             also a standalone CLI: run_from_config(...) for a single transition
pin_baseline_closure.py       one stationary solve → pin fiscal.other_net_spending_over_Y (O_t, §9)
build_survival_GR.py          builds data/survival_GR.npz (cohort survival table, π(j))
eval_fiscal_results.py        post-simulation validation of fiscal_results.json
```

Excluded as dead/one-off diagnostics (not reachable from a production entry point): `check_a0_predetermination.py`, `diag_bequest_decomp.py`, `diag_ss_vs_transition.py`. Test files: `test_olg_transition.py`, `test_fiscal_experiments.py`, `test_calibrate.py`, `test_income_process.py`.

## 2. File-by-file map

| File | Role | Computes (model object) |
| :-- | :-- | :-- |
| `lifecycle_perfect_foresight.py` | household lifecycle solve + simulate, NumPy/Numba reference | utility (1) in `utility` `:604-609` (labor disutility `:872`); budget (4)–(6) in `_compute_budget` `:693`; Bellman (7)–(8) in `_solve_period`/`_solve_state_choice`, terminal `:928-940`, `a'` grid search `:947-991`, survival-adjusted `E[V_{j+1}]` `:985`; labor FOC (9) in `_solve_labor_newton` `:816-866`; income process (2)–(3) `:445-520`, permanent FE `α` process `:513`; Monte-Carlo panel in `simulate` `:1260+` |
| `lifecycle_jax.py` | JAX backend, same interface | CRRA utility `utility_jax` `:35-41`; HSV `_hsv_tax` `:44-47`; robust labor FOC `solve_labor_robust_jax`; batched solve/simulate via `lax.scan`/`vmap` |
| `olg_transition.py` | transition orchestration, firm, aggregation, government | production (10) `_production_function_njit` `:815-818`; prices (11) `_marginal_products_njit` `:822-829`, `K/L` inversion `:1939`; public capital (12) `:1908-1917`; aggregation (13) `_aggregate_capital_labor_njit` `:833-849`; NFA (14) `:2118-2119`; primary deficit (15) `compute_government_budget` `:1684-1797`, `:1773`; debt (16) and trust fund `compute_government_budget_path` `:2197-2266`; cohort solve + MIT stitching `solve_cohort_problems` `:1033`, per-`α` stitching `:1238`/`:1306`; main loop `simulate_transition` `:1832-2195` |
| `fiscal_experiments.py` | fiscal scenarios, financing bisection, multiplier | `FiscalScenario` dataclass `:76-133`; `FiscalScenarioResult` `:137-175`; debt evolution (16) `:242-261`; terminal condition (17) residual `:299`; multiplier (18); dispatcher `run_fiscal_scenario`; Type A/B/C `run_debt_financed`/`run_tax_financed`/`run_nfa_constrained` |
| `calibrate.py` | config loader + SMM | `load_config` `:954`, `build_lifecycle_config` `:1009`; `_derived.theta` override `:1060-1075`, `:1161-1168`; `compute_equilibrium_prices` returns `w`, `K/L`, `Y/L` from `{r, α, δ, A, K^g}`; `build_olg_transition` `:1098`; SMM moments `run_model_moments`, objective `Q(θ)` |
| `run_fiscal_figures.py` | experiment CLI, ratio-mode spending, figures | `--shock G|Ig|both`; `--config` ratio mode (shocks as deltas of `Y(t)`, `B_0 = (B/Y)·Y_0`) |
| `pin_baseline_closure.py` | pins `O_t` (§9) | one stationary solve, `--write` updates `fiscal.other_net_spending_over_Y` |
| `build_survival_GR.py` | survival data builder | writes `data/survival_GR.npz` → `π(j)` cohort table |
| `eval_fiscal_results.py` | post-run validation | checks identities on `fiscal_results.json` |

Required inherited helpers run unmodified: `calibration_input_GR.json` (the single config driving calibration, transition, and experiments), `data/survival_GR.npz` (cohort survival, opt-in via `transition.survival_data_file`).

## 3. How the solve works

**Household lifecycle (inner).** Backward induction over `j = J-1 … 0` (`solve` `lifecycle_perfect_foresight.py:611-691`). The terminal age consumes all resources (`:928-940`); earlier ages run the state loop (`_solve_period` `:881-912`) and grid-search `a'` over `n_a = 100` points (`_solve_state_choice` `:947-991`), solving the labor FOC (9) per candidate when employed and accumulating the survival-adjusted expectation `β·π(j)·E[V_{j+1}]` over the income (and, if `n_h>1`, health) transition. The labor FOC solver is a projected Newton bracketed to `c(\ell) > 0` with a bisection fallback, step tolerance `< 1e-12` (`_solve_labor_newton:816-866`); the JAX path uses `solve_labor_robust_jax`. State space: assets on an exponentially-spaced grid `[0, 200]`, income via Tauchen with `n_y-1 = 4` employed states (`:471`), permanent fixed effect on a 5-node grid (`solve` `:611-691`, outer loop over `α` `:650-680`).

**Transition (outer).** `simulate_transition` (`olg_transition.py:1832-2195`): for each cohort solve the lifecycle under that cohort's diagonal of the exogenous `(r_t, w_t, τ_t, K^g_t)` paths, simulate `n_sim` agents (2000 in transition, 10000 in SS calibration; seed 42), then aggregate per (education, age) cross-section into `{K, C, L}` (`_aggregate_capital_labor_njit` returns `(K, C, L)`; `L` is then divided by `w` to efficiency units, `:2102`). `K_domestic` is computed from the firm FOC (11) *before* `Y_path`, and `Y` uses `K_domestic`, not household wealth `A`; `NFA = A − K_domestic` is returned and `B` subtracted by the fiscal caller (eq. 14). Public capital is integrated forward by (12) (`:1908-1917`).

**Outer fixed points.** (i) Bequest circuit (`recompute_bequests=True` when `survival_probs` is set): iterate cohort-solve → simulate → recompute aggregate bequests → update the lump-sum transfer until `max|Δ| < 1e-4`, capped at 5 iterations (`:1990-2047`); the production CLI defaults ON, the test CLI OFF. (ii) Financing bisection (labor-tax case): bisect the tax increment until the terminal balance residual (17) reaches zero (`fiscal_experiments.py`, residual `:299`). No capital-market fixed point is solved — `r` is exogenous and `w` follows from (11).

**MIT-shock stitching (§7).** `solve_cohort_problems` predetermines `A[0]` across counterfactuals: pre-transition ages are padded with baseline values, and a pure-baseline lifecycle solve overwrites the pre-transition slice of both the scalar policies and the per-`α` policies `a/c/l_policy_alpha[:, :pre]` (both simulate paths read the per-`α` arrays). The MIT baseline model must use pure baseline `r/w/τ/transfer_floor` for all ages; `pre_transition_paths` carries `w_path` and `transfer_floor` for this reason. `A[0]` predetermination is verified exactly (diff 0.0) for both backends.

## 4. How the experiment is applied

The shock enters as explicit arguments to `simulate_transition`: `govt_spending_path=` (G) and `I_g_path=` (I^g), or in ratio mode `G_over_Y=`/`I_g_over_Y=`/`defense_over_Y=`/`other_net_over_Y=` (scalar or `(T,)`), where the budget uses `level = ratio · Y_path[t]` so each run's spending tracks its own realized output. `run_fiscal_figures.py --config` uses ratio mode with the shock as a delta of `Y(t)` (e.g. 0.02 = 2% of `Y(t)`) and `B_initial = (B/Y)·Y(0)`; the hardcoded test branch stays in level mode. `I_g_over_Y` is rejected when `η_g ≠ 0` (the I^g→K^g→Y simultaneity needs a fixed point), so the public-investment experiment with public capital on passes an `I_g` level path. Financing is selected by `FiscalScenario.financing ∈ {'debt','tau_l','tau_c','tau_k','tau_p','transfer_floor'}` with the adjustment distributed over time by a profile `ψ_t`; the balance condition `terminal_debt_gdp` targets `B_T/Y_T` (or the flow target (17)).

## 5. How the objective / welfare is extracted

No welfare/CEV measure exists (eq. 8). The reported outputs are: the debt-to-GDP path `B_t/Y_t` and the financing tax increment `Δτ^l` (the debt-sustainability outcome), the fiscal multiplier `𝓜_t = (Y^{cf}_t − Y^{base}_t)/Δ` (18), and the per-stock terminal drift on `FiscalScenarioResult`. The SMM estimation objective `Q(θ)` (`calibrate.py`, `run_model_moments`) matches five moments — mean hours, `A/Y`, payroll-revenue/`Y`, pensions/`Y`, government-health/`Y` — by Nelder–Mead on logit-transformed parameters, writing the result to `_derived.theta`. `eval_fiscal_results.py` validates `fiscal_results.json` against the budget, production, and aggregation identities; `compute_fiscal_ratios` also reports `primary_balance_full_over_Y` and `closure_other_over_Y`.

## 6. Current state and limits

- **Public capital is on in the live config** (`production.eta_g = 0.05`, `K_g = 1.0`, `delta_g = 0.0`), unlike the source baseline (eq. 8 / Doc 1 §10.1); activation is in progress (see `docs/PUBLIC_CAPITAL_KG_PLAN.md`).
- **Live calibration (2026-06-09) differs from `DSA-LSA calibration.tex`** (Doc 1 §10.2): `_derived.theta` = `{ν 36.91, β 0.943, τ_p 0.198, ρ_pens 0.166, m 0.0428}` overrides the base config fields (`calibrate.py:1060-1075`); the `.tex` table reports the pre-SMM starting values.
- **Debt service uses `r^B = 0.021`**, distinct from `r = 0.04` (Doc 1 §10.3).
- **Defense and an other-net residual** are explicit budget lines; the residual `O/Y = −0.091122` is pinned at the initial steady state by `pin_baseline_closure.py` so the initial-point budget matches `primary_balance_target_over_Y = 0.0195` (Doc 1 §10.4). The baseline transition's `t=0` primary balance need not equal the target exactly.
- **Medical cost is age-dependent** via `m_age_profile` (Doc 1 §10.5).
- **JAX backend on macOS ARM** requires native ARM Python and `JAX_PLATFORM_NAME=cpu`; the Apple Metal backend lacks float64 and breaks the solver. 15 JAX tests fail on macOS ARM for this reason (backend issue, not model logic).
- **Off by default:** HSV progressive labor tax (`tax_progressive=false`), endogenous retirement window, child/schooling costs (`schooling_years=0`), means-tested transfer floor (`transfer_floor=0`), bequest tax (`τ^β=0`), age/health-dependent income transitions (`n_h=1`).

## 7. How to run

```bash
source ~/venvs/jax-arm/bin/activate            # JAX backend only
export JAX_PLATFORM_NAME=cpu                    # macOS ARM

# Calibrate (writes _derived.theta back to the config)
python calibrate.py --config calibration_input_GR.json --backend jax

# Single transition from config
python olg_transition.py --config calibration_input_GR.json --backend jax

# Fiscal shock figures (G, public investment, or both)
python run_fiscal_figures.py --config calibration_input_GR.json --shock G  --backend jax
python run_fiscal_figures.py --config calibration_input_GR.json --shock Ig --backend jax

# Pin the fiscal closure (one stationary solve)
python pin_baseline_closure.py --config calibration_input_GR.json --write

# Validate a fiscal run
python eval_fiscal_results.py            # reads fiscal_results.json

# Tests (exclude pre-existing macOS-ARM JAX failures for a clean baseline)
pytest test_olg_transition.py -v -k "not (TestJAXBackend or TestNewFeaturesJAX or TestLaborSupplyJAX or TestEndogenousRetirementJAX)"
pytest test_fiscal_experiments.py -v
```
