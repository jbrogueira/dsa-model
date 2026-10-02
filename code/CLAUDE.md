# OLG Transition Model

Overlapping Generations (OLG) model simulating demographic and policy transitions with heterogeneous agents.

## Co-authors

- K.Slawinska@esm.europa.eu
- Ramon.Marimon@eui.eu
- L.Zavalloni@esm.europa.eu

## Structure

```
olg_transition.py              # Entry point - OLG transition simulation
lifecycle_perfect_foresight.py # Lifecycle model with perfect foresight prices (NumPy)
lifecycle_jax.py               # JAX-accelerated lifecycle model (solve + simulate)
fiscal_experiments.py          # Fiscal scenario framework (debt/tax/NFA-constrained experiments)
test_olg_transition.py         # Pytest tests (83 tests, incl. 15 JAX tests, 4 cross-validation classes)
run_fiscal_figures.py          # Fiscal shock figures: G, Ig, or both shocks; CLI --shock, --output-dir flags
regen_fiscal_figures_from_json.py  # Re-plot fiscal figures from a saved fiscal_results.json (no simulation); --budget-components SHOCK SCENARIO draws the stacked budget lines behind the primary balance (levels + deviations)
test_fiscal_experiments.py     # Pytest tests for fiscal experiments (39 tests)
build_health_flag_data.py      # Health-flag data side: coverage/CHE/GDP/population from DATA_GR.xlsx -> data/health_flag_GR.csv
build_health_model_baseline.py # Health-flag model side (analytic, no solve): demographics/Abar/g_model -> data/health_model_baseline_GR.npz
health_flag_decomposition.py   # Shapley decomposition of gov-health/GDP gap into coverage/demographics/residual (g=kappa*Abar*psi) + figures
test_health_flag_decomposition.py  # Pytest tests (10) for the health-flag Shapley decomposition
docs/HEALTH_FLAG_DECOMPOSITION.md  # Spec for the health-expenditure flag decomposition
validate_backends.py           # NumPy-vs-JAX equivalence check (per-path n_sim-scaling test)
pin_baseline_closure.py        # Pin fiscal.other_net_spending_over_Y at the initial SS (--write)
normalize_A_tfp.py             # Root-find A_tfp s.t. initial-SS Y = 1 at fixed _derived.theta (--write)
run_scale_loop.sh              # Outer loop: SMM <-> A_tfp normalization until joint fixed point, then closure re-pin
chain_fiscal_after_loop.sh     # Waits for run_scale_loop.sh, gates on the A[0] check, runs the G+Ig set
docs/IMPLEMENTATION_PLAN.md    # Feature implementation plan & progress
docs/OPEN_ISSUES_2026-07-30.md # Six implementation gaps found in the draft-vs-code audit; items 1-4 share one batched re-run
docs/TREND_GROWTH_PLAN.md      # Plan for r_B = g = 1% (detrended units); audited against the code 2026-09-21, not implemented
docs/archive/                  # Completed/superseded plans (index in its README.md); not the live pipeline
../lit-review/                 # Literature review (2026-09-10/11): PLAN_* (candidates H1-H5, kill tests), review .md (verdict table), .bib (173 entries); no code impact
```

## Environment

JAX venv lives at `~/venvs/jax-arm/`. Create it once with the setup script (auto-detects platform):

```bash
bash setup_jax.sh          # macOS ARM → jax[cpu], Linux x86_64 → jax[cuda12]
```

Activate before running the JAX backend:

```bash
source ~/venvs/jax-arm/bin/activate
```

**Platform notes:**
- macOS ARM (Apple Silicon): use native ARM Python — x86 Python via Rosetta hits AVX issues with jaxlib
- Linux x86_64 + CUDA: requires a working NVIDIA driver; if unavailable at runtime, set `JAX_PLATFORM_NAME=cpu`

## Run

```bash
# Fast test mode (NumPy backend, default)
python olg_transition.py --test

# Fast test mode (JAX backend)
python olg_transition.py --test --backend jax

# Full simulation (recompute_bequests=True by default)
python olg_transition.py
python olg_transition.py --backend jax
python olg_transition.py --no-recompute-bequests   # skip bequest loop (open circuit)

# Lifecycle standalone tests
python lifecycle_perfect_foresight.py --test
python lifecycle_jax.py --test              # JAX cross-validation vs NumPy

# Run pytest suite (includes JAX cross-validation tests)
pytest test_olg_transition.py -v

# Fiscal shock figures (G = govt spending, Ig = public investment)
python run_fiscal_figures.py --shock G
python run_fiscal_figures.py --shock Ig
python run_fiscal_figures.py --shock both

# Config-based workflow (reads all parameters from JSON)
python calibrate.py --config calibration_input_GR.json --backend jax
python olg_transition.py --config calibration_input_GR.json --backend jax
python run_fiscal_figures.py --config calibration_input_GR.json --shock G --backend jax
```

## Config System

Country-specific parameters live in a JSON file (e.g., `calibration_input_GR.json`). The same file drives calibration, OLG transitions, and fiscal experiments. Key functions in `calibrate.py`:

- `load_config(path)` — loads JSON, derives `w` from firm FOC, computes age weights, builds `CalibrationSpec`
- `build_lifecycle_config(raw, w)` — constructs `LifecycleConfig` from parsed JSON dict
- `build_olg_transition(config_data, backend)` — constructs `OLGTransition` + transition paths from JSON
- `compute_equilibrium_prices(config_data)` — derives `w`, `K/L`, `Y/L` from `{r, alpha, delta, A_tfp, K_g}`

In `olg_transition.py`:
- `run_from_config(config_path, backend, recompute_bequests, n_sim)` — runs a transition from JSON

## Key Classes

- `OLGTransition`: Manages transition dynamics, aggregation, government budget. Accepts `backend='numpy'|'jax'`
- `LifecycleModelPerfectForesight`: Solves individual lifecycle problem via backward induction (NumPy/Numba)
- `LifecycleModelJAX`: JAX-accelerated lifecycle model. Same interface, vectorized solve via `jax.lax.scan`, simulation via `jax.vmap`
- `LifecycleConfig`: Configuration dataclass for lifecycle parameters
- `FiscalScenario` (`fiscal_experiments.py`): dataclass specifying a policy shock, financing instrument, and budget balance condition
- `FiscalScenarioResult` (`fiscal_experiments.py`): dataclass holding baseline + counterfactual macro/budget paths, debt path, NFA/CA paths, and convergence info

## Key Methods

- `_compute_budget()`: Computes after-tax income and budget for any state (retirement/working). Handles pension floor, progressive tax, child costs, transfer floor, age-dependent medical.
- `_solve_period()`: Solves a single period. Handles retirement window (discrete choice between working/retired).
- `_solve_state_choice()`: Core solver for a single state — survival risk, labor supply FOC, age-dependent P_y.
- `_simulate_sequential()`: Monte Carlo forward simulation of agent paths.
- `_get_P_y()` / `_get_P_y_row()`: Index into 2D or 4D P_y (age-dependent transitions).
- `_survival_prob()`: Returns survival probability π(j,s), or 1.0 if no survival risk.
- `_solve_labor_newton()` (NumPy) / `solve_labor_robust_jax()` (JAX): robust projected-Newton solve of the intratemporal labor FOC `ν·l^φ = c^{−γ}·MW/(1+τ_c)`, `MW = w·κ(j)·y·h·e^α·(1−τ_p)(1−τ_l)`, bracketed to the feasible region `c(l)>0`. (Replaced a 2-iter Newton that froze at the consumption clamp; see FISCAL_EXPERIMENTS_STATUS 2026-06-14.) `_solve_labor_hours()` (NumPy) is a legacy closed-form helper, not on the live path.
- `_print_income_diagnostics()`: Verbose income process diagnostics (called from `__init__` when `verbose=True`).
- `simulate_transition()`: Accepts `recompute_bequests=False`, `bequest_tol=1e-4`, `max_bequest_iters=5` — runs a fixed-point bequest loop when `recompute_bequests=True` and `survival_probs` is set; stores `_bequest_converged` and `_bequest_iter_count` on `self`.
- `run_fiscal_scenario()` (`fiscal_experiments.py`): dispatcher — runs baseline + counterfactual via Type A/B/C experiment.
- `run_debt_financed()` / `run_tax_financed()` / `run_nfa_constrained()` (`fiscal_experiments.py`): Type A (one sim), Type B (Illinois/modified-regula-falsi root-find on scalar Δτ to hit `balance_condition` — keeps the opposite-sign bracket, secant step, midpoint fallback; replaced pure bisection 2026-06-17, ~1 interior iter vs ~10–15), Type C (NFA/CA band around baseline; Mode I: shock scale bisect; Mode II: tax rate bisect). Type B `balance_condition` includes `terminal_nfa_gdp` (full NFA/Y at T_bal = target, the external-balance analogue of `terminal_debt_gdp`); Type C floor is per-period `NFA_t ≥ NFA_base_t − nfa_limit` (half-width 0 = exact baseline tracking).
- `compare_scenarios()` / `fiscal_multiplier()` / `debt_fan_chart()` (`fiscal_experiments.py`): output utilities for plotting and multiplier calculation. `compare_scenarios` accepts a generic `<line>_gdp` key (any `cf_macro`/`cf_budget` line ÷ Y, e.g. `A_gdp`, `NFA_gdp`, `primary_deficit_gdp`, `tax_l_gdp`); `NFA_gdp` uses the full `NFA_path`. `MACRO_VARS`/`FISCAL_VARS` live in `run_fiscal_figures.py` and the regen script (keep in sync). The plotted `interest_payments` line is `r_B·B` (falls back to `r` when `olg.r_B` is unset; fixed 2026-07-07 — it was `r·B` while the B law of motion used `r_B`); `regen_fiscal_figures_from_json.py --r-b <rate>` recomputes the line from stored paths when re-plotting JSONs written before the fix.
- **NFA in results is full on both sides.** `run_debt_financed`/`run_tax_financed`/`run_nfa_constrained` correct `cf_macro['NFA']` AND `base_macro['NFA']` from the partial `A − K_domestic` to the full `A − K_domestic − B` via `_correct_base_macro_nfa()` (fixed 2026-06-17). Plot/compare both sides on the same definition.
- **`eval_fiscal_results.py` conventions (since 2026-07-14):** debt accumulation is checked at `r_B` (from the JSON `params` or `--config` `prices.r_B`; pre-r_B JSONs without `--config` fall back to `r` and fail spuriously — pass `--config`); `bisection_target` compares B/Y at `T_balance` against the same shock's baseline `B_gdp_path[T_balance]`; shock-path checks are mode-aware via `shock_mode_G`/`shock_mode_Ig` embedded in `params` by `run_fiscal_figures.py` (inferred for older JSONs: ratio for config-run G, level for Ig with `eta_g != 0`). The eval main loop iterates baseline/debt_financed/tax_financed only — `nfa_constrained` blocks are not checked.

## Model Features

- Perfect foresight over price paths (r, w)
- Income risk (Tauchen discretization)
- Age-dependent health expenditure with government/household split (`n_h=1`, `kappa`, `m_good`, `m_age_profile`)
- Retirement with pensions (based on last working income)
- Minimum pension floor (`pension_min_floor`)
- UI benefits for unemployed
- Multiple tax instruments (consumption, labor, payroll, capital)
- Progressive HSV taxation (`tax_progressive`, `tax_kappa`, `tax_eta`)
- Means-tested transfers / consumption floor (`transfer_floor`) — booked since 2026-10-02: the simulations record the top-up per agent (`transfer_sim`), `compute_government_budget` adds `transfers` to `total_spending`, and the production config sets 0.0807. `financing='transfer_floor'` is usable again
- Survival risk / stochastic mortality (`survival_probs`)
- Age-dependent medical expenditure (`m_age_profile`)
- Age-dependent productivity transitions (`P_y_by_age_health`)
- Endogenous labor supply via FOC (`labor_supply`, `nu`, `phi`)
- Wage age profile (`wage_age_profile` in LifecycleConfig) — age-dependent wage multiplier κ(j), effective wage = w · κ(j) · y
- Career-average pension (`pension_avg_weight`, `mean_kappa_working`, `mean_y_employed` in LifecycleConfig) — pension base blends last income state with career average; `pension_avg_weight=1.0` recovers last-state-only pension. **Both GR configs leave `pension_avg_weight` unset**, so `calibrate.py:1093` derives λ = (1−ρ_z^{J_R})/(J_R(1−ρ_z)) = **0.443**; β, ν and ρ^pens were all fitted against that blended base
- Endogenous retirement window (`retirement_window`) — **raises** (2026-10-01). Honoured by the NumPy solve only: the JAX solve ignores it and *neither* simulation consults it, both retiring mechanically at `retirement_age` (`docs/bug_report.md:487`), so policies and realised incomes disagreed even on NumPy while `docs/model_vs_implementation.md` advertised it as cross-validated. Wanted later; implement it in the JAX solve and both simulations first
- Schooling phase with child costs (`schooling_years`, `child_cost_profile`)
- Government spending on goods (`govt_spending_path` in OLGTransition)
- Public capital in production (`eta_g`, `K_g_initial`, `delta_g` in OLGTransition). **Active in the GR config since 2026-07-10:** `eta_g=0.05`, `K_g=0.745` (= K_g/Y at the Y_ss=1 normalization; IMF ICSD 2019), `delta_g=0.04738255` (= (I_g/Y)/(K_g/Y) = 0.0353/0.745, keeps baseline K_g stationary at the level-I_g path)
- Public investment path (`I_g_path` in OLGTransition)
- Small open economy with sovereign debt (`economy_type`, `r_star`, `B_path` in OLGTransition)
- Net foreign assets accounting (NFA) in SOE mode
- Pension trust fund (`S_pens_initial` in OLGTransition)
- Defense spending (`defense_spending_path` in OLGTransition)
- Other net primary spending residual (`other_net_spending_path` in OLGTransition) — exogenous (other expenditure − other revenue) line absent from explicit tax/transfer/spending; added to `total_spending`, no household-side effect. Baseline fiscal closure: `fiscal.other_net_spending_over_Y` is a structural constant pinned at the **initial steady state** (not measured off a transition) so the initial-point government budget matches `fiscal.primary_balance_target_over_Y`; the baseline transition takes it as given and its t=0 primary balance need not equal the target exactly. Pin it with `pin_baseline_closure.py` (one stationary solve, no transition; `--write` updates the config); `compute_fiscal_ratios` also reports `primary_balance_full_over_Y` and `closure_other_over_Y`. Defaults None/0.
- Bequest redistribution fixed-point loop (`recompute_bequests` in `simulate_transition()`) — closed bequest circuit iterates until bequests converge; production CLI defaults to `True`, test CLI defaults to `False` (opt-in via `--recompute-bequests`). **`FiscalScenario.recompute_bequests` defaults to `False` and `run_fiscal_figures.py` never sets it**, so newborns receive nothing in every fiscal run. **Since 2026-10-02 the production config sets `external_params.tau_beq = 1.0`**: accidental bequests are taxed away in full and enter `total_revenue` as `bequest_tax` (olg_transition budget) and `bequest_tax_over_Y` / `primary_balance_over_Y` on the calibration side (`compute_fiscal_ratios`), so the circuit is closed through the government and both sides book the same line. The closure `other_net_spending_over_Y` must be re-pinned after this change (not yet done). The runs before 2026-10-02 had τ_beq = 0: 3.77 % of Y per period left the economy
- **Fixes of 2026-10-02 (audit report §3.8, `reports/model_audit_2026-10-01.tex`):**
  - Retired continuation value read at the household's own `i_y_last` in both backends (was index 0 for everyone; the pension depends on `i_y_last`), `lifecycle_perfect_foresight._solve_state_choice` and `lifecycle_jax.solve_period_jax`.
  - Hours are no longer capped at 1: both labour solvers bracket upward by doubling, so the intratemporal FOC holds with equality at every interior solution.
  - **Means-tested consumption floor is booked:** `external_params.transfer_floor = 0.0807` (Greek guaranteed minimum income, EUR 200/month single adult, `build_transfer_floor_GR.py` → `data/transfer_floor_GR.json`); the simulations record the top-up per agent (`transfer_sim`, 23rd panel element; `_compute_budget` returns it as a 5th value), the per-age means carry it as the 12th element (`N_AGE_MEANS`), `compute_government_budget` books `transfers` in `total_spending`, and `compute_fiscal_ratios` nets it in the primary balance. The constructor and budget refusals are gone. JAX refuses a floor together with `schooling_years > 0` (the simulated transfer omits child costs).
  - **L excludes UI:** `L = (wage income + UI − UI)/w` everywhere (`simulate_transition`, `compute_aggregates`, `_compute_ss_aggregates`, `compute_fiscal_ratios`), so `w·L` is the wage bill.
  - `_entrant_weights` extrapolates past the entering-cohort table at the terminal rate `n_path[-1]` instead of raising (the fiscal runs' 20 post-horizon periods overran the table, which ends in 2210).
  - `_check_terminal_convergence(..., B_path=...)` tracks the debt stock at the slow tolerance; the three fiscal runners pass it. No rest-point rule is imposed.
  - `normalize_A_tfp.py` and `pin_baseline_closure.py` read θ with `theta_from_config` (no KeyError on a parameter the last SMM did not fit); the closure script writes the level-based `other_net_spending_over_Y`.
  - `run_scale_loop.sh` converges only if the moments at the written `(θ, A_tfp)` pair are within `MOM_TOL` (5e-3) as well as the pre-update residual.
  - `run_fiscal_figures.py`: the preliminary run is at the full `N_SIM` (so `B/Y(0) = B_over_Y` exactly and the `I_g` shock is sized consistently); the τ_l target is dated `B[T_bal−1]/Y[T_bal−1]`; the results stamp `r`, `kappa`, `tau_beq`, `transfer_floor`.
  - Resource-constraint checks rewritten: `fill_report.goods_market_residual(..., budget=)` tests `C = Y + r·NFA_p − I − G − I_g − D − O − M − ΔNFA_p + PD` on the budget's own lines (the baseline now receives the spending shares), warning above 1%; `eval_fiscal_results.chk_goods_market` tests `C = Y + r·NFA + (r−r_B)B − I − purchases − M − ΔNFA`, failing above 2%.
  - `fill_report.py`: parameter table prints `n_0` and `n_∞`, every configured SMM parameter, the floor, `τ^beq` and `f`; UI/Y is not listed as untargeted when it is a target; the two identity rows of the implied table are labelled; header carries `\nTval` and `\GammaTval` (Γ_0 and Γ_T).
  - Panel: the unemployed record `l_sim = 0` (was the policy array's placeholder 1).
  - `compute_gini` with weights used a right-endpoint sum (understated every weighted Gini, e.g. 0.067 for 1..5 with equal weights instead of 0.267); fixed to the trapezoid rule. Pre-existing; affects the untargeted Gini rows of every calibration report to date.
  - `job_finding_rate`: `build_job_finding_GR.py` derives `f = 1 − long-term unemployment share` from Eurostat `une_ltu_a` (→ `data/job_finding_GR.json`); the configuration value is set from it.
- Bequest taxation with revenue accounting (`tau_beq` in OLGTransition budget). **Bequest measure (since 2026-10-02):** the recorded bequest of a dying agent is the wealth it carried out of the period, `(1+g)·a_grid[a_policy]` in current detrended units (both backends), not its beginning-of-period assets; zero at the terminal age, where a' = 0
- Simulation mortality draws with bequest tracking (`alive_sim`, `bequest_sim` — 21-tuple output)
- Population aging comes from the demographic sidecar (`transition.demography_file`, built by `build_demography_GR.py`): the EUROPOP2023 entering-cohort series IS the fertility path and the projected life tables ARE the longevity improvement. `fertility_path` and `_build_population_weights` were **deleted** 2026-10-01 — the latter multiplied the weights by cumulative survival, double-counting mortality the per-cohort means already carry, and discarded the measured entrant path. `fertility_path` now raises; `survival_improvement_rate` survives only for the legacy no-table path
- Data-driven cohort survival (`survival_table=(years, px)` in OLGTransition; opt-in via `transition.survival_data_file` → `data/survival_GR.npz`, built by `build_survival_GR.py`) — each cohort solved/simulated along its calendar diagonal of period life tables, clamped to the data range (cohort-historical past, held at last year for the future). Population weights stay births-only: survival is already baked into per-cohort means (dead agents hold 0, means divide by `n_sim`), so it must NOT also enter the weights. JAX batched solve/simulate carry survival per-cohort (`in_axes=0`).
- Per-cohort survival schedules (`_cohort_survival_schedule`, `_build_population_weights`)
- Heterogeneous initial wealth distribution (`initial_asset_distribution` in LifecycleConfig)
- Education-based heterogeneity
- All new features default OFF for backward compatibility

## Conventions

- **Aggregates are per living person.** Per-cohort means divide by `n_sim` with the dead at zero, so survival is already inside every mean and the weights must be cohort sizes **at entry** — putting survival in the weights too would double-count it. `_aggregation_weights(t)` divides those entry weights by the living share `_alive_fraction(t) = Σ_j w_j S_j`, which makes a weighted sum a per-capita average rather than a total per person *ever entered*. The invariant is `Σ_j W_j S_j = 1`. Until 2026-10-01 the division was missing, so every transition aggregate was 10.1% below the calibration's at t=0 — invisible in every ratio, since the factor cancels, and invisible to the balanced-growth flatness check, since the living share is constant on the balanced path. `calibrate.py` reaches the same convention from the other side: means among the alive, weighted by the measured living cross-section
- Policies indexed by `lifecycle_age` (not simulation time)
- Policy array shape: `(T, n_a, n_y, n_h, n_y_last)` — last dimension is previous income state (for pension calculation), not earnings history
- `m_grid` is `(T, 1)` — age-dependent medical costs; with `n_h=1`, effectively a `(T,)` age profile scaled by `m_good`
- `P_y` is `(n_y, n_y)` when constant, or `(T, n_h, n_y, n_y)` when age-dependent; `P_y_2d` always holds a 2D version
- Pensions use `i_y_last` state (last working period income state). `i_y_last` is the **previous period's** income state, not the last *employed* one: `z'_last = z` unconditionally while working (`lifecycle_perfect_foresight.py:1244`, `lifecycle_jax.py:765`), so it is 0 after one period of unemployment and UI pays zero from the second period of a spell onward
- Payroll tax applies to wages only, not pensions/UI
- `w_at_retirement` is cached in `__init__` (not recomputed per period)
- `n_sim` controls Monte Carlo simulation size
- Output plots saved to `output/` directory
- `simulate_transition` `results['L']` is in efficiency units (wage-valued `effective_y_sim` aggregate divided by `w_path`), matching calibrate.py's `L = (labor_income − ui) / w`; `_aggregate_capital_labor_njit` returns `(K, C, L)` — keep unpack order aligned. **Since 2026-10-02 UI is netted out before dividing by `w`, so `w·L` is the wage bill. Before that `effective_y_sim = wage_income + ui_sim` meant `L` carried `UI/w`** — 1.680 % of `w·L` at the 2026-10-01 calibration (1.887 % was the July r_B=0 run; the figure tracks the UI replacement and unemployment rates, so it is calibration-dependent), verified as `w·L − tax_p/τ^p = UI` at every t. Consequence: `L` exceeds `∫κ_j z e^α ℓ dμ`, `𝓑^lab ≠ w·L` (labour share 0.670 vs payroll base 0.657 of Y), and Y and K_domestic are above their model definitions by the same share; `results['K']` (household wealth A) is NOT — it is aggregated from `a_sim` and never touches L, so stripping UI leaves it unchanged and raises A/Y
- `_solve_period_wrapper` must stay module-level (required for `multiprocessing` pickling)
- All new features default to OFF (0.0, False, None) — setting defaults recovers pre-feature behavior exactly
- Fiscal G/I_g shocks pass `govt_spending_path=` and `I_g_path=` as explicit args to `simulate_transition()`; `transfer_floor=` (absolute value) is also an explicit arg — no external mutation needed
- GDP-share spending mode: G, I_g, defense, and other-net spending can be passed as **ratios of Y(t)** via `G_over_Y=/I_g_over_Y=/defense_over_Y=/other_net_over_Y=` (scalar or `(T,)`) instead of level paths. The budget then uses `level = ratio · Y_path[t]`, so each run's spending tracks its own realized output and the SS shares are preserved. A set ratio takes precedence over the level path for that line; ratios default None → level-path behavior (backward compatible). `run_fiscal_figures.py --config` uses ratio mode (shocks are ratio deltas, e.g. 0.02 = 2% of Y(t); `B_initial = B_over_Y · Y(0)`); the hardcoded test branch stays in level mode. `I_g_over_Y` is rejected when `eta_g != 0` (I_g→K_g→Y simultaneity needs a fixed point — pass an I_g level there)
- Mixed ratio/level mode (since 2026-07-10): with `eta_g != 0`, `run_fiscal_figures.py --config` passes I_g as a constant **level** `delta_g · K_g` (the stationary value, so baseline K_g stays flat) and a **level** Ig-shock delta `0.02 · Y(0)`, while G/defense/other stay ratios of Y(t). `_apply_shock` (fiscal_experiments.py) treats the I_g line per-line: level mode when `base_paths` has no `I_g_over_Y` key, ratio mode otherwise. Note the resulting G-vs-Ig shock asymmetry: G is 2% of each run's realized Y(t), Ig is 2% of initial Y, constant

## JAX Backend

- `lifecycle_jax.py` provides `LifecycleModelJAX` — same interface as `LifecycleModelPerfectForesight`
- Solve: vectorized grid search over all `(n_a, n_y, n_y_last)` states per period (with `n_h=1`), backward induction via `jax.lax.scan`
- Simulate: `jax.vmap` over agents, `jax.lax.scan` over time steps
- Uses `jax_enable_x64=True` for float64 precision (matches NumPy reference within ~1e-14)
- Different PRNG (ThreeFry vs MT19937): simulation paths differ individually but match distributionally
- `OLGTransition(backend='jax')` uses JAX for all cohort solves and simulations
- Batched cohort solve outer-loops the permanent-FE grid (`n_alpha` sweeps, one per α node); per-α policies stacked as `*_policy_alpha` with shape `(n_alpha, T, ...)`, scalar policies alias α=0. MIT stitching must write the `*_policy_alpha` arrays — both simulate paths read them, not the scalar 5-D policies
- Aggregation stays in NumPy (already fast with Numba, not a bottleneck)
- macOS ARM: use native ARM Python (x86 Python via Rosetta hits AVX issues with jaxlib); Linux x86_64: use `jax[cuda12]` for GPU or `jax[cpu]` for CPU-only
