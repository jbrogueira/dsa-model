# OLG Transition Model

Overlapping Generations (OLG) model simulating demographic and policy transitions with heterogeneous agents.

## Structure

```
olg_transition.py              # Entry point - OLG transition simulation
lifecycle_perfect_foresight.py # Lifecycle model with perfect foresight prices (NumPy)
lifecycle_jax.py               # JAX-accelerated lifecycle model (solve + simulate)
fiscal_experiments.py          # Fiscal scenario framework (debt/tax/NFA-constrained experiments)
test_olg_transition.py         # Pytest tests (83 tests, incl. 12 JAX tests, 7 cross-validation)
test_fiscal_experiments.py     # Pytest tests for fiscal experiments (39 tests)
docs/IMPLEMENTATION_PLAN.md    # Feature implementation plan & progress
```

## Environment

JAX venv lives at `~/venvs/jax/`. Create it once with the setup script (auto-detects platform):

```bash
bash setup_jax.sh          # macOS ARM → jax[cpu], Linux x86_64 → jax[cuda12]
```

Activate before running the JAX backend:

```bash
source ~/venvs/jax/bin/activate
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
```

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
- `_solve_state_choice()`: Core solver for a single state — survival risk, labor supply FOC, age/health-dependent P_y.
- `_simulate_sequential()`: Monte Carlo forward simulation of agent paths.
- `_get_P_y()` / `_get_P_y_row()`: Index into 2D or 4D P_y (age/health-dependent transitions).
- `_survival_prob()`: Returns survival probability π(j,s), or 1.0 if no survival risk.
- `_solve_labor_hours()`: FOC-based optimal labor hours given consumption and wages.
- `_print_income_diagnostics()`: Verbose income process diagnostics (called from `__init__` when `verbose=True`).
- `simulate_transition()`: Accepts `recompute_bequests=False`, `bequest_tol=1e-4`, `max_bequest_iters=5` — runs a fixed-point bequest loop when `recompute_bequests=True` and `survival_probs` is set; stores `_bequest_converged` and `_bequest_iter_count` on `self`.
- `run_fiscal_scenario()` (`fiscal_experiments.py`): dispatcher — runs baseline + counterfactual via Type A/B/C experiment.
- `run_debt_financed()` / `run_tax_financed()` / `run_nfa_constrained()` (`fiscal_experiments.py`): Type A (one sim), Type B (bisection on scalar Δτ), Type C (Mode I: shock scale bisect; Mode II: tax rate bisect for NFA feasibility).
- `compare_scenarios()` / `fiscal_multiplier()` / `debt_fan_chart()` (`fiscal_experiments.py`): output utilities for plotting and multiplier calculation.

## Model Features

- Perfect foresight over price paths (r, w)
- Income risk (Tauchen discretization)
- Health shocks affecting productivity
- Retirement with pensions (based on last working income)
- Minimum pension floor (`pension_min_floor`)
- UI benefits for unemployed
- Multiple tax instruments (consumption, labor, payroll, capital)
- Progressive HSV taxation (`tax_progressive`, `tax_kappa`, `tax_eta`)
- Means-tested transfers / consumption floor (`transfer_floor`)
- Survival risk / stochastic mortality (`survival_probs`)
- Age-dependent medical expenditure (`m_age_profile`)
- Age/health-dependent productivity transitions (`P_y_by_age_health`)
- Endogenous labor supply via FOC (`labor_supply`, `nu`, `phi`)
- Endogenous retirement window (`retirement_window`)
- Schooling phase with child costs (`schooling_years`, `child_cost_profile`)
- Government spending on goods (`govt_spending_path` in OLGTransition)
- Public capital in production (`eta_g`, `K_g_initial`, `delta_g` in OLGTransition)
- Public investment path (`I_g_path` in OLGTransition)
- Small open economy with sovereign debt (`economy_type`, `r_star`, `B_path` in OLGTransition)
- Net foreign assets accounting (NFA) in SOE mode
- Pension trust fund (`S_pens_initial` in OLGTransition)
- Defense spending (`defense_spending_path` in OLGTransition)
- Bequest redistribution fixed-point loop (`recompute_bequests` in `simulate_transition()`) — closed bequest circuit iterates until bequests converge; production CLI defaults to `True`, test CLI defaults to `False` (opt-in via `--recompute-bequests`)
- Bequest taxation with revenue accounting (`tau_beq` in OLGTransition budget)
- Simulation mortality draws with bequest tracking (`alive_sim`, `bequest_sim` — 21-tuple output)
- Population aging: fertility path + longevity improvement (`fertility_path`, `survival_improvement_rate` in OLGTransition)
- Per-cohort survival schedules (`_cohort_survival_schedule`, `_build_population_weights`)
- Heterogeneous initial wealth distribution (`initial_asset_distribution` in LifecycleConfig)
- Education-based heterogeneity
- All new features default OFF for backward compatibility

## Conventions

- Policies indexed by `lifecycle_age` (not simulation time)
- Policy array shape: `(T, n_a, n_y, n_h, n_y_last)` — last dimension is previous income state (for pension calculation), not earnings history
- `m_grid` is `(T, n_h)` — age-dependent medical costs (was 1D `(n_h,)` before age-dependence)
- `P_y` is `(n_y, n_y)` when constant, or `(T, n_h, n_y, n_y)` when age/health-dependent; `P_y_2d` always holds a 2D version
- Pensions use `i_y_last` state (last working period income state)
- Payroll tax applies to wages only, not pensions/UI
- `w_at_retirement` is cached in `__init__` (not recomputed per period)
- `n_sim` controls Monte Carlo simulation size
- Output plots saved to `output/` directory
- `_solve_period_wrapper` must stay module-level (required for `multiprocessing` pickling)
- All new features default to OFF (0.0, False, None) — setting defaults recovers pre-feature behavior exactly
- Fiscal G/I_g shocks pass `govt_spending_path=` and `I_g_path=` as explicit args to `simulate_transition()`; `transfer_floor=` (absolute value) is also an explicit arg — no external mutation needed

## JAX Backend

- `lifecycle_jax.py` provides `LifecycleModelJAX` — same interface as `LifecycleModelPerfectForesight`
- Solve: vectorized grid search over all `(n_a, n_y, n_h, n_y_last)` states per period, backward induction via `jax.lax.scan`
- Simulate: `jax.vmap` over agents, `jax.lax.scan` over time steps
- Uses `jax_enable_x64=True` for float64 precision (matches NumPy reference within ~1e-14)
- Different PRNG (ThreeFry vs MT19937): simulation paths differ individually but match distributionally
- `OLGTransition(backend='jax')` uses JAX for all cohort solves and simulations
- Aggregation stays in NumPy (already fast with Numba, not a bottleneck)
- macOS ARM: use native ARM Python (x86 Python via Rosetta hits AVX issues with jaxlib); Linux x86_64: use `jax[cuda12]` for GPU or `jax[cpu]` for CPU-only
