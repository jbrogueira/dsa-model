# OLG Transition Model

Overlapping Generations (OLG) model simulating demographic and policy transitions with heterogeneous agents.

## Structure

```
olg_transition.py              # Entry point - OLG transition simulation
lifecycle_perfect_foresight.py # Lifecycle model with perfect foresight prices (NumPy)
lifecycle_jax.py               # JAX-accelerated lifecycle model (solve + simulate)
test_olg_transition.py         # Pytest tests (39 tests, incl. 10 JAX cross-validation)
docs/IMPLEMENTATION_PLAN.md    # Feature implementation plan & progress
```

## Environment

JAX venv is stored at `~/venvs/jax-arm/` (Python 3.11, ARM/Homebrew, native Apple Silicon). Activate before running JAX backend:

```bash
source ~/venvs/jax-arm/bin/activate
```

## Run

```bash
# Fast test mode (NumPy backend, default)
python olg_transition.py --test

# Fast test mode (JAX backend)
python olg_transition.py --test --backend jax

# Full simulation
python olg_transition.py
python olg_transition.py --backend jax

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

## Key Methods

- `_compute_budget()`: Computes after-tax income and budget for any state (retirement/working). Handles pension floor, progressive tax, child costs, transfer floor, age-dependent medical.
- `_solve_period()`: Solves a single period. Handles retirement window (discrete choice between working/retired).
- `_solve_state_choice()`: Core solver for a single state — survival risk, labor supply FOC, age/health-dependent P_y.
- `_simulate_sequential()`: Monte Carlo forward simulation of agent paths.
- `_get_P_y()` / `_get_P_y_row()`: Index into 2D or 4D P_y (age/health-dependent transitions).
- `_survival_prob()`: Returns survival probability π(j,s), or 1.0 if no survival risk.
- `_solve_labor_hours()`: FOC-based optimal labor hours given consumption and wages.
- `_print_income_diagnostics()`: Verbose income process diagnostics (called from `__init__` when `verbose=True`).

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
- Bequest taxation config (`tau_beq`)
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

## JAX Backend

- `lifecycle_jax.py` provides `LifecycleModelJAX` — same interface as `LifecycleModelPerfectForesight`
- Solve: vectorized grid search over all `(n_a, n_y, n_h, n_y_last)` states per period, backward induction via `jax.lax.scan`
- Simulate: `jax.vmap` over agents, `jax.lax.scan` over time steps
- Uses `jax_enable_x64=True` for float64 precision (matches NumPy reference within ~1e-14)
- Different PRNG (ThreeFry vs MT19937): simulation paths differ individually but match distributionally
- `OLGTransition(backend='jax')` uses JAX for all cohort solves and simulations
- Aggregation stays in NumPy (already fast with Numba, not a bottleneck)
- Requires ARM Python on Apple Silicon (x86 Python via Rosetta hits AVX issues with jaxlib)
