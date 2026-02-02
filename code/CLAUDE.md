# OLG Transition Model

Overlapping Generations (OLG) model simulating demographic and policy transitions with heterogeneous agents.

## Structure

```
olg_transition.py              # Entry point - OLG transition simulation
lifecycle_perfect_foresight.py # Lifecycle model with perfect foresight prices (NumPy)
lifecycle_jax.py               # JAX-accelerated lifecycle model (solve + simulate)
test_olg_transition.py         # Pytest tests (20 tests, incl. 3 JAX cross-validation)
ISSUES.md                      # Bug tracking
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

- `_compute_budget()`: Computes after-tax income and budget for any state (retirement/working). Used by `_solve_period`, `_solve_period_wrapper`, avoids duplication.
- `_solve_period()`: Solves a single period including the terminal period (`t == T-1` consumes everything, `a' = 0`).
- `_simulate_sequential()`: Monte Carlo forward simulation of agent paths.
- `_print_income_diagnostics()`: Verbose income process diagnostics (called from `__init__` when `verbose=True`).

## Model Features

- Perfect foresight over price paths (r, w)
- Income risk (Tauchen discretization)
- Health shocks affecting productivity
- Retirement with pensions (based on last working income)
- UI benefits for unemployed
- Multiple tax instruments (consumption, labor, payroll, capital)
- Education-based heterogeneity

## Conventions

- Policies indexed by `lifecycle_age` (not simulation time)
- Policy array shape: `(T, n_a, n_y, n_h, n_y_last)` — last dimension is previous income state (for pension calculation), not earnings history
- Pensions use `i_y_last` state (last working period income state)
- Payroll tax applies to wages only, not pensions/UI
- `w_at_retirement` is cached in `__init__` (not recomputed per period)
- `n_sim` controls Monte Carlo simulation size
- Output plots saved to `output/` directory
- `_solve_period_wrapper` must stay module-level (required for `multiprocessing` pickling)

## JAX Backend

- `lifecycle_jax.py` provides `LifecycleModelJAX` — same interface as `LifecycleModelPerfectForesight`
- Solve: vectorized grid search over all `(n_a, n_y, n_h, n_y_last)` states per period, backward induction via `jax.lax.scan`
- Simulate: `jax.vmap` over agents, `jax.lax.scan` over time steps
- Uses `jax_enable_x64=True` for float64 precision (matches NumPy reference within ~1e-14)
- Different PRNG (ThreeFry vs MT19937): simulation paths differ individually but match distributionally
- `OLGTransition(backend='jax')` uses JAX for all cohort solves and simulations
- Aggregation stays in NumPy (already fast with Numba, not a bottleneck)
- Requires ARM Python on Apple Silicon (x86 Python via Rosetta hits AVX issues with jaxlib)
