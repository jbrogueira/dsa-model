# OLG Transition Model

Overlapping Generations (OLG) model simulating demographic and policy transitions with heterogeneous agents.

## Structure

```
olg_transition.py              # Entry point - OLG transition simulation
lifecycle_perfect_foresight.py # Lifecycle model with perfect foresight prices
test_olg_transition.py         # Pytest tests (17 tests)
ISSUES.md                      # Bug tracking
```

## Run

```bash
# Fast test mode
python olg_transition.py --test

# Full simulation (edit main() parameters)
python olg_transition.py

# Lifecycle standalone test
python lifecycle_perfect_foresight.py --test

# Run pytest suite
pytest test_olg_transition.py -v
```

## Key Classes

- `OLGTransition`: Manages transition dynamics, aggregation, government budget
- `LifecycleModelPerfectForesight`: Solves individual lifecycle problem via backward induction
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
