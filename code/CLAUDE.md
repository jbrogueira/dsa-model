# OLG Transition Model

Overlapping Generations (OLG) model simulating demographic and policy transitions with heterogeneous agents.

## Structure

```
olg_transition.py              # Entry point - OLG transition simulation
lifecycle_perfect_foresight.py # Lifecycle model with perfect foresight prices
test_olg_transition.py         # Pytest tests
ISSUES.md                      # Bug tracking
```

## Run

```bash
# Fast test mode
python olg_transition.py --test

# Full simulation (edit main() parameters)
python olg_transition.py
```

## Key Classes

- `OLGTransition`: Manages transition dynamics, aggregation, government budget
- `LifecycleModelPerfectForesight`: Solves individual lifecycle problem via backward induction
- `LifecycleConfig`: Configuration dataclass for lifecycle parameters

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
- Pensions use `y_last` state (last working period income)
- Payroll tax applies to wages only, not pensions/UI
- `n_sim` controls Monte Carlo simulation size
- Output plots saved to `output/` directory
