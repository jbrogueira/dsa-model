# OLG Equilibrium Model with JAX

Overlapping Generations Economy with heterogeneous agents, incomplete markets, and equilibrium prices.

## Setup Instructions

### 1. Create a new conda environment
```bash
conda create -n olg python=3.10 -y
conda activate olg
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Select the environment in VS Code
- Open Command Palette (Cmd+Shift+P)
- Type "Python: Select Interpreter"
- Select the olg environment

## Running the Models

### Lifecycle Model with Perfect Foresight

The lifecycle model solves household optimization with time-varying prices and taxes.

#### Quick Test (Fast Mode)
```bash
cd code/
python lifecycle_perfect_foresight.py --test
```

#### Production Run (Full Resolution)
```bash
python lifecycle_perfect_foresight.py
```

#### Command-Line Options
```bash
# Test mode with custom parameters
python lifecycle_perfect_foresight.py --test --n-sim 500 --n-a 20 --T 30

# Production mode without plots (for batch runs)
python lifecycle_perfect_foresight.py --no-plots

# Sequential solving (disable parallel processing)
python lifecycle_perfect_foresight.py --no-parallel

# Custom simulations
python lifecycle_perfect_foresight.py --n-sim 20000 --n-a 150

# Show all options
python lifecycle_perfect_foresight.py --help
```

**Parameters:**
- `--test`: Fast mode with reduced grid (n_a=30, n_sim=1000, n_y=3)
- `--n-sim N`: Number of Monte Carlo simulations (default: 10000)
- `--n-a N`: Asset grid points (default: 100)
- `--T N`: Number of lifecycle periods (default: 40)
- `--no-plots`: Skip plotting (useful for server/batch runs)
- `--parallel/--no-parallel`: Enable/disable parallel solving across education types

**Default Settings:**
- Test mode: `n_a=30`, `n_y=3`, `n_sim=1000` (~5-10x faster)
- Production mode: `n_a=100`, `n_y=4`, `n_sim=10000` (full resolution)

### OLG Equilibrium Model
```bash
pethon olg_equilibrium_jax.py
```

## Model Features

### Lifecycle Model (`lifecycle_perfect_foresight.py`)

#### Heterogeneous Agents
- **Education heterogeneity**: Three education types (low, medium, high)
  - Different income processes (mean, volatility, persistence)
  - Low education: higher volatility (σ=0.25), lower persistence (ρ=0.93), lower mean (μ=-0.3)
  - Medium education: baseline (σ=0.20, ρ=0.95, μ=0.0)
  - High education: lower volatility (σ=0.15), higher persistence (ρ=0.97), higher mean (μ=0.4)
- **Income shocks**: Stochastic labor productivity with unemployment
  - Discretized using Tauchen method
  - Persistent AR(1) process
- **Health shocks**: Three health states (good, moderate, poor)
  - Age-dependent transition probabilities
  - Health affects labor productivity multiplicatively
  - Young (age < 40): mostly good health
  - Middle age (40 ≤ age < 60): health deteriorates
  - Old (age ≥ 60): high probability of poor health

#### Labor Market
- **Unemployment insurance**: Replacement rate based on last period's wage
  - UI benefit = ui_replacement_rate × w_t × y_last
  - Tracks last period's income state for UI calculation
- **Job dynamics**:
  - Endogenous job separation rate (targeted to steady-state unemployment)
  - Job finding rate for unemployed workers (δ=0.5)
  - Unemployment state in income process (y=0)
  - Target unemployment rate: 6%

#### Health Expenditures
- **Out-of-pocket costs**: Households pay (1-κ) × m(h)
  - κ = 0.7 (government covers 70%)
- **Government coverage**: Government covers κ × m(h)
- **Age-dependent health**: Worse health in old age
- **Expenditure by health state**:
  - Good health: m = 0.05 (5% of income)
  - Moderate health: m = 0.15 (15% of income)
  - Poor health: m = 0.30 (30% of income)
- **Budget impact**: Health expenditure enters budget constraint

#### Taxes and Government Policy
- **Consumption tax** (τ_c): Ad-valorem tax on consumption
  - Applied to final consumption: (1 + τ_c) × c
- **Labor income tax** (τ_l): Progressive taxation on labor income
  - Applied to gross labor income minus payroll tax
- **Payroll tax** (τ_p): Social security contributions
  - Applied to wage income only (not UI benefits)
- **Capital income tax** (τ_k): Tax on asset returns
  - Applied to interest income: τ_k × r × a
- **Time-varying tax rates**: Perfect foresight over lifecycle
  - Can be constant or follow arbitrary time path

#### Perfect Foresight
Agents know future paths of:
- **Interest rates** {r_t}: Return on savings
- **Wages** {w_t}: Labor market prices
- **Tax rates** {τ_c,t, τ_l,t, τ_p,t, τ_k,t}: All tax instruments
- Used for transition dynamics in OLG equilibrium

#### Budget Constraint
```
(1 + τ_c) × c + a' + (1 - κ) × m(h) = 
    a + (1 - τ_k) × r × a + after_tax_labor_income + UI
```
where:
- `after_tax_labor_income = gross_labor_income - τ_p × wage_income - τ_l × (gross_labor_income - τ_p × wage_income)`
- `UI = ui_replacement_rate × w × y_last` (if unemployed)

#### Financial Markets
- **Incomplete markets**: Borrowing constraint (a ≥ a_min)
  - Default: a_min = -2.0 (limited borrowing)
- **Self-insurance**: Precautionary savings against income and health shocks

#### Solution Method
- **Backward induction**: Value function iteration from terminal period
- **State space**: (t, a, y, h, y_last)
  - t: age/period
  - a: assets
  - y: current income state
  - h: health state
  - y_last: last period's income (for UI calculation)
- **Grid search**: Optimize over discrete asset choice
- **Non-linear asset grid**: More points near borrowing constraint
- **Expected continuation value**: Integrate over future income and health shocks
- **Parallel solving**: Education types solved simultaneously (multiprocessing)
- **Monte Carlo simulation**: Large-scale simulations for aggregation (10,000 paths)

#### Output
- **Policy functions**: Savings and consumption by state
  - Stored as arrays: `a_policy[t, i_a, i_y, i_h, i_y_last]`
  - Value function: `V[t, i_a, i_y, i_h, i_y_last]`
- **Lifecycle profiles**: Age-profiles of assets, consumption, income by education
  - Average assets over lifecycle
  - Average consumption over lifecycle
  - Average effective income (y × h)
- **Statistics by education**: 
  - Mean assets, consumption, income
  - Unemployment rates
  - Tax payments (by type: consumption, labor, payroll, capital)
  - Government spending (UI + health coverage)
- **Plots**: Comparison across education types
  - 6-panel figure showing lifecycle profiles
  - Saved to `output/` directory
- **No aggregation**: Each education type solved separately
  - Aggregation done in `olg_equilibrium.py` with education distribution
  - Returns dictionary: `models[edu_type]` for each education type

### File Organization
```
code/
├── lifecycle_perfect_foresight.py  # Household lifecycle problem
├── olg_equilibrium_jax.py          # General equilibrium solver
├── output/                          # Generated plots and results
│   ├── education_comparison.png    # Lifecycle profiles by education
│   └── education_comparison_test.png
└── README.md
```

### Workflow
1. **Lifecycle model** solves household problem for each education type
   - Backward induction with perfect foresight
   - Three education types in parallel
2. **Simulation** generates individual histories
   - Monte Carlo with 10,000 paths per education type
   - Tracks assets, consumption, income, employment, health

## Output

Running the lifecycle model generates:
- **Console output** with solution progress and statistics
  - Solution time for each education type
  - Summary statistics (means, unemployment rates)
  - Tax revenues and government spending
- **Plots** comparing education types (unless `--no-plots`)
  - Assets, consumption, income over lifecycle
  - Taxes and government spending
  - Unemployment rates
- **Saved figures** in `output/` directory

**Test mode** output: `education_comparison_test.png`
**Production mode** output: `education_comparison.png`

## Performance

### Test Mode (`--test`)
- Solution time: ~5-10 seconds (3 education types in parallel)
- Memory usage: ~500 MB
- Grid size: 30 × 3 × 3 × 3 = 810 states per age
- Simulations: 1,000 paths per education type
- Total states: 40 ages × 810 states × 3 = ~97,200

### Production Mode
- Solution time: ~30-60 seconds (3 education types in parallel)
- Memory usage: ~2 GB
- Grid size: 100 × 4 × 3 × 4 = 4,800 states per age
- Simulations: 10,000 paths per education type
- Total states: 40 ages × 4,800 states × 4 = ~768,000

### Scaling
- **Asset grid** (n_a): Linear impact on memory, solution time
- **Income states** (n_y): Quadratic impact (more transitions)
- **Simulations** (n_sim): Linear impact on simulation time only
- **Parallel solving**: Near-linear speedup with CPU cores

## Troubleshooting

### Multiprocessing Issues on Mac
If parallel solving fails with pickle errors:
```bash
python lifecycle_perfect_foresight.py --test --no-parallel
```

This solves education types sequentially (slower but more reliable).

### Memory Issues
Reduce grid size for large-scale runs:
```bash
python lifecycle_perfect_foresight.py --n-a 50 --n-sim 5000
```

Or use test mode parameters:
```bash
python lifecycle_perfect_foresight.py --test --T 30
```

### Plotting Issues
Skip plots if display not available (e.g., on server):
```bash
python lifecycle_perfect_foresight.py --test --no-plots
```

### Convergence Issues
If value function iteration doesn't converge:
- Check parameter values (especially β and r)
- Ensure r < 1/β - 1 for finite value
- Try smaller time horizon (--T 30)

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Note for Apple Silicon Macs
This uses CPU-only JAX. The Metal backend line is commented out to ensure compatibility across all machines.

## Technical Details

### State Space Dimensions
- **Time** (t): 40 periods (age 20-60)
- **Assets** (a): 100 points (production), 30 points (test)
- **Income** (y): 4 states including unemployment (production), 3 (test)
- **Health** (h): 3 states (good, moderate, poor)
- **Last income** (y_last): 4 states (for UI calculation)

### Numerical Methods
- **Tauchen discretization**: Income process
- **Grid search**: Asset optimization
- **Backward induction**: Dynamic programming
- **Monte Carlo**: Simulation and aggregation
- **Multiprocessing**: Parallel across education types

### Computational Optimizations
- **Non-linear asset grid**: More points near constraint
- **Median smoothing**: Robust policy averaging
- **Numba JIT**: Fast utility function
- **Vectorization**: Efficient array operations
