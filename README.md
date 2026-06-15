# OLG Equilibrium Model with JAX

Overlapping Generations Economy with heterogeneous agents, incomplete markets, and equilibrium prices. Application: Greek fiscal transition (debt sustainability under G / I_g shocks).

---

## Current status (handoff 2026-06-15)

### Scenario assumptions documented + τ_l closure change (2026-06-15)

- **`code/docs/FISCAL_SCENARIOS.md`** (new): the assumptions behind each fiscal scenario, in the paper's notation (`DSA-LSA model.tex` / `experiments.tex`) — environment, primary balance, debt law, shocks, the two financing rules, and a section flagging where the implementation currently diverges from the draft's experiments section.
- **τ_l closure changed** (`run_fiscal_figures.py`): the labour-tax scenarios now target the **baseline transition's terminal debt/GDP** (`terminal_debt_gdp`, resolved at runtime), replacing the static `terminal_flow_balance` at `b*=1.64`. Committed; not yet re-run.
- **Public capital (K_g) activation — in progress, NOT committed.** `code/docs/PUBLIC_CAPITAL_KG_PLAN.md` records a change set that turns on `η_g=0.05`, `K_g=1`; the `calibration_input_GR.json` edit is held back because `δ_g=0` makes the baseline `K_g` non-stationary (the plan lists δ_g + A_tfp-normalization as the remaining steps). Pick this up from that plan.

### Baseline closure pinned at the initial SS + spending as GDP shares (2026-06-14, later passes)

Two consistency changes to the baseline fiscal setup, both under the recalibrated θ. Detail: `code/docs/FISCAL_EXPERIMENTS_STATUS.md` (`## Session 2026-06-14 (cont.)` and `(cont. 2)`).

- **Closure pinned at the initial steady state (no transition).** `other_net_spending_over_Y` is now a structural constant that makes the initial-point government budget match `fiscal.primary_balance_target_over_Y = 0.0195`; the baseline transition takes it as given and its t=0 primary balance need not equal the target exactly. Computed from the stationary calibration panels — one solve, no transition — via `pin_baseline_closure.py` (replaces the transition-based `measure_baseline_closure.py`). `compute_fiscal_ratios` now also returns `primary_balance_full_over_Y` and `closure_other_over_Y`. Value: `other_net_spending_over_Y = −0.091122` (was −0.0889, the transition-t=0 value; the gap is the SS-vs-t=0 cross-section difference, documented in the status doc).
- **Exogenous spending lines are fixed shares of Y(t).** G, I_g, defense, and other-net were fixed levels (`ratio × meanY`); they are now `ratio × Y(t)`, so the SS shares are held and levels track each run's realized output. Decisions: each scenario indexes to its **own** Y(t); G/I_g shocks are **2% of Y(t)**; `B_initial = B_over_Y · Y(0)`; **gov_health unchanged** (real per-agent medical cost, not an imposed GDP share). Additive implementation (`G_over_Y/I_g_over_Y/defense_over_Y/other_net_over_Y` kwargs on `simulate_transition`; budget forms `level = ratio · Y_path[t]`); ratios default None → exact level-path behavior. `I_g_over_Y` rejected when `eta_g ≠ 0` (I_g→K_g→Y simultaneity).
- **Verified:** ratio mode gives G/Y exactly the target share each period; debt-financed G shock leaves Y unchanged (no SOE feedback); A[0] predetermination exact; level mode unchanged; `test_fiscal_experiments.py` 39/39 pass.
- **STILL STALE — next step:** re-run the fiscal figures under new θ + SS-pinned closure + GDP-share spending: `python run_fiscal_figures.py --config calibration_input_GR.json --shock both --backend jax`. Everything in `output/fiscal_test/` predates these.

### Labor-supply FOC fixed + recalibrated (2026-06-14)

The one-period labor "spike" at transition t=1 (seen in the fiscal figures) was traced to the labor-supply FOC, fixed in both backends, and the SMM recalibrated under the corrected solver. Detail: `code/docs/FISCAL_EXPERIMENTS_STATUS.md` (`## Session 2026-06-14`).

- **Cause:** the 2-iteration Newton labor solver was seeded in a region where implied consumption is negative (clamped), freezing it at a spurious non-root — hours off by up to ~0.4 at borrowing-constrained ages; MIT alignment turned each cohort's onset wobble into the aggregate t=1 spike. An independent audit (theorist + code-extractor) also found the FOC equation itself wrong three ways: additive tax wedge `(1−τ_l−τ_p)` instead of the budget's `(1−τ_p)(1−τ_l)`, κ(t) omitted in NumPy, and the `1/(1+τ_c)` consumption-tax wedge omitted; plus a spurious labor disutility charged to retired/unemployed agents.
- **Fix:** robust **projected-Newton** root-find on the monotone residual, bracketed to the feasible region, targeting the corrected FOC `ν·l^φ = c^{−γ}·MW/(1+τ_c)` with `MW = w·κ(t)·y·h·e^α·(1−τ_p)(1−τ_l)`; disutility only for working-age employed. `lifecycle_jax.py: solve_labor_robust_jax` (8 iters, vectorized), `lifecycle_perfect_foresight.py: _solve_labor_newton` (scalar, early-break).
- **Verified:** both solvers match an exact bracketing root-finder to ~1e-14; A[0] predetermination exact; NumPy and JAX agree to MC-noise; the t=1 spike is gone (per-age deviation now mixed-sign noise vs the old uniform lift); suites green (fiscal 39/39, OLG 73 passed/17 deselected).
- **Recalibrated:** SMM re-run under the corrected solver; converged (obj ≈ 2e-8), all five moments match targets to ≤0.01%. New θ in `calibration_input_GR.json._derived.theta`:

  | param | new | old (pre-fix) |
  |---|---|---|
  | ν | 36.907 | 28.670 |
  | β | 0.94317 | 0.98541 |
  | τ_p | 0.19776 | 0.19786 |
  | ρ_pens | 0.16629 | 0.16122 |
  | m | 0.04277 | 0.04162 |

  ν +29% and β −4.2pp are the FOC corrections propagating (ν up to still hit 0.41 hours, β down to still hit A/Y = 4).
- **Follow-up (done in later passes this session):** the closure was re-pinned under the new θ — now at the initial SS via `pin_baseline_closure.py` (see the section above), giving `other_net_spending_over_Y = −0.091122`. The fiscal figures remain to be re-run.

### Prior session (2026-06-11/12)

Detailed notes in `code/docs/FISCAL_EXPERIMENTS_STATUS.md` (`## Session 2026-06-12` and `## Session 2026-06-11`).

### SS-vs-transition gap RESOLVED (2026-06-12)

A multi-agent code audit (4 blind auditors + 3 adversarial verifiers, code-only) found the gap was bugs, not economics. All fixed and verified same day:

1. **K/L/C unpack swap** (since `9496ee1`, 2026-03-04): `_aggregate_capital_labor_njit` returns `(K, C, L)`; `simulate_transition` unpacked `(K, L, C)` — Y, K_domestic, and NFA were built from aggregate **consumption**. Fixed at `code/olg_transition.py` (~line 2072).
2. **L-units convention**: the transition fed the wage-valued labor aggregate into production; `calibrate.py` divides by w first. Fixed: `L_path = L_path / w_path` — `results['L']` is now in efficiency units.
3. **JAX batched α bug**: batched solve used only `alpha_mult=1.0` while the batched simulate drew α over the full grid (lookup clamped to the singleton axis → α-neutral policies with α-scaled incomes). Fixed: outer loop over α nodes; per-α policies bitwise-match the standalone JAX solve.
4. **MIT-stitching staleness**: stitching rebound only the scalar 5-D policies while both simulate paths read `*_policy_alpha` — a no-op for the simulation. Fixed: α arrays stitched too; A[0] predetermination re-verified exactly 0.0 on both backends with n_alpha=3 (`code/check_a0_predetermination.py`).

**Post-fix validation**: all item ratios match the calibration SS to ±0.6% (tax_p/Y 0.130 = SMM target exactly); the remaining −11.5% Y-level difference is the per-birth vs per-living normalization, which cancels in ratios (expected). Closure re-measured: baseline primary surplus −6.94% of Y → `other_net_spending_over_Y = −0.0889` (config updated; the plug is now below the data's 9.4% other-revenue line). Suites green: fiscal 39/39, OLG 73 passed / 17 deselected (documented exclusions). **All transition-based results produced 2026-03-04 → 2026-06-12 are invalid; the SMM θ is unaffected** (standalone pipeline, no bug touches it).

### Also this session (2026-06-11)

- **Multiplier 0.000 diagnosed — structural, not numerical.** A debt-financed G shock leaves every household-relevant path (r, w, taxes) unchanged in SOE mode, so ΔY ≡ 0 by construction; only B/GDP and NFA move. `tax_financed` and I_g shocks give ΔY ≠ 0. Which scenario/shock to report a multiplier from is an open modeling decision. (Unaffected by the bugs.)
- Secondary audit findings recorded in the status doc: the SMM's pooled moments (e.g. `average_hours`) double-count survival relative to its ratio moments; `_at` zero-fills exogenous spending paths beyond length while `B_path` clamps; `transition.recompute_bequests` in the JSON is never read (CLI flag only).

### Pre-counterfactual reconciliation + cleanup (2026-06-12, second pass)

- **Pipeline wiring checked, ready for counterfactual runs**: `fiscal.B_over_Y = 1.64` is the live debt input everywhere (baseline interest = 3.44% of Y, matching 2023 data); post-horizon spending paths clamp-extend correctly; the closure −0.0889 carries over to `run_fiscal_figures` as measured (same n_sim, scaling, shares; bequest circuit open on both).
- **Multiplier item dropped**: ΔY ≡ 0 for debt-financed G is the model's theoretical implication (exogenous r, wasteful G, foreign borrowing); the 0.000 output line is that fact, nothing to decide.
- **Dead code/config eliminated**, output verified bit-identical pre/post: dead `transition.{B_over_Y, G_over_Y, I_g_over_Y, recompute_bequests}` JSON keys; the unconsumed SS-profiles block in `solve_cohort_problems` (3 lifecycle solves per call — speedup); `use_initial_distribution`; `_jax_policy_batch`; `compute_aggregates()` L now in efficiency units; pyflakes-driven unused-import/dead-local removal across all modules.

### Next steps

1. **Re-run fiscal figures** under the fixed code and new closure (`run_fiscal_figures.py --config calibration_input_GR.json --shock both --backend jax`) — all existing output in `code/output/fiscal_test/` predates the fixes.
2. Optional: SMM pooled-moment survival double-count (e.g. `average_hours` weights alive observations by survival-inclusive ω(t) — survival enters twice relative to the ratio moments).

### What landed (2026-05-26 → 28, previously uncommitted)

- **Plumbing + `other_net_spending`.** Wired `I_g_path` / `defense_spending_path` into the baseline budget (were silently 0); added an exogenous `other_net_spending` net-primary-spending line to pin the baseline primary balance.
- **5-parameter SMM.** Calibrated `(ν, β, τ_p, ρ_pens, m)` to (hours, A/Y, SSC/Y, pensions/Y, health_gov/Y), all matching ≤1.2%. `_derived.theta`: ν=27.30, β=0.977, τ_p=0.197, ρ_pens=0.147, m=0.0393.
- **Baseline fiscal closure.** `other_net_spending_over_Y = −0.1056` so the transition baseline primary balance equals Greek 2023 (+1.95%).

### What landed (2026-06-09)

- **SS-vs-transition diagnostic.** The transition no-shock baseline is flat over all 60 periods (a steady state) but at a different level than the calibration SS (Y +7.3%, budget items/Y ~12–16% lower). It is a level mismatch, not a transient.
- **Data-driven cohort survival.** Greek Eurostat life tables (`demo_mlifetable` px, 1961–2023, real ages 25–84) → `data/survival_GR.npz` via `build_survival_GR.py`. `OLGTransition` gains `survival_table=(years, px)`, opt-in via `transition.survival_data_file`. Each cohort is solved/simulated along its calendar diagonal — cohort-historical for the past, held at 2023 for future transition years.
- **JAX per-cohort survival bug fixed.** The batched solve/simulate passed cohort-0's survival as a shared vmap arg; now `in_axes=0` (per-cohort). Validated: 2-cohort solve with survival 1.0 vs 0.7 gives distinct policies. No regressions (14 JAX tests pass; 2 failures are pre-existing, confirmed by stashing).
- **Recalibrated against the data 2020 life table** (`023bdfa`). The prior `survival_probs` was hand-entered, off the Eurostat data by up to 0.037 (px₈₄ 0.960 vs data 0.923). Re-sourced from `survival_GR.npz` (year 2020 = transition `current_year`); both calibration uses (lifecycle solve, `compute_age_weights`) read the JSON array. 5-param SMM converged, all moments match. New `_derived.theta`: **ν=28.67, β=0.985, τ_p=0.198, ρ_pens=0.161, m=0.0416** (was 27.30 / 0.977 / 0.197 / 0.147 / 0.0393). Lower old-age survival raises β (weaker effective discounting β·π) and ρ_pens / m (fewer survivors reach pension / high-medical ages).

### Correction recorded 2026-06-09 (weighting algebra — still correct)

The transition bakes survival into its per-cohort means (dead agents hold 0; means divide by `n_sim`), so `mean = survival · E[X|alive]` — algebraically identical to calibrate's `births·S · E[X|alive]` for **ratios**; adding survival to the weights double-counts. (The 2026-06-09 attribution of the residual gap to bequests/behavior was superseded on 2026-06-12: the gap was the unpack-swap and L-units bugs above.)

### Code state at end of session (2026-06-12)

- Calibration unchanged at `_derived.theta` = {ν=28.67, β=0.985, τ_p=0.198, ρ_pens=0.161, m=0.0416}, survival from the data 2020 life table. **SMM θ unaffected by the bugs.**
- `code/olg_transition.py`: unpack fix + `L_path / w_path` in `simulate_transition`; per-α outer loop in `_solve_cohorts_jax_batched`; `*_policy_alpha` stitching in both MIT blocks. `results['L']` is now in efficiency units.
- `code/calibration_input_GR.json`: `other_net_spending_over_Y = −0.0889`.
- New scripts kept: `code/measure_baseline_closure.py` (closure re-derivation), `code/diag_bequest_decomp.py` (bequest on/off decomposition; its 2026-06-11 numbers are pre-fix), `code/check_a0_predetermination.py` (A[0] invariant with α heterogeneity).
- All output under `code/output/fiscal_test/` predates the fixes — stale.
- Working tree: all of the above uncommitted.

---

## Setup Instructions

### 1. Create the virtual environment (once per machine)
```bash
bash make_venv.sh
```
Creates `.venv/` in the project root. Installs `jax[cuda12]` on Linux, plain `jax` on macOS, plus `requirements.txt`.

### 2. Activate before each coding session
```bash
source .venv/bin/activate
```

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
- Test mode (`--test`): `n_a=100`, `n_y=2`, `n_sim=100`
- Full simulation (no flags): `n_a=100`, `n_y=2`, `n_sim=5000`
- Config mode (`--config <file>`): dimensions set in the JSON file; `LifecycleConfig` default is `n_a=100`, `n_y=5`

### OLG Transition Model
```bash
python olg_transition.py --test                                    # fast test (hardcoded params)
python olg_transition.py --config calibration_input_GR.json        # from JSON config
python olg_transition.py --config calibration_input_GR.json --backend jax --n-sim 5000
```

### Fiscal Experiments
```bash
python run_fiscal_figures.py --shock G                             # fast test (hardcoded params)
python run_fiscal_figures.py --config calibration_input_GR.json --shock G
python run_fiscal_figures.py --config calibration_input_GR.json --shock both --backend jax
```

### SMM Calibration
```bash
python calibrate.py --config calibration_input_GR.json --backend jax
python calibrate.py --config calibration_input_GR.json --n-sim 20000 --maxiter 500
python calibrate.py --test                                         # smoke test
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
- **Health expenditure**: Deterministic age-dependent medical cost
  - No stochastic health states (`n_h=1`)
  - Government covers fraction `κ`; household pays `(1 − κ)`
  - Age profile from national health accounts (`m_age_profile`)

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
- **Out-of-pocket costs**: Households pay (1-κ) × m(j)
  - κ calibrated from national health accounts (government coverage share)
- **Government coverage**: Government covers κ × m(j)
- **Age-dependent level**: `m(j) = m_age_profile[j] × m_good`, rising with age
- **Budget impact**: Health expenditure enters budget constraint as a deterministic cost

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
  - Default: a_min = 0.0 (no borrowing)
- **Self-insurance**: Precautionary savings against income shocks

#### Solution Method
- **Backward induction**: Value function iteration from terminal period
- **State space**: (t, a, y, y_last)
  - t: age/period
  - a: assets
  - y: current income state
  - y_last: last period's income (for UI calculation)
- **Grid search**: Optimize over discrete asset choice
- **Non-linear asset grid**: More points near borrowing constraint
- **Expected continuation value**: Integrate over future income shocks
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
  - Aggregation done in `olg_transition.py` with education distribution
  - Returns dictionary: `models[edu_type]` for each education type

### File Organization
```
code/
├── lifecycle_perfect_foresight.py  # Lifecycle household problem (NumPy backend)
├── lifecycle_jax.py                # Lifecycle household problem (JAX backend)
├── olg_transition.py               # OLG transition dynamics + aggregation + budget
├── fiscal_experiments.py           # Fiscal-scenario framework (debt/tax/NFA branches)
├── run_fiscal_figures.py           # CLI: --shock {G, Ig, both} → figures + JSON
├── calibrate.py                    # SMM calibration of (ν, β); JSON config I/O
├── eval_fiscal_results.py          # Post-processing on fiscal_results.json
├── test_*.py                       # pytest suites
├── calibration_input_GR.json       # Greek-country JSON config
├── docs/                           # Plan, status, audit, architecture docs
└── output/                         # Generated figures, calibration reports, fiscal_results.json
```

See `code/CLAUDE.md` for the full method/class index.

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

## JAX backends by platform
- **Linux + NVIDIA GPU**: `make_venv.sh` installs `jax[cuda12]`; JAX runs on CUDA automatically.
- **macOS (Apple Silicon or Intel)**: `make_venv.sh` installs plain `jax`; JAX runs on CPU. Metal/MPS is not enabled.

## Technical Details

### State Space Dimensions
- **Time** (t): `T` periods (default 45; test mode 20, full simulation 60)
- **Assets** (a): `n_a` grid points (default 100)
- **Income** (y): `n_y` states including unemployment (default 5; built-in demos use 2)
- **Health** (h): `n_h` states (default 1 — no stochastic health process)
- **Last income** (y_last): `n_y` states (for pension/UI calculation)

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
