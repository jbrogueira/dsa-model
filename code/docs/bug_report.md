# Bug Report

Chronological log of bugs found, diagnosed, and fixed in the OLG transition model.

---

## BUG-001 — JAX vmap `in_axes` length mismatch

**Date:** 2026-02-24
**Status:** Fixed
**Files:** `olg_transition.py`

### Symptom
```
ValueError: vmap in_axes must be an int, None, or a tuple of entries corresponding
to the positional arguments passed to the function, but got len(in_axes)=37, len(args)=36
```
Raised when running `python olg_transition.py --test --backend jax`.

### Root Cause
`_simulate_lifecycle_jax_batched` is defined with 37 `in_axes` entries (the last being `None` for `survival_probs`), but the call site in `_simulate_cohorts_jax_batched` only passed 36 positional arguments — `survival_probs` was missing.

### Fix
Added `ref.survival_probs` as the last positional argument to the `_simulate_lifecycle_jax_batched` call in `_simulate_cohorts_jax_batched`.

---

## BUG-002 — GPU out-of-memory in full JAX simulation

**Date:** 2026-02-24
**Status:** Fixed
**Files:** `olg_transition.py`

### Symptom
```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to
allocate 453.19MiB
```
Raised during the batched JAX simulation in full production mode (99 cohorts × 15 000 agents).

### Root Cause
`_simulate_lifecycle_jax_batched` was called with all 99 cohorts simultaneously, requiring ~15 GB of GPU VRAM for the full (cohorts × agents × T × 21 arrays × 8 bytes) tensor.

### Fix
Added `jax_sim_chunk_size` parameter to `OLGTransition.__init__`. The simulation loop in `_simulate_cohorts_jax_batched` now iterates over fixed-size cohort chunks. The last chunk is padded to the same size as other chunks to keep XLA tensor shapes static and avoid recompilation. Default for full simulation: `jax_sim_chunk_size=10`.

---

## BUG-003 — Constant aggregate cross-sections throughout transition

**Date:** 2026-02-24
**Status:** Fixed
**Files:** `olg_transition.py`

### Symptom
- Aggregate capital K, labor L, output Y, and all government budget quantities were exactly constant for the entire transition.
- Lifecycle profiles of cohorts born at t=0 and t=5 were indistinguishable.
- Capital income tax revenue constant despite changing interest rate and wages.

### Root Cause
In `solve_cohort_problems`, old cohorts (born at `birth_period < 0`) were assigned their steady-state wealth at lifecycle **age k** (where k = |birth_period|), but this value was placed at **age 0** of the simulation. The age-0 policy function then mapped age-k wealth incorrectly, producing distorted trajectories. Because every cohort's simulation was anchored to the SS asset level regardless of age, the cross-sectional aggregates remained frozen at SS values throughout the transition.

The offending block:
```python
for edu_type in self.education_shares.keys():
    for birth_period in range(-(self.T - 1), 0):
        cohort_age_at_transition = -birth_period
        initial_assets = self.ss_asset_profiles[edu_type][cohort_age_at_transition]
        initial_avg_earnings = self.ss_earnings_profiles[edu_type][cohort_age_at_transition]
        model = birth_cohort_solutions[edu_type][birth_period]
        model.config = model.config._replace(
            initial_assets=initial_assets,
            initial_avg_earnings=initial_avg_earnings,
        )
```

### Fix
Removed the initial-conditions block entirely. Old cohorts now simulate from age 0 with `a=0` and `avg_earnings=0`, exactly like new cohorts. Their price path is padded with `r_path[0]` for ages `0…k-1`, so the steady-state policy functions apply during those pre-transition years and the cohort arrives at age k (calendar t=0) in the correct steady-state asset distribution organically.

See `docs/TRANSITION_ALGORITHM.md` §"Initial conditions for old cohorts" for the theoretical justification.

---

## BUG-004 — Production mode: labor supply disabled (constant l = 1)

**Date:** 2026-02-24
**Status:** Fixed
**Files:** `olg_transition.py`

### Symptom
The lifecycle comparison plots showed mean labor supply = 1.0 for all ages and both cohorts across all education types in production mode.

### Root Cause
`run_full_simulation()` constructed `LifecycleConfig` without `labor_supply=True`, so it defaulted to `False`. With `labor_supply=False`:
- `_solve_labor_hours` immediately returns `1.0`
- `l_policy` is initialised to `np.ones(shape)` and never overwritten
- The JAX path: `l_all = jnp.where(labor_supply, l_star, 1.0)` always returns `1.0`

The test config (`get_test_config`) correctly included `labor_supply=True`; it was simply absent from the production config.

### Fix
Added `labor_supply=True, nu=1.0, phi=2.0` to `LifecycleConfig` in `run_full_simulation()`. (Note: `nu` was subsequently revised — see BUG-005.)

---

## BUG-005 — Unconstrained labor supply (l > 1) and aggregate L spike at t ≈ 30

**Date:** 2026-02-24
**Status:** Fixed
**Files:** `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`, `olg_transition.py`

### Symptom
Two related symptoms appeared after BUG-004 was fixed:

1. **l > 1**: Mean labor supply reached 3.5 for low-education agents at pre-retirement ages. Medium- and high-education agents showed l ≈ 0.1–1, but low-education young agents (low assets, low consumption) hit l >> 1.

2. **Spike in aggregate L, K, Y at t ≈ 27–32**: Aggregate labor jumped ~2.7% above trend at t = 27–29, then dipped ~2.7% below at t = 30–32, before settling. A matching oscillation appeared in K and Y.

### Root Cause

**Unconstrained FOC.** The labor supply FOC is:
```
l* = (c^{-γ} · net_wage / ν)^{1/φ}
```
The implementation enforced only `l* ≥ 0`, with no upper bound. For agents with low consumption (young, low-asset, low-education), `c^{-γ}` is large, driving `l*` well above 1. With `nu=1.0`, `phi=2.0`, `gamma=2.0` and `c ≈ 0.5`, `net_wage ≈ 0.5`:
```
l* = (0.5^{-2} · 0.5 / 1)^{0.5} = (2)^{0.5} ≈ 1.41
```
For even lower consumption states the value reached 3.5.

**Spike mechanism.** The plot x-axis shows period index with a burn-in of 20 dropped, so the visible x = 27–32 corresponds exactly to transition periods t = 27–32. Cohort b=0 (born at the start of the transition) has its highest l* at ages 27–29 (pre-retirement peak). Because `effective_y = w · y · h · l`, aggregate L rises sharply as this cohort approaches retirement, then collapses to zero when the cohort crosses `retirement_age = 30` at calendar t = 30 (effective_y = 0 for retired agents). The spike and the retirement boundary coincide exactly.

This is not an initialisation bug. It is a direct consequence of unconstrained `l*` propagating into `effective_y` and hence into the aggregate cross-section.

### Fix

**1. Clamp l\* ∈ [0, 1] in both backends.**

`lifecycle_perfect_foresight.py`, `_solve_labor_hours`:
```python
# Before
return max(l_star, 0.0)

# After
return min(max(l_star, 0.0), 1.0)  # clamp to [0, 1] — time endowment is 1
```

`lifecycle_jax.py`, `solve_labor_hours_jax`:
```python
# Before
return jnp.maximum(l, 0.0)

# After
return jnp.clip(l, 0.0, 1.0)  # time endowment is 1
```

**2. Increase `nu` in production config.**

With `nu=1.0` most interior solutions are already at or above the l=1 bound, leaving no room for the Frisch elasticity (φ) to act. Raising `nu` lowers the interior solution so agents operate below the ceiling for a wider range of states.

`olg_transition.py`, `run_full_simulation`:
```python
# Before
nu=1.0,

# After
nu=10.0,   # calibrated so FOC gives l≈0.1–0.3 for most agents (clamp l≤1 enforced)
```

With `nu=10`, `phi=2`, `gamma=2` and typical consumption `c ≈ 5–15`, the interior FOC gives `l* ≈ 0.05–0.2` for most agents, well inside the [0,1] constraint. Note: a single `nu` cannot simultaneously target the same average hours across education types whose consumption and wage scales differ by ~10×. Proper calibration should target education-type-specific moments and may require either heterogeneous `nu` values or income renormalization.

### Effect of fix
- Aggregate L, K, Y transition smoothly through t = 30 with no spike.
- Mean labor supply remains ≤ 1 for all ages and education types.
- The Frisch elasticity is active (interior solution) for most agents in most states.

---

## BUG-006 — l_sim = 1.0 for retired agents (incorrect simulation output)

**Date:** 2026-02-24
**Status:** Fixed
**Files:** `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`

### Symptom
Lifecycle labor supply plots showed `mean_l` jumping to 1.0 at `retirement_age=30` and staying there for all subsequent ages. This autoscaled the y-axis to `[0, 1]`, making the working-age profiles (l ≈ 0.05–0.9) appear tiny at the bottom of the chart.

### Root Cause
`l_policy` is initialized to `np.ones(shape)` and the solve leaves retirement ages at 1.0 — a harmless convention because retired agents have `y=0`, making wage income zero regardless of `l`. However, the simulation reads `l_policy` directly for all ages, including retired ones, and stores the result in `l_sim`. Since `mean_l = 1.0` for ages 30–39, the plot y-axis is forced to `[0, 1]`.

### Fix

`lifecycle_perfect_foresight.py`, `_simulate_sequential`:
```python
# Before
l_sim[t_sim, i] = self.l_policy[lifecycle_age, i_a[i], i_y[i], i_h[i], i_y_last[i]]

# After
if is_retired:
    l_sim[t_sim, i] = 0.0
else:
    l_sim[t_sim, i] = self.l_policy[lifecycle_age, i_a[i], i_y[i], i_h[i], i_y_last[i]]
```

`lifecycle_jax.py`, `_agent_step_jax` step_out tuple:
```python
# Before
jnp.where(alive, l_pol_val, 0.0),

# After
jnp.where(alive & ~is_retired, l_pol_val, 0.0),  # retired supply l=0
```

---

## BUG-007 — Single-step FOC approximation underestimates labor supply

**Date:** 2026-02-24
**Status:** Fixed
**Files:** `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`

### Symptom
Mean working-age labor supply for medium education was ~0.07–0.17 and for high education ~0.03–0.08 — systematically too low relative to what the calibrated `nu=10` should produce.

### Root Cause
The labor supply FOC and budget constraint are mutually dependent:
```
l* = (c*^{-γ} · net_wage / ν)^{1/φ}
c* = (budget(l*) − a') / (1 + τ_c)
```
The previous code performed only a single step: compute `l` from `c_guess = budget(l=1)`, then update `c` from `budget(l)`, but never re-solve `l` from the updated `c`. Since `budget(l=1) > budget(l*)` for interior solutions (`l* < 1`), `c_guess > c*`, so the FOC gave `l_step1 < l*`. The stored `(l_policy, c_policy)` pair was internally inconsistent. For medium education the error in `l` was ~35%.

### Fix

**NumPy** — `_solve_state_choice` and the terminal-period block now iterate the `(c, l)` fixed-point to convergence (up to 20 steps, exits early when `|Δl| < 1e-10`):
```python
l = self._solve_labor_hours(c_guess, ...)
for _ in range(20):
    _, _, _, budget_l = self._compute_budget(..., labor_hours=l)
    c_new = (budget_l - a_next) / (1 + tau_c_t)
    if c_new <= 0: break
    l_new = self._solve_labor_hours(c_new, ...)
    if abs(l_new - l) < 1e-10:
        l, c = l_new, c_new; break
    l, c = l_new, c_new
```

**JAX** — 5 unrolled refinement iterations replace the single step in both the non-terminal and terminal period solves:
```python
for _ in range(5):
    delta_budget = effective_wage * (l_star - 1.0) * (1.0 - tau_eff)
    delta_budget = jnp.where(labor_supply, delta_budget, 0.0)
    c_iter = (budget[..., None] + delta_budget - a_next) / (1.0 + tau_c_t)
    l_new = solve_labor_hours_jax(jnp.maximum(c_iter, 1e-10), net_wage_5d, nu, phi, gamma)
    l_new = jnp.where(is_unemployed_5d | is_retired, 1.0, l_new)
    l_star = l_new
```

---

## Notes on calibration (open issue)

The single `nu` parameter cannot perfectly match average labor supply across all three education groups simultaneously. The income grid for high-education agents is ~10× larger than for low-education agents, so the FOC gives very different `l*` values at the same `nu`. Options for future work:

- Use education-type-specific `nu` (requires extending `LifecycleConfig` to accept a dict).
- Normalize income grids so that mean labor income is comparable across education types before applying `nu`.
- Target average hours = 1/3 per education group as a calibration moment.
