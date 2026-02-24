# Transition Algorithm

## Overview

The model simulates an economy transitioning between two steady states.
The interest rate path `r(t)` is exogenous (small open economy).
Wages `w(t)` are derived from the production function given `K/L`.
Because prices are taken as given, **no outer fixed-point iteration is needed** —
the transition is solved in a single forward pass.

---

## Step 1 — Enumerate cohorts

Every birth cohort alive at any point during the transition gets its own
lifecycle problem. Let `T` be the lifecycle length and `T_trans` the number
of transition periods. The cohorts to solve span:

```
birth_period ∈ [-(T-1),  T_trans - 1]
```

- Cohorts with `birth_period < 0` were already alive when the transition began.
- Cohorts with `birth_period ≥ 0` are born during the transition.

---

## Step 2 — Build each cohort's price path

Each cohort sees a length-`T` slice of the global price path indexed by their
own lifecycle age, not by calendar time.

**Cohorts born during the transition** (`birth_period ≥ 0`):

```
cohort_r[j] = r_path[birth_period + j],   j = 0, …, T-1
```

**Cohorts already alive at t=0** (`birth_period = -k < 0`):

The cohort was born `k` periods before the transition. Their lifecycle ages
`0, …, k-1` fall in the pre-transition steady state, while ages `k, …, T-1`
fall during the transition. The price path they take as given is:

```
cohort_r[j] = r_path[0]          for j = 0, …, k-1   (pre-transition SS price, repeated)
cohort_r[j] = r_path[j - k]      for j = k, …, T-1   (actual transition path)
```

This is implemented in `_extract_cohort_path`. The agent knows at birth that
prices will be at the initial steady-state level for their first `k` periods,
and then follow the announced transition path from age `k` onward.
This is **perfect foresight**: the full price path is known from birth.

**Initial conditions for old cohorts:**

All cohorts — old and new — are simulated from age 0 with `a=0` and
`avg_earnings=0`. For old cohorts born at `b = -k`, their price path is
padded with `r_path[0]` (the initial SS price) for ages `0, …, k-1`.
Because those prices are identical to the steady state, the policy functions
at those ages reproduce the SS lifecycle exactly, and the cohort arrives at
age `k` (calendar `t=0`) in the correct steady-state asset distribution.

Placing SS age-k wealth at age 0 (a previous approach) is wrong: the age-0
policy function maps it incorrectly, producing a distorted trajectory that
makes all cohort cross-sections look like the SS and keeps aggregates constant
throughout the transition.

---

## Step 3 — Solve each cohort (backward induction)

Each cohort's lifecycle problem is solved by **backward induction** from age
`T-1` to age `0`, working with the cohort's own length-`T` price path.

At each age `t` the value function

```
V_t(a, y, h, y_last)
```

and policy functions for savings `a'`, consumption `c`, and labor `l` are
computed by grid search over the asset grid, maximising:

```
u(c, l) + β · π(t,h) · E[ V_{t+1}(a', y', h') ]
```

where `π(t, h)` is the survival probability, `y'` and `h'` are next-period
income and health states drawn from their transition matrices.

State space per period: `n_a × n_y × n_h × n_y_last`
(asset grid × current income × health × last working income for pension).

---

## Step 4 — Simulate each cohort (Monte Carlo)

After solving, `n_sim` agents per cohort are simulated **forward** from age 0
using the computed policy functions. Random draws for income shocks, health
shocks, and mortality are pre-generated. Old cohorts start from their
steady-state initial conditions (Step 2 above) rather than from age 0.

---

## Step 5 — Aggregate

At each calendar period `t`, the cross-section is assembled by slicing the
appropriate age from each cohort's simulated panel:

```
age of cohort b at time t  =  t - b
```

Cohorts are weighted by demographic size (population weights by birth period).
Fiscal quantities — taxes, transfers, pensions, bequests — are summed across
all cohorts alive at `t` to produce period-level aggregates.

---

## JAX acceleration

### Batched solve (`vmap` over cohorts)

The NumPy backend solves cohorts one at a time in a Python loop.
`_solve_lifecycle_jax_batched` stacks all cohorts and calls `jax.vmap`,
so all cohort solves run in one compiled XLA call, fully parallelised.

### Batched simulation (`vmap` over cohorts and agents)

`_simulate_lifecycle_jax_batched` vmaps over cohorts, and within each cohort
over agents. All Monte Carlo paths for all cohorts are computed simultaneously.

### `lax.scan` for the time loop

Inside each cohort solve and each agent simulation there is a sequential loop
over `T` periods. This loop **cannot** be parallelised (each step depends on the
previous one), but it can still be compiled efficiently.

`lax.scan(f, carry, xs)` works as follows:

- `carry` is the state threaded from one step to the next.
- `xs` is a stack of inputs, pre-sliced as arrays (one slice per step).
- `f(carry, x)` returns `(new_carry, output)` for a single step.

**In the backward induction solve**, `carry = V_next` (the value function at
age `t+1`) and `xs` are the period parameters (prices, taxes, health matrices…)
stacked for ages `T-2` down to `0`. `lax.scan` runs the grid search for one
age at a time, passing `V_t` as the carry into the next (earlier) age.

**In the simulation**, `carry` is an agent's state `(i_a, i_y, i_h, …)` and
`xs` are the pre-drawn random numbers for each period. `lax.scan` steps the
agent through their life one period at a time.

The critical difference from a Python loop:

| | Python loop | `lax.scan` |
|---|---|---|
| JAX traces the body | `T` times | once |
| Computation graph depth | `T` nodes | 1 fused op |
| Memory | all intermediates live at once | XLA can reuse buffers |
| GPU | steps dispatched serially | XLA pipelines the loop |

JAX traces `scan_fn` once, then XLA compiles the repeated application into a
single efficient kernel. For the grid in this model
(`n_a × n_y × n_h × n_y_last ≈ 7500` states, `T` periods),
this avoids `T` separate Python-dispatched kernel launches.
