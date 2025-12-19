````markdown
# Distribution-propagation transition algorithm (future work)

## Motivation / current limitation
The current transition code can compute aggregates by simulating each birth cohort (or each (t, age) cell).
This is correct but can be expensive because it involves repeated full-life simulations and storing large arrays.

Goal: compute aggregate paths (K, L, taxes, etc.) by propagating the cross-sectional **state distribution**
forward in **calendar time**, avoiding repeated simulation of full lifecycles.

## Core idea
At each calendar period `t`, for each education group `e` and lifecycle age `a`, maintain a distribution over
individual states:
- assets grid index `i_a` (or asset level)
- income shock index `i_y`
- health index `i_h`
- employment / UI state (if applicable)
- any other discrete states (retirement status can be inferred from age)

Denote this distribution as:
`mu[t, e, a, i_a, i_y, i_h, ...]`.

Given policy functions (consumption/asset choice, labor, transition probabilities for shocks), update:

1. **Newborn inflow** (age 0):
   - initialize `mu[t, e, a=0, ...]` from `initial_assets` and stationary distributions of shocks
     (or model-implied initial distributions).

2. **Aging / survival / transitions**:
   - for ages `a=0..T-2`, push mass to `a+1` at `t+1` using:
     - the endogenous savings choice `a' = a_policy[a, i_a, i_y, i_h, ...]` (mapping indices)
     - exogenous Markov transitions for income and health: `P_y`, `P_h`
     - employment transitions (if modeled)
   - retirement happens deterministically with age.

This produces `mu[t+1, e, a+1, ...]`.

3. **Aggregation**
Compute aggregates directly from `mu[t, ...]`:
- capital `K_t = sum_{e,a,states} weight(e,a) * a_level(i_a) * mu[t,e,a,states]`
- labor `L_t = sum ... effective_labor(state) * mu[...]`
- taxes/spending similarly.

## Benefits
- Time complexity roughly O(T_transition * T * n_states) with no Monte Carlo noise.
- Much lower memory if we store distributions sparsely or as float32.
- Exact aggregation on grids.

## Required refactor / API changes
- Expose discrete transition matrices used in simulation (income/health/employment) in a reusable form.
- Ensure `a_policy` is an index mapping into the asset grid (already seems to be the case).
- Add a function that advances one period of the distribution:
  `mu_next = advance_distribution(mu, policies, transitions, t, e, a)`
- Add an aggregation function that computes all required moments from `mu[t]`.

## Implementation sketch
- New module: `transition_distribution.py`
  - `initialize_mu(...)`
  - `advance_mu_one_calendar_period(mu_t, t, params, policies) -> mu_{t+1}`
  - `compute_aggregates_from_mu(mu_t) -> (K_t, L_t, taxes_t, transfers_t, ...)`

## Validation plan
- For small `T` and small grids, compare moments from distribution-propagation to Monte Carlo simulation:
  - K_t, L_t, mean assets by age, taxes/spending
- Ensure cohort accounting is correct (mass conservation, aging, newborn inflows).

## Open questions
- How to treat borrowing constraints and off-grid choices (if any).
- Whether to include continuous shocks (then need quadrature / discretization).
- Whether to store joint distribution across all states or factorize (approximation).