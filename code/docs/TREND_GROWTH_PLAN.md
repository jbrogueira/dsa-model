# Trend growth: g = 1%, r_B = 1%

## Objective

Solve the G and I_g experiments in an economy that grows at 1% per year through
labour-augmenting productivity, with the sovereign rate at 1%. The model is
written in detrended units — every quantity divided by the productivity trend —
so the solved equilibrium is stationary and the reported ratios (B/Y, K_g/Y,
Δτ_l) stay comparable across experiments. Population growth is unchanged at the
Greek value (−0.573%); the growth rate added here is productivity.

## What growth changes

A unit saved today carries 1/(1+g) units into next period in detrended terms,
and utility from a given detrended consumption level shrinks by (1+g)^(1−γ) per
period. The household therefore faces an effective return of (1+r)/(1+g) = 2.97%
instead of 4%, and an effective discount factor β(1+g)^(1−γ). Stocks that
accumulate outside the household problem — sovereign debt, public capital, the
pension fund, net foreign assets — each lose a factor (1+g) per period.

Factor prices are unaffected. The firm condition r + δ = αY/K is a flow
condition, so K/L and w are what they are now.

## Step 1 — config and wiring

- Add `trend_growth: 0.01` to `external_params` in `calibration_input_GR.json`.
  `calibrate.py:1044` assigns `external_params` keys to `LifecycleConfig` fields
  by name, so adding a `trend_growth` field to `LifecycleConfig` is enough for
  the household side.
- Pass it to `OLGTransition` (public capital, pension fund) and to
  `FiscalScenario` (debt, current account) — neither reads `external_params`.
- Set `prices.r_B = 0.01`.

## Step 2 — household budget constraint

Replace `budget - a_next` with `budget - (1 + g) * a_next` at:

- `lifecycle_perfect_foresight.py:949` and `:956`
- `lifecycle_jax.py:273` and `:294`

The terminal-age blocks (`lifecycle_perfect_foresight.py:996`,
`lifecycle_jax.py:397` and `:414`) hold no next-period assets and stay as they
are. The simulation reads `c_policy` and `a_policy` directly, so it needs no
change.

The discount factor needs no code change: the calibrated β is now the effective
discount factor β(1+g)^(1−γ), and SMM estimates it directly.

Level objects inside the household problem — medical costs, the minimum pension
floor, child costs, the bequest lump sum — are already stationary in detrended
units and take no growth factor. This makes each of them grow with the trend in
levels, which is the intended reading for all four.

## Step 3 — stock accumulation

| Object | File | New law of motion |
|---|---|---|
| Sovereign debt | `fiscal_experiments.py:264` | `B[t+1] = (1+r_B)/(1+g) · B[t] + PD[t]` |
| Public capital | `olg_transition.py:1915` | `K_g[t] = ((1−δ_g)·K_g[t−1] + I_g[t]) / (1+g)` |
| Pension fund | `olg_transition.py:2238` | `S[t+1] = (1+r)/(1+g) · S[t] + tax_p[t] − pension[t]` |
| Current account | `fiscal_experiments.py:613` | `CA[t] = (1+g)·NFA[t+1] − NFA[t]` |

Two follow-ons:

- `eval_fiscal_results.py:149` recomputes the debt recursion to check it. Mirror
  the new form or the check reports a false violation.
- `fiscal_experiments.py:319` (`terminal_flow_balance`) becomes
  `PD/Y = (g − r_B)/(1+g) · b`, with g the trend growth rate rather than
  `FiscalScenario.pop_growth`. The runs in `run_fiscal_figures.py` select the
  terminal debt and NFA targets, which are ratio targets and need no change.

## Step 4 — parameters re-identified from the same data

`delta_g = 0.047383` in the config is the data ratio I_g/K_g = 0.0353/0.745.
With growth that ratio identifies δ_g + g, so set `delta_g = 0.037383`. Baseline
I_g/Y then stays at 3.53% and the long-run public capital response to a given
ΔI_g is unchanged. Holding δ_g at its current value instead would raise baseline
I_g/Y to 4.28% and cut the long-run K_g response by 17%.

The steady-state interest line becomes r_B·B/Y = 1.64% (data: 3.4%). Nothing
else in the fiscal block moves — `primary_balance_over_Y` in `calibrate.py:1291`
nets interest out, so the closure residual is identified exactly as before.

## Step 5 — recalibration

Run on the V100 instance (needs ≥32GB RAM), in this order:

1. SMM for θ = (ν, β, τ_p, ρ_pens, m_good). β should rise: hitting the same A/Y
   needs roughly β·(1+g)^γ, about 0.961 against the current 0.9419. Start the
   search from the current θ with β scaled by that factor.
2. `normalize_A_tfp.py` — restore Y_ss = 1. Use the Aitken seed trick for the
   SMM ↔ A_tfp loop.
3. `pin_baseline_closure.py --write` — re-pin `other_net_spending_over_Y`.
4. Write θ back into the config.

## Step 6 — experiments

Re-run G and I_g, debt-financed and τ_l-financed, into a fresh output directory.
Then `eval_fiscal_results.py` and the figures.

## Step 7 — checks

- `check_a0_predetermination.py`: A[0] must still be identical across scenarios.
  A missed growth factor in the MIT stitching path shows up here.
- Public capital: at the baseline I_g level, K_g/Y must hold at 0.745 across the
  whole transition.
- Debt: with r_B = g, B/Y must be flat in any period with a zero primary balance.
- Asset grid: confirm the top of `a_grid` still does not bind after
  recalibration.

## Three points to settle before writing this up

**Labour disutility.** With γ = 2 and separable preferences, constant hours along
a growth path require the disutility weight ν to grow with the trend. In
detrended terms ν is a constant and SMM estimates it as it does now, so there is
no code change, but the model write-up needs to carry the trend in ν.

**Pension indexation.** A retiree's benefit is ρ · w at the retirement date, and
w is constant in detrended units, so benefits grow with the trend — retirees are
indexed to aggregate wages. Price indexation would need a factor
(1+g)^−(years since retirement) in `_compute_budget`.

**Interest on government debt.** Households earn r on all wealth including
government debt, while debt accrues at r_B. At r = 4%, r_B = 1%, B/Y = 1.64,
that is 4.9% of Y in interest income the government does not pay. Moving r_B to
1% narrows the gap without closing it. Either leave it and state it, or pay
households r_B on their holdings of B.
