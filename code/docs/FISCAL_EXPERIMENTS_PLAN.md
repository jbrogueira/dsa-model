# Implementation Plan: Fiscal Scenario Framework

Covers two independent features:
- **Feature A** — Bequest redistribution loop in `OLGTransition` (closes the currently open bequest circuit; independent of fiscal experiments)
- **Feature B** — Fiscal Scenario Framework (`fiscal_experiments.py`) — computes budget-balanced transition paths under different financing regimes given exogenous policy shocks

---

## Background: Current Architecture

`OLGTransition.simulate_transition()` is a **one-pass simulator**. All fiscal instruments are exogenous inputs:

```
Inputs: r_path, tau_c/l/p/k_path, G_path, I_g_path, B_path, pension_replacement_path
  → solve_cohort_problems()          (lifecycle backward induction, all cohorts)
  → _ensure_cohort_panel_cache()     (Monte Carlo simulation)
  → compute_aggregates()             (K, L, Y paths)

Post-hoc accounting (no feedback):
  → compute_government_budget_path() (deficit = residual; B_path exogenous)
```

The government budget deficit is an **accounting residual** — it never feeds back into agent decisions or prices. In SOE mode (`economy_type='soe'`), `r = r*` is exogenous and `w` is pinned by the production function given `r*` and `K_g`. Domestic fiscal policy therefore affects agent behavior and revenue/spending flows, but **not prices**. This is the key property that makes the fiscal experiment iterations tractable.

### Current bequest gap

`_compute_bequest_lumpsum_path()` exists and computes the per-newborn after-tax bequest transfer, but it is never called automatically within `simulate_transition()`. The `bequest_lumpsum_path` argument to `simulate_transition()` defaults to `None`. So the bequest redistribution loop is **open by default**: dying agents' assets are recorded in `bequest_sim`, bequest tax revenue enters `compute_government_budget()`, but the after-tax transfer never reaches newborns' budget constraints unless the caller manually constructs and passes `bequest_lumpsum_path`.

---

## Feature A — Bequest Redistribution Loop in `OLGTransition`

**Scope:** `olg_transition.py` only. No changes to lifecycle backends.

**Motivation:** Closing the bequest loop is economically correct whenever `survival_probs` is set and `tau_beq < 1`. It is independent of fiscal experiments and should be available in baseline transition simulations.

### Mechanism

The fixed-point iteration on the bequest lumpsum path:

```
bequest_lumpsum_path ← None   (initial guess: no transfer)

repeat:
    solve_cohort_problems(..., bequest_lumpsum_path=bequest_lumpsum_path)
    _ensure_cohort_panel_cache(n_sim, seed_base)
    new_path = _compute_bequest_lumpsum_path()
    if max_t |new_path[t] - old_path[t]| < bequest_tol:
        break
    bequest_lumpsum_path = new_path
```

Since bequests are typically a small fraction of newborn wealth, convergence in **1–2 iterations** is expected in practice. The loop is only behaviorally meaningful when `survival_probs` is set; when `survival_probs is None` (the default), `_compute_bequest_lumpsum_path()` returns all zeros and the loop exits after one pass with no change — **fully backward compatible**.

### Changes to `olg_transition.py`

**`simulate_transition()` signature additions:**
```python
def simulate_transition(self, r_path, ...,
                        bequest_lumpsum_path=None,   # existing — now used as initial guess
                        recompute_bequests=False,    # NEW: close the bequest loop
                        bequest_tol=1e-4,            # NEW: convergence threshold
                        max_bequest_iters=5,         # NEW: safety cap
                        ...):
```

**Logic (inside `simulate_transition()`, before the aggregation loop):**
```python
if recompute_bequests and self.lifecycle_config.survival_probs is not None:
    current_bequest_path = bequest_lumpsum_path  # use caller's value as initial guess
    for _bequest_iter in range(max_bequest_iters):
        self.solve_cohort_problems(..., bequest_lumpsum_path=current_bequest_path)
        self._ensure_cohort_panel_cache(n_sim=n_sim, seed_base=42, verbose=False)
        new_bequest_path = self._compute_bequest_lumpsum_path(n_sim=n_sim)
        old_vals = np.array([current_bequest_path.get(t, 0.0)
                             for t in range(self.T_transition)])
            if current_bequest_path is not None else np.zeros(self.T_transition))
        new_vals = np.array([new_bequest_path.get(t, 0.0) for t in range(self.T_transition)])
        if np.max(np.abs(new_vals - old_vals)) < bequest_tol:
            current_bequest_path = new_bequest_path
            break
        current_bequest_path = new_bequest_path
    # final solve with converged bequest path (or use already-cached solve if converged early)
    bequest_lumpsum_path = current_bequest_path
    # the cohort panel cache is already up to date from the last iteration
else:
    self.solve_cohort_problems(..., bequest_lumpsum_path=bequest_lumpsum_path)
    self._ensure_cohort_panel_cache(n_sim=n_sim, seed_base=42, verbose=verbose)
```

Store convergence diagnostics on `self`: `self._bequest_iter_count`, `self._bequest_converged`.

### Tests

- `recompute_bequests=False` (default): result identical to current behavior ✓
- `survival_probs=None` + `recompute_bequests=True`: loop exits after 1 pass, same result ✓
- With `survival_probs` set: newborn budget constraint changes; verify `a_sim[0, :]` shifts by the bequest transfer amount in a simple calibration
- Convergence: verify `max |Δbequest_path|` falls below `bequest_tol` within `max_bequest_iters`
- Budget identity: `bequest_tax_revenue + bequest_transfers = total_bequests` holds at convergence

---

## Feature B — Fiscal Scenario Framework

**Scope:** New file `fiscal_experiments.py`. Imports `OLGTransition`; **no changes to any existing file**.

### Concept

An **outer policy loop** wraps `olg.simulate_transition()` + `olg.compute_government_budget_path()` and searches for the path of a single **balancing instrument** that satisfies a chosen **budget balance condition**, given an exogenous **policy shock**.

```
FiscalScenario
    │
    ▼
run_fiscal_scenario(olg, scenario, base_paths)
    │
    ├─ financing='debt'        → run_debt_financed()      [1 simulate call]
    │                               + compute_debt_path()
    │
    ├─ financing='tau_*'       → run_tax_financed()       [~10–15 simulate calls]
    │  or 'transfer_floor'          bisect on scalar Δτ
    │
    └─ nfa_limit / ca_limit    → run_nfa_constrained()    [~15–20 simulate calls]
       set                          outer: check NFA/CA feasibility
                                    inner: bisect on shock scale Δ or on Δτ
```

### `FiscalScenario` Dataclass

```python
@dataclass
class FiscalScenario:
    name: str

    # ── Policy shock ───────────────────────────────────────────────────────────
    # Additive changes relative to the base paths, length T_transition.
    # Any combination may be set simultaneously.
    delta_G_path:        Optional[np.ndarray] = None   # Δ government consumption
    delta_I_g_path:      Optional[np.ndarray] = None   # Δ public investment
    delta_tau_l_path:    Optional[np.ndarray] = None   # Δ labor income tax
    delta_tau_c_path:    Optional[np.ndarray] = None   # Δ consumption tax
    delta_tau_k_path:    Optional[np.ndarray] = None   # Δ capital income tax
    delta_tau_p_path:    Optional[np.ndarray] = None   # Δ payroll tax
    delta_pension_path:  Optional[np.ndarray] = None   # Δ pension replacement rate

    # ── Financing instrument ───────────────────────────────────────────────────
    # 'debt'          : B_path is the residual (no iteration on agent problems)
    # 'tau_l' | 'tau_c' | 'tau_k' | 'tau_p' : tax rate adjusts to hit balance target
    # 'transfer_floor': means-tested transfer floor adjusts
    financing: str = 'debt'

    # ── Adjustment profile ─────────────────────────────────────────────────────
    # The balancing instrument adjustment enters as: Δinstrument_t = Δ * psi_t
    # None  → uniform: psi_t = 1 for all t  (single scalar Δ, cleanest)
    # array of length T_transition → pre-specified phase-in schedule, scalar Δ still
    #         searched by bisection (e.g., gradual phase-in, back-loaded, etc.)
    adjustment_profile: Optional[np.ndarray] = None

    # ── Budget balance condition ───────────────────────────────────────────────
    # 'terminal_debt_gdp' : B_T / Y_T = target_debt_gdp
    # 'pv_balance'        : Σ_t δ^t * PrimaryDeficit_t = 0
    # 'period_balance'    : find Δ minimizing max_t |PrimaryDeficit_t|
    #                       (best uniform approximation; exact period balance
    #                        requires a T-dimensional path search, future work)
    balance_condition: str = 'terminal_debt_gdp'
    target_debt_gdp: float = 0.0       # target B_T / Y_T
    discount_rate: float = 0.04        # used by 'pv_balance'

    # ── Initial debt ───────────────────────────────────────────────────────────
    B_initial: float = 0.0

    # ── External constraint (optional) ────────────────────────────────────────
    # Per-period NFA stock floor: NFA_t >= -nfa_limit for all t
    nfa_limit: Optional[float] = None
    # Per-period CA flow floor: CA_t = NFA_{t+1} - NFA_t >= -ca_limit for all t
    ca_limit: Optional[float] = None

    # ── Bequest handling ───────────────────────────────────────────────────────
    # Forwarded to olg.simulate_transition(recompute_bequests=...)
    recompute_bequests: bool = False
```

### `FiscalScenarioResult` Dataclass

```python
@dataclass
class FiscalScenarioResult:
    scenario: FiscalScenario

    # Macroeconomic paths (length T_transition)
    base_macro:  dict   # r, w, K, L, Y  from base run
    cf_macro:    dict   # r, w, K, L, Y  from counterfactual run

    # Fiscal paths (dicts of np.ndarray, length T_transition)
    base_budget: dict   # output of compute_government_budget_path() on base
    cf_budget:   dict   # output of compute_government_budget_path() on counterfactual

    # Debt paths (length T_transition + 1)
    B_path:     np.ndarray   # debt level
    B_gdp_path: np.ndarray   # B_t / Y_t

    # NFA/CA paths (length T_transition), populated in SOE mode
    NFA_path: Optional[np.ndarray] = None
    CA_path:  Optional[np.ndarray] = None   # CA_t = NFA_{t+1} - NFA_t

    # Adjustment found by bisection
    adjustment_scalar: float          # Δ (the single scalar found)
    adjustment_path:   np.ndarray     # Δ * psi_t (length T_transition)
    adjustment_label:  str            # e.g. 'Δτ_l', 'shock scale Δ'

    # Solver diagnostics
    converged:        bool
    n_iterations:     int
    residual_history: list            # balance residual at each bisection step
```

---

## Algorithms

### Type A — Debt-financed (no iteration)

```
1. Apply shock to base paths:
       G_path_cf    = G_path_base    + delta_G_path
       I_g_path_cf  = I_g_path_base  + delta_I_g_path
       tau_l_path_cf = tau_l_path_base + delta_tau_l_path   [etc.]

2. olg.simulate_transition(r_path, tau_paths_cf, G_cf, I_g_cf, ...,
                            recompute_bequests=scenario.recompute_bequests)

3. olg.compute_government_budget_path()

4. Accumulate debt forward:
       B[0] = B_initial
       B[t+1] = (1 + r[t]) * B[t] + PrimaryDeficit[t]
       B_gdp[t] = B[t] / Y[t]

5. Compute NFA_path, CA_path from cf_macro (SOE mode)

6. Return FiscalScenarioResult (adjustment_scalar=0, adjustment_path=zeros)
```

One `simulate_transition()` call. As fast as the baseline.

**Note on public capital:** When `delta_I_g_path` is set, the production function `Y = A · K_g^{η_g} · K^α · L^{1−α}` means higher `K_g` raises `w`, which raises the labor income tax base. This partial self-financing is automatically captured because `simulate_transition()` recomputes `K_g_path` and `w_path` from `I_g_path`. No special logic needed.

### Type B — Tax/transfer-financed (scalar bisection)

The balancing instrument adjustment path is:
```
instrument_t^cf = instrument_t^base + Δ * psi_t
```
where `psi_t = adjustment_profile[t]` (or 1 for all `t` if uniform).

**Bisection target functions:**

| `balance_condition` | Residual `f(Δ)` | Monotone in `Δ`? |
|---|---|---|
| `terminal_debt_gdp` | `B_T/Y_T − target_debt_gdp` | Yes (higher tax → lower deficit → lower debt) |
| `pv_balance` | `Σ_t δ^t · PD_t` | Yes |
| `period_balance` | `max_t |PD_t|` | Not guaranteed; use bounded minimization instead |

For `terminal_debt_gdp` and `pv_balance`, standard bisection:
```
Δ_lo = −instrument_base_mean   (instrument can't go negative)
Δ_hi = large positive value (e.g. 0.5)

while (Δ_hi - Δ_lo) > tol and iter < max_iter:
    Δ_mid = (Δ_lo + Δ_hi) / 2
    run simulate_transition(instrument_path + Δ_mid * psi)
    run compute_government_budget_path()
    accumulate B_path
    residual = f(Δ_mid)
    if residual > 0: Δ_hi = Δ_mid   [too much debt → raise tax more]
    else:            Δ_lo = Δ_mid
```

For `period_balance`: find `Δ` minimizing `max_t |PD_t|` via bounded scalar minimization (e.g., `scipy.optimize.minimize_scalar` with `method='bounded'`). Each function evaluation is one `simulate_transition()` call. This is the best achievable with a single uniform instrument — it does not guarantee zero deficit every period (that requires a T-dimensional path search, deferred to future work).

**Bracket initialization:** Before bisection, verify the bracket signs:
- At `Δ = Δ_lo`: residual should be positive (deficit, not enough adjustment).
- At `Δ = Δ_hi`: residual should be negative (surplus, too much adjustment).
- If the bracket is not valid, expand `Δ_hi` geometrically until it is.

### Type C — NFA/CA-constrained (outer feasibility + inner bisection)

```
Step 1: Run debt-financed scenario with full shock (Δ=1)
        Compute NFA_path^cf and CA_path^cf = NFA_{t+1}^cf - NFA_t^cf

Step 2: Check which constraint is active (if any):
        nfa_violation = any(NFA_path^cf < -nfa_limit)  if nfa_limit is set
        ca_violation  = any(CA_path^cf  < -ca_limit)   if ca_limit  is set

Step 3a — No violation: return the debt-financed result as-is.

Step 3b — Violation exists. Two sub-modes:

  Mode I  (trim the shock):
    Find the largest shock scale Δ_shock ∈ [0, 1] such that
    the scaled shock Δ_shock * delta_G_path (or delta_I_g_path) satisfies the constraint.
    Bisect on Δ_shock; each evaluation = one simulate_transition() + NFA check.
    Interpretation: "how much of this G increase can be externally financed?"

  Mode II (mixed: full shock + tax adjustment to compress domestic absorption):
    Apply full shock. Find Δτ on the specified financing instrument such that
    NFA_t ≥ -nfa_limit (or CA_t ≥ -ca_limit) for all t.
    This reduces household savings (higher tax → lower after-tax income → less saving
    if income effect dominates) or directly reduces absorption.
    Bisect on Δτ; NFA check replaces the budget balance condition.
    Note: the NFA/income effect direction depends on tax type and model parameters;
    verify monotonicity empirically before relying on bisection.

Default: Mode I. Mode II activated by setting `financing != 'debt'` alongside nfa_limit/ca_limit.
```

**CA_path computation:**
```python
NFA_path = cf_macro['NFA']                 # length T_transition
CA_path  = np.diff(NFA_path, append=NFA_path[-1])   # CA_t = NFA_{t+1} - NFA_t
# (last period: use steady-state NFA or repeat last value)
```

---

## External Constraint: NFA vs CA — Full Options Table

Currently only `nfa_limit` (per-period NFA stock) and `ca_limit` (per-period CA flow) are implemented. Four natural formulations exist; the PV variants are deferred to future work:

| Variant | Formula | Implemented | Interpretation |
|---|---|---|---|
| NFA per-period | `NFA_t ≥ −nfa_limit ∀t` | **Yes** | Foreign debt ceiling: total external indebtedness cap |
| CA per-period | `CA_t ≥ −ca_limit ∀t` | **Yes** | No-sudden-stop: external position can't deteriorate by more than `ca_limit` per year |
| NFA present-value | `Σ_t δ^t · NFA_t ≥ −PV_NFA_limit` | Future | Soft stock constraint: allows short-run violations if offset later |
| CA present-value | `Σ_t δ^t · CA_t ≥ −PV_CA_limit` | Future | Telescopes to terminal NFA condition: limits cumulative external deterioration |

The CA present-value constraint collapses usefully: `Σ_t δ^t · CA_t ≈ NFA_T^{discounted} − NFA_0`, so it is essentially a long-run external solvency condition.

---

## Adjustment Profile Helpers

```python
def uniform_profile(T: int) -> np.ndarray:
    """psi_t = 1 for all t. Δ is a permanent level shift."""
    return np.ones(T)

def linear_phase_in(T: int, n_ramp: int) -> np.ndarray:
    """psi_t ramps linearly from 0 to 1 over n_ramp periods, then stays at 1."""
    ramp = np.linspace(0, 1, n_ramp + 1)[1:]
    return np.concatenate([ramp, np.ones(T - n_ramp)])

def back_loaded(T: int, n_delay: int) -> np.ndarray:
    """psi_t = 0 for first n_delay periods, then 1."""
    return np.concatenate([np.zeros(n_delay), np.ones(T - n_delay)])

def exponential_convergence(T: int, half_life: float) -> np.ndarray:
    """psi_t = 1 - 0.5^(t / half_life). Asymptotically approaches 1."""
    t = np.arange(T)
    return 1.0 - 0.5 ** (t / half_life)
```

---

## Debt Accumulation Utility

Standalone function (no OLG dependency):
```python
def compute_debt_path(primary_deficit_path: np.ndarray,
                      r_path: np.ndarray,
                      B_initial: float = 0.0) -> np.ndarray:
    """
    Forward-recursive debt accumulation.
        B[0] = B_initial
        B[t+1] = (1 + r[t]) * B[t] + PrimaryDeficit[t]
    Returns B_path of length T_transition + 1.
    """
    T = len(primary_deficit_path)
    B = np.zeros(T + 1)
    B[0] = B_initial
    for t in range(T):
        B[t + 1] = (1 + r_path[t]) * B[t] + primary_deficit_path[t]
    return B
```

This is the fundamental identity underlying all three experiment types.

---

## Output Utilities

```python
def compare_scenarios(
    base: FiscalScenarioResult,
    *counterfactuals: FiscalScenarioResult,
    variables: list = None,   # e.g. ['B_gdp_path', 'Y', 'tau_l', 'NFA', 'primary_deficit']
    save: bool = True,
    filename: str = None,
) -> plt.Figure:
    """Side-by-side line plots of base vs. counterfactual paths."""

def fiscal_multiplier(
    base: FiscalScenarioResult,
    cf: FiscalScenarioResult,
    shock_variable: str = 'G',    # key in budget_path or macro
    output_variable: str = 'Y',
) -> np.ndarray:
    """
    Compute ΔOutput_t / ΔShock_t at each period.
    Cumulative multiplier: Σ_t ΔY_t / Σ_t ΔG_t.
    """

def debt_fan_chart(
    scenarios: list,           # list of FiscalScenarioResult
    labels: list,              # legend labels
    save: bool = True,
    filename: str = None,
) -> plt.Figure:
    """Plot B/Y paths for multiple scenarios — standard DSA output."""
```

---

## Implementation Steps

### Feature A (bequest loop in `OLGTransition`)

| Step | Change | File |
|---|---|---|
| A1 | Add `recompute_bequests`, `bequest_tol`, `max_bequest_iters` to `simulate_transition()` signature | `olg_transition.py` |
| A2 | Add bequest fixed-point loop inside `simulate_transition()`, guarded by `recompute_bequests and survival_probs is not None` | `olg_transition.py` |
| A3 | Store `_bequest_iter_count` and `_bequest_converged` on `self` for diagnostics | `olg_transition.py` |
| A4 | Tests: backward compatibility, convergence, budget identity, newborn wealth shift | `test_olg_transition.py` |

### Feature B (fiscal experiments)

| Step | Change | File |
|---|---|---|
| B1 | `FiscalScenario` and `FiscalScenarioResult` dataclasses | `fiscal_experiments.py` (new) |
| B2 | Adjustment profile helpers (`uniform_profile`, `linear_phase_in`, etc.) | `fiscal_experiments.py` |
| B3 | `compute_debt_path()` standalone utility | `fiscal_experiments.py` |
| B4 | `_balance_residual()` — maps `(budget_path, Y_path, B_path, scenario)` to scalar | `fiscal_experiments.py` |
| B5 | `_apply_shock()` — builds counterfactual policy paths from base + scenario deltas | `fiscal_experiments.py` |
| B6 | `run_debt_financed()` — 1 simulate call + debt accumulation | `fiscal_experiments.py` |
| B7 | `run_tax_financed()` — bisection loop; handles all 3 balance conditions | `fiscal_experiments.py` |
| B8 | `run_nfa_constrained()` — outer NFA/CA check + bisection (Mode I and II) | `fiscal_experiments.py` |
| B9 | `run_fiscal_scenario()` — dispatcher | `fiscal_experiments.py` |
| B10 | `compare_scenarios()`, `fiscal_multiplier()`, `debt_fan_chart()` | `fiscal_experiments.py` |
| B11 | Tests | `test_fiscal_experiments.py` (new) |

**Dependency:** Feature A should be merged before Feature B (B uses `recompute_bequests` via `olg.simulate_transition()`).

---

## Testing Strategy

### Feature A tests

1. `recompute_bequests=False` (default): result byte-identical to current output
2. `survival_probs=None` + `recompute_bequests=True`: one pass, no behavioral change
3. With mortality active: `bequest_lumpsum_path` converges within `max_bequest_iters`; newborn period-0 assets shift by the transfer amount
4. Budget identity: `tau_beq * total_bequests + (1 - tau_beq) * total_bequests == total_bequests` holds at every period

### Feature B tests

1. **Debt accumulation identity**: `B[t+1] = (1+r[t])*B[t] + PD[t]` holds exactly for Type A
2. **Tax-financed convergence**: `|balance_residual| < tol` at convergence for `terminal_debt_gdp` and `pv_balance`
3. **Period balance**: `max_t |PD_t|` is minimized; result is smaller than with `Δ=0`
4. **Bracket validity**: bisection always finds a valid bracket (test for edge cases where target is infeasible)
5. **NFA constraint**: `min_t NFA_t >= -nfa_limit` and `min_t CA_t >= -ca_limit` hold at solution
6. **Public investment self-financing**: with `eta_g > 0`, `delta_I_g > 0` produces higher `w_path` and thus higher `tax_l` revenue relative to `eta_g = 0`; debt path is lower
7. **Adjustment profile**: uniform and `linear_phase_in` profiles produce different `adjustment_path` arrays but both satisfy the balance condition
8. **Multiplier sign**: positive G shock → positive output multiplier (for any financing mode)
9. **Backward compatibility**: calling `run_fiscal_scenario()` with `delta_G_path=zeros` and `financing='debt'` reproduces the base budget path exactly

---

## Key Design Constraints

- **No changes to existing files** for Feature B (imports only)
- **All new features default OFF** — `recompute_bequests=False` recovers current behavior exactly
- **SOE assumption**: prices `(r, w)` do not respond to domestic fiscal policy; bisection residual is monotone in `Δτ` for non-Laffer-curve parameter ranges; verify empirically if labor supply elasticity is high
- **Bequest loop in fiscal iterations**: `recompute_bequests` in `FiscalScenario` is forwarded to `olg.simulate_transition()` — no separate logic in `fiscal_experiments.py`
- **JAX backend compatibility**: all functions operate on `OLGTransition` instances; the backend choice (`'numpy'` or `'jax'`) is a property of the passed `olg` object and requires no changes

---

## Computational Cost Summary

| Experiment | `simulate_transition()` calls | Expected time (NumPy) | Notes |
|---|---|---|---|
| Debt-financed | 1 | 1× baseline | No iteration |
| Tax-financed (`terminal` or `pv`) | 10–15 (bisection) | 10–15× baseline | ~8–12 iters typical |
| Tax-financed (`period_balance`) | 15–25 (minimization) | 15–25× baseline | `minimize_scalar` evaluations |
| NFA-constrained Mode I | 10–15 (shock scale bisect) | 10–15× baseline | Outer bisect on Δ_shock |
| NFA-constrained Mode II | 15–25 (nested) | 15–25× baseline | Outer NFA check + inner Δτ |
| With `recompute_bequests=True` | ×2–3 inner iters each | 2–3× per call | Only if `survival_probs` set |

JAX backend reduces per-call cost by ~5–10× and is recommended for `period_balance` and NFA-constrained experiments.
