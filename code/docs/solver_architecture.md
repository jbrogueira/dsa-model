# Solver architecture — steady state and fiscal transition

Two trees, framed by the economic problem each solves. Function names are exact; equations are simplified to the essentials.

---

## Tree 1 — Steady-state calibration

**Problem.** Find structural parameters `θ = (ν, β)` such that simulated moments match data targets, with SOE prices (`r` exogenous, `w` from firm FOC) and a stationary cohort distribution.

```
calibrate.py : main()
│
├── load_config(path)                                 JSON → CalibrationSpec
│   └── compute_equilibrium_prices()                  Firm FOC at SOE r*:
│                                                       K/L = [(r+δ)/(α A K_g^η)]^{1/(α−1)}
│                                                       w   = (1−α) A K_g^η (K/L)^α
│
├── default_spec(config)                              Free params {ν, β}, targets {avg_hours, A/Y}
│
└── calibrate(spec)                                   Nelder–Mead on bounded θ
    │                                                 J(θ) = (m_model − m_data)' W (m_model − m_data)
    │
    └── smm_objective_bounded(θ)         ┐
        │                                │ ← evaluated each NM step
        └── run_model_moments(θ)         │
            │
            ├── apply_params(cfg, θ)                  Inject candidate (ν, β) into LifecycleConfig
            ├── build_olg_transition()                Steady-state OLG object
            │
            ├── OLGTransition.simulate_transition(r_path = const, w_path = const, taxes = const)
            │   │                                     Stationary equilibrium: prices and policies time-invariant
            │   │
            │   ├── solve_cohort_problems()           ∀ edu ∈ {low, med, high}: solve PE lifecycle
            │   │   │
            │   │   └── LifecycleModelJAX / LifecycleModelPerfectForesight
            │   │       │
            │   │       ├── _income_process()         Tauchen AR(1):  log y_jt = (1−ρ)μ + ρ log y_{j−1} + ε
            │   │       │                             × permanent FE  α ∼ N(0, σ_α²)   (Gauss–Hermite, n_alpha pts)
            │   │       │                             × age-wage profile κ(j)
            │   │       │
            │   │       └── solve()                   Backward induction j = T, T−1, …, 0
            │   │           │
            │   │           └── _solve_period(j)
            │   │               │                     V_j(a, y, ŷ, α) = max_{c, ℓ, a'}
            │   │               │                         u(c, ℓ) + β · π(j,s) · E[V_{j+1}]
            │   │               │
            │   │               └── _solve_state_choice()
            │   │                   ├── _compute_budget()
            │   │                   │       Working j < R:
            │   │                   │         c (1+τ_c) + a' = (1−τ_l−τ_p) w κ(j) α y ℓ
            │   │                   │                          + (1+r(1−τ_k)) a + bequest_j
            │   │                   │                          − m_oop(j) + UI/transfer floor
            │   │                   │       Retired j ≥ R:
            │   │                   │         c (1+τ_c) + a' = pension(ŷ, κ̄, ȳ_emp)
            │   │                   │                          + (1+r(1−τ_k)) a − m_oop(j)
            │   │                   │
            │   │                   └── discrete grid-search on (a', ℓ)
            │   │
            │   ├── _simulate_sequential(n_sim)       Monte-Carlo: draw initial wealth + shocks;
            │   │                                     forward-simulate each (edu, cohort) panel
            │   │
            │   ├── (optional) bequest fixed point    If recompute_bequests=True:
            │   │                                     iterate solve+simulate until {bequest_j} converges
            │   │                                     (unintentional bequests of decedents redistributed
            │   │                                      to survivors — closes bequest circuit)
            │   │
            │   └── compute_aggregates(t)             Stationary pop weights ω_j:
            │                                          L  = Σ_j ω_j ℓ_j
            │                                          A  = Σ_j ω_j a_j
            │                                          K_dom = (K/L)·L
            │                                          Y  = A · K_g^η · K_dom^α · L^{1−α}
            │                                          NFA = A − K_dom − B
            │
            └── _moment_* estimators
                ├── _moment_average_hours              ∫ ℓ_j · 1{employed_j} / ∫ 1{employed_j}
                ├── _moment_A_over_Y                   household wealth / GDP
                └── (untargeted: Gini, p90/p10, …)     diagnostic fit

After convergence:
└── Write θ̂ → JSON _derived.theta                    Persist for downstream solvers (calibrate → transition link)
    + theta_metadata (date, report path)
```

---

## Tree 2 — Fiscal-experiment transition

**Problem.** Given a path of policy shocks `{ΔG_t, ΔI_{g,t}, Δτ_t, Δtransfers_t}`, compute the perfect-foresight transition such that
(i) households optimize given announced price/tax paths;
(ii) the government budget closes period by period under the chosen financing rule (debt, taxes, or NFA constraint);
(iii) firm capital demand is pinned by `r*` (SOE);
(iv) `A_0` is predetermined at the pre-shock SS distribution (MIT-shock convention).

Audited against the live code on 2026-05-20.

```
run_fiscal_figures.py   --shock {G, Ig, both}
│
└── fiscal_experiments.run_fiscal_scenario(olg, scenario, base_paths)
    │
    │   ┌──────────────────────────────────────────────────────────────────────┐
    │   │  Branch selector (run_fiscal_scenario:1067–1081):                    │
    │   │    if scenario.nfa_limit or scenario.ca_limit → NFA-constrained      │
    │   │    elif scenario.financing == 'debt'          → debt-residual        │
    │   │    else                                       → instrument-financed  │
    │   │  Three regimes:                                                       │
    │   │    A   debt absorbs the shock (no instrument adjustment)             │
    │   │    B   financing instrument adjusts to satisfy balance_condition     │
    │   │    C   NFA/CA floor binds; bisect shock-scale OR instrument          │
    │   └──────────────────────────────────────────────────────────────────────┘
    │
    ├── (preamble — common to A, B, C)
    │   ├── (1) Baseline solve via _run_one_simulation()
    │   │                                                 No shock; pins A[0], B[0], NFA[0]
    │   ├── _build_pre_transition_paths(olg, base_paths)  Snapshot baseline r, w, τ_*, transfer_floor
    │   │                                                  for ages BEFORE the announced shock
    │   │                                                  ─ MIT-shock invariant: A_0 identical across cf
    │   └── _apply_shock(scenario, base_paths, T, instrument_delta=0, shock_scale=1)
    │                                                     Build cf paths:
    │                                                       cf[k]_t = base[k]_t + shock_scale · Δshock_k(t)
    │                                                                              + instrument_delta · ψ_t
    │                                                                                (financing instrument only)
    │                                                     ψ_t from scenario.adjustment_profile; helpers:
    │                                                       uniform_profile, linear_phase_in,
    │                                                       back_loaded, exponential_convergence
    │
    ├──── Type A   run_debt_financed() ───────────────────────────────────
    │     └── _run_one_simulation(olg, base_paths, cf)    Single solve; debt absorbs deficit residually
    │
    ├──── Type B   run_tax_financed() ────────────────────────────────────
    │     │   financing ∈ {tau_l, tau_c, tau_k, tau_p, pension_replacement, transfer_floor}
    │     │   balance_condition ∈ {terminal_debt_gdp, terminal_flow_balance, pv_balance, period_balance}
    │     └── bisection on scalar Δ for terminal_debt_gdp / terminal_flow_balance / pv_balance
    │         │ bounded scalar minimization (minimize_scalar) for period_balance
    │         └── _simulate_and_residual(Δ)
    │             ├── _apply_shock(..., instrument_delta=Δ)
    │             ├── _run_one_simulation(...)
    │             └── _balance_residual(budget, Y, B, scenario, r_terminal, T_balance)
    │                                              (r_terminal = r_B_path[-1], sovereign rate)
    │                 ├── 'terminal_debt_gdp'    → B[T_bal]/Y[T_bal−1] − target_debt_gdp → 0
    │                 ├── 'terminal_flow_balance'→ PD[T_bal−1]/Y[T_bal−1] − (g − r_B)·target → 0
    │                 ├── 'pv_balance'           → Σ_t (1+δ)^{−t} · PD[t] → 0
    │                 └── 'period_balance'       → minimize max_t |PD[t]|
    │
    └──── Type C   run_nfa_constrained() ─────────────────────────────────
          ├── Step 1 : run debt-financed with full shock; check NFA/CA via _check_nfa_violation()
          ├── Mode I  (financing == 'debt') : bisect shock_scale ∈ [0, 1]
          │                                   → LARGEST feasible scale with NFA_t ≥ −nfa_limit
          │                                                              and CA_t  ≥ −ca_limit ∀ t
          │                                                              (one-sided floors)
          └── Mode II (financing != 'debt'): full shock + bisect instrument Δ
                                            → SMALLEST Δ that restores NFA/CA feasibility

╔══════════════════════════════════════════════════════════════════════╗
║  Inside each OLGTransition.simulate_transition(cf, pre_paths):       ║
╚══════════════════════════════════════════════════════════════════════╝
│
├── solve_cohort_problems(r_path, w_path, tax_paths, pre_transition_paths)
│   │
│   │  For each (edu_type, birth_period b ∈ [1 − T, T_transition − 1]):
│   │    extract calendar-time slice of (r, w, τ_*, pension) the cohort faces
│   │    from age 0 to T, then solve the PF lifecycle.
│   │
│   ├── _extract_cohort_path(path, birth_period, T, default=0.0, pre_value=None)
│   │                                                Pre-transition ages → padded with `pre_value`
│   │                                                 (baseline scalar), so K_0 is identical across cf
│   │
│   ├── self.lifecycle_config._replace(             CRITICAL: preserves edu_params, n_alpha, m_good,
│   │       education_type=edu, current_age=0,        κ, wage_age_profile, … from JSON.
│   │       r_path=cohort_r, w_path=cohort_w, …)
│   │
│   ├── self._lifecycle_model_class(cohort_cfg).solve()
│   │                                                LifecycleModelPerfectForesight (NumPy) or
│   │                                                LifecycleModelJAX, dispatched on `backend`.
│   │                                                Same backward induction as steady state, but
│   │                                                with TIME-VARYING (r_t, w_t, τ_t) along the
│   │                                                cohort's PF path.
│   │
│   └── MIT-shock stitching                         Cohorts with birth_period < 0:
│                                                    build "MIT baseline" lifecycle (pure baseline
│                                                    r/w/τ for ALL ages, baseline transfer_floor);
│                                                    solve; overwrite a_policy[:pre], c_policy[:pre],
│                                                    l_policy[:pre] on the counterfactual model.
│                                                    Reason: backward induction propagates terminal
│                                                    values; cf contamination in post-shock ages
│                                                    corrupts pre-shock policies.
│
├── _ensure_cohort_panel_cache(n_sim, seed_base=42)
│                                                   Forward-simulate every (edu, cohort) panel along
│                                                   its PF price path; initial wealth drawn from the
│                                                   baseline-SS distribution (A_0 predetermination).
│                                                   Internally invokes LifecycleModel.simulate(), which
│                                                   in turn calls _simulate_sequential (NumPy) or
│                                                   _simulate_cohorts_jax_batched (JAX).
│
├── (optional) bequest fixed point                  If recompute_bequests=True and survival_probs:
│                                                    iterate solve + simulate until {bequest_j}
│                                                    converges (max_bequest_iters, bequest_tol).
│                                                    Default in fiscal scenarios: False.
│
├── _compute_all_cross_sections(n_sim)              At each calendar t, aggregate cohorts present:
│                                                     K_t = Σ_{j,edu} ω_{j,t,edu} · a_{j,t,edu}
│                                                     L_t, C_t analogously.
│                                                   Populates self._period_cache so that
│                                                   compute_government_budget_path() can reuse results.
│
├── K_domestic / NFA (SOE block)                    K_domestic_t = (K/L)_t · L_t   (firm FOC at r*)
│                                                                where (K/L)_t = [(r_t+δ)/(α·A·K_g_t^{η_g})]^{1/(α−1)}
│                                                   NFA_t        = A_t − K_domestic_t
│                                                                (partial; B subtracted by fiscal callers)
│
├── _compute_output_path_njit(K, L, alpha, A, K_g, eta_g)
│                                                   Y_t = A · K_g_t^{η_g} · K_t^α · L_t^{1−α}
│                                                     SOE   : K = K_domestic
│                                                     closed: K = K_path = A
│
└── compute_government_budget_path() / compute_government_budget(t)
                                                   Rev_t   = tax_c_t + tax_l_t + tax_p_t + tax_k_t
                                                              + τ_β · TotalBequests_t
                                                            (tax_*_t are realized per-cohort revenues
                                                             summed over (edu, age) — captures progressive
                                                             HSV tax and means-tested behaviour.)
                                                   Spd_t   = G_t + I_{g,t} + UI_t + Pens_t
                                                              + gov_health_t + defense_t
                                                                              [debt service NOT included]
                                                   primary_deficit_t = Spd_t − Rev_t          [textbook primary]
                                                   debt_service_t    = r_B,t · B_t            [reported separately]
                                                   fiscal_deficit_t  = primary_deficit_t − new_borrowing_t
                                                   B-path accumulation (compute_debt_path):
                                                     B[t+1] = (1 + r_B,t) · B[t] + primary_deficit_t
                                                   r_B path: built per simulate_transition() call as
                                                     r_B_path = (np.full(T, r_B) if r_B is not None
                                                                 else r_path)              [capital fallback]
                                                   plumbed into fiscal helpers via base_paths['r_B_path'].

╔════════════════════════════════════════════════════════════════════╗
║  Post-processing                                                   ║
╚════════════════════════════════════════════════════════════════════╝
│
├── compare_scenarios(base, *cfs, variables=None)
│           default variables = ['Y', 'primary_deficit', 'B_gdp_path', 'NFA']
│
├── fiscal_multiplier(base, cf, shock_variable='G', output_variable='Y')
│           per-period m_t = (Y_cf_t − Y_base_t) / (G_cf_t − G_base_t)    (NaN where Δ shock = 0)
│           cumulative print: Σ_t ΔY_t / Σ_t ΔG_t                          (undiscounted)
│
└── debt_fan_chart(scenarios, labels)
            Overlaid B/Y trajectories (no stochastic band — name is conventional).
```

---

## Where the two trees diverge economically

| Aspect | Steady state (calibration) | Transition (fiscal) |
|---|---|---|
| Time index | Lifecycle age `j` only | Lifecycle age `j` **and** calendar `t` |
| Prices, taxes | Constant in `t` | PF path in `t` |
| Cohorts solved | Three (one per `edu`) | `3 × (T_pre + T_transition)` — one per `(edu, birth_period)` |
| `A_0` | Output of the SS distribution | Input: pinned from baseline SS (MIT-shock) |
| Outer loop | SMM Nelder–Mead on `θ` | Bisection on `Δτ` or shock scale (Types B, C) |
| Goal | Match data moments | Solve for equilibrium paths; report multipliers |
| Government budget | Implicit (stationary identity) | Explicit period-by-period; ΔB closes the gap |

The lifecycle solver (`LifecycleModelJAX` / `LifecycleModelPerfectForesight`) is identical in both trees — only the price/tax paths it receives change.
