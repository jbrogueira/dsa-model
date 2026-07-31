# Model Writeup

Independent, code-anchored description of the model implemented in `/Users/joaob.sousa/Work/research/dsa/code`. Equations describe what the code computes; the code is the source of truth. Notation follows the default glossary; new symbols are defined at first use. Parameter values are the live configuration (`calibration_input_GR.json`, Greece), with calibrated values taken from its `_derived.theta` block (the post-estimation override applied at load). Issues are collected in Appendix B; symbol↔code anchors in Appendix A.

Notation note: `κ_j` is the deterministic wage age profile; the government health-coverage share is written `ξ` to avoid collision; `σ` is CRRA risk aversion, `ν` the labor-disutility weight, `φ` the inverse Frisch parameter.

## 1. Steady State

The model is an overlapping-generations small open economy in discrete (annual) time, indexed by calendar period `t` and lifecycle age `j = 0, …, J-1`. Households face idiosyncratic productivity and employment risk and age-dependent mortality, supply labor on the intensive margin, save in a single asset, pay age-dependent out-of-pocket medical costs, and retire at a fixed age `j_R`. Aggregate household wealth is decomposed ex post into domestic capital, government debt, and net foreign assets. The government taxes consumption, labor, payroll, capital, and bequests, pays pensions, unemployment insurance, and a health subsidy, consumes goods, invests in public capital, and issues sovereign debt at an exogenous rate. The interest rate is exogenous (small open economy); the wage follows from the firm's first-order condition.

### 1.1 Demography

Each period a cohort is born and lives up to `J = 60` ages; retirement is fixed at `j_R = 39`. Survival from age `j` to `j+1` is `π_j ∈ [0,1]`, age-varying and, in the data-driven mode, cohort-specific. A cohort is split across education types `e ∈ {low, med, high}` with population shares `{φ_e} = {0.234, 0.471, 0.295}`, fixed at birth.

Cross-sectional cohort weights at calendar time `t` are reduced-form,
$$\omega_{j,t} = \frac{\exp\!\big(g_t\,(y^{b}_{j,t} - \bar y^{b})\big)}{\sum_{j'=0}^{J-1}\exp\!\big(g_t\,(y^{b}_{j',t} - \bar y^{b})\big)}, \qquad y^{b}_{j,t} = (t_0 + t) - j, \tag{1.1}$$
with population growth `g_t`, birth year `y^b`, and reference birth year `\bar y^b`. Since `y^b_{j,t}` is linear in `-j`, this is `ω_{j,t} ∝ (1+g_t)^{-j}` up to normalization. Survival is not multiplied into `ω_{j,t}`: mortality is carried in the per-cohort simulation means (agents who fail the survival draw contribute zero), so weights remain births-only to avoid double counting.¹ At steady state `g_t = g` and `{π_j}` are constant.

### 1.2 Preferences

Period utility is CRRA in consumption with separable iso-elastic labor disutility,
$$u(c,\ell) = \frac{c^{1-\sigma}}{1-\sigma} - \nu\,\frac{\ell^{1+\varphi}}{1+\varphi}, \qquad \ell\in[0,1], \tag{1.2}$$
risk aversion `σ = 2`, disutility weight `ν`, inverse Frisch `φ = 1.5`. Labor disutility applies only to the employed working-age. A household discounts at `β` and weights continuation by survival; the discounted-utility objective is
$$\mathbb{E}\sum_{j=0}^{J-1}\beta^{j}\Big(\textstyle\prod_{k<j}\pi_k\Big)\,u(c_j,\ell_j). \tag{1.3}$$

### 1.3 Income, wages, pensions

Idiosyncratic productivity is a Markov chain `z ∈ {0, z_1, …, z_{n_y-1}}`, `n_y = 5`, with `z = 0` unemployment; the `n_y-1 = 4` employed states discretize a log-normal AR(1) via Tauchen,
$$\log z' = \rho_z\log z + \eta', \quad \eta'\sim N(0,\sigma_\eta^2), \tag{1.4}$$
education-specific `ρ_z = 0.95`, `σ_η ∈ {0.054, 0.086, 0.075}`. A permanent fixed effect `α_i` (5-node Gauss–Hermite grid, education-specific variance `σ_α²`) enters productivity multiplicatively through `e^{α_i}`. Employed wage income is
$$y^L_j = w_t\,\kappa_j\,z\,e^{\alpha_i}, \tag{1.5}$$
with deterministic age profile `κ_j`. When `z = 0` the household receives unemployment insurance `b^{ui}_j = ρ^{ui} w_t κ_j z_{last} e^{α_i}`, where `z_{last}` is the last employed state, evolving as `z_{last}' = z·𝟙[z>0] + z_{last}·𝟙[z=0]`. Movements in and out of unemployment use a job-finding rate `λ^{f} = 0.5` and a separation rate `λ^{s} = min((u/(1-u))λ^{f}, \barλ^{s})`, `\barλ^{s} = 0.1`, with education-specific unemployment rate `u`.

At retirement the pension is fixed from the wage and a blend of last-employed and career-average productivity, with a floor:
$$\text{pens} = \max\!\Big(\rho^{p}\,w_t\,\kappa_{j_R}\big[\lambda z_{last} + (1-\lambda)\bar z\big]e^{\alpha_i},\; b_{\min}\Big), \tag{1.6}$$
replacement rate `ρ^p`, floor `b_min = 0.15`, and blend weight `λ`. The configuration leaves `λ` unset, so the code derives it from the career-average approximation `λ = (1 - ρ_z^{j_R})/(j_R(1-ρ_z))`, which at `ρ_z = 0.95`, `j_R = 39` gives `λ ≈ 0.443`²: the live pension base is a near-even blend of the last-employed state and career-average productivity, not last-state-only (`λ = 1`).

### 1.4 Household budget

With assets `a` (beginning of period, bound `a' ≥ a̲ = 0`), the working-age budget is
$$(1+\tau^c)c + a' = \big(1+(1-\tau^k)r_t\big)a + y^L_j + b^{ui}_j - \tau^p y^L_j - \tau^l\big(y^L_j - \tau^p y^L_j + b^{ui}_j\big) - (1-\xi)m_j + \beta^{B}_t, \tag{1.7}$$
where the payroll tax `τ^p` falls on wage income only, the labor income tax `τ^l` on the base `y^L_j - τ^p y^L_j + b^{ui}_j` (payroll-deductible, unemployment insurance taxable),³ `m_j` is age-dependent medical cost of which the government covers share `ξ = 0.662` and the household pays `(1-ξ)m_j`, and `β^B_t` is the lump-sum bequest transfer (§1.6). A minimum-consumption floor is enforced by an additional transfer when (1.7) would otherwise imply `c < c̲`. The retired budget drops labor income and adds the pension, taxed at `τ^l` but not `τ^p`:
$$(1+\tau^c)c + a' = \big(1+(1-\tau^k)r_t\big)a + (1-\tau^l)\,\text{pens} - (1-\xi)m_j + \beta^{B}_t. \tag{1.8}$$

### 1.5 Technology, prices, and the open economy

Output is Cobb–Douglas in private capital, effective labor, and public capital,
$$Y_t = A\,(K^g_t)^{\eta_g}K_t^{\alpha}L_t^{1-\alpha}, \tag{1.9}$$
`α = 0.33`, public-capital elasticity `η_g = 0.05`,⁴ `A = 1`. Given the exogenous `r_t`, the firm's first-order conditions pin the capital–labor ratio and the wage:
$$\frac{K_t}{L_t} = \left(\frac{\alpha A(K^g_t)^{\eta_g}}{r_t+\delta}\right)^{\frac{1}{1-\alpha}}, \qquad w_t = (1-\alpha)A(K^g_t)^{\eta_g}\left(\frac{K_t}{L_t}\right)^{\alpha}, \tag{1.10}$$
`δ = 0.05`. Public capital accumulates from public investment,
$$K^g_{t+1} = (1-\delta_g)K^g_t + I^g_t, \qquad \delta_g = 0. \tag{1.11}$$

Aggregate effective labor and the private capital stock are
$$L_t = \sum_e φ_e\sum_{j<j_R}\omega_{j,t}\,\overline{\kappa_j z\,\ell\,e^{\alpha}}_{\,e,j}, \qquad K_t = \frac{K_t}{L_t}\,L_t \;\;(\text{domestic capital}). \tag{1.12}$$
Household wealth `A_t = \sum_e φ_e\sum_j ω_{j,t}\,\bar a_{e,j}` need not equal `K_t + B_t`; the residual is net foreign assets,
$$\text{NFA}_t = A_t - K_t - B_t, \tag{1.13}$$
`B_t` sovereign debt. Production uses domestic capital `K_t`, not household wealth `A_t`; the two coincide only in the closed-economy limit. The goods-market identity is
$$C_t + I_t + G_t + I^g_t + \Delta\text{NFA}_t = Y_t, \qquad I_t = K_{t+1}-(1-\delta)K_t. \tag{1.14}$$

### 1.6 Bequests

Assets of the deceased are taxed at `τ^β` (live value 0) and the remainder is redistributed lump-sum to surviving members of the same cohort, giving `β^B_t` in (1.7)–(1.8). The transfer is a fixed point: it depends on aggregate accidental bequests, which depend on saving, which depends on the transfer (§5.4).

## 2. Transition

### 2.1 Setup

A transition is a perfect-foresight path of length `T_{trans} = 60` [@AuerbachKotlikoff1987]. The exogenous interest rate path `{r_t}` is given; `{w_t}` and `{K_t/L_t}` follow from (1.10) period by period given `{K^g_t}`. Every cohort born at or after `t = 0` knows the entire future of `(r_t, w_t, τ_t, ρ^p_t, K^g_t)` and solves (1.2)–(1.8) along its own calendar diagonal. Aggregates are formed period by period from the simulated cohort cross-sections weighted by `ω_{j,t}` (1.1).

### 2.2 Predetermination and the link to steady state

Initial wealth `A_0` and the age distribution `μ_0` are predetermined: cohorts alive before `t = 0` hold their pre-shock policies over their pre-transition ages, so a counterfactual cannot move wealth chosen before the announcement. The mechanism overwrites the pre-transition slice of each cohort's policy with a pure-baseline solve (an MIT-shock boundary condition); the baseline solve must use baseline `r/w/τ` for all ages, since backward induction propagates future prices into earlier policies.⁵ The result is that `A_0` is identical across counterfactuals; this is checked by a standalone diagnostic script (`check_a0_predetermination.py`, reporting zero difference on both backends), not by the pytest suite.

### 2.3 Regime structure (the fiscal experiment)

Two permanent fiscal shocks of size `Δ` (2% of mean baseline output) are studied, each financed two ways:

- **G shock:** `G^{cf}_t = G^{base}_t + Δ`. With exogenous `r_t`, (1.10) fixes `K_t/L_t`, `Y_t`, `w_t`, so output is unchanged under debt financing.
- **`I^g` shock:** `I^{g,cf}_t = I^{g,base}_t + Δ`, raising `K^g_t` via (1.11), hence `Y_t` via (1.9) and `w_t` via (1.10).

Financing: (i) **debt-financed** — tax rates at baseline, debt `B_t` is the residual of the law of motion (2.1); (ii) **labor-tax-financed** — a constant increment `Δτ^l` on the baseline labor-tax path, set by a one-dimensional search so a terminal debt target is met. All unshocked spending lines stay at their baseline paths (the output-share lines re-scale to realized output). Debt evolves at the sovereign service rate `r^B = 0.021`,
$$B_{t+1} = (1+r^B)B_t + \text{PD}_t, \tag{2.1}$$
with primary deficit `PD_t` from (3.4). The live figure script's labor-tax closure matches the counterfactual terminal debt ratio to the baseline transition's terminal ratio,
$$\frac{B_{T_{bal}}}{Y_{T_{bal}-1}} = \left.\frac{B_{T_{bal}}}{Y_{T_{bal}-1}}\right|_{\text{baseline}}, \tag{2.2}$$
a stock-ratio target (the baseline is debt-financed, so its terminal ratio is known only after the baseline run). The code also implements a flow target `PD_{T_{trans}-1}/Y_{T_{trans}-1} = (g - r)b^*`, `b^* = B_0/Y_0` (the primary balance that stabilizes debt at its initial ratio), but this `terminal_flow_balance` branch is not the one the figure script selects.⁶ Either target is a finite-horizon condition, not a converged long-run object. The debt-financed fiscal multiplier is `𝓜_t = (Y^{cf}_t - Y^{base}_t)/Δ`.

## 3. Equilibrium

### 3.1 Government

Revenues and outlays, integrated over the period cross-section (`A_t` household wealth, `𝓑^{lab}_t = \sum_e φ_e\sum_{j<j_R}ω_{j,t}\,\overline{w_tκ_j z\ell e^{α}}_{e,j}` the wage bill):

| Line | Definition |
| :-- | :-- |
| Consumption tax | `τ^c_t C_t` |
| Labor income tax | `τ^l_t` on taxable labor income (base `𝓑^{lab}_t - τ^p𝓑^{lab}_t + \text{UI}_t`) |
| Payroll tax | `τ^p_t 𝓑^{lab}_t` |
| Capital income tax | `τ^k_t r_t A_t` |
| Bequest tax | `τ^β` × aggregate accidental bequests |
| Unemployment insurance | `\text{UI}_t = \sum_e φ_e\sum_{j<j_R}ω_{j,t}\,\overline{b^{ui}}_{e,j}` |
| Pensions | `\text{PENS}_t = \sum_e φ_e\sum_{j\ge j_R}ω_{j,t}\,\overline{\text{pens}}_{e,j}` |
| Health subsidy | `\text{HS}_t = ξ\sum_e φ_e\sum_j ω_{j,t}\,\bar m_{e,j}` |
| Government consumption | `G_t` |
| Public investment | `I^g_t` |
| Defense | `D_t` |
| Other net primary | `O_t` (residual closure) |

Total revenue and primary deficit:
$$\text{Rev}_t = τ^c_t C_t + τ^l_t\,\mathcal{T}^{lab}_t + τ^p_t 𝓑^{lab}_t + τ^k_t r_t A_t + \text{BeqTax}_t, \tag{3.1}$$
$$\text{PD}_t = G_t + I^g_t + D_t + O_t + \text{UI}_t + \text{PENS}_t + \text{HS}_t - \text{Rev}_t. \tag{3.4}$$
An optional pension trust fund accumulates `S^{p}_{t+1} = (1+r_t)S^{p}_t + τ^p_t𝓑^{lab}_t - \text{PENS}_t`. The residual line `O_t` is held at a share of output pinned at the initial steady state so the initial-period budget matches the primary-balance target (`primary_balance_target/Y = 0.0195`); the live value is `O/Y = -0.091122`.

### 3.2 Markets and equilibrium concept

Given the exogenous `{r_t}`: the labor market clears through `L_t` (1.12); the goods market clears by the resource identity (1.14) with NFA absorbing the gap between national saving and domestic investment; the capital market does not clear domestically — domestic capital is pinned by the firm FOC (1.10) and any wealth–capital gap is net foreign assets (1.13). The government budget closes through debt (2.1) or the labor tax (2.2). The equilibrium is a stationary recursive steady state (constant prices and policies, stationary age distribution) or a perfect-foresight transition between two such steady states with predetermined `A_0` (§2.2). No domestic capital-market or interest-rate fixed point is solved; the interest rate is a small-open-economy primitive.

## 4. Calibration

Greece. The idiosyncratic income process is estimated externally (LIS/EU-SILC) and fixed; five parameters are estimated internally by the method of simulated moments (SMM); the rest are set externally.

### 4.1 Externally set

| Symbol | Meaning | Value | Source |
| :-- | :-- | :-- | :-- |
| `J` | lifecycle length | 60 | demographic |
| `j_R` | retirement age | 39 | Eurostat |
| `σ` | CRRA | 2.0 | external |
| `φ` | inverse Frisch (`1/φ = 0.67`) | 1.5 | [@Chetty_AER2011] |
| `a̲` | borrowing limit | 0 | institutional |
| `α` | capital share | 0.33 | [@Gollin_JPE2002] |
| `δ` | depreciation | 0.05 | OECD |
| `A` | TFP | 1.0 | normalized (units) |
| `η_g` | public-capital elasticity | 0.05 | [@BaxterKing_AER93] |
| `δ_g` | public depreciation | 0.0 | — |
| `K^g_0` | initial public capital | 1.0 | normalized (units) |
| `τ^l` | labor income tax | 0.10 | EC implicit |
| `τ^c` | consumption tax | 0.1818 | Eurostat implicit |
| `τ^k` | capital income tax | 0.2236 | Eurostat effective |
| `τ^β` | bequest tax | 0.0 | — |
| `ρ^{ui}` | UI replacement | 0.0945 | OAED rules |
| `b_min` | pension floor (wage units) | 0.15 | Greek min/avg pension |
| `ξ` | health coverage | 0.662 | OECD Health |
| `λ^{f}` | job-finding rate | 0.50 | Eurostat |
| `\barλ^{s}` | separation cap | 0.10 | — |
| `B_0/Y` | initial debt/GDP | 1.64 | Eurostat 2023 |
| `G/Y` | govt consumption/GDP | 0.13 | Eurostat |
| `I^g/Y` | public investment/GDP | 0.03 | Eurostat |
| `D/Y` | defense/GDP | 0.03 | SIPRI/NATO |
| `g` | population growth | −0.00573 | Eurostat |
| `r` | capital return | 0.04 | 10y Greek bond |
| `r^B` | sovereign service rate | 0.021 | effective Greek yield |
| primary-balance target/`Y` | terminal closure | 0.0195 | — |
| `O/Y` | other net primary | −0.091122 | pinned at initial steady state |
| `T_{trans}` | transition horizon | 60 | — |

### 4.2 Internally calibrated (SMM)

Five parameters `θ = (β, ν, τ^p, ρ^p, m)` minimize the weighted distance between model and data moments,
$$Q(θ) = \big[m^{d} - m^{m}(θ)\big]'W\big[m^{d} - m^{m}(θ)\big], \tag{4.1}$$
`W` diagonal with entries `1/(m^d_i)^2`. The solver is Nelder–Mead on logit-transformed parameters; moments are computed from a stationary solve with `n_{sim} = 10{,}000` per education type; no domestic price loop (`w` from (1.10) given `r`).

| Symbol | Meaning | Value | Dominant target |
| :-- | :-- | :-- | :-- |
| `β` | discount factor | 0.943 | `A/Y = 4.00` |
| `ν` | disutility weight | 36.91 | mean hours `= 0.41` |
| `τ^p` | payroll tax | 0.198 | social contributions/`Y = 0.13` |
| `ρ^p` | pension replacement | 0.166 | pensions/`Y = 0.16` |
| `m` | medical cost (share of mean income) | 0.0428 | government health/`Y = 0.054` |

### 4.3 Income process (external)

| | Low | Med | High |
| :-- | :-- | :-- | :-- |
| share `φ_e` | 0.234 | 0.471 | 0.295 |
| unemployment `u` | 0.165 | 0.158 | 0.101 |
| mean log productivity (dev.) | −0.153 | 0.000 | +0.259 |
| persistence `ρ_z` | 0.95 | 0.95 | 0.95 |
| innovation `σ_η` | 0.054 | 0.086 | 0.075 |
| fixed-effect sd `σ_α` | 0.367 | 0.259 | 0.318 |

Estimated by weighted NLLS on the cross-sectional log-income variance profile `Var(u_j) = σ_α² + σ_η²(1-ρ^{2j})/(1-ρ²)`, LIS/EU-SILC Greece, `ρ_z` fixed at 0.95. [@Tauchen1986]; [CITATION: Heathcote, Storesletten, Violante (2017), income process / progressive tax].

## 5. Numerical Algorithm

### 5.1 Household solve

The lifecycle problem (1.2)–(1.8) is solved by backward induction from `j = J-1` to `0`. The state is `(a, z, z_{last})` with `a` on an exponentially-spaced grid of `n_a = 100` points on `[0, 200]`, `z` on `n_y = 5` points (one unemployment + four Tauchen). For each state the algorithm grid-searches next assets `a'` over the `n_a` points; for the employed the intratemporal labor choice solves the first-order condition
$$\nu\,\ell^{\varphi} = c^{-\sigma}\,\frac{\text{MW}}{1+\tau^c}, \qquad \text{MW} = w_t\kappa_j z\,e^{\alpha_i}(1-\tau^p)(1-\tau^l), \tag{5.1}$$
by a projected-Newton iteration bracketed to the feasible region `c(\ell) > 0` with a bisection fallback (step tolerance `1e-12`). The continuation value is survival-discounted, `β π_j 𝔼[V_{j+1} | z]`, the expectation taken over the income transition (and a health transition when `n_h > 1`). The terminal age consumes all resources. A permanent fixed effect is handled by an outer loop over the 5-node `α` grid; policies are stored per node.

### 5.2 Simulation

A Monte-Carlo panel of `n_{sim}` agents per cohort (10,000 in calibration, 2,000 in transition; seed 42) draws income, employment, mortality, and computes assets, consumption, labor, taxes, pensions, and accidental bequests along each cohort's life. Dead agents contribute zero; per-age means divide by `n_{sim}`, so the survival fraction enters means directly (consistent with births-only weights, §1.1). The JAX backend vectorizes the solve (grid search via `vmap`, backward induction via `scan`) and the simulation (`vmap` over agents); it matches the NumPy reference to float64 tolerance and is the production path for calibration.

### 5.3 Transition aggregation

`simulate_transition` solves all cohorts, simulates them, and aggregates per-(education, age) cross-sections into `{K, C, L}` weighted by `ω_{j,t}`. Domestic capital `K_t` is computed from the firm FOC (1.10) before output, and (1.9) uses `K_t`, not household wealth `A_t`; `NFA_t = A_t - K_t - B_t` (1.13). The government budget path (3.1)–(3.4) is computed from the same cross-sections.

### 5.4 Outer fixed points

Two outer loops, no domestic price loop:
1. **Bequest fixed point** (active when survival is on): iterate cohort-solve → simulate → recompute aggregate bequests → update the lump-sum transfer until `max|Δ| < 10^{-4}`, capped at 5 iterations. The production CLI enables it; the test CLI does not.
2. **Financing search** (labor-tax case): a one-dimensional root-find (bisection) on the labor-tax increment `Δτ^l` until the terminal debt-ratio residual (2.2) is zero (default branch `terminal_debt_gdp`; the alternative `terminal_flow_balance` branch implements the flow target but is not selected by the figure script).

The fiscal-scenario dispatcher runs a baseline and a counterfactual: Type A (debt-financed, one pass), Type B (tax-financed, root-find on a scalar tax increment), Type C (net-foreign-asset-constrained, an outer feasibility check around an inner search).

## Notes

¹ Survival is not multiplied into the cohort weights because per-cohort simulation means already net out the dead (Appendix B1).
² The configuration leaves `pension_avg_weight` unset, so the code derives `λ ≈ 0.443` from the career-average approximation (`calibrate.py:1093`); `λ = 1` would be last-state-only.
³ Unemployment insurance is taxed at `τ^l`; the payroll tax `τ^p` is deductible from the `τ^l` base and falls on wage income only.
⁴ Public capital is active in the live configuration (`η_g = 0.05`, `K^g_0 = 1`), so the `I^g` shock operates through (1.9)–(1.11).
⁵ The pre-transition (MIT) baseline solve must use pure-baseline `r/w/τ` for all ages; backward induction would otherwise propagate counterfactual future prices into pre-transition policies and break predetermination of `A_0`.
⁶ The live figure script selects `terminal_debt_gdp` (2.2); the `terminal_flow_balance` branch (the flow target `PD/Y = (g-r)b^*`) is implemented but not used.

## Appendix A — Code–Equation Correspondence

| Equation / Object | Symbol | Code reference | Notes |
| :-- | :-- | :-- | :-- |
| Cohort weights | (1.1) | `olg_transition.py:890-930` (`set_cohort_sizes_path_from_pop_growth`), `:932-936` (`_cohort_weights`) | reduced-form `exp(g·years)`, normalized |
| Births-only vs survival weights | ¹ | `olg_transition.py:983-1031` (`_build_population_weights`), guard `:2072` | fertility/longevity branch (dead for live config) — see Appendix B |
| CRRA + labor disutility | (1.2) | `lifecycle_perfect_foresight.py:604-609` (utility), `:872` (disutility) | `σ`=gamma, `ν`=nu, `φ`=phi |
| Discounted objective | (1.3) | backward induction `:611-691`; survival discount `:985` | — |
| Income AR(1)/Tauchen | (1.4) | `lifecycle_perfect_foresight.py:445-520`; `n_employed=n_y-1` `:469` | — |
| Permanent effect `e^{α}` | (1.5) | `α` outer loop `:650-680`; income process `:513` | — |
| Wage income / UI | (1.5) | `_compute_budget` `:693`, `:720-726` | — |
| Pension | (1.6) | `lifecycle_perfect_foresight.py:702-708` | blend `λ`=pension_avg_weight |
| Pension blend default `λ` | (1.6) | `calibrate.py:1083-1093` | unset in config → `λ=(1-ρ^{j_R})/(j_R(1-ρ))≈0.443` |
| Working budget | (1.7) | `_compute_budget :693`, `:724-743` | UI taxed at `τ^l`, payroll deductible |
| Retired budget | (1.8) | `_compute_budget :693` | pension taxed at `τ^l` not `τ^p` |
| Production | (1.9) | `_production_function_njit :815-818` | uses `K_domestic` |
| Firm FOCs | (1.10) | `_marginal_products_njit :822-829`; `K/L` inversion `:1939` | — |
| Public capital | (1.11) | `:1908-1917` | — |
| Aggregation `L`, `K` | (1.12) | `_aggregate_capital_labor_njit :833-849`; `L/=w` `:2102` | `L` in efficiency units |
| NFA | (1.13) | `K_domestic` inversion `:2114-2118`; `NFA` assignment `:2119` | `NFA = A − K_domestic − B` |
| Resource identity | (1.14) | not computed in code (accounting identity) | unanchored — see Appendix B |
| Bequest transfer | §1.6 | bequest loop `:1990-2047` | `τ^β`=tau_beq=0 |
| Pension trust fund | §3.1 | `olg_transition.py:2234-2238` | optional `S^p` accumulation |
| Predetermination / MIT stitching | §2.2 | `solve_cohort_problems :1033`; per-`α` stitch `:1238`/`:1306`; check `check_a0_predetermination.py` | verified by script, not pytest |
| Debt law of motion | (2.1) | `fiscal_experiments.py:242-261`; `r^B` `olg_transition.py:1763-1768` | uses `r^B` not `r` |
| Terminal closure (live) | (2.2) | `terminal_debt_gdp` residual `fiscal_experiments.py:299`; target set `run_fiscal_figures.py:273-276` | matches baseline terminal B/Y |
| Terminal flow target (unused) | — | `terminal_flow_balance` branch `fiscal_experiments.py:301-307` (formula `:306`) | implemented, not selected by figure script |
| Revenue / primary deficit | (3.1),(3.4) | `compute_government_budget :1684-1797`, `:1773`; budget path `:2197-2266` | incl. `D_t`, `O_t` |
| SMM moments / objective | (4.1) | moments `run_model_moments :665`; objective `Q` `smm_objective :699-710`; ratios `compute_fiscal_ratios :1206` | `W=diag(1/(m^d)^2)` |
| Job-finding / separation | §1.3 | `lifecycle_perfect_foresight.py:460-497`; config `external_params` (0.5, 0.1) | overrides dataclass defaults |
| Fiscal multiplier `𝓜_t` | §2.3 | `fiscal_experiments.py` `fiscal_multiplier` | `(Y^{cf}-Y^{base})/Δ` |
| Calibrated `θ` override | §4.2 | `calibrate.py:1060-1075`, `:1161-1168` | `_derived.theta` overrides base fields |
| Equilibrium prices | (1.10) | `compute_equilibrium_prices :901` | `w`, `K/L`, `Y/L` from `{r,α,δ,A,K^g}` |
| Labor FOC solver | (5.1) | `_solve_labor_newton :816-866`; JAX `solve_labor_robust_jax` | projected Newton, tol 1e-12 |
| Backward induction | §5.1 | `solve :611-691`; state loop `_solve_period :881-912`; `a'` search `_solve_state_choice :947-991` | — |
| Simulation | §5.2 | `simulate :1260+`; JAX `lifecycle_jax.py` `vmap`/`scan` | seed 42 |
| Transition driver | §5.3 | `simulate_transition :1832-2195` | — |
| Bequest fixed point | §5.4 | `:1990-2047` | tol 1e-4, ≤5 iters |
| Fiscal dispatcher | §5.4 | `fiscal_experiments.py run_fiscal_scenario`, `run_debt_financed`/`run_tax_financed`/`run_nfa_constrained` | Type A/B/C |

## Appendix B — Issues Found in Code

1. **Survival double-counting in the fertility/longevity weight branch — present but inactive for the live config.** The default cross-sectional weights (1.1) are births-only because per-cohort simulation means already embed survival (dead agents contribute zero), and the code states survival must not also enter the weights. The alternative branch `_build_population_weights` (`olg_transition.py:1004-1024`) sets `cohort_sizes_path[t,age] = fert_val · cum_surv` with `cum_surv = ∏_{j<age} mean(survival)`; feeding the same survival-netted per-cohort means, this would count survival twice. The branch is gated at `olg_transition.py:2072` by `fertility_path is not None or survival_improvement_rate != 0.0`. The live Greek config sets neither (`fertility_path` absent, `survival_improvement_rate → 0`; `survival_data_file` drives per-cohort simulation survival, not this branch), so the guard is False on every live entry point and weights come from the births-only `set_cohort_sizes_path_from_pop_growth`. The double-count is latent: it would activate only if a `fertility_path` or nonzero longevity-improvement rate were supplied.

2. **Debt service rate vs capital return.** Equation (2.1) services debt at `r^B = 0.021`, distinct from the capital return `r = 0.04` used in household and firm problems. This is an internally consistent modeling choice (a sovereign spread), recorded here because the two rates are easy to conflate; not a defect.

3. **`O_t` residual closure is a level pinned at the initial steady state, not a behavioral object.** The other-net primary line `O_t` (3.4) is set so the initial-period budget matches the primary-balance target and is then held fixed (as an output share) along the transition. The baseline transition's `t = 0` primary balance therefore need not equal the target exactly. Documented behavior, not a bug.

4. **The resource identity (1.14) is an accounting statement, not computed or tested.** No code path forms `C + I + G + I^g + ΔNFA - Y`, and no test asserts it; it is stated for completeness and follows from (1.13) and the budget. Treat it as unverified by the code. Likewise A[0] predetermination (§2.2) is checked only by `check_a0_predetermination.py`, not the pytest suite.

No violated identities were found in the inspected tests: the production-identity test uses `K_domestic` with `η_g = 0`, the debt-law tests match (2.1), the government-budget test is consistent with (3.4), and the SMM-objective test asserts `Q ≥ 0`, consistent with (4.1). The aggregation unpack order `(K, C, L)` is consistent with (1.12).

## Appendix C — Citations

Resolved to the project bibliography (`docs/ref.bib`) and applied inline: `[@Tauchen1986]` (AR(1) discretization), `[@Chetty_AER2011]` (Frisch elasticity), `[@Gollin_JPE2002]` (labor share), `[@BaxterKing_AER93]` (public capital), `[@AuerbachKotlikoff1987]` (OLG transition). Note: `ref.bib` entry `Tauchen1986` has a typo in its journal field ("Economic Letters" → "Economics Letters").

Unresolved — add to the library before final compilation:
- **Heathcote, Storesletten, Violante (2017)**, "Optimal Tax Progressivity: An Analysis of Effects and Optimal Settings," *Quarterly Journal of Economics* 132(4): 1693–1754 — the income-process / progressive-tax reference. The only HSV entry in Zotero is the 2009 *Annual Review* survey (`SSNZXXV4`), not this paper; verify the exact subtitle before inserting. Placeholder kept at §4.3.
- **MIT-shock / predetermined-distribution transition** (§2.2) — no entry exists. Suggested canonical reference: Boppart, Krusell, Mitman (2018), "Exploiting MIT shocks in heterogeneous-agent economies," *JEDC* 89: 68–92 (closest match), or `[@AuerbachKotlikoff1987]` for the perfect-foresight OLG transition technique generally.

## Appendix D — Audit Trail

- **Code Mapper (Subagent 1):** reused this session's code-only mapping (an `Explore` agent instructed to read only source/config/tests and skip all `.tex` and `docs/` writeups), supplemented by direct reads of `olg_transition.py:890-1031` (weights) and `calibrate.py` SS-solve functions.
- **Files read (code only):** `olg_transition.py`, `lifecycle_perfect_foresight.py`, `lifecycle_jax.py`, `fiscal_experiments.py`, `calibrate.py`, `run_fiscal_figures.py`, `pin_baseline_closure.py`, `build_survival_GR.py`, `eval_fiscal_results.py`, `calibration_input_GR.json`, `test_*.py`.
- **Blacklist (not read by any agent, for independence):** `../docs/*.tex`; `code/docs/*.md` (including `dsa_economic_problem.md`, `dsa_implementation.md`, `model_vs_implementation.md`, `bug_report.md`, `FISCAL_*`, `IMPLEMENTATION_PLAN.md`, `solver_architecture.md`, `TRANSITION_ALGORITHM.md`); the prior `code_report.md`/`.pdf`; README Model/Theory sections.
- **Subagent 2 (Auditor):** verified Appendix-A anchors against code; found two substantive errors (pension blend `λ ≈ 0.443` not 1; live τ_l closure uses `terminal_debt_gdp` not the flow condition), several wrong-branch line cites, and confirmed Appendix-B1's double-count branch is dead for the live config. All applied above.
- **Subagent 3 (Citation Matcher):** resolved 5 of 7 placeholders against `docs/ref.bib`; HSV (2017) and the MIT-shock reference unresolved (Appendix C).
