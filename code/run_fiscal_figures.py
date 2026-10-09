"""
Fiscal experiment figures — demo script.

Runs three scenarios on the fast-test OLG configuration and produces figures
for each shock type.

Run:
    cd code
    python run_fiscal_figures.py               # G shock, NumPy backend (default)
    python run_fiscal_figures.py --shock Ig    # I_g (public investment) shock
    python run_fiscal_figures.py --shock both  # both shock types (G,Ig)
    python run_fiscal_figures.py --backend jax # JAX backend
    # The policy exercises (docs/POLICY_EXERCISES_PLAN.md): unanticipated in 2026
    python run_fiscal_figures.py --config calibration_input_GR.json --backend jax \
        --shock Ig,health --scenarios debt,tau_l_debt,tau_l_window

Shocks: G (government consumption, 2% of output), Ig (public investment, a
constant detrended level of 2% of baseline output in the year before the
shock), health (a cut in coverage kappa and in the level of medical spending
over --health-years years, calibrated to --health-targets). Scenarios: debt,
tau_l_debt (a permanent labour-tax change from the shock year that returns
debt/output at T_balance to the baseline's), tau_l_nfa (the same with a
terminal net-foreign-asset target), tau_l_window (health only: the labour tax
moves over the years of the cut only, same debt target). The health exercise
adds two debt-financed runs that cut coverage only and medical spending only.
Every scenario is unanticipated in --shock-year; --shock-year 2023 (the base
year) reproduces the runs with the shock at the start of the transition.
"""

import argparse
import functools
import gc
import json
import os
import platform
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # avoid Metal backend on macOS
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display required
import matplotlib.pyplot as plt  # noqa: E402

from lifecycle_perfect_foresight import LifecycleConfig  # noqa: E402
from olg_transition import OLGTransition  # noqa: E402
from fiscal_experiments import (  # noqa: E402
    FiscalScenario,
    run_baseline,
    run_fiscal_scenario,
    compare_scenarios,
    debt_fan_chart,
)

# Force unbuffered print so progress is visible over SSH / pipes
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['numpy', 'jax'], default='numpy')
parser.add_argument('--shock', default='G',
                    help='Comma list from {G, Ig, health}; both = G,Ig')
parser.add_argument('--scenarios', default='debt,tau_l_debt,tau_l_nfa',
                    help='Comma list from {debt, tau_l_debt, tau_l_nfa, tau_l_window}')
parser.add_argument('--shock-year', type=int, default=2026,
                    help='Calendar year in which the shock becomes known (unanticipated)')
parser.add_argument('--health-targets', default=None,
                    help='d_gov,d_hh: changes in government and household health spending, '
                         'shares of output (default: from the data, see --health-window)')
parser.add_argument('--health-window', default='2011-2015',
                    help='Years whose mean change relative to the year before gives the targets')
parser.add_argument('--health-household', choices=['che_minus_gov', 'hf3'],
                    default='che_minus_gov',
                    help='Household health spending: CHE less government schemes, or HF.3')
parser.add_argument('--health-years', type=int, default=5, help='Length of the health cut')
parser.add_argument('--no-distribution', action='store_true',
                    help='Skip the distributional and welfare outputs')
parser.add_argument('--tiny', action='store_true',
                    help='The small economy of policy_reference_case.py (tests)')
parser.add_argument('--config', type=str, default=None,
                    help='JSON config file (same format as calibration input)')
parser.add_argument('--n-sim', type=int, default=None,
                    help='Override simulation size')
parser.add_argument('--output-dir', type=str, default='output/fiscal_test',
                    help='Directory for figures and fiscal_results.json')
args = parser.parse_args()

SHOCKS = ['G', 'Ig'] if args.shock == 'both' else [x.strip() for x in args.shock.split(',') if x.strip()]
SCENARIO_KEYS = [x.strip() for x in args.scenarios.split(',') if x.strip()]
for _x in SHOCKS:
    if _x not in ('G', 'Ig', 'health'):
        parser.error(f'unknown shock {_x!r}')
for _x in SCENARIO_KEYS:
    if _x not in ('debt', 'tau_l_debt', 'tau_l_nfa', 'tau_l_window'):
        parser.error(f'unknown scenario {_x!r}')

_t_start = time.perf_counter()  # wall-clock start (includes model build + warmup)

if args.backend == 'jax':
    import jax
    devices = jax.devices()
    dev_type = devices[0].platform.upper() if devices else 'UNKNOWN'
    print(f"JAX backend: {dev_type} ({len(devices)} device(s): {devices})")

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extra periods beyond T_transition to show post-target dynamics
N_POST = 2 if args.tiny else 20

# ---------------------------------------------------------------------------
# 1. Build OLG model
# ---------------------------------------------------------------------------

# Baseline-only fiscal lines; populated in the config branch, left None in the
# hardcoded fast-test branch.
defense_path = None
other_path   = None

if args.config:
    # Build from JSON config file
    from calibrate import build_olg_transition

    with open(args.config) as f:
        config_data = json.load(f)

    economy, paths, T_TR = build_olg_transition(config_data, backend=args.backend)
    economy.output_dir = OUTPUT_DIR
    N_SIM = args.n_sim or config_data.get('transition', {}).get('n_sim', 2000)

    r_path = paths['r_path']
    tax_paths = {k: paths[k] for k in
                 ['tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path',
                  'pension_replacement_path']}

    # The I_g level delta_g*K_g is the stationary public-investment level (keeps
    # K_g flat at K_g_initial); with eta_g != 0 it is the baseline I_g path.
    prod = config_data.get('production', {})
    eta_g_cfg = prod.get('eta_g', 0.0)
    # Stationary public-investment level: (delta_g + Gamma_t - 1) * K_g holds
    # K_g flat in per-capita detrended units. Gamma_t varies while the
    # population is in transition, so the level does too.
    I_g_warmup = ((prod.get('delta_g', 0.05) + economy.growth_factors(T_TR) - 1.0)
                  * prod.get('K_g', 0.0))

    # Government spending lines are fixed shares of Y(t): pass the SS-calibrated
    # ratios and let the budget multiply by each run's realized Y_path, so levels
    # move with output and the shares stay at their SS values (GDP-share mode).
    # Exception: with eta_g != 0 the I_g line is a LEVEL (delta_g + Gamma_t - 1)*K_g, an array not a constant
    # (the stationary level; a GDP-share I_g would need an I_g↔K_g↔Y fixed
    # point and is rejected by simulate_transition).
    G_over_Y       = paths.get('G_over_Y', 0.13)
    I_g_over_Y     = paths.get('I_g_over_Y', 0.03)
    defense_over_Y = paths.get('defense_over_Y', 0.0)
    other_over_Y   = paths.get('other_net_spending_over_Y', 0.0)
    G_path = I_g_path = defense_path = other_path = None  # ratio mode → no levels
    if eta_g_cfg != 0.0:
        I_g_path = I_g_warmup      # level mode for I_g only
    # The lines of 2026-10-07 (BUDGET_ALIGNMENT_PLAN.md): the output tax (a
    # rate path, constant until the fixed point below sets its terminal
    # ramp), the lump-sum transfer (a level path, lambda times output, set by
    # the fixed point), education, the transfer from abroad, the real
    # sovereign-rate path and the unemployment index.
    tau_y_base     = float(paths.get('tau_y', 0.0) or 0.0)
    lump_over_Y    = float(paths.get('lump_sum_over_Y', 0.0) or 0.0)
    new_lines = {
        'tau_y_path': np.full(T_TR, tau_y_base),
        'lump_sum_path': np.full(T_TR, lump_over_Y),
        'education_over_Y0': paths.get('education_over_Y0', 0.0),
        'education_index_path': paths.get('education_index_path'),
        'foreign_transfer_over_Y': paths.get('foreign_transfer_over_Y'),
        'unemployment_index_path': paths.get('unemployment_index_path'),
    }
    if paths.get('r_B_path') is not None:
        new_lines['r_B_path'] = np.asarray(paths['r_B_path'], dtype=float)

elif args.tiny:
    # The small economy of the tests: one education group, every cohort
    # split between two retirement ages, exact aggregation.
    import policy_reference_case as prc
    economy = prc.build_economy(args.backend)
    economy.output_dir = OUTPUT_DIR
    N_SIM = args.n_sim or 100
    T_TR = prc.T_TR
    _bp = prc.base_paths()
    r_path, G_path, I_g_path = _bp['r_path'], _bp['G_path'], _bp['I_g_path']
    tax_paths = {k: _bp[k] for k in ('tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path',
                                     'pension_replacement_path')}
    B_initial = 0.3
    target_B_Y = 0.3
    Y_path = None

else:
    # Hardcoded fast-test parameters (backward compatible)
    T_LC  = 20
    N_H   = 1
    N_SIM = args.n_sim or 2000

    config = LifecycleConfig(
        T              = T_LC,
        beta           = 0.96,
        gamma          = 2.0,
        n_a            = 50,
        n_y            = 4,
        n_h            = N_H,
        retirement_age = 15,
        education_type = 'medium',
        labor_supply   = True,
        nu             = 1.0,
        phi            = 2.0,
        survival_probs = np.linspace(0.995, 0.90, T_LC).reshape(T_LC, N_H),
    )

    economy = OLGTransition(
        lifecycle_config  = config,
        alpha             = 0.33,
        delta             = 0.05,
        A                 = 1.0,
        birth_year        = 2005,
        current_year      = 2020,
        education_shares  = {'medium': 1.0},
        eta_g             = 0.10,
        K_g_initial       = 1.0,
        backend           = args.backend,
        output_dir        = OUTPUT_DIR,
    )

    T_TR = 40
    r_path   = np.full(T_TR, 0.04)
    # Stationary public investment: (delta_g + G - 1) * K_g keeps K_g flat.
    I_g_path = np.full(T_TR, (economy.delta_g + economy.growth_factor - 1.0)
                             * economy.K_g_initial)

    tax_paths = dict(
        tau_l_path               = np.full(T_TR, 0.15),
        tau_c_path               = np.full(T_TR, 0.18),
        tau_p_path               = np.full(T_TR, 0.20),
        tau_k_path               = np.full(T_TR, 0.20),
        pension_replacement_path = np.full(T_TR, 0.60),
    )

    print("Calibrating baseline G (30 %% of Y) …")
    _calib = economy.simulate_transition(
        r_path=r_path, I_g_path=I_g_path, n_sim=50, verbose=False, **tax_paths
    )
    Y_path = np.asarray(_calib['Y'])
    G_path = np.full(T_TR, 0.30 * Y_path.mean())
    B_initial = 0.0
    target_B_Y = 0.0  # hardcoded test: no initial debt, target stays at zero
    print(f"  mean(Y) = {Y_path.mean():.4f},  mean(G) = {G_path.mean():.4f}")

base_paths = dict(r_path=r_path, G_path=G_path, I_g_path=I_g_path, **tax_paths)
if args.config:
    # GDP-share mode: spending lines are ratios of Y(t) (constant shares across
    # scenarios; each scenario's budget multiplies by its own Y_path). With
    # eta_g != 0 the I_g line stays a level (base_paths['I_g_path'] above);
    # omitting 'I_g_over_Y' keeps _build_cf_paths in level mode for I_g only.
    base_paths['G_over_Y']         = G_over_Y
    if eta_g_cfg == 0.0:
        base_paths['I_g_over_Y']   = I_g_over_Y
    base_paths['defense_over_Y']   = defense_over_Y
    base_paths['other_net_over_Y'] = other_over_Y
    base_paths.update({k: v for k, v in new_lines.items() if v is not None})
elif args.tiny:
    base_paths['r_B_path'] = _bp['r_B_path']
else:
    # Baseline-only fiscal lines (no shock applied to them; constant across scenarios).
    if defense_path is not None:
        base_paths['defense_spending_path'] = defense_path
    if other_path is not None:
        base_paths['other_net_spending_path'] = other_path

# The period in which the shock becomes known (unanticipated): households
# alive then follow the baseline before it and re-optimise from their state.
T_S = int(args.shock_year) - int(economy.current_year)
if not 0 <= T_S < T_TR:
    raise SystemExit(f'--shock-year {args.shock_year} is outside the transition '
                     f'({economy.current_year}-{int(economy.current_year) + T_TR - 1})')
print(f"Shock known in {args.shock_year} (period t_s = {T_S}); shocks {SHOCKS}; "
      f"scenarios {SCENARIO_KEYS}")

# Health coverage and the level of medical spending as calendar paths, at the
# configured values in the baseline; the health shock moves them.
KAPPA0 = float(economy.lifecycle_config.kappa)
base_paths['kappa_path'] = np.full(T_TR, KAPPA0)
base_paths['m_scale_path'] = np.ones(T_TR)

# Runs whose household inputs equal those of an earlier run reuse its cohort
# age means: the G shock under debt financing, the first step of each tax
# search, and the steps the two searches of a shock have in common. The
# distributional and welfare outputs need the cohort models (and their value
# functions) of every run, so the reuse is off when they are produced.
DISTRIBUTION = not args.no_distribution
economy.household_cache_size = 0 if DISTRIBUTION else 16
if DISTRIBUTION and economy.jax_policies_on_device:
    raise SystemExit('the welfare outputs need the value functions: jax_policies_on_device must be off')

# One baseline run for all scenarios, at the full N_SIM and horizon.
print("Running the baseline …")
base_paths = run_baseline(economy, base_paths, n_post=N_POST, n_sim=N_SIM, shock_period=T_S)


def _release_models():
    """Drop the cohort models of the last run (policies and value functions,
    tens of GB at production size) once nothing reads them, so that the next
    run does not hold two sets. The baseline models the MIT stitching needs
    stay in economy._mit_baseline_cache."""
    if DISTRIBUTION:
        economy.birth_cohort_solutions = None
        economy.birth_cohort_later = {}
        gc.collect()


_release_models()

if args.config:
    # B_initial = B_over_Y * Y0 and the I_g shock level 0.02 * Y0 are read off
    # the baseline the scenarios are compared with, so B/Y(0) equals B_over_Y
    # to the digit. Y(0) does not depend on the horizon of the run.
    Y_path = np.asarray(base_paths['base_macro']['Y'])[:T_TR]
    Y0 = float(Y_path[0])
    B_over_Y = config_data.get('fiscal', {}).get('B_over_Y', 0.0)
    B_initial = B_over_Y * Y0          # initial debt level pins B/Y at t=0
    target_B_Y = B_over_Y  # tax-financed: return to initial debt ratio
    print(f"  Y(0) = {Y0:.4f}  (baseline run, n_sim={N_SIM})")

    # The baseline's fixed point (baseline_closure.solve_baseline): the
    # lump-sum transfer is lambda times the run's own output and the output
    # tax ramps after 2060 to the rate that holds the debt ratio at its 2070
    # value; both are household inputs, so each iteration is a full
    # transition. The stock-flow adjustment (the data's 2024-25 ratios, the
    # projection's rows to 2060) then enters the debt recursion as levels and
    # the start-of-base-year stock makes end-of-2023 debt equal fiscal.B_over_Y.
    from baseline_closure import solve_baseline
    dsa_cfg = config_data.get('fiscal', {}).get('dsa_projection_file')
    G_growth = economy.growth_factors(T_TR)
    r_B_full = np.asarray(base_paths['r_B_path'], dtype=float)
    # The education line is anchored on the baseline's base-year output in
    # every scenario (e_0 Y_2023 (w_t/w_0) s_t), not on each run's own Y(0).
    base_paths['education_Y0'] = 1.0        # the base-year cross-section's output

    def _run(lump, tau):
        global base_paths
        base_paths['lump_sum_path'] = np.asarray(lump, dtype=float)
        base_paths['tau_y_path'] = np.asarray(tau, dtype=float)
        base_paths = run_baseline(economy, base_paths, n_post=N_POST, n_sim=N_SIM,
                                  shock_period=T_S)
        return (np.asarray(base_paths['base_macro']['Y'])[:T_TR],
                {k: np.asarray(v)[:T_TR] for k, v in base_paths['base_budget'].items()})

    fx = solve_baseline(_run, config_data, T_TR, int(economy.current_year), lump_over_Y,
                        tau_y_base, r_B_full, G_growth, Y_init=Y_path,
                        ramp_years=int(paths.get('tau_y_ramp_years', 10)),
                        tol_Y=1e-4, tol_pb=1e-4, verbose=True,
                        match_projection=(config_data.get('fiscal', {}).get('tau_y_mode', 'constant')
                                          == 'projection'),
                        match_debt_year=(int(config_data.get('fiscal', {}).get('tau_y_debt_year', 2060))
                                         if config_data.get('fiscal', {}).get('tau_y_mode', 'constant') == 'debt' else None),
                        first_mid_year=int(config_data.get('fiscal', {}).get('tau_y_first_year', 2026)),
                        terminal_rule=(config_data.get('fiscal', {}).get('tau_y_mode', 'constant') != 'pinned_throughout'
                                       and bool(config_data.get('fiscal', {}).get('tau_y_terminal_rule', True))))
    Y_path = np.asarray(base_paths['base_macro']['Y'])[:T_TR]
    Y0 = float(Y_path[0])
    debt = fx['debt']
    base_paths['sfa_path'] = np.asarray(debt['sfa'], float) * Y_path
    PD0 = float(np.asarray(base_paths['base_budget']['primary_deficit'])[0])
    r_B0 = float(r_B_full[0])
    B_initial = (B_over_Y * Y0 - PD0) / (1.0 + r_B0)
    t70 = debt['terminal_year'] - int(economy.current_year)
    _tau_y_now = np.asarray(base_paths['tau_y_path'], float)
    _t26 = min(max(int(config_data.get('fiscal', {}).get('tau_y_first_year', 2026))
                   - int(economy.current_year), 0), T_TR - 1)
    print(f"  baseline: tau_y {_tau_y_now[0]:.5f} to {int(economy.current_year) + _t26 - 1}, "
          f"{_tau_y_now[_t26]:.5f} from {int(economy.current_year) + _t26}, "
          f"{fx['tau_terminal']:.5f} from "
          f"{debt['terminal_year']} ({fx['iterations']} iterations); debt "
          f"{100 * debt['debt'][2]:.1f}% in 2025, {100 * debt['debt'][t70]:.1f}% in "
          f"{debt['terminal_year']}; end-{int(economy.current_year)} debt/Y = {B_over_Y}, "
          f"start-of-year stock B_initial = {B_initial:.4f}")
    if eta_g_cfg != 0.0:
        print(f"  G/Y = {G_over_Y}, defense/Y = {defense_over_Y}, "
              f"other_net/Y = {other_over_Y}  (fixed shares of Y(t))")
        print(f"  I_g = {I_g_path[0]:.4f} (level = (delta_g + G - 1)*K_g, K_g flat at "
              f"{prod.get('K_g', 0.0)})")
    else:
        print(f"  G/Y = {G_over_Y}, I_g/Y = {I_g_over_Y}, defense/Y = {defense_over_Y}, "
              f"other_net/Y = {other_over_Y}  (fixed shares of Y(t))")
    print(f"  B/Y = {B_over_Y},  B_initial = {B_initial:.4f}")

# ---------------------------------------------------------------------------
# 3. Define scenarios
# ---------------------------------------------------------------------------

from fiscal_experiments import back_loaded, window_profile, health_cut_paths  # noqa: E402
import distribution_stats as dstat  # noqa: E402

T_TOT = T_TR + N_POST
BASE_YEAR = int(economy.current_year)
# Zero before the shock, one from the shock year on
AFTER = (np.arange(T_TR) >= T_S).astype(float)
# The labour tax moves from the shock year on (permanent schemes) or over the
# years of the health cut only (window scheme).
PSI_PERM = back_loaded(T_TOT, T_S)
PSI_WINDOW = window_profile(T_TOT, T_S, args.health_years)


def health_targets_from_data(window, household):
    """Mean change over the window years relative to the year before, in
    shares of output: government health spending (HF.1) and household health
    spending (CHE less HF.1, or out-of-pocket HF.3)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'health_flag_GR.csv')
    d = np.genfromtxt(path, delimiter=',', names=True)
    y0, y1 = (int(x) for x in window.split('-'))
    years = d['year'].astype(int)
    gov = d['gov_health_gdp']
    hh = d['che_gdp'] - d['gov_health_gdp'] if household == 'che_minus_gov' else d['oop_gdp']
    win = (years >= y0) & (years <= y1)
    ref = years == y0 - 1
    return float(gov[win].mean() - gov[ref][0]), float(hh[win].mean() - hh[ref][0])


scn_base = FiscalScenario(name='baseline', financing='debt', B_initial=B_initial,
                          n_post=N_POST, shock_period=T_S)

# G: 2% of Y(t) from the shock year (ratio in config mode; level in test mode)
delta_G = (0.02 * AFTER if args.config
           else 0.02 * float(np.mean(Y_path if Y_path is not None
                                     else base_paths['base_macro']['Y'][:T_TR])) * AFTER)

# I_g: config mode with eta_g != 0, a constant detrended level equal to 2% of
# baseline output in the year before the shock (Y(0) for a shock in the base
# year); a ratio of Y(t) when I_g is in GDP-share mode; 2% of mean output in
# the test economies.
_Y_base = np.asarray(base_paths['base_macro']['Y'], float)
if args.config:
    if eta_g_cfg == 0.0:
        delta_Ig = 0.02 * AFTER
        IG_LEVEL = None
    else:
        IG_LEVEL = 0.02 * float(_Y_base[T_S - 1] if T_S > 0 else _Y_base[0])
        delta_Ig = IG_LEVEL * AFTER
else:
    IG_LEVEL = 0.02 * float(np.mean(_Y_base[:T_TR]))
    delta_Ig = IG_LEVEL * AFTER
if 'Ig' in SHOCKS and IG_LEVEL is not None:
    print(f"  I_g shock: {IG_LEVEL:.5f} from {args.shock_year} "
          f"({100 * IG_LEVEL / _Y_base[T_S]:.2f}% of baseline output in {args.shock_year})")

# Health: a cut in coverage and in the level of medical spending over the
# window, calibrated on baseline output to the targets.
HEALTH = None
if 'health' in SHOCKS:
    if args.health_targets:
        D_GOV, D_HH = (float(x) for x in args.health_targets.split(','))
        _src = 'command line'
    else:
        D_GOV, D_HH = health_targets_from_data(args.health_window, args.health_household)
        _src = f'data, mean {args.health_window} less {int(args.health_window[:4]) - 1}, ' \
               f'household = {args.health_household}'
    HEALTH = health_cut_paths(base_paths, D_GOV, D_HH, T_S, args.health_years, T_TR)
    HEALTH.update(d_gov_gdp=D_GOV, d_hh_gdp=D_HH, source=_src, years=int(args.health_years),
                  window=args.health_window, household=args.health_household)
    print(f"  health targets ({_src}): gov {100 * D_GOV:+.2f} pp, household {100 * D_HH:+.2f} pp "
          f"of output; baseline window shares gov {100 * HEALTH['g0']:.2f}%, household "
          f"{100 * HEALTH['o0']:.2f}%; kappa {HEALTH['kappa0']:.3f} -> {HEALTH['kappa1']:.3f}, "
          f"medical spending x {HEALTH['mu1']:.3f}")


def _health_deltas(kappa1, mu1):
    n = args.health_years
    dk, dm = np.zeros(T_TR), np.zeros(T_TR)
    dk[T_S:T_S + n] = kappa1 - HEALTH['kappa0']
    dm[T_S:T_S + n] = mu1 - 1.0
    return {'delta_kappa_path': dk, 'delta_m_scale_path': dm}


def shock_kwargs(shock):
    if shock == 'G':
        return {'delta_G_path': delta_G}
    if shock == 'Ig':
        return {'delta_I_g_path': delta_Ig}
    return _health_deltas(HEALTH['kappa1'], HEALTH['mu1'])


SCENARIO_LABELS = {
    'debt': ('debt_financed', 'debt-financed'),
    'tau_l_debt': ('tax_financed', 'τ_l, debt-ratio target'),
    'tau_l_nfa': ('nfa_constrained', 'τ_l, NFA target'),
    'tau_l_window': ('tax_financed_window', 'τ_l over the cut, debt-ratio target'),
}


def make_scenario(shock, key, **over):
    kw = dict(name=f'{shock}_{key}', B_initial=B_initial, n_post=N_POST, shock_period=T_S)
    kw.update(shock_kwargs(shock))
    if key == 'debt':
        kw['financing'] = 'debt'
    else:
        kw['financing'] = 'tau_l'
        kw['adjustment_profile'] = PSI_WINDOW if key == 'tau_l_window' else PSI_PERM
    kw.update(over)
    return FiscalScenario(**kw)


# ---------------------------------------------------------------------------
# 4. Run experiments
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Running fiscal experiments …")
print("=" * 60)

t0 = time.time()

PERIODS = dstat.reporting_periods(T_TR, BASE_YEAR)
if T_S not in PERIODS:
    PERIODS = sorted(PERIODS + [T_S])
# Cohorts entering after the shock year whose welfare is reported: entry
# years up to 2070 (or the end of the transition).
NEWBORN_BPS = list(range(T_S + 1, min(T_TR, 2070 - BASE_YEAR + 1)))


def _extract(res):
    """Distribution of the run that produced *res* (the last one)."""
    if not DISTRIBUTION:
        return None
    if not np.array_equal(np.asarray(economy.Y_path), np.asarray(res.cf_macro['Y'])):
        raise RuntimeError(f'{res.scenario.name}: the last run is not the reported one')
    t1 = time.time()
    ext = dstat.extract(economy, PERIODS, t_s=T_S, newborn_bps=NEWBORN_BPS, period_chunk=7)
    print(f"      distribution and value functions extracted in {time.time() - t1:.1f}s")
    return ext


def _run(scn, **kw):
    t1 = time.time()
    res = run_fiscal_scenario(economy, scn, base_paths, n_sim=N_SIM, verbose=False, **kw)
    extra = (f", {res.n_iterations} evaluations, Δτ_l = {100 * res.adjustment_scalar:+.3f} pp, "
             f"converged = {res.converged}" if scn.financing != 'debt' else '')
    print(f"      {scn.name}: {time.time() - t1:.1f}s{extra}")
    ext = _extract(res)
    _release_models()
    return res, ext


def run_experiment_set(shock):
    """{scenario key: (result, extraction)} for the baseline and the requested
    scenarios of one shock, with the labels."""
    out, labels = {}, {}
    print(f"\n[{shock}] baseline …")
    out['baseline'] = _run(scn_base)
    res_base = out['baseline'][0]

    # The tau_l schemes return debt/output at T_balance to the baseline's
    # (dated as _balance_residual dates terminal_debt_gdp: B[T_bal-1] over
    # Y[T_bal-1]); the NFA scheme does the same for NFA/Y.
    T_bal = res_base.T_balance or len(res_base.cf_macro['Y'])
    target_base = float(res_base.B_path[T_bal - 1] / res_base.cf_macro['Y'][T_bal - 1])
    target_nfa = float(res_base.NFA_path[T_bal - 1] / res_base.cf_macro['Y'][T_bal - 1])
    print(f"      τ_l target: baseline B/Y at T_bal = {target_base:.4f}; NFA/Y = {target_nfa:.4f}")

    keys = [k for k in SCENARIO_KEYS if k != 'tau_l_window' or shock == 'health']
    for key in keys:
        name, lab = SCENARIO_LABELS[key]
        over = {}
        if key in ('tau_l_debt', 'tau_l_window'):
            over = dict(balance_condition='terminal_debt_gdp', target_debt_gdp=target_base)
        elif key == 'tau_l_nfa':
            over = dict(balance_condition='terminal_nfa_gdp', target_nfa_gdp=target_nfa)
        print(f"[{shock}] {lab} …")
        init = 0.0
        if key == 'tau_l_nfa' and 'tax_financed' in out and out['tax_financed'][0].converged:
            init = out['tax_financed'][0].adjustment_scalar
        out[name] = _run(make_scenario(shock, key, **over),
                         **({} if key == 'debt' else dict(bisect_tol=1e-3, bisect_init=init)))
        labels[name] = lab

    if shock == 'health':
        # Decomposition: coverage only (kappa1, medical spending unchanged)
        # and medical spending only (kappa0, mu1), both debt-financed.
        for name, (k1, m1), lab in (('debt_financed_kappa_only', (HEALTH['kappa1'], 1.0),
                                     'coverage only (debt)'),
                                    ('debt_financed_m_only', (HEALTH['kappa0'], HEALTH['mu1']),
                                     'medical spending only (debt)')):
            print(f"[{shock}] {lab} …")
            scn = make_scenario(shock, 'debt', name=f'health_{name}', **_health_deltas(k1, m1))
            out[name] = _run(scn)
            labels[name] = lab
    return out, labels


experiment_results = {st: run_experiment_set(st) for st in SHOCKS}

print(f"\nAll experiments done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 5. Post-processing: attach derived quantities to each result
# ---------------------------------------------------------------------------

def _post(res):
    """Interest payments at r_B, K_g/Y and the labour-tax rate of a result."""
    _T = len(res.cf_macro['Y'])
    _B = res.B_path[:_T]
    # Interest at the sovereign rate r_B (the rate in the B law of motion,
    # a path by period since 2026-10-07), not the capital return r.
    _rbp = base_paths.get('r_B_path')
    if _rbp is not None:
        _r = economy._as_period_path(np.asarray(_rbp, dtype=float), _T)
    else:
        _rB = getattr(economy, 'r_B', None)
        _r = (np.full(_T, float(_rB)) if _rB is not None
              else np.asarray(res.cf_macro['r']))
    res.cf_budget['interest_payments'] = _r * _B
    _Kg = res.cf_macro.get('K_g')
    if _Kg is not None:
        res.cf_macro['K_g_Y'] = np.asarray(_Kg)[:_T] / np.asarray(res.cf_macro['Y'])
    # The labour-tax rate: the base path plus the scheme's change
    _tl = economy._as_period_path(np.asarray(base_paths['tau_l_path'], float), _T)
    res.cf_macro['tau_l'] = _tl + np.asarray(res.adjustment_path, float)[:_T]


for runs, _ in experiment_results.values():
    for res, _ext in runs.values():
        _post(res)


def multipliers(base, cf, shock):
    """Output response over the change in the spending line, both as aggregate
    flows (the detrended per-capita paths re-trended by Gamma): on impact (the
    shock year), cumulated over the shock year and the nine after, and over
    the horizon to T_balance. For the health shock the spending line is
    government health spending plus the means-tested transfers, which absorb
    part of the cut; a cut is a negative spending change."""
    dY = np.asarray(cf.cf_macro['Y'], float) - np.asarray(base.cf_macro['Y'], float)
    if shock == 'health':
        line = lambda r: (np.asarray(r.cf_budget['gov_health'], float)
                          + np.asarray(r.cf_budget['transfers'], float))
    else:
        key = 'govt_spending' if shock == 'G' else 'public_investment'
        line = lambda r: np.asarray(r.cf_budget[key], float)
    dS = line(cf) - line(base)
    G = economy.growth_factors(len(dY))
    scale = np.concatenate([[1.0], np.cumprod(np.asarray(G, float)[:-1])])
    T_bal = cf.T_balance or len(dY)

    def cum(a, b):
        den = float(np.sum((scale * dS)[a:b]))
        return float(np.sum((scale * dY)[a:b]) / den) if den != 0.0 else float('nan')
    return {'impact': float(dY[T_S] / dS[T_S]) if dS[T_S] != 0.0 else float('nan'),
            'cumulative_10y': cum(T_S, T_S + 10),
            'cumulative_horizon': cum(T_S, T_bal),
            'path': [float(x) for x in np.where(dS != 0.0, dY / np.where(dS != 0.0, dS, 1.0),
                                                np.nan)]}


def health_realised(base, cf):
    """Mean change over the window in government and household health
    spending over output: each run's spending over its own output, against
    the baseline's."""
    w = slice(T_S, T_S + args.health_years)
    s = lambda r, k: np.asarray(r.cf_budget[k], float)[w] / np.asarray(r.cf_macro['Y'], float)[w]
    return {'d_gov_gdp': float(np.mean(s(cf, 'gov_health') - s(base, 'gov_health'))),
            'd_hh_gdp': float(np.mean(s(cf, 'oop_health') - s(base, 'oop_health'))),
            'd_transfers_gdp': float(np.mean(s(cf, 'transfers') - s(base, 'transfers')))}


if HEALTH is not None:
    _hb = experiment_results['health'][0]['baseline'][0]
    _hd = experiment_results['health'][0].get('debt_financed', (None,))[0]
    if _hd is not None:
        HEALTH['realised_debt'] = health_realised(_hb, _hd)
        _gap = max(abs(HEALTH['realised_debt']['d_gov_gdp'] - HEALTH['d_gov_gdp']),
                   abs(HEALTH['realised_debt']['d_hh_gdp'] - HEALTH['d_hh_gdp']))
        print(f"  health, debt-financed: realised change in gov {100 * HEALTH['realised_debt']['d_gov_gdp']:+.3f} pp, "
              f"household {100 * HEALTH['realised_debt']['d_hh_gdp']:+.3f} pp of output "
              f"(gap to the targets {100 * _gap:.3f} pp)")
        if _gap > 0.001:
            # One pass on counterfactual output: the levels are reset so that
            # the debt-financed run's spending over its own output meets the
            # targets, and the health set is run again.
            _first = {k: HEALTH[k] for k in ('kappa1', 'mu1')}
            _new = health_cut_paths(base_paths, HEALTH['d_gov_gdp'], HEALTH['d_hh_gdp'], T_S,
                                    args.health_years, T_TR,
                                    Y_eval=np.asarray(_hd.cf_macro['Y'], float))
            HEALTH.update({k: _new[k] for k in ('kappa1', 'mu1', 'delta_kappa_path',
                                                 'delta_m_scale_path')})
            HEALTH['first_pass'] = _first
            print(f"  gap above 0.1 pp: medical spending x {HEALTH['mu1']:.4f} on counterfactual "
                  f"output; rerunning the health set")
            experiment_results['health'] = run_experiment_set('health')
            for res, _ext in experiment_results['health'][0].values():
                _post(res)
            _hb = experiment_results['health'][0]['baseline'][0]
            _hd = experiment_results['health'][0]['debt_financed'][0]
            HEALTH['realised_debt'] = health_realised(_hb, _hd)

# ---------------------------------------------------------------------------
# 6. Figure variables
# ---------------------------------------------------------------------------

MACRO_VARS = ['Y', 'C', 'K_domestic', 'L', 'w', 'B_gdp_path', 'A', 'A_gdp', 'NFA_gdp',
              'K_g_Y', 'tau_l']
MACRO_LABELS = {
    'Y':          'Output (Y)',
    'C':          'Consumption (C)',
    'K_domestic': 'Domestic capital (K)',
    'L':          'Labour, efficiency units (L)',
    'w':          'Wage rate (w)',
    'B_gdp_path': 'Debt / GDP (B/Y)',
    'A':          'Household wealth (A)',
    'A_gdp':      'Household wealth / Y (A/Y)',
    'NFA_gdp':    'Net foreign assets / Y (NFA/Y)',
    'K_g_Y':      'Public capital / Y (K_g/Y)',
    'tau_l':      'Labour income tax rate',
}

PRICE_VARS   = ['w', 'r']
PRICE_LABELS = {'w': 'Wage rate (w)', 'r': 'Interest rate (r)'}

FISCAL_VARS = [
    'primary_deficit_gdp', 'total_revenue_gdp', 'total_spending_gdp',
    'tax_l_gdp', 'tax_c_gdp', 'tax_p_gdp', 'tax_k_gdp',
    'ui_gdp', 'pension_gdp', 'govt_spending', 'public_investment_gdp',
    'gov_health_gdp', 'oop_health_gdp', 'medical_total_gdp', 'transfers_gdp',
    'defense_spending', 'other_net_spending',
    'tax_y_gdp', 'foreign_transfer', 'education', 'lump_sum',
    'interest_payments',
]
FISCAL_LABELS = {
    'primary_deficit_gdp': 'Primary deficit / Y',
    'total_revenue_gdp': 'Revenue / Y',
    'total_spending_gdp': 'Primary spending / Y',
    'tax_l_gdp':         'Labour tax / Y',
    'tax_c_gdp':         'Consumption tax / Y',
    'tax_p_gdp':         'Payroll tax / Y',
    'tax_k_gdp':         'Capital tax / Y',
    'ui_gdp':            'UI benefits / Y',
    'pension_gdp':       'Pensions / Y',
    'govt_spending':     'Govt spending (G)',
    'public_investment_gdp': 'Public investment / Y (I_g/Y)',
    'gov_health_gdp':    'Government health spending / Y',
    'oop_health_gdp':    'Household health spending / Y',
    'medical_total_gdp': 'Total medical spending / Y',
    'transfers_gdp':     'Means-tested transfers / Y',
    'defense_spending':  'Defense',
    'other_net_spending':'Other net spending',
    'tax_y_gdp':         'Output tax / Y',
    'foreign_transfer':  'Transfer from abroad',
    'education':         'Education',
    'lump_sum':          'Lump-sum transfer',
    'interest_payments': 'Interest payments (r_B·B)',
}

# ---------------------------------------------------------------------------
# 7. Print key scalars + produce figures per experiment set
# ---------------------------------------------------------------------------

print("\n--- Key results ---")
MULTIPLIERS = {}
for shock_type, (runs, labels) in experiment_results.items():
    res_base = runs['baseline'][0]
    print(f"\n  [{shock_type}] baseline: final B/Y = {res_base.B_gdp_path[-1] * 100:.1f}%")
    MULTIPLIERS[shock_type] = {}
    for name, (res, _ext) in runs.items():
        if name == 'baseline':
            continue
        line = f"  [{shock_type}] {labels[name]:<38}: final B/Y = {res.B_gdp_path[-1] * 100:.1f}%"
        if res.scenario.financing != 'debt':
            line += (f", Δτ_l = {res.adjustment_scalar * 100:+.3f} pp, "
                     f"converged = {res.converged}")
        m = multipliers(res_base, res, shock_type)
        MULTIPLIERS[shock_type][name] = m
        line += (f"; multiplier impact {m['impact']:.3f}, 10 years {m['cumulative_10y']:.3f}, "
                 f"horizon {m['cumulative_horizon']:.3f}")
        print(line)

print("\n--- Saving figures ---")
for shock_type, (runs, labels) in experiment_results.items():
    p = shock_type.lower() + '_'
    order = ['baseline'] + [k for k in runs if k != 'baseline']
    results = [runs[k][0] for k in order]
    title_suffix = f'({shock_type} shock, {args.shock_year})'
    for variables, var_labels, name, fname in (
            (MACRO_VARS, MACRO_LABELS, 'Macro Overview', 'macro_overview.png'),
            (PRICE_VARS, PRICE_LABELS, 'Prices — SOE sanity check', 'prices_sanity.png'),
            (FISCAL_VARS, FISCAL_LABELS, 'Fiscal Decomposition', 'fiscal_decomp.png')):
        compare_scenarios(results[0], *results[1:], variables=variables, var_labels=var_labels,
                          title=f'{name} {title_suffix}', output_dir=OUTPUT_DIR,
                          filename=f'{p}{fname}', base_year=BASE_YEAR)
        plt.close('all')
    debt_fan_chart(results, ['baseline'] + [labels[k] for k in order[1:]],
                   output_dir=OUTPUT_DIR, filename=f'{p}debt_fan_chart.png', base_year=BASE_YEAR)
    plt.close('all')

# ---------------------------------------------------------------------------
# 8. Save numerical results to JSON for remote inspection
# ---------------------------------------------------------------------------

def _result_to_dict(res):
    """Convert a FiscalScenarioResult to a JSON-serializable dict."""
    d = {
        'converged': bool(res.converged) if res.converged is not None else None,
        'terminal_converged': bool(res.terminal_converged),
        'terminal_drift': {k: float(v) for k, v in res.terminal_drift.items()},
        'balance_condition': res.scenario.balance_condition,
        'T_balance': res.T_balance,
        'n_post': res.scenario.n_post,
        'shock_period': int(res.scenario.shock_period),
        'adjustment_scalar': float(res.adjustment_scalar) if res.adjustment_scalar else None,
        'adjustment_path': [float(x) for x in np.asarray(res.adjustment_path, float)],
        'n_iterations': int(res.n_iterations),
        'residual_history': [float(x) for x in res.residual_history],
        'B_gdp_path': [float(x) for x in res.B_gdp_path],
    }
    for label, macro in [('baseline', res.base_macro), ('counterfactual', res.cf_macro)]:
        d[label] = {}
        for k, v in macro.items():
            try:
                d[label][k] = [float(x) for x in np.asarray(v)]
            except (TypeError, ValueError):
                pass
    for label, budget in [('base_budget', res.base_budget), ('cf_budget', res.cf_budget)]:
        d[label] = {}
        for k, v in budget.items():
            try:
                d[label][k] = [float(x) for x in np.asarray(v)]
            except (TypeError, ValueError):
                pass
    if res.scenario.delta_kappa_path is not None:
        d['health_paths'] = {
            'delta_kappa_path': [float(x) for x in res.scenario.delta_kappa_path],
            'delta_m_scale_path': [float(x) for x in res.scenario.delta_m_scale_path]}
    return d


def _welfare(base_ext, ext):
    """lambda_c by year of entry into the model (cohorts alive in the shock
    year and later entrants) and lambda by quintile of baseline income."""
    w = dstat.welfare_summary(base_ext, ext, economy.education_shares)
    return {'entry_year': [BASE_YEAR + int(b) for b in w['cohort']],
            'cohort_cev': list(w['cohort'].values()),
            'alive_in_shock_year': [int(b) <= T_S for b in w['cohort']],
            'alive_mean': w['alive_mean'],
            'quintile_cev': w['quintile']}


def health_decomposition(block):
    """Total effect of the debt-financed cut, its two parts (coverage only,
    medical spending only) and the residual total - parts, for the main
    aggregates (% deviation of Y and C, change in debt/output in pp, at
    reporting years) and for the mean consumption-equivalent variation of the
    cohorts alive in the shock year."""
    names = ('debt_financed', 'debt_financed_kappa_only', 'debt_financed_m_only')
    out = {}
    for y in (args.shock_year, args.shock_year + 4, 2040, 2070):
        t = y - BASE_YEAR
        if not 0 <= t < T_TR:
            continue
        for var in ('Y', 'C'):
            v = [100 * (block[n]['counterfactual'][var][t] / block[n]['baseline'][var][t] - 1)
                 for n in names]
            out[f'{var}_{y}'] = {'total': v[0], 'kappa_only': v[1], 'm_only': v[2],
                                 'residual': v[0] - v[1] - v[2]}
        v = [100 * (block[n]['B_gdp_path'][t] - block['baseline']['B_gdp_path'][t]) for n in names]
        out[f'B_gdp_{y}'] = {'total': v[0], 'kappa_only': v[1], 'm_only': v[2],
                             'residual': v[0] - v[1] - v[2]}
    if all('welfare' in block[n] for n in names):
        v = [block[n]['welfare']['alive_mean'] for n in names]
        out['cev_alive_mean'] = {'total': v[0], 'kappa_only': v[1], 'm_only': v[2],
                                 'residual': v[0] - v[1] - v[2]}
    return out


results_out = {}
for shock_type, (runs, labels) in experiment_results.items():
    base_ext = runs['baseline'][1]
    block = {}
    for name, (res, ext) in runs.items():
        d = _result_to_dict(res)
        if ext is not None:
            d['distribution'] = dstat.serialisable(ext)
            if name != 'baseline' and 'welfare' in ext:
                d['welfare'] = _welfare(base_ext, ext)
        if name != 'baseline':
            d['label'] = labels[name]
            d['multiplier'] = MULTIPLIERS[shock_type][name]
        if shock_type == 'health' and name != 'baseline':
            d['health_realised'] = health_realised(runs['baseline'][0], res)
        block[name] = d
    if 'debt_financed' in runs and shock_type in ('G', 'Ig'):
        # The per-period multiplier of the debt-financed run (as before)
        mp = MULTIPLIERS[shock_type]['debt_financed']['path']
        block['fiscal_multiplier_mean'] = float(np.nanmean(mp[T_S:]))
        block['fiscal_multiplier_path'] = mp
    if shock_type == 'health' and all(k in block for k in ('debt_financed', 'debt_financed_kappa_only',
                                                          'debt_financed_m_only')):
        block['decomposition'] = health_decomposition(block)
    results_out[shock_type] = block

params_out = {
    'alpha':         float(economy.alpha),
    'delta':         float(economy.delta),
    'eta_g':         float(economy.eta_g),
    'labor_supply':  bool(getattr(economy.lifecycle_config, 'labor_supply', False)),
    'n_sim':         int(N_SIM),
    'T_transition':  int(T_TR),
    'B_initial':     float(B_initial),
    'target_debt_gdp': float(target_B_Y),
    'base_year':     int(economy.current_year),
    # The baseline closure (baseline_closure.py): other net spending as a
    # path of shares of each run's output, the stock-flow adjustment as
    # levels entering the debt recursion, both None without a projection.
    'other_net_over_Y_path': ([float(x) for x in base_paths['other_net_over_Y']]
                              if args.config and np.ndim(base_paths.get('other_net_over_Y', 0.0)) == 1
                              else None),
    'sfa_path':      ([float(x) for x in base_paths['sfa_path']]
                      if args.config and base_paths.get('sfa_path') is not None else None),
    # The lines of 2026-10-07: the output-tax path, the lump-sum level path,
    # the education inputs, the transfer from abroad and the real
    # sovereign-rate path, as the runs used them.
    'tau_y_path':    ([float(x) for x in base_paths['tau_y_path']]
                      if args.config and base_paths.get('tau_y_path') is not None else None),
    'lump_sum_path': ([float(x) for x in base_paths['lump_sum_path']]
                      if args.config and base_paths.get('lump_sum_path') is not None else None),
    'education_over_Y0': (float(base_paths.get('education_over_Y0') or 0.0) if args.config else None),
    'education_index_path': ([float(x) for x in base_paths['education_index_path']]
                             if args.config and base_paths.get('education_index_path') is not None
                             else None),
    'foreign_transfer_over_Y_path': (
        [float(x) for x in np.atleast_1d(base_paths['foreign_transfer_over_Y'])]
        if args.config and base_paths.get('foreign_transfer_over_Y') is not None else None),
    'r_B_path':      ([float(x) for x in np.atleast_1d(base_paths['r_B_path'])]
                      if base_paths.get('r_B_path') is not None else None),
    'tau_l_path':    [float(x) for x in base_paths['tau_l_path']],
    'tau_c_path':    [float(x) for x in base_paths['tau_c_path']],
    'tau_p_path':    [float(x) for x in base_paths['tau_p_path']],
    'tau_k_path':    [float(x) for x in base_paths['tau_k_path']],
    'delta_G_path':  [float(x) for x in delta_G] if 'G'  in SHOCKS else None,
    'delta_Ig_path': [float(x) for x in delta_Ig] if 'Ig' in SHOCKS else None,
    # The shock year: every scenario is unanticipated in period shock_period
    # (t = 0 is base_year); households alive then follow the baseline before.
    'shock_year':    int(args.shock_year),
    'shock_period':  int(T_S),
    'scenarios':     SCENARIO_KEYS,
    # Health coverage and the medical-spending multiplier of the baseline
    'kappa_path':    [float(x) for x in base_paths['kappa_path']],
    'm_scale_path':  [float(x) for x in base_paths['m_scale_path']],
    'health':        ({k: ([float(x) for x in v] if isinstance(v, np.ndarray) else v)
                       for k, v in HEALTH.items()} if HEALTH is not None else None),
    # Distributional outputs: reporting periods, and model age 0 is real age 25
    'reporting_periods': PERIODS,
    'entry_age':     25,
    'r_B':           (float(economy.r_B) if getattr(economy, 'r_B', None) is not None
                      else None),
    # The return on household wealth, so the evaluator can form net factor
    # income r*NFA + (r - r_B)*B in the resource constraint; without it the
    # checker silently tested the closed-economy form.
    'r':             float(r_path[0]),
    # Household-side parameters the checker needs: the public share of medical
    # spending (to form total medical spending from the gov_health line), the
    # bequest tax and the consumption floor in force.
    'kappa':         float(getattr(economy.lifecycle_config, 'kappa', 1.0)),
    'tau_beq':       float(getattr(economy.lifecycle_config, 'tau_beq', 0.0)),
    'transfer_floor': float(getattr(economy.lifecycle_config, 'transfer_floor', 0.0) or 0.0),
    # Which population the aggregates are divided by. Consumers of this JSON
    # must refuse it if they work on a different convention: before 2026-10-01
    # the transition divided by everyone ever entered, which is 11% smaller than
    # per living person at t=0, and nothing recorded the difference.
    'aggregation':   'per_living_person',
    'trend_growth':  float(getattr(economy, 'trend_growth', 0.0)),
    'pop_growth':    float(economy.pop_growth),
    # Gamma_t over the run's horizon. pop_growth alone no longer pins it when
    # the population comes from a demographic path, and the evaluator needs
    # the same sequence the recursions used.
    'growth_factor_path': [float(x) for x in economy.growth_factors(T_TR)],
    'delta_g':       float(economy.delta_g),
    # Production parameters, so a consumer can verify identities rather than
    # guess at them. Their absence is why no invocation of eval_fiscal_results
    # could validate the July runs: it fell back to r for r_B and to Gamma = 1.
    'delta':         float(economy.delta),
    'alpha':         float(economy.alpha),
    'eta_g':         float(economy.eta_g),
    'K_g_initial':   float(economy.K_g_initial),
    'A_tfp':         float(economy.A),
    'shock_mode_G':  'ratio' if args.config else 'level',
    'shock_mode_Ig': ('ratio' if (args.config and eta_g_cfg == 0.0) else 'level')
                     if 'Ig' in SHOCKS else None,
}
if args.config:
    fiscal = config_data.get('fiscal', {})
    for key in ['pensions_over_Y', 'ui_over_Y', 'health_gov_over_Y', 'health_oop_over_Y', 'health_total_over_Y', 'G_over_Y', 'tax_revenue_over_Y']:
        if key in fiscal:
            params_out[key] = fiscal[key]

results_out['params'] = params_out

results_path = os.path.join(OUTPUT_DIR, 'fiscal_results.json')
with open(results_path, 'w') as f:
    json.dump(results_out, f, indent=2)

print(f"\nFigures saved to {OUTPUT_DIR}/")
print(f"Numerical results saved to {results_path}")

_elapsed = time.perf_counter() - _t_start
print(f"Total run time: {_elapsed:.1f}s ({_elapsed / 60:.1f} min)  "
      f"[backend={args.backend}, shock={args.shock}, n_sim={N_SIM}]")
print(f"Household solves reused from an earlier run: {economy._household_cache_hits}")
