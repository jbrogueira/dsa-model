"""
Targeted check: A[0] predetermination under MIT stitching, with permanent-FE
heterogeneity active (n_alpha>1, sigma_alpha>0), on both backends.

A tau_l shock at t>=0 must leave the t=0 cross-section of household wealth
unchanged: pre-transition cohorts' policies for ages < -birth_period are
stitched from a pure-baseline solve, so simulated assets entering t=0 are
identical between baseline and counterfactual. Exercises the *_policy_alpha
stitching (both simulate paths read the per-alpha arrays).

The '+ret' cases give cohorts three different retirement ages and split some
cohorts between two, so the batched JAX solve and simulation run one group per
age and the MIT baseline models, the later-retiring parts' included, must
inherit each cohort's ages.
The Ig case exercises the K_g→w channel: an I_g (level) shock with eta_g != 0
moves K_g and hence the wage path, so the MIT baseline model must be built
from pure-baseline wages. A[0] must still be exactly baseline.
The 'Ig+rerun' case follows the fiscal driver's sequence: a baseline run, a
second baseline run with another closure path served from the household
cache (which keeps no cohort models), then the I_g scenario, with a shock
large enough to move the wage by several percent. The stitching models must
then come from the pre-transition dict's own wage path.

With --shock-period t_s the shock is announced in t_s instead of t = 0: the
shock is zero before t_s, every cohort alive in t_s keeps the baseline's
policies at the ages it lived before t_s, and household wealth through t_s
must equal the baseline's.

With --savings-solver egm the households solve the savings choice by the
endogenous grid method (log utility, a minimum income benefit of 0.1).

Usage: python check_a0_predetermination.py [--shock-period 3] [--savings-solver egm]
"""
import argparse
import os
import platform
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

from olg_transition import OLGTransition
from lifecycle_perfect_foresight import LifecycleConfig
from fiscal_experiments import FiscalScenario, run_fiscal_scenario, run_baseline

T_TR = 10
N_SIM = 50
_ap = argparse.ArgumentParser()
_ap.add_argument('--shock-period', type=int, default=0)
_ap.add_argument('--savings-solver', choices=('grid', 'egm'), default='grid')
_args = _ap.parse_args()
T_S = _args.shock_period
SOLVER_KW = (dict(savings_solver='egm', gamma=1.0, minimum_income=0.1)
             if _args.savings_solver == 'egm' else {})
PRE = np.r_[np.zeros(T_S), np.ones(T_TR - T_S)]   # zero before the shock
TREND_GROWTH = 0.017   # balanced-growth rate under test; 0.0 recovers the old harness

def run_backend(backend, shock, cohort_retirement=False, rerun=False):
    cfg = LifecycleConfig(T=20, n_a=30, n_y=3, n_alpha=3, retirement_age=12,
                          trend_growth=TREND_GROWTH, **SOLVER_KW)
    ep = dict(cfg.edu_params)
    ep['medium'] = dict(ep['medium'], sigma_alpha=0.3)
    cfg = cfg._replace(edu_params=ep)
    olg_kwargs = dict(lifecycle_config=cfg, backend=backend,
                      education_shares={'medium': 1.0})
    if shock == 'Ig':
        olg_kwargs.update(eta_g=0.05, K_g_initial=0.745, delta_g=0.05)
    if cohort_retirement:
        # Entry years current_year + birth period (2020 - 19 ... 2020 + 9).
        lam = lambda J: (1 - 0.95 ** J) / (J * (1 - 0.95))
        # Three ages, and cohorts entering 2010-2015 split 70/30 between 12
        # and 13, so the later-retiring parts are solved and stitched too.
        def parts(k):
            if 2010 <= k <= 2015:
                return ((12, lam(12), 0.7), (13, lam(13), 0.3))
            J = 12 if k < 2010 else 13 if k < 2022 else 14
            return ((J, lam(J), 1.0),)
        olg_kwargs['cohort_retirement'] = {k: parts(k) for k in range(2001, 2030)}
    olg = OLGTransition(**olg_kwargs)
    bp = dict(
        r_path=np.full(T_TR, 0.04),
        tau_l_path=np.full(T_TR, 0.15),
        tau_c_path=np.full(T_TR, 0.0),
        tau_p_path=np.full(T_TR, 0.0),
        tau_k_path=np.full(T_TR, 0.0),
        pension_replacement_path=np.full(T_TR, 0.4),
    )
    if shock == 'Ig':
        # stationary baseline I_g level keeps K_g flat; the shock is a level delta
        bp['I_g_path'] = (0.05 + olg.growth_factors(T_TR) - 1.0) * 0.745
        scen = FiscalScenario(
            name='Ig_shock',
            delta_I_g_path=(0.5 if rerun else 0.02) * PRE,
            financing='debt',
            balance_condition='terminal_debt_gdp',
            B_initial=0.0,
            shock_period=T_S,
        )
    else:
        scen = FiscalScenario(
            name='tau_l_shock',
            delta_tau_l_path=0.05 * PRE,
            financing='debt',
            balance_condition='terminal_debt_gdp',
            B_initial=0.0,
            shock_period=T_S,
        )
    if rerun:
        olg.household_cache_size = 16
        bp['other_net_over_Y'] = np.full(T_TR, -0.05)
        bp = run_baseline(olg, bp, n_post=0, n_sim=N_SIM, shock_period=T_S)
        bp['other_net_over_Y'] = np.full(T_TR, -0.06)
        bp = run_baseline(olg, bp, n_post=0, n_sim=N_SIM, shock_period=T_S)
        assert olg._household_cache_hits == 1, "the rerun was not served from the household cache"
    elif T_S > 0:
        bp = run_baseline(olg, bp, n_post=0, n_sim=N_SIM, shock_period=T_S)
    res = run_fiscal_scenario(olg, scen, bp, n_sim=N_SIM, verbose=False)
    # Wealth entering t = 0, ..., t_s
    A0_base = np.asarray(res.base_macro['A'])[:T_S + 1]
    A0_cf = np.asarray(res.cf_macro['A'])[:T_S + 1]
    return A0_base, A0_cf

for shock, ret, rerun in (('tau_l', False, False), ('Ig', False, False),
                          ('tau_l', True, False), ('Ig', True, False),
                          ('Ig', False, True)):
    for backend in ('numpy', 'jax'):
        A0_base, A0_cf = run_backend(backend, shock, cohort_retirement=ret, rerun=rerun)
        diff = float(np.max(np.abs(A0_cf - A0_base)))
        status = "OK" if diff == 0.0 else "FAIL"
        label = shock + ('+ret' if ret else '') + ('+rerun' if rerun else '')
        print(f"{label:9s} {backend:6s}: A[{T_S}] base = {A0_base[-1]:.10f}, "
              f"cf = {A0_cf[-1]:.10f}, max |diff| through t_s = {diff:.3e}  {status}")
print("DONE")
