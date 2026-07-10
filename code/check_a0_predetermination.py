"""
Targeted check: A[0] predetermination under MIT stitching, with permanent-FE
heterogeneity active (n_alpha>1, sigma_alpha>0), on both backends.

A tau_l shock at t>=0 must leave the t=0 cross-section of household wealth
unchanged: pre-transition cohorts' policies for ages < -birth_period are
stitched from a pure-baseline solve, so simulated assets entering t=0 are
identical between baseline and counterfactual. Exercises the *_policy_alpha
stitching (both simulate paths read the per-alpha arrays).

The Ig case exercises the K_g→w channel: an I_g (level) shock with eta_g != 0
moves K_g and hence the wage path, so the MIT baseline model must be built
from pure-baseline wages. A[0] must still be exactly baseline.

Usage: python check_a0_predetermination.py
"""
import os, platform
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

from olg_transition import OLGTransition
from lifecycle_perfect_foresight import LifecycleConfig
from fiscal_experiments import FiscalScenario, run_fiscal_scenario

T_TR = 10
N_SIM = 50

def run_backend(backend, shock):
    cfg = LifecycleConfig(T=20, n_a=30, n_y=3, n_alpha=3, retirement_age=12)
    ep = dict(cfg.edu_params)
    ep['medium'] = dict(ep['medium'], sigma_alpha=0.3)
    cfg = cfg._replace(edu_params=ep)
    olg_kwargs = dict(lifecycle_config=cfg, backend=backend,
                      education_shares={'medium': 1.0})
    if shock == 'Ig':
        olg_kwargs.update(eta_g=0.05, K_g_initial=0.745, delta_g=0.05)
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
        bp['I_g_path'] = np.full(T_TR, 0.05 * 0.745)
        scen = FiscalScenario(
            name='Ig_shock',
            delta_I_g_path=np.full(T_TR, 0.02),
            financing='debt',
            balance_condition='terminal_debt_gdp',
            B_initial=0.0,
        )
    else:
        scen = FiscalScenario(
            name='tau_l_shock',
            delta_tau_l_path=np.full(T_TR, 0.05),
            financing='debt',
            balance_condition='terminal_debt_gdp',
            B_initial=0.0,
        )
    res = run_fiscal_scenario(olg, scen, bp, n_sim=N_SIM, verbose=False)
    A0_base = float(np.asarray(res.base_macro['A'])[0])
    A0_cf = float(np.asarray(res.cf_macro['A'])[0])
    return A0_base, A0_cf

for shock in ('tau_l', 'Ig'):
    for backend in ('numpy', 'jax'):
        A0_base, A0_cf = run_backend(backend, shock)
        diff = abs(A0_cf - A0_base)
        status = "OK" if diff == 0.0 else "FAIL"
        print(f"{shock:5s} {backend:6s}: A[0] base = {A0_base:.10f}, "
              f"cf = {A0_cf:.10f}, |diff| = {diff:.3e}  {status}")
print("DONE")
