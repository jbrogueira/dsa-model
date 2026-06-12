"""
Targeted check: A[0] predetermination under MIT stitching, with permanent-FE
heterogeneity active (n_alpha>1, sigma_alpha>0), on both backends.

A tau_l shock at t>=0 must leave the t=0 cross-section of household wealth
unchanged: pre-transition cohorts' policies for ages < -birth_period are
stitched from a pure-baseline solve, so simulated assets entering t=0 are
identical between baseline and counterfactual. Exercises the *_policy_alpha
stitching (both simulate paths read the per-alpha arrays).

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

def run_backend(backend):
    cfg = LifecycleConfig(T=20, n_a=30, n_y=3, n_alpha=3, retirement_age=12)
    ep = dict(cfg.edu_params)
    ep['medium'] = dict(ep['medium'], sigma_alpha=0.3)
    cfg = cfg._replace(edu_params=ep)
    olg = OLGTransition(lifecycle_config=cfg, backend=backend,
                        education_shares={'medium': 1.0})
    bp = dict(
        r_path=np.full(T_TR, 0.04),
        tau_l_path=np.full(T_TR, 0.15),
        tau_c_path=np.full(T_TR, 0.0),
        tau_p_path=np.full(T_TR, 0.0),
        tau_k_path=np.full(T_TR, 0.0),
        pension_replacement_path=np.full(T_TR, 0.4),
    )
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

for backend in ('numpy', 'jax'):
    A0_base, A0_cf = run_backend(backend)
    diff = abs(A0_cf - A0_base)
    status = "OK" if diff == 0.0 else "FAIL"
    print(f"{backend:6s}: A[0] base = {A0_base:.10f}, cf = {A0_cf:.10f}, "
          f"|diff| = {diff:.3e}  {status}")
print("DONE")
