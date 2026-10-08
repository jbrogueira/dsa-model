"""
A small economy for the tests of the policy exercises (test_policy_exercises.py).

build_economy() and base_paths() use only arguments that predate the health
paths and the shock period, so reference_runs() runs on the code of e80b119
too. `python policy_reference_case.py OUT.npz` writes its arrays; the stored
file tests_data/policy_reference_e80b119.npz was written by the code of
e80b119 (POLICY_EXERCISES_PLAN.md section 4, test 11).
"""
import sys

import numpy as np

from lifecycle_perfect_foresight import LifecycleConfig
from olg_transition import OLGTransition

T, J_R, T_TR, N_POST = 8, 5, 10, 2
KAPPA0 = 0.6


def lifecycle_config(**kw):
    base = dict(
        T=T, beta=0.96, gamma=1.0, n_a=30, n_y=3, n_h=1, n_alpha=1, a_max=20.0,
        retirement_age=J_R, education_type='medium', labor_supply=True, nu=5.0, phi=1.5,
        trend_growth=0.017, r_default=0.04, w_default=1.0,
        tau_c_default=0.18, tau_l_default=0.10, tau_p_default=0.20, tau_k_default=0.2,
        pension_replacement_default=0.4, ui_replacement_rate=0.3, kappa=KAPPA0, m_good=0.05,
        m_age_profile=np.linspace(0.5, 2.0, T),
        job_finding_rate=0.44, max_job_separation_rate=0.1, transfer_floor=0.05,
        survival_probs=np.linspace(0.999, 0.90, T).reshape(T, 1),
        edu_params={'medium': {'mu_y': 0.0, 'sigma_y': 0.1, 'rho_y': 0.95,
                               'sigma_alpha': 0.0, 'unemployment_rate': 0.12}},
    )
    base.update(kw)
    return LifecycleConfig(**base)


def build_economy(backend='jax', split=True, **kw):
    """One education group; every cohort split between retirement at 4 and 5."""
    cohort_retirement = ({y: ((4, None, 0.6), (5, None, 0.4)) for y in range(1990, 2060)}
                         if split else None)
    return OLGTransition(
        lifecycle_config=lifecycle_config(), education_shares={'medium': 1.0},
        backend=backend, pop_growth=0.0, economy_type='soe', r_star=0.04,
        alpha=0.33, delta=0.05, A=1.0, eta_g=0.05, K_g_initial=0.5, delta_g=0.05,
        birth_year=2015, current_year=2023, cohort_retirement=cohort_retirement,
        aggregation='exact', output_dir='output/test', **kw)


def base_paths():
    return dict(
        r_path=np.full(T_TR, 0.04),
        tau_l_path=np.full(T_TR, 0.10), tau_c_path=np.full(T_TR, 0.18),
        tau_p_path=np.full(T_TR, 0.20), tau_k_path=np.full(T_TR, 0.20),
        pension_replacement_path=np.full(T_TR, 0.4),
        G_path=np.full(T_TR, 0.08),
        I_g_path=np.full(T_TR, 0.05 * 0.5 + 0.017 * 0.5),
        r_B_path=np.full(T_TR, 0.02),
    )


def reference_runs(backend='jax'):
    """Baseline, an I_g shock under debt financing and under a labour tax with
    a terminal debt target, all from t = 0."""
    from fiscal_experiments import FiscalScenario, run_baseline, run_fiscal_scenario
    olg = build_economy(backend)
    bp = run_baseline(olg, base_paths(), n_post=N_POST, n_sim=100)
    dI = np.full(T_TR, 0.02)
    out = {}
    for name, scn in (
            ('base', FiscalScenario(name='b', financing='debt', B_initial=0.3, n_post=N_POST)),
            ('debt', FiscalScenario(name='d', delta_I_g_path=dI, financing='debt',
                                    B_initial=0.3, n_post=N_POST)),
            ('taul', FiscalScenario(name='t', delta_I_g_path=dI, financing='tau_l',
                                    balance_condition='terminal_debt_gdp',
                                    target_debt_gdp=0.3, B_initial=0.3, n_post=N_POST))):
        res = run_fiscal_scenario(olg, scn, bp, n_sim=100, bisect_tol=1e-6)
        for k in ('Y', 'C', 'A', 'L', 'w'):
            out[f'{name}_{k}'] = np.asarray(res.cf_macro[k], dtype=float)
        for k in ('primary_deficit', 'gov_health', 'transfers', 'tax_l', 'pension'):
            out[f'{name}_{k}'] = np.asarray(res.cf_budget[k], dtype=float)
        out[f'{name}_B'] = np.asarray(res.B_path, dtype=float)
        out[f'{name}_Delta'] = np.array([res.adjustment_scalar])
    return out


def cross_section_reference():
    """The calibration's batched exact cross-section of the JAX model."""
    from lifecycle_jax import LifecycleModelJAX
    m = LifecycleModelJAX(lifecycle_config(), verbose=False)
    surv = np.stack([np.linspace(0.999, 0.90 - 0.01 * c, T).reshape(T, 1) for c in range(T)])
    panel, mass = m.cross_section_exact(surv)
    return {'xs_c': np.asarray(panel[1], float), 'xs_a': np.asarray(panel[0], float),
            'xs_gov_m': np.asarray(panel[10], float), 'xs_mass': np.asarray(mass, float)}


if __name__ == '__main__':
    arrays = reference_runs('jax')
    arrays.update(cross_section_reference())
    np.savez(sys.argv[1], **arrays)
    print(f"wrote {len(arrays)} arrays to {sys.argv[1]}")
