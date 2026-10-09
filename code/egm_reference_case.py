"""
Reference arrays for the tests of the minimum income benefit, the savings
policy as a level and the endogenous grid method (docs/EGM_PLAN.md).

lifecycle_reference() and the transition runs of policy_reference_case use
only arguments that predate the plan, so they run on the code of 2580d6e too.
`python egm_reference_case.py OUT.npz` writes the arrays; the stored file
tests_data/egm_reference_2580d6e.npz was written by the code of 2580d6e. The
savings policy is stored as a level, a_grid[a_policy] on that code.
"""
import sys

import numpy as np

from lifecycle_perfect_foresight import LifecycleConfig, LifecycleModelPerfectForesight

T, J_R = 8, 5
EXACT_ROWS = (0, 3, 6)
PANEL_FIELDS = (0, 1, 7, 16, 18, 20, 22)   # a, c, ui, pension, hours, bequest, transfer
N_SIM, SEED = 300, 3


def lifecycle_config(**kw):
    base = dict(
        T=T, beta=0.96, gamma=1.0, n_a=25, n_y=3, n_h=1, n_alpha=2, a_max=15.0,
        retirement_age=J_R, education_type='medium', labor_supply=True, nu=5.0, phi=1.5,
        trend_growth=0.017, r_default=0.04, w_default=1.0,
        tau_c_default=0.18, tau_l_default=0.10, tau_p_default=0.20, tau_k_default=0.2,
        pension_replacement_default=0.3, pension_min_floor=0.08,
        ui_replacement_rate=0.3, ui_eligibility_prob=0.5, kappa=0.6, m_good=0.05,
        m_age_profile=np.linspace(0.5, 2.0, T), lump_sum_path=np.full(T, 0.03),
        job_finding_rate=0.44, max_job_separation_rate=0.1,
        survival_probs=np.linspace(0.999, 0.90, T).reshape(T, 1),
        initial_asset_distribution=np.array([0.0, 0.0, 0.5, 1.0, 2.5]),
        edu_params={'medium': {'mu_y': 0.0, 'sigma_y': 0.1, 'rho_y': 0.95,
                               'sigma_alpha': 0.3, 'unemployment_rate': 0.12}},
    )
    base.update(kw)
    return LifecycleConfig(**base)


def savings_level(m):
    """The savings policy as a level, (n_alpha, T, n_a, n_y, n_h, n_y)."""
    if getattr(m, 'a_next_policy_alpha', None) is not None:
        return np.asarray(m.a_next_policy_alpha, dtype=float)
    return np.asarray(m.a_grid)[np.asarray(m.a_policy_alpha)]


def _model(backend, **kw):
    cfg = lifecycle_config(**kw)
    if backend == 'jax':
        from lifecycle_jax import LifecycleModelJAX
        return LifecycleModelJAX(cfg, verbose=False)
    return LifecycleModelPerfectForesight(cfg, verbose=False)


def lifecycle_reference(backend, floor):
    """Policies, exact means, exact cross-sections and a simulated panel of
    the test household at transfer_floor = floor."""
    m = _model(backend, transfer_floor=floor)
    m.solve(verbose=False)
    tag = f'{backend}_f{int(round(floor * 100)):02d}'
    out = {f'{tag}_a_next': savings_level(m),
           f'{tag}_c': np.asarray(m.c_policy_alpha, float),
           f'{tag}_l': np.asarray(m.l_policy_alpha, float),
           f'{tag}_V': np.asarray(m.V_alpha, float),
           f'{tag}_means': np.asarray(m.exact_age_means(), float)}
    panel, mass = m.exact_panel(rows=np.array(EXACT_ROWS))
    for i in PANEL_FIELDS:
        out[f'{tag}_xpanel{i}'] = np.asarray(panel[i], float)
    out[f'{tag}_xmass'] = np.asarray(mass, float)
    sim = m.simulate(n_sim=N_SIM, seed=SEED)
    for i in PANEL_FIELDS:
        out[f'{tag}_sim{i}'] = np.asarray(sim[i], float)
    return out


def calibration_cross_sections():
    """The batched exact and simulated cross-sections of the JAX model."""
    m = _model('jax', transfer_floor=0.05)
    surv = np.stack([np.linspace(0.999, 0.90 - 0.01 * c, T).reshape(T, 1) for c in range(T)])
    out = {}
    panel, mass = m.cross_section_exact(surv)
    for i in PANEL_FIELDS:
        out[f'xs_exact{i}'] = np.asarray(panel[i], float)
    out['xs_exact_mass'] = np.asarray(mass, float)
    sim = m.cross_section_batched(surv, list(range(T)), 200)
    for i in PANEL_FIELDS:
        out[f'xs_sim{i}'] = np.asarray(sim[i], float)
    return out


def all_references():
    import policy_reference_case as prc
    out = {}
    for backend in ('numpy', 'jax'):
        for floor in (0.0, 0.05):
            out.update(lifecycle_reference(backend, floor))
    out.update(calibration_cross_sections())
    out.update({f'tr_{k}': v for k, v in prc.reference_runs('jax').items()})
    return out


if __name__ == '__main__':
    arrays = all_references()
    np.savez(sys.argv[1], **arrays)
    print(f"wrote {len(arrays)} arrays to {sys.argv[1]}")
