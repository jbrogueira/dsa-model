"""
Tests of UI eligibility at the start of a spell (docs/UI_ELIGIBILITY_PLAN.md):
a household that moves from employment into unemployment at a working age is
eligible with probability p; an ineligible one carries z_last = 0 and receives
no UI over the spell. Small model, both backends.
"""
import numpy as np
import pytest

from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
from test_fiscal_restructure import small_config, _budget_identity, _economy, T, J_R, N_SIM

P_ELIG = 0.33
UI, PENSION, EMPLOYED, RETIRED, ALIVE = 7, 16, 6, 17, 19


def _solved(Model, p):
    m = Model(small_config(ui_eligibility_prob=p), verbose=False)
    m.solve(verbose=False)
    return m


def _jax():
    from lifecycle_jax import LifecycleModelJAX
    return LifecycleModelJAX


def _recipient_and_first_year(m):
    """Per working age: mass of unemployed with z_last > 0 (UI recipients),
    and mass of all unemployed, from the exact distribution."""
    _, dists = m.exact_age_means(return_dist=True)
    d = np.asarray(dists)                       # (T_sim, n_alpha, n_a, n_y, n_h, n_y)
    unemp = d[:, :, :, 0, :, :]
    recip = unemp[..., 1:].sum(axis=(1, 2, 3, 4))
    return recip[:J_R], unemp.sum(axis=(1, 2, 3, 4))[:J_R]


class TestEligibility:
    def test_backends_agree(self):
        m_np = _solved(LifecycleModelPerfectForesight, P_ELIG)
        m_jx = _solved(_jax(), P_ELIG)
        assert np.array_equal(np.asarray(m_np.a_next_policy), np.asarray(m_jx.a_next_policy))
        assert np.abs(np.asarray(m_np.c_policy) - np.asarray(m_jx.c_policy)).max() < 1e-9
        assert np.abs(np.asarray(m_np.V) - np.asarray(m_jx.V)).max() < 1e-9
        e_np, e_jx = m_np.exact_age_means(), m_jx.exact_age_means()
        cols = [i for i in range(23) if i not in (2, 15)]
        np.testing.assert_allclose(e_np[:, cols], e_jx[:, cols], rtol=1e-9, atol=1e-12)

    def test_policies_change(self):
        m1 = _solved(LifecycleModelPerfectForesight, 1.0)
        mp = _solved(LifecycleModelPerfectForesight, P_ELIG)
        # The value of an employed household falls when UI is less likely.
        assert np.asarray(mp.V)[0, :, 1:].mean() < np.asarray(m1.V)[0, :, 1:].mean()

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_recipient_share_is_p_times_first_year_share(self, backend):
        Model = LifecycleModelPerfectForesight if backend == 'numpy' else _jax()
        r1, u1 = _recipient_and_first_year(_solved(Model, 1.0))
        rp, up = _recipient_and_first_year(_solved(Model, P_ELIG))
        # Income and survival are exogenous: the unemployed mass is unchanged.
        np.testing.assert_allclose(up, u1, rtol=1e-12)
        # At p = 1 every unemployed household in the first year of a spell
        # (z_last > 0) receives UI; at p only the eligible share does.
        np.testing.assert_allclose(rp, P_ELIG * r1, rtol=1e-10, atol=1e-15)
        assert r1[1:].min() > 0

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_ui_spending_scales_with_p_and_pensions_do_not_move(self, backend):
        Model = LifecycleModelPerfectForesight if backend == 'numpy' else _jax()
        e1 = _solved(Model, 1.0).exact_age_means()
        ep = _solved(Model, P_ELIG).exact_age_means()
        np.testing.assert_allclose(ep[:J_R, UI], P_ELIG * e1[:J_R, UI], rtol=1e-10, atol=1e-15)
        # The pension base of a household unemployed in its last working year
        # is its last income state, with no eligibility draw at retirement.
        np.testing.assert_allclose(ep[J_R:, PENSION], e1[J_R:, PENSION], rtol=1e-12)

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_monte_carlo_matches_exact(self, backend):
        Model = LifecycleModelPerfectForesight if backend == 'numpy' else _jax()
        m = _solved(Model, P_ELIG)
        n = 20000
        if backend == 'numpy':
            np.random.seed(11)
        sim = [np.asarray(x) for x in m.simulate(n_sim=n, seed=11)]
        work = slice(1, J_R)
        unemp = (~sim[EMPLOYED][work].astype(bool)) & sim[ALIVE][work].astype(bool)
        recip = unemp & (sim[UI][work] > 0)
        share_mc = recip.sum() / unemp.sum()
        r, u = _recipient_and_first_year(m)
        share_ex = r[work].sum() / u[work].sum()
        se = np.sqrt(share_ex * (1 - share_ex) / unemp.sum())
        assert abs(share_mc - share_ex) < 4 * se

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_budget_identity(self, backend):
        Model = LifecycleModelPerfectForesight if backend == 'numpy' else _jax()
        # A lump sum keeps every household's resources positive (without one a
        # household with no assets and no income is at the consumption bound).
        cfg = small_config(ui_eligibility_prob=P_ELIG, lump_sum_path=np.linspace(0.02, 0.05, T))
        resid, _ = _budget_identity(cfg, Model)
        assert resid < 1e-10

    def test_rejects_a_value_outside_the_unit_interval(self):
        with pytest.raises(ValueError):
            LifecycleModelPerfectForesight(small_config(ui_eligibility_prob=1.2), verbose=False)


class TestTransition:
    def test_backends_agree_in_a_transition(self):
        cfg = small_config(ui_eligibility_prob=P_ELIG)
        out = {}
        for backend in ('numpy', 'jax'):
            olg = _economy(cfg, backend=backend, aggregation='exact',
                           unemployment_index_path=np.array([1.0, 0.9, 0.8, 0.8]))
            res = olg.simulate_transition(np.full(4, 0.04), n_sim=N_SIM, verbose=False)
            bud = olg.compute_government_budget_path(n_sim=N_SIM, verbose=False)
            out[backend] = (np.asarray(res['Y']), np.asarray(res['K']), np.asarray(bud['ui']))
        for a, b in zip(out['numpy'], out['jax']):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)

    def test_cross_section_exact_threads_p(self):
        Model = _jax()
        m = _solved(Model, P_ELIG)
        surv = np.stack([np.asarray(m.survival_probs)] * 3)
        panel, mass = m.cross_section_exact(surv, rows=np.array([1, 2, 3]))
        e = m.exact_age_means()
        ui_cs = (panel[UI] * mass).sum(axis=1)
        np.testing.assert_allclose(ui_cs, e[1:4, UI], rtol=1e-10)
