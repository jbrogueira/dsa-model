"""
Tests of the fiscal restructure of 2026-10-07: the output tax in the firm
conditions, the lump-sum transfer in both household solvers, the education
and foreign-transfer lines, the sovereign-rate path and the unemployment path
(per-cohort income matrices), on a small model.
"""
import numpy as np
import pytest

from lifecycle_perfect_foresight import (LifecycleModelPerfectForesight, LifecycleConfig,
                                         income_transition_matrix, income_matrices_by_age,
                                         employed_transition_matrix)
from olg_transition import OLGTransition
from firm_conditions import firm_conditions, marginal_products

T, J_R, N_SIM = 8, 5, 400


def small_config(**kw):
    base = dict(
        T=T, beta=0.96, gamma=1.0, n_a=30, n_y=3, n_h=1, n_alpha=1, a_max=20.0,
        retirement_age=J_R, education_type='medium', labor_supply=True, nu=5.0, phi=1.5,
        trend_growth=0.017, r_default=0.04, w_default=1.0,
        tau_c_default=0.18, tau_l_default=0.10, tau_p_default=0.20, tau_k_default=0.2,
        pension_replacement_default=0.4, ui_replacement_rate=0.1, kappa=0.6, m_good=0.03,
        job_finding_rate=0.44, max_job_separation_rate=0.1,
        survival_probs=np.linspace(0.999, 0.93, T).reshape(T, 1),
        edu_params={'medium': {'mu_y': 0.0, 'sigma_y': 0.1, 'rho_y': 0.95,
                               'sigma_alpha': 0.0, 'unemployment_rate': 0.12}},
    )
    base.update(kw)
    return LifecycleConfig(**base)


def _budget_identity(cfg, Model):
    """(1+tau_c) c + (1+g) a' = a + r a - tax_k + after-tax income - oop + floor
    top-up + lump sum, for every living household of a simulated panel."""
    mdl = Model(cfg, verbose=False)
    mdl.solve(verbose=False)
    out = [np.asarray(x) for x in mdl.simulate(n_sim=N_SIM, seed=3)]
    a, c, eff, oop, tl, tp, tk, pens, ret, alive, tr = (
        out[0], out[1], out[5], out[9], out[12], out[13], out[14], out[16],
        out[17].astype(bool), out[19].astype(bool), out[22])
    a_next = np.zeros_like(a); a_next[:-1] = a[1:]
    alive_next = np.zeros_like(alive); alive_next[:-1] = alive[1:]
    after_tax = np.where(ret, pens - tl, eff - tp - tl)
    ls = np.asarray(cfg.lump_sum_path)[:, None]
    resid = ((1 + cfg.tau_c_default) * c + (1 + cfg.trend_growth) * a_next
             - (a + cfg.r_default * a - tk + after_tax - oop + tr + ls))
    return float(np.abs(resid[alive & alive_next]).max()), mdl


class TestLumpSum:
    def test_budget_identity_both_backends(self):
        from lifecycle_jax import LifecycleModelJAX
        cfg = small_config(lump_sum_path=np.linspace(0.02, 0.05, T))
        resid_np, m_np = _budget_identity(cfg, LifecycleModelPerfectForesight)
        resid_jx, m_jx = _budget_identity(cfg, LifecycleModelJAX)
        assert resid_np < 1e-10 and resid_jx < 1e-10
        assert np.array_equal(np.asarray(m_np.a_next_policy), np.asarray(m_jx.a_next_policy))
        assert np.abs(np.asarray(m_np.c_policy) - np.asarray(m_jx.c_policy)).max() < 1e-9

    def test_lump_sum_raises_consumption(self):
        cfg0 = small_config()
        cfg1 = small_config(lump_sum_path=np.full(T, 0.05))
        m0 = LifecycleModelPerfectForesight(cfg0, verbose=False); m0.solve(verbose=False)
        m1 = LifecycleModelPerfectForesight(cfg1, verbose=False); m1.solve(verbose=False)
        # On a discrete asset grid the extra resources can move the saving
        # node up and consumption down at single states; the mean rises.
        assert np.asarray(m1.c_policy).mean() > np.asarray(m0.c_policy).mean()
        assert np.asarray(m1.V).mean() > np.asarray(m0.V).mean()

    def test_exact_means_carry_the_lump_sum(self):
        from lifecycle_jax import LifecycleModelJAX
        cfg = small_config(lump_sum_path=np.full(T, 0.04))
        m_np = LifecycleModelPerfectForesight(cfg, verbose=False); m_np.solve(verbose=False)
        m_jx = LifecycleModelJAX(cfg, verbose=False); m_jx.solve(verbose=False)
        e_np, e_jx = m_np.exact_age_means(), m_jx.exact_age_means()
        # Column 2 (y of the retired) differs by convention between the
        # backends (the NumPy panel records 0, the JAX panel the last state).
        mask = ~np.isnan(e_np)
        mask[:, 2] = False
        assert np.abs(e_np[mask] - e_jx[mask]).max() < 1e-10
        # column 22 is the floor top-up; the lump sum is in the budget, so
        # consumption (column 1) is higher than without it
        m0 = LifecycleModelPerfectForesight(small_config(), verbose=False); m0.solve(verbose=False)
        assert e_np[:, 1].mean() > m0.exact_age_means()[:, 1].mean()


class TestFirmConditions:
    def test_identities_at_a_positive_output_tax(self):
        tau = 0.05
        K_L, w, Y_L = firm_conditions(0.04, 1.3, 1.0, 0.33, 0.05, tau)
        assert abs((1 - tau) * 0.33 * Y_L / K_L - 0.05 - 0.04) < 1e-12
        assert abs(w - (1 - tau) * 0.67 * Y_L) < 1e-12
        assert abs(tau * Y_L + w + (0.04 + 0.05) * K_L - Y_L) < 1e-12
        r_b, w_b = marginal_products(K_L, 1.0, 0.33, 0.05, 1.3, tau_y=tau)
        assert abs(r_b - 0.04) < 1e-12 and abs(w_b - w) < 1e-12

    def test_zero_tax_reproduces_the_untaxed_conditions(self):
        K_L, w, _ = firm_conditions(0.04, 1.3, 1.0, 0.33, 0.05, 0.0)
        assert abs(K_L - ((0.04 + 0.05) / (0.33 * 1.3)) ** (1 / (0.33 - 1))) < 1e-12
        assert abs(w - 0.67 * 1.3 * K_L ** 0.33) < 1e-12


def _economy(cfg, **kw):
    return OLGTransition(lifecycle_config=cfg, education_shares={'medium': 1.0},
                         backend=kw.pop('backend', 'numpy'), pop_growth=0.0, economy_type='soe',
                         r_star=0.04, alpha=0.33, delta=0.05, A=1.0, **kw)


class TestTransitionBudget:
    def test_output_tax_and_new_lines_in_the_budget(self):
        cfg = small_config()
        T_tr = 4
        tau_y = np.array([0.03, 0.04, 0.05, 0.05])
        olg = _economy(cfg, tau_y=tau_y, lump_sum_path=np.full(T_tr, 0.02),
                       education_over_Y0=0.03, education_index_path=np.array([1.0, 0.98, 0.96, 0.95]),
                       foreign_transfer_over_Y=np.array([0.02, 0.02, 0.01, 0.01]),
                       r_B_path=np.array([-0.01, 0.0, 0.01, 0.02]),
                       B_path=np.full(T_tr + 1, 0.5))
        res = olg.simulate_transition(np.full(T_tr, 0.04), n_sim=N_SIM, verbose=False)
        bud = olg.compute_government_budget_path(n_sim=N_SIM, verbose=False)
        Y, L, K, w = (np.asarray(res[k]) for k in ('Y', 'L', 'K_domestic', 'w'))
        # firm conditions with the tax
        np.testing.assert_allclose(K / Y, (1 - tau_y) * 0.33 / 0.09, rtol=1e-10)
        np.testing.assert_allclose(w * L, (1 - tau_y) * 0.67 * Y, rtol=1e-10)
        np.testing.assert_allclose(tau_y * Y + w * L + 0.09 * K, Y, rtol=1e-10)
        # the lines
        np.testing.assert_allclose(bud['tax_y'], tau_y * Y, rtol=1e-12)
        np.testing.assert_allclose(bud['foreign_transfer'], np.array([0.02, 0.02, 0.01, 0.01]) * Y, rtol=1e-12)
        np.testing.assert_allclose(bud['lump_sum'], 0.02, rtol=1e-12)
        np.testing.assert_allclose(bud['education'],
                                   0.03 * Y[0] * (w / w[0]) * np.array([1.0, 0.98, 0.96, 0.95]), rtol=1e-12)
        spend = sum(np.asarray(bud[k]) for k in ('ui', 'pension', 'gov_health', 'transfers',
                                                  'govt_spending', 'public_investment',
                                                  'defense_spending', 'other_net_spending',
                                                  'education', 'lump_sum'))
        rev = sum(np.asarray(bud[k]) for k in ('tax_c', 'tax_l', 'tax_p', 'tax_k', 'bequest_tax',
                                                'tax_y', 'foreign_transfer'))
        np.testing.assert_allclose(spend, bud['total_spending'], atol=1e-12)
        np.testing.assert_allclose(rev, bud['total_revenue'], atol=1e-12)
        # the rate path reaches the debt-service line
        np.testing.assert_allclose(bud['debt_service'], np.array([-0.01, 0.0, 0.01, 0.02]) * 0.5, atol=1e-12)

    def test_defaults_leave_the_lines_at_zero(self):
        cfg = small_config()
        olg = _economy(cfg)
        olg.simulate_transition(np.full(3, 0.04), n_sim=N_SIM, verbose=False)
        bud = olg.compute_government_budget_path(n_sim=N_SIM, verbose=False)
        for k in ('tax_y', 'foreign_transfer', 'education', 'lump_sum'):
            assert np.all(np.asarray(bud[k]) == 0.0)


class TestUnemploymentPath:
    def test_matrix_builder_has_the_stationary_rate(self):
        cfg = small_config()
        P_e = employed_transition_matrix(cfg, 'medium')
        for u in (0.12, 0.08, 0.055):
            P = income_transition_matrix(P_e, cfg.n_y, u, 0.44, 0.1)
            w, v = np.linalg.eig(P.T)
            st = v[:, np.argmax(w.real)].real
            st = st / st.sum()
            assert abs(st[0] - u) < 1e-10
        rates = np.linspace(0.12, 0.06, T)
        P4 = income_matrices_by_age(P_e, cfg.n_y, 1, rates, 0.44, 0.1)
        assert P4.shape == (T, 1, cfg.n_y, cfg.n_y)
        np.testing.assert_allclose(P4[-1, 0], income_transition_matrix(P_e, cfg.n_y, rates[-1], 0.44, 0.1))

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_cohorts_face_the_index_along_their_diagonal(self, backend):
        cfg = small_config()
        T_tr = 5
        index = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        olg = _economy(cfg, backend=backend, unemployment_index_path=index)
        olg.simulate_transition(np.full(T_tr, 0.04), n_sim=N_SIM, verbose=False)
        sols = olg.birth_cohort_solutions['medium']
        # a cohort born at t = 2 faces index[2 + age] (held at the last value)
        m = sols[2]
        P = np.asarray(m.P_y)
        assert P.ndim == 4
        P_e = employed_transition_matrix(cfg, 'medium')
        for age in range(T):
            u = 0.12 * index[min(2 + age, T_tr - 1)]
            np.testing.assert_allclose(P[age, 0], income_transition_matrix(P_e, cfg.n_y, u, 0.44, 0.1),
                                       atol=1e-12)
        # a cohort born before the transition faces the base rate until t = 0
        m_old = sols[-3]
        P_old = np.asarray(m_old.P_y)
        np.testing.assert_allclose(P_old[0, 0], income_transition_matrix(P_e, cfg.n_y, 0.12, 0.44, 0.1), atol=1e-12)
        np.testing.assert_allclose(P_old[3, 0], income_transition_matrix(P_e, cfg.n_y, 0.12 * index[0], 0.44, 0.1), atol=1e-12)
        np.testing.assert_allclose(P_old[4, 0], income_transition_matrix(P_e, cfg.n_y, 0.12 * index[1], 0.44, 0.1), atol=1e-12)
        assert m.config.edu_params['medium']['unemployment_rate'] == pytest.approx(0.12 * index[2])

    def test_backends_agree_under_the_path(self):
        cfg = small_config()
        T_tr = 4
        index = np.array([1.0, 0.85, 0.7, 0.6])
        out = {}
        for backend in ('numpy', 'jax'):
            olg = _economy(cfg, backend=backend, unemployment_index_path=index,
                           lump_sum_path=np.full(T_tr, 0.02), aggregation='exact')
            res = olg.simulate_transition(np.full(T_tr, 0.04), n_sim=N_SIM, verbose=False)
            bud = olg.compute_government_budget_path(n_sim=N_SIM, verbose=False)
            out[backend] = (np.asarray(res['Y']), np.asarray(res['L']), np.asarray(bud['ui']))
        for a, b in zip(out['numpy'], out['jax']):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)
        # a falling unemployment rate lowers UI relative to a flat one
        olg0 = _economy(cfg, aggregation='exact')
        olg0.simulate_transition(np.full(T_tr, 0.04), n_sim=N_SIM, verbose=False)
        ui0 = np.asarray(olg0.compute_government_budget_path(n_sim=N_SIM, verbose=False)['ui'])
        assert out['numpy'][2][-1] < ui0[-1]


class TestPriceConsistency:
    def test_calibration_wage_equals_transition_wage_at_the_output_tax(self):
        """The SMM's wage (compute_equilibrium_prices) and the transition's
        w_path[0] come from the same firm conditions at the configured tau_y."""
        import json, os
        from calibrate import compute_equilibrium_prices
        here = os.path.dirname(os.path.abspath(__file__))
        raw = json.load(open(os.path.join(here, 'calibration_input_GR.json')))
        raw['fiscal']['tau_y'] = 0.07
        eq = compute_equilibrium_prices(raw)
        prod = raw['production']
        K_g_factor = prod['K_g'] ** prod['eta_g']
        K_L, w, Y_L = firm_conditions(raw['prices']['r'], prod['A_tfp'], K_g_factor,
                                      prod['alpha'], prod['delta'], 0.07)
        assert abs(eq['w'] - w) < 1e-12 and abs(eq['K_over_L'] - K_L) < 1e-12
        # untaxed: the wage is higher by the factor (1 - tau)^(1/(1-alpha))
        raw['fiscal']['tau_y'] = 0.0
        eq0 = compute_equilibrium_prices(raw)
        assert abs(eq['w'] / eq0['w'] - (1 - 0.07) ** (1 / (1 - prod['alpha']))) < 1e-12


class TestDebtMatchingRate:
    """solve_baseline with match_debt_year: a constant rate over 2026-60 puts
    the debt ratio of 2060 at the projection's, on a toy budget whose balance
    is linear in the rate."""

    def test_rate_hits_the_projection_in_2060(self):
        import json
        import os
        from baseline_closure import solve_baseline
        raw = json.load(open(os.path.join(os.path.dirname(__file__), 'calibration_input_GR.json')))
        T_tr, base = 180, 2023
        G = np.full(T_tr, 1.017)
        r_B = np.full(T_tr, 0.02)

        def run(lump, tau):
            Y = np.ones(T_tr)
            budget = {'total_revenue': 0.3665 + 0.7 * np.asarray(tau), 'total_spending': np.full(T_tr, 0.386)}
            return Y, budget

        fx = solve_baseline(run, raw, T_tr, base, 0.035, 0.0595, r_B, G, verbose=False,
                            match_debt_year=2060, first_mid_year=2026, max_iter=20)
        d = fx['debt']
        t60 = 2060 - base
        assert abs(d['debt'][t60] - d['debt_projection'][t60]) < 2e-3
        assert fx['tau_mid'] < 0.0595                      # a lower rate raises the ratio
        tau = fx['tau_y_path']
        assert np.allclose(tau[:2026 - base], 0.0595)       # 2023-25 at the pinned rate
        assert np.allclose(tau[2026 - base:2061 - base], fx['tau_mid'])
        # the terminal window still holds
        assert abs(d['window_residual']) < 1e-3 * d['window_years'] + 1e-6

    def test_pinned_rate_throughout_keeps_the_rate(self):
        import json
        import os
        from baseline_closure import solve_baseline
        raw = json.load(open(os.path.join(os.path.dirname(__file__), 'calibration_input_GR.json')))
        T_tr, base = 180, 2023

        def run(lump, tau):
            return np.ones(T_tr), {'total_revenue': 0.3665 + 0.7 * np.asarray(tau), 'total_spending': np.full(T_tr, 0.386)}

        fx = solve_baseline(run, raw, T_tr, base, 0.035, 0.0595, np.full(T_tr, 0.02), np.full(T_tr, 1.017),
                            verbose=False, terminal_rule=False, max_iter=6)
        assert np.allclose(fx['tau_y_path'], 0.0595)
        assert fx['tau_terminal'] == 0.0595
        assert np.isfinite(fx['window_residual'])          # reported, not solved

    def test_debt_match_without_terminal_rule_holds_the_2026_rate_after_2060(self):
        import json
        import os
        from baseline_closure import solve_baseline
        raw = json.load(open(os.path.join(os.path.dirname(__file__), 'calibration_input_GR.json')))
        T_tr, base = 180, 2023

        def run(lump, tau):
            return np.ones(T_tr), {'total_revenue': 0.3665 + 0.7 * np.asarray(tau), 'total_spending': np.full(T_tr, 0.386)}

        fx = solve_baseline(run, raw, T_tr, base, 0.035, 0.0595, np.full(T_tr, 0.02), np.full(T_tr, 1.017),
                            verbose=False, match_debt_year=2060, first_mid_year=2026, terminal_rule=False, max_iter=20)
        tau = fx['tau_y_path']
        assert np.allclose(tau[:2026 - base], 0.0595)
        assert np.allclose(tau[2026 - base:], fx['tau_mid'])          # one step, constant after
        d = fx['debt']
        assert abs(d['debt'][2060 - base] - d['debt_projection'][2060 - base]) < 2e-3


class TestOutputTaxRules:
    """The three output-tax rules of closure_options on the toy budget: a
    linear ramp to the debt-matching rate of 2060, the base-year rate
    throughout, and one rate from the base year."""

    @staticmethod
    def _setup():
        import json
        import os
        raw = json.load(open(os.path.join(os.path.dirname(__file__), 'calibration_input_GR.json')))
        T_tr = 180

        def run(lump, tau):
            return np.ones(T_tr), {'total_revenue': 0.3665 + 0.7 * np.asarray(tau),
                                   'total_spending': np.full(T_tr, 0.386)}
        return raw, T_tr, run

    def test_ramp_is_linear_to_the_debt_matching_rate(self):
        from baseline_closure import solve_baseline, closure_options
        raw, T_tr, run = self._setup()
        base = 2023
        opts = closure_options({'tau_y_mode': 'debt_ramp', 'tau_y_debt_year': 2060,
                                'tau_y_terminal_rule': False})
        fx = solve_baseline(run, raw, T_tr, base, 0.035, 0.0595, np.full(T_tr, 0.02),
                            np.full(T_tr, 1.017), verbose=False, max_iter=25, **opts)
        tau, t60 = fx['tau_y_path'], 2060 - base
        assert abs(tau[0] - 0.0595) < 1e-12                       # the base-year pin
        assert np.allclose(np.diff(tau[:t60 + 1]), (fx['tau_mid'] - 0.0595) / t60)  # linear, no jump
        assert np.allclose(tau[t60:], fx['tau_mid'])
        d = fx['debt']
        assert abs(d['debt'][t60] - d['debt_projection'][t60]) < 2e-3

    def test_one_rate_from_the_base_year(self):
        from baseline_closure import solve_baseline, closure_options
        raw, T_tr, run = self._setup()
        opts = closure_options({'tau_y_mode': 'debt', 'tau_y_debt_year': 2060,
                                'tau_y_first_year': 2023, 'tau_y_terminal_rule': False})
        fx = solve_baseline(run, raw, T_tr, 2023, 0.035, 0.0595, np.full(T_tr, 0.02),
                            np.full(T_tr, 1.017), verbose=False, max_iter=25, **opts)
        assert np.allclose(fx['tau_y_path'], fx['tau_mid'])
        d = fx['debt']
        assert abs(d['debt'][37] - d['debt_projection'][37]) < 2e-3

    def test_pinned_throughout(self):
        from baseline_closure import closure_options
        o = closure_options({'tau_y_mode': 'pinned_throughout'})
        assert o['match_debt_year'] is None and o['terminal_rule'] is False

    def test_constant_foreign_transfer_overrides_the_file(self):
        from calibrate import foreign_transfer_by_year
        raw, _, _ = self._setup()
        raw['fiscal']['foreign_transfer_over_Y'] = 0.010
        assert np.allclose(foreign_transfer_by_year(raw, [2023, 2025, 2040]), 0.010)
        raw['fiscal'].pop('foreign_transfer_over_Y')
        assert abs(foreign_transfer_by_year(raw, [2025])[0] - 0.027) < 1e-9
