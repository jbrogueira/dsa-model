"""
Tests of the endogenous grid method (docs/EGM_PLAN.md section 4): Euler and
hours conditions, concavity and monotonicity, convergence to a fine grid
search, continuity of aggregate assets in r, agreement of the backends, UI
eligibility and the MIT stitching under the method.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import egm_reference_case as erc
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight

HERE = os.path.dirname(os.path.abspath(__file__))
Y_MIN = 0.12


def _jax():
    from lifecycle_jax import LifecycleModelJAX
    return LifecycleModelJAX


def _model(backend):
    return LifecycleModelPerfectForesight if backend == 'numpy' else _jax()


def egm_config(**kw):
    base = dict(minimum_income=Y_MIN, savings_solver='egm')
    base.update(kw)
    return erc.lifecycle_config(**base)


def _solved(backend, **kw):
    m = _model(backend)(egm_config(**kw), verbose=False)
    m.solve(verbose=False)
    return m


def _uprime(c, gamma):
    return c ** (-gamma)


def _expected_marginal_value_loop(m, t, i_a_next, i_y, i_h, i_yl):
    """E_t [R_{t+1} u'(c_{t+1}) / (1+tau_c_{t+1})] at the node a' = a_grid[i_a_next],
    by explicit sums over next period's states (independent of the solver's
    vectorised operator)."""
    R1 = 1.0 + (1.0 - m.tau_k_path[t + 1]) * m.r_path[t + 1]
    c1 = m.c_policy[t + 1]
    dV = lambda iy, ih, iyl: R1 * _uprime(c1[i_a_next, iy, ih, iyl], m.gamma) / (1.0 + m.tau_c_path[t + 1])
    out = 0.0
    if t >= m.retirement_age:
        for ih2 in range(m.n_h):
            out += m.P_h[t, i_h, ih2] * dV(0, ih2, i_yl)
        return out
    p = m.ui_eligibility_prob
    draw = p < 1.0 and i_y > 0 and t + 1 < m.retirement_age
    for iy2 in range(m.n_y):
        for ih2 in range(m.n_h):
            prob = m._get_P_y(t, i_h, i_y, iy2) * m.P_h[t, i_h, ih2]
            v = dV(iy2, ih2, i_y)
            if draw and iy2 == 0:
                v = p * v + (1.0 - p) * dV(0, ih2, 0)
            out += prob * v
    return out


class TestOptimality:
    def test_euler_equation_at_the_endogenous_points(self):
        """At each endogenous point (a_e, a'_k) of an unconstrained state the
        consumption the method assigns satisfies the Euler equation with the
        expectation formed by explicit sums."""
        m = _solved('numpy', n_alpha=1)
        worst = 0.0
        for t in (1, 3, m.retirement_age + 1):
            R1 = 1.0 + (1.0 - m.tau_k_path[t + 1]) * m.r_path[t + 1]
            dV_next = R1 * _uprime(m.c_policy[t + 1], m.gamma) / (1.0 + m.tau_c_path[t + 1])
            a_e, c_e, l_e = m._egm_endogenous(t, dV_next)
            surv = m.survival_probs[t]
            for i_y in range(m.n_y):
                for i_yl in range(m.n_y):
                    for k in (2, 8, 15, 22):
                        Ev = _expected_marginal_value_loop(m, t, k, i_y, 0, i_yl)
                        lhs = _uprime(c_e[k, i_y, 0, i_yl], m.gamma) * (1 + m.trend_growth) \
                            / (1 + m.tau_c_path[t])
                        worst = max(worst, abs(1.0 - m.beta * surv[0] * Ev / lhs))
        assert worst < 1e-10, worst

    def test_euler_residuals_off_the_endogenous_points(self):
        """At the grid nodes with a' above the borrowing limit the residual
        carries the interpolation error of next period's marginal value
        between nodes. Reported (mean and maximum log10), not bounded."""
        m = _solved('numpy', n_alpha=1)
        res = []
        for t in range(1, m.T - 1):
            for i_a in range(m.n_a):
                for i_y in range(m.n_y):
                    for i_yl in range(m.n_y):
                        ap = m.a_next_policy[t, i_a, i_y, 0, i_yl]
                        if ap <= m.a_grid[0] + 1e-12 or ap >= m.a_grid[-1] - 1e-12:
                            continue
                        k = np.clip(np.searchsorted(m.a_grid, ap) - 1, 0, m.n_a - 2)
                        w = (m.a_grid[k + 1] - ap) / (m.a_grid[k + 1] - m.a_grid[k])
                        Ev = (w * _expected_marginal_value_loop(m, t, k, i_y, 0, i_yl)
                              + (1 - w) * _expected_marginal_value_loop(m, t, k + 1, i_y, 0, i_yl))
                        c = m.c_policy[t, i_a, i_y, 0, i_yl]
                        lhs = _uprime(c, m.gamma) * (1 + m.trend_growth) / (1 + m.tau_c_path[t])
                        res.append(abs(1.0 - m.beta * m.survival_probs[t, 0] * Ev / lhs))
        res = np.log10(np.maximum(np.array(res), 1e-17))
        print(f"\noff-node Euler residuals: mean log10 {res.mean():.2f}, max {res.max():.2f}")
        assert np.isfinite(res).all()

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_hours_condition(self, backend):
        m = _solved(backend)
        t_work = np.arange(m.retirement_age)
        c = np.asarray(m.c_policy_alpha)[:, t_work]
        l = np.asarray(m.l_policy_alpha)[:, t_work]
        y = np.asarray(m.y_grid)[None, None, None, :, None, None]
        mult = np.exp(np.asarray(m.alpha_grid))[:, None, None, None, None, None]
        tp, tl, tc = (np.asarray(p)[t_work][None, :, None, None, None, None]
                      for p in (m.tau_p_path, m.tau_l_path, m.tau_c_path))
        kap = np.asarray(m.wage_age_profile)[t_work][None, :, None, None, None, None]
        w = np.asarray(m.w_path)[t_work][None, :, None, None, None, None]
        mw = w * kap * y * mult * (1 - tp) * (1 - tl)
        emp = np.broadcast_to(y > 0, c.shape) & (l < 64.0 - 1e-9)
        lhs = m.nu * l ** m.phi
        rhs = mw * c ** (-m.gamma) / (1 + tc)
        rel = np.abs(lhs / np.broadcast_to(rhs, c.shape) - 1.0)[emp]
        assert emp.any() and rel.max() < 1e-10

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_value_concave_and_policies_monotone(self, backend):
        m = _solved(backend)
        a = np.asarray(m.a_grid)
        V = np.asarray(m.V_alpha)[:, :m.T - 1]
        slope = np.diff(V, axis=2) / np.diff(a)[None, None, :, None, None, None]
        assert np.diff(slope, axis=2).max() <= 1e-12
        ap = np.asarray(m.a_next_policy_alpha)
        c = np.asarray(m.c_policy_alpha)
        assert np.diff(ap, axis=2).min() >= -1e-12
        assert np.diff(c, axis=2).min() >= -1e-12


class TestBackends:
    def test_backends_agree(self):
        m_np, m_jx = _solved('numpy'), _solved('jax')
        for f in ('a_next_policy_alpha', 'c_policy_alpha', 'l_policy_alpha', 'V_alpha'):
            d = np.abs(np.asarray(getattr(m_np, f)) - np.asarray(getattr(m_jx, f))).max()
            assert d < 1e-10, (f, d)
        e_np, e_jx = m_np.exact_age_means(), m_jx.exact_age_means()
        cols = [i for i in range(23) if i not in (2, 15)]
        np.testing.assert_allclose(e_np[:, cols], e_jx[:, cols], rtol=1e-10, atol=1e-12)

    def test_policies_lie_off_the_nodes(self):
        m = _solved('jax')
        ap = np.asarray(m.a_next_policy_alpha)[:, :m.T - 1]
        on = np.isin(ap, np.asarray(m.a_grid))
        assert on.mean() < 0.5

    def test_refusals(self):
        for kw in (dict(transfer_floor=0.05, minimum_income=0.0), dict(tax_progressive=True),
                   dict(gamma=2.0), dict(savings_solver='vfi')):
            with pytest.raises((ValueError, NotImplementedError)):
                LifecycleModelPerfectForesight(egm_config(**kw), verbose=False)
        # resources at the borrowing limit must be positive
        with pytest.raises(ValueError):
            LifecycleModelPerfectForesight(
                egm_config(minimum_income=0.0, lump_sum_path=np.zeros(erc.T),
                           ui_replacement_rate=0.0, m_good=0.2), verbose=False)
        # gamma != 1 is accepted without trend growth
        LifecycleModelPerfectForesight(egm_config(gamma=2.0, trend_growth=0.0), verbose=False)


def _economy_n_y3(**kw):
    return erc.lifecycle_config(n_alpha=1, n_y=3, minimum_income=Y_MIN, **kw)


class TestConvergence:
    def test_converges_to_a_fine_grid_search(self):
        from lifecycle_jax import LifecycleModelJAX
        fine = LifecycleModelJAX(_economy_n_y3(n_a=2000, savings_solver='grid'), verbose=False)
        fine.solve(verbose=False)
        egm100 = LifecycleModelJAX(_economy_n_y3(n_a=100, savings_solver='egm'), verbose=False)
        egm100.solve(verbose=False)
        egm400 = LifecycleModelJAX(_economy_n_y3(n_a=400, savings_solver='egm'), verbose=False)
        egm400.solve(verbose=False)
        # policies at the EGM nodes, read off the fine grid at the nearest node
        a_f, a_e = np.asarray(fine.a_grid), np.asarray(egm100.a_grid)
        idx = np.abs(a_f[:, None] - a_e[None, :]).argmin(axis=0)
        ap_f = np.asarray(fine.a_next_policy)[:-1][:, idx]
        ap_e = np.asarray(egm100.a_next_policy)[:-1]
        spacing = np.diff(a_f).max()
        # a' of the fine search is within its own spacing of the EGM policy,
        # up to the fine grid's offset from the EGM node (<= one spacing)
        assert np.abs(ap_f - ap_e).max() <= 3 * spacing + 1e-12
        # exact age means: EGM-100 against the fine search within the
        # EGM-100 vs EGM-400 gap plus the fine grid's own error
        e_f, e_1, e_4 = (m.exact_age_means() for m in (fine, egm100, egm400))
        for col in (0, 1, 18):
            tol = 3 * np.abs(e_1[:, col] - e_4[:, col]).max() + 1e-4
            assert np.abs(e_1[:, col] - e_f[:, col]).max() < tol, col

    @pytest.mark.parametrize('solver', ['egm', 'grid'])
    def test_aggregate_assets_continuous_in_r(self, solver):
        from lifecycle_jax import LifecycleModelJAX
        rs = 0.04 + np.linspace(-1e-4, 1e-4, 21)
        A = []
        for r in rs:
            m = LifecycleModelJAX(_economy_n_y3(n_a=40, savings_solver=solver, r_default=r),
                                  verbose=False)
            m.solve(verbose=False)
            A.append(m.exact_age_means()[:, 0].sum())
        # Continuous: assets respond at every step and no step exceeds three
        # times the median step. Grid search gives a step function in r.
        d = np.abs(np.diff(A))
        smooth = d.min() > 0.0 and d.max() <= 3 * np.median(d)
        assert smooth == (solver == 'egm'), (solver, d)


class TestUIEligibility:
    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_recipient_share(self, backend):
        J_R = erc.J_R

        def recip(p):
            m = _solved(backend, ui_eligibility_prob=p)
            _, dists = m.exact_age_means(return_dist=True)
            unemp = np.asarray(dists)[:, :, :, 0, :, :]
            return unemp[..., 1:].sum(axis=(1, 2, 3, 4))[:J_R], unemp.sum(axis=(1, 2, 3, 4))[:J_R]
        r1, u1 = recip(1.0)
        rp, up = recip(0.4)
        np.testing.assert_allclose(up, u1, rtol=1e-12)
        np.testing.assert_allclose(rp, 0.4 * r1, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize('shock_period', [0, 3])
def test_mit_stitching_assets_predetermined(shock_period):
    cmd = [sys.executable, os.path.join(HERE, 'check_a0_predetermination.py'),
           '--shock-period', str(shock_period), '--savings-solver', 'egm']
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert 'DONE' in proc.stdout and 'FAIL' not in proc.stdout, proc.stdout
