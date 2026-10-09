"""
Tests of the minimum income benefit (docs/EGM_PLAN.md section 2): the
unemployed of working age and retirees receive max(0, y_min - y), y the lump
sum plus after-tax UI or pension less out-of-pocket medical spending; the
employed receive nothing. Small model, both backends.
"""
import os

import numpy as np
import pytest

import egm_reference_case as erc
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
from test_fiscal_restructure import _budget_identity, _economy, N_SIM

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'tests_data', 'egm_reference_2580d6e.npz')
Y_MIN = 0.12
OOP, UI, TAX_L, PENSION, TRANSFER = 9, 7, 12, 16, 22


def _jax():
    from lifecycle_jax import LifecycleModelJAX
    return LifecycleModelJAX


def _model(backend):
    return LifecycleModelPerfectForesight if backend == 'numpy' else _jax()


def _solved(backend, **kw):
    m = _model(backend)(erc.lifecycle_config(minimum_income=Y_MIN, **kw), verbose=False)
    m.solve(verbose=False)
    return m


def _income_and_benefit(m, rows):
    """At each state of the exact cross-sections at ages *rows*: the income
    the benefit tests, the benefit, and the masks of the retired, the
    unemployed of working age and the employed, each restricted to states with
    positive mass (the NumPy cross-section records the benefit only there).
    Shapes (R, n_states)."""
    panel, mass = m.exact_panel(rows=np.asarray(rows))
    occupied = np.asarray(mass) > 0
    shape = (int(m.n_alpha), int(m.n_a), int(m.n_y), int(m.n_h), int(m.n_y))
    Y = np.broadcast_to(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij')[2].reshape(-1),
                        panel[0].shape)
    ages = np.asarray(rows)[:, None] + int(m.current_age)
    retired = np.broadcast_to(ages >= int(m.retirement_age), panel[0].shape)
    unemployed = (~retired) & (Y == 0) & occupied
    employed = (~retired) & (Y > 0) & occupied
    retired = retired & occupied
    lump = np.asarray(m.config.lump_sum_path)[ages]
    after_tax = np.where(retired, panel[PENSION], panel[UI]) - panel[TAX_L]
    income = lump + after_tax - panel[OOP]
    return income, np.asarray(panel[TRANSFER]), retired, unemployed, employed


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
@pytest.mark.parametrize('floor', [0.0, 0.05])
def test_zero_benefit_reproduces_2580d6e(backend, floor):
    """With minimum_income = 0 the code reproduces 2580d6e bit for bit."""
    ref = np.load(REF)
    out = erc.lifecycle_reference(backend, floor)
    for k, v in out.items():
        assert np.array_equal(v, ref[k], equal_nan=True), k


def test_backends_agree():
    m_np, m_jx = _solved('numpy'), _solved('jax')
    assert np.array_equal(np.asarray(m_np.a_next_policy), np.asarray(m_jx.a_next_policy))
    assert np.abs(np.asarray(m_np.c_policy_alpha) - np.asarray(m_jx.c_policy_alpha)).max() < 1e-9
    assert np.abs(np.asarray(m_np.V_alpha) - np.asarray(m_jx.V_alpha)).max() < 1e-9
    e_np, e_jx = m_np.exact_age_means(), m_jx.exact_age_means()
    cols = [i for i in range(23) if i not in (2, 15)]
    np.testing.assert_allclose(e_np[:, cols], e_jx[:, cols], rtol=1e-9, atol=1e-12)
    assert e_np[:, TRANSFER].min() > 0.0


def test_benefit_depends_on_the_discrete_state_only():
    """b is equal across the asset nodes at every (age, alpha, z, h, z_last)."""
    m = _solved('jax')
    rows = np.arange(int(m.T))
    _, b, _, _, _ = _income_and_benefit(m, rows)
    b = b.reshape(len(rows), int(m.n_alpha), int(m.n_a), int(m.n_y), int(m.n_h), int(m.n_y))
    assert np.abs(b - b[:, :, :1]).max() == 0.0
    # the same in the NumPy budget function, which the solve calls
    n = _solved('numpy')
    for t in (1, int(n.retirement_age) + 1):
        retired = t >= n.retirement_age
        for i_y in range(n.n_y):
            for i_yl in range(n.n_y):
                tr = [n._compute_budget(retired, t, n.r_path[t], n.w_path[t], n.tau_l_path[t],
                                        n.tau_p_path[t], n.tau_k_path[t], a, n.y_grid[i_y],
                                        n.h_grid[0], i_y, i_yl, 0)[4] for a in n.a_grid]
                assert np.ptp(tr) == 0.0


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_income_reaches_the_guaranteed_level(backend):
    m = _solved(backend)
    rows = np.arange(int(m.T))
    income, b, retired, unemployed, employed = _income_and_benefit(m, rows)
    covered = retired | unemployed
    assert (income + b)[covered].min() >= Y_MIN - 1e-12
    short = covered & (income < Y_MIN)
    np.testing.assert_allclose((income + b)[short], Y_MIN, rtol=0, atol=1e-12)
    assert short.any() and (covered & ~short).any()
    assert np.all(b[covered & ~short] == 0.0)
    assert np.all(b[employed] == 0.0)


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_budget_identity(backend):
    cfg = erc.lifecycle_config(minimum_income=Y_MIN)
    resid, m = _budget_identity(cfg, _model(backend))
    assert resid < 1e-10


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_no_benefit_at_zero_level(backend):
    """The benefit is gated on y_min > 0: at y_min = 0 the formula would pay
    max(0, -y) where medical spending exceeds the lump sum."""
    cfg = erc.lifecycle_config(lump_sum_path=np.zeros(erc.T), ui_replacement_rate=0.0,
                               m_good=0.2)
    m = _model(backend)(cfg, verbose=False)
    m.solve(verbose=False)
    panel, mass = m.exact_panel()
    assert np.all(np.asarray(panel[TRANSFER]) == 0.0)
    assert (np.asarray(panel[OOP])[np.asarray(mass) > 0] > 0).any()


def test_refusals():
    with pytest.raises(ValueError):
        LifecycleModelPerfectForesight(erc.lifecycle_config(minimum_income=0.1, transfer_floor=0.05),
                                       verbose=False)
    with pytest.raises(ValueError):
        LifecycleModelPerfectForesight(erc.lifecycle_config(minimum_income=-0.1), verbose=False)


def test_transition_books_the_benefit_and_backends_agree():
    cfg = erc.lifecycle_config(minimum_income=Y_MIN, n_alpha=1)
    out = {}
    for backend in ('numpy', 'jax'):
        olg = _economy(cfg, backend=backend, aggregation='exact')
        res = olg.simulate_transition(np.full(4, 0.04), n_sim=N_SIM, verbose=False)
        bud = olg.compute_government_budget_path(n_sim=N_SIM, verbose=False)
        out[backend] = (np.asarray(res['Y']), np.asarray(res['K']), np.asarray(bud['transfers']),
                        np.asarray(bud['total_spending']))
    assert out['jax'][2].min() > 0.0
    for a, b in zip(out['numpy'], out['jax']):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)
