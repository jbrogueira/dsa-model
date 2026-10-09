"""
Tests of the savings policy as a level and the two-node lottery
(docs/EGM_PLAN.md section 3). A level a' between grid nodes a_k <= a' <=
a_{k+1} puts weight omega = (a_{k+1} - a')/(a_{k+1} - a_k) on a_k and 1 - omega
on a_{k+1}; a grid-search policy sits on a node and reproduces the code of
2580d6e bit for bit.
"""
import os

import numpy as np
import pytest

import egm_reference_case as erc
import policy_reference_case as prc
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight, lottery_np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'tests_data', 'egm_reference_2580d6e.npz')


def _jax():
    from lifecycle_jax import LifecycleModelJAX
    return LifecycleModelJAX


def _lottery(backend):
    if backend == 'numpy':
        return lottery_np
    from lifecycle_jax import lottery_jax
    return lambda grid, x: tuple(np.asarray(v) for v in lottery_jax(grid, x))


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_lottery_on_and_between_nodes(backend):
    lot = _lottery(backend)
    grid = erc.lifecycle_config().a_min + np.linspace(0, 1, 25) ** 1.5 * 15.0
    n = len(grid)
    k, w = lot(grid, grid)
    # on node n: k = n - 1 with omega = 0 (all mass on the upper node), and
    # k = 0 with omega = 1 at the first node
    assert k[0] == 0 and w[0] == 1.0
    np.testing.assert_array_equal(k[1:], np.arange(n - 1))
    np.testing.assert_array_equal(w[1:], 0.0)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, grid[-1] + 1.0, 500)
    k, w = lot(grid, x)
    xc = np.clip(x, grid[0], grid[-1])
    assert k.min() >= 0 and k.max() <= n - 2
    assert np.all((w >= 0.0) & (w <= 1.0))
    np.testing.assert_allclose(w * grid[k] + (1 - w) * grid[k + 1], xc, rtol=0, atol=1e-13)


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_lottery_preserves_mean_assets(backend):
    lot = _lottery(backend)
    grid = np.linspace(0, 1, 40) ** 1.5 * 20.0
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 20.0, 2000)
    mu = rng.dirichlet(np.ones(2000))
    k, w = lot(grid, x)
    moved = np.zeros(len(grid))
    np.add.at(moved, k, w * mu)
    np.add.at(moved, k + 1, (1 - w) * mu)
    assert abs(moved @ grid - mu @ x) < 1e-14
    assert abs(moved.sum() - 1.0) < 1e-14


def test_grid_search_reproduces_2580d6e_cross_sections():
    ref = np.load(REF)
    out = erc.calibration_cross_sections()
    for k, v in out.items():
        assert np.array_equal(v, ref[k], equal_nan=True), k


def test_grid_search_reproduces_2580d6e_transition():
    ref = np.load(REF)
    out = prc.reference_runs('jax')
    for k, v in out.items():
        assert np.array_equal(v, ref[f'tr_{k}'], equal_nan=True), k


def _off_node(m):
    """Moves every savings level a third of the way to the next node, so the
    lottery splits the mass of every household that saves."""
    grid = np.asarray(m.a_grid)
    a = np.asarray(m.a_next_policy_alpha, dtype=float).copy()
    k = np.clip(np.searchsorted(grid, a, side='left'), 0, len(grid) - 2)
    a = np.where(a > grid[0], a + (grid[k + 1] - a) / 3.0, a)
    m.a_next_policy_alpha = a
    m.a_next_policy = a[0]


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_monte_carlo_matches_exact_off_the_nodes(backend):
    Model = LifecycleModelPerfectForesight if backend == 'numpy' else _jax()
    m = Model(erc.lifecycle_config(), verbose=False)
    m.solve(verbose=False)
    _off_node(m)
    exact = m.exact_age_means()
    n = 40000 if backend == 'jax' else 8000
    sim = m.simulate(n_sim=n, seed=5)
    a = np.asarray(sim[0])
    se = a.std(axis=1) / np.sqrt(n)
    gap = np.abs(a.mean(axis=1) - exact[:, 0])
    assert np.all(gap < 4.5 * se + 1e-12), (gap, se)
    # assets carried out (the bequest of those who die) use the level itself
    beq = np.asarray(sim[20]).mean(axis=1)
    se_b = np.asarray(sim[20]).std(axis=1) / np.sqrt(n)
    assert np.all(np.abs(beq - exact[:, 20]) < 4.5 * se_b + 1e-12)
