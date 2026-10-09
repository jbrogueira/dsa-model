"""The pension-per-pensioner index: the data file, the replacement-rate path
it implies by cohort, and the pension floor that follows it."""
import dataclasses
import json
import os

import numpy as np
import pytest

from lifecycle_perfect_foresight import (LifecycleConfig, LifecycleModelPerfectForesight,
                                         encoded_pension_floor, pension_floor_at)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'calibration_input_GR.json')
NPZ = os.path.join(HERE, '..', 'data', 'pension_index_GR.npz')


@pytest.fixture(scope='module')
def raw():
    if not os.path.exists(CONFIG) or not os.path.exists(NPZ):
        pytest.skip('country config or pension index not present')
    return json.load(open(CONFIG))


def test_index_is_one_to_the_base_year_and_follows_the_fiche(raw):
    """Country Fiche EL, Tables 6 and 10: pension spending over GDP divided by
    pensioners over employment, 2022 = 1, is 0.836 in 2030 and 0.654 in 2050."""
    from calibrate import pension_index_path
    d = np.load(NPZ)
    assert np.allclose(d['fiche_index'], [1.0, 0.8358, 0.7498, 0.6541, 0.6036, 0.6112],
                       atol=5e-4)
    idx = pension_index_path(raw, [1990, 2022, 2023, 2030, 2050, 2070, 2071, 2200])
    base = 1.0 + (d['fiche_index'][1] - 1.0) / 8.0          # the fiche index in 2023
    assert np.allclose(idx[:3], 1.0)
    assert idx[3] == pytest.approx(d['fiche_index'][1] / base)
    assert idx[4] == pytest.approx(d['fiche_index'][3] / base)
    assert idx[5] == idx[6] == idx[7] == pytest.approx(d['fiche_index'][5] / base)


def test_cohort_rows_follow_the_calendar_diagonal(raw):
    from calibrate import base_year_cohort_pension_index, pension_index_path
    T = raw['model']['T']
    base = raw['transition']['current_year']
    X = base_year_cohort_pension_index(raw, T)
    assert X.shape == (T, T)
    for j in (0, 10, 39, 59):
        assert np.allclose(X[j, :j + 1], 1.0)               # ages already lived
        assert np.allclose(X[j], pension_index_path(raw, base + np.arange(T) - j))
    assert X[0, 27] == pytest.approx(float(pension_index_path(raw, [base + 27])[0]))


def test_transition_path_is_the_calibrated_rate_times_the_index(raw):
    from calibrate import build_olg_transition, pension_index_path
    _, paths, T_tr = build_olg_transition(raw, backend='numpy')
    rho = raw['_derived']['theta']['pension_replacement_default']
    idx = pension_index_path(raw, raw['transition']['current_year'] + np.arange(T_tr))
    assert np.allclose(paths['pension_replacement_path'], rho * idx)
    assert paths['pension_replacement_path'][0] == pytest.approx(rho)


def _config(**kw):
    T = 12
    base = dict(T=T, beta=0.96, gamma=1.0, n_a=30, a_max=20.0, n_y=3, n_h=1,
                retirement_age=8, education_type='medium',
                r_path=np.full(T, 0.03), w_path=np.ones(T),
                pension_replacement_default=0.4, pension_avg_weight=1.0)
    base.update(kw)
    return LifecycleConfig(**base)


def test_floor_encoding():
    assert encoded_pension_floor(_config(pension_min_floor=0.3)) == 0.3
    assert encoded_pension_floor(_config(pension_min_floor=0.0,
                                         pension_floor_indexed=True)) == 0.0
    f = encoded_pension_floor(_config(pension_min_floor=0.3, pension_floor_indexed=True))
    assert f == pytest.approx(-0.75)
    assert pension_floor_at(f, 0.4) == pytest.approx(0.3)
    assert pension_floor_at(f, 0.2) == pytest.approx(0.15)
    assert pension_floor_at(0.3, 0.2) == 0.3


def test_indexed_floor_is_the_absolute_floor_at_the_reference_rate():
    a = LifecycleModelPerfectForesight(_config(pension_min_floor=0.3), verbose=False)
    b = LifecycleModelPerfectForesight(
        _config(pension_min_floor=0.3, pension_floor_indexed=True), verbose=False)
    a.solve(verbose=False)
    b.solve(verbose=False)
    assert np.allclose(a.V, b.V, rtol=0, atol=1e-12)
    assert np.array_equal(a.a_next_policy, b.a_next_policy)


def test_indexed_floor_moves_with_the_replacement_rate():
    """Where the rate is half the reference rate the floor is half as high."""
    T, R = 12, 8
    path = np.r_[np.full(R, 0.4), np.full(T - R, 0.2)]
    lowest = {}
    for indexed in (False, True):
        m = LifecycleModelPerfectForesight(
            _config(pension_min_floor=5.0, pension_floor_indexed=indexed,
                    pension_replacement_path=path), verbose=False)
        m.solve(verbose=False)
        pension = np.asarray(m.simulate(T_sim=T, n_sim=200, seed=1)[16])   # pension_sim
        assert np.allclose(pension[:R], 0.0)
        lowest[indexed] = pension[R:].min()
    assert lowest[False] == pytest.approx(5.0)
    assert lowest[True] == pytest.approx(2.5)


def test_jax_matches_numpy_with_an_indexed_floor_and_a_falling_rate():
    pytest.importorskip('jax')
    from lifecycle_jax import LifecycleModelJAX
    T, R = 12, 8
    path = 0.4 * np.r_[np.ones(R), np.linspace(1.0, 0.6, T - R)]
    cfg = _config(pension_min_floor=0.25, pension_floor_indexed=True,
                  pension_replacement_path=path)
    n = LifecycleModelPerfectForesight(cfg, verbose=False)
    n.solve(verbose=False)
    j = LifecycleModelJAX(cfg, verbose=False)
    j.solve(verbose=False)
    assert np.allclose(np.asarray(j.V), n.V, rtol=0, atol=1e-9)
    assert np.array_equal(np.asarray(j.a_next_policy), n.a_next_policy)


def _small_spec(backend, aggregation):
    from test_olg_transition import TestBaseYearCrossSection
    helper = TestBaseYearCrossSection()
    cfg, spec, theta = helper._small()
    T = helper.T
    S = np.tile(np.asarray(spec.base_config.survival_probs, dtype=float).reshape(1, T),
                (T, 1))
    age = np.arange(T)
    idx = np.stack([np.where(age <= j, 1.0, 1.0 - 0.04 * (age - j)) for j in range(T)])
    spec = dataclasses.replace(
        spec, backend=backend, aggregation=aggregation, cohort_pension_index=idx,
        cohort_unemployment_index=None,   # the production index is for the production T
        # A fine asset grid: on the country grid shrunk to 25 points this small
        # economy holds no assets, and the path of future pensions would leave
        # no trace in the base-year cross-section.
        base_config=spec.base_config._replace(pension_min_floor=0.05,
                                              pension_floor_indexed=True,
                                              a_max=6.0, n_a=60))
    return cfg, spec, theta, S


def test_cross_section_with_an_index_changes_the_retired_and_the_savers():
    from calibrate import base_year_cross_section
    cfg, spec, theta, S = _small_spec('numpy', 'exact')
    with_idx = base_year_cross_section(theta, spec, cfg, survival=S)['medium']
    flat = base_year_cross_section(
        theta, dataclasses.replace(spec, cohort_pension_index=None), cfg,
        survival=S)['medium']
    def mean(p, f):
        w = np.asarray(p.weight_sim)
        return (np.asarray(getattr(p, f)) * w).sum(1) / w.sum(1)
    # Pensions in payment in the base year are those of the flat path: the
    # index is one up to the base year.
    assert np.allclose(mean(with_idx, 'pension_sim'), mean(flat, 'pension_sim'),
                       rtol=0, atol=1e-12)
    # Working cohorts expect lower pensions and hold more assets.
    assert mean(flat, 'a_sim')[3:9].sum() > 0.0
    assert mean(with_idx, 'a_sim')[3:9].sum() > 1.01 * mean(flat, 'a_sim')[3:9].sum()


def test_cross_section_with_an_index_batched_equals_one_at_a_time():
    pytest.importorskip('jax')
    from calibrate import base_year_cross_section
    cfg, spec, theta, S = _small_spec('jax', 'exact')
    one = base_year_cross_section(theta, spec, cfg, survival=S, batched=False)['medium']
    bat = base_year_cross_section(theta, spec, cfg, survival=S, batched=True)['medium']
    cfg_n, spec_n, _, _ = _small_spec('numpy', 'exact')
    ref = base_year_cross_section(theta, spec_n, cfg_n, survival=S)['medium']
    for f in ('a_sim', 'c_sim', 'pension_sim', 'weight_sim'):
        assert np.allclose(np.asarray(getattr(bat, f)), np.asarray(getattr(one, f)),
                           rtol=0, atol=1e-10), f
        assert np.allclose(np.asarray(getattr(bat, f)), np.asarray(getattr(ref, f)),
                           rtol=0, atol=1e-9), f
