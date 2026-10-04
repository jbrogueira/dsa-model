"""The statutory retirement age path built by build_retirement_age_GR.py."""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, '..', 'data', 'retirement_age_GR.npz')
RAW = os.path.join(HERE, '..', 'data', 'europop2023_raw')


@pytest.fixture(scope='module')
def d():
    if not os.path.exists(NPZ):
        pytest.skip('run build_retirement_age_GR.py first')
    return np.load(NPZ)


def test_e65_reproduces_the_ageing_report(d):
    """Country Fiche EL (2024 Ageing Report), life expectancy at 65:
    men 18.7 (2022) and 23.9 (2070), women 21.7 and 26.7."""
    y = list(d['e65_years'])
    for yr, men, women in ((2022, 18.7, 21.7), (2070, 23.9, 26.7)):
        assert round(float(d['e65_men'][y.index(yr)]), 1) == men
        assert round(float(d['e65_women'][y.index(yr)]), 1) == women


def test_statutory_age_is_67_in_2023_and_moves_one_for_one_at_reviews(d):
    years, S = list(d['years']), d['statutory_age']
    e = dict(zip(d['e65_years'].tolist(), d['e65'].tolist()))
    assert S[years.index(2023)] == 67.0
    assert all(S[years.index(y)] == 67.0 for y in range(1939, 2024))
    for r in (2024, 2027, 2030, 2060, 2099):
        assert S[years.index(r)] == pytest.approx(67 + e[r] - e[2023])
        assert S[years.index(r + 1)] == S[years.index(r)]   # constant between reviews
    assert np.all(np.diff(S) >= -1e-12)


def test_each_cohort_retires_at_the_rounded_age_in_force(d):
    S = dict(zip(d['years'].tolist(), d['statutory_age'].tolist()))
    for k, a in zip(d['entry_years'].tolist(), d['retirement_real_age'].tolist()):
        assert a == int(np.floor(S[k + a - 25] + 0.5)), k
    assert np.array_equal(d['J_R'], d['retirement_real_age'] - 25)


def test_config_base_age_matches_the_table(d):
    import json
    cfg = json.load(open(os.path.join(HERE, 'calibration_input_GR.json')))
    J = cfg['model']['retirement_age']
    k = cfg['transition']['current_year'] - J      # turns 25 + J in the base year
    assert d['J_R'][list(d['entry_years']).index(k)] == J == 42
