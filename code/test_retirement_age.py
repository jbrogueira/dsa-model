"""The retirement age path built by build_retirement_age_GR.py."""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, '..', 'data', 'retirement_age_GR.npz')


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


def test_effective_age_follows_the_fiche_to_2070_and_the_review_rule_after(d):
    """Country Fiche EL, Table 4: 63.8 (2022), 65.5 (2030), 66.4 (2040),
    66.6 (2050), 67.4 (2060), 67.9 (2070)."""
    years, R = list(d['years']), d['retirement_age_path']
    assert all(R[years.index(y)] == pytest.approx(63.8) for y in range(1939, 2023))
    for y, a in ((2022, 63.8), (2030, 65.5), (2040, 66.4), (2050, 66.6),
                 (2060, 67.4), (2070, 67.9)):
        assert R[years.index(y)] == pytest.approx(a)
    assert R[years.index(2026)] == pytest.approx(63.8 + 0.5 * (65.5 - 63.8))
    e = dict(zip(d['e65_years'].tolist(), d['e65'].tolist()))
    for r in (2072, 2075, 2099):          # reviews after 2070
        assert R[years.index(r)] == pytest.approx(67.9 + e[r] - e[2069])
        assert R[years.index(r + 1)] == R[years.index(r)]   # constant between reviews
    assert np.all(np.diff(R) >= -1e-12)


def test_rule_path_is_63_8_in_2023_and_moves_one_for_one_at_reviews(d):
    years, R = list(d['years']), d['rule_path']
    e = dict(zip(d['e65_years'].tolist(), d['e65'].tolist()))
    assert all(R[years.index(y)] == pytest.approx(63.8) for y in range(1939, 2024))
    for r in (2024, 2027, 2030, 2060, 2099):
        assert R[years.index(r)] == pytest.approx(63.8 + e[r] - e[2023])
        assert R[years.index(r + 1)] == R[years.index(r)]
    assert np.all(np.diff(R) >= -1e-12)


def test_each_cohort_is_split_around_the_age_in_force(d):
    R = dict(zip(d['years'].tolist(), d['retirement_age_path'].tolist()))
    for k, A, J, s in zip(d['entry_years'].tolist(), d['retirement_real_age'].tolist(),
                          d['J_R'].tolist(), d['share_later'].tolist()):
        assert A == pytest.approx(R[k + int(round(A)) - 25]), k
        assert J == int(np.floor(A)) - 25, k
        assert (J + 25) * (1 - s) + (J + 26) * s == pytest.approx(A), k
        assert 0.0 <= s < 1.0, k


def test_config_base_age_is_where_most_of_the_2023_retirees_retire(d):
    cfg = json.load(open(os.path.join(HERE, 'calibration_input_GR.json')))
    J = cfg['model']['retirement_age']
    k = cfg['transition']['current_year'] - J
    i = list(d['entry_years']).index(k)
    major = d['J_R'][i] + (1 if d['share_later'][i] > 0.5 else 0)
    assert major == J == 39
