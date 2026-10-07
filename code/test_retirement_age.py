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


def test_effective_age_follows_the_fiche_to_2070_and_e65_after(d):
    """Country Fiche EL, Table 4: 63.8 (2022), 65.5 (2030), 66.4 (2040),
    66.6 (2050), 67.4 (2060), 67.9 (2070); after 2070, 0.75 of the change in
    e65 every year."""
    years, R = list(d['years']), d['retirement_age_path']
    assert all(R[years.index(y)] == pytest.approx(63.8) for y in range(1939, 2023))
    for y, a in ((2022, 63.8), (2030, 65.5), (2040, 66.4), (2050, 66.6),
                 (2060, 67.4), (2070, 67.9)):
        assert R[years.index(y)] == pytest.approx(a)
    assert R[years.index(2026)] == pytest.approx(63.8 + 0.5 * (65.5 - 63.8))
    assert float(d['pass_through']) == 0.75
    e = dict(zip(d['e65_years'].tolist(), d['e65'].tolist()))
    for y in (2071, 2072, 2085, 2100):
        assert R[years.index(y)] == pytest.approx(67.9 + 0.75 * (e[y] - e[2070]))
    assert R[years.index(2101)] == R[years.index(2100)]   # e65 held at 2100
    assert np.all(np.diff(R) >= -1e-12)
    late = np.diff(R[years.index(2070):years.index(2100)])
    assert late.max() < 0.09 and late.min() > 0.05       # no three-year steps


def test_effective_age_in_2100_and_continuity_at_2070(d):
    """R(2100) = 67.9 + 0.75 (e65(2100) - e65(2070)), 69.7; the fiche's
    segment and the e65 rule meet at 67.9 in 2070 with one-year steps on
    both sides (0.05 from the fiche's 2060-70 slope, 0.75 of the e65 change
    after)."""
    years, R = list(d['years']), d['retirement_age_path']
    e = dict(zip(d['e65_years'].tolist(), d['e65'].tolist()))
    r2100 = 67.9 + 0.75 * (e[2100] - e[2070])
    assert R[years.index(2100)] == pytest.approx(r2100)
    assert round(r2100, 1) == 69.7
    assert R[years.index(2070)] == pytest.approx(67.9)
    assert R[years.index(2070)] == pytest.approx(67.9 + 0.75 * (e[2070] - e[2070]))
    assert R[years.index(2070)] - R[years.index(2069)] == pytest.approx(0.05)
    step_after = R[years.index(2071)] - R[years.index(2070)]
    assert step_after == pytest.approx(0.75 * (e[2071] - e[2070]))
    assert 0.0 < step_after < 0.1


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
