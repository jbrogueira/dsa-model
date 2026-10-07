"""The unemployment-rate index built by build_unemployment_path_GR.py."""
import os

import numpy as np
import pytest

import build_unemployment_path_GR as bu

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, '..', 'data', 'unemployment_index_GR.npz')


@pytest.fixture(scope='module')
def d():
    if not os.path.exists(NPZ):
        pytest.skip('run build_unemployment_path_GR.py first')
    return np.load(NPZ)


@pytest.fixture(scope='module')
def at(d):
    return {int(y): i for i, y in enumerate(d['years'])}


def test_shapes(d):
    years = d['years']
    assert years[0] == 1900 and years[-1] == 2400
    assert np.array_equal(years, np.arange(1900, 2401))
    assert d['index'].shape == d['rate_25_64'].shape == years.shape
    assert np.all(np.isfinite(d['index']))
    assert np.allclose(d['rate_25_64'], d['data_rates'][0] * d['index'])
    assert int(d['base_year']) == 2023
    assert len(d['sources']) == 3


def test_index_is_one_before_and_in_the_base_year(d, at):
    assert np.all(d['index'][:at[2023] + 1] == 1.0)
    assert d['rate_25_64'][at[2023]] == d['data_rates'][0]


def test_outturns_and_forecast(d, at):
    """lfsa_urgaed, ages 25-64: 10.2 (2023), 9.5 (2024), 8.3 (2025); Spring
    2026 forecast, ages 15-74: 8.3 (2026), 7.9 (2027), converted by the 2023
    ratio 10.2 / 11.1."""
    years, rates = d['data_years'].tolist(), d['data_rates']
    assert years == [2023, 2024, 2025]
    assert rates[0] == pytest.approx(10.2, abs=0.2)        # the 2023 base
    for y, r in zip(years, rates):
        assert d['rate_25_64'][at[y]] == pytest.approx(r)
    assert d['forecast_years'].tolist() == [2026, 2027]
    assert d['forecast_rates'].tolist() == [8.3, 7.9]
    ratio = rates[0] / float(d['rate_15_74_2023'])
    assert ratio == pytest.approx(float(d['ratio_15_74']))
    for y, f in zip(d['forecast_years'], d['forecast_rates']):
        assert d['rate_25_64'][at[int(y)]] == pytest.approx(f * ratio)
        assert d['index'][at[int(y)]] == pytest.approx(f / float(d['rate_15_74_2023']))


def test_monotone_non_increasing_from_2025(d, at):
    assert np.all(np.diff(d['index'][at[2025]:]) <= 1e-12)


def test_2050_and_2055_are_the_ageing_report_levels(d, at):
    """Table II.1.50, row EL: 6.6 in 2050 and 6.5 in 2055, times the 2023
    ratio of the 25-64 rate to the Report's 20-64 rate, 12.0875 (between
    12.4 in 2022 and 9.9 in 2030)."""
    ar = dict(zip(d['ar_years'].tolist(), d['ar_rates'].tolist()))
    assert (ar[2022], ar[2030], ar[2050], ar[2055]) == (12.4, 9.9, 6.6, 6.5)
    ar_2023 = 12.4 + (9.9 - 12.4) / 8.0
    assert float(d['ar_rate_2023']) == pytest.approx(ar_2023)
    base = d['data_rates'][0]
    assert float(d['ratio_ar']) == pytest.approx(base / ar_2023)
    assert d['rate_25_64'][at[2050]] == pytest.approx(6.6 * base / ar_2023)
    assert d['index'][at[2050]] == pytest.approx(6.6 / ar_2023, abs=5e-4)
    assert d['index'][at[2055]] == pytest.approx(6.5 / ar_2023, abs=5e-4)
    assert round(float(d['index'][at[2050]]), 3) == 0.546


def test_linear_between_2027_and_2050_and_to_2055(d, at):
    idx = d['index']
    assert np.allclose(np.diff(idx[at[2027]:at[2050] + 1], n=2), 0.0, atol=1e-12)
    assert np.allclose(np.diff(idx[at[2050]:at[2055] + 1], n=2), 0.0, atol=1e-12)
    assert idx[at[2028]] < idx[at[2027]]


def test_constant_after_2055(d, at):
    assert np.all(d['index'][at[2055]:] == d['index'][at[2055]])


def test_rate_path_honours_its_knots():
    data = {2023: 10.0, 2024: 9.0, 2025: 8.0}
    fc = {2026: 8.0, 2027: 7.0}
    r = bu.rate_path([1990, 2023, 2024, 2025, 2026, 2027, 2050, 2055, 2300],
                     data, fc, ratio_15_74=0.5, ratio_ar=0.25)
    assert np.allclose(r, [10.0, 10.0, 9.0, 8.0, 4.0, 3.5, 6.6 * 0.25, 6.5 * 0.25,
                           6.5 * 0.25])
    assert bu.ar_rate(2023) == pytest.approx(12.4 + (10.9 - 12.4) / 3.0)
    assert bu.ar_rate_base_year() == pytest.approx(12.0875)
