"""
Tests of the calendar-year input paths built by build_r_B_path_GR.py,
build_school_age_GR.py and build_eu_transfers_GR.py.
"""
import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FIRST_YEAR, LAST_YEAR = 1900, 2400


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip(f'{name} not built')
    return np.load(path, allow_pickle=False)


def assert_full_coverage(years, values):
    assert years[0] == FIRST_YEAR and years[-1] == LAST_YEAR
    assert np.array_equal(years, np.arange(FIRST_YEAR, LAST_YEAR + 1))
    assert values.shape == years.shape
    assert not np.isnan(values).any()


class TestRBPath:
    @pytest.fixture(scope='class')
    def rb(self):
        return load('r_B_path_GR.npz')

    @pytest.fixture(scope='class')
    def dsa(self):
        return load('dsa_projection_GR.npz')

    def test_coverage(self, rb):
        assert_full_coverage(rb['years'], rb['r_B'])

    @pytest.mark.parametrize('year', [2030, 2060])
    def test_equals_projection(self, rb, dsa, year):
        model = float(rb['r_B'][rb['years'] == year][0])
        proj = float(dsa['real_effective_rate'][dsa['years'] == year][0])
        assert model == pytest.approx(proj, abs=1e-12)

    def test_terminal_rate(self, rb):
        assert np.all(rb['r_B'][rb['years'] >= 2070] == 0.02)

    def test_mean_2026_2060(self, rb, dsa):
        m = (rb['years'] >= 2026) & (rb['years'] <= 2060)
        d = (dsa['years'] >= 2026) & (dsa['years'] <= 2060)
        assert abs(rb['r_B'][m].mean() - dsa['real_effective_rate'][d].mean()) < 1e-6


class TestSchoolAge:
    @pytest.fixture(scope='class')
    def sa(self):
        return load('school_age_GR.npz')

    def test_coverage(self, sa):
        assert_full_coverage(sa['years'], sa['index'])

    def test_one_in_base_year(self, sa):
        assert float(sa['index'][sa['years'] == int(sa['base_year'])][0]) == pytest.approx(1.0, abs=1e-12)

    def test_range(self, sa):
        assert np.all(sa['index'] >= 0.5) and np.all(sa['index'] <= 1.5)


class TestEUTransfers:
    @pytest.fixture(scope='class')
    def eu(self):
        return load('eu_transfers_GR.npz')

    def test_coverage(self, eu):
        assert_full_coverage(eu['years'], eu['transfer_over_Y'])

    def test_nonnegative(self, eu):
        assert np.all(eu['transfer_over_Y'] >= 0.0)
