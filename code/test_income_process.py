"""Regression tests for income-process discretization.

Asserts that the Tauchen-discretized y_grid and Gauss-Hermite alpha_grid match
the data targets recorded in each country's calibration JSON. Data-driven:
every calibration_input_*.json (excluding _test variants) is auto-discovered.

mu_y convention: in calibration_input_*.json, edu_params[edu]['mu_y'] is the
unconditional mean of log y for that education stratum (as estimated from the
country's earnings microdata; see data/lis/02_estimate_ar1.py for Greece).
"""
import glob
import json
import pathlib

import numpy as np
import pytest

from calibrate import build_lifecycle_config
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight


_HERE = pathlib.Path(__file__).resolve().parent
CONFIGS = sorted(
    p for p in glob.glob(str(_HERE / 'calibration_input_*.json'))
    if not p.endswith('_test.json')
)


def _config_id(path):
    return pathlib.Path(path).stem.replace('calibration_input_', '')


def _build_model(raw, edu):
    base, _ = build_lifecycle_config(raw)
    return LifecycleModelPerfectForesight(base._replace(education_type=edu),
                                          verbose=False)


@pytest.mark.parametrize('config_path', CONFIGS, ids=_config_id)
def test_y_grid_uncond_mean_log_matches_mu_y(config_path):
    with open(config_path) as f:
        raw = json.load(f)
    for edu, p in raw['edu_params'].items():
        model = _build_model(raw, edu)
        # Tauchen state values live in log space; the model stores exp() of
        # them as y_grid[1:]. y_grid[0] is the unemployment state (y=0).
        log_y_grid = np.log(model.y_grid[1:])
        got = float(log_y_grid.mean())
        want = float(p['mu_y'])
        assert abs(got - want) < 1e-12, (
            f"{_config_id(config_path)} [{edu}]: "
            f"arithmetic mean of log(y_grid[1:]) = {got:.6f}, want mu_y = {want:.6f}"
        )


@pytest.mark.parametrize('config_path', CONFIGS, ids=_config_id)
def test_alpha_grid_weighted_mean_is_zero(config_path):
    with open(config_path) as f:
        raw = json.load(f)
    for edu in raw['edu_params']:
        model = _build_model(raw, edu)
        got = float(np.dot(model.alpha_probs, model.alpha_grid))
        assert abs(got) < 1e-12, (
            f"{_config_id(config_path)} [{edu}]: "
            f"weighted mean of alpha_grid = {got:.3e}, want 0"
        )
