"""
Pin the baseline fiscal closure (other_net_spending_over_Y) at the BASE-YEAR
EQUILIBRIUM — no transition.

other_net_spending is a structural constant that makes the base-year
government budget consistent with the data primary balance. It is pinned at the
calibration equilibrium (the same panels that match A/Y, tax_p/Y,
... to the SMM targets), NOT by forcing the transition's t=0 primary balance to
the target. The transition takes the pinned constant as given and produces a
time-varying primary-deficit path as a model output. (See
docs/FISCAL_EXPERIMENTS_STATUS.md, "Baseline closure: pinned at the initial
steady state", for why the two cross-sections differ.)

The base-year equilibrium is one lifecycle problem at constant detrended
prices, aggregated over the measured 2023 cross-section. It is not a steady
state: with growth no level is stationary, and with a population still in
transition neither is the detrended aggregate.

Procedure (interest excluded throughout, matching the transition's primary
balance; pb_house is the household-side balance from compute_fiscal_ratios):

    pb_house = (tax_revenue - pension - ui - gov_health) / Y
    s_SS     = pb_house - (G_over_Y + I_g_over_Y + defense_over_Y)   # full, other=0
    other_net_spending_over_Y = s_SS - primary_balance_target_over_Y

This equals ratios['closure_other_over_Y'] from compute_fiscal_ratios.

Usage:
    python pin_baseline_closure.py [--backend jax|numpy] [--config FILE] [--write]
"""
import os
import sys
import platform
import json
import argparse
import dataclasses
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

from calibrate import (load_config, run_model_moments, compute_fiscal_ratios,
                       _demography_path)

DEFAULT_CONFIG = 'calibration_input_GR.json'

p = argparse.ArgumentParser()
p.add_argument('--backend', default='jax', choices=['jax', 'numpy'])
p.add_argument('--config', default=DEFAULT_CONFIG)
p.add_argument('--write', action='store_true',
               help='write the pinned value into fiscal.other_net_spending_over_Y')
args = p.parse_args()

loaded = load_config(args.config)
spec = loaded['spec']
config_data = loaded['config_data']

theta_dict = config_data.get('_derived', {}).get('theta')
if theta_dict is None:
    sys.exit("No _derived.theta in config — run calibration first.")
theta = np.array([theta_dict[pp.name] for pp in spec.params])
spec = dataclasses.replace(spec, backend=args.backend)

print("theta:", {pp.name: float(t) for pp, t in zip(spec.params, theta)})
print(f"backend={spec.backend}, n_sim={spec.n_sim}")
print("solving the stationary lifecycle problem (base-year equilibrium) ...", flush=True)

_m, panels = run_model_moments(theta, spec, return_panels=True)
ratios = compute_fiscal_ratios(panels, spec, config_data)
if 'error' in ratios:
    sys.exit(f"compute_fiscal_ratios failed: {ratios['error']}")

fiscal = config_data.get('fiscal', {})
G_over_Y       = fiscal.get('G_over_Y', 0.0)
# The transition spends I_g as a LEVEL, (delta_g + Gamma_0 - 1) * K_g, because
# eta_g != 0 makes ratio mode a fixed point. Subtracting the config's
# I_g_over_Y here instead would pin the closure against a different number than
# the budget actually pays: 0.079pp of output at t=0 on the 2026-10-01
# calibration, widening as output falls through the ageing transition. So use
# the level the transition will spend, over this routine's own output.
prod           = loaded['config_data']['production']
_delta_g       = float(prod.get('delta_g', 0.05))
_K_g           = float(prod.get('K_g', 0.0))
_g             = float(loaded['config_data']['external_params'].get('trend_growth', 0.0))
_n0            = 0.0
_demog         = _demography_path(loaded['config_data'])
if _demog is not None:
    import numpy as _np
    _d = _np.load(_demog)
    _i = list(_np.asarray(_d['pop_years'], dtype=int)).index(
        int(loaded['config_data']['transition'].get('current_year',
                                                    int(_d['base_year']))))
    _n0 = float(_np.asarray(_d['n_path'])[_i])
_Gamma0        = (1.0 + _g) * (1.0 + _n0)
_I_g_level     = (_delta_g + _Gamma0 - 1.0) * _K_g
I_g_over_Y     = _I_g_level / float(ratios['Y'])
print(f"  I_g: level {_I_g_level:.6f} / Y {float(ratios['Y']):.6f} = {I_g_over_Y:.6f}"
      f"   (config ratio {fiscal.get('I_g_over_Y', 0.0)})")
defense_over_Y = fiscal.get('defense_over_Y', 0.0)
target         = fiscal.get('primary_balance_target_over_Y', 0.0195)
discretionary  = G_over_Y + I_g_over_Y + defense_over_Y

pb_house     = ratios['primary_balance_over_Y']          # household-side base-year balance
s_SS         = pb_house - discretionary                  # full base-year primary surplus, other=0
other_over_Y = ratios['closure_other_over_Y']            # = s_SS - target

print(f"\nbase-year household primary balance / Y : {pb_house:+.4f}")
print(f"  (G + I_g + defense)/Y               : {discretionary:.4f}  "
      f"[G={G_over_Y} I_g={I_g_over_Y} def={defense_over_Y}]")
print(f"base-year full primary surplus / Y (other=0) : {s_SS:+.4f}")
print(f"target primary surplus / Y            : {target:+.4f}")
print(f"\nother_net_spending_over_Y             : {other_over_Y:+.6f}")
print(f"(current config value                 : {fiscal.get('other_net_spending_over_Y')})")
print(f"\ncheck: full base-year primary balance at pinned other = "
      f"{pb_house - discretionary - other_over_Y:+.4f} (should equal target {target:+.4f})")

if args.write:
    with open(args.config) as f:
        raw_disk = json.load(f)
    raw_disk.setdefault('fiscal', {})['other_net_spending_over_Y'] = round(float(other_over_Y), 6)
    with open(args.config, 'w') as f:
        json.dump(raw_disk, f, indent=2)
    print(f"\nWrote other_net_spending_over_Y={other_over_Y:.6f} to {args.config} (fiscal block)")

print("\nDONE")
