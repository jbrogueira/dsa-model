"""
Report the base-year pin of the tax on gross output, fiscal.tau_y, and
optionally write it.

The government budget has no residual line since 2026-10-07. The primary
balance of the base year is made equal to fiscal.primary_balance_target_over_Y
by the rate tau_y of a tax on gross output paid by firms, which enters the
firm's conditions and so the wage. The joint solution of (A_tfp, tau_y) for
output of one and the target balance is normalize_A_tfp.py --pin-tau-y; this
script evaluates the base-year cross-section once at the configuration's
values and reports the balance and the rate that would hit the target at
these household outcomes (compute_fiscal_ratios['closure_tau_y']). With
--write it stores that rate, a one-step update that the normalisation's
iteration supersedes.

The balance is the full base-year primary balance as the transition books
it: the household-side balance plus the output tax and the transfer from
abroad, less G, defence, education, the lump-sum transfer and public
investment at the level the transition spends, (delta_g + Gamma_0 - 1) K_g,
over this routine's own output.

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
                       _demography_path, theta_from_config)

DEFAULT_CONFIG = 'calibration_input_GR.json'

p = argparse.ArgumentParser()
p.add_argument('--backend', default='jax', choices=['jax', 'numpy'])
p.add_argument('--config', default=DEFAULT_CONFIG)
p.add_argument('--write', action='store_true',
               help='write the one-step update of fiscal.tau_y')
args = p.parse_args()

loaded = load_config(args.config)
spec = loaded['spec']
config_data = loaded['config_data']

theta_dict = config_data.get('_derived', {}).get('theta')
if theta_dict is None:
    sys.exit("No _derived.theta in config — run calibration first.")
theta = theta_from_config(config_data, spec, verbose=True)
spec = dataclasses.replace(spec, backend=args.backend)

print("theta:", {pp.name: float(t) for pp, t in zip(spec.params, theta)})
print(f"backend={spec.backend}, n_sim={spec.n_sim}")
print("solving the base-year cross-section ...", flush=True)

_m, panels = run_model_moments(theta, spec, return_panels=True)
ratios = compute_fiscal_ratios(panels, spec, config_data)
if 'error' in ratios:
    sys.exit(f"compute_fiscal_ratios failed: {ratios['error']}")

fiscal = config_data.get('fiscal', {})
prod = config_data['production']
ext = config_data['external_params']
n0 = 0.0
demog = _demography_path(config_data)
if demog is not None:
    d = np.load(demog)
    yrs = list(np.asarray(d['pop_years'], dtype=int))
    n0 = float(np.asarray(d['n_path'])[yrs.index(
        int(config_data['transition'].get('current_year', int(d['base_year']))))])
Gamma0 = (1.0 + float(ext.get('trend_growth', 0.0))) * (1.0 + n0)
I_g_level = (float(prod.get('delta_g', 0.05)) + Gamma0 - 1.0) * float(prod.get('K_g', 0.0))
I_g_over_Y = I_g_level / float(ratios['Y'])
target = float(fiscal.get('primary_balance_target_over_Y', 0.0195))
tau_y = float(fiscal.get('tau_y', 0.0) or 0.0)

pb_full = ratios['primary_balance_full_over_Y'] + float(fiscal.get('I_g_over_Y', 0.0)) - I_g_over_Y
tau_pin = tau_y + (target - pb_full)

print(f"\nbase-year output                        : {float(ratios['Y']):.6f}")
print(f"household primary balance / Y           : {ratios['primary_balance_over_Y']:+.4f}")
print(f"output tax / Y                          : {tau_y:+.4f}")
print(f"transfer from abroad / Y                : {ratios['foreign_transfer_over_Y']:+.4f}")
print(f"G, defence, education, lump sum / Y     : {fiscal.get('G_over_Y', 0.0):.4f}, "
      f"{fiscal.get('defense_over_Y', 0.0):.4f}, {ratios['education_over_Y']:.4f}, "
      f"{ratios['lump_sum_over_Y']:.4f}")
print(f"I_g / Y (level the transition spends)   : {I_g_over_Y:.4f}  "
      f"(config ratio {fiscal.get('I_g_over_Y', 0.0)})")
print(f"full primary balance / Y                : {pb_full:+.4f}   target {target:+.4f}")
print(f"\ntau_y that hits the target at these outcomes : {tau_pin:.6f}  (config {tau_y:.6f})")
print("(the household block moves with tau_y through the wage; normalize_A_tfp.py "
      "--pin-tau-y iterates to the fixed point)")

if args.write:
    with open(args.config) as f:
        raw_disk = json.load(f)
    raw_disk.setdefault('fiscal', {})['tau_y'] = round(float(tau_pin), 6)
    with open(args.config, 'w') as f:
        json.dump(raw_disk, f, indent=2)
    print(f"\nWrote fiscal.tau_y={tau_pin:.6f} to {args.config}")

print("\nDONE")
