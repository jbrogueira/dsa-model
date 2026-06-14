"""
Pin the baseline fiscal closure (other_net_spending_over_Y) at the INITIAL
STEADY STATE — no transition.

other_net_spending is a structural constant that makes the initial-point
government budget consistent with the data primary balance. It is pinned at the
stationary calibration equilibrium (the same panels that match A/Y, tax_p/Y,
... to the SMM targets), NOT by forcing the transition's t=0 primary balance to
the target. The transition takes the pinned constant as given and produces a
time-varying primary-deficit path as a model output. (See
docs/FISCAL_EXPERIMENTS_STATUS.md, "Baseline closure: pinned at the initial
steady state", for why the SS and transition-t=0 cross-sections differ.)

Procedure (interest excluded throughout, matching the transition's primary
balance; pb_house is the household-side SS balance from compute_fiscal_ratios):

    pb_house = (tax_revenue - pension - ui - gov_health) / Y
    s_SS     = pb_house - (G_over_Y + I_g_over_Y + defense_over_Y)   # full, other=0
    other_net_spending_over_Y = s_SS - primary_balance_target_over_Y

This equals ratios['closure_other_over_Y'] from compute_fiscal_ratios.

Usage:
    python pin_baseline_closure.py [--backend jax|numpy] [--config FILE] [--write]
"""
import os, sys, platform, json, argparse, dataclasses
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

from calibrate import load_config, run_model_moments, compute_fiscal_ratios

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
print("solving stationary lifecycle (initial steady state) ...", flush=True)

_m, panels = run_model_moments(theta, spec, return_panels=True)
ratios = compute_fiscal_ratios(panels, spec, config_data)
if 'error' in ratios:
    sys.exit(f"compute_fiscal_ratios failed: {ratios['error']}")

fiscal = config_data.get('fiscal', {})
G_over_Y       = fiscal.get('G_over_Y', 0.0)
I_g_over_Y     = fiscal.get('I_g_over_Y', 0.0)
defense_over_Y = fiscal.get('defense_over_Y', 0.0)
target         = fiscal.get('primary_balance_target_over_Y', 0.0195)
discretionary  = G_over_Y + I_g_over_Y + defense_over_Y

pb_house     = ratios['primary_balance_over_Y']          # household-side SS balance
s_SS         = pb_house - discretionary                  # full SS primary surplus, other=0
other_over_Y = ratios['closure_other_over_Y']            # = s_SS - target

print(f"\nSS household primary balance / Y      : {pb_house:+.4f}")
print(f"  (G + I_g + defense)/Y               : {discretionary:.4f}  "
      f"[G={G_over_Y} I_g={I_g_over_Y} def={defense_over_Y}]")
print(f"SS full primary surplus / Y (other=0) : {s_SS:+.4f}")
print(f"target primary surplus / Y            : {target:+.4f}")
print(f"\nother_net_spending_over_Y             : {other_over_Y:+.6f}")
print(f"(current config value                 : {fiscal.get('other_net_spending_over_Y')})")
print(f"\ncheck: full SS primary balance at pinned other = "
      f"{pb_house - discretionary - other_over_Y:+.4f} (should equal target {target:+.4f})")

if args.write:
    with open(args.config) as f:
        raw_disk = json.load(f)
    raw_disk.setdefault('fiscal', {})['other_net_spending_over_Y'] = round(float(other_over_Y), 6)
    with open(args.config, 'w') as f:
        json.dump(raw_disk, f, indent=2)
    print(f"\nWrote other_net_spending_over_Y={other_over_Y:.6f} to {args.config} (fiscal block)")

print("\nDONE")
