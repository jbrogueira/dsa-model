"""
The firm's first-order conditions in the small open economy, with a tax on
gross output.

Output is Y = A K_g^{eta_g} K^alpha L^{1-alpha}. A tax tau_y on gross output is
paid by the firm, so the conditions for capital and labour are

    (1 - tau_y) alpha A K_g^{eta_g} (K/L)^{alpha-1} = r + delta,
    w = (1 - tau_y)(1 - alpha) A K_g^{eta_g} (K/L)^alpha.

With the world return r given, the first condition fixes K/L and the second
the wage. At tau_y = 0 they are the conditions the model used until
2026-10-07. Every site that needs K/L or w from (r, A, K_g) calls this
function: the transition's wage path and implied domestic capital
(olg_transition.py), the calibration's price derivation
(calibrate.compute_equilibrium_prices) and the evaluator's check of the
conditions (eval_fiscal_results.chk_firm_foc).
"""
import numpy as np


def firm_conditions(r, A, K_g_factor=1.0, alpha=0.33, delta=0.05, tau_y=0.0):
    """K/L, the wage and output per unit of labour from the two conditions.

    r, K_g_factor (= K_g^{eta_g}) and tau_y may be scalars or arrays of one
    shape; the result broadcasts. Returns (K_over_L, w, Y_over_L).
    """
    r = np.asarray(r, dtype=float)
    tau_y = np.asarray(tau_y, dtype=float)
    K_g_factor = np.asarray(K_g_factor, dtype=float)
    net = (1.0 - tau_y) * A * K_g_factor
    K_over_L = np.power((r + delta) / (alpha * net), 1.0 / (alpha - 1.0))
    w = (1.0 - alpha) * net * np.power(K_over_L, alpha)
    Y_over_L = A * K_g_factor * np.power(K_over_L, alpha)
    return K_over_L, w, Y_over_L


def marginal_products(K, L, alpha, delta, A, K_g=1.0, eta_g=0.0, tau_y=0.0):
    """The return and the wage implied by (K, L): the conditions read the
    other way. Returns (r, w)."""
    K_g_factor = K_g ** eta_g if eta_g != 0.0 else 1.0
    Y = A * K_g_factor * K ** alpha * L ** (1.0 - alpha)
    r = (1.0 - tau_y) * alpha * Y / K - delta
    w = (1.0 - tau_y) * (1.0 - alpha) * Y / L
    return r, w
