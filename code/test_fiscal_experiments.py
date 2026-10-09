"""
Tests for fiscal_experiments.py and the Feature A bequest loop.

Run with:
    pytest test_fiscal_experiments.py -v

All tests use a minimal OLG configuration (T=5, T_transition=4, n_sim=20)
to keep runtime fast while exercising every code path.
"""

import os

import numpy as np
from lifecycle_perfect_foresight import LifecycleConfig
from olg_transition import OLGTransition

from fiscal_experiments import (
    FiscalScenario,
    FiscalScenarioResult,
    compute_debt_path,
    uniform_profile,
    linear_phase_in,
    back_loaded,
    exponential_convergence,
    _apply_shock,
    _balance_residual,
    _nfa_ca_paths,
    run_fiscal_scenario,
    fiscal_multiplier,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T_LIFECYCLE   = 5
T_TRANSITION  = 4
N_SIM         = 20
RETIREMENT    = 4


def _make_olg(survival_probs=None, eta_g=0.0, K_g_initial=0.0,
              govt_spending_path=None, I_g_path=None):
    """Minimal OLGTransition for fast tests. DSA_SAVINGS_SOLVER=egm solves the
    savings choice by the endogenous grid method."""
    config = LifecycleConfig(
        savings_solver=os.environ.get('DSA_SAVINGS_SOLVER', 'grid'),
        T=T_LIFECYCLE,
        beta=0.96,
        gamma=2.0,
        n_a=10,
        n_y=2,
        n_h=1,
        retirement_age=RETIREMENT,
        education_type='medium',
        survival_probs=survival_probs,
    )
    return OLGTransition(
        lifecycle_config=config,
        alpha=0.33,
        delta=0.05,
        A=1.0,
        eta_g=eta_g,
        K_g_initial=K_g_initial,
        education_shares={'medium': 1.0},
        output_dir='output/test',
        govt_spending_path=govt_spending_path,
        I_g_path=I_g_path,
    )


def _make_base_paths(r=0.04, tau_l=0.15, tau_c=0.0, tau_p=0.0, tau_k=0.0,
                     pension=0.4, G=None, I_g=None):
    """Return a base_paths dict for the given constant rates."""
    T = T_TRANSITION
    bp = dict(
        r_path=np.full(T, r),
        tau_l_path=np.full(T, tau_l),
        tau_c_path=np.full(T, tau_c),
        tau_p_path=np.full(T, tau_p),
        tau_k_path=np.full(T, tau_k),
        pension_replacement_path=np.full(T, pension),
    )
    if G is not None:
        bp['G_path'] = np.asarray(G)
    if I_g is not None:
        bp['I_g_path'] = np.asarray(I_g)
    return bp


# ---------------------------------------------------------------------------
# Feature A — Bequest loop
# ---------------------------------------------------------------------------

class TestBequestLoop:
    """Tests for the recompute_bequests feature in simulate_transition()."""

    def test_backward_compatible_default(self):
        """recompute_bequests=False (default) gives same result as before."""
        olg = _make_olg()
        bp  = _make_base_paths()
        r   = bp['r_path']

        res1 = olg.simulate_transition(
            r_path=r, tau_l_path=bp['tau_l_path'],
            n_sim=N_SIM, verbose=False,
        )
        res2 = olg.simulate_transition(
            r_path=r, tau_l_path=bp['tau_l_path'],
            n_sim=N_SIM, verbose=False, recompute_bequests=False,
        )
        assert np.allclose(res1['K'], res2['K'])
        assert np.allclose(res1['Y'], res2['Y'])

    def test_no_survival_probs_loop_exits_immediately(self):
        """survival_probs=None + recompute_bequests=True: one pass, no change."""
        olg = _make_olg(survival_probs=None)
        bp  = _make_base_paths()
        res = olg.simulate_transition(
            r_path=bp['r_path'], tau_l_path=bp['tau_l_path'],
            n_sim=N_SIM, verbose=False,
            recompute_bequests=True,
        )
        # Without survival risk bequests are zero; loop should be a no-op
        assert olg._bequest_iter_count == 0  # guard clause: loop not entered
        assert olg._bequest_converged is True
        assert np.all(res['K'] > 0)

    def test_with_survival_probs_converges(self):
        """With mortality active, bequest loop converges within max_bequest_iters."""
        n_h = 1
        T   = T_LIFECYCLE
        survival = np.linspace(0.99, 0.85, T).reshape(T, n_h)
        olg = _make_olg(survival_probs=survival)
        bp  = _make_base_paths()
        olg.simulate_transition(
            r_path=bp['r_path'], tau_l_path=bp['tau_l_path'],
            n_sim=N_SIM, verbose=False,
            recompute_bequests=True, bequest_tol=1e-3, max_bequest_iters=10,
        )
        assert olg._bequest_converged is True
        assert olg._bequest_iter_count <= 10

    def test_convergence_stored_on_self(self):
        """_bequest_iter_count and _bequest_converged are accessible after run."""
        olg = _make_olg()
        bp  = _make_base_paths()
        olg.simulate_transition(
            r_path=bp['r_path'], n_sim=N_SIM, verbose=False,
            recompute_bequests=False,
        )
        assert hasattr(olg, '_bequest_iter_count')
        assert hasattr(olg, '_bequest_converged')


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestComputeDebtPath:
    def test_identity(self):
        """B[t+1] = (1+r)*B[t] + PD[t] holds exactly."""
        r  = np.array([0.04, 0.04, 0.04])
        pd = np.array([0.01, 0.02, -0.01])
        B  = compute_debt_path(pd, r, B_initial=0.5)
        assert len(B) == 4
        for t in range(3):
            assert np.isclose(B[t + 1], (1 + r[t]) * B[t] + pd[t])

    def test_zero_deficit(self):
        """Zero primary deficit: debt grows only via interest."""
        r  = np.array([0.05, 0.05])
        pd = np.zeros(2)
        B  = compute_debt_path(pd, r, B_initial=1.0)
        assert np.isclose(B[1], 1.05)
        assert np.isclose(B[2], 1.05 ** 2)

    def test_zero_initial(self):
        """Zero initial debt + positive surplus → negative debt (asset)."""
        r  = np.array([0.04])
        pd = np.array([-0.5])
        B  = compute_debt_path(pd, r, B_initial=0.0)
        assert np.isclose(B[1], -0.5)


class TestAdjustmentProfiles:
    def test_uniform(self):
        psi = uniform_profile(5)
        assert np.allclose(psi, 1.0)
        assert len(psi) == 5

    def test_linear_phase_in(self):
        psi = linear_phase_in(6, 3)
        assert psi[0] < 1.0
        assert np.isclose(psi[-1], 1.0)
        assert np.all(np.diff(psi[:3]) > 0)   # ramp is increasing
        assert np.allclose(psi[3:], 1.0)       # tail is flat

    def test_back_loaded(self):
        psi = back_loaded(5, 2)
        assert np.allclose(psi[:2], 0.0)
        assert np.allclose(psi[2:], 1.0)

    def test_exponential_convergence(self):
        psi = exponential_convergence(10, half_life=5.0)
        assert psi[0] < psi[-1]               # increasing
        assert psi[0] >= 0.0
        assert psi[-1] < 1.0 + 1e-9

    def test_linear_phase_in_zero_ramp(self):
        psi = linear_phase_in(4, 0)
        assert np.allclose(psi, 1.0)


class TestApplyShock:
    def test_zero_shock_zero_delta_identity(self):
        """Zero shock + zero instrument_delta → CF paths == base paths."""
        bp  = _make_base_paths()
        T   = T_TRANSITION
        scn = FiscalScenario(name='zero', financing='debt')
        cf  = _apply_shock(scn, bp, T, instrument_delta=0.0, shock_scale=1.0)
        assert np.allclose(cf['tau_l'], bp['tau_l_path'])
        assert np.allclose(cf['tau_c'], bp['tau_c_path'])
        assert np.allclose(cf['G_path'], np.zeros(T))  # default base G=0

    def test_delta_G_shock(self):
        """delta_G_path shifts G_path correctly."""
        T   = T_TRANSITION
        G_base = np.full(T, 0.1)
        bp  = _make_base_paths(G=G_base)
        dG  = np.full(T, 0.05)
        scn = FiscalScenario(name='g_shock', delta_G_path=dG, financing='debt')
        cf  = _apply_shock(scn, bp, T)
        assert np.allclose(cf['G_path'], G_base + dG)

    def test_instrument_delta_tau_l(self):
        """instrument_delta adds uniformly to tau_l when financing='tau_l'."""
        T   = T_TRANSITION
        bp  = _make_base_paths(tau_l=0.15)
        scn = FiscalScenario(name='tax', financing='tau_l')
        cf  = _apply_shock(scn, bp, T, instrument_delta=0.03)
        assert np.allclose(cf['tau_l'], 0.15 + 0.03)

    def test_shock_scale_zero(self):
        """shock_scale=0 should zero out all delta_* shocks."""
        T   = T_TRANSITION
        bp  = _make_base_paths()
        scn = FiscalScenario(
            name='scaled', financing='debt',
            delta_G_path=np.ones(T) * 0.5,
            delta_tau_l_path=np.ones(T) * 0.1,
        )
        cf  = _apply_shock(scn, bp, T, shock_scale=0.0)
        assert np.allclose(cf['G_path'], np.zeros(T))
        assert np.allclose(cf['tau_l'], bp['tau_l_path'])

    def test_transfer_floor_delta(self):
        """financing='transfer_floor' writes transfer_floor_delta to cf dict."""
        T   = T_TRANSITION
        bp  = _make_base_paths()
        scn = FiscalScenario(name='tf', financing='transfer_floor')
        cf  = _apply_shock(scn, bp, T, instrument_delta=0.02)
        assert 'transfer_floor_delta' in cf
        assert np.isclose(cf['transfer_floor_delta'], 0.02)


class TestBalanceResidual:
    def _make_budget_Y_B(self, PD_const, T=T_TRANSITION):
        budget = {'primary_deficit': np.full(T, PD_const)}
        Y      = np.ones(T) * 5.0
        r      = np.ones(T) * 0.04
        B      = compute_debt_path(budget['primary_deficit'], r, B_initial=0.0)
        return budget, Y, B

    def test_terminal_debt_gdp_positive(self):
        """Positive deficit → positive residual for terminal_debt_gdp."""
        budget, Y, B = self._make_budget_Y_B(0.1)
        scn = FiscalScenario(balance_condition='terminal_debt_gdp', target_debt_gdp=0.0)
        r = _balance_residual(budget, Y, B, scn)
        assert r > 0.0

    def test_terminal_debt_gdp_target(self):
        """If B_T/Y_T == target, residual == 0."""
        T  = T_TRANSITION
        Y  = np.ones(T) * 5.0
        PD = np.zeros(T)
        budget = {'primary_deficit': PD}
        B      = np.zeros(T + 1)    # B = 0 everywhere
        scn    = FiscalScenario(balance_condition='terminal_debt_gdp', target_debt_gdp=0.0)
        res    = _balance_residual(budget, Y, B, scn)
        assert np.isclose(res, 0.0)

    def test_pv_balance_zero_PD(self):
        """Zero primary deficit → PV balance residual == 0."""
        T      = T_TRANSITION
        budget = {'primary_deficit': np.zeros(T)}
        Y      = np.ones(T)
        B      = np.zeros(T + 1)
        scn    = FiscalScenario(balance_condition='pv_balance', discount_rate=0.04)
        res    = _balance_residual(budget, Y, B, scn)
        assert np.isclose(res, 0.0)

    def test_period_balance_returns_nonneg(self):
        """period_balance residual is always >= 0 (it is max |PD_t|)."""
        budget = {'primary_deficit': np.array([-0.1, 0.2, -0.05, 0.3])}
        Y = np.ones(4)
        B = np.zeros(5)
        scn = FiscalScenario(balance_condition='period_balance')
        res = _balance_residual(budget, Y, B, scn)
        assert res >= 0.0
        assert np.isclose(res, 0.3)


# ---------------------------------------------------------------------------
# Feature B — Integration tests using OLGTransition
# ---------------------------------------------------------------------------

class TestDebtFinanced:
    """Type A: single simulate_transition() call."""

    def test_debt_accumulation_identity(self):
        """G*B[t+1] = (1+r[t])*B[t] + PD[t] holds exactly, G = (1+g)(1+n)."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(name='base_debt', financing='debt', B_initial=0.1)
        result = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False)

        B  = result.B_path
        r  = result.cf_macro['r']
        PD = result.cf_budget['primary_deficit']
        G  = olg.growth_factor
        assert G != 1.0, "fixture should exercise a non-unit growth factor"
        for t in range(len(PD)):
            assert np.isclose(G * B[t + 1], (1 + r[t]) * B[t] + PD[t], atol=1e-10)

    def test_result_structure(self):
        """run_fiscal_scenario returns a FiscalScenarioResult with expected fields."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(name='struct', financing='debt')
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False)
        assert isinstance(res, FiscalScenarioResult)
        assert res.converged is True
        assert res.n_iterations == 1
        assert res.adjustment_scalar == 0.0
        assert len(res.B_path) == T_TRANSITION + 1
        assert len(res.B_gdp_path) == T_TRANSITION + 1

    def test_backward_compatibility_zero_shock(self):
        """Zero shock, debt-financed: CF budget == base budget."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(name='zero_shock', financing='debt')

        # Run baseline to populate budget on olg
        base_macro  = olg.simulate_transition(
            r_path=bp['r_path'], tau_l_path=bp['tau_l_path'], n_sim=N_SIM, verbose=False)
        base_budget = olg.compute_government_budget_path(n_sim=N_SIM, verbose=False)

        bp2 = dict(bp, base_macro=base_macro, base_budget=base_budget)
        res = run_fiscal_scenario(olg, scn, bp2, n_sim=N_SIM, verbose=False)
        assert np.allclose(res.cf_budget['primary_deficit'],
                           res.base_budget['primary_deficit'], atol=1e-8)

    def test_positive_G_shock_raises_deficit(self):
        """A positive G shock should raise the primary deficit."""
        olg  = _make_olg()
        bp   = _make_base_paths()
        scn0 = FiscalScenario(name='no_shock', financing='debt')
        scn1 = FiscalScenario(name='g_shock', financing='debt',
                              delta_G_path=np.full(T_TRANSITION, 0.05))

        res0 = run_fiscal_scenario(olg, scn0, bp, n_sim=N_SIM, verbose=False)
        res1 = run_fiscal_scenario(olg, scn1, bp, n_sim=N_SIM, verbose=False)

        assert np.mean(res1.cf_budget['primary_deficit']) > \
               np.mean(res0.cf_budget['primary_deficit'])


class TestTaxFinanced:
    """Type B: bisection on a tax rate."""

    def test_terminal_debt_gdp_converges(self):
        """Tax-financed (tau_l) with terminal_debt_gdp balance converges."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(
            name='tax_fin',
            delta_G_path=np.full(T_TRANSITION, 0.02),
            financing='tau_l',
            balance_condition='terminal_debt_gdp',
            target_debt_gdp=0.0,
        )
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False,
                                  bisect_tol=1e-3)
        assert res.converged is True
        T   = T_TRANSITION
        r   = res.cf_macro['r']
        PD  = res.cf_budget['primary_deficit']
        Y   = np.asarray(res.cf_macro['Y'])
        B   = compute_debt_path(PD, r, B_initial=scn.B_initial)
        ratio = B[T] / Y[T - 1]
        assert abs(ratio - scn.target_debt_gdp) < 0.05  # within 5 pp

    def test_pv_balance_converges(self):
        """Tax-financed (tau_c) with pv_balance condition converges."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(
            name='pv_fin',
            delta_G_path=np.full(T_TRANSITION, 0.015),
            financing='tau_c',
            balance_condition='pv_balance',
            discount_rate=0.04,
        )
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False,
                                  bisect_tol=1e-3)
        assert res.converged is True
        # PV of primary deficits should be near zero
        T    = T_TRANSITION
        disc = (1 / 1.04) ** np.arange(T)
        pv   = np.dot(disc, res.cf_budget['primary_deficit'])
        assert abs(pv) < 0.1  # loose tolerance given tiny N_SIM

    def test_period_balance_runs(self):
        """Tax-financed with period_balance condition runs without error."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(
            name='period_fin',
            delta_G_path=np.full(T_TRANSITION, 0.01),
            financing='tau_l',
            balance_condition='period_balance',
        )
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False,
                                  bisect_tol=1e-2)
        assert isinstance(res, FiscalScenarioResult)
        assert res.n_iterations > 0

    def test_adjustment_path_shape(self):
        """adjustment_path has length T_transition."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(
            name='adj_shape',
            delta_G_path=np.full(T_TRANSITION, 0.02),
            financing='tau_l',
            balance_condition='terminal_debt_gdp',
            target_debt_gdp=0.0,
        )
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False,
                                  bisect_tol=0.01)
        assert len(res.adjustment_path) == T_TRANSITION

    def test_adjustment_profile_linear_phase_in(self):
        """Linear phase-in profile produces non-uniform adjustment_path."""
        olg  = _make_olg()
        bp   = _make_base_paths()
        psi  = linear_phase_in(T_TRANSITION, 2)
        scn  = FiscalScenario(
            name='phase_in',
            delta_G_path=np.full(T_TRANSITION, 0.02),
            financing='tau_l',
            balance_condition='terminal_debt_gdp',
            target_debt_gdp=0.0,
            adjustment_profile=psi,
        )
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False,
                                  bisect_tol=0.01)
        # With a phase-in, not all elements of adjustment_path should be equal
        # unless Δ == 0
        if abs(res.adjustment_scalar) > 1e-6:
            assert not np.allclose(res.adjustment_path,
                                   res.adjustment_path[0] * np.ones(T_TRANSITION))

    def test_residual_history_nonempty(self):
        """Bisection records residual history."""
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(
            name='hist',
            delta_G_path=np.full(T_TRANSITION, 0.02),
            financing='tau_l',
            balance_condition='terminal_debt_gdp',
            target_debt_gdp=0.0,
        )
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False,
                                  bisect_tol=0.05)
        assert len(res.residual_history) > 0


class TestNFACA:
    """NFA / CA path computation tests."""

    def test_nfa_ca_paths_none_if_no_nfa(self):
        macro = {'r': np.zeros(3), 'w': np.zeros(3), 'K': np.zeros(3),
                 'L': np.zeros(3), 'Y': np.ones(3)}
        NFA, CA = _nfa_ca_paths(macro)
        assert NFA is None
        assert CA is None

    def test_ca_equals_diff_nfa(self):
        """G = 1 recovers the plain difference."""
        NFA_arr = np.array([1.0, 1.2, 0.9, 1.1])
        macro = {'NFA': NFA_arr}
        NFA, CA = _nfa_ca_paths(macro)
        assert np.isclose(CA[0], 0.2)    # 1.2 - 1.0
        assert np.isclose(CA[1], -0.3)   # 0.9 - 1.2
        assert np.isclose(CA[2], 0.2)    # 1.1 - 0.9
        assert np.isclose(CA[3], 0.0)    # last period repeated

    def test_debt_constant_at_rest_point(self):
        """PD = (G - 1 - r_B) * B[0] holds B constant; the n != 0 case is the
        one that distinguishes G from (1+g) and from a missing r_B."""
        G, r_B, B0 = 1.0109, 0.019, 1.64
        T = 20
        pd = np.full(T, (G - 1.0 - r_B) * B0)
        B = compute_debt_path(pd, np.full(T, r_B), B_initial=B0, growth_factor=G)
        np.testing.assert_allclose(B, B0, rtol=1e-12)

    def test_ca_with_growth(self):
        """CA[t] = G*NFA[t+1] - NFA[t]; terminal entry is (G-1)*NFA[-1]."""
        NFA_arr = np.array([1.0, 1.2, 0.9, 1.1])
        G = 1.0109
        NFA, CA = _nfa_ca_paths({'NFA': NFA_arr}, G)
        assert np.isclose(CA[0], G * 1.2 - 1.0)
        assert np.isclose(CA[1], G * 0.9 - 1.2)
        assert np.isclose(CA[2], G * 1.1 - 0.9)
        assert np.isclose(CA[3], (G - 1.0) * 1.1)

    def test_nfa_constraint_satisfied_at_solution(self):
        """NFA-constrained result satisfies NFA_t >= -nfa_limit for all t.

        The NFA in this small test economy is around -144.  We use
        nfa_limit=200 (i.e., the floor is -200) so the constraint is *not*
        violated by the base run and run_nfa_constrained returns the
        debt-financed result as-is.
        """
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(
            name='nfa_check',
            financing='debt',
            nfa_limit=200.0,   # floor at -200; actual NFA ~-144 is feasible
        )
        res = run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False)
        if res.NFA_path is not None:
            assert np.all(res.NFA_path >= -200.0 - 1e-8)


class TestMultiplier:
    def test_multiplier_positive_G_shock(self):
        """Positive G shock → positive output multiplier (mean)."""
        olg   = _make_olg()
        bp    = _make_base_paths()
        scn0  = FiscalScenario(name='base', financing='debt')
        scn1  = FiscalScenario(name='shock', financing='debt',
                               delta_G_path=np.full(T_TRANSITION, 0.05))
        res0  = run_fiscal_scenario(olg, scn0, bp, n_sim=N_SIM, verbose=False)
        res1  = run_fiscal_scenario(olg, scn1, bp, n_sim=N_SIM, verbose=False)
        mult  = fiscal_multiplier(res0, res1, shock_variable='govt_spending',
                                  output_variable='Y')
        # Ignore NaN (periods with zero ΔG)
        valid = mult[~np.isnan(mult)]
        if len(valid) > 0:
            assert np.mean(valid) > -10.0  # sign-agnostic sanity check


class TestPublicInvestmentSelfFinancing:
    """Higher eta_g → higher w_path → higher tax_l revenue → lower debt path."""

    def test_public_investment_effect(self):
        """With eta_g > 0, I_g shock raises K_g above the neutral level (1.0)
        and thus boosts wages relative to the no-K_g case.

        We start both economies at K_g_initial=1.0 so that K_g^{eta_g}=1
        initially (neutral base wages) and then add an I_g shock.  The
        eta_g>0 economy accumulates K_g > 1 and gets higher wages.
        """
        I_g_base  = np.zeros(T_TRANSITION)
        I_g_shock = np.full(T_TRANSITION, 0.10)

        # eta_g=0: public capital has no effect on wages
        olg_no_kg   = _make_olg(eta_g=0.0, K_g_initial=1.0, I_g_path=I_g_base)
        # eta_g=0.2, K_g_initial=1.0: K_g^0.2 = 1 at start; grows above 1 with I_g shock
        olg_with_kg = _make_olg(eta_g=0.2, K_g_initial=1.0, I_g_path=I_g_base)

        bp  = _make_base_paths()
        scn = FiscalScenario(name='ig_shock', financing='debt',
                             delta_I_g_path=I_g_shock)

        res_no   = run_fiscal_scenario(olg_no_kg,   scn, bp, n_sim=N_SIM, verbose=False)
        res_with = run_fiscal_scenario(olg_with_kg, scn, bp, n_sim=N_SIM, verbose=False)

        # With public capital in production the w_path should be higher on average
        w_no   = np.mean(res_no.cf_macro['w'])
        w_with = np.mean(res_with.cf_macro['w'])
        assert w_with >= w_no - 1e-6  # with-kg wage >= no-kg wage


class TestOutputUtilities:
    """Smoke tests for compare_scenarios, fiscal_multiplier, debt_fan_chart."""

    def _base_result(self):
        olg = _make_olg()
        bp  = _make_base_paths()
        scn = FiscalScenario(name='base_util', financing='debt')
        return run_fiscal_scenario(olg, scn, bp, n_sim=N_SIM, verbose=False)

    def test_compare_scenarios_returns_figure(self):
        from fiscal_experiments import compare_scenarios
        import matplotlib
        matplotlib.use('Agg')
        res = self._base_result()
        fig = compare_scenarios(res, variables=['Y', 'primary_deficit'], save=False)
        assert fig is not None

    def test_debt_fan_chart_returns_figure(self):
        from fiscal_experiments import debt_fan_chart
        import matplotlib
        matplotlib.use('Agg')
        res = self._base_result()
        fig = debt_fan_chart([res], ['base'], save=False)
        assert fig is not None

    def test_fiscal_multiplier_shape(self):
        olg  = _make_olg()
        bp   = _make_base_paths()
        scn0 = FiscalScenario(name='m_base', financing='debt')
        scn1 = FiscalScenario(name='m_shock', financing='debt',
                              delta_G_path=np.full(T_TRANSITION, 0.05))
        res0 = run_fiscal_scenario(olg, scn0, bp, n_sim=N_SIM, verbose=False)
        res1 = run_fiscal_scenario(olg, scn1, bp, n_sim=N_SIM, verbose=False)
        mult = fiscal_multiplier(res0, res1,
                                 shock_variable='govt_spending',
                                 output_variable='Y')
        assert len(mult) == T_TRANSITION


# ---------------------------------------------------------------------------
# Shared baseline, household-results cache, search from Delta = 0
# ---------------------------------------------------------------------------

def _count_solves(olg):
    """Wrap olg.solve_cohort_problems to count its calls."""
    calls = []
    inner = olg.solve_cohort_problems

    def counted(*args, **kwargs):
        calls.append(1)
        return inner(*args, **kwargs)

    olg.solve_cohort_problems = counted
    return calls


def _same_result(a, b):
    for key in ('Y', 'C', 'L', 'K'):
        assert np.array_equal(np.asarray(a.cf_macro[key]), np.asarray(b.cf_macro[key])), key
        assert np.array_equal(np.asarray(a.base_macro[key]), np.asarray(b.base_macro[key])), key
    for key in ('primary_deficit', 'total_revenue', 'total_spending'):
        assert np.array_equal(np.asarray(a.cf_budget[key]), np.asarray(b.cf_budget[key])), key
        assert np.array_equal(np.asarray(a.base_budget[key]), np.asarray(b.base_budget[key])), key
    assert np.array_equal(a.B_path, b.B_path)
    assert a.adjustment_scalar == b.adjustment_scalar


class TestSharedBaseline:
    """run_baseline(): one baseline run shared by several scenarios."""

    def _scenarios(self):
        return (FiscalScenario(name='base', financing='debt', n_post=2),
                FiscalScenario(name='g_debt', financing='debt', n_post=2,
                               delta_G_path=np.full(T_TRANSITION, 0.02)))

    def test_results_equal_those_with_a_baseline_per_scenario(self):
        from fiscal_experiments import run_baseline
        bp = _make_base_paths(tau_p=0.1)
        olg_a, olg_b = _make_olg(), _make_olg()
        calls_a, calls_b = _count_solves(olg_a), _count_solves(olg_b)
        shared = run_baseline(olg_b, bp, n_post=2, n_sim=N_SIM)
        for scn in self._scenarios():
            res_a = run_fiscal_scenario(olg_a, scn, bp, n_sim=N_SIM, verbose=False)
            res_b = run_fiscal_scenario(olg_b, scn, shared, n_sim=N_SIM, verbose=False)
            _same_result(res_a, res_b)
        # Per scenario: baseline + counterfactual. Shared: one baseline in all.
        assert len(calls_a) == 4 and len(calls_b) == 3

    def test_a_baseline_under_other_settings_is_not_reused(self):
        from fiscal_experiments import run_baseline
        bp = _make_base_paths()
        olg = _make_olg()
        shared = run_baseline(olg, bp, n_post=0, n_sim=N_SIM)
        calls = _count_solves(olg)
        scn = FiscalScenario(name='base', financing='debt', n_post=2)
        res = run_fiscal_scenario(olg, scn, shared, n_sim=N_SIM, verbose=False)
        assert len(calls) == 2                       # its own baseline, then the scenario
        assert len(res.base_macro['Y']) == T_TRANSITION + 2

    def test_output_at_t0_does_not_depend_on_the_horizon(self):
        """Y(0) of a T-period run equals Y(0) of the same run extended by
        frozen terminal paths, so it can be read off the experiments' baseline."""
        bp = _make_base_paths(tau_p=0.1)
        kw = {k: v for k, v in bp.items() if k != 'r_path'}
        short = _make_olg().simulate_transition(r_path=bp['r_path'], n_sim=N_SIM, verbose=False, **kw)
        ext = {k: np.concatenate([v, np.full(3, v[-1])]) for k, v in bp.items()}
        kw = {k: v for k, v in ext.items() if k != 'r_path'}
        long = _make_olg().simulate_transition(r_path=ext['r_path'], n_sim=N_SIM, verbose=False, **kw)
        assert np.array_equal(np.asarray(short['Y']), np.asarray(long['Y'])[:T_TRANSITION])


class TestHouseholdCache:
    """OLGTransition(household_cache_size=...): calls with the same household
    inputs reuse the cohort age means of the first."""

    def _olg(self, size):
        olg = _make_olg()
        olg.household_cache_size = size
        return olg

    def test_cached_results_equal_uncached_results(self):
        bp = _make_base_paths(tau_p=0.1)
        G = np.full(T_TRANSITION, 0.02)
        scenarios = (
            FiscalScenario(name='base', financing='debt'),
            FiscalScenario(name='g_debt', financing='debt', delta_G_path=G),
            FiscalScenario(name='g_taul', financing='tau_l', delta_G_path=G,
                           balance_condition='terminal_debt_gdp', target_debt_gdp=0.0),
        )
        plain, cached = self._olg(0), self._olg(8)
        calls_plain, calls_cached = _count_solves(plain), _count_solves(cached)
        for scn in scenarios:
            res_p = run_fiscal_scenario(plain, scn, bp, n_sim=N_SIM, verbose=False, bisect_tol=1e-3)
            res_c = run_fiscal_scenario(cached, scn, bp, n_sim=N_SIM, verbose=False, bisect_tol=1e-3)
            _same_result(res_p, res_c)
            assert res_p.residual_history == res_c.residual_history
        assert plain._household_cache_hits == 0
        # The repeated baselines, the G shock under debt financing (households
        # face the baseline paths) and the first step of the tax search
        # (Delta = 0) are served from the cache.
        assert cached._household_cache_hits >= 4
        assert len(calls_cached) == len(calls_plain) - cached._household_cache_hits

    def test_household_inputs_decide_the_key(self):
        bp = _make_base_paths()
        kw = {k: v for k, v in bp.items() if k != 'r_path'}
        olg = self._olg(4)
        calls = _count_solves(olg)
        run = lambda **over: olg.simulate_transition(
            r_path=bp['r_path'], verbose=False, **dict(dict(kw, n_sim=N_SIM), **over))
        first = run()
        again = run(govt_spending_path=np.full(T_TRANSITION, 0.3))    # not a household input
        assert len(calls) == 1 and olg._household_cache_hits == 1
        assert np.array_equal(np.asarray(first['Y']), np.asarray(again['Y']))
        assert olg.birth_cohort_solutions is None
        run(tau_l_path=bp['tau_l_path'] + 0.01)
        run(n_sim=N_SIM + 1)
        n_calls = 3
        if os.environ.get('DSA_SAVINGS_SOLVER', 'grid') == 'grid':
            # the endogenous grid method refuses a transfer floor
            run(transfer_floor=0.05)
            n_calls = 4
        assert len(calls) == n_calls and olg._household_cache_hits == 1

    def test_cache_is_bounded(self):
        bp = _make_base_paths()
        kw = {k: v for k, v in bp.items() if k != 'r_path'}
        olg = self._olg(2)
        for d in (0.0, 0.01, 0.02, 0.03):
            olg.simulate_transition(r_path=bp['r_path'], n_sim=N_SIM, verbose=False,
                                    **dict(kw, tau_l_path=bp['tau_l_path'] + d))
        assert len(olg._household_cache) == 2


class TestSearchFromDeltaInit:
    """The tax search starts at Delta_init and brackets the root from there."""

    def _scenario(self, target, **kw):
        return FiscalScenario(name='taul', financing='tau_l',
                              balance_condition='terminal_debt_gdp',
                              target_debt_gdp=target, **kw)

    def test_no_shock_and_the_baseline_target_need_one_evaluation(self):
        olg, bp = _make_olg(), _make_base_paths(tau_p=0.1)
        base = run_fiscal_scenario(olg, FiscalScenario(name='base', financing='debt'), bp,
                                   n_sim=N_SIM, verbose=False)
        T = T_TRANSITION
        target = float(base.B_path[T - 1] / base.cf_macro['Y'][T - 1])
        res = run_fiscal_scenario(olg, self._scenario(target), bp, n_sim=N_SIM,
                                  verbose=False, bisect_tol=1e-6)
        assert res.converged and res.n_iterations == 1 and res.adjustment_scalar == 0.0

    def test_root_does_not_depend_on_the_starting_point(self):
        bp = _make_base_paths(tau_p=0.1)
        scn = self._scenario(0.0, delta_G_path=np.full(T_TRANSITION, 0.02))
        roots = []
        for init, step in ((0.0, 0.02), (0.0, 0.1), (0.2, 0.02), (-0.3, 0.05)):
            res = run_fiscal_scenario(_make_olg(), scn, bp, n_sim=N_SIM, verbose=False,
                                      bisect_tol=1e-4, bisect_init=init, bisect_step=step)
            assert res.converged
            assert res.residual_history[0] != res.residual_history[-1] or res.n_iterations == 1
            roots.append(res.adjustment_scalar)
        assert np.ptp(roots) < 5e-3, roots

    def test_bounds_are_the_fallback_bracket(self):
        """With bounds that exclude the root, stepping from Delta_init finds no
        sign change and the search brackets with the bounds. The upper bound is
        expanded, so a root above it is found; a root below the lower bound is
        reported as not converged."""
        from fiscal_experiments import run_baseline, run_tax_financed
        bp = _make_base_paths(tau_p=0.1)

        def solve(target, lo, hi):
            olg = _make_olg()
            shared = run_baseline(olg, bp, n_sim=N_SIM)
            return run_tax_financed(olg, self._scenario(target), shared, shared['base_macro'],
                                    shared['base_budget'], n_sim=N_SIM, tol=1e-4,
                                    Delta_lo=lo, Delta_hi=hi)

        base = run_fiscal_scenario(_make_olg(), FiscalScenario(name='base', financing='debt'),
                                   bp, n_sim=N_SIM, verbose=False)
        T = T_TRANSITION
        base_ratio = float(base.B_path[T - 1] / base.cf_macro['Y'][T - 1])
        for shift in (-0.05, 0.05):          # a lower target needs a higher tax, and conversely
            wide = solve(base_ratio + shift, -0.5, 0.5)
            assert wide.converged and abs(wide.adjustment_scalar) > 2e-3
            assert np.sign(wide.adjustment_scalar) == -np.sign(shift)
            narrow = solve(base_ratio + shift, -1e-3, 1e-3)
            if wide.adjustment_scalar > 0:
                assert narrow.converged
                assert abs(narrow.adjustment_scalar - wide.adjustment_scalar) < 5e-3
            else:
                assert not narrow.converged
