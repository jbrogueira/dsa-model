"""
Tests of the public-investment and health-coverage policy exercises
(docs/POLICY_EXERCISES_PLAN.md section 4): calendar-time paths for coverage
kappa and medical spending, the unanticipated shock in a period after the
start of the transition, the memo lines of medical spending, the health-cut
calibration, and the distributional and welfare outputs. Small economy, one
education group, every cohort split between two retirement ages.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import policy_reference_case as prc
from calibrate import compute_gini, _quantile
from fiscal_experiments import (FiscalScenario, run_baseline, run_fiscal_scenario,
                                health_cut_paths, back_loaded, window_profile)
import distribution_stats as ds

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'tests_data', 'policy_reference_e80b119.npz')
T_S = 3
N_TOT = prc.T_TR + prc.N_POST


def _health_paths(kappa_window=0.5, mu_window=0.85, t_s=T_S, n=3, T=prc.T_TR):
    kap = np.full(T, prc.KAPPA0)
    ms = np.ones(T)
    kap[t_s:t_s + n] = kappa_window
    ms[t_s:t_s + n] = mu_window
    return kap, ms


def _base_paths_with_health():
    bp = prc.base_paths()
    bp['kappa_path'] = np.full(prc.T_TR, prc.KAPPA0)
    bp['m_scale_path'] = np.ones(prc.T_TR)
    return bp


def _baseline(backend='jax', shock_period=T_S, **kw):
    olg = prc.build_economy(backend, **kw)
    bp = run_baseline(olg, _base_paths_with_health(), n_post=prc.N_POST, n_sim=100,
                      shock_period=shock_period)
    return olg, bp


def _run(olg, bp, **scn_kw):
    scn_kw.setdefault('financing', 'debt')
    scn = FiscalScenario(name=scn_kw.pop('name', 'x'), B_initial=0.3, n_post=prc.N_POST, **scn_kw)
    return run_fiscal_scenario(olg, scn, bp, n_sim=100, bisect_tol=1e-6)


def _delta_health(kap_w=0.5, mu_w=0.85, n=3):
    dk = np.zeros(prc.T_TR)
    dm = np.zeros(prc.T_TR)
    dk[T_S:T_S + n] = kap_w - prc.KAPPA0
    dm[T_S:T_S + n] = mu_w - 1.0
    return dk, dm


# ---------------------------------------------------------------------------
# 1, 11a. With the new options off the results are those of the earlier code
# ---------------------------------------------------------------------------

class TestUnchangedWhenOff:
    def test_reference_runs_equal_stored_arrays(self):
        """Test 11, shock_period = 0: baseline, I_g under debt and under a
        labour tax equal the arrays written by the code of e80b119."""
        ref = np.load(REF)
        new = prc.reference_runs('jax')
        for k, v in new.items():
            np.testing.assert_array_equal(v, ref[k], err_msg=k)

    def test_cross_section_unchanged(self):
        """Test 3: the calibration's batched exact cross-section is unchanged
        by the transition-only kernel variants."""
        ref = np.load(REF)
        new = prc.cross_section_reference()
        for k, v in new.items():
            np.testing.assert_array_equal(v, ref[k], err_msg=k)

    @pytest.mark.parametrize('backend', ['numpy', 'jax'])
    def test_constant_health_paths_bit_identical(self, backend):
        """Test 1: paths at the baseline constants give the same results as None."""
        out = []
        for paths in ({}, {'kappa_path': np.full(prc.T_TR, prc.KAPPA0),
                           'm_scale_path': np.ones(prc.T_TR)}):
            olg = prc.build_economy(backend)
            bp = prc.base_paths()
            res = olg.simulate_transition(bp['r_path'], tau_l_path=bp['tau_l_path'],
                                          tau_c_path=bp['tau_c_path'], tau_p_path=bp['tau_p_path'],
                                          tau_k_path=bp['tau_k_path'],
                                          pension_replacement_path=bp['pension_replacement_path'],
                                          I_g_path=bp['I_g_path'], govt_spending_path=bp['G_path'],
                                          n_sim=100, verbose=False, **paths)
            bud = olg.compute_government_budget_path(verbose=False)
            out.append((res, bud))
        for k in ('Y', 'C', 'A', 'L'):
            np.testing.assert_array_equal(out[0][0][k], out[1][0][k])
        for k in ('gov_health', 'transfers', 'primary_deficit', 'tax_l'):
            np.testing.assert_array_equal(out[0][1][k], out[1][1][k])


# ---------------------------------------------------------------------------
# 2. Backends agree with nonconstant paths
# ---------------------------------------------------------------------------

def test_backends_agree_with_health_paths():
    kap, ms = _health_paths()
    res = {}
    for backend in ('numpy', 'jax'):
        olg = prc.build_economy(backend)
        bp = prc.base_paths()
        r = olg.simulate_transition(bp['r_path'], tau_l_path=bp['tau_l_path'],
                                    tau_c_path=bp['tau_c_path'], tau_p_path=bp['tau_p_path'],
                                    tau_k_path=bp['tau_k_path'],
                                    pension_replacement_path=bp['pension_replacement_path'],
                                    I_g_path=bp['I_g_path'], govt_spending_path=bp['G_path'],
                                    kappa_path=kap, m_scale_path=ms, n_sim=100, verbose=False)
        res[backend] = (r, olg.compute_government_budget_path(verbose=False))
    for k in ('Y', 'C', 'A', 'L'):
        np.testing.assert_allclose(res['numpy'][0][k], res['jax'][0][k], rtol=1e-10, atol=1e-12)
    for k in ('gov_health', 'oop_health', 'transfers', 'tax_l'):
        np.testing.assert_allclose(res['numpy'][1][k], res['jax'][1][k], rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 4, 5. Deduplication and the household cache see the new inputs
# ---------------------------------------------------------------------------

def test_dedup_key_includes_health_paths():
    from lifecycle_jax import LifecycleModelJAX
    from olg_transition import OLGTransition
    kap, ms = _health_paths(T=prc.T)
    m0 = LifecycleModelJAX(prc.lifecycle_config(), verbose=False)
    m1 = LifecycleModelJAX(prc.lifecycle_config(kappa_path=kap), verbose=False)
    m2 = LifecycleModelJAX(prc.lifecycle_config(m_scale_path=ms), verbose=False)
    keys = {OLGTransition._solve_inputs_key(m) for m in (m0, m1, m2)}
    assert len(keys) == 3


def test_household_cache_sees_health_paths_and_shock_period():
    olg = prc.build_economy('jax', household_cache_size=8)
    bp = prc.base_paths()
    pre_tp = {k: bp.get(k) for k in ('r_path', 'tau_l_path', 'tau_c_path', 'tau_p_path',
                                     'tau_k_path', 'pension_replacement_path')}
    kw = dict(tau_l_path=bp['tau_l_path'], tau_c_path=bp['tau_c_path'],
              tau_p_path=bp['tau_p_path'], tau_k_path=bp['tau_k_path'],
              pension_replacement_path=bp['pension_replacement_path'],
              I_g_path=bp['I_g_path'], govt_spending_path=bp['G_path'], n_sim=100,
              verbose=False, pre_transition_paths=pre_tp)
    kap, ms = _health_paths()
    olg.simulate_transition(bp['r_path'], **kw)
    olg.simulate_transition(bp['r_path'], kappa_path=kap, **kw)
    olg.simulate_transition(bp['r_path'], m_scale_path=ms, **kw)
    olg.simulate_transition(bp['r_path'], shock_period=T_S, **kw)
    assert olg._household_cache_hits == 0
    olg.simulate_transition(bp['r_path'], m_scale_path=ms, **kw)
    assert olg._household_cache_hits == 1


# ---------------------------------------------------------------------------
# 6, 7. Medical spending in the accounts; the health-cut calibration
# ---------------------------------------------------------------------------

def test_booked_medical_spending():
    kap, ms = _health_paths(mu_window=1.25)
    out = {}
    for name, m_path in (('one', np.ones(prc.T_TR)), ('scaled', ms)):
        olg = prc.build_economy('jax')
        bp = prc.base_paths()
        olg.simulate_transition(bp['r_path'], tau_l_path=bp['tau_l_path'],
                                tau_c_path=bp['tau_c_path'], tau_p_path=bp['tau_p_path'],
                                tau_k_path=bp['tau_k_path'],
                                pension_replacement_path=bp['pension_replacement_path'],
                                I_g_path=bp['I_g_path'], govt_spending_path=bp['G_path'],
                                kappa_path=kap, m_scale_path=m_path, n_sim=100, verbose=False)
        out[name] = olg.compute_government_budget_path(verbose=False)
    b = out['scaled']
    np.testing.assert_allclose(b['medical_total'], b['gov_health'] / kap, rtol=1e-12)
    np.testing.assert_allclose(b['gov_health'] + b['oop_health'], b['medical_total'], rtol=1e-12)
    np.testing.assert_allclose(b['kappa'], kap, rtol=0, atol=0)
    # M_t does not depend on household choices: it scales with the multiplier
    np.testing.assert_allclose(out['scaled']['medical_total'],
                               out['one']['medical_total'] * ms, rtol=1e-10)


def test_health_cut_paths_hits_targets_and_special_cases():
    T = 12
    Y = np.linspace(1.0, 1.1, T)
    kappa0 = 0.63
    gov = 0.054 * Y * np.linspace(1.0, 1.05, T)
    base = {'base_macro': {'Y': Y}, 'base_budget': {'gov_health': gov},
            'kappa_path': np.full(T, kappa0)}
    dg, dh = -0.0129, 0.0026
    p = health_cut_paths(base, dg, dh, t_s=3, n_years=5, T=T)
    win = slice(3, 8)
    M = gov / kappa0
    g_new = (kappa0 + p['delta_kappa_path']) * (1 + p['delta_m_scale_path']) * M
    o_new = (1 - kappa0 - p['delta_kappa_path']) * (1 + p['delta_m_scale_path']) * M
    g0 = np.mean(gov[win] / Y[win])
    o0 = g0 * (1 - kappa0) / kappa0
    assert abs(np.mean(g_new[win] / Y[win]) - (g0 + dg)) < 1e-14
    assert abs(np.mean(o_new[win] / Y[win]) - (o0 + dh)) < 1e-14
    assert np.all(p['delta_kappa_path'][:3] == 0) and np.all(p['delta_kappa_path'][8:] == 0)
    assert np.all(p['delta_m_scale_path'][:3] == 0) and np.all(p['delta_m_scale_path'][8:] == 0)
    # kappa-only cut: household spending rises one for one
    q = health_cut_paths(base, dg, -dg, t_s=3, n_years=5, T=T)
    assert abs(q['mu1'] - 1.0) < 1e-14
    # m-only cut: both shares fall in proportion
    r = health_cut_paths(base, dg, dg * (1 - kappa0) / kappa0, t_s=3, n_years=5, T=T)
    assert abs(r['kappa1'] - kappa0) < 1e-14


# ---------------------------------------------------------------------------
# 8, 9, 10. Statistics, weights and welfare
# ---------------------------------------------------------------------------

def test_weighted_statistics_equal_replicated_sample():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=40)
    w = rng.integers(1, 6, size=40).astype(float)
    rep = np.repeat(x, w.astype(int))
    assert abs(compute_gini(x, w) - compute_gini(rep)) < 1e-12
    assert abs(compute_gini(x, w) - compute_gini(rep, np.ones(len(rep)))) < 1e-12
    for q in (0.1, 0.5, 0.9):
        assert _quantile(x, q, w) == _quantile(rep, q, np.ones(len(rep)))
    top = np.sort(rep)[int(round(0.9 * len(rep))):]
    if (0.9 * len(rep)) % 1 == 0:
        assert abs(ds.top_share(x, w) - top.sum() / rep.sum()) < 1e-12


def test_cross_section_weights_and_aggregates():
    """Test 9: extract() checks that the weights sum to one and that the
    weighted means of assets and consumption are A_t and C_t."""
    olg, bp = _baseline('jax')
    _run(olg, bp, name='b', shock_period=T_S)
    ext = ds.extract(olg, list(range(prc.T_TR)), t_s=T_S, newborn_bps=[T_S + 1])
    np.testing.assert_allclose(ext['weight_sum'], 1.0, rtol=1e-12)
    np.testing.assert_allclose(ext['mean_assets'], olg.K_path[:prc.T_TR], rtol=1e-10)


def test_welfare_zero_when_counterfactual_is_baseline():
    """Test 10: lambda = 0 for a zero shock in t_s (JAX, deduplicated models)."""
    olg, bp = _baseline('jax')
    periods = list(range(prc.T_TR))
    _run(olg, bp, name='b', shock_period=T_S)
    eb = ds.extract(olg, periods, t_s=T_S, newborn_bps=range(T_S + 1, prc.T_TR))
    _run(olg, bp, name='z', shock_period=T_S, delta_I_g_path=np.zeros(prc.T_TR))
    ec = ds.extract(olg, periods, t_s=T_S, newborn_bps=range(T_S + 1, prc.T_TR))
    w = ds.welfare_summary(eb, ec, olg.education_shares)
    assert max(abs(v) for v in w['cohort'].values()) < 1e-12
    assert max(abs(v) for v in w['quintile']) < 1e-12


def test_cev_log_utility_toy():
    """Scaling consumption by (1 + x) at every age raises V by log(1+x) D(j),
    and the measure returns x."""
    x, D = 0.03, 7.3
    rng = np.random.default_rng(1)
    V = rng.normal(size=20)
    mu = rng.random(20)
    rec = {('medium', 5, 0): {'j': 2, 'V': V, 'mass': mu, 'share': 1.0, 'D': D}}
    rec_cf = {('medium', 5, 0): {'j': 2, 'V': V + np.log1p(x) * D, 'mass': mu,
                                 'share': 1.0, 'D': D}}
    out = ds.cohort_cev({'alive': rec, 'newborn': {}}, {'alive': rec_cf, 'newborn': {}},
                        {'medium': 1.0})
    assert abs(out[5] - x) < 1e-12


# ---------------------------------------------------------------------------
# 11b, 12, 13, 14. The shock in t_s
# ---------------------------------------------------------------------------

class TestShockPeriod:
    def test_before_the_shock_equals_baseline(self):
        """Test 11, shock_period = 3: aggregates, budget lines and every
        cohort's 2026 asset distribution equal the baseline's."""
        olg, bp = _baseline('jax')
        res_b = _run(olg, bp, name='b', shock_period=T_S)
        pb = ds.cohort_panels(olg, [T_S])
        dk, dm = _delta_health()
        res_c = _run(olg, bp, name='h', shock_period=T_S, delta_kappa_path=dk,
                     delta_m_scale_path=dm,
                     delta_I_g_path=np.r_[np.zeros(T_S), np.full(prc.T_TR - T_S, 0.02)])
        pc = ds.cohort_panels(olg, [T_S])
        for k in ('Y', 'C', 'L', 'w'):
            np.testing.assert_array_equal(res_c.cf_macro[k][:T_S], res_b.cf_macro[k][:T_S])
        np.testing.assert_array_equal(res_c.cf_macro['A'][:T_S + 1], res_b.cf_macro['A'][:T_S + 1])
        for k in ('primary_deficit', 'gov_health', 'tax_l', 'transfers', 'pension'):
            np.testing.assert_array_equal(res_c.cf_budget[k][:T_S], res_b.cf_budget[k][:T_S])
        np.testing.assert_array_equal(res_c.B_path[:T_S + 1], res_b.B_path[:T_S + 1])
        assert set(pb) == set(pc)
        assert any(k[2] == 1 for k in pb) and any(0 <= k[1] < T_S for k in pb)
        for key in pb:
            if not pb[key][1][0]:
                continue
            np.testing.assert_array_equal(pb[key][3][0], pc[key][3][0], err_msg=str(key))
            np.testing.assert_array_equal(pb[key][2][0][0], pc[key][2][0][0], err_msg=str(key))
        # and the shock moves things from t_s on
        assert np.max(np.abs(res_c.cf_budget['gov_health'][T_S:] -
                             res_b.cf_budget['gov_health'][T_S:])) > 1e-4

    def test_zero_shock_equals_baseline_everywhere(self):
        """Test 12."""
        olg, bp = _baseline('jax')
        res_b = _run(olg, bp, name='b', shock_period=T_S)
        res_z = _run(olg, bp, name='z', shock_period=T_S, delta_I_g_path=np.zeros(prc.T_TR))
        for k in ('Y', 'C', 'A', 'L'):
            np.testing.assert_array_equal(res_z.cf_macro[k], res_b.cf_macro[k])
        np.testing.assert_array_equal(res_z.B_path, res_b.B_path)

    def test_numpy_backend_stitching(self):
        olg, bp = _baseline('numpy')
        res_b = _run(olg, bp, name='b', shock_period=T_S)
        res_c = _run(olg, bp, name='c', shock_period=T_S,
                     delta_I_g_path=np.r_[np.zeros(T_S), np.full(prc.T_TR - T_S, 0.02)])
        np.testing.assert_array_equal(res_c.cf_macro['A'][:T_S + 1], res_b.cf_macro['A'][:T_S + 1])
        assert np.max(np.abs(res_c.cf_macro['A'][T_S + 1:] - res_b.cf_macro['A'][T_S + 1:])) > 0

    def test_adjustment_profiles_zero_before_and_outside(self):
        """Test 13."""
        olg, bp = _baseline('jax')
        res_b = _run(olg, bp, name='b', shock_period=T_S)
        T_bal = res_b.T_balance
        target = float(res_b.B_path[T_bal - 1] / res_b.cf_macro['Y'][T_bal - 1])
        dk, dm = _delta_health()
        for psi in (back_loaded(N_TOT, T_S), window_profile(N_TOT, T_S, 3)):
            res = _run(olg, bp, name='t', shock_period=T_S, financing='tau_l',
                       balance_condition='terminal_debt_gdp', target_debt_gdp=target,
                       delta_kappa_path=dk, delta_m_scale_path=dm, adjustment_profile=psi)
            assert res.converged
            assert np.all(res.adjustment_path[:T_S] == 0.0)
            if psi[-1] == 0.0:
                assert np.all(res.adjustment_path[T_S + 3:] == 0.0)
            np.testing.assert_array_equal(res.cf_macro['A'][:T_S + 1], res_b.cf_macro['A'][:T_S + 1])

    def test_refuses_bequest_loop(self):
        """Test 14."""
        olg = prc.build_economy('jax')
        bp = prc.base_paths()
        with pytest.raises(ValueError, match='shock_period'):
            olg.simulate_transition(bp['r_path'], n_sim=100, verbose=False,
                                    shock_period=T_S, recompute_bequests=True,
                                    pre_transition_paths={'r_path': bp['r_path']})


# ---------------------------------------------------------------------------
# 15, 16. Driver, evaluator and report on a health run
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tiny_driver_run(tmp_path_factory):
    out = tmp_path_factory.mktemp('policy')
    cmd = [sys.executable, os.path.join(HERE, 'run_fiscal_figures.py'), '--backend', 'jax',
           '--tiny', '--shock', 'Ig,health', '--scenarios', 'debt,tau_l_debt,tau_l_window',
           '--output-dir', str(out)]
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return out


def test_evaluator_on_health_json(tiny_driver_run):
    """Tests 15 and 16: the evaluator runs on a JSON with a health block and
    the goods-market check passes with the booked medical spending."""
    path = os.path.join(tiny_driver_run, 'fiscal_results.json')
    with open(path) as f:
        res = json.load(f)
    assert 'health' in res and 'Ig' in res and 'G' not in res
    assert 'nfa_constrained' not in res['health']
    proc = subprocess.run([sys.executable, os.path.join(HERE, 'eval_fiscal_results.py'),
                           '--input', path],
                          cwd=HERE, capture_output=True, text=True, timeout=600)
    out = proc.stdout
    assert 'Traceback' not in proc.stderr, proc.stderr[-3000:]
    goods = [ln for ln in out.splitlines() if 'goods_market' in ln and 'health' in ln]
    assert goods and all('FAIL' not in ln for ln in goods), out[-3000:]


def test_report_figures_without_g_or_nfa(tiny_driver_run):
    path = os.path.join(tiny_driver_run, 'fiscal_results.json')
    proc = subprocess.run([sys.executable, os.path.join(HERE, 'reports', 'fiscal_figures.py'),
                           '--results', path, '--baseline', '', '--out-dir', str(tiny_driver_run)],
                          cwd=HERE, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert os.path.exists(os.path.join(tiny_driver_run, 'fiscal_health.pdf'))
