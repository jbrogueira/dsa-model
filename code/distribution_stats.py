"""
Distributional outputs of the policy experiments (POLICY_EXERCISES_PLAN.md
sections 3.2-3.4): the cross-section of households in a calendar period,
means by age group, inequality, and the consumption-equivalent welfare
change by birth cohort and by income quintile.

Everything is read off the cohort models that the last simulate_transition()
call of an OLGTransition left in birth_cohort_solutions and
birth_cohort_later, on the exact distribution over states. A household state
of the cohort born in period b, at age j = t - b in period t, carries the
weight

    W_t(j) * (education share) * (retirement-part share) * mass(state),

where W_t is _aggregation_weights(t) and mass is the exact mass at the start
of age j, which already includes the cohort's survival to j. The weights of a
period sum to one, and the weighted means of assets and consumption are the
transition's A_t and C_t.

Call extract() right after the run whose distribution is wanted: the next run
replaces the cohort models. The run must keep its cohort models
(household_cache_size = 0) and its value functions
(jax_policies_on_device = False).
"""
from __future__ import annotations

import numpy as np

from calibrate import compute_gini, _quantile

# Elements of the 23-element panel (simulate() layout) used here
_FIELDS = {'a': 0, 'c': 1, 'eff_y': 5, 'employed': 6, 'ui': 7, 'oop': 9,
           'tax_l': 12, 'tax_p': 13, 'tax_k': 14, 'pension': 16, 'retired': 17,
           'l': 18, 'transfer': 22}
_FIELD_IDX = tuple(_FIELDS.values())

# Model age 0 is real age 25
AGE_GROUPS = (('25-44', 0, 20), ('45-64', 20, 40), ('65+', 40, 10_000))

# Measures of the inequality block: (key, population, label)
INEQ_MEASURES = (
    ('labour_income', 'employed', 'Labour income of the employed'),
    ('disp_income', 'all', 'Disposable income'),
    ('disp_income_net_health', 'all', 'Disposable income less household health spending'),
    ('disp_income_calib', 'all', 'Disposable income, calibration definition'),
    ('consumption', 'all', 'Consumption'),
    ('wealth', 'all', 'Wealth'),
)
RATIO_MEASURES = ('labour_income', 'disp_income', 'disp_income_net_health',
                  'disp_income_calib', 'consumption')


def reporting_periods(T_transition: int, base_year: int) -> list:
    """Periods t of the distributional outputs: every year 2023-2040, then
    2045-2070 every five years, within the transition."""
    years = list(range(2023, 2041)) + list(range(2045, 2071, 5))
    return [y - int(base_year) for y in years if 0 <= y - int(base_year) < int(T_transition)]


# ---------------------------------------------------------------------------
# Cohort models and their state-level panels
# ---------------------------------------------------------------------------

def cohort_parts(olg) -> list:
    """[(key, model, part_share)] for every cohort model of the last run, key =
    (edu_type, birth_period, part); part 1 is the later-retiring part of a
    split cohort, with share later_share[bp], and part 0 has the rest."""
    if olg.birth_cohort_solutions is None:
        raise RuntimeError("the last run kept no cohort models (served from the household "
                           "cache); run with household_cache_size = 0")
    later = getattr(olg, 'birth_cohort_later', None) or {}
    out = []
    for edu, by_bp in olg.birth_cohort_solutions.items():
        for bp, m in by_bp.items():
            has_later = bp in later.get(edu, {})
            sh = float(olg.later_share[bp]) if has_later else 0.0
            out.append(((edu, int(bp), 0), m, 1.0 - sh))
            if has_later:
                out.append(((edu, int(bp), 1), later[edu][bp], sh))
    return out


def _np_model(m):
    return getattr(m, '_np_model', m)


def _panels_numpy(models_rows):
    out = {}
    for key, (m, rows) in models_rows.items():
        panel, mass = m.exact_panel(rows)
        vals = np.stack([np.asarray(panel[i], dtype=float) for i in _FIELD_IDX], axis=1)
        out[key] = (vals, np.asarray(mass, dtype=float))
    return out


def _panels_jax(olg, models_rows, chunk=None):
    """State-level panels of many cohort models in batched calls, one per
    chunk of models that share an education group, a retirement age and the
    retirement scalars. Rows of every model must have the same length."""
    import jax.numpy as jnp
    from lifecycle_jax import _exact_panel_jax_batched_tr, _exact_panel_jax_batched_tr_pyc

    groups = {}
    for key, (m, rows) in models_rows.items():
        g = (key[0], int(m.retirement_age), float(m.pension_avg_weight),
             float(m.mean_kappa_working))
        groups.setdefault(g, []).append(key)
    chunk = int(chunk or olg.jax_sim_chunk_size or 32)
    fidx = jnp.asarray(_FIELD_IDX)
    out = {}
    for keys in groups.values():
        ref = models_rows[keys[0]][0]
        per_cohort_py = bool(ref.P_y_age_health)
        kernel = _exact_panel_jax_batched_tr_pyc if per_cohort_py else _exact_panel_jax_batched_tr
        if float(ref.transfer_floor) > 0.0 and int(getattr(ref.config, 'schooling_years', 0) or 0) > 0:
            raise NotImplementedError("transfer floor with schooling years")
        initial_dist = jnp.array(ref._np_model._initial_distribution())
        ones_surv = np.ones((ref.T, olg.n_h))
        # Every batch padded to the chunk length: one compiled kernel per
        # retirement age whatever the size of the group.
        n = chunk
        for start in range(0, len(keys), n):
            sel = keys[start:start + n]
            padded = sel + [sel[-1]] * (n - len(sel))
            ms = [models_rows[k][0] for k in padded]
            rows = jnp.asarray(np.stack([models_rows[k][1] for k in padded]).astype(np.int32))

            def stack(f):
                return jnp.stack([jnp.asarray(f(m)) for m in ms])
            if per_cohort_py:
                init_arg = stack(lambda m: m._np_model._initial_distribution())
                py_arg = stack(lambda m: m.P_y_4d)
            else:
                init_arg, py_arg = initial_dist, (ref.P_y_4d if ref.P_y_age_health else None)
            _, panel, mass = kernel(
                stack(lambda m: m.a_policy_alpha), stack(lambda m: m.c_policy_alpha),
                stack(lambda m: m.l_policy_alpha),
                ref.a_grid, ref.y_grid, ref.h_grid, stack(lambda m: m.m_grid),
                ref.P_y_2d, ref.P_h,
                stack(lambda m: m.w_path), jnp.array([m.w_at_retirement for m in ms]),
                stack(lambda m: m.tau_c_path), stack(lambda m: m.tau_l_path),
                stack(lambda m: m.tau_p_path), stack(lambda m: m.tau_k_path),
                stack(lambda m: m.r_path), stack(lambda m: m.pension_replacement_path),
                ref.ui_replacement_rate, stack(lambda m: m.kappa_path),
                ref.retirement_age, ref.T, ref.current_age,
                init_arg, ref.alpha_grid,
                ref.pension_min_floor, ref.tax_progressive,
                ref.tax_kappa_hsv, ref.tax_eta,
                ref.P_y_age_health, py_arg,
                stack(lambda m: m.survival_probs if m.survival_probs is not None else ones_surv),
                ref.wage_age_profile,
                ref.pension_avg_weight, ref.mean_kappa_working, ref.mean_y_employed,
                ref.trend_growth,
                ref.transfer_floor,
                jnp.array([float(m.bequest_lumpsum) for m in ms]),
                stack(lambda m: m.lump_sum_path),
                False,
                rows,
                ref.ui_eligibility_prob,
            )
            vals = np.asarray(panel[:, :, fidx, :])        # (C, R, F, S)
            mass = np.asarray(mass)                         # (C, R, S)
            for i, k in enumerate(sel):
                out[k] = (vals[i], mass[i])
    return out


def cohort_panels(olg, periods, extra_rows=None):
    """{key: (rows, alive, vals (R, F, S), mass (R, S))} for every cohort
    model alive in at least one of *periods*. Row r of a cohort born in b is
    its age periods[r] - b (clipped to the life span; alive[r] says whether
    the cohort is alive then). *extra_rows* {key: age} appends one row per
    model (the welfare ages)."""
    periods = np.asarray(periods, dtype=int)
    T = int(olg.T)
    models_rows, meta = {}, {}
    for key, m, share in cohort_parts(olg):
        ages = periods - key[1]
        alive = (ages >= 0) & (ages < T)
        extra = (extra_rows or {}).get(key)
        if not alive.any() and extra is None:
            continue
        rows = np.clip(ages, 0, T - 1)
        rows = np.append(rows, extra if extra is not None else 0)
        models_rows[key] = (m, rows)
        meta[key] = (rows, alive, share)
    if olg.backend == 'jax':
        panels = _panels_jax(olg, models_rows)
    else:
        panels = _panels_numpy(models_rows)
    return {k: (meta[k][0], meta[k][1], panels[k][0], panels[k][1], meta[k][2], models_rows[k][0])
            for k in panels}


def _derived(vals, model, age):
    """Per-state measures at one age from the panel fields (F, S)."""
    f = {k: vals[i] for i, k in enumerate(_FIELDS)}
    npm = _np_model(model)
    r_t = float(np.asarray(npm.r_path)[age])
    lump = float(np.asarray(npm.lump_sum_path)[age])
    labour = f['eff_y'] - f['ui']
    disp = (f['eff_y'] + f['pension'] + r_t * f['a'] - f['tax_k'] + lump + f['transfer']
            - f['tax_l'] - f['tax_p'])
    employed = f['employed'] > 0.5
    return {
        'assets': f['a'], 'consumption': f['c'], 'wealth': f['a'],
        'labour_income': labour,
        'disp_income': disp,
        'disp_income_net_health': disp - f['oop'],
        'disp_income_calib': f['eff_y'] + f['pension'] - f['tax_l'] - f['tax_p'],
        'hours': np.where(employed, f['l'], 0.0),
        'employed': employed.astype(float),
    }


def cross_section(olg, panels, r, t):
    """The cross-section of period t, reporting row r of *panels*: {measure:
    values} by state, the weights and the model age of each state."""
    W = np.asarray(olg._aggregation_weights(int(t)), dtype=float)
    parts = {k: [] for k in ('assets', 'consumption', 'wealth', 'labour_income', 'disp_income',
                             'disp_income_net_health', 'disp_income_calib', 'hours', 'employed')}
    weights, ages = [], []
    for key, (rows, alive, vals, mass, share, model) in panels.items():
        if not alive[r]:
            continue
        age = int(rows[r])
        assert key[1] + age == int(t)
        d = _derived(vals[r], model, age)
        w = W[age] * float(olg.education_shares[key[0]]) * share * mass[r]
        keep = w > 0
        for k in parts:
            parts[k].append(d[k][keep])
        weights.append(w[keep])
        ages.append(np.full(int(keep.sum()), age))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out['weight'] = np.concatenate(weights)
    out['age'] = np.concatenate(ages)
    out['t'] = t
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def top_share(x, w, q=0.9):
    """Share of the total of x held by the top 1 - q of the weighted population."""
    order = np.argsort(x, kind='stable')
    x, w = np.asarray(x, float)[order], np.asarray(w, float)[order]
    cum = np.cumsum(w) / w.sum()
    prev = np.concatenate([[0.0], cum[:-1]])
    # fraction of each state's mass above the q-th population quantile
    frac = np.clip((cum - q) / np.maximum(cum - prev, 1e-300), 0.0, 1.0)
    total = np.sum(w * x)
    return float(np.sum(frac * w * x) / total) if total != 0 else float('nan')


def inequality(cs):
    """Gini of every measure, P90/P10 and P90/P50 on positive values with the
    share of non-positive values, and the top-10% wealth share."""
    out = {}
    w_all = cs['weight']
    emp = cs['employed'] > 0.5
    for key, pop, _ in INEQ_MEASURES:
        x, w = cs[key], w_all
        if pop == 'employed':
            x, w = x[emp], w[emp]
        rec = {'gini': float(compute_gini(x, w)),
               'share_nonpositive': float(w[x <= 0].sum() / w.sum())}
        if key in RATIO_MEASURES:
            pos = x > 0
            p10, p50, p90 = (_quantile(x[pos], q, w[pos]) for q in (0.1, 0.5, 0.9))
            rec.update(p90_p10=float(p90 / p10), p90_p50=float(p90 / p50))
        out[key] = rec
    out['wealth']['top10_share'] = top_share(cs['wealth'], w_all, 0.9)
    return out


def age_group_means(cs):
    """Means of consumption, assets and disposable income over all households,
    and of hours and labour income over the employed, by age group."""
    out = {}
    w, emp = cs['weight'], cs['employed'] > 0.5
    for name, lo, hi in AGE_GROUPS:
        g = (cs['age'] >= lo) & (cs['age'] < hi)
        ge = g & emp
        with np.errstate(invalid='ignore', divide='ignore'):
            rec = {k: float(np.sum(w[g] * cs[k][g]) / np.sum(w[g]))
                   for k in ('consumption', 'assets', 'disp_income')}
        if ge.any():
            rec['hours'] = float(np.sum(w[ge] * cs['hours'][ge]) / np.sum(w[ge]))
            rec['labour_income'] = float(np.sum(w[ge] * cs['labour_income'][ge]) / np.sum(w[ge]))
        else:
            rec['hours'] = rec['labour_income'] = float('nan')
        rec['population'] = float(np.sum(w[g]))
        if not g.any():
            rec = {k: float('nan') for k in rec}
        out[name] = rec
    return out


def aggregate_hours(cs):
    """Hours of the employed over all employed households, and the employed
    share of the living population."""
    w, emp = cs['weight'], cs['employed'] > 0.5
    return {'hours_employed': float(np.sum(w[emp] * cs['hours'][emp]) / np.sum(w[emp])),
            'employment_share': float(np.sum(w[emp]) / np.sum(w))}


# ---------------------------------------------------------------------------
# Welfare
# ---------------------------------------------------------------------------

def annuity_factor(model, j):
    """D(j) = sum_{k=j}^{T-1} beta^{k-j} prod_{i=j}^{k-1} s(i): the change in
    lifetime utility from age j of a one-unit rise in log consumption at every
    age (log utility). Survival is the cohort's own schedule."""
    npm = _np_model(model)
    T = int(npm.T)
    surv = (np.ones(T) if npm.survival_probs is None
            else np.asarray(npm.survival_probs, dtype=float)[:, 0])
    beta = float(npm.beta)
    D, disc = 0.0, 1.0
    for k in range(j, T):
        D += disc
        disc *= beta * surv[k]
    return D


def _V_slice(model, j):
    V = getattr(model, 'V_alpha', None)
    if V is None:
        raise RuntimeError("the cohort model has no value function (jax_policies_on_device?)")
    return np.asarray(V)[:, j].reshape(-1)


def welfare_inputs(olg, panels, t_s, r_ts, newborn_bps):
    """The value functions and masses the welfare measures need, from the
    models of the last run.

    Cohorts alive at t_s (row r_ts of *panels*): V at age t_s - b and the
    mass at the start of that age. Cohorts born after t_s (*newborn_bps*):
    V at age 0 and the initial distribution."""
    gamma = float(getattr(olg.lifecycle_config, 'gamma', 1.0))
    if abs(gamma - 1.0) > 1e-12:
        raise NotImplementedError("the consumption-equivalent measure assumes log utility")
    alive = {}
    for key, (rows, alv, vals, mass, share, model) in panels.items():
        if not alv[r_ts]:
            continue
        j = int(rows[r_ts])
        if t_s - key[1] != j:
            raise RuntimeError(f"row {r_ts} of cohort {key} is age {j}, not {t_s - key[1]}")
        alive[key] = {'j': j, 'V': _V_slice(model, j), 'mass': mass[r_ts], 'share': share,
                      'D': annuity_factor(model, j)}
    newborn = {}
    newborn_bps = set(int(b) for b in newborn_bps)
    for key, model, share in cohort_parts(olg):
        if key[1] in newborn_bps:
            mu = np.asarray(_np_model(model)._initial_distribution(), dtype=float).reshape(-1)
            newborn[key] = {'j': 0, 'V': _V_slice(model, 0), 'mass': mu, 'share': share,
                            'D': annuity_factor(model, 0)}
    return {'alive': alive, 'newborn': newborn}


def _check_state_order(model, vals_row):
    """The flattening of V_alpha[:, j] must follow the panel's state order over
    (alpha, a, y, h, y_last): assets vary along the second axis."""
    npm = _np_model(model)
    shape = (int(npm.n_alpha), int(npm.n_a), int(npm.n_y), int(npm.n_h), int(npm.n_y))
    a_state = np.broadcast_to(np.asarray(npm.a_grid)[None, :, None, None, None], shape).reshape(-1)
    if not np.allclose(vals_row[_FIELD_IDX.index(0)], a_state):
        raise RuntimeError("panel state order differs from the value function's")


def cohort_cev(base, cf, education_shares, W_ts=None):
    """lambda_c = exp[(E V_cf - E V_base) / D_c] - 1 by birth period, the
    expectation over the cohort's distribution at the evaluation age, mixed
    over education groups and retirement parts (weighted by their mass).
    With W_ts also returns the population of each cohort alive in t_s
    (zero for later entrants)."""
    out, pop = {}, {}
    for group in ('alive', 'newborn'):
        by_bp = {}
        for key, b in base[group].items():
            c = cf[group][key]
            wt = float(education_shares[key[0]]) * b['share']
            mu = b['mass']
            rec = by_bp.setdefault(key[1], [0.0, 0.0, 0.0, b['D']])
            rec[0] += wt * float(np.sum(mu * b['V']))
            rec[1] += wt * float(np.sum(mu * c['V']))
            rec[2] += wt * float(np.sum(mu))
        for bp, (eb, ec, m, D) in by_bp.items():
            out[bp] = float(np.expm1((ec - eb) / m / D))
            if W_ts is not None:
                j = next(b['j'] for k, b in base[group].items() if k[1] == bp)
                pop[bp] = float(W_ts[j] * m) if group == 'alive' else 0.0
    out = dict(sorted(out.items()))
    return out if W_ts is None else (out, {k: pop[k] for k in out})


def quintile_cev(base, cf, base_cs_ts, education_shares, W_ts, n_q=5):
    """State-level lambda(s) = exp[(V_cf(s) - V_base(s)) / D] - 1 averaged
    within quintiles of baseline disposable income in t_s over all cohorts
    alive in t_s, weighted by the cross-section weights."""
    lam, inc, wts = [], [], []
    for key, b in base['alive'].items():
        c = cf['alive'][key]
        w = W_ts[b['j']] * float(education_shares[key[0]]) * b['share'] * b['mass']
        keep = w > 0
        lam.append(np.expm1((c['V'] - b['V'])[keep] / b['D']))
        inc.append(b['income'][keep])
        wts.append(w[keep])
    lam, inc, wts = (np.concatenate(x) for x in (lam, inc, wts))
    order = np.argsort(inc, kind='stable')
    lam, inc, wts = lam[order], inc[order], wts[order]
    cum = np.cumsum(wts) / wts.sum()
    mid = cum - 0.5 * wts / wts.sum()
    q = np.minimum((mid * n_q).astype(int), n_q - 1)
    return [float(np.sum(wts[q == k] * lam[q == k]) / np.sum(wts[q == k])) for k in range(n_q)]


# ---------------------------------------------------------------------------
# One call per run
# ---------------------------------------------------------------------------

def extract(olg, periods, t_s=None, newborn_bps=(), check_aggregates=True):
    """Distributional outputs of the last run of *olg*.

    Returns {'periods': [...], 'inequality': [...], 'age_groups': [...],
    'hours': [...], 'weight_sum': [...]} by reporting period, and with t_s
    (which must be in *periods*) the welfare inputs under 'welfare' (value
    functions and masses, not serialisable) and the baseline income of the
    t_s cross-section by state under the same keys.
    """
    periods = [int(t) for t in periods]
    panels = cohort_panels(olg, periods)
    out = {'periods': periods, 'inequality': [], 'age_groups': [], 'hours': [],
           'weight_sum': [], 'mean_assets': [], 'mean_consumption': []}
    for r, t in enumerate(periods):
        cs = cross_section(olg, panels, r, t)
        out['inequality'].append(inequality(cs))
        out['age_groups'].append(age_group_means(cs))
        out['hours'].append(aggregate_hours(cs))
        out['weight_sum'].append(float(cs['weight'].sum()))
        out['mean_assets'].append(float(np.sum(cs['weight'] * cs['assets'])))
        out['mean_consumption'].append(float(np.sum(cs['weight'] * cs['consumption'])))
    # Under simulated aggregation the transition's means carry sampling error
    # that the exact cross-section does not.
    if check_aggregates and getattr(olg, 'aggregation', 'simulation') == 'exact':
        A = np.asarray(olg.K_path)[periods]
        C = np.asarray(olg.C_path)[periods]
        for name, got, ref in (('weights', np.asarray(out['weight_sum']), np.ones(len(periods))),
                               ('A', np.asarray(out['mean_assets']), A),
                               ('C', np.asarray(out['mean_consumption']), C)):
            gap = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)))
            if gap > 1e-6:
                raise RuntimeError(f"cross-section {name} differs from the transition's by {gap:.2e}")
    if t_s is not None and abs(float(getattr(olg.lifecycle_config, 'gamma', 1.0)) - 1.0) > 1e-12:
        print("      note: the consumption-equivalent measure assumes log utility; "
              "no welfare outputs at gamma != 1")
        t_s = None
    if t_s is not None:
        r_ts = periods.index(int(t_s))
        wel = welfare_inputs(olg, panels, int(t_s), r_ts, newborn_bps)
        for key, rec in wel['alive'].items():
            rows, alive, vals, mass, share, model = panels[key]
            _check_state_order(model, vals[r_ts])
            rec['income'] = _derived(vals[r_ts], model, rec['j'])['disp_income']
        wel['W_ts'] = np.asarray(olg._aggregation_weights(int(t_s)), dtype=float)
        out['welfare'] = wel
    return out


def welfare_summary(base_ext, cf_ext, education_shares):
    """lambda_c by birth period (cohorts alive in t_s and later newborns) and
    the state-level lambda by quintile of baseline income in t_s."""
    b, c = base_ext['welfare'], cf_ext['welfare']
    lam, pop = cohort_cev(b, c, education_shares, b['W_ts'])
    w = np.array([pop[k] for k in lam])
    alive_mean = float(np.sum(w * np.array(list(lam.values()))) / w.sum())
    return {'cohort': {str(k): v for k, v in lam.items()},
            'alive_mean': alive_mean,
            'quintile': quintile_cev(b, c, None, education_shares, b['W_ts'])}


def serialisable(ext):
    """The JSON part of an extract() result (drops the welfare arrays)."""
    return {k: v for k, v in ext.items() if k != 'welfare'}
