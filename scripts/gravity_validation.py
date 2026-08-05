#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/gravity_validation.py — statistical validation of the temporal calibrated
gravity mobility model (formulae_sheet.md §4/§7) against the raw BTS T-100
state→state monthly flows (data/bts_state_pairs_monthly.csv, US carriers,
2020-01 → 2023-03, 39 months).

Model under test (origin i → dest j, per day t):

    Ŵ_ij(t) = s_i(t) · μ0_i · W^grav_ij                       [travellers/day]
    μ0_i    = inter-state outbound_i(2020-01) / T_mo   (T_mo = 30.44 d)
    s_i(t)  = inter-state outbound_i(t) / outbound_i(2020-01)
    W^grav_ij ∝ √(N_i N_j) / max(d_ij, 100 km)^power,  row-normalised, power=2

Intra-state self-pairs (i==j) are excluded throughout: they never leave the
patch.  Because s_i(t)·μ0_i·T_mo = outbound_i(t) by construction, the outbound
*totals* carry no information — the belief "the temporally calibrated gravity
model is correct" is tested on the directional fingerprint (outbound shares),
the uniform temporal-scaling claim, and the *uncalibrated* inbound totals.

Tests:
  [A]  outbound totals identity (numerical exactness)
  [B]  outbound directional shares p̂_ij(t) vs W^grav_ij  (Pearson/Spearman/log)
  [B2] per-pair predicted vs observed flow (raw + log–log)
  [C]  top-1 / top-3 destination agreement
  [D]  uniform temporal scaling: F_ij(t)/F_ij(Jan) vs s_i(t)  (corr + pooled slope)
  [E]  inbound totals, cross-sectional per month (UNcalibrated prediction)
  [F]  inbound totals, per-state time series
  [G]  gravity regression  log F ~ b_N·log N_j + b_d·log d_ij + origin/month FE
  [H]  robustness: contiguous-49 subset only

Run:  venv/bin/python scripts/gravity_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from fused_metapopulation import (  # noqa: E402
    JURISDICTIONS, CENSUS_POPULATIONS, CONTIGUOUS, CODE_TO_NAME,
    build_gravity_kernel, build_distance_matrix, calibrate_mobility,
    build_month_index, T_MO,
)

POWER = 2.0
D_MIN = 100.0


def _r(x, y, fn):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return fn(x, y)[0]


def pear(x, y):
    return _r(x, y, pearsonr)


def spear(x, y):
    return _r(x, y, spearmanr)


def main():
    DATA = os.path.join(ROOT, 'data')
    states = list(JURISDICTIONS)
    n = len(states)
    months = build_month_index()
    T = len(months)
    N = np.array([CENSUS_POPULATIONS[s] for s in states], dtype=float)

    line = '=' * 108
    print(line)
    print('GRAVITY-MODEL VALIDATION vs BTS T-100 state flows (US carriers)')
    print(f'{n} jurisdictions | {T} months '
          f'({months[0][0]:04d}-{months[0][1]:02d} … {months[-1][0]:04d}-{months[-1][1]:02d}) | '
          f'kernel: sqrt(Ni·Nj)/d^{POWER:g}, d_min={D_MIN:g} km, row-normalised | T_mo={T_MO} d')
    print(line)

    pairs = pd.read_csv(os.path.join(DATA, 'bts_state_pairs_monthly.csv'))
    print(f'[0] raw file: {len(pairs):,} rows | columns: {list(pairs.columns)} | '
          f'years {pairs.YEAR.min()}–{pairs.YEAR.max()}')

    mu0, s, months, F = calibrate_mobility(states, data_dir=DATA)
    out = F.sum(axis=1)                        # (n,T) inter-state outbound/month
    out0 = out[:, 0]
    active = out0 > 0
    inactive = [states[i] for i in np.where(~active)[0]]
    print(f'    model-active origins (Jan-2020 outbound > 0): {active.sum()}/{n}'
          + (f' | inactive: {inactive}' if inactive else ''))

    # intra-state share (data note) + consistency with bts_state_monthly.csv
    F_full = pd.read_csv(os.path.join(DATA, 'bts_state_pairs_monthly.csv'))
    Ff = np.zeros((n, n, T))
    sidx = {s: i for i, s in enumerate(states)}
    mkey = {(y, m): t for t, (y, m) in enumerate(months)}
    for r in F_full.itertuples(index=False):
        o, d = r.origin_state, r.dest_state
        if o not in sidx or d not in sidx:
            continue
        t = mkey.get((int(r.YEAR), int(r.MONTH)))
        if t is not None:
            Ff[sidx[o], sidx[d], t] += r.PASSENGERS
    intra_jan = np.diag(Ff[:, :, 0]) / np.maximum(Ff[:, :, 0].sum(axis=1), 1)
    print(f'    data note: intra-state share of Jan-2020 outbound — '
          f'median {np.median(intra_jan) * 100:.1f}%, max '
          f'{intra_jan.max() * 100:.1f}% ({states[int(np.argmax(intra_jan))]}); '
          f'excluded from μ0/s calibration')
    try:
        sm = pd.read_csv(os.path.join(DATA, 'bts_state_monthly.csv'))
        smt = sm.pivot_table(index='state', columns='month', values='outbound') \
                 .reindex(index=states).fillna(0)
        obs_cols = [f'{y:04d}-{m:02d}' for y, m in months]
        rel_inter = np.abs(out - smt[obs_cols].values) / np.maximum(smt[obs_cols].values, 1)
        rel_full = np.abs(Ff.sum(1) - smt[obs_cols].values) / np.maximum(smt[obs_cols].values, 1)
        print(f'    bts_state_monthly.csv outbound vs inter-state sum: max rel diff '
              f'{np.nanmax(rel_inter):.2e} | vs FULL sum: {np.nanmax(rel_full):.2e} '
              f'(→ file already excludes intra-state if inter≈0)')
    except Exception as e:
        print('    bts_state_monthly consistency check skipped:', e)

    D = build_distance_matrix(states)
    W = build_gravity_kernel(states, power=POWER, min_distance_km=D_MIN)

    # ── [A] outbound totals identity ──────────────────────────────────────────
    print('\n' + line)
    print('[A] OUTBOUND TOTALS — calibration identity (expect ~0 error)')
    pred_month = s * mu0[:, None] * T_MO
    err = np.abs(pred_month - out)
    print(f'    max |pred − obs| over active states: {err[active].max():.6f} pax/month '
          f'(relative: {np.max(err[active] / np.maximum(out[active], 1)):.2e})')
    for i in np.where(~active)[0]:
        nz = (out[i] > 0).sum()
        print(f'    {states[i]} (μ0=0, no extrapolation): model outbound ≡ 0; CSV has '
              f'{nz}/{T} nonzero months (max {out[i].max():,.0f} pax/mo) → excluded below')

    # ── [B] outbound directional shares ───────────────────────────────────────
    print('\n' + line)
    print('[B] OUTBOUND DIRECTIONAL SHARES  p̂_ij(t) = F_ij(t)/out_i(t)  vs  W^grav_ij')
    print('    (per active origin, per month, across 56 destinations)')
    pooled_p, pooled_s, pooled_lx, pooled_ly = [], [], [], []
    per_state = []
    for i in range(n):
        if not active[i]:
            continue
        med_p, med_s, mn_p, mx_p = [], [], [], []
        months_used = 0
        for t in range(T):
            o = F[i, :, t]
            tot = o.sum()
            if tot <= 0:
                continue
            months_used += 1
            phat = o / tot
            pg = W[i]
            rp, rs = pear(phat, pg), spear(phat, pg)
            if not np.isnan(rp):
                med_p.append(rp); mn_p.append(rp); mx_p.append(rp)
            if not np.isnan(rs):
                med_s.append(rs)
            pooled_p.append(phat); pooled_s.append(pg)
            m = phat > 0
            if m.sum() >= 2:
                pooled_lx.append(np.log(phat[m])); pooled_ly.append(np.log(pg[m]))
        if med_p:
            per_state.append([states[i], months_used, np.median(med_p),
                              np.median(med_s), np.min(mn_p), np.max(mx_p)])

    pp = np.concatenate(pooled_p); ps = np.concatenate(pooled_s)
    lx = np.concatenate(pooled_lx); ly = np.concatenate(pooled_ly)
    print(f'    pooled (all active state·month·dest, N={len(pp):,}):')
    print(f'      Pearson(raw)  = {pear(pp, ps):.4f}')
    print(f'      Spearman      = {spear(pp, ps):.4f}')
    print(f'      Pearson(log)  = {pear(lx, ly):.4f}   (only observed dests, N={len(lx):,})')

    print('    per-month pooled correlations (landmark months):')
    print(f'      {"month":>7} | {"n(state-mo)":>10} | {"Pearson":>7} | '
          f'{"Spearman":>8} | {"Pearson(log)":>12}')
    for (y, mo) in [(2020, 1), (2020, 4), (2020, 7), (2020, 12),
                    (2021, 12), (2022, 12), (2023, 3)]:
        t = months.index((y, mo))
        pp2, ps2, lx2, ly2 = [], [], [], []
        for i in np.where(active)[0]:
            o = F[i, :, t]; tot = o.sum()
            if tot <= 0:
                continue
            phat = o / tot; pg = W[i]
            pp2.append(phat); ps2.append(pg)
            m = phat > 0
            if m.sum() >= 2:
                lx2.append(np.log(phat[m])); ly2.append(np.log(pg[m]))
        a = np.concatenate(pp2); b = np.concatenate(ps2)
        c = np.concatenate(lx2); d2 = np.concatenate(ly2)
        print(f'      {y}-{mo:02d} | {len(pp2):10d} | {pear(a, b):7.4f} | '
              f'{spear(a, b):8.4f} | {pear(c, d2):12.4f}')

    ps_df = pd.DataFrame(per_state,
                         columns=['state', 'months_used', 'medPearson',
                                  'medSpearman', 'minPearson', 'maxPearson'])
    ps_df['name'] = ps_df['state'].map(CODE_TO_NAME)
    ps_df = ps_df.sort_values('medPearson', ascending=False)
    out_csv = os.path.join(ROOT, 'scripts', 'gravity_corr_states.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ps_df.to_csv(out_csv, index=False)
    print(f'    per-state table (median over months; saved → '
          f'{os.path.relpath(out_csv, ROOT)}):')
    print(f'      {"state":<24} {"mo":>3} | {"medP":>6} {"medS":>6} '
          f'{"minP":>6} {"maxP":>6}')
    for _, rw in ps_df.iterrows():
        print(f'      {rw["name"]:<24} {rw.months_used:3d} | {rw.medPearson:6.3f} '
              f'{rw.medSpearman:6.3f} {rw.minPearson:6.3f} {rw.maxPearson:6.3f}')
    meds = ps_df['medPearson'].values
    print(f'    state-median Pearson: median {np.median(meds):.3f} | IQR '
          f'[{np.percentile(meds, 25):.3f}, {np.percentile(meds, 75):.3f}] | '
          f'range [{meds.min():.3f}, {meds.max():.3f}]')

    # ── [B2] per-pair predicted vs observed ───────────────────────────────────
    pred_pair = (s[:, None, :] * mu0[:, None, None] * W[:, :, None] * T_MO)  # (n,n,T)
    m2 = F > 0
    print('\n' + line)
    print('[B2] PER-PAIR PREDICTED vs OBSERVED flow, pooled over all (i,j,t)')
    print(f'    (N = {m2.sum():,} nonzero inter-state pairs; '
          f'predicted = s_i·μ0_i·W_ij·T_mo, pax/month)')
    a = pred_pair[m2]; b = F[m2]
    print(f'      Pearson(raw) = {pear(a, b):.4f} | Spearman = {spear(a, b):.4f}')
    lg = (a > 0) & (b > 0)
    print(f'      Pearson(log–log) = {pear(np.log(a[lg]), np.log(b[lg])):.4f} '
          f'| Spearman(log–log) = {spear(np.log(a[lg]), np.log(b[lg])):.4f}')

    # ── [C] top destinations ──────────────────────────────────────────────────
    print('\n' + line)
    print('[C] TOP-DESTINATION AGREEMENT (per active state-month)')
    h1, h3 = [], []
    for i in np.where(active)[0]:
        for t in range(T):
            o = F[i, :, t]
            if o.sum() <= 0:
                continue
            h1.append(np.argmax(o) == np.argmax(W[i]))
            s3o = set(np.argsort(o)[-3:]); s3g = set(np.argsort(W[i])[-3:])
            h3.append(len(s3o & s3g) / 3)
    print(f'    top-1 destination match: {np.mean(h1) * 100:.1f}%  '
          f'({np.sum(h1)}/{len(h1)} state-months)')
    print(f'    top-3 destination overlap: {np.mean(h3) * 100:.1f}%')

    # ── [D] uniform temporal scaling ──────────────────────────────────────────
    print('\n' + line)
    print('[D] UNIFORM TEMPORAL SCALING  F_ij(t)/F_ij(Jan-2020)  vs  s_i(t)')
    rs_d, Xd, Yd = [], [], []
    for i in np.where(active)[0]:
        for j in range(n):
            if i == j or F[i, j, 0] <= 0:
                continue
            x = F[i, j, :] / F[i, j, 0]
            m = (x > 0) & (s[i] > 0)
            if m.sum() < 5:
                continue
            rv = pear(x[m], s[i][m])
            if not np.isnan(rv):
                rs_d.append(rv)
            Xd.append(np.log(s[i][m])); Yd.append(np.log(x[m]))
    Xd = np.concatenate(Xd); Yd = np.concatenate(Yd)
    bfit, bres = np.polyfit(Xd, Yd, 1)
    r2d = 1 - np.sum((Yd - np.polyval([bfit, bres], Xd)) ** 2) / np.sum((Yd - Yd.mean()) ** 2)
    rs_d = np.array(rs_d)
    print(f'    pairs used: {len(rs_d):,} (origin–dest with Jan flow, ≥5 nonzero months)')
    print(f'    correlation median {np.median(rs_d):.3f} | IQR '
          f'[{np.percentile(rs_d, 25):.3f}, {np.percentile(rs_d, 75):.3f}] | '
          f'frac > 0.5: {np.mean(rs_d > 0.5) * 100:.1f}% | frac > 0.8: '
          f'{np.mean(rs_d > 0.8) * 100:.1f}%')
    print(f'    pooled regression log(F_ij/F_jan) = {bfit:.3f}·log(s_i) '
          f'(expect 1.00), R² = {r2d:.3f}, N = {len(Xd):,}')

    # ── [E] inbound totals (uncalibrated) ─────────────────────────────────────
    print('\n' + line)
    print('[E] INBOUND TOTALS (UNcalibrated) — predicted Σ_i s_i·μ0_i·W_ij vs observed')
    pred_in = W.T @ (s * mu0[:, None]) * T_MO      # (n,T) pax/month into each dest
    obs_in = F.sum(axis=0)
    r_cs = [pear(pred_in[:, t], obs_in[:, t]) for t in range(T)]
    r_cs = np.array([r for r in r_cs if not np.isnan(r)])
    print(f'    cross-sectional, per month (corr across {n} dests):')
    print(f'      median {np.median(r_cs):.3f} | IQR '
          f'[{np.percentile(r_cs, 25):.3f}, {np.percentile(r_cs, 75):.3f}] | '
          f'min {r_cs.min():.3f} | max {r_cs.max():.3f} | frac > 0.7: '
          f'{np.mean(r_cs > 0.7) * 100:.0f}%')
    a = pred_in.ravel(); b = obs_in.ravel()
    print(f'    pooled (all {len(a):,} dest·month): Pearson = {pear(a, b):.4f} | '
          f'Spearman = {spear(a, b):.4f}')
    lg = b > 0
    print(f'    pooled log–log (obs>0, N={lg.sum():,}): Pearson = '
          f'{pear(np.log(a[lg]), np.log(b[lg])):.4f}')

    # ── [F] inbound per-state time series ─────────────────────────────────────
    print('\n' + line)
    print('[F] INBOUND per-state TIME SERIES (39 months; shares the COVID trend)')
    r_ts = []
    for j in range(n):
        if obs_in[j].sum() <= 0:
            continue
        rv = pear(pred_in[j], obs_in[j])
        if not np.isnan(rv):
            r_ts.append((states[j], rv))
    rv_arr = np.array([r for _, r in r_ts])
    worst = min(r_ts, key=lambda x: x[1]); best = max(r_ts, key=lambda x: x[1])
    print(f'    states with observed inflow: {len(r_ts)}')
    print(f'    correlation median {np.median(rv_arr):.3f} | IQR '
          f'[{np.percentile(rv_arr, 25):.3f}, {np.percentile(rv_arr, 75):.3f}] | '
          f'worst {worst[1]:.3f} ({worst[0]}) | best {best[1]:.3f} ({best[0]})')

    # ── [G] gravity regression ────────────────────────────────────────────────
    print('\n' + line)
    print('[G] GRAVITY REGRESSION  log F_ij = b_N·log N_j + b_d·log d_ij + FE')
    print('    (model hypothesis: b_N = 0.50, b_d = −2.00; origin FE + month FE)')
    rows = np.argwhere(F > 0)
    o = rows[:, 0]; d = rows[:, 1]; tt = rows[:, 2]
    y = np.log(F[o, d, tt].astype(float))
    X = np.column_stack([np.log(np.maximum(D[o, d], 10.0)), np.log(N[d]),
                         np.eye(n)[o], np.eye(T)[tt]])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    sigma2 = ss_res / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    print(f'    pooled (N={len(y):,}, df={len(y) - X.shape[1]:,}):')
    print(f'      b_d  = {coef[0]:+.3f} ± {se[0]:.3f}   (model: −2.00)')
    print(f'      b_N  = {coef[1]:+.3f} ± {se[1]:.3f}   (model: +0.50)')
    print(f'      R² = {r2:.3f}')
    bs_d, bs_n, rs2 = [], [], []
    for t in range(T):
        m = tt == t
        Xt = np.column_stack([np.log(np.maximum(D[o[m], d[m]], 10.0)),
                              np.log(N[d[m]]), np.eye(n)[o[m]]])
        yt = y[m]
        ct, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
        pt = Xt @ ct
        r2t = 1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        bs_d.append(ct[0]); bs_n.append(ct[1]); rs2.append(r2t)
    bs_d = np.array(bs_d); bs_n = np.array(bs_n); rs2 = np.array(rs2)
    print(f'    per-month: b_d  mean {bs_d.mean():+.3f} ± {bs_d.std():.3f} '
          f'(range [{bs_d.min():+.3f}, {bs_d.max():+.3f}])')
    print(f'              b_N  mean {bs_n.mean():+.3f} ± {bs_n.std():.3f} '
          f'(range [{bs_n.min():+.3f}, {bs_n.max():+.3f}])')
    print(f'              R²   mean {rs2.mean():.3f} ± {rs2.std():.3f} '
          f'(range [{rs2.min():.3f}, {rs2.max():.3f}])')

    # ── [H] contiguous-49 robustness ──────────────────────────────────────────
    print('\n' + line)
    print('[H] ROBUSTNESS: CONTIGUOUS 49 ONLY (48 states + DC, excl AK/HI/territories)')
    idx = [states.index(c) for c in CONTIGUOUS]
    Wc = build_gravity_kernel(CONTIGUOUS, power=POWER, min_distance_km=D_MIN)
    nc = len(CONTIGUOUS)
    ppc, psc = [], []
    for ii, i in enumerate(idx):
        if not active[i]:
            continue
        for t in range(T):
            o = F[i, idx, t]; tot = o.sum()
            if tot <= 0:
                continue
            ppc.append(o / tot); psc.append(Wc[ii])
    a = np.concatenate(ppc); b = np.concatenate(psc)
    print(f'    pooled Pearson = {pear(a, b):.4f} | Spearman = {spear(a, b):.4f} '
          f'| N = {len(a):,}')
    lgc = (a > 0) & (b > 0)
    print(f'    pooled Pearson(log) = {pear(np.log(a[lgc]), np.log(b[lgc])):.4f}')

    print('\n' + line)
    print('DONE — full per-state table: scripts/gravity_corr_states.csv')
    print(line)


if __name__ == '__main__':
    main()
