"""Builder: regenerate notebooks/appendix_B.ipynb — the Omicron (wave 5,
2022-01-01 -> 2022-04-30) run of the fused degree-based SEIR x gravity
metapopulation with the SEIRV immune-pool machinery (vaccination + prior
infection), adding Omicron's immune escape. Edit this script, not the .ipynb."""
import os, nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nb = nbf.v4.new_notebook()
cells = []
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ── 0. title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""# Appendix B — Omicron (Wave 5): Does the SEIRV Model Explain the 2022 Surge?

**Built 2026-08-06** on the notebook-08 model (degree-based SEIR × temporal
gravity metapopulation, SEIRV with vaccination + prior-infection immunity,
formulae_sheet §1–§8). Modelling window: **2022-01-01 → 2022-04-30**
(days 731–850 since 2020-01-01). All data already in `data/` — no new
downloads needed (NYT cases, BTS T-100 mobility, CDC vaccination incl.
boosters, Google CMR, OxCGRT).

**What is different about Omicron (vs the Delta window, notebook 08 §8.14)**

1. **Transmissibility**: consensus R₀ ≈ 6–10 (vs ~5–8 Delta); observed
   national growth r ≈ 0.1–0.2/d — the steepest wave of the pandemic.
2. **Immune escape**: prior infection and 2-dose vaccination give much less
   protection *against infection* (≈ 30–50 % vs ≈ 60–80 % for Delta) — the
   susceptible pool at the window start is therefore *larger* than the
   naive "1 − vaccinated − prior" count.
3. **Boosters** (Sep 2021 →) restore some protection but mostly against
   severe disease; not modelled against infection (primary-series only).
4. **Reporting**: home-test undercount — reported cases likely ≈ 1/3 of
   true infections, so a model in true-infection units *should* show
   π̂ ≈ 3, not ≈ 1.

| Section | Content |
|---|---|
| §B.1 | data recap & the observed Omicron wave |
| §B.2 | β calibration on the Jan-2022 growth window |
| §B.3 | initial conditions: immunity stack (vaccine + prior, with escape) |
| §B.4 | baseline: constant-β, no immunity |
| §B.5 | SEIRV grid: vaccine efficacy × prior-immunity protection |
| §B.6 | honest reading: what Omicron explains & what it needs |"""))

# ── imports ───────────────────────────────────────────────────────────────────
cells.append(code("""import sys, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, '../src')
from fused_metapopulation import (
    JURISDICTIONS, CENSUS_POPULATIONS, CODE_TO_NAME, NAME_TO_CODE,
    T_MO, build_distance_matrix, build_gravity_kernel, fit_gravity_exponents,
    calibrate_mobility, s_interpolator, truncated_power_law,
    power_law_alpha_for_mean_k, FusedMetapopulationSEIR,
)
from data_loader import load_us_states

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150})
ASSETS = '../assets'
DAY0 = pd.Timestamp('2020-01-01')
T0  = (pd.Timestamp('2022-01-01') - DAY0).days    # 731
TMAX = (pd.Timestamp('2022-04-30') - DAY0).days   # 850
sigma, gamma = 1.0 / 5, 1.0 / 10

def show_fig(path):
    from IPython.display import Image, display
    display(Image(filename=path))

us = load_us_states()
print(f'window: day {T0}..{TMAX} = {DAY0 + pd.Timedelta(days=T0):%Y-%m-%d} '
      f'.. {DAY0 + pd.Timedelta(days=TMAX):%Y-%m-%d}')"""))

# ── B.1 data recap ────────────────────────────────────────────────────────────
cells.append(md(r"""## §B.1 The observed Omicron wave

The US entered 2022 in the middle of the Omicron surge: reported smoothed
cases went from ≈ 120k/day in mid-December 2021 to ≈ 800k/day in
mid-January 2022 (≈ 6–7× in a month), then collapsed to ≈ 30–60k/day by
April (with a BA.2 mini-surge in April). Window total ≈ 30M reported cases.
Vaccination coverage by 2022-01-01: ≈ 62 % fully vaccinated, ≈ 35 %
boosted; cumulative reported prior cases ≈ 55M (true ≈ 150M+)."""))
cells.append(code("""nat_obs_all = us.groupby('date')['new_cases_smooth'].sum()
w = nat_obs_all.loc[pd.Timestamp('2022-01-01'):pd.Timestamp('2022-04-30')]
print(f'window observed: total {w.sum():,.0f} | peak {w.idxmax().date()} '
      f'({w.max():,.0f}/d) | end {w.index[-1].date()} ({w.iloc[-1]:,.0f}/d)')

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(nat_obs_all.index, nat_obs_all.values, lw=1.0, color='#333')
ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-04-30'),
           color='C3', alpha=0.15, label='Omicron window (modelled)')
ax.set_title('US daily new cases (NYT, 7-day smoothed) — Omicron')
ax.set_ylabel('new cases / day')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig78_appB_cases.png'); plt.show()
show_fig(f'{ASSETS}/fig78_appB_cases.png')

# vaccination / booster coverage at window start
vax = pd.read_csv('../data/cdc_vax_state_monthly.csv')
cum = vax.pivot_table(index='state', columns='month',
                      values='series_complete').reindex(index=JURISDICTIONS).fillna(0)
boo = vax.pivot_table(index='state', columns='month',
                      values='boosters').reindex(index=JURISDICTIONS).fillna(0)
print('2021-12 coverage: fully vaccinated %.1fM (%.0f%% of pop) | boosted %.1fM (%.0f%%)'
      % (cum['2021-12'].sum() / 1e6, 100 * cum['2021-12'].sum() / sum(CENSUS_POPULATIONS.values()),
         boo['2021-12'].sum() / 1e6, 100 * boo['2021-12'].sum() / sum(CENSUS_POPULATIONS.values())))

us['cum_cases'] = us.groupby('state')['new_cases'].cumsum()
prior_cases = us[us.date <= pd.Timestamp('2021-12-31')].groupby('state')['cum_cases'].max()
print('cumulative reported cases by 2021-12-31: %.1fM (true ≈ 2.5-3x)' % (prior_cases.sum() / 1e6))"""))

# ── B.2 beta calibration ──────────────────────────────────────────────────────
cells.append(md(r"""## §B.2 β calibration — the pre-window Omicron ramp

Same slope-based calibration as notebook 08 §8.5, but on the Omicron
takeoff: log-linear slope of 7-day-smoothed new cases over
**2021-12-15 → 2022-01-10** — the explosive pre-peak ramp (the national
peak hit 2022-01-11, so a window inside Jan-2022 would straddle the peak
and under-estimate r).  The cap is raised to $R_0 = 8$ (Omicron consensus
6–10):

$$\beta_i = \\frac{(r_i+\\gamma)(r_i+\\sigma)}{\\sigma}\\,\\frac{\\langle k\\rangle}{\\langle k^2\\rangle}, \\qquad
\\beta_i \\leftarrow \\min(\\beta_i,\\ 8\\gamma\\langle k\\rangle/\\langle k^2\\rangle)$$

Note the fitted slope R₀ is an *effective* value (it already contains the
immune suppression of growth); the §B.5 runs rescale β by 1/s_eff so the
model reproduces the observed ramp whatever the immune pool."""))
cells.append(code("""alpha = power_law_alpha_for_mean_k(6.0)
k_vals, P_k = truncated_power_law(alpha=alpha)
mean_k = float(k_vals @ P_k); mean_k2 = float((k_vals ** 2) @ P_k)
mom_ratio = mean_k2 / mean_k

def fit_growth(df_sub):
    v = df_sub['new_cases_smooth'].values
    t = np.arange(len(v)); m = v > 0
    if m.sum() < 5:
        return np.nan
    sl = np.polyfit(t[m], np.log(v[m]), 1)[0]
    return sl if sl > 0 else np.nan

win_start = pd.Timestamp('2021-12-15'); win_days = 27
rs, bs = {}, {}
for code in JURISDICTIONS:
    name = CODE_TO_NAME[code]
    sub = us[(us.state == name) & (us.date >= win_start)
             & (us.date < win_start + pd.Timedelta(days=win_days))]
    rv = fit_growth(sub)
    rs[code] = rv
    bs[code] = (rv + gamma) * (rv + sigma) / sigma / mom_ratio if not np.isnan(rv) else np.nan
n_fit = int(pd.Series(rs).notna().sum())
bs = pd.Series(bs).fillna(pd.Series(bs).median()).clip(upper=8.0 * gamma / mom_ratio)
rs = pd.Series(rs).fillna(float(pd.Series(rs).median()))
print(f'usable growth windows (Dec15-Jan10): {n_fit}/56 | fallback median applied to {56 - n_fit}')
print('r: median %.3f/d | R0: median %.2f, range [%.2f, %.2f]'
      % (rs.median(), (bs * mom_ratio / gamma).median(),
         (bs * mom_ratio / gamma).min(), (bs * mom_ratio / gamma).max()))"""))

# ── B.3 initial conditions ────────────────────────────────────────────────────
cells.append(md(r"""## §B.3 Initial conditions (2022-01-01) — the immunity stack

- $I_0$ = reported new cases on 2022-01-01 ÷ π₀ = 0.15 (home-test
  era: CDC estimates ≈ 1 in 4 infections reported in Jan-2022).
- $V_0$ = fully vaccinated at 2021-12-31 × **ε_vax** (Omicron escape:
  fitted, expected ≈ 0.3–0.5).
- $R_{\\text{prior}}$ = cumulative reported prior cases × **p_prior**
  (effective prior-infection protection; fitted, expected ≈ 0.5–1.5 —
  cases × p_prior ≈ 30–80M effective).
- $S_0 = N - I_0 - E_0 - V_0 - R_{\\text{prior}}$; boosters not modelled
  against infection (they protect severe disease).
- Vaccination continues through the window: `series_complete_flow`
  × ε_vax.

Both ε_vax and p_prior are grid-searched (§B.5); the β rescaling by
1/s_eff (susceptible share at window start) is applied as in §8.14."""))
cells.append(code("""pi0 = 0.15
seed_date = pd.Timestamp('2022-01-01')
I0, E0 = {}, {}
for code in JURISDICTIONS:
    row = us[(us.state == CODE_TO_NAME[code]) & (us.date == seed_date)]
    i0 = float(row['new_cases'].iloc[0]) if len(row) else 0.0
    I0[code] = i0 / pi0
    E0[code] = I0[code] * (rs[code] + gamma) / sigma
print('national I0 (true-scale): %.0f' % sum(I0.values()))

from fused_metapopulation import build_month_index
months = build_month_index()
grid_m = [f'{y:04d}-{m:02d}' for y, m in months]
pv = vax.pivot_table(index='state', columns='month',
                     values='series_complete_flow').reindex(index=JURISDICTIONS,
                                                            columns=grid_m).fillna(0)
V0_raw = {s: float(cum.loc[s, '2021-12']) for s in JURISDICTIONS}
print('V0 (fully vaccinated at 2021-12-31): %.1fM' % (sum(V0_raw.values()) / 1e6))

# model setup shared by all runs
mu0, s, months, F = calibrate_mobility(JURISDICTIONS, data_dir='../data')
s_func = s_interpolator(s, months)
D = build_distance_matrix(JURISDICTIONS)
Npop = np.array([CENSUS_POPULATIONS[c] for c in JURISDICTIONS], dtype=float)
r_ppml = fit_gravity_exponents(F, D, Npop, method='ppml')
W_cal = build_gravity_kernel(JURISDICTIONS, b_N=r_ppml['b_N'], b_d=-r_ppml['b_d'])
dates = [DAY0 + pd.Timedelta(days=int(t)) for t in range(T0, TMAX + 1)]
nat_obs = nat_obs_all.reindex(dates).fillna(0).values"""))

# ── B.4 baseline ──────────────────────────────────────────────────────────────
cells.append(md(r"""## §B.4 Baseline — constant-β, no immunity

With Omicron-scale β and **no** immunity in the initial condition
(S₀ = N), the model grows unboundedly to the window edge — the reference
that shows what the immune pool must explain."""))
cells.append(code("""def run(eps_vax, p_prior, vac_on=True):
    V0 = {s: v * eps_vax for s, v in V0_raw.items()} if vac_on else {s: 0.0 for s in V0_raw}
    R0init = {s: float(prior_cases.get(CODE_TO_NAME[s], 0)) * p_prior for s in JURISDICTIONS}
    s_eff = np.clip(np.array([1.0 - (V0[s] + R0init[s]) / CENSUS_POPULATIONS[s]
                              for s in JURISDICTIONS]), 0.02, None)
    beta_v = bs.values / s_eff
    flow = (pv.values / T_MO) if vac_on else np.zeros_like(pv.values)
    m = FusedMetapopulationSEIR(
        JURISDICTIONS, CENSUS_POPULATIONS, W_cal, mu0, s_func,
        k_vals, P_k, beta=beta_v, sigma=sigma, gamma=gamma,
        months=months, vac_flow=flow, vac_efficacy=eps_vax, V0_dict=V0,
        R_init_dict=R0init)
    m.simulate(TMAX, I0, E0, t0=T0)
    ni = m.daily_new_infections().sum(0)
    ph = ni.sum() / nat_obs.sum()
    rm = np.sqrt(np.mean((np.log(ni / ph + 1) - np.log(nat_obs + 1)) ** 2))
    return ni, ph, rm, int(np.argmax(ni)), m

ni0, ph0, rm0, pk0, m0 = run(0.0, 0.0, vac_on=False)
r2l0 = 1 - np.sum((np.log(ni0 / ph0 + 1) - np.log(nat_obs + 1)) ** 2) / np.sum((np.log(nat_obs + 1) - np.log(nat_obs + 1).mean()) ** 2)
print(f'baseline (no immunity): pi-hat = {ph0:.1f} | peak = {dates[pk0].date()} | R2(log) = {r2l0:.2f}')"""))

# ── B.5 SEIRV grid ────────────────────────────────────────────────────────────
cells.append(md(r"""## §B.5 SEIRV with immune escape — ε_vax × p_prior grid

Two fitted parameters: ε_vax (2-dose efficacy vs Omicron infection) and
p_prior (effective prior-infection protection, cases × p_prior).
Omicron's escape means both should come out *lower* than the Delta fit
(ε = 1.0, p ≈ 3)."""))
cells.append(code("""rows = []
for eps_vax in (0.2, 0.3, 0.4, 0.6):
    for p_prior in (0.5, 1.0, 1.5):
        ni, ph, rm, pk, m = run(eps_vax, p_prior)
        rows.append((p_prior, eps_vax, rm, ph, pk))
grid = pd.DataFrame(rows, columns=['p_prior', 'eps_vax', 'log_RMSE', 'pi_hat', 'peak_day'])
print('grid (p_prior = prior-immunity factor, eps_vax = vaccine efficacy):')
print(grid.sort_values('log_RMSE').head(8).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
best = grid.sort_values('log_RMSE').iloc[0]
ni2, ph2, rm2, pk2, m2 = run(float(best.eps_vax), float(best.p_prior))
r2l2 = 1 - np.sum((np.log(ni2 / ph2 + 1) - np.log(nat_obs + 1)) ** 2) / np.sum((np.log(nat_obs + 1) - np.log(nat_obs + 1).mean()) ** 2)
r2v0 = 1 - np.sum((ni0 / ph0 - nat_obs) ** 2) / np.sum((nat_obs - nat_obs.mean()) ** 2)
r2v2 = 1 - np.sum((ni2 / ph2 - nat_obs) ** 2) / np.sum((nat_obs - nat_obs.mean()) ** 2)
pk_o = nat_obs_all.loc[dates].idxmax()
print(f'best: eps_vax={best.eps_vax:.1f}, p_prior={best.p_prior:.1f}')
print(f'pi-hat: no-immunity {ph0:.1f} -> SEIRV {ph2:.2f}')
print(f'peak: no-immunity {dates[pk0].date()} -> SEIRV {dates[pk2].date()} (observed {pk_o.date()})')
print(f'R2(log): no-immunity {r2l0:.2f} -> SEIRV {r2l2:.2f}')
print(f'R2(linear): no-immunity {r2v0:.2f} -> SEIRV {r2v2:.2f}')
apr = float(ni2[-1] / ni2[int(np.argmax(ni2))])
print(f'end/peak ratio (SEIRV): {apr:.2f} (observed {nat_obs[-1]/nat_obs.max():.2f})')"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
axes[0].plot(dates, nat_obs, lw=1.4, color='k', label='observed (smoothed)')
axes[0].plot(dates, ni0 / ph0, lw=1.3, color='C0', ls=':', label='no-immunity (own π̂)')
axes[0].plot(dates, ni2 / ph2, lw=1.3, color='C2',
             label=f'SEIRV ε={best.eps_vax:.1f}, p={best.p_prior:.1f} (own π̂)')
axes[0].set_ylabel('reported-scale new cases / day')
axes[0].set_title(f'Omicron national fit\\nπ̂: {ph0:.1f} -> {ph2:.2f}')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

# immunity stack at window start by state
sus = []
for s in JURISDICTIONS:
    v = V0_raw[s] * best.eps_vax
    r = float(prior_cases.get(CODE_TO_NAME[s], 0)) * best.p_prior
    sus.append(100 * (1 - (v + r) / CENSUS_POPULATIONS[s]))
order = np.argsort(sus)
axes[1].barh(np.arange(56), [sus[i] for i in order], color='C6', alpha=0.8)
axes[1].set_yticks([])
axes[1].set_xlabel('effective susceptible (%)')
axes[1].set_title(f'Effective S₀ share by state (ε={best.eps_vax:.1f}, p={best.p_prior:.1f})\\n'
                  f'national mean = {np.mean(sus):.0f}%')
axes[1].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ASSETS}/fig79_appB_fit.png'); plt.show()
show_fig(f'{ASSETS}/fig79_appB_fit.png')"""))

# ── B.6 honest reading ────────────────────────────────────────────────────────
cells.append(md(r"""## §B.6 Honest reading

**The model explains Omicron's *level* but not its *shape*.**

- **Peak level right**: the model's national true new-infections peak
  (≈ 750k/day) matches the observed reported peak (≈ 807k/day).
- **π̂ ≈ 1.3 is *too low*** — for the home-test era the model should sit at
  π̂ ≈ 3–5 against reported cases (CDC: ≈ 1-in-4–1-in-5 infections
  reported in Jan-2022).  The model under-produces true infections ≈ 2–3×,
  so its susceptible pool depletes ≈ 2× too slowly.
- **Peak 2 months late** (2022-03-10 vs observed 2022-01-11) and the
  January collapse is too shallow (end/peak 0.36 vs observed 0.08).  Two
  structural causes:
  1. **IC under-seeding**: I₀ = reported/π₀ ≈ 1.1M active, but the real
     active pool on Jan-1-2022 was ≈ 3–5M (home tests) — the infection
     flux (∝ I·S) starts 3–5× low, so depletion is slow even though the
     growth rate matches (r ≈ 0.075/d, R₀_eff ≈ 2.4, 54/56 states).
  2. **No behavioural Rₜ collapse**: the real January decline
     (5× fall in 3 weeks, r ≈ −0.15/d) is 2–3× faster than pure
     S-depletion can produce — it needs a transmission-modulation
     channel (post-holiday behaviour, mass quarantine of cases), i.e. a
     β(t) collapse that the CMR/stringency drivers (§8.13) cannot see
     (mobility barely moved in January 2022).
- **What the fit *does* reveal — immune escape quantified**: the grid
  prefers ε_vax = 0.6, p_prior = 1.5 vs the Delta window's ε = 1.0,
  p ≈ 3.0 — i.e. 2-dose vaccine and prior infection are only ~half as
  protective against Omicron infection, and the effective susceptible pool
  at the window start is ≈ 50–60 % (vs ≈ 34 % for Delta).  Without the
  immune-pool machinery the model explodes (π̂ ≈ 3.3, no peak, §B.4).
- **Omicron vs Delta (explanatory power)**: Delta closes at R²(log) +0.85
  with vaccination + prior immunity; Omicron stays at ≈ −1.2 because its
  defining features — extreme transmissibility × immune escape × home-test
  undercount × behavioural collapse — jointly sit beyond a
  susceptible-depletion SEIRV.  The documented next step: a two-regime
  β(t) (pre/post-peak) fitted to the January data, plus π₀ as an explicit
  fitted parameter."""))
cells.append(code("""print('appendix B complete | figures fig78, fig79 -> assets/')"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14'},
}
out = os.path.join(ROOT, 'notebooks', 'appendix_B.ipynb')
with open(out, 'w') as f:
    f.write(nbf.writes(nb, indent=1))
print('wrote', out, '| cells:', len(cells))
