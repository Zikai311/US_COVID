"""Builder: regenerate notebooks/appendix_A.ipynb from scratch — data
reconstruction experiment for the temporal calibrated gravity model vs the
raw BTS T-100 state->state monthly flows (reuses the tests of
scripts/gravity_validation.py, re-run on the current calibrated-kernel model
with PPML exponents). Edit this script, not the .ipynb."""
import json, os, nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nb = nbf.v4.new_notebook()
cells = []
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ── 0. title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""# Appendix A — Data Reconstruction: Does the Gravity Model Reproduce US Aviation Flows?

**Built 2026-08-06** on top of the fused metapopulation model of notebook 08
(`src/fused_metapopulation.py`). All tests reuse `scripts/gravity_validation.py`
but are re-run on the **current** gravity kernel — the PPML-calibrated one
($b_N \approx 1.0$, $b_d \approx 0.55$) instead of the classic
$\sqrt{N_iN_j}/d^2$.

**Experiment (one sentence).** From (i) 2020-Census populations, (ii)
jurisdiction distances and (iii) each state's *monthly outbound totals* we
reconstruct the full directional flow matrix $\hat F_{ij}(t)$, then measure
how well it reproduces the raw BTS T-100 flows $F_{ij}(t)$.

**What is calibrated vs what is predicted — the honest split:**

| Component | Input used | Status |
|---|---|---|
| $\mu_{0,i}$, $s_i(t)$ | observed per-state monthly **totals** | calibrated (exact by construction, §A.3) |
| directional shares $W_{ij}^{\text{grav}}$ | populations + distances only | **predicted** (the gravity hypothesis) |
| per-pair flows $\hat F_{ij}(t)$ | the two above | reconstruction (§A.5) |
| inbound totals $\hat I_j(t)$ | outbound calibration only | **predicted** (§A.7) |

The outbound *totals* carry no information about the gravity hypothesis —
they are matched by construction. Where the model stands or falls is the
**directional fingerprint**: shares, per-pair structure, top destinations,
inbound (which is not calibrated at all).

| Section | Test |
|---|---|
| §A.1 | data & calibration inputs |
| §A.2 | gravity kernels: classic vs PPML-calibrated |
| §A.3 | [A] outbound-totals identity |
| §A.4 | [B] outbound directional shares (pooled / per-month / per-state) |
| §A.5 | [B2] per-pair reconstruction · [C] top destinations |
| §A.6 | [D] uniform temporal scaling |
| §A.7 | [E] inbound cross-sectional · [F] inbound time series |
| §A.8 | [G] gravity regression ($b_N$, $b_d$) |
| §A.9 | [H] robustness: contiguous-49 subset |
| §A.10 | reconstruction summary & limitations |"""))

# ── 0. imports ────────────────────────────────────────────────────────────────
cells.append(code("""import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, '../src')
from fused_metapopulation import (
    JURISDICTIONS, CENSUS_POPULATIONS, CONTIGUOUS, CODE_TO_NAME,
    T_MO, build_distance_matrix, build_gravity_kernel, fit_gravity_exponents,
    calibrate_mobility, build_month_index,
)
from data_loader import load_bts_state_pairs

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150})
ASSETS = '../assets'
POWER, D_MIN = 2.0, 100.0   # classic kernel params (Viboud 2006)

def pear(x, y):
    if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]

def spear(x, y):
    if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return spearmanr(x, y)[0]

def show_fig(path):
    from IPython.display import Image, display
    display(Image(filename=path))

print('ready | nodes:', len(JURISDICTIONS))"""))

# ── A.1 data ──────────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.1 Data & calibration inputs

Raw source: BTS T-100 Market (US carriers, domestic markets), aggregated to
jurisdiction pairs by `data/bts_state_pairs_monthly.csv` — **63,721 rows**,
directional monthly passengers for every (origin, dest) among the 56 NYT/BTS
jurisdictions, 2020-01 → 2023-03 (39 months). Missing rows = zero passengers.

`calibrate_mobility()` builds the inter-state tensor $F_{ij}(t)$
(56×56×39), **excluding intra-state self-pairs** (they never leave the
patch), then per state:

$$\mu_{0,i} = \frac{\sum_{j\ne i} F_{ij}(\text{Jan-2020})}{T_{\text{mo}}},
\qquad
s_i(t) = \frac{\sum_{j\ne i} F_{ij}(t)}{\sum_{j\ne i} F_{ij}(\text{Jan-2020})},
\qquad T_{\text{mo}} = 30.44\ \text{d}$$

States with no Jan-2020 outbound flow at all ($\mu_{0,i}=0$) are kept but
decoupled (no outbound mobility) — **never extrapolated**."""))
cells.append(code("""states = list(JURISDICTIONS)
n = len(states)
months = build_month_index()
T = len(months)
Npop = np.array([CENSUS_POPULATIONS[s] for s in states], dtype=float)

mu0, s, months, F = calibrate_mobility(states, data_dir='../data')
out = F.sum(axis=1)               # (n, T) inter-state outbound / month
out0 = out[:, 0]
active = out0 > 0
inactive = [states[i] for i in np.where(~active)[0]]

pairs = load_bts_state_pairs('../data')
print(f'jurisdictions: {n} | months: {T} ({months[0][0]:04d}-{months[0][1]:02d} '
      f'.. {months[-1][0]:04d}-{months[-1][1]:02d}) | F tensor: {F.shape}')
print(f'raw rows: {len(pairs):,} | years {pairs.YEAR.min()}..{pairs.YEAR.max()}')
print(f'model-active origins (Jan-2020 inter-state outbound > 0): '
      f'{int(active.sum())}/{n}' + (f' | inactive: {inactive}' if inactive else ''))

# intra-state share of Jan-2020 outbound (data note: why self-flows are excluded)
Ff = np.zeros((n, n, T))
sidx = {s: i for i, s in enumerate(states)}
mkey = {(y, m): t for t, (y, m) in enumerate(months)}
for rr in pairs.itertuples(index=False):
    o, d = rr.origin_state, rr.dest_state
    if o not in sidx or d not in sidx:
        continue
    t = mkey.get((int(rr.YEAR), int(rr.MONTH)))
    if t is not None:
        Ff[sidx[o], sidx[d], t] += rr.PASSENGERS
intra_jan = np.diag(Ff[:, :, 0]) / np.maximum(Ff[:, :, 0].sum(axis=1), 1)
print('intra-state share of Jan-2020 outbound: median %.1f%% | max %.1f%% (%s) '
      '-> excluded from mu0/s' % (np.median(intra_jan) * 100, intra_jan.max() * 100,
                                  CODE_TO_NAME[states[int(np.argmax(intra_jan))]]))
print('Jan-2020 outbound (inter-state), top 5:')
for i in np.argsort(mu0)[::-1][:5]:
    print(f'  {CODE_TO_NAME[states[i]]:<24} {out0[i]:12,.0f} pax/month')"""))

# ── A.2 kernels ───────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.2 Gravity kernels — classic vs PPML-calibrated

Two candidate kernels for the directional shares $W_{ij}^{\text{grav}}$
(row-normalised, $\sum_j W = 1$):

- **classic (Viboud 2006):** $\sqrt{N_i N_j}\,/\,d_{ij}^{\,2}$
- **data-calibrated (default, this model):** $N_j^{\,b_N}\,/\,d_{ij}^{\,b_d}$
  with exponents fitted from the full 39-month panel via **PPML**
  (Poisson pseudo-ML, origin + month fixed effects — robust to the Jensen
  bias of log-linear OLS):

$$\log F_{ijt} = \alpha_i + \gamma_t + b_N\log N_j + b_d\log d_{ij} + \varepsilon
\qquad\text{(OLS on logs, for comparison)}$$

The kernel uses only $N_j$ (destination population) and $d_{ij}$ — no flows
leak into the directional prediction beyond the per-state totals."""))
cells.append(code("""D = build_distance_matrix(states)
r_ols  = fit_gravity_exponents(F, D, Npop, method='ols')
r_ppml = fit_gravity_exponents(F, D, Npop, method='ppml')
print('OLS : b_d = %+.3f +- %.3f | b_N = %+.3f +- %.3f | R2 = %.3f'
      % (r_ols['b_d'], r_ols['se_d'], r_ols['b_N'], r_ols['se_N'], r_ols['r2']))
print('PPML: b_d = %+.3f | b_N = %+.3f | deviance R2 = %.3f | converged = %s'
      % (r_ppml['b_d'], r_ppml['b_N'], r_ppml['r2'], r_ppml['converged']))
b_N, b_d = r_ppml['b_N'], -r_ppml['b_d']   # positive distance exponent

W_classic = build_gravity_kernel(states, power=POWER, min_distance_km=D_MIN)
W_cal = build_gravity_kernel(states, b_N=b_N, b_d=b_d)
print('kernels built: classic sqrt(NiNj)/d^2 | calibrated N_j^%.3f / d^%.3f'
      % (b_N, b_d))"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for axx, Wk, lab in zip(axes, (W_classic, W_cal),
                        ('classic $\\\\sqrt{N_i N_j}/d^2$',
                         f'calibrated $N_j^{{{b_N:.2f}}}/d^{{{b_d:.2f}}}$')):
    x, y = [], []
    for i in np.where(active)[0]:
        o = F[i, :, 0]
        x += list(Wk[i]); y += list(o / o.sum())
    x, y = np.array(x), np.array(y)
    m = (x > 0) & (y > 0)
    axx.scatter(np.log(x[m]), np.log(y[m]), s=4, alpha=0.45, color='C2')
    axx.set_title(f'{lab}\\nPearson(log) = {pear(np.log(x[m]), np.log(y[m])):.3f} '
                  f'| Spearman = {spear(np.log(x[m]), np.log(y[m])):.3f}', fontsize=10)
    axx.set_xlabel('log predicted outbound share (kernel)')
    axx.set_ylabel('log observed outbound share (Jan-2020)')
    axx.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ASSETS}/fig68_appA_kernel_shares.png'); plt.show()
show_fig(f'{ASSETS}/fig68_appA_kernel_shares.png')"""))

# ── A.3 identity ──────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.3 [A] Outbound-totals identity (calibration exactness)

Sanity check first: with $\hat F_{ij}(t) = s_i(t)\,\mu_{0,i}\,W_{ij}^{\text{grav}}\,T_{\text{mo}}$,
each state's predicted **total** outbound equals its observed total **by
construction** (whatever the kernel). The reconstruction experiment therefore
rests entirely on the directional tests that follow."""))
cells.append(code("""pred_month = s * mu0[:, None] * T_MO
err = np.abs(pred_month - out)
print('max |pred - obs| over active states: %.3e pax/month (relative %.2e)'
      % (err[active].max(), np.max(err[active] / np.maximum(out[active], 1))))
for i in np.where(~active)[0]:
    nz = (out[i] > 0).sum()
    print(f'{states[i]} (mu0=0, no extrapolation): model outbound = 0; '
          f'data has {nz}/{T} nonzero months (max {out[i].max():,.0f} pax/mo) '
          f'-> excluded below')"""))

# ── A.4 shares ────────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.4 [B] Outbound directional shares

The core test: does the gravity kernel reproduce **where** the outbound
passengers go? For each active origin $i$ and month $t$,
$\hat p_{ij} = W_{ij}^{\text{grav}}$ is compared with the observed share
$p_{ij}(t) = F_{ij}(t)/\sum_j F_{ij}(t)$ across all 56 destinations."""))
cells.append(code("""def pooled_shares(Wk, month=None):
    x, y, lx, ly = [], [], [], []
    tsel = range(T) if month is None else [months.index(month)]
    for i in np.where(active)[0]:
        for t in tsel:
            o = F[i, :, t]; tot = o.sum()
            if tot <= 0:
                continue
            phat = o / tot
            x += list(Wk[i]); y += list(phat)
            m = phat > 0
            if m.sum() >= 2:
                lx += list(np.log(phat[m])); ly += list(np.log(Wk[i][m]))
    return np.array(x), np.array(y), np.array(lx), np.array(ly)

rows = []
for lab, Wk in (('classic sqrt(NiNj)/d^2', W_classic),
                (f'calibrated N_j^{b_N:.2f}/d^{b_d:.2f}', W_cal)):
    x, y, lx, ly = pooled_shares(Wk)
    print(f'{lab:<28} pooled  P = {pear(x, y):.4f} | S = {spear(x, y):.4f} '
          f'| log-P = {pear(lx, ly):.4f}   (N={len(x):,}, Nlog={len(lx):,})')
    x0, y0, lx0, ly0 = pooled_shares(Wk, month=(2020, 1))
    print(f'{"":<28} Jan-2020 P = {pear(x0, y0):.4f} | S = {spear(x0, y0):.4f} '
          f'| log-P = {pear(lx0, ly0):.4f}')
    rows.append((lab, pear(x, y), spear(x, y), pear(x0, y0), spear(x0, y0)))
share_summary = pd.DataFrame(rows, columns=['kernel', 'P_pooled', 'S_pooled',
                                            'P_jan', 'S_jan'])"""))

cells.append(code("""# per-month pooled correlations (all months)
mm_p, mm_s = [], []
for t in range(T):
    a, b = [], []
    for i in np.where(active)[0]:
        o = F[i, :, t]; tot = o.sum()
        if tot <= 0:
            continue
        a += list(o / tot); b += list(W_cal[i])
    mm_p.append(pear(np.array(a), np.array(b)))
    mm_s.append(spear(np.array(a), np.array(b)))
mm_pc, mm_sc = [], []
for t in range(T):
    a, b = [], []
    for i in np.where(active)[0]:
        o = F[i, :, t]; tot = o.sum()
        if tot <= 0:
            continue
        a += list(o / tot); b += list(W_classic[i])
    mm_pc.append(pear(np.array(a), np.array(b)))
    mm_sc.append(spear(np.array(a), np.array(b)))
print('monthly pooled share correlation, calibrated vs classic:')
print(f'{"month":>7} | {"cal P":>6} {"cal S":>6} | {"cla P":>6} {"cla S":>6}')
for t in [0, 3, 6, 11, 23, 35, 38]:
    y, m = months[t]
    print(f'{y}-{m:02d} | {mm_p[t]:6.3f} {mm_s[t]:6.3f} | {mm_pc[t]:6.3f} {mm_sc[t]:6.3f}')
print(f'calibrated: median P {np.median(mm_p):.3f} (range {np.min(mm_p):.3f}..'
      f'{np.max(mm_p):.3f}) | classic median P {np.median(mm_pc):.3f}')"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(range(T), mm_p, 'o-', ms=3, lw=1, color='C0', label='calibrated kernel (Pearson)')
ax.plot(range(T), mm_s, 's--', ms=3, lw=1, color='C0', alpha=0.55, label='calibrated (Spearman)')
ax.plot(range(T), mm_pc, 'o-', ms=3, lw=1, color='C3', alpha=0.8, label='classic kernel (Pearson)')
ax.axhline(np.median(mm_p), color='C0', lw=0.6, ls=':')
ax.axhline(np.median(mm_pc), color='C3', lw=0.6, ls=':')
ax.set_xticks(range(0, T, 6))
ax.set_xticklabels([f'{y}-{m:02d}' for y, m in months[::6]], rotation=45, fontsize=8)
ax.set_ylabel('pooled outbound-share correlation')
ax.set_title('Monthly directional-share fit: calibrated vs classic gravity kernel')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ASSETS}/fig70_appA_share_corr_monthly.png'); plt.show()
show_fig(f'{ASSETS}/fig70_appA_share_corr_monthly.png')"""))

cells.append(code("""# per-state median Pearson (calibrated) vs classic
def per_state_medians(Wk):
    out_l, ms_used = [], []
    for i in np.where(active)[0]:
        ps = []
        for t in range(T):
            o = F[i, :, t]; tot = o.sum()
            if tot <= 0:
                continue
            rv = pear(o / tot, Wk[i])
            if not np.isnan(rv):
                ps.append(rv)
        if ps:
            out_l.append((states[i], np.median(ps), np.min(ps), np.max(ps), len(ps)))
    return out_l

med_cal = per_state_medians(W_cal)
med_cla = per_state_medians(W_classic)
med_cal_v = np.array([r[1] for r in med_cal])
med_cla_v = np.array([r[1] for r in med_cla])
print('per-state median Pearson (calibrated): median %.3f | IQR [%.3f, %.3f] | '
      'range [%.3f, %.3f]' % (np.median(med_cal_v),
                              np.percentile(med_cal_v, 25), np.percentile(med_cal_v, 75),
                              med_cal_v.min(), med_cal_v.max()))
print('  (classic comparison: median %.3f)' % np.median(med_cla_v))
worst = sorted(med_cal, key=lambda t: t[1])[:3]
best = sorted(med_cal, key=lambda t: -t[1])[:3]
print('  worst 3:', [(CODE_TO_NAME[s], round(v, 3)) for s, v, _, _, _ in worst])
print('  best  3:', [(CODE_TO_NAME[s], round(v, 3)) for s, v, _, _, _ in best])

ps_df = pd.DataFrame(
    [{'state': s, 'name': CODE_TO_NAME[s], 'months_used': mu,
      'medPearson_cal': v, 'minPearson_cal': mn, 'maxPearson_cal': mx}
     for s, v, mn, mx, mu in med_cal])
clad = {s: v for s, v, _, _, _ in med_cla}
ps_df['medPearson_classic'] = ps_df['state'].map(clad)
ps_df = ps_df.sort_values('medPearson_cal', ascending=False).reset_index(drop=True)
ps_df.to_csv('../scripts/appendixA_state_corr.csv', index=False)
print('per-state table saved -> scripts/appendixA_state_corr.csv')"""))

cells.append(code("""order = np.argsort(med_cal_v)
fig, ax = plt.subplots(figsize=(7, 11))
ypos = np.arange(len(order))
ax.barh(ypos, med_cla_v[order], height=0.45, color='C3', alpha=0.35, label='classic kernel')
ax.barh(ypos + 0.45, med_cal_v[order], height=0.45, color='C0', alpha=0.85, label='calibrated kernel')
ax.set_yticks(ypos + 0.225)
ax.set_yticklabels([CODE_TO_NAME[med_cal[i][0]] for i in order], fontsize=6.5)
ax.axvline(0, color='grey', lw=0.6)
ax.set_xlabel('median Pearson (outbound shares, 39 months)')
ax.set_title('Per-state directional-share fit: calibrated vs classic')
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ASSETS}/fig71_appA_state_corr.png'); plt.show()
show_fig(f'{ASSETS}/fig71_appA_state_corr.png')"""))

# ── A.5 per-pair + top ────────────────────────────────────────────────────────
cells.append(md(r"""## §A.5 [B2] Per-pair reconstruction · [C] Top destinations

**Reconstruction.** The full predicted tensor
$\hat F_{ij}(t) = s_i(t)\,\mu_{0,i}\,W_{ij}^{\text{grav}}\,T_{\text{mo}}$
(pax/month) vs the observed $F_{ij}(t)$, pooled over all nonzero pairs —
the literal "data restoration" test of the gravity model.

**Top destinations.** Per active state-month, does the kernel name the same
largest destination(s) as the data?"""))
cells.append(code("""pred_cal = (s[:, None, :] * mu0[:, None, None] * W_cal[:, :, None] * T_MO)
m2 = F > 0
a = pred_cal[m2]; b = F[m2]
lg = (a > 0) & (b > 0)
logy = np.log(b[lg]); logp = np.log(a[lg])
r2log = 1 - np.sum((logy - logp) ** 2) / np.sum((logy - logy.mean()) ** 2)
fac = a[lg] / b[lg]
print(f'nonzero pairs (i,j,t): {m2.sum():,}')
print(f'Pearson(raw) = {pear(a, b):.4f} | Spearman = {spear(a, b):.4f}')
print(f'Pearson(log-log) = {pear(logp, logy):.4f} | Spearman(log-log) = {spear(logp, logy):.4f}')
print(f'R2 (log-log) = {r2log:.3f} | median |log2(pred/obs)| = {np.median(np.abs(np.log2(fac))):.3f} '
      f'(median factor error = {2 ** np.median(np.abs(np.log2(fac))):.2f}x)')

pred_cla = (s[:, None, :] * mu0[:, None, None] * W_classic[:, :, None] * T_MO)
a2 = pred_cla[m2]
print(f'classic: Pearson(raw) = {pear(a2, b):.4f} | Spearman = {spear(a2, b):.4f} '
      f'| Pearson(log-log) = {pear(np.log(a2[lg]), logy):.4f}')"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(logp, logy, s=3, alpha=0.35, color='C0')
lo, hi = min(logp.min(), logy.min()), max(logp.max(), logy.max())
ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='1:1')
ax.set_xlabel('log predicted flow  s_i mu0_i W_ij T_mo  (pax/month)')
ax.set_ylabel('log observed flow  F_ij(t)  (pax/month)')
ax.set_title(f'Per-pair reconstruction, calibrated kernel\\n'
             f'Pearson(log) = {pear(logp, logy):.3f} | R2(log) = {r2log:.3f} | N = {len(logp):,}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ASSETS}/fig69_appA_pair_reconstruction.png'); plt.show()
show_fig(f'{ASSETS}/fig69_appA_pair_reconstruction.png')"""))

cells.append(code("""h1_cal, h1_cla, h3_cal = [], [], []
for i in np.where(active)[0]:
    for t in range(T):
        o = F[i, :, t]
        if o.sum() <= 0:
            continue
        h1_cal.append(np.argmax(o) == np.argmax(W_cal[i]))
        h1_cla.append(np.argmax(o) == np.argmax(W_classic[i]))
        s3o = set(np.argsort(o)[-3:]); s3g = set(np.argsort(W_cal[i])[-3:])
        h3_cal.append(len(s3o & s3g) / 3)
print(f'state-months with traffic: {len(h1_cal):,}')
print(f'top-1 destination match  : calibrated {np.mean(h1_cal) * 100:.1f}% '
      f'({np.sum(h1_cal)}/{len(h1_cal)}) | classic {np.mean(h1_cla) * 100:.1f}% '
      f'({np.sum(h1_cla)}/{len(h1_cla)})')
print(f'top-3 destination overlap: calibrated {np.mean(h3_cal) * 100:.1f}%')"""))

# ── A.6 temporal ──────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.6 [D] Uniform temporal scaling

The model's time dependence is a **uniform multiplicative scale per
origin-month**: every pair $i\!\to\!j$ scales by the same $s_i(t)$. If that
holds, pair-level relative flows $\frac{F_{ij}(t)}{F_{ij}(\text{Jan-2020})}$
should track $s_i(t)$ with pooled slope 1.00. This tests the *temporal* half
of the model (independent of the kernel)."""))
cells.append(code("""rs_d, Xd, Yd = [], [], []
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
print(f'origin-dest pairs used: {len(rs_d):,} (Jan flow > 0, >= 5 nonzero months)')
print(f'pair correlation: median {np.median(rs_d):.3f} | IQR '
      f'[{np.percentile(rs_d, 25):.3f}, {np.percentile(rs_d, 75):.3f}] | '
      f'frac > 0.5: {np.mean(rs_d > 0.5) * 100:.1f}% | frac > 0.8: {np.mean(rs_d > 0.8) * 100:.1f}%')
print(f'pooled regression log(F_ij/F_jan) = {bfit:.3f} * log(s_i) + {bres:.3f} '
      f'(expect slope 1.00) | R2 = {r2d:.3f} | N = {len(Xd):,}')"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(Xd, Yd, s=3, alpha=0.3, color='C1')
xs = np.linspace(Xd.min(), Xd.max(), 50)
ax.plot(xs, bfit * xs + bres, 'k-', lw=1.4, label=f'OLS slope {bfit:.3f}')
ax.plot(xs, xs, 'k--', lw=0.8, label='1:1')
ax.set_xlabel('log s_i(t)  (origin mobility scale)')
ax.set_ylabel('log F_ij(t)/F_ij(Jan-2020)  (pair relative flow)')
ax.set_title(f'Uniform temporal scaling test\\nR2 = {r2d:.3f} | N = {len(Xd):,}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ASSETS}/fig72_appA_temporal_scaling.png'); plt.show()
show_fig(f'{ASSETS}/fig72_appA_temporal_scaling.png')"""))

# ── A.7 inbound ───────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.7 [E]·[F] Inbound totals — *uncalibrated* prediction

Inbound is never calibrated: $\hat I_j(t) = \sum_i s_i(t)\mu_{0,i}W_{ij}T_{\text{mo}}$
uses only the **outbound** calibration. Two views:

- **[E] cross-sectional** — per month, predicted vs observed inbound across
  the 56 destinations (does gravity rank the hubs correctly?);
- **[F] time series** — per destination, the 39-month curve (shares the
  national COVID trend, so expect high correlations)."""))
cells.append(code("""pred_in = W_cal.T @ (s * mu0[:, None]) * T_MO      # (n, T) pax/month in
obs_in = F.sum(axis=0)
r_cs = np.array([pear(pred_in[:, t], obs_in[:, t]) for t in range(T)])
r_cs = r_cs[~np.isnan(r_cs)]
print('[E] cross-sectional (per month, across %d dests): median %.3f | IQR '
      '[%.3f, %.3f] | min %.3f | max %.3f | frac > 0.7: %.0f%%'
      % (n, np.median(r_cs), np.percentile(r_cs, 25), np.percentile(r_cs, 75),
         r_cs.min(), r_cs.max(), np.mean(r_cs > 0.7) * 100))
a = pred_in.ravel(); b = obs_in.ravel()
lg = b > 0
print('[E] pooled: Pearson = %.4f | Spearman = %.4f | Pearson(log) = %.4f'
      % (pear(a, b), spear(a, b), pear(np.log(a[lg]), np.log(b[lg]))))

r_ts = []
for j in range(n):
    if obs_in[j].sum() <= 0:
        continue
    rv = pear(pred_in[j], obs_in[j])
    if not np.isnan(rv):
        r_ts.append((states[j], rv))
rv_arr = np.array([v for _, v in r_ts])
print('[F] inbound time series: n = %d | median %.3f | IQR [%.3f, %.3f]'
      % (len(r_ts), np.median(rv_arr), np.percentile(rv_arr, 25), np.percentile(rv_arr, 75)))
r_sorted = sorted(r_ts, key=lambda t: t[1])
print('[F] worst 3:', [(CODE_TO_NAME[s], round(v, 3)) for s, v in r_sorted[:3]])
print('[F] best  3:', [(CODE_TO_NAME[s], round(v, 3)) for s, v in r_sorted[-3:]])"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].plot(range(T), r_cs, 'o-', ms=3, lw=1, color='C5')
axes[0].axhline(np.median(r_cs), color='C5', lw=0.6, ls=':')
axes[0].set_xticks(range(0, T, 6))
axes[0].set_xticklabels([f'{y}-{m:02d}' for y, m in months[::6]], rotation=45, fontsize=8)
axes[0].set_ylabel('cross-sectional Pearson')
axes[0].set_title('[E] Inbound: monthly cross-sectional fit (uncalibrated)')
axes[0].grid(alpha=0.3)
lg2 = obs_in.ravel() > 0
axes[1].scatter(np.log(pred_in.ravel()[lg2]), np.log(obs_in.ravel()[lg2]),
                s=4, alpha=0.4, color='C5')
axes[1].plot([np.log(pred_in.ravel()[lg2]).min(), np.log(pred_in.ravel()[lg2]).max()],
             [np.log(pred_in.ravel()[lg2]).min(), np.log(pred_in.ravel()[lg2]).max()],
             'k--', lw=0.8)
axes[1].set_xlabel('log predicted inbound (pax/month)')
axes[1].set_ylabel('log observed inbound (pax/month)')
axes[1].set_title(f'[E] pooled inbound, log-log\\nPearson = '
                  f'{pear(np.log(pred_in.ravel()[lg2]), np.log(obs_in.ravel()[lg2])):.3f}')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ASSETS}/fig73_appA_inbound.png'); plt.show()
show_fig(f'{ASSETS}/fig73_appA_inbound.png')"""))

# ── A.8 regression ────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.8 [G] Gravity regression

The reduced-form gravity equation on the **raw flows** (no calibration),
pooled and per-month:

$$\log F_{ijt} = \alpha_i + \gamma_t + b_N\log N_j + b_d\log d_{ij} + \varepsilon$$

The estimated $b_N$, $b_d$ tell us what exponents the data itself wants.
The calibrated kernel uses the PPML versions ($b_N \approx 1.0$,
$b_d \approx 0.55$), not these OLS ones — OLS on logs is Jensen-biased
(underestimates magnitude) and is shown for comparison."""))
cells.append(code("""rows_g = np.argwhere(F > 0)
o = rows_g[:, 0]; d = rows_g[:, 1]; tt = rows_g[:, 2]
y = np.log(F[o, d, tt].astype(float))
X = np.column_stack([np.log(np.maximum(D[o, d], 10.0)), np.log(Npop[d]),
                     np.eye(n)[o], np.eye(T)[tt]])
coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
r2g = 1 - ss_res / ss_tot
sigma2 = ss_res / (len(y) - X.shape[1])
se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
print(f'pooled OLS (N={len(y):,}, df={len(y) - X.shape[1]:,}):')
print(f'  b_d = {coef[0]:+.3f} +- {se[0]:.3f}   (kernel exponent used: -{b_d:.2f})')
print(f'  b_N = {coef[1]:+.3f} +- {se[1]:.3f}   (kernel exponent used: {b_N:.2f})')
print(f'  R2  = {r2g:.3f}')
bs_d, bs_n, rs2 = [], [], []
for t in range(T):
    m = tt == t
    Xt = np.column_stack([np.log(np.maximum(D[o[m], d[m]], 10.0)),
                          np.log(Npop[d[m]]), np.eye(n)[o[m]]])
    yt = y[m]
    ct, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
    pt = Xt @ ct
    r2t = 1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    bs_d.append(ct[0]); bs_n.append(ct[1]); rs2.append(r2t)
bs_d = np.array(bs_d); bs_n = np.array(bs_n); rs2 = np.array(rs2)
print(f'per-month: b_d mean {bs_d.mean():+.3f} +- {bs_d.std():.3f} '
      f'(range [{bs_d.min():+.3f}, {bs_d.max():+.3f}])')
print(f'          b_N mean {bs_n.mean():+.3f} +- {bs_n.std():.3f} '
      f'(range [{bs_n.min():+.3f}, {bs_n.max():+.3f}])')
print(f'          R2  mean {rs2.mean():.3f} +- {rs2.std():.3f} '
      f'(range [{rs2.min():.3f}, {rs2.max():.3f}])')"""))

# ── A.9 robustness ────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.9 [H] Robustness — contiguous 49 only

Re-run the directional-share test on the **contiguous 48 states + DC**
(excluding AK, HI and the five territories) to make sure results are not
driven by the small long-haul/territory markets. The kernel is rebuilt and —
for the calibrated variant — the exponents refit on the subset."""))
cells.append(code("""idx = [states.index(c) for c in CONTIGUOUS]
Wc_cla = build_gravity_kernel(CONTIGUOUS, power=POWER, min_distance_km=D_MIN)
r_ppml49 = fit_gravity_exponents(F[np.ix_(idx, idx, range(T))],
                                 D[np.ix_(idx, idx)], Npop[idx], method='ppml')
Wc_cal = build_gravity_kernel(CONTIGUOUS, b_N=r_ppml49['b_N'], b_d=-r_ppml49['b_d'])
print(f'contiguous-49 PPML refit: b_N = {r_ppml49["b_N"]:.3f} | b_d = '
      f'{-r_ppml49["b_d"]:.3f}')

def pooled_contig(Wk):
    x, y = [], []
    for ii, i in enumerate(idx):
        if not active[i]:
            continue
        for t in range(T):
            o = F[i, idx, t]; tot = o.sum()
            if tot <= 0:
                continue
            x += list(Wk[ii]); y += list(o / tot)
    return np.array(x), np.array(y)

a, b = pooled_contig(Wc_cla)
lg = (a > 0) & (b > 0)
print(f'classic    : pooled P = {pear(a, b):.4f} | S = {spear(a, b):.4f} | '
      f'log-P = {pear(np.log(a[lg]), np.log(b[lg])):.4f} | N = {len(a):,}')
a, b = pooled_contig(Wc_cal)
print(f'calibrated : pooled P = {pear(a, b):.4f} | S = {spear(a, b):.4f} | N = {len(a):,}')
print('(full-56 reference: classic P = 0.35, calibrated P = 0.45, §A.4)')"""))

# ── A.10 summary ──────────────────────────────────────────────────────────────
cells.append(md(r"""## §A.10 Reconstruction summary & limitations

**Does the gravity model reproduce the BTS aviation flows?** Partly —
the numbers separate cleanly into a strong temporal half and a moderate
spatial half:

| What | Result |
|---|---|
| outbound totals (calibrated) | exact to 1e-16 (by construction) |
| uniform temporal scaling | pair-correlation median **0.87**, pooled slope **1.15** vs 1.00 |
| directional shares (pooled) | calibrated **P 0.45 / S 0.58** vs classic 0.35 / 0.46 |
| directional shares (per-state median) | **0.60** (IQR 0.49–0.78) |
| per-pair reconstruction | Pearson(raw) **0.82**; log-log **0.60**, R²(log) 0.25, median error ×2.6 |
| top-1 destination | calibrated **34.5%** vs classic 20.0% |
| inbound (uncalibrated), cross-sectional | median **0.83** per month |
| inbound (uncalibrated), time series | median **0.98** (shares the COVID trend) |
| gravity regression | $b_N \approx 1.10$, $b_d \approx -0.72$, R² 0.41; stable across months |
| contiguous-49 robustness | calibrated P **0.56**, classic 0.28 |

**Interpretation.** Population and distance alone (plus per-state totals)
recover the *rank order* of the directional structure and almost all of the
*temporal* evolution; but the residual spatial scatter is large (median
per-pair error ×2.6), and the top-1 destination is right only a third of
the time. The PPML-calibrated kernel is a clear improvement over
$\sqrt{N_iN_j}/d^2$ on every axis.

**Limitations (honest).**

1. **Air ≠ all travel.** BTS T-100 covers US air carriers only. Most
   inter-state travel — and most COVID-19 seeding — is by **car**
   (FHWA/StreetLight data show autos dominate the state-pair volumes and
   collapsed differently during the pandemic). The calibrated $s_i(t)$ is
   an *aviation* mobility scale; the metapopulation uses it as a proxy for
   all human mobility (see also notebook 08 §8.11 item 4).
2. **Calibrated totals carry no information** — the experiment's power lies
   in the directional tests, which are moderate.
3. **Distance metric**: state centroid distances with a 100 km floor;
   intra-state hubs (airport pairs) would sharpen the kernel.
4. **Monthly granularity** — March 2020's aviation collapse enters only on
   the monthly grid.
5. The inbound time series ([F], median 0.98) is **not** evidence of
   predictive power: every state shares the same national COVID trend, so
   the curves are collinear."""))
cells.append(code("""pp = pred_cal[m2]; ff = F[m2]
mm = (pp > 0) & (ff > 0)
summary = pd.DataFrame({
    'metric': [
        'outbound totals (max rel err)',
        'share P pooled (cal/classic)', 'share S pooled (cal/classic)',
        'share P per-state median',
        'per-pair P raw (cal)', 'per-pair P log-log (cal)', 'per-pair R2 log (cal)',
        'median |log2 factor err| (cal)',
        'top-1 dest (cal/classic)',
        'temporal pair-corr median', 'temporal pooled slope',
        'inbound cross-sectional median', 'inbound time-series median',
        'OLS b_N (pooled)', 'OLS b_d (pooled)', 'OLS R2 (pooled)',
        'contiguous-49 P (cal/classic)',
    ],
    'value': [
        '%.1e' % np.max(err[active] / np.maximum(out[active], 1)),
        '%.3f / %.3f' % (share_summary['P_pooled'][1], share_summary['P_pooled'][0]),
        '%.3f / %.3f' % (share_summary['S_pooled'][1], share_summary['S_pooled'][0]),
        '%.3f' % np.median(med_cal_v),
        '%.3f' % pear(pp, ff),
        '%.3f' % pear(np.log(pp[mm]), np.log(ff[mm])),
        '%.3f' % r2log,
        '%.2f' % np.median(np.abs(np.log2(pp[mm] / ff[mm]))),
        '%.1f%% / %.1f%%' % (np.mean(h1_cal) * 100, np.mean(h1_cla) * 100),
        '%.3f' % np.median(rs_d), '%.3f' % bfit,
        '%.3f' % np.median(r_cs), '%.3f' % np.median(rv_arr),
        '%.3f' % coef[1], '%.3f' % coef[0], '%.3f' % r2g,
        '%.3f / %.3f' % (pear(pooled_contig(Wc_cal)[0], pooled_contig(Wc_cal)[1]),
                         pear(pooled_contig(Wc_cla)[0], pooled_contig(Wc_cla)[1])),
    ],
})
from IPython.display import display
display(summary)
print('appendix A complete | figures fig68..fig73 -> assets/')"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14'},
}
out = os.path.join(ROOT, 'notebooks', 'appendix_A.ipynb')
with open(out, 'w') as f:
    f.write(nbf.writes(nb, indent=1))
print('wrote', out, '| cells:', len(cells))
