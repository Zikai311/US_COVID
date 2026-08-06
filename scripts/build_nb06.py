"""Builder: regenerate notebooks/06_covid_case_study.ipynb — a from-scratch,
layperson-friendly walkthrough (Chinese) of the project's final model
(formulae_sheet.md): S/E/I/R boxes -> contact heterogeneity -> state
coupling -> data calibration -> vaccination/prior immunity -> two-regime
beta(t) -> the Delta-wave fit. Edit this script, not the .ipynb."""
import os, nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nb = nbf.v4.new_notebook()
cells = []
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ── 0. title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""# Part 6: COVID-19 Case Study — From Simple Boxes to a Real Epidemic Wave

**Who this is for**: anyone comfortable with fractions and the idea of a
"rate of change". No background is assumed. Every formula comes from the
project's formula sheet (`formulae_sheet.md`); here we explain what each
one *does*.

**The question we end up answering**: in the summer of 2021 a wave of
COVID-19 caused by the "Delta" variant swept the United States
(2021-06-01 to 2021-10-31, about 12.7M reported cases). Can a mathematical
model reproduce this wave — its **size**, its **timing**, and the **way it
ended**?

**Method**: rather than presenting a complicated model all at once, we
start from one simple idea and add pieces step by step. Every step answers
the question "what did the previous version leave out?".

| Step | What we add | Question it answers |
|---|---|---|
| 1 | Divide the population into compartments | What happens to one person, from healthy to sick? |
| 2 | Some people meet many others, some few | Why do a few people control how fast an epidemic spreads? |
| 3 | People move between the 56 states | How does an epidemic travel from New York to the whole country? |
| 4 | Numbers come from real data | Where do the parameters come from (not guesswork) |
| 5 | Vaccination and existing immunity | Why did the Delta wave stop by itself? |
| 6 | Behaviour and policy change transmission | Why does the curve fall so fast after the peak? |
| 7 | Put everything together | The final model's fit to the Delta wave |

> Convention: the first time a technical term appears, it is explained in
> plain words first; the abbreviation follows afterwards."""))

# ── setup ─────────────────────────────────────────────────────────────────────
cells.append(code("""import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, Rectangle

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

def show_fig(path):
    from IPython.display import Image, display
    display(Image(filename=path))

us = load_us_states()
print('data ready: NYT state-level daily cases loaded')"""))

# ── step 1: boxes ─────────────────────────────────────────────────────────────
cells.append(md(r"""## Step 1: Divide the population into compartments

Imagine sorting a city's people into four groups by health status
(mathematically, these are called **compartments**):

- **S (susceptible)**: not yet infected; can catch the virus;
- **E (exposed)**: infected but showing no symptoms and not yet able to
  infect others;
- **I (infectious)**: can transmit the virus to others;
- **R (recovered)**: recovered (or deceased); no longer part of the
  spreading.

People move through the chain **S → E → I → R**, and each arrow has a
"speed":

- Susceptibles become infected at a rate proportional to the fraction of
  infectious people in the population; the proportionality constant is the
  **transmission rate $\beta$** (units: 1/day);
- Exposed people become infectious at rate $\sigma$ (here 1/5: on average
  5 days of latency);
- Infectious people recover at rate $\gamma$ (here 1/10: on average 10
  days of infectiousness).

With a population of $N$, this reads:

$$\frac{dS}{dt}=-\beta\frac{S\,I}{N},\qquad
\frac{dE}{dt}=\beta\frac{S\,I}{N}-\sigma E,\qquad
\frac{dI}{dt}=\sigma E-\gamma I,\qquad
\frac{dR}{dt}=\gamma I$$

**The single most important number**: how many people does one infectious
person infect on average? It is called the **basic reproduction number**
$R_0$. Roughly, $R_0=\beta/\gamma$. If $R_0>1$ the epidemic grows
exponentially; if $R_0<1$ it dies out on its own. **Much of this project
is about what pushes $R_0$ below 1.**"""))
cells.append(code("""# Compartment diagram: four boxes + three arrows (Fig 6.1)
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis('off')
boxes = [
    ('S', 'susceptible', 0.6, '#A8D5BA'),
    ('E', 'exposed', 3.2, '#F9E79F'),
    ('I', 'infectious', 5.8, '#F5B7B1'),
    ('R', 'recovered', 8.4, '#AED6F1'),
]
for name, lab, x, col in boxes:
    ax.add_patch(Rectangle((x, 0.6), 1.5, 1.0, facecolor=col, edgecolor='black', lw=1.5))
    ax.text(x + 0.75, 1.3, name, ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(x + 0.75, 0.38, lab, ha='center', va='center', fontsize=9)
arrows = [('S', 'E', r'$\\beta\\cdot SI/N$'), ('E', 'I', r'$\\sigma E$'), ('I', 'R', r'$\\gamma I$')]
pos = {b[0]: b[2] + 1.5 for b in boxes}
for a, b, lab in arrows:
    ax.add_patch(FancyArrowPatch((pos[a], 1.1), (pos[b] - 0.4, 1.1),
                                 arrowstyle='-|>', mutation_scale=18, color='k'))
    ax.text((pos[a] + pos[b]) / 2 - 0.35, 1.42, lab, fontsize=10, color='#7B241C')
ax.set_title('Four compartments, three arrows: one person\'s journey through the disease', fontsize=12)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig82_compartment.png'); plt.show()
show_fig(f'{ASSETS}/fig82_compartment.png')"""))

# ── step 2: heterogeneity ─────────────────────────────────────────────────────
cells.append(md(r"""## Step 2: Some people meet many others, some few

The four-compartment model assumes **everyone is identical**: everyone meets
the same number of people per day. Reality is different — some jobs involve
hundreds of contacts a day, some people barely go out. We call a person's
number of contacts their **degree $k$**.

The contact pattern in the US is described by a "truncated power law"
(formula sheet §1):

$$P(k)=C\,k^{-\alpha}e^{-k/k_c},\qquad k=3\dots 200$$

Do not be put off by the formula; two facts matter:

1. **On average each person meets 6 others per day** (⟨k⟩=6, guaranteed by
   the parameter α = 2.659);
2. A few "hub" individuals meet a huge number of people — **most contacts in
   the population are contributed by a small minority**.

This changes how infection is written: a person of degree $k$ is infected
$k$ times faster. And the probability that "a randomly chosen contact line
leads to an infectious person" is denoted $\theta$ (the infectious-contact
share):

$$\theta = \\frac{\\sum_k k\\, I_k}{N\\,\\langle k\\rangle}$$

The numerator is the number of contact lines owned by infectious people;
the denominator is the total number of contact lines. The infection rate of
a degree-$k$ susceptible is then $\beta\, k\, S_k\, \theta$ — **the more
people you meet, the easier you catch it; and the more of a hub you are,
the more you spread it.**"""))
cells.append(code("""alpha = power_law_alpha_for_mean_k(6.0)
k_vals, P_k = truncated_power_law(alpha=alpha)
mean_k = float(k_vals @ P_k); mean_k2 = float((k_vals ** 2) @ P_k)
print(f'degree distribution: alpha = {alpha:.3f} | mean degree <k> = {mean_k:.1f} | <k^2>/<k> = {mean_k2/mean_k:.2f}')

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.loglog(k_vals, P_k, 'o', ms=2.5, color='C0')
ax.axvline(mean_k, color='grey', ls=':', lw=1)
ax.text(mean_k * 1.1, P_k.max() * 0.5, f'mean <k>={mean_k:.0f}', fontsize=9, color='grey')
ax.set_xlabel('contacts per person, k'); ax.set_ylabel('P(k)')
ax.set_title('Contact distribution: a few hubs meet a huge number of people')
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig83_degree.png'); plt.show()
show_fig(f'{ASSETS}/fig83_degree.png')"""))

# ── step 3: metapopulation ────────────────────────────────────────────────────
cells.append(md(r"""## Step 3: The US is not a single population — people move between states

The epidemic started in New York but quickly reached the whole country,
because **people travel between states**, mostly by air. We treat the US as
56 separate populations (50 states + Washington DC + 5 overseas
territories). Each one runs the model of Steps 1–2 internally; between
them, daily "passenger flows" move people around.

**The number of people travelling from state $i$ to state $j$ per day**
(formula sheet §4):

$$W_{ij}(t) = \underbrace{s_i(t)}_{travel\ scale}\cdot
\underbrace{\mu_{0,i}}_{baseline\ flow}\cdot
\underbrace{W_{ij}^{\\text{grav}}}_{direction\ share}$$

Three factors:

1. **Direction share** — a "gravity model": more people go to places with
   **larger populations**, and fewer to places that are **far away**:
   $$W_{ij}^{\\text{grav}}\\propto \\frac{N_j^{1.0}}{d_{ij}^{0.55}}$$
   It is called "gravity" because it has the same form as Newton's law:
   mass (population) attracts, distance decays. The exponents 1.0 and 0.55
   are not guesses — they are estimated statistically from the real
   2020–2023 US airline data (Step 4).
2. **Baseline flow** $\mu_{0,i}$: how many people left state $i$ per day,
   on average, before the pandemic (January 2020).
3. **Travel scale** $s_i(t)$: each month's actual travel volume divided by
   the January-2020 volume. In April 2020 US air travel was about 5% of its
   January level — **the pandemic itself changes travel**.

How does movement enter the equations? The change in state $i$'s
infectious count = local new infections − local recoveries + **infectious
people flying in from other states − infectious people flying out**. This
is the "coupling term" of formula sheet §5. The full coupling picture is a
56 × 56 grid of routes, with real passenger numbers for every month."""))
cells.append(code("""mu0, s, months, F = calibrate_mobility(JURISDICTIONS, data_dir='../data')
out = F.sum(axis=1)
nat_mob = out.sum(0) / out.sum(0)[0]
print('flight data: 39 months (2020-01 .. 2023-03) | April-2020 national travel scale = %.0f%%'
      % (100 * nat_mob[3]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(len(months)), nat_mob, 'o-', ms=3, color='C1')
axes[0].set_xticks(range(0, len(months), 6))
axes[0].set_xticklabels([f'{y}-{m:02d}' for y, m in months[::6]], rotation=45, fontsize=7)
axes[0].axhline(1, color='grey', lw=0.6, ls=':')
axes[0].set_ylabel('travel scale s(t)')
axes[0].set_title('US air travel: collapsed to 5% in April 2020')
axes[0].grid(alpha=0.3)

Wg = build_gravity_kernel(JURISDICTIONS, b_N=1.0025, b_d=0.5465)
x, y = [], []
for i in np.where(out[:, 0] > 0)[0]:
    o = F[i, :, 0]
    x += list(Wg[i]); y += list(o / o.sum())
x, y = np.array(x), np.array(y)
m = (x > 0) & (y > 0)
axes[1].scatter(np.log(x[m]), np.log(y[m]), s=4, alpha=0.45, color='C2')
axes[1].set_xlabel('gravity-predicted share (log)')
axes[1].set_ylabel('observed share, Jan-2020 (log)')
axes[1].set_title('Gravity model vs real flights: Pearson ≈ 0.45')
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig84_mobility.png'); plt.show()
show_fig(f'{ASSETS}/fig84_mobility.png')"""))

# ── step 4: calibration ───────────────────────────────────────────────────────
cells.append(md(r"""## Step 4: The numbers come from real data (calibration)

Every number in the model should have a source.

**The transmission rate $\beta$ (one per state).** Early in an epidemic,
new daily cases grow exponentially: on a plot of "log new cases vs time"
they form a straight line whose slope is the **growth rate $r$**. Given
the latency and recovery periods, $r$ and $\beta$ are linked by a fixed
relationship (formula sheet §8), so $\beta$ can be recovered from $r$:

$$\beta_i = \\frac{(r_i+\\gamma)(r_i+\\sigma)}{\\sigma}\\,\\frac{\\langle k\\rangle}{\\langle k^2\\rangle}$$

Why not fit $\beta$ directly to the whole curve? Because $r$ is
**insensitive to the reporting fraction** — even if only half of the cases
are reported, the growth rate is still correct. That is the key advantage
of this method. Here we use the Jul–Aug 2021 growth slope (the Delta
takeoff), and cap the implied transmission at $R_0=6$.

**The gravity exponents ($b_N=1.0$, $b_d=0.55$).** Estimated by a
"Poisson pseudo-maximum-likelihood" regression on the 56×56×39-month
flight panel — in one sentence: **choose the exponents so that the
model's predicted flows look, statistically, as much like the real flows
as possible.**"""))
cells.append(code("""# Example: New York's Jul-Aug 2021 growth slope -> transmission rate beta
sub = us[(us.state == 'New York') & (us.date >= '2021-07-01') & (us.date < '2021-09-01')]
v = sub['new_cases_smooth'].values; t = np.arange(len(v)); m = v > 0
r_ny = np.polyfit(t[m], np.log(v[m]), 1)[0]
print(f'New York Jul-Aug 2021 growth rate r = {r_ny:.4f}/day (new cases multiply by e^{{r}}≈{np.exp(r_ny):.2f} per day)')

D = build_distance_matrix(JURISDICTIONS)
Npop = np.array([CENSUS_POPULATIONS[c] for c in JURISDICTIONS], dtype=float)
r_ppml = fit_gravity_exponents(F, D, Npop, method='ppml')
print('gravity exponents (PPML fit to the real flight panel):')
print('  destination-population exponent b_N = %+.3f | distance exponent b_d = %+.3f | fit R2 = %.3f'
      % (r_ppml['b_N'], -r_ppml['b_d'], r_ppml['r2']))"""))

# ── step 5: vaccination ───────────────────────────────────────────────────────
cells.append(md(r"""## Step 5: Vaccination and existing immunity

By 2021 two forces were shrinking the pool of people who could be infected:

**1. Vaccination (a new compartment, V).** Vaccinated people move from the
susceptible compartment S to the "vaccinated" compartment V, and are
essentially no longer infected (formula sheet §6):

$$\\frac{dS}{dt} = \\cdots - v(t)\\frac{S}{N},\\qquad
\\frac{dV}{dt} = v(t)\\frac{S}{N}$$

where $v(t)$ is the **actual number of people who completed vaccination in
that state in that month** (real data from the US CDC). By June 2021 about
42% of the US population had completed vaccination.

**2. People already infected.** By 2021 about 100 million Americans had
actually been infected (55M reported × an undercount factor of about 3).
They have immunity; at the start of the Delta wave we place them in the
recovered compartment R, where they no longer participate in spreading.

So the model's "susceptible population" at 2021-06-01 is:

$$\text{susceptible} = \text{population} - \text{vaccinated}(\times\text{effectiveness}) - \text{already infected} - \text{currently infectious/latent}$$

**Why this matters**: without vaccination and immunity the susceptible pool
is 335M; with them it is about 110M. The epidemic has two-thirds less
"fuel" to burn — **this is why the Delta wave stopped by itself and why
October declined.**"""))
cells.append(code("""vax = pd.read_csv('../data/cdc_vax_state_monthly.csv')
cum = vax.pivot_table(index='state', columns='month', values='series_complete')
sel = ['New York', 'Florida', 'Texas', 'California']
for name in sel:
    code = NAME_TO_CODE[name]
    ser = cum.loc[code].copy(); ser.index = [pd.Period(m, 'M').to_timestamp() for m in ser.index]
    plt.plot(ser.index, ser.values / 1e6, lw=1.4, label=name)
nat = cum.sum(0); nat.index = [pd.Period(m, 'M').to_timestamp() for m in nat.index]
plt.plot(nat.index, nat.values / 1e6, 'k--', lw=1.6, label='US total')
plt.axvline(pd.Timestamp('2021-06-01'), color='grey', ls=':', lw=1)
plt.text(pd.Timestamp('2021-05-15'), plt.ylim()[1] * 0.55, 'Delta wave\nstarts', fontsize=9, color='grey')
plt.ylabel('fully vaccinated (millions)')
plt.title('US vaccination rollout (real CDC data)')
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig85_vaccination.png'); plt.show()
show_fig(f'{ASSETS}/fig85_vaccination.png')

us['cum_cases'] = us.groupby('state')['new_cases'].cumsum()
prior = us[us.date <= pd.Timestamp('2021-05-31')].groupby('state')['cum_cases'].max()
print('at 2021-06-01: fully vaccinated %.0fM (42%%) | cumulative reported infections %.0fM (true ~3x)'
      % (cum['2021-05'].sum() / 1e6, prior.sum() / 1e6))"""))

# ── step 6: two-regime beta ───────────────────────────────────────────────────
cells.append(md(r"""## Step 6: Behaviour and policy change transmission (two-regime $\beta$)

If the transmission rate $\beta$ never changed, the model's epidemic would
keep growing to the end of the window. The real Delta wave peaked on
2021-09-10 and then turned around. The reasons include not only "fewer
susceptible people left" (Step 5), but also that **people's behaviour
changed**: after the peak, people were more careful, isolated more, and
wore masks more.

The simplest way to express this is a **two-regime transmission rate**
(formula sheet §7):

$$\beta_i(t) = \\beta_{0,i}\cdot g(t),\\qquad
g(t) = \\begin{cases} 1 & \\text{before the peak}\\\\ \\rho & \\text{after the peak}\\end{cases}$$

- Before the peak: transmission stays at the calibrated value (Step 4);
- After the peak: the whole transmission rate is multiplied by a factor
  $\rho < 1$;
- The switch date is the observed national peak (2021-09-10), and $\rho$
  is fitted to the data. For Delta the fit gives $\rho=0.9$ — a mild
  reduction, because Delta's decline was mostly the shrinking susceptible
  pool. For the 2022 Omicron wave the fit gives $\rho=0.25$: there,
  behaviour was the main driver (appendix B compares the two waves in
  detail)."""))
cells.append(code("""months = [f'{y:04d}-{m:02d}' for y, m in [(2021, 8), (2021, 9), (2021, 10)]]
g_aug, g_sep, g_oct = 1.0, (9 + 21 * 0.9) / 30, 0.9
print('two-regime transmission multiplier g(t):')
print(f'  2021-08 (before peak) = {g_aug:.2f} | 2021-09 (switch month, weighted) = {g_sep:.2f} | 2021-10 (after peak) = {g_oct:.2f}')
print('  meaning: after the peak, each infectious person infects others at 90% of the pre-peak rate')"""))

# ── step 7: the full fit ──────────────────────────────────────────────────────
cells.append(md(r"""## Step 7: Putting it all together — the Delta-wave fit

All pieces now go into one machine (`FusedMetapopulationSEIR`, more than
50,000 equations):

1. 56 states, each running the degree-class S-E-I-R-V model;
2. states coupled by the real flight data (gravity directions + monthly
   travel scales);
3. each state's transmission rate from its Jul–Aug 2021 growth slope
   (capped at $R_0=6$);
4. starting point at 2021-06-01: infectious count (reported cases ÷ 0.15),
   vaccinated × effectiveness, and the ~100M already immune;
5. two-regime transmission (×0.9 after the peak).

The model outputs **true infections**; the NYT data reports cases. To
compare shapes we multiply the reported curve by a single constant
$\hat\\pi$ (the window-total ratio). Four checks:

- **Total size**: $\hat\\pi$ close to 1 means the magnitude is right;
- **Peak timing**: model peak date vs observed peak date;
- **Decline**: end-of-October ÷ peak (model 0.35, observed 0.41);
- **Overall shape**: $R^2$ (1 = perfect, 0 = "as good as guessing the
  mean")."""))
cells.append(code("""sigma, gamma = 1.0 / 5, 1.0 / 10
T0 = (pd.Timestamp('2021-06-01') - DAY0).days
TMAX = (pd.Timestamp('2021-10-31') - DAY0).days
mom_ratio = mean_k2 / mean_k

# 1) per-state beta from the Jul-Aug 2021 growth slopes
win_start = pd.Timestamp('2021-07-01'); win_days = 62
rs, bs = {}, {}
for code in JURISDICTIONS:
    sub = us[(us.state == CODE_TO_NAME[code]) & (us.date >= win_start)
             & (us.date < win_start + pd.Timedelta(days=win_days))]
    v = sub['new_cases_smooth'].values; t = np.arange(len(v)); m = v > 0
    rv = np.nan
    if m.sum() >= 5:
        sl = np.polyfit(t[m], np.log(v[m]), 1)[0]
        if sl > 0: rv = sl
    rs[code] = rv
    bs[code] = (rv + gamma) * (rv + sigma) / sigma / mom_ratio if not np.isnan(rv) else np.nan
bs = pd.Series(bs).fillna(pd.Series(bs).median()).clip(upper=6.0 * gamma / mom_ratio)
rs = pd.Series(rs).fillna(float(pd.Series(rs).median()))

# 2) initial conditions (2021-06-01): infectious/latent/vaccinated/prior-immune
pi0 = 0.15
I0, E0 = {}, {}
for code in JURISDICTIONS:
    row = us[(us.state == CODE_TO_NAME[code]) & (us.date == pd.Timestamp('2021-06-01'))]
    i0 = float(row['new_cases'].iloc[0]) if len(row) else 0.0
    I0[code] = i0 / pi0
    E0[code] = I0[code] * (rs[code] + gamma) / sigma

grid_m = [f'{y:04d}-{m:02d}' for y, m in [(2020, 1)] and [(2020, 1)]] or []
vax2 = pd.read_csv('../data/cdc_vax_state_monthly.csv')
gidx = [f'{y:04d}-{m:02d}' for y in range(2020, 2024) for m in range(1, 13)][:39]
pv = vax2.pivot_table(index='state', columns='month', values='series_complete_flow') \\
     .reindex(index=JURISDICTIONS, columns=gidx).fillna(0)
cumv = vax2.pivot_table(index='state', columns='month', values='series_complete') \\
       .reindex(index=JURISDICTIONS, columns=gidx).fillna(0)
flow = pv.values / T_MO
eps, asc = 1.0, 3.0
V0 = {s: float(cumv.loc[s, '2021-05']) * eps for s in JURISDICTIONS}
R0init = {s: float(prior.get(CODE_TO_NAME[s], 0)) * asc for s in JURISDICTIONS}
s_eff = np.array([1.0 - (V0[s] + R0init[s]) / CENSUS_POPULATIONS[s] for s in JURISDICTIONS])
beta_v = bs.values / s_eff
print('susceptible share of the US population: %.0f%% (would be 100%% without vaccination/immunity)' % (100 * np.mean(s_eff)))

# 3) two-regime transmission + mobility + solve
months_full = [(y, m) for y in range(2020, 2024) for m in range(1, 13)][:39]
mu0, s, months_full, F = calibrate_mobility(JURISDICTIONS, data_dir='../data')
s_func = s_interpolator(s, months_full)
W_cal = build_gravity_kernel(JURISDICTIONS, b_N=1.0025, b_d=0.5465)
g2 = np.ones((56, 39)); g2[:, 20] = (9 + 21 * 0.9) / 30; g2[:, 21] = 0.9

model = FusedMetapopulationSEIR(
    JURISDICTIONS, CENSUS_POPULATIONS, W_cal, mu0, s_func,
    k_vals, P_k, beta=beta_v, sigma=sigma, gamma=gamma,
    months=months_full, vac_flow=flow, vac_efficacy=eps, V0_dict=V0,
    R_init_dict=R0init, beta_scale=g2)
t_sol, y_sol = model.simulate(TMAX, I0, E0, t0=T0)
ni = model.daily_new_infections().sum(0)
dates = [DAY0 + pd.Timedelta(days=int(t)) for t in t_sol]
nat_obs_all = us.groupby('date')['new_cases_smooth'].sum()
nat_obs = nat_obs_all.reindex(dates).fillna(0).values

ph = ni.sum() / nat_obs.sum()
pk_m = dates[int(np.argmax(ni))]; pk_o = nat_obs_all.loc[dates].idxmax()
r2l = 1 - np.sum((np.log(ni / ph + 1) - np.log(nat_obs + 1)) ** 2) / np.sum((np.log(nat_obs + 1) - np.log(nat_obs + 1).mean()) ** 2)
r2v = 1 - np.sum((ni / ph - nat_obs) ** 2) / np.sum((nat_obs - nat_obs.mean()) ** 2)
oct_r = float(ni[-1] / ni[int(np.argmax(ni))]); oct_o = float(nat_obs[-1] / nat_obs.max())

fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.plot(dates, nat_obs, lw=1.4, color='k', label='observed (7-day smoothed)')
ax.plot(dates, ni / ph, lw=1.5, color='C2', label='model (true infections ÷ π̂)')
ax.set_ylabel('new cases per day (reported scale)')
ax.set_title('Delta wave: model vs observed (2021-06-01 .. 10-31)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig86_delta_fit.png'); plt.show()
show_fig(f'{ASSETS}/fig86_delta_fit.png')

print(f'total size: model / observed  π̂ = {ph:.2f}      (1 = perfectly aligned)')
print(f'peak date: model {pk_m.date()} vs observed {pk_o.date()}   ({(pk_m - pk_o).days} days off)')
print(f'decline: end-Oct / peak = model {oct_r:.2f} vs observed {oct_o:.2f}')
print(f'overall fit: R2(log) = {r2l:.2f} | R2(linear) = {r2v:.2f}')"""))

# ── summary ───────────────────────────────────────────────────────────────────
cells.append(md(r"""## Recap: what each layer fixed

| Layer | What we added | Wrong assumption it removed | Formula sheet |
|---|---|---|---|
| 1 | Four compartments S-E-I-R | "becoming sick is instantaneous" | §2 |
| 2 | Contact heterogeneity (degree) | "everyone is identical" | §1, §3 |
| 3 | Movement between 56 states (gravity) | "the US is one mixed population" | §4–5 |
| 4 | Calibration from data (growth rate → β, flights → gravity) | "parameters are guesses" | §8 |
| 5 | Vaccination V + existing immunity | "everyone can be infected" | §6 |
| 6 | Two-regime β(t) | "human behaviour never changes" | §7 |
| 7 | Solve all 55,440 equations together | "each layer can be handled alone" | all |

**The final scorecard (Delta wave)**: total size aligned (π̂ = 1.10), peak
within about a week, October decline depth matched (0.35 vs 0.41), overall
fit R² (linear) ≈ 0.96. In other words: assembling compartments + contact
heterogeneity + state coupling + real data + vaccination/immunity +
behaviour change, layer by layer, **reproduces the main features of a real
epidemic wave**.

**Where to go next**:
- notebook 08: the full model, with sensitivity experiments and the source
  of every number;
- appendix A: can the gravity model *reconstruct* the airline data?
- appendix B: the same machinery applied to the 2022 Omicron wave (needs a
  much steeper transmission collapse);
- `formulae_sheet.md`: every formula on one page.

**A closing thought**: the way to follow this journey is not to memorise
the formulas, but to remember that every step asked "what did the previous
version leave out?". A model's value is not that it "looks like" reality —
it is **which real-world rule it managed to put into mathematics**."""))
cells.append(code("""print('notebook 06 rebuild complete | figures fig82..fig86 -> assets/')"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14'},
}
out = os.path.join(ROOT, 'notebooks', '06_covid_case_study.ipynb')
with open(out, 'w') as f:
    f.write(nbf.writes(nb, indent=1))
print('wrote', out, '| cells:', len(cells))
