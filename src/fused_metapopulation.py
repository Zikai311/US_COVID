"""
fused_metapopulation.py — Fused degree-based SEIR × gravity metapopulation
(56 US jurisdictions), per formulae_sheet.md §1–§7 (PART II, 2026-08-05).

Model summary
-------------
Each node i ∈ 56 jurisdictions (50 states + DC + PR/GU/VI/MP/AS) runs a
degree-based SEIR over degree classes k with distribution
P(k) ∝ k^{-α} e^{-k/k_c}.  Nodes are coupled by a directional, time-varying
gravity mobility matrix

    W_ij(t) = s_i(t) · μ_{0,i} · W_ij^gravity     [travellers/day]

with W^gravity row-normalised, μ_{0,i} = Jan-2020 total outflow / T_mo
(daily units, T_mo = 30.44 d), and s_i(t) = monthly outflow / Jan-2020
outflow (dimensionless).

Within-state (counts per class k):
    dS_ik/dt = -β_i k S_ik θ_i
    dE_ik/dt =  β_i k S_ik θ_i - σ E_ik
    dI_ik/dt =  σ E_ik - γ I_ik
    dR_ik/dt =  γ I_ik
    θ_i      = (1/<k>) Σ_k k P(k) I_ik / N_i        (dimensionless)

Coupling (non-symmetric two-term form; inflow uses W_ji, outflow W_ij):
    dX_ik/dt += Σ_j [ W_ji X_jk/N_j - W_ij X_ik/N_i ]   X ∈ {S,E,I,R}

Notes
-----
- Exposed individuals do NOT transmit (θ depends only on I).
- Deaths are NOT removed: total population is conserved (model R = recovered
  incl. the ~1 % fatality, which is ignored at the modelling level).
- The reporting fraction π (observed/true cases) is deliberately NOT in the
  model yet; growth-rate fits are slope-based, so they are π-independent.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from math import radians, sin, cos, sqrt, atan2

T_MO = 30.44  # days per month (monthly→daily conversion)

# ── 56 jurisdictions ──────────────────────────────────────────────────────────

JURISDICTIONS = sorted([
    'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN',
    'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
    'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
    'VT','VA','WA','WV','WI','WY','PR','GU','VI','MP','AS',
])

# 2020 Census populations (51 from metapopulation.STATE_POPULATIONS +
# PR/GU/VI/MP/AS verified 2026-08-05)
CENSUS_POPULATIONS = {
    'AL': 5024279, 'AK': 733391, 'AZ': 7151502, 'AR': 3011524, 'CA': 39538223,
    'CO': 5773714, 'CT': 3605944, 'DE': 989948, 'DC': 689545, 'FL': 21538187,
    'GA': 10711908, 'HI': 1455271, 'ID': 1839106, 'IL': 12812508, 'IN': 6785528,
    'IA': 3190369, 'KS': 2937880, 'KY': 4505836, 'LA': 4657757, 'ME': 1362359,
    'MD': 6177224, 'MA': 7029917, 'MI': 10077331, 'MN': 5706494, 'MS': 2961279,
    'MO': 6154913, 'MT': 1084225, 'NE': 1961504, 'NV': 3104614, 'NH': 1377529,
    'NJ': 9288994, 'NM': 2117522, 'NY': 20201249, 'NC': 10439388, 'ND': 779094,
    'OH': 11799448, 'OK': 3959353, 'OR': 4237256, 'PA': 13002700, 'RI': 1097379,
    'SC': 5118425, 'SD': 886667, 'TN': 6910840, 'TX': 29145505, 'UT': 3271616,
    'VT': 643077, 'VA': 8631393, 'WA': 7705281, 'WV': 1793716, 'WI': 5893718,
    'WY': 576851,
    'PR': 3285874, 'GU': 153836, 'VI': 87146, 'MP': 47329, 'AS': 49710,
}

# Geographic centroids (lat, lon): 51 from metapopulation.STATE_CENTROIDS +
# the five territories (approximate population centroids)
CENTROIDS = {
    'AL': (32.779, -86.829), 'AK': (64.069, -153.369), 'AZ': (34.274, -111.660),
    'AR': (34.894, -92.443), 'CA': (37.184, -119.470), 'CO': (38.997, -105.548),
    'CT': (41.622, -72.727), 'DE': (38.990, -75.505), 'DC': (38.905, -77.016),
    'FL': (28.631, -82.450), 'GA': (32.642, -83.443), 'HI': (20.293, -156.374),
    'ID': (44.351, -114.613), 'IL': (40.042, -89.197), 'IN': (39.894, -86.282),
    'IA': (42.075, -93.496), 'KS': (38.494, -98.380), 'KY': (37.535, -85.302),
    'LA': (31.069, -91.997), 'ME': (45.370, -69.243), 'MD': (39.055, -76.791),
    'MA': (42.260, -71.808), 'MI': (44.347, -85.410), 'MN': (46.281, -94.305),
    'MS': (32.736, -89.668), 'MO': (38.357, -92.458), 'MT': (47.053, -109.633),
    'NE': (41.538, -99.795), 'NV': (39.329, -116.631), 'NH': (43.681, -71.581),
    'NJ': (40.191, -74.673), 'NM': (34.407, -106.113), 'NY': (42.954, -75.527),
    'NC': (35.556, -79.388), 'ND': (47.450, -100.466), 'OH': (40.286, -82.794),
    'OK': (35.589, -97.494), 'OR': (43.934, -120.558), 'PA': (40.878, -77.800),
    'RI': (41.676, -71.556), 'SC': (33.917, -80.896), 'SD': (44.444, -100.226),
    'TN': (35.858, -86.351), 'TX': (31.476, -99.331), 'UT': (39.321, -111.094),
    'VT': (44.069, -72.666), 'VA': (37.522, -78.854), 'WA': (47.383, -120.447),
    'WV': (38.641, -80.623), 'WI': (44.624, -89.994), 'WY': (42.996, -107.551),
    'PR': (18.221, -66.590), 'GU': (13.444, 144.794), 'VI': (18.336, -64.896),
    'MP': (15.098, 145.674), 'AS': (-14.271, -170.132),
}

# NYT full name → code (50 states + DC from metapopulation + 5 territories)
NAME_TO_CODE = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',
    'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY',
    'Puerto Rico': 'PR', 'Guam': 'GU', 'Virgin Islands': 'VI',
    'Northern Mariana Islands': 'MP', 'American Samoa': 'AS',
}
CODE_TO_NAME = {v: k for k, v in NAME_TO_CODE.items()}

CONTIGUOUS = [s for s in JURISDICTIONS
              if s not in ('AK', 'HI', 'PR', 'GU', 'VI', 'MP', 'AS')]


# ── Distance & gravity ────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_distance_matrix(state_list=None):
    states = list(state_list or JURISDICTIONS)
    n = len(states)
    D = np.zeros((n, n))
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i != j:
                la1, lo1 = CENTROIDS[si]; la2, lo2 = CENTROIDS[sj]
                D[i, j] = haversine_km(la1, lo1, la2, lo2)
    return D


def build_gravity_kernel(state_list=None, power=2.0, min_distance_km=100.0,
                         b_N=None, b_d=None, origin_elasticity=0.5):
    """
    Row-normalised gravity kernel, Σ_j W = 1. Two forms:

      classic (b_N is None)      : sqrt(N_i N_j) / d_ij^power      (Viboud 2006)
      calibrated (b_N not None)  : N_i^{origin_el} N_j^{b_N} / d_ij^{b_d}

    The calibrated exponents come from fit_gravity_exponents() on the BTS
    T-100 panel (b_N ≈ 1.1, b_d ≈ 0.7); the origin factor cancels under
    row-normalisation and is kept only for kernel symmetry.
    Returns (n, n) ndarray.
    """
    states = list(state_list or JURISDICTIONS)
    n = len(states)
    N = np.array([CENSUS_POPULATIONS[s] for s in states], dtype=float)
    D = np.maximum(build_distance_matrix(states), min_distance_km)
    if b_N is None:
        # classic (Viboud 2006): sqrt(N_i N_j) / d^power
        raw = np.sqrt(np.outer(N, N)) / D ** power
    else:
        # data-calibrated: N_i^{origin_el} N_j^{b_N} / d^{b_d} (row-normalisation
        # makes the origin factor irrelevant to the shares, kept for symmetry)
        raw = N[:, None] ** origin_elasticity * N[None, :] ** b_N / D ** b_d
    np.fill_diagonal(raw, 0.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return raw / row_sums


def fit_gravity_exponents(F, D, N, method='ppml', min_distance_km=10.0,
                          max_iter=200, tol=1e-9):
    """
    Estimate the gravity exponents from the monthly flow panel F_ij(t)
    ((n, n, n_months) array of inter-state passengers/month):

        log F_ijt = α_i + γ_t + b_N·log N_j + b_d·log d_ij + ε     (OLS)
        F_ijt    ~ Poisson(exp(α_i + γ_t + b_N·log N_j + b_d·log d_ij))   (PPML)

    PPML (Poisson pseudo-maximum-likelihood, Santos Silva & Tenreyro 2006) is
    the standard gravity estimator — consistent under multiplicative
    heteroskedasticity and well-behaved with zero flows; OLS-on-logs is biased
    under the Jensen inequality and is reported for comparison.

    Returns dict(b_N, b_d, se_N, se_d, r2, method, converged).
    """
    n, _, T = F.shape
    rows = np.argwhere(F > 0)
    o = rows[:, 0]; d = rows[:, 1]; tt = rows[:, 2]
    y = F[o, d, tt].astype(float)
    ld = np.log(np.maximum(D[o, d], min_distance_km))
    ln = np.log(N[d])
    X = np.column_stack([ld, ln, np.eye(n)[o], np.eye(T)[tt]])

    if method == 'ols':
        coef, _, _, _ = np.linalg.lstsq(X, np.log(y), rcond=None)
        pred = X @ coef
        ss_res = float(((np.log(y) - pred) ** 2).sum())
        ss_tot = float(((np.log(y) - np.log(y).mean()) ** 2).sum())
        sigma2 = ss_res / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
        r2 = 1 - ss_res / ss_tot
        return dict(b_N=coef[1], b_d=coef[0], se_N=se[1], se_d=se[0],
                    r2=r2, method='ols', converged=True)

    # PPML via Poisson maximum likelihood (BFGS on the exact likelihood; the
    # IRWLS variant oscillates at this data's 6-decade count range)
    from scipy.optimize import minimize
    beta = np.linalg.lstsq(X, np.log(np.maximum(y, 1.0)), rcond=None)[0]

    def nll_grad(b):
        eta = np.clip(X @ b, -700, 700)
        mu = np.exp(eta)
        return (-(y @ eta) + mu.sum(),
                -(X.T @ (y - mu)))

    res = minimize(lambda b: nll_grad(b)[0], beta, jac=lambda b: nll_grad(b)[1],
                   method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12,
                                               'gtol': 1e-6})
    beta = res.x
    converged = res.success
    # deviance-based R² (1 − dev/dev_null)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        eta = np.clip(X @ beta, -700, 700)
        mu_f = np.exp(eta)
        dev = 2 * np.sum(y * np.log(np.maximum(y, 1e-12) / mu_f) - (y - mu_f))
        mu0 = np.full_like(mu_f, y.mean())
        dev0 = 2 * np.sum(y * np.log(np.maximum(y, 1e-12) / mu0) - (y - mu0))
    return dict(b_N=beta[1], b_d=beta[0], se_N=np.nan, se_d=np.nan,
                r2=1 - dev / dev0, method='ppml', converged=converged)


# ── Degree distribution ───────────────────────────────────────────────────────

def truncated_power_law(kmin=3, kmax=200, alpha=3.0, kc=1000.0):
    """P(k) = C k^{-α} e^{-k/kc}, k = kmin…kmax, ΣP = 1. Returns (k, P)."""
    k = np.arange(kmin, kmax + 1, dtype=float)
    w = k ** (-alpha) * np.exp(-k / kc)
    return k, w / w.sum()


def power_law_alpha_for_mean_k(mean_k_target=6.0, kmin=3, kmax=200, kc=1000.0):
    """Solve α so that ⟨k⟩ = mean_k_target for the truncated power law."""
    def f(alpha):
        k, P = truncated_power_law(kmin, kmax, alpha, kc)
        return (k @ P) - mean_k_target
    return brentq(f, 1.5, 4.5)


def density_scaled_mean_k(dens_per_km2, dens_ref, k0, xi):
    """⟨k⟩_i = ⟨k⟩_0 (dens_i / dens_ref)^ξ  (formulae_sheet §6 decision 1)."""
    dens = np.asarray(dens_per_km2, dtype=float)
    return k0 * np.power(dens / dens_ref, xi)


# ── Mobility calibration (BTS) ────────────────────────────────────────────────

def build_month_index():
    """Ordered (year, month) pairs from 2020-01 to 2023-03."""
    months = []
    for y in range(2020, 2024):
        for m in range(1, 13):
            months.append((y, m))
            if (y, m) == (2023, 3):
                return months
    return months


def calibrate_mobility(state_list=None, data_dir='../data', pair_level=False):
    """
    From the BTS state→state monthly flows compute, per state:
      μ_{0,i}  : Jan-2020 outbound / T_mo           [travellers/day]
      s_i(t)   : monthly outbound / Jan-2020 outbound  [dimensionless]
    Returns (mu0 array, s matrix (n × n_months), months, F_ij(t) (n×n×n_months)).
    Intra-state self-pairs (i == j) are excluded: they never leave the patch
    and W_ij couples distinct patches only.

    pair_level=True (the "saturated" data-replay limit, used for the
    per-pair mobility experiment) instead returns:
      mu0_mat[i,j]   = F_ij(Jan-2020) / T_mo       [travellers/day]
      s_tensor[i,j,t]= F_ij(t) / F_ij(Jan-2020)    [dimensionless]
    (zero wherever the Jan-2020 baseline is zero), so that
    W_ij(t) = mu0_mat[i,j]·s_tensor[i,j,t] reproduces the observed flow
    tensor exactly — one parameter per data point, no gravity involved.
    """
    import os
    states = list(state_list or JURISDICTIONS)
    n = len(states)
    pairs = pd.read_csv(os.path.join(data_dir, 'bts_state_pairs_monthly.csv'))
    months = build_month_index()
    month_idx = {(y, m): i for i, (y, m) in enumerate(months)}

    F = np.zeros((n, n, len(months)))
    for _, r in pairs.iterrows():
        i = states.index(r['origin_state']) if r['origin_state'] in states else None
        j = states.index(r['dest_state']) if r['dest_state'] in states else None
        if i is None or j is None or i == j:
            continue  # self-pairs (intra-state) never leave the patch
        mi = month_idx.get((int(r['YEAR']), int(r['MONTH'])))
        if mi is not None:
            F[i, j, mi] += r['PASSENGERS']

    if pair_level:
        base = F[:, :, 0]
        mu0_mat = base / T_MO
        s_tensor = np.divide(F, base[:, :, None], out=np.zeros_like(F, dtype=float),
                             where=base[:, :, None] > 0)
        return mu0_mat, s_tensor, months, F

    out = F.sum(axis=1)                      # (n, n_months) total outbound/month
    out0 = out[:, 0]                          # Jan-2020
    mu0 = out0 / T_MO                         # travellers/day
    s = np.divide(out, out0[:, None], out=np.ones_like(out),
                  where=out0[:, None] > 0)    # (n, n_months), 1.0 for μ0=0 states
    s[out0 == 0, :] = 0.0                     # no baseline → no outbound mobility
    return mu0, s, months, F


def s_interpolator(s_mat, months):
    """
    Return callable s(t) → (n,) giving each state's mobility scale at day t.
    Day 0 = 2020-01-01; monthly steps (constant within each month).
    """
    s_mat = np.asarray(s_mat, dtype=float)
    start = pd.Timestamp('2020-01-01')
    month_starts = [pd.Timestamp(f'{y:04d}-{m:02d}-01') for y, m in months]

    def s_func(t):
        ts = start + pd.Timedelta(days=int(np.floor(t)))
        idx = 0
        for i, ms in enumerate(month_starts):
            if ts >= ms:
                idx = i
        return s_mat[:, idx]
    return s_func


# ── Fused model ───────────────────────────────────────────────────────────────

class FusedMetapopulationSEIR:
    """
    Degree-based SEIR × gravity metapopulation (formulae_sheet §1–§7).

    State vector layout: for each state i, block [S_ik..., E_ik..., I_ik...,
    R_ik...] over degree classes k; total length 4·n·K.
    """

    def __init__(self, state_list, populations, W_grav, mu0, s_func,
                 k_vals, P_k, beta, sigma, gamma, W_tensor=None, months=None,
                 beta_scale=None, vac_flow=None, vac_efficacy=1.0,
                 V0_dict=None, waning=0.0):
        self.states = list(state_list)
        self.n = len(self.states)
        self.idx = {s: i for i, s in enumerate(self.states)}
        self.N = np.array([populations[s] for s in self.states], dtype=float)
        self.W_grav = np.asarray(W_grav, dtype=float)
        self.mu0 = np.asarray(mu0, dtype=float)
        self.s_func = s_func
        # saturated per-pair mode: W_ij(t) = W_tensor[i,j,month(t)] (daily),
        # reproduces the observed flows exactly; overrides W_grav·μ0·s.
        self.W_tensor = np.asarray(W_tensor, dtype=float) if W_tensor is not None else None
        self.month_starts = self._build_month_starts(months)
        # time-varying transmission: beta_scale (n, T) monthly multipliers,
        # beta_i(t) = beta0_i * beta_scale[i, month(t)]  (None -> constant β)
        if beta_scale is not None and months is None:
            raise ValueError('beta_scale requires months= (monthly grid)')
        self.beta_scale = (np.asarray(beta_scale, dtype=float)
                           if beta_scale is not None else None)
        # SEIRV: vaccination depletes the susceptible pool.
        #   dS/dt = -βkSθ - v_i(t)·(S/N) + w·V      (v = vac_flow·efficacy, people/day)
        #   dV/dt =  v_i(t)·(S/N) - w·V
        # vac_flow (n, T) monthly people/day; None -> 4-compartment SEIR (default).
        if vac_flow is not None and months is None:
            raise ValueError('vac_flow requires months= (monthly grid)')
        self.vac_flow = (np.asarray(vac_flow, dtype=float)
                         if vac_flow is not None else None)
        self.vac_efficacy = float(vac_efficacy)
        self.waning = float(waning)
        self.V0 = np.array([(V0_dict or {}).get(s, 0.0) for s in self.states],
                           dtype=float)
        self.n_comp = 5 if self.vac_flow is not None else 4
        self.k = np.asarray(k_vals, dtype=float)
        self.P = np.asarray(P_k, dtype=float)
        self.K = len(self.k)
        self.mean_k = float(self.k @ self.P)
        self.mean_k2 = float((self.k ** 2) @ self.P)

        beta_arr = np.atleast_1d(np.asarray(beta, dtype=float))
        if beta_arr.shape == (1,):
            beta_arr = np.full(self.n, beta_arr[0])
        elif beta_arr.shape != (self.n,):
            raise ValueError(f'beta must be scalar or length-{self.n}')
        self.beta = beta_arr
        self.sigma = float(sigma)
        self.gamma = float(gamma)

        self.t_sol = None
        self.y_sol = None  # (4·n·K, T)

    # ── state layout helpers ──────────────────────────────────────────────────

    def _block(self, y, comp):
        """(n, K) view of compartment comp (0=S,1=E,2=I,3=R,4=V)."""
        return y[comp * self.n * self.K:(comp + 1) * self.n * self.K].reshape(self.n, self.K)

    def _flatten(self, S, E, I, R, V=None):
        arrs = [a.ravel() for a in (S, E, I, R)]
        if V is not None:
            arrs.append(V.ravel())
        return np.concatenate(arrs)

    # ── initial conditions ────────────────────────────────────────────────────

    def make_initial_state(self, I0_dict, E0_dict=None):
        """
        I0_dict: {abbrev: infectious count} (distributed across classes ∝ P_k).
        E0_dict: optional {abbrev: exposed count}; default E0 = I0·(r+γ)/σ ≈ I0.
        SEIRV mode: V0_dict gives the vaccinated pool, S0 = N - I0 - E0 - V0.
        """
        S = np.zeros((self.n, self.K))
        E = np.zeros((self.n, self.K))
        I = np.zeros((self.n, self.K))
        R = np.zeros((self.n, self.K))
        V = np.zeros((self.n, self.K)) if self.n_comp == 5 else None
        for i, st in enumerate(self.states):
            i0 = I0_dict.get(st, 0.0)
            e0 = (E0_dict or {}).get(st, i0)
            pop = self.N[i]
            v0 = self.V0[i] if V is not None else 0.0
            s0 = max(pop - i0 - e0 - v0, 0.0)
            # counts per class: distribute proportional to P_k
            S[i] = s0 * self.P
            E[i] = e0 * self.P
            I[i] = i0 * self.P
            R[i] = 0.0
            if V is not None:
                V[i] = v0 * self.P
        return self._flatten(S, E, I, R, V)

    # ── RHS ───────────────────────────────────────────────────────────────────

    def _build_month_starts(self, months):
        """Day indices (since 2020-01-01) of each month's first day."""
        if not months:
            return np.array([0])
        starts = []
        for y, m in months:
            d = (pd.Timestamp(f'{y:04d}-{m:02d}-01') - pd.Timestamp('2020-01-01')).days
            starts.append(d)
        return np.array(starts)

    def _month_index(self, t):
        idx = 0
        for i, ms in enumerate(self.month_starts):
            if t >= ms:
                idx = i
        return idx

    def _W_at(self, t):
        if self.W_tensor is not None:
            return self.W_tensor[:, :, self._month_index(int(np.floor(t)))]
        s = self.s_func(t)
        return self.W_grav * (s * self.mu0)[:, None]  # (n,n), W_ij = s_i μ0_i W^grav

    def _rhs(self, t, y):
        S = self._block(y, 0)
        E = self._block(y, 1)
        I = self._block(y, 2)
        R = self._block(y, 3)
        V = self._block(y, 4) if self.n_comp == 5 else None
        N = self.N

        # theta_i = fraction of infectious stubs = Σ_k k I_ik / (N_i <k>)
        # (P(k) cancels: an infectious class-k node owns k stubs; total stubs = N_i<k>)
        theta = (I @ self.k) / (self.N * self.mean_k)  # (n,)

        # time-varying transmission: beta_i(t) = beta0_i · beta_scale[i, month(t)]
        beta_t = self.beta
        if self.beta_scale is not None:
            beta_t = beta_t * self.beta_scale[:, self._month_index(int(np.floor(t)))]

        # local dynamics (counts)
        inf = beta_t[:, None] * self.k[None, :] * S * theta[:, None]
        dS_l = -inf
        dE_l = inf - self.sigma * E
        dI_l = self.sigma * E - self.gamma * I
        dR_l = self.gamma * I

        # vaccination: v_i(t) people/day drawn from the susceptible share;
        # waning (optional) returns V to S.
        if V is not None:
            vt = self.vac_flow[:, self._month_index(int(np.floor(t)))] * self.vac_efficacy
            vac_out = vt[:, None] * (S / N[:, None])
            dS_l = dS_l - vac_out
            dV_l = vac_out - self.waning * V
            if self.waning > 0:
                dS_l = dS_l + self.waning * V
        else:
            dV_l = None

        # mobility: net_ik = Σ_j [W_ji f_jk − W_ij f_ik],  f = X/N
        W_t = self._W_at(t)
        rowsum = W_t.sum(axis=1)                      # = s_i μ_0,i
        comps = [(S, dS_l), (E, dE_l), (I, dI_l), (R, dR_l)]
        if V is not None:
            comps.append((V, dV_l))
        for X, dX in comps:
            frac = X / N[:, None]
            net = W_t.T @ frac - frac * rowsum[:, None]
            dX += net

        return self._flatten(dS_l, dE_l, dI_l, dR_l, dV_l)

    # ── simulate ──────────────────────────────────────────────────────────────

    def simulate(self, t_max, I0_dict, E0_dict=None, t0=0.0, method='RK45'):
        y0 = self.make_initial_state(I0_dict, E0_dict)
        t_eval = np.arange(t0, t_max + 1, dtype=float)
        sol = solve_ivp(self._rhs, [t0, t_max], y0, t_eval=t_eval,
                        method=method, rtol=1e-5, atol=1e-6)
        self.t_sol = sol.t
        self.y_sol = sol.y
        return self.t_sol, self.y_sol

    # ── outputs ───────────────────────────────────────────────────────────────

    def compartment(self, comp):
        """(n, K, T) or aggregated (n, T) arrays."""
        y = self.y_sol
        T = y.shape[1]
        blk = y[comp * self.n * self.K:(comp + 1) * self.n * self.K]
        return blk.reshape(self.n, self.K, T)

    def state_totals(self):
        S = self.compartment(0).sum(axis=1)   # (n, T)
        E = self.compartment(1).sum(axis=1)
        I = self.compartment(2).sum(axis=1)
        R = self.compartment(3).sum(axis=1)
        if self.n_comp == 5:
            V = self.compartment(4).sum(axis=1)
            return S, E, I, R, V
        return S, E, I, R

    def national_totals(self):
        tot = self.state_totals()
        return tuple(t.sum(0) for t in tot)

    def daily_new_infections(self):
        """True new infections per day per state: β_i Σ_k k S_ik θ_i (n, T)."""
        S = self.compartment(0)
        I = self.compartment(2)
        T = S.shape[2]
        theta_t = np.zeros((self.n, T))
        for ti in range(T):
            It = I[:, :, ti]
            theta_t[:, ti] = (It @ self.k) / (self.N * self.mean_k)
        kS = self.k[None, :, None] * S
        return (self.beta[:, None, None] * kS * theta_t[:, None, :]).sum(axis=1)


# ── Parameter estimation ───────────────────────────────────────────────────────

def fit_beta_from_growth(df_states, state_list, sigma, gamma,
                         start_date, window_days=30,
                         k_vals=None, P_k=None):
    """
    Per-state β_i from the early exponential growth rate r_i:
        β_i = (r_i + γ)(r_i + σ)/σ · ⟨k⟩/⟨k²⟩        (formulae_sheet §4)
    r_i = slope of log(7-day-smoothed new cases) over the window
    [start_date, start_date + window_days).  Slope fits are invariant to a
    constant reporting fraction, so no π is needed here.

    Returns (beta (n,), r (n,), ok_mask (n,)).
    """
    states = list(state_list)
    n = len(states)
    df = df_states.copy()
    df['date'] = pd.to_datetime(df['date'])
    win_end = pd.Timestamp(start_date) + pd.Timedelta(days=window_days)
    win = df[(df.date >= start_date) & (df.date < win_end)].copy()
    win['t'] = (win['date'] - pd.Timestamp(start_date)).dt.days

    k = np.asarray(k_vals, dtype=float)
    P = np.asarray(P_k, dtype=float)
    mom = (k ** 2 @ P) / (k @ P)

    beta = np.full(n, np.nan)
    r = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

    for i, st in enumerate(states):
        name = CODE_TO_NAME.get(st)
        if name is None:
            continue
        sub = win[win['state'] == name].sort_values('t')
        if len(sub) < 7:
            continue
        nc = sub['new_cases_smooth'].values if 'new_cases_smooth' in sub.columns \
             else sub['new_cases'].values
        nc = np.maximum(nc, 0)
        if nc.max() <= 0:
            continue
        tt = sub['t'].values
        keep = nc > 0
        if keep.sum() < 5:
            continue
        slope, _ = np.polyfit(tt[keep], np.log(nc[keep]), 1)
        if slope <= 0:
            continue
        r[i] = slope
        beta[i] = (slope + gamma) * (slope + sigma) / sigma * 1.0 / mom
        ok[i] = True

    # fallback for states without usable growth window: national median β
    med = np.nanmedian(beta)
    beta[~ok] = med
    return beta, r, ok


# ── time-varying β: mobility/NPI-driven beta_scale builders ────────────────

def cmr_beta_scale(cmrs, months, state_list=None, category='transit',
                   theta=1.0, min_scale=0.05):
    """
    Build the (n, T) monthly transmission multipliers from Google Community
    Mobility Reports (data/google_cmr_state_monthly.csv):

        g_i(t) = clip(1 + theta·pct_i(t)/100, min_scale, None)

    pct_i(t) = monthly mean % change vs the Jan-2020 baseline of category
    (default 'transit' — stations visits, the most direct contact proxy;
    alternatives: 'workplaces', 'retail', 'residential' (inverse)).

    beta_i(t) = beta0_i · g_i(t).  theta = 1 -> transmission scales exactly
    with mobility; theta < 1 -> partial contact reduction reaches β.

    Missing data (territories, months before 2020-02, after 2022-10):
    falls back to the national 'US' series, then to 1.0 (no reduction).
    """
    states = list(state_list or JURISDICTIONS)
    n = len(states)
    T = len(months)
    grid = [f'{y:04d}-{m:02d}' for y, m in months]
    pivot = cmrs.pivot_table(index='code', columns='month', values=category)
    scale = np.ones((n, T))
    nat = pivot.loc['US'] if 'US' in pivot.index else None
    for i, st in enumerate(states):
        if st in pivot.index:
            ser = pivot.loc[st].reindex(grid)
            base = ser.to_numpy(dtype=float)
        elif nat is not None:
            base = nat.reindex(grid).to_numpy(dtype=float)
        else:
            continue
        g = 1.0 + theta * base / 100.0
        g = np.where(np.isnan(g), 1.0, np.clip(g, min_scale, None))
        scale[i] = g
    return scale


def stringency_beta_scale(ox, months, state_list=None, lambda_=0.5,
                          min_scale=0.05):
    """
    Alternative transmission driver from the Oxford stringency index
    (data/oxcgrt_state_monthly.csv, 0-100):

        g_i(t) = clip(exp(-lambda_·str_i(t)/100), min_scale, None)

    beta_i(t) = beta0_i · g_i(t).  lambda_ = 0.5 -> 39% β reduction at
    stringency 100; lambda_ = 1.0 -> 63%.  Missing (territories/DC,
    months without data): fall back to national mean, then 1.0.
    """
    states = list(state_list or JURISDICTIONS)
    n = len(states)
    T = len(months)
    grid = [f'{y:04d}-{m:02d}' for y, m in months]
    pivot = ox.pivot_table(index='code', columns='month', values='stringency')
    scale = np.ones((n, T))
    nat_mean = float(pivot.to_numpy().mean())
    for i, st in enumerate(states):
        if st in pivot.index:
            ser = pivot.loc[st].reindex(grid)
            base = ser.to_numpy(dtype=float)
        else:
            base = np.full(T, nat_mean)
        base = np.where(np.isnan(base), nat_mean, base)
        scale[i] = np.clip(np.exp(-lambda_ * base / 100.0), min_scale, None)
    return scale
