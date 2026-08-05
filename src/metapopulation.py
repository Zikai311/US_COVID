"""
metapopulation.py — Metapopulation SIR model over a US state network.

Each of the 51 nodes (50 states + DC) is an independent SIR compartment.
Inter-state mobility is modelled via a gravity-model weighted network:

    W_ij  ∝  sqrt(N_i · N_j) / d_ij^power

A traveller from state k has probability I_k/N_k of being infectious
(proportional-mixing / random-flux assumption, as specified).

Migration ODEs (diffusion form, population-conserving):
    dS_k/dt = −β · S_k · I_k / N_k  +  μ · Σ_j W_kj (S_j/N_j − S_k/N_k)
    dI_k/dt =  β · S_k · I_k / N_k − γ · I_k  +  μ · Σ_j W_kj (I_j/N_j − I_k/N_k)
    dR_k/dt =  γ · I_k              +  μ · Σ_j W_kj (R_j/N_j − R_k/N_k)

where μ is a global mobility scaling factor (1.0 = baseline; 0.0 = full lockdown).

References
----------
Viboud et al. (2006)  Science  — gravity model for influenza spread in the US
Balcan et al. (2009)  PNAS     — metapopulation network epidemics
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.integrate import solve_ivp
from math import radians, sin, cos, sqrt, atan2

# ── State constants ───────────────────────────────────────────────────────────

# Geographic centroids (lat, lon) — contiguous US + AK + HI + DC
STATE_CENTROIDS = {
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
}

# 2020 Census populations
STATE_POPULATIONS = {
    'AL':  5024279, 'AK':   733391, 'AZ':  7151502, 'AR':  3011524,
    'CA': 39538223, 'CO':  5773714, 'CT':  3605944, 'DE':   989948,
    'DC':   689545, 'FL': 21538187, 'GA': 10711908, 'HI':  1455271,
    'ID':  1839106, 'IL': 12812508, 'IN':  6785528, 'IA':  3190369,
    'KS':  2937880, 'KY':  4505836, 'LA':  4657757, 'ME':  1362359,
    'MD':  6177224, 'MA':  7029917, 'MI': 10077331, 'MN':  5706494,
    'MS':  2961279, 'MO':  6154913, 'MT':  1084225, 'NE':  1961504,
    'NV':  3104614, 'NH':  1377529, 'NJ':  9288994, 'NM':  2117522,
    'NY': 20201249, 'NC': 10439388, 'ND':   779094, 'OH': 11799448,
    'OK':  3959353, 'OR':  4237256, 'PA': 13002700, 'RI':  1097379,
    'SC':  5118425, 'SD':   886667, 'TN':  6910840, 'TX': 29145505,
    'UT':  3271616, 'VT':   643077, 'VA':  8631393, 'WA':  7705281,
    'WV':  1793716, 'WI':  5893718, 'WY':   576851,
}

# Full name → abbreviation mapping (aligns with NYT data)
NAME_TO_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}
ABBREV_TO_NAME = {v: k for k, v in NAME_TO_ABBREV.items()}

# Contiguous US states (excludes AK, HI) — used for map plots
CONTIGUOUS_STATES = [s for s in STATE_CENTROIDS if s not in ('AK', 'HI')]


# ── Distance utilities ────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two (lat, lon) points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_distance_matrix(state_list):
    """Return (n×n) distance matrix in km for the given state abbreviations."""
    n = len(state_list)
    D = np.zeros((n, n))
    for i, si in enumerate(state_list):
        for j, sj in enumerate(state_list):
            if i != j:
                lat1, lon1 = STATE_CENTROIDS[si]
                lat2, lon2 = STATE_CENTROIDS[sj]
                D[i, j] = haversine_km(lat1, lon1, lat2, lon2)
    return D


# ── Network construction ──────────────────────────────────────────────────────

def build_gravity_network(state_list=None, power=2.0, min_distance_km=100.0,
                           top_k=None):
    """
    Build a weighted NetworkX graph using the gravity model.

        W_ij = sqrt(N_i * N_j) / max(d_ij, min_distance_km)^power

    Parameters
    ----------
    state_list   : list of state abbreviations; defaults to all 51 (50+DC)
    power        : distance decay exponent (default 2.0)
    min_distance_km : floor for distance (avoids division-by-zero for DC/MD/VA)
    top_k        : if set, keep only top-k strongest edges per node (sparsify)

    Returns
    -------
    G  : nx.Graph with node attributes 'pop' and 'centroid',
         edge attribute 'weight' (gravity score, un-normalised)
    W  : np.ndarray (n×n) weight matrix, rows normalised so Σ_j W_ij = 1
    """
    if state_list is None:
        state_list = sorted(STATE_POPULATIONS.keys())

    n = len(state_list)
    pops = np.array([STATE_POPULATIONS[s] for s in state_list], dtype=float)
    D = build_distance_matrix(state_list)

    # gravity weights
    W_raw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = max(D[i, j], min_distance_km)
                W_raw[i, j] = sqrt(pops[i] * pops[j]) / (d ** power)

    # optional: keep only top-k edges per node
    if top_k is not None:
        W_sparse = np.zeros_like(W_raw)
        for i in range(n):
            top_idx = np.argsort(W_raw[i])[-top_k:]
            W_sparse[i, top_idx] = W_raw[i, top_idx]
            W_sparse[top_idx, i] = W_raw[top_idx, i]   # symmetrise
        W_raw = W_sparse

    # row-normalise: W[i, j] = fraction of state-i population that travels to j per day
    # (will be multiplied by global mobility rate μ externally)
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W_raw / row_sums

    # build graph
    G = nx.Graph()
    for i, s in enumerate(state_list):
        G.add_node(s, pop=int(pops[i]),
                   centroid=STATE_CENTROIDS.get(s, (0, 0)))

    for i in range(n):
        for j in range(i + 1, n):
            w = 0.5 * (W[i, j] + W[j, i])   # symmetrise
            if w > 0:
                G.add_edge(state_list[i], state_list[j], weight=w,
                           distance_km=D[i, j])

    return G, W


# ── BTS aviation statistics (W matrix calibration) ───────────────────────────

# US domestic air passengers as fraction of Jan-2020 baseline.
# Source: BTS T-100 Domestic Segment + TSA checkpoint data.
# Apr-2020 nadir: ~87k passengers/day vs pre-COVID ~2.5M/day → only 4% of baseline.
BTS_MONTHLY_SCALE = {
    (2020,  1): 1.000,  # baseline
    (2020,  2): 0.970,  # early warnings, no impact yet
    (2020,  3): 0.550,  # collapse after 13-Mar national emergency
    (2020,  4): 0.040,  # nadir — 96% reduction
    (2020,  5): 0.075,
    (2020,  6): 0.135,
    (2020,  7): 0.220,  # summer partial recovery
    (2020,  8): 0.260,
    (2020,  9): 0.225,
    (2020, 10): 0.240,
    (2020, 11): 0.215,
    (2020, 12): 0.265,
    (2021,  1): 0.255,
    (2021,  2): 0.280,
    (2021,  3): 0.380,
    (2021,  4): 0.500,
    (2021,  5): 0.620,
    (2021,  6): 0.720,
    (2021,  7): 0.780,
    (2021,  8): 0.800,
    (2021,  9): 0.770,
    (2021, 10): 0.810,
    (2021, 11): 0.820,
    (2021, 12): 0.830,
    (2022,  1): 0.840,
    (2022,  2): 0.860,
    (2022,  3): 0.900,
    (2022,  4): 0.920,
}

import datetime as _dt

def aviation_mobility_scale(day_offset, start_year=2020, start_month=3):
    """
    Return the BTS-calibrated aviation mobility scale for a given day offset
    from the simulation start date.

    Parameters
    ----------
    day_offset    : int or float — days from simulation start
    start_year    : int — year of day 0 (default 2020)
    start_month   : int — month of day 0 (default 3 = March)

    Returns
    -------
    float in [0, 1] — fraction of Jan-2020 baseline mobility
    """
    start  = _dt.date(start_year, start_month, 1)
    current = start + _dt.timedelta(days=int(day_offset))
    key = (current.year, current.month)
    return BTS_MONTHLY_SCALE.get(key, 0.85)   # post-2022 plateau


def build_time_varying_W(W_base, mu_base, start_year=2020, start_month=3):
    """
    Return a callable ``W_func(t) → (n, n) ndarray`` that scales the static
    gravity matrix W_base by the BTS monthly aviation statistics.

    Drop-in replacement for the static ``mobility_rate × W_base`` used in
    MetapopulationSIR.simulate().

    Parameters
    ----------
    W_base       : (n, n) row-stochastic gravity matrix
    mu_base      : float — Jan-2020 baseline mobility rate (e.g. 5e-4)
    start_year   : int — year of simulation day 0
    start_month  : int — month of simulation day 0

    Returns
    -------
    callable(t) → (n, n) ndarray
    """
    W_base = np.asarray(W_base, dtype=float)
    def W_func(t):
        scale = aviation_mobility_scale(t, start_year, start_month)
        return W_base * (mu_base * scale)
    return W_func


# ── Metapopulation SIR ────────────────────────────────────────────────────────

class MetapopulationSIR:
    """
    Coupled SIR model over a US state network.

    Equations (diffusion / proportional-mixing form):
        dS_k/dt = −β · S_k · I_k / N_k  +  μ(t) · Σ_j W_kj (S_j/N_j − S_k/N_k)
        dI_k/dt =  β · S_k · I_k / N_k  −  γ · I_k
                +  μ(t) · Σ_j W_kj (I_j/N_j − I_k/N_k)
        dR_k/dt =  γ · I_k
                +  μ(t) · Σ_j W_kj (R_j/N_j − R_k/N_k)

    Parameters
    ----------
    state_list    : ordered list of state abbreviations (length n)
    populations   : dict {abbrev: N_k} or array of length n
    W             : (n×n) row-stochastic weight matrix from build_gravity_network
    beta          : float OR length-n array — transmission rate(s) per day.
                    Pass a scalar for a shared β, or an array for per-state β_k
                    (see fit_state_betas()).
    gamma         : float — recovery rate (per day)
    mobility_rate : float — baseline daily mobility (fraction of pop that travels)
                    e.g. 0.001 = 0.1 % per day ≈ ~330k travellers/day nationally
    """

    def __init__(self, state_list, populations, W, beta, gamma,
                 mobility_rate=5e-4, rho=1.0):
        self.states = list(state_list)
        self.n = len(self.states)
        self.idx = {s: i for i, s in enumerate(self.states)}

        if isinstance(populations, dict):
            self.N = np.array([populations[s] for s in self.states], dtype=float)
        else:
            self.N = np.asarray(populations, dtype=float)

        self.W  = np.asarray(W, dtype=float)      # (n, n)

        # beta: accept a scalar (shared) OR a length-n array (one β per state)
        beta_arr = np.atleast_1d(np.asarray(beta, dtype=float))
        if beta_arr.shape == (1,):
            beta_arr = np.full(self.n, beta_arr[0])
        elif beta_arr.shape != (self.n,):
            raise ValueError(
                f"beta must be scalar or length-{self.n} array, "
                f"got shape {beta_arr.shape}"
            )
        self.beta  = beta_arr                       # always (n,) — per-state β
        self.gamma = float(gamma)
        self.mu    = float(mobility_rate)          # overall scale
        # reporting fraction: observed = rho * true infections
        # rho << 1 in early COVID waves due to limited testing
        self.rho   = float(rho)

        # results storage
        self.t_sol = None
        self.S_sol = None   # (n, T)
        self.I_sol = None
        self.R_sol = None

    # ── initial conditions ────────────────────────────────────────────────────

    def make_initial_state(self, I0_dict):
        """
        Build initial state vector y0 of length 3n.

        Parameters
        ----------
        I0_dict : dict {state_abbrev: (I0, R0)}  — states with seed infections.
                  States not listed get I0=0, R0=0.
        """
        S = self.N.copy()
        I = np.zeros(self.n)
        R = np.zeros(self.n)

        for abbrev, (i0, r0) in I0_dict.items():
            if abbrev in self.idx:
                k = self.idx[abbrev]
                I[k] = max(float(i0), 0)
                R[k] = max(float(r0), 0)
                S[k] = max(self.N[k] - I[k] - R[k], 0)

        y0 = np.concatenate([S, I, R])
        return y0

    # ── ODE right-hand side ───────────────────────────────────────────────────

    def _rhs(self, t, y, W_t):
        """
        ODE right-hand side (3n equations).

        Parameters
        ----------
        t   : float — current time (days)
        y   : (3n,) state vector [S_0…S_{n-1}, I_0…, R_0…]
        W_t : (n, n) ndarray — effective mobility matrix at time t.
              Each entry W_t[k,j] = effective migration rate from j→k
              (already includes μ and any time-scaling).
              Callers build this via ``W_base * (mu * scale(t))``.
        """
        S = y[          :   self.n]
        I = y[  self.n  : 2*self.n]
        R = y[2*self.n  :          ]
        N = self.N

        # fractional prevalence per state
        fS = S / N
        fI = I / N
        fR = R / N

        # local transmission & recovery
        new_inf  = self.beta  * S * I / N
        recovery = self.gamma * I

        # diffusion: net inflow = Σ_j W_t[k,j]*(f_j - f_k)
        # W_t.sum(axis=1) = outflow rate from state k
        diffS = (W_t @ fS - fS * W_t.sum(axis=1)) * N
        diffI = (W_t @ fI - fI * W_t.sum(axis=1)) * N
        diffR = (W_t @ fR - fR * W_t.sum(axis=1)) * N

        dS = -new_inf  + diffS
        dI =  new_inf  - recovery + diffI
        dR =  recovery + diffR

        return np.concatenate([dS, dI, dR])

    # ── simulation ────────────────────────────────────────────────────────────

    def simulate(self, t_max, I0_dict,
                 mobility_schedule=None,
                 W_func=None,
                 method='RK45'):
        """
        Run the metapopulation SIR simulation.

        Parameters
        ----------
        t_max             : int — number of days to simulate
        I0_dict           : dict {abbrev: (I0, R0)} — initial infected per state
        mobility_schedule : callable(t) → float in [0,1], or None (constant=1).
                            Scalar multiplier on μ.  Ignored when W_func is given.
        W_func            : callable(t) → (n, n) ndarray, or None.
                            If provided, replaces mobility_schedule entirely.
                            Use ``build_time_varying_W(W_base, mu_base, ...)``
                            for BTS-calibrated time-varying mobility.
        method            : ODE solver method (default 'RK45')

        Returns
        -------
        t  : np.ndarray (T,)
        S  : np.ndarray (n, T)
        I  : np.ndarray (n, T)
        R  : np.ndarray (n, T)
        """
        # Build the W(t) callable that _rhs will use
        if W_func is not None:
            # time-varying matrix path (BTS or user-supplied)
            _W_at = W_func
        else:
            # legacy scalar-schedule path
            if mobility_schedule is None:
                mobility_schedule = lambda t: 1.0
            W_static = self.W * self.mu        # (n,n) baseline effective matrix
            def _W_at(t):
                return W_static * mobility_schedule(t)

        y0 = self.make_initial_state(I0_dict)
        t_eval = np.arange(0, t_max + 1, dtype=float)

        def rhs(t, y):
            return self._rhs(t, y, _W_at(t))

        sol = solve_ivp(rhs, [0, t_max], y0, t_eval=t_eval,
                        method=method, rtol=1e-5, atol=1e-6)

        self.t_sol = sol.t
        self.S_sol = sol.y[          :   self.n]
        self.I_sol = sol.y[  self.n  : 2*self.n]
        self.R_sol = sol.y[2*self.n  :          ]

        return self.t_sol, self.S_sol, self.I_sol, self.R_sol

    # ── analysis helpers ──────────────────────────────────────────────────────

    def get_national_trajectory(self):
        """Aggregate all states → national S, I, R."""
        return (self.S_sol.sum(axis=0),
                self.I_sol.sum(axis=0),
                self.R_sol.sum(axis=0))

    def get_peak_day(self, state_abbrev):
        """Return (peak_day, peak_I) for a specific state."""
        k = self.idx[state_abbrev]
        pk = int(np.argmax(self.I_sol[k]))
        return pk, self.I_sol[k, pk]

    def get_daily_new_cases(self):
        """
        Approximate daily new infections per state = β_k * S_k * I_k / N_k.
        Returns (n, T) array of **true infections** (not reported cases).
        For reported/observed cases, use get_reported_cases().
        """
        return self.beta[:, None] * self.S_sol * self.I_sol / self.N[:, None]

    def get_reported_cases(self):
        """
        Reported daily new cases = rho * (daily new infections).
        Returns (n, T) array. Use this to compare with observed data (NYT, JHU, etc.).
        """
        return self.rho * self.get_daily_new_cases()

    def fit_reporting_fraction(self, obs_new, per_state=False):
        """
        Fit reporting fraction ρ by minimizing least-squares error between
        model predictions and observed case counts.

        Analytical solution: ρ = sum(obs) / sum(model).

        Parameters
        ----------
        obs_new   : (n, T) ndarray — observed daily new cases per state
        per_state : bool — if True, fit separate ρ_k per state;
                    if False (default), fit a single global ρ.

        Returns
        -------
        rho : float or (n,) array — fitted reporting fraction(s)

        Notes
        -----
        After calling this, you can either:
        - Create a new model instance with the fitted rho:
          `model_new = MetapopulationSIR(..., rho=fitted_rho)`
        - Or manually set `self.rho = fitted_rho` on this instance.
        """
        model_new = self.get_daily_new_cases()
        T_cmp = min(obs_new.shape[1], model_new.shape[1])
        obs_trunc = obs_new[:, :T_cmp]
        mod_trunc = model_new[:, :T_cmp]

        if per_state:
            # per-state ρ_k = sum_t(obs[k,t]) / sum_t(model[k,t])
            rho_arr = np.zeros(self.n)
            for k in range(self.n):
                denom = mod_trunc[k].sum()
                if denom > 0:
                    rho_arr[k] = obs_trunc[k].sum() / denom
                else:
                    rho_arr[k] = 0.0
            return rho_arr
        else:
            # global ρ = sum_{k,t}(obs[k,t]) / sum_{k,t}(model[k,t])
            denom = mod_trunc.sum()
            if denom > 0:
                return obs_trunc.sum() / denom
            else:
                return 0.0

    def fit_state_betas(self, obs_new, rho_arr=None, t_max=None,
                        beta_bounds=(0.05, 1.50)):
        """
        Fit per-state transmission rates β_k by independently minimising
        SSE(obs_k, ρ_k · SIR_k(β_k)) for each state.

        A scalar ρ (§8.8) only rescales amplitude; it cannot fix the *shape*
        of the epidemic curve — its peak height-to-total ratio and peak timing.
        Those are governed by β. Here each state is fitted as an *isolated*
        SIR model (decoupled approximation, valid because μ ≪ β so inter-state
        diffusion contributes < 1 % of peak dynamics). Initial conditions are
        taken from day 0 of the already-run metapopulation simulation, so
        ``simulate()`` must be called first.

        Parameters
        ----------
        obs_new     : (n, T) ndarray — observed daily new cases per state
        rho_arr     : float or (n,) array — reporting fraction(s) applied to the
                      model before comparison. If None, uses self.rho.
        t_max       : int — simulation horizon in days. Defaults to obs length.
        beta_bounds : (lo, hi) — search range for β_k (per-day units)

        Returns
        -------
        beta_fitted : (n,) ndarray — fitted β_k for each state
        r2_fitted   : (n,) ndarray — coefficient of determination per state
        """
        from scipy.optimize import minimize_scalar

        if self.I_sol is None:
            raise RuntimeError("Run simulate() before fit_state_betas().")

        gamma  = self.gamma
        T_obs  = obs_new.shape[1]
        T_sim  = T_obs if t_max is None else max(int(t_max), T_obs)
        t_eval = np.arange(1, T_obs + 1, dtype=float)   # days 1…T_obs

        if rho_arr is None:
            rho_vec = np.full(self.n, self.rho, dtype=float)
        else:
            rho_vec = np.broadcast_to(np.asarray(rho_arr, dtype=float),
                                      (self.n,)).astype(float).copy()

        beta_fitted = np.empty(self.n)
        r2_fitted   = np.empty(self.n)
        lb, ub = np.log(beta_bounds[0]), np.log(beta_bounds[1])

        for k in range(self.n):
            Nk    = self.N[k]
            obs_k = obs_new[k, :T_obs]
            rho_k = float(rho_vec[k])

            # initial conditions from day 0 of the metapopulation run;
            # floor the seed at 1 infectious so zero-seed states can still grow
            I0 = max(float(self.I_sol[k, 0]), 1.0)
            R0 = max(float(self.R_sol[k, 0]), 0.0)
            S0 = max(Nk - I0 - R0, 0.0)
            y0 = [S0, I0, R0]

            def _sir_rhs(t, y, bk, _N=Nk):
                S, I, R = y
                inf = bk * S * I / _N
                rec = gamma * I
                return [-inf, inf - rec, rec]

            def _pred(bk):
                sol = solve_ivp(lambda t, y: _sir_rhs(t, y, bk),
                                [0, T_obs], y0, t_eval=t_eval,
                                method='RK45', rtol=1e-4, atol=1e-5)
                if not sol.success or sol.y.shape[1] < T_obs:
                    return None
                S_t, I_t = sol.y[0], sol.y[1]
                return rho_k * bk * S_t * I_t / Nk

            def sse(log_bk):
                pred = _pred(np.exp(log_bk))
                if pred is None:
                    return 1e18
                return float(np.sum((obs_k - pred) ** 2))

            res    = minimize_scalar(sse, bounds=(lb, ub), method='bounded')
            beta_k = float(np.exp(res.x))
            beta_fitted[k] = beta_k

            pred_opt = _pred(beta_k)
            if pred_opt is not None:
                ss_res = np.sum((obs_k - pred_opt) ** 2)
                ss_tot = np.sum((obs_k - obs_k.mean()) ** 2)
                r2_fitted[k] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                r2_fitted[k] = np.nan

        return beta_fitted, r2_fitted

    def get_all_peak_days(self):
        """Return dict {abbrev: peak_day} for all states."""
        return {s: int(np.argmax(self.I_sol[self.idx[s]]))
                for s in self.states}


# ── Data alignment utilities ──────────────────────────────────────────────────

def align_nyt_with_model(df_states, state_list, start_date, end_date):
    """
    Slice NYT state data to [start_date, end_date] and align with model state_list.

    Returns
    -------
    obs_new  : np.ndarray (n, T) — smoothed daily new cases per state
    obs_peak : dict {abbrev: peak_day_in_window}
    """
    n = len(state_list)

    mask_date = (df_states['date'] >= start_date) & (df_states['date'] <= end_date)
    df_w = df_states[mask_date].copy()

    dates = sorted(df_w['date'].unique())
    T = len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    col = 'new_cases_smooth' if 'new_cases_smooth' in df_w.columns else 'new_cases'
    obs_new = np.zeros((n, T))

    for i, abbrev in enumerate(state_list):
        name = ABBREV_TO_NAME.get(abbrev)
        if name is None:
            continue
        sub = df_w[df_w['state'] == name]
        for _, row in sub.iterrows():
            j = date_idx.get(row['date'])
            if j is not None:
                obs_new[i, j] = max(row[col], 0)

    obs_peak = {}
    for i, abbrev in enumerate(state_list):
        if obs_new[i].max() > 0:
            obs_peak[abbrev] = int(np.argmax(obs_new[i]))

    return obs_new, obs_peak


def stepwise_mobility(lockdown_start, lockdown_end, lockdown_factor=0.35,
                       recovery_factor=0.70):
    """
    Return a callable mobility_schedule(t) that models:
    - 1.0      before lockdown_start
    - lockdown_factor  during [lockdown_start, lockdown_end]
    - recovery_factor  after lockdown_end
    """
    def schedule(t):
        if t < lockdown_start:
            return 1.0
        elif t <= lockdown_end:
            return lockdown_factor
        else:
            return recovery_factor
    return schedule
