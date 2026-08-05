"""
data_loader.py — Real-world COVID-19 data loading and preprocessing.

Supports NYT US national and state-level data.
Provides daily new cases/deaths, 7-day rolling averages, and
wave segmentation for model fitting.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── path resolution ──────────────────────────────────────────────────────────
_HERE = Path(__file__).parent          # src/
_ROOT = _HERE.parent                   # project root
DATA_DIR = _ROOT / "data"

# US population (2020 Census)
US_POPULATION = 331_449_281

# State populations (2020 Census estimates, top 10 + commonly studied)
STATE_POPULATIONS = {
    "California":   39_538_223,
    "Texas":        29_145_505,
    "Florida":      21_538_187,
    "New York":     20_201_249,
    "Pennsylvania": 13_002_700,
    "Illinois":     12_812_508,
    "Ohio":         11_799_448,
    "Georgia":      10_711_908,
    "North Carolina":10_439_388,
    "Michigan":      10_077_331,
    "Washington":     7_705_281,
    "Massachusetts":  7_029_917,
    "New Jersey":     9_288_994,
    "Colorado":       5_773_714,
    "Oregon":         4_237_256,
}


# ── loaders ──────────────────────────────────────────────────────────────────

def load_us_national(data_dir=None, smooth=True, window=7):
    """
    Load NYT US national cumulative data; derive daily new cases/deaths.

    Returns
    -------
    pd.DataFrame with columns:
        date, cum_cases, cum_deaths,
        new_cases, new_deaths,
        new_cases_smooth, new_deaths_smooth,
        active_cases (rough proxy: new_cases rolling 14-day sum)
    """
    path = Path(data_dir or DATA_DIR) / "nyt_us_national.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"cases": "cum_cases", "deaths": "cum_deaths"})

    # daily new cases/deaths (forward difference of cumulative)
    df["new_cases"]  = df["cum_cases"].diff().clip(lower=0).fillna(0)
    df["new_deaths"] = df["cum_deaths"].diff().clip(lower=0).fillna(0)

    if smooth:
        df["new_cases_smooth"]  = df["new_cases"].rolling(window, center=True,
                                                          min_periods=1).mean()
        df["new_deaths_smooth"] = df["new_deaths"].rolling(window, center=True,
                                                           min_periods=1).mean()

    # rough active-case proxy: sum of new cases over past 14 days
    df["active_cases"] = df["new_cases"].rolling(14, min_periods=1).sum()

    df["N"] = US_POPULATION
    return df


def load_us_states(state=None, data_dir=None, smooth=True, window=7):
    """
    Load NYT state-level cumulative data.

    Parameters
    ----------
    state : str or None
        State name (e.g. "New York"). None returns all states.

    Returns
    -------
    pd.DataFrame (same structure as load_us_national, plus 'state' column)
    """
    path = Path(data_dir or DATA_DIR) / "nyt_us_states.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)
    df = df.rename(columns={"cases": "cum_cases", "deaths": "cum_deaths"})

    if state is not None:
        df = df[df["state"] == state].reset_index(drop=True)

    # per-state daily new
    df["new_cases"]  = df.groupby("state")["cum_cases"].diff().clip(lower=0).fillna(0)
    df["new_deaths"] = df.groupby("state")["cum_deaths"].diff().clip(lower=0).fillna(0)

    if smooth:
        df["new_cases_smooth"] = (
            df.groupby("state")["new_cases"]
              .transform(lambda x: x.rolling(window, center=True, min_periods=1).mean())
        )
        df["new_deaths_smooth"] = (
            df.groupby("state")["new_deaths"]
              .transform(lambda x: x.rolling(window, center=True, min_periods=1).mean())
        )

    df["active_cases"] = (
        df.groupby("state")["new_cases"]
          .transform(lambda x: x.rolling(14, min_periods=1).sum())
    )

    # attach population
    df["N"] = df["state"].map(STATE_POPULATIONS).fillna(5_000_000).astype(int)
    return df


# ── wave segmentation ─────────────────────────────────────────────────────────

# Hard-coded US COVID wave boundaries (approximate peak dates)
US_WAVES = {
    "Wave 1 (Spring 2020)":  ("2020-02-01", "2020-06-30"),
    "Wave 2 (Summer 2020)":  ("2020-07-01", "2020-09-30"),
    "Wave 3 (Winter 2020)":  ("2020-10-01", "2021-03-31"),
    "Wave 4 (Delta 2021)":   ("2021-06-01", "2021-10-31"),
    "Wave 5 (Omicron 2022)": ("2022-01-01", "2022-04-30"),
}


def get_wave(df, wave_name):
    """Slice a DataFrame to a named wave period."""
    start, end = US_WAVES[wave_name]
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df[mask].reset_index(drop=True)


def get_wave_by_dates(df, start, end):
    """Slice a DataFrame to an arbitrary date range (strings 'YYYY-MM-DD')."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df[mask].reset_index(drop=True)


# ── SIR-compatible formatting ─────────────────────────────────────────────────

def to_sir_arrays(df, N=None):
    """
    Convert a wave DataFrame to numpy arrays suitable for SIR fitting.

    Returns
    -------
    t      : np.ndarray  — days from start (0, 1, 2, …)
    I_obs  : np.ndarray  — smoothed new cases (proxy for dI/dt * N)
    R_obs  : np.ndarray  — cumulative removed = cum_deaths + recovered proxy
    N      : int         — population size
    """
    if N is None:
        N = int(df["N"].iloc[0]) if "N" in df.columns else US_POPULATION

    t     = np.arange(len(df), dtype=float)
    I_obs = df["new_cases_smooth"].values if "new_cases_smooth" in df.columns \
            else df["new_cases"].values
    # cumulative confirmed cases ≈ cumulative removed + still infectious
    cum   = df["cum_cases"].values
    R_obs = np.maximum(cum - df["active_cases"].values, 0)

    return t, I_obs, R_obs, N


def compute_daily_growth_rate(df, col="new_cases_smooth"):
    """Compute day-over-day growth rate r(t) = (I(t) - I(t-1)) / I(t-1)."""
    vals = df[col].replace(0, np.nan)
    r = vals.pct_change()
    return r.values


def estimate_R_effective(df, gamma=1/10, col="new_cases_smooth"):
    """
    Simple Wallinga-Lipsitch approximation:
        R_eff(t) ≈ (I(t) / I(t-7)) ^ (gamma * 7)

    Parameters
    ----------
    gamma : float  — recovery rate (default: 10-day infectious period)
    """
    I = df[col].replace(0, np.nan).values
    gen_time = int(round(1 / gamma))    # generation time in days (≈ 1/γ)
    ratio = np.full_like(I, np.nan)
    for i in range(gen_time, len(I)):
        if I[i - gen_time] > 0:
            ratio[i] = (I[i] / I[i - gen_time]) ** 1.0
    # R_eff ≈ ratio^(gamma * gen_time)  but gen_time=1/gamma → exponent = 1
    return ratio


# ── BTS aviation data ─────────────────────────────────────────────────────────
# NOTE: the derived *national* aviation scale CSVs (bts_aviation_monthly.csv /
# bts_aviation_daily.csv) were removed from data/ — the national series is
# only used for the §7.5 analysis figures in notebook 07, where it is computed
# in-memory from the raw files.  The project models state-level mobility, so
# only the raw sources and the state-level tables below are persisted.

def load_tsa_daily(data_dir=None):
    """
    Load the raw TSA checkpoint travel numbers (daily screened travellers).

    Downloaded from <https://www.tsa.gov/travel/passenger-volumes> (pages
    2019-2023), covering 2019-01-01 .. 2023-12-31.  Note: TSA revised some
    early-2020 figures after they were first published.

    Returns
    -------
    pd.DataFrame with columns: date, travelers
    """
    path = Path(data_dir or DATA_DIR) / "tsa_checkpoint_daily.csv"
    return pd.read_csv(path, parse_dates=["date"])


def load_bts_state_pairs(data_dir=None):
    """
    Load the state-to-state monthly aviation flow matrix (the W_ij input).

    Directional monthly passengers for every (origin state, dest state) pair
    among the 56 NYT jurisdictions, 2020-01 .. 2023-03. Derived from BTS
    T-100 Market (US carriers, domestic markets) via
    ``data/t100_market_2020_2023.csv.gz``. Jurisdiction mapping notes:

    - DCA (Washington National) is physically in DC but T-100 codes it VA;
      it is re-attributed to DC.  IAD stays VA, BWI stays MD.
    - T-100 codes the Pacific territories under the single code 'TT'
      (Trust Territory); it is split by airport: GU = GUM+UAM,
      MP = SPN+TIQ+ROP, AS = PPG.  (MP = NMI, AS = American Samoa.)
    - A missing (origin, dest, month) row means zero passengers.
    - American Samoa (PPG) and NMI (SPN/TIQ/ROP) service is small but present.

    Returns
    -------
    pd.DataFrame with columns: origin_state, dest_state, YEAR, MONTH, PASSENGERS
    """
    path = Path(data_dir or DATA_DIR) / "bts_state_pairs_monthly.csv"
    return pd.read_csv(path)


def load_bts_state_monthly(data_dir=None):
    """
    Load the per-state monthly aviation table (the per-state traffic series).

    Each row is a state-month with a single time column 'month' (YYYY-MM) and
    outbound / inbound / net (= inbound - outbound) passengers.  Missing
    months = zero traffic.  Source: BTS T-100 Market, same jurisdiction
    mapping as ``load_bts_state_pairs``.  The per-state Jan-2020 baseline
    (outbound_jan2020) is stored separately in ``load_bts_state_jan2020``,
    and no mobility scale is stored here — the model computes its own
    per-state miu(t).

    Returns
    -------
    pd.DataFrame with columns: month, state, outbound, inbound, net
    """
    path = Path(data_dir or DATA_DIR) / "bts_state_monthly.csv"
    return pd.read_csv(path)


def load_bts_state_jan2020(data_dir=None):
    """
    Load the per-state Jan-2020 outbound baseline (state-constant).

    One row per state: that state's T-100 outbound passengers in 2020-01,
    used as the denominator for a per-state mobility factor miu(t).  States
    without any Jan-2020 outbound traffic (e.g. DE) have no row here.

    Returns
    -------
    pd.DataFrame with columns: state, outbound_jan2020
    """
    path = Path(data_dir or DATA_DIR) / "bts_state_jan2020.csv"
    return pd.read_csv(path)


def load_state_land_area_2020(data_dir=None):
    """
    Load per-jurisdiction land area + population density (2020 Census data,
    compiled 2026-08-05; land areas from the Wikipedia-derived census table,
    cross-checked against the 2020 Island Areas censuses).

    All 56 NYT/BTS jurisdictions (50 states + DC + PR/GU/VI/MP/AS) present.
    Used for the per-state degree-distribution scaling
    <k>_i = <k>_0 (dens_i / dens_0)^gamma (see formulae_sheet.md §1).

    Returns
    -------
    pd.DataFrame with columns: code, name, land_area_km2, land_area_mi2,
    pop_2020, density_per_km2
    """
    path = Path(data_dir or DATA_DIR) / "state_land_area_2020.csv"
    return pd.read_csv(path)


# ── summary statistics ─────────────────────────────────────────────────────────

def wave_summary(df):
    """Print key statistics for a wave slice."""
    total_cases  = df["cum_cases"].iloc[-1] - df["cum_cases"].iloc[0]
    total_deaths = df["cum_deaths"].iloc[-1] - df["cum_deaths"].iloc[0]
    peak_day     = df["new_cases_smooth"].idxmax() if "new_cases_smooth" in df.columns \
                   else df["new_cases"].idxmax()
    peak_date    = df.loc[peak_day, "date"].date()
    peak_val     = df.loc[peak_day, "new_cases_smooth" if "new_cases_smooth" in df.columns
                          else "new_cases"]
    cfr = total_deaths / max(total_cases, 1) * 100

    print(f"  Date range  : {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"  Total cases : {int(total_cases):,}")
    print(f"  Total deaths: {int(total_deaths):,}")
    print(f"  Peak date   : {peak_date}  ({int(peak_val):,} new cases/day)")
    print(f"  CFR (approx): {cfr:.2f}%")
    return dict(total_cases=total_cases, total_deaths=total_deaths,
                peak_date=peak_date, peak_cases_per_day=peak_val, cfr=cfr)
