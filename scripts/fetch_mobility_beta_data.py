#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/fetch_mobility_beta_data.py — download & parse the two datasets that
drive the time-varying beta_t(t) extension (model unchanged by this script):

  1. Google COVID-19 Community Mobility Reports (CMR), US state level:
     daily % change vs baseline (5-week period 2020-01-03..2020-02-06) for
     retail & recreation, grocery & pharmacy, parks, transit stations,
     workplaces, residential.  Zip: Region_Mobility_Report_CSVs.zip from
     gstatic.com (public).  Coverage 2020-02-15 .. 2022-10.
  2. Oxford COVID-19 Government Response Tracker (OxCGRT), US states:
     daily StringencyIndex (0-100) from OxCGRT_USA_latest.csv (GitHub).

Outputs (data/):
  google_cmr_state_daily.csv    code, date, retail, grocery, parks, transit,
                                workplaces, residential  (% vs baseline; code
                                'US' = national fallback; territories absent)
  google_cmr_state_monthly.csv  same, averaged onto the model's monthly grid
                                (YYYY-MM); months before 2020-02 have no data
                                -> NaN (use 0 when constructing beta(t))
  oxcgrt_state_daily.csv        code, date, stringency (0-100)
  oxcgrt_state_monthly.csv      monthly mean stringency per state

Run: venv/bin/python scripts/fetch_mobility_beta_data.py
"""
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
TMP = os.path.join(ROOT, 'data', '.cmr_tmp')
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0'}

CMR_ZIP_URL = 'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip'
OX_URL = ('https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/'
          'data/United%20States/OxCGRT_USA_latest.csv')

CATS = ['retail', 'grocery', 'parks', 'transit', 'workplaces', 'residential']
CMR_COLS = {
    'retail': 'retail_and_recreation_percent_change_from_baseline',
    'grocery': 'grocery_and_pharmacy_percent_change_from_baseline',
    'parks': 'parks_percent_change_from_baseline',
    'transit': 'transit_stations_percent_change_from_baseline',
    'workplaces': 'workplaces_percent_change_from_baseline',
    'residential': 'residential_percent_change_from_baseline',
}

sys.path.insert(0, os.path.join(ROOT, 'src'))
from fused_metapopulation import NAME_TO_CODE, build_month_index  # noqa: E402


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return 'cached'
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{url}: HTTP {e.code}')
    with open(dest, 'wb') as f:
        f.write(data)
    return 'downloaded'


def extract_us_cmr(zip_path):
    """Parse the US Region files from the CMR zip (they contain state rows
    in sub_region_1); return daily state-level df."""
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
        us = [n for n in names if '_US_Region_' in n]
        for n in sorted(us):
            df = pd.read_csv(z.open(n))
            st = df[df['sub_region_1'].notna() & df['sub_region_2'].isna()].copy()
            st['code'] = st['sub_region_1'].map(NAME_TO_CODE)
            st = st.dropna(subset=['code'])
            if st.empty:
                continue
            sub = st[['date'] + list(CMR_COLS.values()) + ['code']].copy()
            sub.columns = ['date'] + CATS + ['code']
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(TMP, exist_ok=True)
    zip_path = os.path.join(TMP, 'Region_Mobility_Report_CSVs.zip')
    status = fetch(CMR_ZIP_URL, zip_path)
    print(f'CMR zip: {status} ({os.path.getsize(zip_path)/1e6:.1f} MB)')

    cmr = extract_us_cmr(zip_path)
    cmr['date'] = pd.to_datetime(cmr['date'])
    cmr = cmr.sort_values(['code', 'date']).reset_index(drop=True)
    print(f'CMR daily: {len(cmr):,} rows | states: {cmr["code"].nunique()} '
          f'| {cmr.date.min().date()} .. {cmr.date.max().date()}')
    out_d = os.path.join(DATA, 'google_cmr_state_daily.csv')
    cmr.to_csv(out_d, index=False)
    print(f'saved -> {os.path.relpath(out_d, ROOT)}')

    # monthly aggregation on the model grid (2020-01 .. 2023-03)
    months = build_month_index()
    cmr['month'] = cmr['date'].dt.strftime('%Y-%m')
    grid = [f'{y:04d}-{m:02d}' for y, m in months]
    cmr = cmr[cmr['month'].isin(grid)]
    mm = cmr.groupby(['code', 'month'])[CATS].mean().reset_index()
    # national fallback series (from the 'US' region file when present)
    # -> not in extract (Region file has no state name); compute national
    #    separately by reading the Region files
    nat = _national_cmr(zip_path)
    nat['date'] = pd.to_datetime(nat['date'])
    nat['month'] = nat['date'].dt.strftime('%Y-%m')
    nat = nat[nat['month'].isin(grid)].groupby('month')[CATS].mean().reset_index()
    nat['code'] = 'US'
    mm = pd.concat([mm, nat], ignore_index=True)
    mm = mm.sort_values(['code', 'month']).reset_index(drop=True)
    out_m = os.path.join(DATA, 'google_cmr_state_monthly.csv')
    mm.to_csv(out_m, index=False)
    print(f'monthly: {len(mm):,} rows | saved -> {os.path.relpath(out_m, ROOT)}')
    print(mm[mm.code == 'NY'].head(3).to_string(index=False))

    # ── OxCGRT US states ────────────────────────────────────────────────────
    ox_path = os.path.join(TMP, 'OxCGRT_USA_latest.csv')
    status = fetch(OX_URL, ox_path)
    print(f'OxCGRT: {status}')
    ox = pd.read_csv(ox_path)
    ox = ox[ox['RegionName'].notna()].copy()
    ox['code'] = ox['RegionName'].map(NAME_TO_CODE)
    ox = ox.dropna(subset=['code'])
    ox['date'] = pd.to_datetime(ox['Date'].astype(str), format='%Y%m%d')
    ox = ox[['code', 'date', 'StringencyIndex_Average']].rename(
        columns={'StringencyIndex_Average': 'stringency'})
    ox = ox.sort_values(['code', 'date']).reset_index(drop=True)
    out_od = os.path.join(DATA, 'oxcgrt_state_daily.csv')
    ox.to_csv(out_od, index=False)
    print(f'OxCGRT daily: {len(ox):,} rows | states: {ox["code"].nunique()} '
          f'| saved -> {os.path.relpath(out_od, ROOT)}')
    ox['month'] = ox['date'].dt.strftime('%Y-%m')
    om = ox[ox['month'].isin(grid)].groupby(['code', 'month'])['stringency'].mean().reset_index()
    out_om = os.path.join(DATA, 'oxcgrt_state_monthly.csv')
    om.to_csv(out_om, index=False)
    print(f'OxCGRT monthly: {len(om):,} rows | saved -> {os.path.relpath(out_om, ROOT)}')
    print(om[om.code == 'NY'].head(3).to_string(index=False))


def _national_cmr(zip_path):
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.endswith('.csv') or '_US_Region_' not in n:
                continue
            df = pd.read_csv(z.open(n))
            nat = df[df['sub_region_1'].isna() & df['sub_region_2'].isna()].copy()
            sub = nat[['date'] + list(CMR_COLS.values())].copy()
            sub.columns = ['date'] + CATS
            sub['code'] = 'US'
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    main()
