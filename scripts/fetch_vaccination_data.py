#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/fetch_vaccination_data.py — download & parse US COVID-19 vaccination
coverage (state level, 2020-12 .. 2023-05) from CDC:

  data.cdc.gov dataset "COVID-19 Vaccinations in the United States, County"
  (id 8xkx-amqh) — county-level, aggregated server-side to state x date via
  SoQL ($group).  Columns used:
    series_complete_yes  : people fully vaccinated (cumulative)
    administered_dose1   : people with >=1 dose (cumulative)
    booster_doses        : people with a booster (cumulative)

Outputs (data/):
  cdc_vax_state_daily.csv   : state, date, series_complete, dose1, boosters
  cdc_vax_state_monthly.csv : state, month(YYYY-MM),
                              series_complete (cumulative at month end),
                              dose1, boosters,
                              series_complete_flow (monthly increase in
                              fully-vaccinated people — the v_i(t) input for
                              the SEIRV susceptible-pool term)

Coverage: all 56 NYT/BTS jurisdictions (50 states + DC + PR/GU/VI/MP/AS).
Territories are patchy in the source (AS/MP/GU/VI have few rows) — the model
will use them as-is or fall back to the national series.

Run: venv/bin/python scripts/fetch_vaccination_data.py
"""
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
URL = ('https://data.cdc.gov/resource/8xkx-amqh.csv?'
       '$select=recip_state,date,sum(series_complete_yes)%20as%20series_complete,'
       'sum(administered_dose1_recip)%20as%20dose1,sum(booster_doses)%20as%20boosters'
       '&$group=recip_state,date&$limit=60000')
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0'}

DROP = {'FM', 'MH', 'PW', 'UNK'}  # non-project jurisdictions


def main():
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        df = pd.read_csv(r)
    df = df[~df['recip_state'].isin(DROP)].copy()
    df = df.rename(columns={'recip_state': 'state'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['state', 'date']).reset_index(drop=True)
    for c in ('series_complete', 'dose1', 'boosters'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    print(f'state x date rows: {len(df):,} | states: {df["state"].nunique()} '
          f'| {df.date.min().date()} .. {df.date.max().date()}')

    out_d = os.path.join(DATA, 'cdc_vax_state_daily.csv')
    df.to_csv(out_d, index=False)
    print(f'saved -> {os.path.relpath(out_d, ROOT)}')

    # monthly aggregation on the model grid (2020-01 .. 2023-03)
    months = []
    for y in range(2020, 2024):
        for m in range(1, 13):
            months.append(f'{y:04d}-{m:02d}')
            if (y, m) == (2023, 3):
                break
    grid = [pd.Period(m, 'M') for m in months]
    df['month'] = df['date'].dt.to_period('M')
    rows = []
    for st, g in df.groupby('state'):
        g = g.set_index('month')
        for per in grid:
            if per in g.index:
                last = g.loc[per].iloc[-1]
                first = g.loc[per].iloc[0]
            else:
                continue
            # month-end cumulative; flow = month-end - previous month-end
            rows.append({'state': st, 'month': str(per),
                         'series_complete': float(last['series_complete']),
                         'dose1': float(last['dose1']),
                         'boosters': float(last['boosters'])})
    mm = pd.DataFrame(rows).sort_values(['state', 'month']).reset_index(drop=True)
    mm['series_complete_prev'] = mm.groupby('state')['series_complete'].shift(1)
    mm['series_complete_flow'] = (
        mm['series_complete'] - mm['series_complete_prev'].fillna(0)).clip(lower=0)
    mm = mm.drop(columns='series_complete_prev')
    out_m = os.path.join(DATA, 'cdc_vax_state_monthly.csv')
    mm.to_csv(out_m, index=False)
    print(f'monthly: {len(mm):,} rows | saved -> {os.path.relpath(out_m, ROOT)}')
    print('sample (NY):')
    print(mm[mm.state == 'NY'].tail(4).to_string(index=False))
    us = mm[mm.state == 'US']
    print('\nnational fully-vaccinated coverage milestones:')
    for m, v in us.set_index('month')['series_complete'].items():
        if m in ('2021-01', '2021-06', '2021-09', '2021-12', '2022-06', '2023-03'):
            print(f'  {m}: {v/1e6:.1f}M')


if __name__ == '__main__':
    main()
