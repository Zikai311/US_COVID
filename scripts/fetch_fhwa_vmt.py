#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/fetch_fhwa_vmt.py — download & parse FHWA Traffic Volume Trends (TVT)
monthly files and build the national + state-level monthly VMT series that
complement the BTS aviation flows (candidate inputs for a future multimodal
mobility factor; the model is NOT changed here).

Sources:
  - monthly files: https://www.fhwa.dot.gov/policyinformation/travel_monitoring/
        {yymm}tvt/{yymm}tvt.{xls,xlsx}   (2020-01 .. 2024-12)
        * Page 6 "Changes on ALL Estimated Roads by Region and State":
          per-state VMT for the file's month (CMILES) vs prior-year same
          month (PMILES), % change (CCHANGE).
  - national series: the 2020 sheet of tvtarchive02-20.xlsx (2002..2020,
        "All Systems" column) + Table 1 (JAN..DEC by system) of each
        December file for 2021..2024.

Outputs (written to data/):
  - fhwa_vmt_state_monthly.csv : state, month(YYYY-MM), vmt_millions,
        vmt_prior_year_millions, pct_change
  - fhwa_vmt_national_monthly.csv : month(YYYY-MM), vmt_millions,
        vmt_prior_year_millions, pct_change  (All Systems)

Run: venv/bin/python scripts/fetch_fhwa_vmt.py
"""
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
TMP = os.path.join(ROOT, 'data', '.tvt_tmp')
BASE = 'https://www.fhwa.dot.gov/policyinformation/travel_monitoring/'
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0'}

START = (2020, 1)
END = (2024, 12)


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return 'cached'
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return f'http-{e.code}'
    except Exception as e:
        return f'error-{type(e).__name__}'
    with open(dest, 'wb') as f:
        f.write(data)
    return 'downloaded'


def month_files():
    out = []
    for y in range(START[0], END[0] + 1):
        for m in range(1, 13):
            if (y, m) < START or (y, m) > END:
                continue
            out.append((y, m))
    return out


def find_file(y, m):
    tag = f'{y:02d}{m:02d}'[2:]  # e.g. '20jan'
    name = f'{y:02d}{m:02d}'[2:]
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    tag = f'{y % 100:02d}{months[m - 1]}'
    for ext in ('xlsx', 'xls'):
        fn = os.path.join(TMP, f'{tag}tvt.{ext}')
        url = f'{BASE}{tag}tvt/{tag}tvt.{ext}'
        status = fetch(url, fn)
        if os.path.exists(fn) and os.path.getsize(fn) > 1000:
            return fn, status
    return None, None


def parse_state_table(path):
    """Page 6: 'Changes on ALL Estimated Roads by Region and State'.
    Returns DataFrame(state, cmiles, pmiles, cchange)."""
    xl = pd.ExcelFile(path)
    sheet = 'Page 6' if 'Page 6' in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    rows = []
    cur_state = None
    for _, r in df.iterrows():
        cells = [str(v) if pd.notna(v) else '' for v in r.tolist()]
        text = ' '.join(cells)
        if 'ALL* Estimated Roads by Region and State' in text or \
           'Region and State' in text:
            continue
        state = None
        for c in r.tolist()[:3]:
            if isinstance(c, str) and c.strip():
                state = c.strip()
                break
        if state is None or state.upper() in ('SUBTOTAL', 'TOTAL', 'NORTHEAST',
                                              'SOUTH ATLANTIC', 'NORTH CENTRAL',
                                              'SOUTH', 'WEST', 'UNITED STATES',
                                              'REGION AND STATE'):
            continue
        if state in ('Connecticut', 'Maine', 'Massachusetts', 'New Hampshire',
                     'New Jersey', 'New York', 'Pennsylvania', 'Rhode Island',
                     'Vermont', 'Delaware', 'Maryland', 'District of Columbia',
                     'Virginia', 'West Virginia', 'North Carolina', 'South Carolina',
                     'Georgia', 'Florida', 'Kentucky', 'Tennessee', 'Alabama',
                     'Mississippi', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas',
                     'Ohio', 'Indiana', 'Illinois', 'Michigan', 'Wisconsin',
                     'Minnesota', 'Iowa', 'Missouri', 'North Dakota',
                     'South Dakota', 'Nebraska', 'Kansas', 'Montana', 'Idaho',
                     'Wyoming', 'Colorado', 'New Mexico', 'Arizona', 'Utah',
                     'Nevada', 'Washington', 'Oregon', 'California', 'Alaska',
                     'Hawaii'):
            nums = []
            for c in r.tolist()[3:8]:
                if isinstance(c, (int, float)) and not isinstance(c, bool):
                    nums.append(float(c))
                elif isinstance(c, str) and c.strip():
                    try:
                        nums.append(float(c.replace(',', '')))
                    except ValueError:
                        pass
            # layout: col3=STATIONS, col4=CMILES, col5=PMILES, col6=CCHANGE
            if len(nums) >= 4:
                rows.append((state, nums[1], nums[2], nums[3]))
    st = pd.DataFrame(rows, columns=['state', 'cmiles', 'pmiles', 'cchange'])
    return st


def parse_national_table(path, target_year):
    """Table 1: 'Estimated Individual Monthly Motor Vehicle Travel in the
    United States' (Page 3 in xlsx, may be another page in xls).
    Returns list of (system, [12 monthly values in millions]).
    Layout: system name in col 0, month values in cols 3..14, values in
    billions — converted to millions (x1000). Blocks are labelled by year
    (e.g. '2021 Individual', '2020 (Preliminary)') in col 3; the block whose
    year matches target_year (and 'Revised' if both preliminary/revised
    blocks exist) is used.
    """
    xl = pd.ExcelFile(path)
    sheet = 'Page 3' if 'Page 3' in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    blocks = {}            # year -> list[(system, vals)]
    cur_year = None
    for _, r in df.iterrows():
        row = r.tolist()
        label = str(row[3]).strip() if len(row) > 3 and pd.notna(row[3]) else ''
        m = re.search(r'(20\d\d)', label)
        if m:
            cur_year = int(m.group(1))
            blocks.setdefault(cur_year, [])
            continue
        sysname = str(row[0]).strip() if pd.notna(row[0]) else ''
        if not sysname or cur_year is None:
            continue
        vals = []
        ok = 0
        for i in range(3, 15):
            v = row[i] if i < len(row) else None
            try:
                vals.append(float(v) * 1000.0)
                ok += 1
            except (ValueError, TypeError):
                vals.append(np.nan)
        if ok < 12:
            continue
        if any(s == sysname for s, _ in blocks[cur_year]):
            continue
        blocks[cur_year].append((sysname, vals))
    if target_year not in blocks:
        return []
    return blocks[target_year]


def main():
    os.makedirs(TMP, exist_ok=True)
    files = []
    for y, m in month_files():
        fn, status = find_file(y, m)
        if fn is None:
            print(f'{y:04d}-{m:02d}: MISSING')
            continue
        files.append(((y, m), fn))
        print(f'{y:04d}-{m:02d}: {status} ({os.path.basename(fn)})')

    # ── state-level monthly VMT ────────────────────────────────────────────
    st_rows = []
    for (y, m), fn in files:
        try:
            st = parse_state_table(fn)
        except Exception as e:
            print(f'  parse state {y}-{m:02d}: {e}')
            continue
        for _, r in st.iterrows():
            st_rows.append((r['state'], f'{y:04d}-{m:02d}',
                            r['cmiles'], r['pmiles'], r['cchange']))
    st_df = pd.DataFrame(st_rows, columns=['state', 'month', 'vmt_millions',
                                           'vmt_prior_year_millions',
                                           'pct_change'])
    st_df = st_df.sort_values(['state', 'month']).reset_index(drop=True)
    out1 = os.path.join(DATA, 'fhwa_vmt_state_monthly.csv')
    st_df.to_csv(out1, index=False)
    print(f'\nstate series: {len(st_df):,} rows | {st_df["state"].nunique()} '
          f'states | saved -> {os.path.relpath(out1, ROOT)}')
    print(st_df.groupby('state').size().describe().to_string())

    # ── national monthly VMT (All Systems, millions) ──────────────────────
    # 2019-2020 from the 2002-2020 archive (All Systems column, millions)
    arch = os.path.join(TMP, 'tvtarchive02-20.xlsx')
    fetch('https://www.fhwa.dot.gov/ohim/tvtw/tvtarchive02-20.xlsx', arch)
    nat_rows = []
    a = pd.ExcelFile(arch)
    months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for year in (2019, 2020):
        df = pd.read_excel(a, sheet_name=str(year), header=None)
        for _, r in df.iterrows():
            c0 = r.tolist()[0]
            if not (isinstance(c0, str) and c0.strip() in months_abbr):
                continue
            v = r.tolist()[13]          # All Systems value (col 13), millions
            try:
                v = float(v)
            except (ValueError, TypeError):
                continue
            nat_rows.append((f'{year:04d}-{months_abbr.index(c0.strip()) + 1:02d}', v))
    # 2021-2024 from the December files' Table 1 ('All Systems', billions -> x1000)
    for (y, m), fn in files:
        if m != 12:
            continue
        try:
            systems = parse_national_table(fn, y)
        except Exception as e:
            print(f'  parse national {y}-12: {e}')
            continue
        for sysname, vals in systems:
            if 'All Systems' not in sysname:
                continue
            for mi, v in enumerate(vals):
                if pd.notna(v):
                    nat_rows.append((f'{y:04d}-{mi + 1:02d}', v))
    nat_df = pd.DataFrame(nat_rows, columns=['month', 'vmt_millions'])
    nat_df = nat_df.sort_values('month').drop_duplicates('month').reset_index(drop=True)
    nat_df['vmt_prior_year_millions'] = nat_df['vmt_millions'].shift(12)
    nat_df['pct_change'] = (nat_df['vmt_millions']
                            / nat_df['vmt_prior_year_millions'] - 1) * 100
    out2 = os.path.join(DATA, 'fhwa_vmt_national_monthly.csv')
    nat_df.to_csv(out2, index=False)
    print(f'\nnational series: {len(nat_df)} months '
          f'({nat_df["month"].iloc[0]} .. {nat_df["month"].iloc[-1]}) '
          f'| saved -> {os.path.relpath(out2, ROOT)}')
    print(nat_df.to_string(index=False))


if __name__ == '__main__':
    main()
