#!/usr/bin/env python3
"""
13c_build_features_v4.py
Feature matrix v4:
  + flow_lag1, flow_lag7   — autoregressive features (biggest NSE gain)
  + drought_flag = spei_90d < -1.0  — fixes wb_30d feature/target leakage
"""
import numpy as np
import pandas as pd

DATA   = '/home/sandbox/huc0505/data'
GAUGES = ['03171000','03176500','03192000','03193000','03197000','03200500','03198000']

def rolling_spei(series, window):
    r  = series.rolling(window, min_periods=window // 2)
    sd = r.std().replace(0, np.nan)
    return (series - r.mean()) / sd

def build_gauge(gid, flow_df, gmet_df):
    pfx  = gid + '_'
    cols = [c for c in gmet_df.columns if c.startswith(pfx)]
    g    = gmet_df[cols].rename(columns=lambda c: c.replace(pfx, ''))
    df   = g.join(flow_df[[gid]].rename(columns={gid: 'flow_m3s'}), how='inner').copy()

    df['wb_mm']   = df['pr_mm'] - df['pet_mm']
    df['tmean_C'] = (df['tmax_C'] + df['tmin_C']) / 2.0

    df['pr_7d']   = df['pr_mm'].rolling(7,  min_periods=4 ).sum()
    df['pr_30d']  = df['pr_mm'].rolling(30, min_periods=15).sum()
    df['pr_90d']  = df['pr_mm'].rolling(90, min_periods=45).sum()
    df['pet_7d']  = df['pet_mm'].rolling(7,  min_periods=4 ).sum()
    df['pet_30d'] = df['pet_mm'].rolling(30, min_periods=15).sum()
    df['wb_30d']  = df['wb_mm'].rolling(30, min_periods=15).sum()
    df['wb_90d']  = df['wb_mm'].rolling(90, min_periods=45).sum()

    doy = df.index.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    df['spei_30d'] = rolling_spei(df['wb_mm'], 30)
    df['spei_90d'] = rolling_spei(df['wb_mm'], 90)

    # ── NEW autoregressive lag features ──────────────────
    df['flow_lag1'] = df['flow_m3s'].shift(1)
    df['flow_lag7'] = df['flow_m3s'].shift(7)

    # Event labels
    q90 = df['flow_m3s'].quantile(0.90)
    q75 = df['pr_7d'].quantile(0.75)
    df['flood_flag']   = ((df['flow_m3s'] > q90) & (df['pr_7d'] > q75)).astype(int)
    # FIXED: drought_flag no longer uses wb_30d (eliminates feature/target leakage)
    df['drought_flag'] = (df['spei_90d'] < -1.0).astype(int)

    df['gauge_id'] = gid
    return df.dropna(subset=['pr_90d', 'spei_90d', 'flow_lag7', 'flow_m3s'])

# ── Load ──────────────────────────────────────────────────────────
flow = pd.read_csv(f'{DATA}/streamflow_6gauges.csv', index_col='date', parse_dates=True)
gmet = pd.read_csv(f'{DATA}/gridmet_gauges.csv',     index_col='date', parse_dates=True)
for gid in GAUGES:
    if gid in flow.columns:
        flow[gid] = flow[gid].ffill(limit=5)

parts = []
for gid in GAUGES:
    df = build_gauge(gid, flow, gmet)
    print(f"  {gid}: {len(df):5d} rows | flood={df['flood_flag'].mean():.3f} | drought={df['drought_flag'].mean():.3f}")
    parts.append(df)

out = pd.concat(parts)
out.index.name = 'date'
out.to_csv(f'{DATA}/features_gridmet_v4.csv')
print(f"\nSaved features_gridmet_v4.csv")
print(f"  Shape      : {out.shape[0]} rows x {out.shape[1]} cols")
print(f"  Date range : {out.index.min().date()} to {out.index.max().date()}")
print(f"  Flood rate : {out['flood_flag'].mean():.3f}")
print(f"  Drought    : {out['drought_flag'].mean():.3f}")
print(f"  NaN        : {out.isna().sum().sum()}")