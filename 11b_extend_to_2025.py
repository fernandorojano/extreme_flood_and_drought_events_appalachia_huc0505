"""
Step 11b – Extend streamflow and gridMET data from 2023-01-01 to 2025-12-31
Appends to existing files:
  data/streamflow_6gauges.csv   → 2000-01-01 to 2025-12-31
  data/gridmet_gauges.csv       → 2000-01-01 to 2025-12-31
"""
import os, json, numpy as np, pandas as pd
import urllib.request, netCDF4 as nc

os.chdir("/home/sandbox/huc0505")
CFS2CMS = 0.0283168

# ── 1. Load gauge registry ────────────────────────────────────────────────────
with open("data/gauge_registry.json") as f:
    reg = json.load(f)
gauges    = [{"site_no": k, **v} for k, v in reg.items()]
gauge_ids = [g["site_no"] for g in gauges]

NEW_START = "2023-01-01"
NEW_END   = "2025-12-31"
NEW_DATES = pd.date_range(NEW_START, NEW_END, freq="D")

# ── 2. Extend streamflow ──────────────────────────────────────────────────────
print("=== Extending streamflow ===")
url = ("https://waterservices.usgs.gov/nwis/dv/?format=json"
       f"&sites={','.join(gauge_ids)}"
       f"&parameterCd=00060&startDT={NEW_START}&endDT={NEW_END}&statCd=00003")
with urllib.request.urlopen(url, timeout=30) as r:
    data = json.loads(r.read())

new_q = pd.DataFrame(index=NEW_DATES)
new_q.index.name = "date"
for ts in data["value"]["timeSeries"]:
    sid  = ts["sourceInfo"]["siteCode"][0]["value"]
    vals = ts["values"][0]["value"]
    s    = pd.Series({pd.to_datetime(v["dateTime"][:10]): float(v["value"])
                      for v in vals if v["value"] not in ["-999999",""]})
    s    = s * CFS2CMS
    new_q[sid] = s.reindex(NEW_DATES)
    valid = new_q[sid].notna().sum()
    print(f"  {sid}: {valid}/{len(NEW_DATES)} valid days")

# Merge with existing
existing_q = pd.read_csv("data/streamflow_6gauges.csv", index_col=0, parse_dates=True)
combined_q = pd.concat([existing_q, new_q[~new_q.index.isin(existing_q.index)]])
combined_q = combined_q.sort_index()
combined_q.to_csv("data/streamflow_6gauges.csv")
print(f"streamflow_6gauges.csv updated: {existing_q.shape[0]} → {combined_q.shape[0]} rows")

# ── 3. Load gridMET cell assignments ─────────────────────────────────────────
print("\n=== Extending gridMET ===")
with open("data/gridmet_cells.json") as f:
    gc = json.load(f)

lat_s = slice(*gc["lat_slice"])
lon_s = slice(*gc["lon_slice"])
sub_lats = np.array(gc["sub_lats"])
sub_lons = np.array(gc["sub_lons"])

# Rebuild gauge_cells from json
gauge_cells = {}
for g in gauges:
    gid = g["site_no"]
    cells_raw = [(int(k.split("_")[0]), int(k.split("_")[1]))
                 for k, v in gc["cell_assignments"].items() if v]
    if gid == "03198000":
        gauge_cells[gid] = cells_raw
    else:
        target = g["huc8"]
        gauge_cells[gid] = [(int(k.split("_")[0]), int(k.split("_")[1]))
                             for k, v in gc["cell_assignments"].items()
                             if target in v]

THREDDS = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET"
VARS = {
    "pr":   ("precipitation_amount",        "pr_mm"),
    "pet":  ("potential_evapotranspiration", "pet_mm"),
    "vpd":  ("mean_vapor_pressure_deficit",  "vpd_kPa"),
    "tmmx": ("air_temperature",              "tmax_C"),
    "tmmn": ("air_temperature",              "tmin_C"),
}
CACHE_DIR = "data/gridmet"
NEW_YEARS = [2023, 2024, 2025]

# Accumulate records per variable per gauge
records = {l: {gid: [] for gid in gauge_ids}
           for _, (_, l) in VARS.items()}
date_list = []

for yr in NEW_YEARS:
    ndays      = 366 if (yr%4==0 and (yr%100!=0 or yr%400==0)) else 365
    cache_file = f"{CACHE_DIR}/yr_{yr}.csv"
    yr_dates   = pd.date_range(f"{yr}-01-01", periods=ndays, freq="D")

    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, index_col=0)
        print(f"── Year {yr}: loaded from cache ({len(cached)} rows)", flush=True)
        for col in cached.columns:
            gid, label = col.split("_", 1)
            records[label][gid].extend(cached[col].tolist())
        date_list.extend([d.strftime("%Y-%m-%d") for d in yr_dates])
        continue

    print(f"\n── Year {yr} ({ndays} days) ──", flush=True)
    yr_data = {}
    for vcode, (vname, label) in VARS.items():
        url = f"{THREDDS}/{vcode}/{vcode}_{yr}.nc"
        ds  = nc.Dataset(url)
        arr = ds.variables[vname][:, lat_s, lon_s]
        if vcode in ("tmmx","tmmn"):
            arr = arr - 273.15
        if hasattr(arr,"filled"):
            arr = arr.filled(np.nan)
        yr_data[label] = np.array(arr, dtype=np.float32)
        ds.close()
        print(f"  {label}: mean={float(np.nanmean(yr_data[label])):.3f}", flush=True)

    date_list.extend([d.strftime("%Y-%m-%d") for d in yr_dates])

    yr_rows = {}
    for label in [l for _,(_,l) in VARS.items()]:
        for gid, cells in gauge_cells.items():
            col = f"{gid}_{label}"
            if not cells:
                yr_rows[col] = [np.nan]*ndays
                records[label][gid].extend([np.nan]*ndays)
                continue
            idx_i = [c[0] for c in cells]
            idx_j = [c[1] for c in cells]
            cell_vals  = yr_data[label][:, idx_i, idx_j]
            daily_mean = np.nanmean(cell_vals, axis=1)
            yr_rows[col] = daily_mean.tolist()
            records[label][gid].extend(daily_mean.tolist())

    pd.DataFrame(yr_rows, index=yr_dates).to_csv(cache_file)
    print(f"  cached → {cache_file}", flush=True)

# Build new-years DataFrame
new_index = pd.to_datetime(date_list)
new_gmet  = pd.DataFrame(index=new_index)
new_gmet.index.name = "date"
for label in [l for _,(_,l) in VARS.items()]:
    for gid in gauge_ids:
        new_gmet[f"{gid}_{label}"] = records[label][gid]

# Merge with existing gridMET
existing_gm = pd.read_csv("data/gridmet_gauges.csv", index_col=0, parse_dates=True)
combined_gm = pd.concat([existing_gm, new_gmet[~new_gmet.index.isin(existing_gm.index)]])
combined_gm = combined_gm.sort_index()
combined_gm.index.name = "date"
combined_gm.to_csv("data/gridmet_gauges.csv")
print(f"\ngridmet_gauges.csv updated: {existing_gm.shape[0]} → {combined_gm.shape[0]} rows")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print(f"\n=== Final dataset extent ===")
print(f"Streamflow : {combined_q.index[0].date()} → {combined_q.index[-1].date()}  ({len(combined_q)} days)")
print(f"gridMET    : {combined_gm.index[0].date()} → {combined_gm.index[-1].date()}  ({len(combined_gm)} days)")
print(f"NaN (streamflow): {combined_q.isna().mean().mean():.4f}")
print(f"NaN (gridMET)   : {combined_gm.isna().mean().mean():.4f}")
print("\nStep 11b complete.")