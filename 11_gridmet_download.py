"""
Step 11 – gridMET 4 km Daily Data Download & Per-Gauge Assignment for HUC 0505
Variables: pr (mm/day), pet (mm/day), vpd (kPa), tmmx (°C), tmmn (°C)
Output: data/gridmet_gauges.csv  (8401 rows × gauge×variable columns)
        data/gridmet_cells.json  (gridMET cell assignments per gauge)
"""
import os, json, numpy as np, pandas as pd
import netCDF4 as nc
from shapely.geometry import Point, shape

os.chdir("/home/sandbox/huc0505")
os.makedirs("data/gridmet", exist_ok=True)

# ── 1. Load gauge registry ────────────────────────────────────────────────────
with open("data/gauge_registry.json") as f:
    reg = json.load(f)
gauges = [{"site_no": k, **v} for k, v in reg.items()]
print(f"Gauges loaded: {len(gauges)}")
for g in gauges:
    print(f"  {g['site_no']} | {g['name'][:40]:40s} | HUC-8: {g['huc8']}")

# ── 2. Re-fetch HUC-8 polygons (WBD REST API) ─────────────────────────────────
import urllib.request
HUC8_IDS = [g["huc8"] for g in gauges]
# Remove duplicates while preserving order
seen = set()
HUC8_UNIQUE = [h for h in HUC8_IDS if not (h in seen or seen.add(h))]

WBD_URL = ("https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query"
           "?where=HUC8+IN+({ids})&outFields=HUC8,NAME&f=geojson"
           "&geometryType=esriGeometryPolygon&returnGeometry=true&outSR=4326")

url = WBD_URL.format(ids=",".join(f"'{h}'" for h in HUC8_UNIQUE))
print(f"\nFetching {len(HUC8_UNIQUE)} HUC-8 polygons …")
with urllib.request.urlopen(url, timeout=60) as resp:
    wbd = json.loads(resp.read())

huc8_shapes = {}
for feat in wbd["features"]:
    hid  = feat["properties"]["huc8"]
    name = feat["properties"].get("name","")
    huc8_shapes[hid] = {"polygon": shape(feat["geometry"]), "name": name}
print(f"HUC-8 polygons fetched: {len(huc8_shapes)}")

# ── 3. Determine gridMET grid for HUC 0505 bounding box ───────────────────────
THREDDS = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET"
print("\nReading gridMET grid coordinates …")
ds_ref = nc.Dataset(f"{THREDDS}/pr/pr_2000.nc")
all_lats = ds_ref.variables["lat"][:]   # shape (585,), N→S
all_lons = ds_ref.variables["lon"][:]   # shape (1386,), W→E
ds_ref.close()

LAT_MIN, LAT_MAX = 36.9, 40.1
LON_MIN, LON_MAX = -83.6, -78.4
lat_idx = np.where((all_lats >= LAT_MIN) & (all_lats <= LAT_MAX))[0]
lon_idx = np.where((all_lons >= LON_MIN) & (all_lons <= LON_MAX))[0]
lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1])+1)
lon_slice  = slice(int(lon_idx[0]), int(lon_idx[-1])+1)
sub_lats = np.array(all_lats[lat_slice])
sub_lons = np.array(all_lons[lon_slice])
NL, NC = len(sub_lats), len(sub_lons)
print(f"gridMET subset: {NL} lat × {NC} lon = {NL*NC} cells "
      f"lat[{sub_lats[0]:.3f}–{sub_lats[-1]:.3f}] "
      f"lon[{sub_lons[0]:.3f}–{sub_lons[-1]:.3f}]")

# ── 4. Assign gridMET cells to HUC-8 watersheds ───────────────────────────────
print("\nAssigning gridMET cells to HUC-8 polygons …")
cell_huc8 = {}  # (i_lat, i_lon) → list of huc8 ids
for i, lat in enumerate(sub_lats):
    for j, lon in enumerate(sub_lons):
        pt = Point(float(lon), float(lat))
        for hid, hs in huc8_shapes.items():
            if hs["polygon"].contains(pt):
                cell_huc8.setdefault((i,j), []).append(hid)

n_assigned = sum(1 for v in cell_huc8.values() if v)
print(f"Cells assigned to ≥1 HUC-8: {n_assigned} / {NL*NC}")

# Build gauge → cell mask (nested: Kanawha gets all its contributing HUC-8s)
# Gauge watershed HUC-8 membership (from gauge_registry)
GAUGE_HUC8 = {g["site_no"]: g["huc8"] for g in gauges}
# Kanawha at Charleston (03198000) covers all 9 HUC-8s
KANAWHA_ID = "03198000"

gauge_cells = {}
for g in gauges:
    gid = g["site_no"]
    target_huc = g["huc8"]
    cells = []
    for (i,j), hlist in cell_huc8.items():
        if gid == KANAWHA_ID:
            if hlist:  # any HUC-8 in the study region
                cells.append((i,j))
        else:
            if target_huc in hlist:
                cells.append((i,j))
    gauge_cells[gid] = cells
    print(f"  {gid}: {len(cells)} cells ({g['name'][:35]})")

# Save assignment
cell_assign = {f"{i}_{j}": v for (i,j),v in cell_huc8.items()}
with open("data/gridmet_cells.json","w") as f:
    json.dump({"lat_slice": [int(lat_slice.start), int(lat_slice.stop)],
               "lon_slice": [int(lon_slice.start), int(lon_slice.stop)],
               "sub_lats":  sub_lats.tolist(),
               "sub_lons":  sub_lons.tolist(),
               "gauge_cell_counts": {k:len(v) for k,v in gauge_cells.items()},
               "cell_assignments": cell_assign}, f)
print("data/gridmet_cells.json saved")

# ── 5. Download gridMET variables year-by-year ────────────────────────────────
VARS = {
    "pr":   ("precipitation_amount",        "pr_mm"),
    "pet":  ("potential_evapotranspiration", "pet_mm"),
    "vpd":  ("mean_vapor_pressure_deficit",  "vpd_kPa"),
    "tmmx": ("air_temperature",              "tmax_C"),
    "tmmn": ("air_temperature",              "tmin_C"),
}
YEARS = list(range(2000, 2023))

# Accumulate per-gauge daily series
gauge_ids = [g["site_no"] for g in gauges]
records   = {}   # var_label → {gauge_id: [daily values]}

for vcode, (vname, label) in VARS.items():
    records[label] = {gid: [] for gid in gauge_ids}

date_list = []
CACHE_DIR = "data/gridmet"
for yr in YEARS:
    ndays = 366 if (yr%4==0 and (yr%100!=0 or yr%400==0)) else 365
    cache_file = f"{CACHE_DIR}/yr_{yr}.csv"

    # Skip if already cached
    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, index_col=0)
        print(f"── Year {yr}: loaded from cache ({len(cached)} rows)", flush=True)
        for col in cached.columns:
            parts = col.split("_", 1)
            gid, label = parts[0], parts[1]
            records[label][gid].extend(cached[col].tolist())
        yr_dates = pd.date_range(f"{yr}-01-01", periods=ndays, freq="D")
        date_list.extend([d.strftime("%Y-%m-%d") for d in yr_dates])
        continue

    print(f"\n── Year {yr} ({ndays} days) ──", flush=True)
    yr_data = {}

    for vcode, (vname, label) in VARS.items():
        url = f"{THREDDS}/{vcode}/{vcode}_{yr}.nc"
        ds  = nc.Dataset(url)
        arr = ds.variables[vname][:, lat_slice, lon_slice]
        if vcode in ("tmmx","tmmn"):
            arr = arr - 273.15
        if hasattr(arr,"filled"):
            arr = arr.filled(np.nan)
        yr_data[label] = np.array(arr, dtype=np.float32)
        ds.close()
        print(f"  {label}: mean={float(np.nanmean(yr_data[label])):.3f}", flush=True)

    yr_dates = pd.date_range(f"{yr}-01-01", periods=ndays, freq="D")
    date_list.extend([d.strftime("%Y-%m-%d") for d in yr_dates])

    # Compute per-gauge means and cache
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

print(f"\nTotal dates accumulated: {len(date_list)}")

# ── 6. Build output DataFrame ─────────────────────────────────────────────────
dates_index = pd.to_datetime(date_list)
# Filter to 2000-01-01 – 2022-12-31 exactly
target_dates = pd.date_range("2000-01-01","2022-12-31", freq="D")
df = pd.DataFrame(index=dates_index)
for label in [l for _,(_,l) in VARS.items()]:
    for gid in gauge_ids:
        col = f"{gid}_{label}"
        df[col] = records[label][gid]

df = df.reindex(target_dates)
df.index.name = "date"

# Basic check
print(f"\nOutput shape: {df.shape}")
print(f"NaN fraction: {df.isna().mean().mean():.4f}")
print(df.describe().round(3))

df.to_csv("data/gridmet_gauges.csv")
print("\nSaved: data/gridmet_gauges.csv")
print("Step 11 complete.")