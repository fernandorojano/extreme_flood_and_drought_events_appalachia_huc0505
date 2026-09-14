"""
18_add_gauge_03193000.py  (v2 — correct NetCDF variable names)
Add gauge 03193000 gridMET data using the same HUC-8 cell-average methodology.
"""
import os, json, numpy as np, pandas as pd
import urllib.request, netCDF4 as nc
from shapely.geometry import Point, shape

os.chdir("/home/sandbox/huc0505")
G    = "03193000"
HUC8 = "05050006"
LAT, LON = 38.1382, -81.2143

# Actual NetCDF variable names inside the gridMET files
VAR_MAP = {
    "pr":   ("pr_mm",   "precipitation_amount"),
    "pet":  ("pet_mm",  "potential_evapotranspiration"),
    "vpd":  ("vpd_kPa", "mean_vapor_pressure_deficit"),
    "tmmx": ("tmax_C",  "air_temperature"),
    "tmmn": ("tmin_C",  "air_temperature"),
}
YEARS  = list(range(2000, 2026))
THREDDS = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET"

# ── 1. Load existing gridMET grid parameters ──────────────────────────────────
with open("data/gridmet_cells.json") as f:
    gc = json.load(f)
sub_lats   = np.array(gc["sub_lats"])
sub_lons   = np.array(gc["sub_lons"])
lat_start  = gc["lat_slice"][0]   # offset into full 585-row grid
lon_start  = gc["lon_slice"][0]   # offset into full 1386-col grid
NL, NC = len(sub_lats), len(sub_lons)
print(f"Existing subgrid: {NL} lat × {NC} lon  (offsets lat={lat_start}, lon={lon_start})")

# ── 2. Fetch HUC-8 05050006 polygon ──────────────────────────────────────────
WBD_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query"
    f"?where=HUC8='{HUC8}'&outFields=HUC8,NAME&f=geojson"
    "&geometryType=esriGeometryPolygon&returnGeometry=true&outSR=4326"
)
print(f"Fetching HUC-8 {HUC8} polygon …")
with urllib.request.urlopen(WBD_URL, timeout=60) as resp:
    wbd = json.loads(resp.read())

huc8_poly = shape(wbd["features"][0]["geometry"])
print(f"  Name: {wbd['features'][0]['properties'].get('name','?')}")
print(f"  Bounds: {[round(x,3) for x in huc8_poly.bounds]}")

# ── 3. Find sub-grid cells inside HUC-8 05050006 ─────────────────────────────
# sub-indices (i, j) and full-grid indices (i+lat_start, j+lon_start)
sub_indices  = []   # sub-grid (used for cell_assignments lookup)
full_indices = []   # full-grid (used for NetCDF extraction)
for i, lat in enumerate(sub_lats):
    for j, lon in enumerate(sub_lons):
        if huc8_poly.contains(Point(float(lon), float(lat))):
            sub_indices.append((i, j))
            full_indices.append((i + lat_start, j + lon_start))
print(f"Cells inside HUC-8 {HUC8}: {len(sub_indices)}")

if len(sub_indices) < 3:
    print("  Falling back to nearest cells to gauge coordinates")
    ilat = int(np.argmin(np.abs(sub_lats - LAT)))
    ilon = int(np.argmin(np.abs(sub_lons - LON)))
    sub_indices, full_indices = [], []
    for di, dj in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = ilat+di, ilon+dj
        if 0 <= ni < NL and 0 <= nj < NC:
            sub_indices.append((ni, nj))
            full_indices.append((ni + lat_start, nj + lon_start))
    print(f"  Using {len(full_indices)} nearest cells")

# ── 4. Download gridMET from THREDDS ─────────────────────────────────────────
rows = {}
for var_code, (col_name, nc_var) in VAR_MAP.items():
    print(f"  {var_code} → {col_name} ({nc_var}) …", flush=True)
    series = {}
    for yr in YEARS:
        url = f"{THREDDS}/{var_code}/{var_code}_{yr}.nc"
        try:
            ds = nc.Dataset(url)
            time_var  = ds.variables["day"]
            data_var  = ds.variables[nc_var]
            # Build date strings
            try:
                dates = nc.num2date(time_var[:], time_var.units,
                                    calendar=getattr(time_var, "calendar", "standard"))
                date_strs = [str(d)[:10] for d in dates]
            except Exception:
                base = pd.Timestamp(f"{yr}-01-01")
                date_strs = [(base + pd.Timedelta(days=int(k))).strftime("%Y-%m-%d")
                             for k in range(len(time_var))]
            # Extract and average across HUC-8 cells
            vals_all = []
            for (fi, fj) in full_indices:
                v = np.array(data_var[:, fi, fj], dtype=float)
                vals_all.append(v)
            cell_mean = np.nanmean(vals_all, axis=0)
            ds.close()
            for d, v in zip(date_strs, cell_mean):
                series[d] = v
        except Exception as e:
            print(f"    WARN {yr}: {e}")
    print(f"    → {len(series)} days")
    rows[col_name] = series

# ── 5. Build DataFrame ────────────────────────────────────────────────────────
all_dates = sorted(set().union(*[set(v.keys()) for v in rows.values()]))
gmet_new = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
gmet_new.index.name = "date"
for col_name, series in rows.items():
    gmet_new[f"{G}_{col_name}"] = pd.Series(
        {pd.Timestamp(d): v for d, v in series.items()}
    ).reindex(gmet_new.index)

print(f"\ngridMET for {G}: shape={gmet_new.shape}")
for c in gmet_new.columns:
    nn = gmet_new[c].notna().sum()
    print(f"  {c}: {nn} valid, mean={gmet_new[c].mean():.3f}")

# ── 6. Merge into gridmet_gauges.csv ─────────────────────────────────────────
gmet_existing = pd.read_csv("data/gridmet_gauges.csv", index_col="date", parse_dates=True)
print(f"\nExisting gridmet_gauges.csv: {gmet_existing.shape}")
for c in [c for c in gmet_existing.columns if G in c]:
    gmet_existing.drop(columns=[c], inplace=True)
gmet_combined = gmet_existing.join(gmet_new, how="left")
gmet_combined.to_csv("data/gridmet_gauges.csv")
print(f"Saved gridmet_gauges.csv — shape: {gmet_combined.shape}")
print("Done.")