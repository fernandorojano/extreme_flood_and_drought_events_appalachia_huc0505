"""
15d_fullperiod_preds_v4.py
Load per-gauge models; predict full 2000-2025 period.
"""
import os, pickle, numpy as np, pandas as pd
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")

DATA    = "/home/sandbox/huc0505/data/features_gridmet_v4.csv"
MDL_DIR = "/home/sandbox/huc0505/models"
OUT     = "/home/sandbox/huc0505/data/predictions_full_v4.csv"
TEST_START = "2023-01-01"

FEAT_COLS = [
    "pr_mm","pet_mm","vpd_kPa","tmax_C","tmin_C",
    "wb_mm","tmean_C","pr_7d","pr_30d","pr_90d",
    "pet_7d","pet_30d","wb_30d","wb_90d",
    "doy_sin","doy_cos","spei_30d","spei_90d",
    "flow_lag1","flow_lag7",
]
GAUGES = [3171000,3176500,3192000,3193000,3197000,3200500,3198000]

df = pd.read_csv(DATA, parse_dates=["date"])
rows = []

for gid in GAUGES:
    g = df[df["gauge_id"]==gid].copy()
    X = g[FEAT_COLS].values

    rf  = pickle.load(open(f"{MDL_DIR}/rf_reg_{gid}.pkl","rb"))
    uq  = pickle.load(open(f"{MDL_DIR}/rf_uq_{gid}.pkl","rb"))
    fl  = pickle.load(open(f"{MDL_DIR}/rf_flood_{gid}.pkl","rb"))
    dr  = pickle.load(open(f"{MDL_DIR}/rf_drought_{gid}.pkl","rb"))

    leaf_preds = np.array([e.predict(X) for e in uq.estimators_])
    q05 = np.percentile(leaf_preds,  5, axis=0)
    q95 = np.percentile(leaf_preds, 95, axis=0)

    tmp = g[["date","gauge_id","flow_m3s","flood_flag","drought_flag"]].copy()
    tmp["pred"]         = rf.predict(X)
    tmp["q05"]          = q05
    tmp["q95"]          = q95
    tmp["prob_flood"]   = fl.predict_proba(X)[:,1]
    tmp["prob_drought"] = dr.predict_proba(X)[:,1]
    tmp["split"]        = np.where(g["date"]>=TEST_START,"test","train")
    rows.append(tmp)
    print(f"  {gid}: {len(tmp)} rows  train={( tmp['split']=='train').sum()}  test={(tmp['split']=='test').sum()}")

out = pd.concat(rows, ignore_index=True)
out.to_csv(OUT, index=False)
print(f"\nSaved predictions_full_v4.csv  shape={out.shape}")