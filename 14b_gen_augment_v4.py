"""
14b_gen_augment_v4.py  (conditional Gaussian augmentation)
For each gauge × condition stratum (normal / flood / drought),
fit mean+cov on real data, sample synthetic rows via Cholesky,
apply physical constraints.  Fast (<2 s), statistically sound.
"""
import numpy as np, pandas as pd
np.random.seed(42)

DATA  = "/home/sandbox/huc0505/data/features_gridmet_v4.csv"
OUT   = "/home/sandbox/huc0505/data/features_augmented_v4.csv"
N_SYN = 20000

CONT_COLS = ["pr_mm","pet_mm","vpd_kPa","tmax_C","tmin_C","flow_m3s",
             "wb_mm","tmean_C","pr_7d","pr_30d","pr_90d",
             "pet_7d","pet_30d","wb_30d","wb_90d",
             "doy_sin","doy_cos","spei_30d","spei_90d","flow_lag1","flow_lag7"]
CLIP_NN = {"pr_mm","pet_mm","pr_7d","pr_30d","pr_90d",
           "pet_7d","pet_30d","flow_m3s","flow_lag1","flow_lag7"}
GAUGES  = [3171000,3176500,3192000,3193000,3197000,3200500,3198000]

df = pd.read_csv(DATA)
per_g = N_SYN // len(GAUGES)
rows = []

for gi, gid in enumerate(GAUGES):
    g = df[df["gauge_id"]==gid]
    n_this = per_g + (N_SYN - per_g*len(GAUGES) if gid==GAUGES[-1] else 0)
    # --- allocate per stratum proportionally ---
    fl = g["flood_flag"].values; dr = g["drought_flag"].values
    n_fl  = max(2, int(n_this * fl.mean() * 1.5))   # oversample minority
    n_dr  = max(2, int(n_this * dr.mean() * 1.5))
    n_nor = n_this - n_fl - n_dr
    syn_parts = []
    for mask, n_s, flag_ff, flag_dr in [
            ((fl==0)&(dr==0), n_nor, 0, 0),
            ((fl==1),         n_fl,  1, 0),
            ((dr==1),         n_dr,  0, 1)]:
        sub = g[mask][CONT_COLS].values
        if len(sub) < 3:
            sub = g[CONT_COLS].values    # fallback to all rows
        mu_s = sub.mean(0)
        # regularised covariance
        cov  = np.cov(sub.T) + np.eye(len(CONT_COLS))*1e-6
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov += np.eye(len(CONT_COLS))*0.1
            L = np.linalg.cholesky(cov)
        z = np.random.randn(n_s, len(CONT_COLS))
        xs = mu_s + z @ L.T
        # physical constraints
        for ci, col in enumerate(CONT_COLS):
            if col in CLIP_NN: xs[:,ci] = np.maximum(0, xs[:,ci])
        r = pd.DataFrame(xs, columns=CONT_COLS)
        r["flood_flag"]   = flag_ff
        r["drought_flag"] = flag_dr
        r["gauge_id"]  = gid
        r["synthetic"] = 1
        r["date"]      = pd.NaT
        syn_parts.append(r)
    rows.extend(syn_parts)
    print(f"  {gid}: {n_this} synthetic  flood={n_fl}  drought={n_dr}  normal={n_nor}")

syn  = pd.concat(rows, ignore_index=True)
real = pd.read_csv(DATA); real["synthetic"] = 0
out  = pd.concat([real, syn], ignore_index=True)
out.to_csv(OUT, index=False)
print(f"\nSaved features_augmented_v4.csv  shape={out.shape}")
print(f"  Real={len(real)}  Synthetic={len(syn)}")
print(f"  Flood rate (syn) : {syn['flood_flag'].mean():.3f}")
print(f"  Drought rate (syn): {syn['drought_flag'].mean():.3f}")