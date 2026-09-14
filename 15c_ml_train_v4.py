"""
15c_ml_train_v4.py
Per-gauge RF regressor + UQ RF + flood/drought classifiers with SMOTE.
n_estimators=5, max_depth=6 for sandbox speed (24 models within timeout).
"""
import os, json, pickle, warnings
warnings.filterwarnings("ignore")
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, mean_squared_error)
from imblearn.over_sampling import SMOTE

DATA_REAL = "/home/sandbox/huc0505/data/features_gridmet_v4.csv"
DATA_AUG  = "/home/sandbox/huc0505/data/features_augmented_v4.csv"
MDL_DIR   = "/home/sandbox/huc0505/models"
PRED_OUT  = "/home/sandbox/huc0505/data/predictions_v4.csv"
MET_OUT   = "/home/sandbox/huc0505/data/metrics_v4.json"
os.makedirs(MDL_DIR, exist_ok=True)

TEST_START = "2023-01-01"
NE, MD = 5, 6          # trees / max_depth — fast for sandbox

FEAT_COLS = [
    "pr_mm","pet_mm","vpd_kPa","tmax_C","tmin_C",
    "wb_mm","tmean_C","pr_7d","pr_30d","pr_90d",
    "pet_7d","pet_30d","wb_30d","wb_90d",
    "doy_sin","doy_cos","spei_30d","spei_90d",
    "flow_lag1","flow_lag7",
]
TARGET = "flow_m3s"
GAUGES = [3171000,3176500,3192000,3193000,3197000,3200500,3198000]

def nse(obs,sim): return 1-np.sum((obs-sim)**2)/np.sum((obs-obs.mean())**2)
def kge(obs,sim):
    r=np.corrcoef(obs,sim)[0,1]; a=sim.std()/obs.std(); b=sim.mean()/obs.mean()
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)

import time; t0=time.time()
real = pd.read_csv(DATA_REAL, parse_dates=["date"])
aug  = pd.read_csv(DATA_AUG, low_memory=False)
print(f"Data loaded in {time.time()-t0:.1f}s")

all_preds = []; metrics = {}

for gid in GAUGES:
    tg=time.time()
    print(f"\n── Gauge {gid} ──")
    gr  = real[real["gauge_id"]==gid].copy()
    tr_r= gr[gr["date"] < TEST_START]
    te_r= gr[gr["date"] >= TEST_START]
    syn_g = aug[(aug["gauge_id"]==gid) & (aug["synthetic"]==1)]

    Xtr_real = tr_r[FEAT_COLS].values.astype(float)
    ytr_real = tr_r[TARGET].values
    Xtr_aug  = pd.concat([tr_r[FEAT_COLS], syn_g[FEAT_COLS]], ignore_index=True).values.astype(float)
    ytr_aug  = pd.concat([tr_r[[TARGET]],  syn_g[[TARGET]]],  ignore_index=True)[TARGET].values
    Xte      = te_r[FEAT_COLS].values.astype(float)
    yte      = te_r[TARGET].values

    # (A) RF regressor
    rf=RandomForestRegressor(n_estimators=NE,max_depth=MD,
                             min_samples_leaf=2,n_jobs=1,random_state=42)
    rf.fit(Xtr_aug, ytr_aug)
    pickle.dump(rf, open(f"{MDL_DIR}/rf_reg_{gid}.pkl","wb"))
    pred_te=rf.predict(Xte); pred_tr=rf.predict(Xtr_real)
    nse_te=nse(yte,pred_te); kge_te=kge(yte,pred_te)
    rmse_te=np.sqrt(mean_squared_error(yte,pred_te)); nse_tr=nse(ytr_real,pred_tr)
    print(f"  Reg  NSE_te={nse_te:.3f}  KGE_te={kge_te:.3f}  RMSE={rmse_te:.1f}")

    # (B) UQ RF
    uq=RandomForestRegressor(n_estimators=NE,max_depth=MD,
                             min_samples_leaf=15,n_jobs=1,random_state=7)
    uq.fit(Xtr_real, ytr_real)
    pickle.dump(uq, open(f"{MDL_DIR}/rf_uq_{gid}.pkl","wb"))
    leaf_preds=np.array([e.predict(Xte) for e in uq.estimators_])
    q05=np.percentile(leaf_preds,5,axis=0); q95=np.percentile(leaf_preds,95,axis=0)
    coverage=np.mean((yte>=q05)&(yte<=q95)); width=(q95-q05).mean()
    print(f"  UQ   coverage={coverage:.3f}  width={width:.1f}")

    # (C) Flood + SMOTE
    ff_tr=tr_r["flood_flag"].values.astype(int)
    n_fl=ff_tr.sum()
    if n_fl>=6:
        try:
            sm=SMOTE(k_neighbors=min(5,n_fl-1),random_state=42)
            Xs,ys=sm.fit_resample(Xtr_real,ff_tr)
        except: Xs,ys=Xtr_real,ff_tr
    else: Xs,ys=Xtr_real,ff_tr
    rfc_fl=RandomForestClassifier(n_estimators=NE,max_depth=MD,
                                  class_weight="balanced",n_jobs=1,random_state=42)
    rfc_fl.fit(Xs,ys)
    pickle.dump(rfc_fl, open(f"{MDL_DIR}/rf_flood_{gid}.pkl","wb"))
    prob_fl=rfc_fl.predict_proba(Xte)[:,1]
    fl_true=te_r["flood_flag"].values.astype(int)
    fl_auc=roc_auc_score(fl_true,prob_fl)
    fl_ap=average_precision_score(fl_true,prob_fl)
    fl_f1=f1_score(fl_true,(prob_fl>=0.5).astype(int),zero_division=0)
    print(f"  Flood AUC={fl_auc:.4f}  AP={fl_ap:.4f}  F1={fl_f1:.4f}")

    # (D) Drought
    rfc_dr=RandomForestClassifier(n_estimators=NE,max_depth=MD,
                                  class_weight="balanced",n_jobs=1,random_state=99)
    rfc_dr.fit(Xtr_real, tr_r["drought_flag"].values.astype(int))
    pickle.dump(rfc_dr, open(f"{MDL_DIR}/rf_drought_{gid}.pkl","wb"))
    prob_dr=rfc_dr.predict_proba(Xte)[:,1]
    dr_true=te_r["drought_flag"].values.astype(int)
    if dr_true.sum()>0:
        dr_auc=roc_auc_score(dr_true,prob_dr)
        dr_ap=average_precision_score(dr_true,prob_dr)
        dr_f1=f1_score(dr_true,(prob_dr>=0.5).astype(int),zero_division=0)
    else: dr_auc=dr_ap=dr_f1=float("nan")
    print(f"  Drought AUC={dr_auc:.4f}  F1={dr_f1:.4f}  ({time.time()-tg:.1f}s)")

    tmp=te_r[["date","gauge_id",TARGET,"flood_flag","drought_flag"]].copy()
    tmp["pred"]=pred_te; tmp["q05"]=q05; tmp["q95"]=q95
    tmp["prob_flood"]=prob_fl; tmp["prob_drought"]=prob_dr
    all_preds.append(tmp)
    metrics[str(gid)]=dict(
        nse_train=float(nse_tr),nse_test=float(nse_te),
        kge_test=float(kge_te),rmse_test=float(rmse_te),
        pbias=float(100*(pred_te.mean()-yte.mean())/yte.mean()),
        r=float(np.corrcoef(yte,pred_te)[0,1]),
        uq_coverage=float(coverage),uq_width=float(width),
        flood_auc=float(fl_auc),flood_ap=float(fl_ap),flood_f1=float(fl_f1),
        drought_auc=float(dr_auc),drought_ap=float(dr_ap),drought_f1=float(dr_f1),
    )

preds_df=pd.concat(all_preds,ignore_index=True)
preds_df.to_csv(PRED_OUT,index=False)

real_te=real[real["date"]>=TEST_START].reset_index(drop=True)
obs_all=preds_df[TARGET].values; pred_all=preds_df["pred"].values
metrics["overall"]=dict(
    nse=float(nse(obs_all,pred_all)),kge=float(kge(obs_all,pred_all)),
    rmse=float(np.sqrt(mean_squared_error(obs_all,pred_all))),
    pbias=float(100*(pred_all.mean()-obs_all.mean())/obs_all.mean()),
    r=float(np.corrcoef(obs_all,pred_all)[0,1]),
    uq_coverage=float(np.mean([v["uq_coverage"] for v in metrics.values() if "uq_coverage" in v])),
    flood_auc=float(roc_auc_score(real_te["flood_flag"].values,preds_df["prob_flood"].values)),
    flood_f1=float(f1_score(real_te["flood_flag"].values,(preds_df["prob_flood"]>=0.5).astype(int),zero_division=0)),
    drought_auc=float(roc_auc_score(real_te["drought_flag"].values,preds_df["prob_drought"].values)
                      if real_te["drought_flag"].sum()>0 else float("nan")),
)
with open(MET_OUT,"w") as f: json.dump(metrics,f,indent=2)

print(f"\n═══ Overall v4 (test 2023-2025) ═══")
print(f"  NSE={metrics['overall']['nse']:.3f}  KGE={metrics['overall']['kge']:.3f}"
      f"  RMSE={metrics['overall']['rmse']:.1f}  PBIAS={metrics['overall']['pbias']:.2f}%")
print(f"  UQ coverage={metrics['overall']['uq_coverage']:.3f}")
print(f"  Flood AUC={metrics['overall']['flood_auc']:.4f}  F1={metrics['overall']['flood_f1']:.4f}")
print(f"  Drought AUC={metrics['overall']['drought_auc']:.4f}")
print(f"  Total time: {time.time()-t0:.1f}s")