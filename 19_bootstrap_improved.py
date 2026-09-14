#!/usr/bin/env python3
"""
19_bootstrap_improved.py
Improved uncertainty quantification for HUC 0505 ML streamflow predictions.

Methods compared:
  1. Current   : naive RF tree variance (baseline from predictions_full_v4.csv)
  2. MBB       : Moving Block Bootstrap on training residuals (l=90 days)
  3. Conformal : Split conformal prediction  (calibration 2020–2022, guaranteed coverage)
  4. GBR-Q     : Gradient Boosting quantile regression (alpha=0.05/0.95, sklearn)

Outputs:
  /home/sandbox/huc0505/data/bootstrap_comparison.csv
  /home/sandbox/huc0505/figures/F7_bootstrap_comparison.png
"""

# Prevent OpenMP / BLAS thread-lock on shared systems — must be set BEFORE sklearn import
import os
os.environ['OMP_NUM_THREADS']       = '1'
os.environ['OPENBLAS_NUM_THREADS']  = '1'
os.environ['MKL_NUM_THREADS']       = '1'
os.environ['NUMEXPR_NUM_THREADS']   = '1'

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_CSV  = '/home/sandbox/huc0505/data/predictions_full_v4.csv'
FEAT_CSV  = '/home/sandbox/huc0505/data/features_augmented_v4.csv'
OUT_CSV   = '/home/sandbox/huc0505/data/bootstrap_comparison.csv'
OUT_FIG   = '/home/sandbox/huc0505/figures/F7_bootstrap_comparison.png'

# ── Date-based split boundaries ───────────────────────────────────────────────
TRAIN_END  = pd.Timestamp('2019-12-31')
VAL_START  = pd.Timestamp('2020-01-01')
VAL_END    = pd.Timestamp('2022-12-31')
TEST_START = pd.Timestamp('2023-01-01')

# ── v4 feature list (20 features) ─────────────────────────────────────────────
FEATURES = [
    'pr_mm','pet_mm','vpd_kPa','tmax_C','tmin_C','wb_mm','tmean_C',
    'pr_7d','pr_30d','pr_90d','pet_7d','pet_30d','wb_30d','wb_90d',
    'doy_sin','doy_cos','spei_30d','spei_90d','flow_lag1','flow_lag7'
]

# ── Metric helpers ────────────────────────────────────────────────────────────

def emp_coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))

def mean_width(lo, hi):
    return float(np.mean(hi - lo))

def winkler(y, lo, hi, alpha=0.10):
    """Winkler interval score – lower is better."""
    w   = hi - lo
    pen = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return float(np.mean(w + pen))

# ── MBB helper ────────────────────────────────────────────────────────────────

def mbb_residual_quantiles(residuals, block_len=90, n_boot=1000, alpha=0.10):
    """
    Moving Block Bootstrap: pool residuals from n_boot randomly sampled
    overlapping blocks of length block_len; return (q_lo, q_hi).
    This preserves temporal autocorrelation structure in the error distribution.
    """
    n = len(residuals)
    if n < block_len:
        return (float(np.quantile(residuals, alpha / 2)),
                float(np.quantile(residuals, 1.0 - alpha / 2)))
    max_start = n - block_len
    pool = []
    for _ in range(n_boot):
        k = np.random.randint(0, max_start + 1)
        pool.extend(residuals[k : k + block_len])
    pool = np.array(pool)
    return float(np.quantile(pool, alpha / 2)), float(np.quantile(pool, 1.0 - alpha / 2))

# ── Conformal helper ─────────────────────────────────────────────────────────

def conformal_q(cal_scores, alpha=0.10):
    """
    Split conformal quantile: smallest level that guarantees ≥(1-alpha) coverage.
    """
    n = len(cal_scores)
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, level))

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
pred_df = pd.read_csv(PRED_CSV, parse_dates=['date'])
feat_df = pd.read_csv(FEAT_CSV, parse_dates=['date'])

if 'synthetic' in feat_df.columns:
    feat_df = feat_df[feat_df['synthetic'] == 0].copy()

gauges = sorted(pred_df['gauge_id'].unique())
print(f"Gauges ({len(gauges)}): {gauges}")

# ── Per-gauge analysis ────────────────────────────────────────────────────────
records = []

for gid in gauges:
    print(f"\n── Gauge {gid} ──")

    gp = pred_df[pred_df['gauge_id'] == gid].sort_values('date').reset_index(drop=True)

    gp_test  = gp[gp['date'] >= TEST_START]
    gp_train = gp[gp['date'] <= TRAIN_END]
    gp_cal   = gp[gp['date'].between(VAL_START, VAL_END)]

    y_test    = gp_test['flow_m3s'].values
    yhat_test = gp_test['pred'].values

    if len(y_test) == 0:
        print("  No test data – skipping"); continue

    # ── 1. Current (RF tree variance) ────────────────────────────────────────
    lo_c = gp_test['q05'].values
    hi_c = gp_test['q95'].values
    records.append(dict(gauge_id=gid, method='Current',
                        coverage=emp_coverage(y_test, lo_c, hi_c),
                        mean_width=mean_width(lo_c, hi_c),
                        winkler_score=winkler(y_test, lo_c, hi_c)))
    r = records[-1]
    print(f"  Current  : cov={r['coverage']:.3f}  width={r['mean_width']:7.1f}  winkler={r['winkler_score']:8.1f}")

    # ── 2. MBB ───────────────────────────────────────────────────────────────
    res_tr = gp_train['flow_m3s'].values - gp_train['pred'].values
    q_lo, q_hi = mbb_residual_quantiles(res_tr, block_len=90, n_boot=1000)
    lo_m = yhat_test + q_lo
    hi_m = yhat_test + q_hi
    records.append(dict(gauge_id=gid, method='MBB',
                        coverage=emp_coverage(y_test, lo_m, hi_m),
                        mean_width=mean_width(lo_m, hi_m),
                        winkler_score=winkler(y_test, lo_m, hi_m)))
    r = records[-1]
    print(f"  MBB      : cov={r['coverage']:.3f}  width={r['mean_width']:7.1f}  winkler={r['winkler_score']:8.1f}")

    # ── 3. Split Conformal ────────────────────────────────────────────────────
    if len(gp_cal) < 20:
        print(f"  Conformal: skip – {len(gp_cal)} cal pts")
        records.append(dict(gauge_id=gid, method='Conformal',
                            coverage=np.nan, mean_width=np.nan, winkler_score=np.nan))
    else:
        scores = np.abs(gp_cal['flow_m3s'].values - gp_cal['pred'].values)
        q_hat  = conformal_q(scores, alpha=0.10)
        lo_cf  = yhat_test - q_hat
        hi_cf  = yhat_test + q_hat
        records.append(dict(gauge_id=gid, method='Conformal',
                            coverage=emp_coverage(y_test, lo_cf, hi_cf),
                            mean_width=mean_width(lo_cf, hi_cf),
                            winkler_score=winkler(y_test, lo_cf, hi_cf)))
        r = records[-1]
        print(f"  Conformal: cov={r['coverage']:.3f}  width={r['mean_width']:7.1f}  winkler={r['winkler_score']:8.1f}  q̂={q_hat:.1f}")

    # ── 4. GBR-Q ─────────────────────────────────────────────────────────────
    gf = feat_df[feat_df['gauge_id'] == gid].copy()
    miss = [f for f in FEATURES if f not in gf.columns]
    if miss:
        print(f"  GBR-Q    : missing {miss} – skip")
        records.append(dict(gauge_id=gid, method='GBR-Q',
                            coverage=np.nan, mean_width=np.nan, winkler_score=np.nan))
        continue

    gf_tr = gf[gf['date'] <= TRAIN_END].dropna(subset=FEATURES + ['flow_m3s'])
    gf_te = gf[gf['date'] >= TEST_START].dropna(subset=FEATURES + ['flow_m3s'])

    if len(gf_tr) < 100 or len(gf_te) < 10:
        print(f"  GBR-Q    : skip – train={len(gf_tr)}, test={len(gf_te)}")
        records.append(dict(gauge_id=gid, method='GBR-Q',
                            coverage=np.nan, mean_width=np.nan, winkler_score=np.nan))
        continue

    X_tr  = gf_tr[FEATURES].values
    y_tr2 = gf_tr['flow_m3s'].values
    X_te  = gf_te[FEATURES].values
    y_te2 = gf_te['flow_m3s'].values

    print(f"  GBR-Q    : fitting on {len(X_tr)} samples ...", end='', flush=True)
    kw = dict(n_estimators=100, max_depth=4, learning_rate=0.05,
              subsample=0.8, min_samples_leaf=10, random_state=42)
    gbr_lo = GradientBoostingRegressor(loss='quantile', alpha=0.05, **kw)
    gbr_hi = GradientBoostingRegressor(loss='quantile', alpha=0.95, **kw)
    gbr_lo.fit(X_tr, y_tr2)
    gbr_hi.fit(X_tr, y_tr2)

    lo_g = gbr_lo.predict(X_te)
    hi_g = gbr_hi.predict(X_te)
    lo_g, hi_g = np.minimum(lo_g, hi_g), np.maximum(lo_g, hi_g)

    records.append(dict(gauge_id=gid, method='GBR-Q',
                        coverage=emp_coverage(y_te2, lo_g, hi_g),
                        mean_width=mean_width(lo_g, hi_g),
                        winkler_score=winkler(y_te2, lo_g, hi_g)))
    r = records[-1]
    print(f" done.  cov={r['coverage']:.3f}  width={r['mean_width']:7.1f}  winkler={r['winkler_score']:8.1f}")

# ── Aggregate ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

means = (df.groupby('method', sort=False)[['coverage','mean_width','winkler_score']]
           .mean().reset_index())
means['gauge_id'] = 'MEAN'
df = pd.concat([df, means], ignore_index=True)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nResults saved → {OUT_CSV}")
print("\n── MEAN across gauges ──")
print(df[df['gauge_id'] == 'MEAN'][['method','coverage','mean_width','winkler_score']]
      .to_string(index=False))

# ── Figure ────────────────────────────────────────────────────────────────────
METHODS = ['Current', 'MBB', 'Conformal', 'GBR-Q']
LABELS  = ['Current\n(RF Tree Var.)', 'MBB\n(Block Bootstrap)',
            'Conformal\n(Split CP)', 'GBR-Q\n(Quantile Reg.)']
COLORS  = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

gauge_ids = [g for g in df['gauge_id'].unique() if g != 'MEAN']
xlabels   = gauge_ids + ['MEAN']
x         = np.arange(len(xlabels))
bw        = 0.18
offsets   = np.linspace(-(len(METHODS)-1)/2*bw, (len(METHODS)-1)/2*bw, len(METHODS))

fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=150)
fig.suptitle(
    'Uncertainty Quantification Method Comparison\n'
    'HUC 0505 – 7 USGS Gauges  |  Test period: 2023–2025',
    fontsize=13, fontweight='bold', y=0.998)

def get_vals(method, metric):
    vals = []
    for gid in xlabels:
        row = df[(df['gauge_id'] == gid) & (df['method'] == method)]
        vals.append(float(row[metric].values[0]) if len(row) else np.nan)
    return vals

# ── Panel (a): Coverage ───────────────────────────────────────────────────────
ax = axes[0]
for m, lbl, col, off in zip(METHODS, LABELS, COLORS, offsets):
    ax.bar(x + off, get_vals(m, 'coverage'), bw, label=lbl,
           color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.axhline(0.90, color='#2c3e50', ls='--', lw=1.8, label='Target 90%', zorder=6)
ax.set_ylabel('Empirical Coverage', fontsize=10)
ax.set_title('(a) Empirical Coverage  [target = 0.90]', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=8)
ax.set_ylim(0, 1.09)
ax.legend(loc='upper left', fontsize=7.5, ncol=5, framealpha=0.8)
ax.grid(axis='y', alpha=0.25)

# ── Panel (b): Mean Width ─────────────────────────────────────────────────────
ax = axes[1]
for m, lbl, col, off in zip(METHODS, LABELS, COLORS, offsets):
    ax.bar(x + off, get_vals(m, 'mean_width'), bw, label=lbl,
           color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.set_ylabel('Mean Width (m³/s)', fontsize=10)
ax.set_title('(b) Mean Interval Width  [narrower is better at equal coverage]', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.25)

# ── Panel (c): Winkler Score ──────────────────────────────────────────────────
ax = axes[2]
for m, lbl, col, off in zip(METHODS, LABELS, COLORS, offsets):
    ax.bar(x + off, get_vals(m, 'winkler_score'), bw, label=lbl,
           color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.set_ylabel('Winkler Score', fontsize=10)
ax.set_title('(c) Winkler Interval Score  [lower = better; penalises under-coverage and wide bands]', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.975])
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved → {OUT_FIG}")
print("Done.")