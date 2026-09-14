"""
16c_figures_v4.py
Regenerate F1 (7-panel time series), F4 (scatter), F6 (bar metrics) for v4.
Updated to include gauge 03193000 (Kanawha R. @ Kanawha Falls,WV).
"""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

PRED = "/home/sandbox/huc0505/data/predictions_full_v4.csv"
MET  = "/home/sandbox/huc0505/data/metrics_v4.json"
FIG  = "/home/sandbox/huc0505/figures"

GAUGES = [3171000,3176500,3192000,3193000,3197000,3200500,3198000]
NAMES  = {3171000:"New R. @ Radford,VA",      3176500:"New R. @ Glen Lyn,VA",
          3192000:"Gauley R. @ Belva,WV",     3193000:"Kanawha R. @ Kanawha Falls,WV",
          3197000:"Elk R. @ Queen Shoals,WV", 3200500:"Coal R. @ Tornado,WV",
          3198000:"Kanawha R. @ Charleston,WV"}

df  = pd.read_csv(PRED, parse_dates=["date"])
met = json.load(open(MET))

def nse(obs,sim): return 1-np.sum((obs-sim)**2)/np.sum((obs-obs.mean())**2)
def kge(obs,sim):
    r=np.corrcoef(obs,sim)[0,1]; a=sim.std()/obs.std(); b=sim.mean()/obs.mean()
    return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)

SPLIT_DATE = pd.Timestamp("2023-01-01")

# ══════════════════════════════════════════════════
# F1 — 7-panel time series 2000-2025
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(7,1, figsize=(14,21), sharex=False)
for ax, gid in zip(axes, GAUGES):
    g  = df[df["gauge_id"]==gid].sort_values("date")
    tr = g[g["split"]=="train"]
    te = g[g["split"]=="test"]
    ax.fill_between(g["date"], g["q05"], g["q95"], alpha=0.18, color="#2196F3", label="90% UQ band")
    ax.plot(tr["date"], tr["flow_m3s"], color="#78909C", lw=0.7, alpha=0.7, label="Observed (train)")
    ax.plot(te["date"], te["flow_m3s"], color="#0D47A1", lw=0.9, label="Observed (test)")
    ax.plot(tr["date"], tr["pred"],     color="#FF8F00", lw=0.7, alpha=0.7, label="Predicted (train)")
    ax.plot(te["date"], te["pred"],     color="#E53935", lw=0.9, label="Predicted (test)")
    ax.axvline(SPLIT_DATE, color="k", lw=1.1, ls="--", alpha=0.6)
    nse_v = met[str(gid)]["nse_test"]
    ax.set_title(f"{NAMES[gid]}  (NSE={nse_v:.3f})", fontsize=9, pad=3)
    ax.set_ylabel("Flow (m³/s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(g["date"].min(), g["date"].max())
axes[0].legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.7)
axes[0].text(SPLIT_DATE, axes[0].get_ylim()[1]*0.95, "  Test →", fontsize=7, color="k")
fig.suptitle("Figure 1 — Observed vs Predicted Streamflow (2000–2025)", fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(f"{FIG}/F1_v4_streamflow.png", dpi=150, bbox_inches="tight")
plt.close()
print("F1 saved")

# ══════════════════════════════════════════════════
# F4 — 2-panel scatter train / test
# ══════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
for ax, split, title in [(ax1,"train","Training (2000–2022)"),
                          (ax2,"test","Test (2023–2025)")]:
    sub = df[df["split"]==split]
    obs = sub["flow_m3s"].values; sim = sub["pred"].values
    nse_v=nse(obs,sim); r=np.corrcoef(obs,sim)[0,1]
    rmse=np.sqrt(mean_squared_error(obs,sim))
    ax.scatter(obs, sim, s=3, alpha=0.3, color="#1565C0" if split=="train" else "#B71C1C")
    mx=max(obs.max(),sim.max())*1.05
    ax.plot([0,mx],[0,mx],"k--",lw=0.9)
    ax.set_xlim(0,mx); ax.set_ylim(0,mx)
    ax.set_xlabel("Observed (m³/s)", fontsize=9); ax.set_ylabel("Predicted (m³/s)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.text(0.04,0.92, f"NSE={nse_v:.3f}\nr={r:.3f}\nRMSE={rmse:.1f} m³/s",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
fig.suptitle("Figure 4 — Validation Scatter: All Gauges Combined", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/F4_v4_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("F4 saved")

# ══════════════════════════════════════════════════
# F6 — 3-panel bar NSE / KGE / RMSE per gauge
# ══════════════════════════════════════════════════
labels = [NAMES[g].split("@")[0].strip() for g in GAUGES]
nse_v  = [met[str(g)]["nse_test"]  for g in GAUGES]
kge_v  = [met[str(g)]["kge_test"]  for g in GAUGES]
rmse_v = [met[str(g)]["rmse_test"] for g in GAUGES]

x = np.arange(len(GAUGES))
fig, axes = plt.subplots(1,3, figsize=(14,4))
colors = ["#1B5E20","#1565C0","#B71C1C"]
for ax, vals, ylabel, col in zip(axes,
                                  [nse_v,kge_v,rmse_v],
                                  ["NSE","KGE","RMSE (m³/s)"],
                                  colors):
    bars=ax.bar(x, vals, color=col, alpha=0.8, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(ylabel, fontsize=10)
    if ylabel!="RMSE (m³/s)": ax.axhline(0, color="k", lw=0.6, ls="--")
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Figure 6 — Per-Gauge Performance Metrics (Test 2023–2025)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/F6_v4_spatial.png", dpi=150, bbox_inches="tight")
plt.close()
print("F6 saved")
print("\nAll v4 figures saved to", FIG)