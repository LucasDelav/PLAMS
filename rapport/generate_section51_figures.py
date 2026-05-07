"""Generate classical ML baseline figures for section 5.1.

Produces:
  - s51_mae_per_task.png  : grouped bar chart MAE(RF/GB/SVR) per task.
  - s51_parity_dG_gb.png  : parity plots dG_red/ox for the best model (GB).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (apply_style, setup_grid, C_RED_DARK, C_BLUE_DARK, C_GREEN,
                   C_GRAY, STATS_BOX)

apply_style()

ROOT = Path("/home/cav/Documents/Ecole/Fac/M2/ML_Batt/checkpoints/baselines")
OUT = Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures/section51")
OUT.mkdir(parents=True, exist_ok=True)

# Task order matches train_baselines.py (12 targets)
TASKS = [
    "EA", "Edef_red", "ddU_red", "TdS_red",
    "IP", "Edef_ox", "ddU_ox", "TdS_ox",
    "residual_red", "residual_ox",
    "dG_red", "dG_ox",
]

PRETTY = {
    "EA": "EA", "IP": "IP",
    "Edef_red": r"$E_{\mathrm{relax}}^{\mathrm{red}}$",
    "Edef_ox":  r"$E_{\mathrm{relax}}^{\mathrm{ox}}$",
    "ddU_red":  r"$\Delta\Delta U_{\mathrm{red}}$",
    "ddU_ox":   r"$\Delta\Delta U_{\mathrm{ox}}$",
    "TdS_red":  r"$T\Delta S_{\mathrm{red}}$",
    "TdS_ox":   r"$T\Delta S_{\mathrm{ox}}$",
    "residual_red": r"res$_{\mathrm{red}}$",
    "residual_ox":  r"res$_{\mathrm{ox}}$",
    "dG_red":   r"$\Delta G_{\mathrm{red}}$",
    "dG_ox":    r"$\Delta G_{\mathrm{ox}}$",
}

# ---------------------------------------------------------------------------
# 1. MAE grouped bar chart
# ---------------------------------------------------------------------------
with (ROOT / "rf" / "metrics.json").open() as f:
    rf_m = json.load(f)
with (ROOT / "gb" / "metrics.json").open() as f:
    gb_m = json.load(f)
with (ROOT / "svr" / "metrics.json").open() as f:
    svr_m = json.load(f)

rf  = [rf_m[f"mae/{t}"] for t in TASKS]
gb  = [gb_m[f"mae/{t}"] for t in TASKS]
svr = [svr_m[f"mae/{t}"] for t in TASKS]

xp = np.arange(len(TASKS))
w = 0.27

fig, ax = plt.subplots(figsize=(10.0, 4.0))
b1 = ax.bar(xp - w, rf,  width=w, color=C_BLUE_DARK, label="Random Forest",   alpha=0.88)
b2 = ax.bar(xp,      gb,  width=w, color=C_GREEN,     label="XGBoost",         alpha=0.88)
b3 = ax.bar(xp + w, svr, width=w, color=C_RED_DARK,  label="SVR",             alpha=0.88)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(xp)
ax.set_xticklabels([PRETTY[t] for t in TASKS], rotation=25, ha="right")
ax.set_ylabel("MAE (kJ/mol)")
ax.set_ylim(0, max(max(rf), max(gb), max(svr)) * 1.18)
ax.legend(loc="upper right", framealpha=0.9)
setup_grid(ax, limit_ticks=False)
fig.tight_layout()
fig.savefig(OUT / "s51_mae_per_task.png")
plt.close(fig)
print("  s51_mae_per_task.png")

# ---------------------------------------------------------------------------
# 2. Parity plots for best model (GB) on dG_red and dG_ox
# ---------------------------------------------------------------------------
data = np.load(ROOT / "gb" / "oof_predictions.npz")
pred = data["pred"]; target = data["target"]

idx_red = TASKS.index("dG_red")
idx_ox  = TASKS.index("dG_ox")

def r2(p, y):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot

def mae(p, y):
    return float(np.mean(np.abs(p - y)))

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
for ax, idx, title, color in [
    (axes[0], idx_red, r"$\Delta G_{\mathrm{red}}$", C_BLUE_DARK),
    (axes[1], idx_ox,  r"$\Delta G_{\mathrm{ox}}$",  C_RED_DARK),
]:
    y = target[:, idx]; p = pred[:, idx]
    ax.scatter(y, p, s=16, alpha=0.55, color=color, edgecolors="white",
               linewidths=0.3)
    lims = [min(y.min(), p.min()) - 20, max(y.max(), p.max()) + 20]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=0.8, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"{title} DFT (kJ/mol)")
    ax.set_ylabel(f"{title} XGBoost (kJ/mol)")
    ax.text(0.05, 0.95,
            f"$R^2 = {r2(p, y):.3f}$\nMAE = {mae(p, y):.1f} kJ/mol\n$n = {len(y)}$",
            transform=ax.transAxes, va="top", fontsize=12, bbox=STATS_BOX,
            family="serif")
    setup_grid(ax)

fig.suptitle("Parité out-of-fold — XGBoost (5-fold mol-aware)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "s51_parity_dG_gb.png")
plt.close(fig)
print("  s51_parity_dG_gb.png")

print("\nFigures written to", OUT)
