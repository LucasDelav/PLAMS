"""Generate RedoxMACE figures for Section 5 of the report.

Reads oof_predictions.npz from checkpoints_v6_fixedweights and produces:
  - s5_parity_dG.png       : parity plots for dG_red and dG_ox
  - s5_mae_comparison.png  : bar chart MAE Padé vs RedoxMACE per task
  - s5_r2_per_task.png     : horizontal bar chart R² per task, v6_fixedweights
  - s5_feature_ablation.png : bar chart dG MAE across v2_best, v4, v5, v6
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (apply_style, setup_grid, add_parity_line, stats_text,
                   C_RED_DARK, C_BLUE_DARK, C_GREEN, C_GRAY, C_BLUE_LIGHT,
                   C_RED_LIGHT, STATS_BOX)

apply_style()

ROOT = Path("/home/cav/Documents/Ecole/Fac/M2/ML_Batt")
OUT = Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures/section5")
OUT.mkdir(parents=True, exist_ok=True)

# Target order (matches training script)
TASK_NAMES = [
    "EA", "Edef_red", "ddU_red", "TdS_red",
    "IP", "Edef_ox", "ddU_ox", "TdS_ox",
    "residual_red", "residual_ox",
    "dG_red", "dG_ox",
]

# ---------------------------------------------------------------------------
# 1. Parity plots for dG_red and dG_ox (v6_fixedweights, out-of-fold)
# ---------------------------------------------------------------------------
data = np.load(ROOT / "checkpoints_v6_fixedweights" / "oof_predictions.npz")
pred = data["pred"]
target = data["target"]
filled = data["filled"]

idx_dg_red = TASK_NAMES.index("dG_red")
idx_dg_ox = TASK_NAMES.index("dG_ox")

mask = filled
y_red = target[mask, idx_dg_red]
p_red = pred[mask, idx_dg_red]
y_ox = target[mask, idx_dg_ox]
p_ox = pred[mask, idx_dg_ox]

def mae(p, y):
    return float(np.mean(np.abs(p - y)))

def r2(p, y):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
for ax, y, p, title, color in [
    (axes[0], y_red, p_red, r"$\Delta G_{\mathrm{red}}$", C_BLUE_DARK),
    (axes[1], y_ox, p_ox, r"$\Delta G_{\mathrm{ox}}$", C_RED_DARK),
]:
    ax.scatter(y, p, s=18, alpha=0.55, color=color, edgecolors="white",
               linewidths=0.3)
    lims = [min(y.min(), p.min()) - 20, max(y.max(), p.max()) + 20]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=0.8, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"{title} DFT (kJ/mol)")
    ax.set_ylabel(f"{title} RedoxMACE (kJ/mol)")
    ax.text(0.05, 0.95,
            f"$R^2 = {r2(p, y):.3f}$\nMAE = {mae(p, y):.1f} kJ/mol\n$n = {len(y)}$",
            transform=ax.transAxes, va="top", fontsize=12, bbox=STATS_BOX,
            family="serif")
    setup_grid(ax)
fig.suptitle("Parité out-of-fold (5-fold, v6\\_fixedweights)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "s5_parity_dG.png")
plt.close(fig)
print("  s5_parity_dG.png")

# ---------------------------------------------------------------------------
# 2. Bar chart MAE Padé vs RedoxMACE per task
# ---------------------------------------------------------------------------
pade_mae = {
    "EA": 3.7, "IP": 5.2,
    "Erelax_red": 3.8, "Erelax_ox": 4.9,
    "ddU_red": 0.7, "ddU_ox": 0.9,
    "TdS_red": 1.4, "TdS_ox": 1.8,
    "dG_red": 5.7, "dG_ox": 6.0,
}
with (ROOT / "checkpoints_v6_fixedweights" / "oof_metrics.json").open() as f:
    m = json.load(f)
mace_mae = {
    "EA": m["mae/EA"], "IP": m["mae/IP"],
    "Erelax_red": m["mae/Edef_red"], "Erelax_ox": m["mae/Edef_ox"],
    "ddU_red": m["mae/ddU_red"], "ddU_ox": m["mae/ddU_ox"],
    "TdS_red": m["mae/TdS_red"], "TdS_ox": m["mae/TdS_ox"],
    "dG_red": m["mae/dG_red"], "dG_ox": m["mae/dG_ox"],
}
labels = list(pade_mae.keys())
pretty = {
    "EA": "EA", "IP": "IP",
    "Erelax_red": r"$E_{\mathrm{relax}}^{\mathrm{red}}$",
    "Erelax_ox": r"$E_{\mathrm{relax}}^{\mathrm{ox}}$",
    "ddU_red": r"$\Delta\Delta U_{\mathrm{red}}$",
    "ddU_ox": r"$\Delta\Delta U_{\mathrm{ox}}$",
    "TdS_red": r"$T\Delta S_{\mathrm{red}}$",
    "TdS_ox": r"$T\Delta S_{\mathrm{ox}}$",
    "dG_red": r"$\Delta G_{\mathrm{red}}$",
    "dG_ox": r"$\Delta G_{\mathrm{ox}}$",
}
xpos = np.arange(len(labels))
w = 0.38

fig, ax = plt.subplots(figsize=(9.0, 3.8))
b1 = ax.bar(xpos - w/2, [pade_mae[k] for k in labels], width=w,
            color=C_BLUE_DARK, label="Padé combiné ($n=1402$)", alpha=0.85)
b2 = ax.bar(xpos + w/2, [mace_mae[k] for k in labels], width=w,
            color=C_RED_DARK, label="RedoxMACE v6 ($n=1682$)", alpha=0.85)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels([pretty[k] for k in labels], rotation=25, ha="right")
ax.set_ylabel("MAE (kJ/mol)")
ax.set_ylim(0, max(mace_mae.values()) * 1.2)
ax.legend(loc="upper left", framealpha=0.9)
setup_grid(ax)
fig.tight_layout()
fig.savefig(OUT / "s5_mae_comparison.png")
plt.close(fig)
print("  s5_mae_comparison.png")

# ---------------------------------------------------------------------------
# 3. R² per task (horizontal) for v6_fixedweights
# ---------------------------------------------------------------------------
task_r2 = [(pretty.get(k, k), m[f"r2/{raw_k}"])
           for k, raw_k in [
               ("dG_red", "dG_red"),
               ("dG_ox", "dG_ox"),
               ("EA", "EA"),
               ("IP", "IP"),
               ("Erelax_red", "Edef_red"),
               ("Erelax_ox", "Edef_ox"),
               ("ddU_red", "ddU_red"),
               ("ddU_ox", "ddU_ox"),
               ("TdS_red", "TdS_red"),
               ("TdS_ox", "TdS_ox"),
           ]]
task_r2 = task_r2[::-1]  # so best on top
names = [t[0] for t in task_r2]
vals = [t[1] for t in task_r2]
colors = [C_GREEN if v >= 0.80 else (C_BLUE_LIGHT if v >= 0.4 else C_RED_LIGHT)
          for v in vals]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.6,
               alpha=0.9)
ax.axvline(0, color=C_GRAY, lw=0.5)
ax.axvline(0.8, color=C_GREEN, lw=0.6, linestyle=":", alpha=0.6)
for bar, v in zip(bars, vals):
    ax.text(v + 0.02 if v >= 0 else v - 0.02, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=11)
ax.set_xlabel(r"$R^2$ (out-of-fold, v6\_fixedweights)")
ax.set_xlim(-0.15, 1.05)
setup_grid(ax, limit_ticks=False)
fig.tight_layout()
fig.savefig(OUT / "s5_r2_per_task.png")
plt.close(fig)
print("  s5_r2_per_task.png")

# ---------------------------------------------------------------------------
# 4. Ablation bar chart: dG avg MAE across configurations
# ---------------------------------------------------------------------------
configs = [
    ("v4\\_baseline\n(sans Koopmans)", 26.39, C_GRAY),
    ("v5\\_koopmans\n(Kendall)", 19.81, C_BLUE_LIGHT),
    ("v2\\_best\\_tuned\n(jeu v6)", 15.86, C_BLUE_DARK),
    ("v6\\_fixedweights\n(meilleur)", 14.02, C_GREEN),
    ("Padé combiné\n(section 4)", 5.85, C_RED_DARK),
]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
xp = np.arange(len(configs))
vals = [c[1] for c in configs]
cols = [c[2] for c in configs]
bars = ax.bar(xp, vals, color=cols, edgecolor="white", linewidth=0.6, alpha=0.9)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}",
            ha="center", va="bottom", fontsize=11)
ax.set_xticks(xp)
ax.set_xticklabels([c[0] for c in configs], fontsize=10)
ax.set_ylabel(r"MAE $\overline{\Delta G}$ (kJ/mol)")
ax.set_ylim(0, max(vals) * 1.2)
setup_grid(ax, limit_ticks=False)
fig.tight_layout()
fig.savefig(OUT / "s5_ablation_mae.png")
plt.close(fig)
print("  s5_ablation_mae.png")

print("\nFigures written to", OUT)
