#!/usr/bin/env python3
"""
Generate figures for improved ΔΔU (η) and TΔS (bader_sumO, μ) models.
Style matches generate_thermodynamic_models.py (steelblue, navy edges, etc.)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import glob
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5"
OUT = f"{BASE}/rapport/figures/thermodynamic_models"
DEA = {"Br-DA", "Br-HMC", "I-DA", "I-HMC"}
RAPPORT15 = ["Al-CA", "CA", "Cl-DA", "Cl-HMC", "DA", "DHI", "F-DA", "F-HMC",
             "H2C-CA", "HMC", "HO-CA", "L-DA", "O-DHI", "P-DHI", "S-DHI"]

# --- Load data ---
files = glob.glob(f"{BASE}/data/Acetonitrile/*/redox/results/*_redox_conformers.csv")
rows = []
for f in sorted(files):
    mol = f.split("/")[-1].replace("_redox_conformers.csv", "")
    df = pd.read_csv(f)
    w = df["weight"].values; w = w / w.sum()
    row = {"molecule": mol}
    for col in ["ddU_red_kJ", "ddU_ox_kJ", "TdS_red_kJ", "TdS_ox_kJ", "EA_kJ", "IP_kJ"]:
        if col in df.columns:
            row[col] = np.sum(w * df[col].values)
    rows.append(row)
redox = pd.DataFrame(rows)
data = redox[redox["molecule"].isin(RAPPORT15)].copy()
data["eta_vert"] = (data["IP_kJ"] - data["EA_kJ"]) / 2

# Load predict_dg for bader_sumO
all_cols = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", nrows=0).columns.tolist()
keep = ["molecule", "conformer", "weight", "bader_sumO_n", "mu_n", "eta_n", "omega_n",
        "DI_sumO_n", "LI_sumC_n"]
keep = [c for c in keep if c in all_cols]
pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", usecols=keep)
pdg = pdg[~pdg["molecule"].isin(DEA)].copy()
numeric = [c for c in pdg.columns if c not in ["molecule", "conformer", "weight"]]


def bw_avg(g):
    w = g["weight"].values; w = w / w.sum()
    return pd.Series({c: np.sum(w * g[c].values) for c in numeric
                       if np.all(np.isfinite(g[c].values))})


bw = pdg.groupby("molecule").apply(bw_avg).reset_index()
data = data.merge(bw, on="molecule", how="left")
data["mu_vert"] = -(data["IP_kJ"] + data["EA_kJ"]) / 2
data = data.sort_values("molecule").reset_index(drop=True)


# ============================================================
# Figure 1: η vs ΔΔU (2 panels, same style as ZPE_vs_ddU.png)
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, target_col, title, y_label in [
    (ax1, "ddU_ox_kJ", r"(a) Oxydation", r"$\Delta\Delta U_{\mathrm{ox}}$ (kJ/mol)"),
    (ax2, "ddU_red_kJ", r"(b) Réduction", r"$\Delta\Delta U_{\mathrm{red}}$ (kJ/mol)"),
]:
    x = data["eta_vert"].values
    y = data[target_col].values
    names = data["molecule"].values

    ax.scatter(x, y, c='steelblue', s=60, zorder=5, edgecolors='navy', linewidths=0.5)

    for i, name in enumerate(names):
        ax.annotate(name, (x[i], y[i]), fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords='offset points')

    # Regression
    sl, inter, r, p, se = linregress(x, y)
    r2 = r**2
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = sl * x_fit + inter
    mae_lin = np.mean(np.abs(y - (sl * x + inter)))
    ax.plot(x_fit, y_fit, 'r-', alpha=0.7, linewidth=1.5,
            label=f'Lin: $R^2={r2:.3f}$, MAE={mae_lin:.2f}')

    # Constant model
    cst = np.mean(y)
    mae_cst = np.mean(np.abs(y - cst))
    ax.axhline(cst, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Const: $\\bar{{y}}={cst:.2f}$, MAE={mae_cst:.2f}')

    chosen = 'Linéaire' if p < 0.05 else 'Constant'
    ax.set_title(f'{title}\nModèle retenu : {chosen} ($p = {p:.4f}$)', fontsize=11)
    ax.set_xlabel(r'$\eta = (\mathrm{IP} - \mathrm{EA})/2$ (kJ/mol)', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/ddU_vs_eta.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/ddU_vs_eta.png")


# ============================================================
# Figure 2: TΔS parity plots (2 panels, same style as dG_parity)
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, target_col, title, label_prefix in [
    (ax1, "TdS_ox_kJ", r"(a) Oxydation", "ox"),
    (ax2, "TdS_red_kJ", r"(b) Réduction", "red"),
]:
    x1 = data["bader_sumO_n"].values.astype(float)
    x2 = data["mu_vert"].values.astype(float)
    y = data[target_col].values
    names = data["molecule"].values
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    n = mask.sum()

    if n >= 6:
        X = np.column_stack([x1[mask], x2[mask]])
        yv = y[mask]
        reg = LinearRegression().fit(X, yv)
        y_pred = reg.predict(X)
        mae_2d = mean_absolute_error(yv, y_pred)
        r2 = r2_score(yv, y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - 3)

        cst = np.mean(y[np.isfinite(y)])
        mae_cst = np.mean(np.abs(y[np.isfinite(y)] - cst))

        # Parity: predicted vs real
        ax.scatter(yv, y_pred, c='steelblue', s=60, zorder=5,
                   edgecolors='navy', linewidths=0.5)

        names_v = names[mask]
        for i, name in enumerate(names_v):
            ax.annotate(name, (yv[i], y_pred[i]), fontsize=6, alpha=0.7,
                        xytext=(4, 4), textcoords='offset points')

        # y=x line
        lims = [min(yv.min(), y_pred.min()), max(yv.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.15
        ax.plot([lims[0] - margin, lims[1] + margin],
                [lims[0] - margin, lims[1] + margin],
                'k--', alpha=0.5, linewidth=1, label='y = x')

        # Molecules without QTAIM (constant fallback)
        no_data = ~mask & np.isfinite(y)
        if no_data.sum() > 0:
            ax.scatter(y[no_data], np.full(no_data.sum(), cst), c='grey', s=40,
                       alpha=0.5, marker='x', zorder=4, linewidths=1.5,
                       label=f'Sans QTAIM (const)')
            for i in np.where(no_data)[0]:
                ax.annotate(names[i], (y[i], cst), fontsize=6, alpha=0.5,
                            xytext=(4, -8), textcoords='offset points', color='grey')

        ax.set_title(f'{title}\n$R^2_{{adj}}={r2_adj:.3f}$, MAE={mae_2d:.2f} (const: {mae_cst:.2f})',
                     fontsize=11)
        ax.set_xlabel(r'$T\Delta S$ DFT (kJ/mol)', fontsize=10)
        ax.set_ylabel(r'$T\Delta S$ prédit (kJ/mol)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/TdS_bader_mu_parity.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/TdS_bader_mu_parity.png")

print("  Done.")
