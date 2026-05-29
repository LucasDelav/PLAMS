#!/usr/bin/env python
"""
Parity plots des approches EA/IP explorées (récit chronologique partie 3) :
 - Koopmans (orbitale frontière unique, calibré) : bon mais imparfait.
 - Politzer-Murray (ALIE / V_S,min, calibré) : échec.
Style maison (rapport/style.py), points coloriés par molécule (muets), CV par molécule.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, FIGSIZE_DOUBLE_V, C_RED_DARK, C_BLUE_DARK, C_GRAY)
import matplotlib.pyplot as plt

apply_style()
EH = 2625.499638
EXCL = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
FIG = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"


def muted(rgba):
    h, s, v = mcolors.rgb_to_hsv(np.array(rgba[:3]))
    return (*mcolors.hsv_to_rgb((h, s * 0.45, min(v, 0.78))), 1.0)


def palette_for(mols):
    u = sorted(set(mols))
    cmap = plt.cm.gist_rainbow(np.linspace(0, 1, len(u)))
    return {m: muted(cmap[i]) for i, m in enumerate(u)}


def cv_calibrate(raw, dft, groups):
    """Calibration linéaire 1D en validation croisée (GroupKFold par molécule)."""
    x = np.asarray(raw, float).reshape(-1, 1)
    y = np.asarray(dft, float)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(x, y, groups):
        lr = LinearRegression().fit(x[tr], y[tr])
        pred[te] = lr.predict(x[te])
    return pred


# Limites d'axes partagées avec parity_plots_p3.py (mêmes échelles Koopmans/Politzer/CDFT)
IP_LIMS = (505, 715)
EA_LIMS = (40, 432)

def panel(ax, dft, pred, mols, pal, color, label, lims=None):
    cols = [pal[m] for m in mols]
    ax.scatter(dft, pred, c=cols, **{k: v for k, v in SCATTER_KW.items() if k != "c"})
    regression_with_band(ax, dft, pred, color=color)
    if lims is None:
        allv = np.concatenate([dft, pred]); mar = (allv.max() - allv.min()) * 0.12
        lims = [allv.min() - mar, allv.max() + mar]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2 = 1 - np.sum((dft - pred)**2) / np.sum((dft - dft.mean())**2)
    stats_text(ax, r2, np.mean(np.abs(pred - dft)), len(dft))
    ax.set_xlabel(f"${label}^{{\\mathrm{{DFT}}}}$ (kJ/mol)")
    ax.set_ylabel(f"${label}$ calibré (kJ/mol)")
    setup_grid(ax)


def one(dft, raw, mols, color, label, fname, lims=None):
    dft = np.asarray(dft, float); raw = np.asarray(raw, float); mols = np.asarray(mols)
    ok = np.isfinite(dft) & np.isfinite(raw)
    dft, raw, mols = dft[ok], raw[ok], mols[ok]
    pal = palette_for(mols)
    pred = cv_calibrate(raw, dft, mols)
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    panel(ax, np.asarray(dft, float), pred, list(mols), pal, color, label, lims=lims)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}.png"); fig.savefig(f"{FIG}/{fname}.pdf"); plt.close()
    r2 = 1 - np.sum((dft - pred)**2) / np.sum((dft - dft.mean())**2)
    print(f"  {fname}: MAE={np.mean(np.abs(pred-dft)):.2f}  R2={r2:.3f}  n={len(dft)}")


# --- Koopmans (154 conformères) ---
m = pd.read_parquet("data/merged_dataset.parquet")
m = m[(m.solvent == "Acetonitrile") & (~m.molecule.isin(EXCL))].copy()
one(m.IP.values * EH, -m.HOMO.values, m.molecule.values, C_RED_DARK, "\\mathrm{IP}", "parity_koopmans_ip", lims=IP_LIMS)
one(m.EA.values * EH, -m.LUMO.values, m.molecule.values, C_BLUE_DARK, "\\mathrm{EA}", "parity_koopmans_ea", lims=EA_LIMS)

# --- Politzer-Murray (74 conformères) ---
p = pd.read_csv("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/data/politzer_per_conformer.csv")
one(p.IP_kJ.values, p.I_S_min_eV.values, p.molecule_name.values, C_RED_DARK, "\\mathrm{IP}", "parity_politzer_ip", lims=IP_LIMS)
one(p.EA_kJ.values, -p.V_S_min_eV.values, p.molecule_name.values, C_BLUE_DARK, "\\mathrm{EA}", "parity_politzer_ea", lims=EA_LIMS)
