#!/usr/bin/env python
"""
Parity plots du MEILLEUR modèle de ML classique (XGBoost, piste holistique
3-SP en mode direct) sur ΔG_red et ΔG_ox, format identique aux figures
multi-solvant de la partie 3 (solvent_artifacts_p3.py / parity_ms_*).

Source : ML_Batt/results/classical/arr_oof_dg.npz, clé xgb_holistic_3sp_direct.
Points coloriés par solvant via viridis ordonné par ε croissant (apolaire en
violet, polaire en jaune), légende externalisée dans solvent_legend.{png,pdf}.

Sorties : rapport/figures/parity_xgb_ox.{png,pdf} et parity_xgb_red.{png,pdf}.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, FIGSIZE_DOUBLE_V, C_RED_DARK, C_BLUE_DARK, C_GRAY)

apply_style()

NPZ = "/home/cav/Documents/Ecole/Fac/M2/ML_Batt/results/classical/arr_oof_dg.npz"
FIGDIR = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"
KEY = "xgb_holistic_3sp_direct"

# ε des dix solvants (mêmes valeurs que solvent_screen_p3.EPS)
EPS = {"Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
       "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93, "Acetone": 20.7,
       "Methanol": 32.6, "Acetonitrile": 35.7, "DMSO": 46.7, "Water": 78.4}

d = np.load(NPZ, allow_pickle=True)
oof = d[KEY]                                # (n, 2) : col 0 = ΔG_red, col 1 = ΔG_ox
y_red_true = d["dG_red_true"].astype(float)
y_red_pred = oof[:, 0]
y_ox_true = d["dG_ox_true"].astype(float)
y_ox_pred = oof[:, 1]
solvents = d["solvent"].astype(str)

# couleur par solvant : viridis ordonné par ε (apolaire = violet, polaire = jaune)
solv_order = sorted(EPS.keys(), key=lambda s: EPS[s])
solv_color = {s: plt.cm.viridis(i / (len(solv_order) - 1))
              for i, s in enumerate(solv_order)}
pt_colors = [solv_color[s] for s in solvents]


def make_panel(ydft, ypred, color, xlab, ylab, fname, lims=None):
    """Patron identique à solvent_artifacts_p3.make_panel."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    scat = dict(SCATTER_KW); scat.pop("c", None)
    ax.scatter(ydft, ypred, c=pt_colors, **scat)
    regression_with_band(ax, ydft, ypred, color=color)
    if lims is None:
        a = np.concatenate([ydft, ypred]); mg = (a.max() - a.min()) * 0.12
        lims = [a.min() - mg, a.max() + mg]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2 = 1 - np.sum((ydft - ypred) ** 2) / np.sum((ydft - ydft.mean()) ** 2)
    mae = float(np.mean(np.abs(ypred - ydft)))
    stats_text(ax, r2, mae, len(ydft))
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); setup_grid(ax)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/{fname}.png", dpi=150)
    plt.savefig(f"{FIGDIR}/{fname}.pdf")
    plt.close()
    print(f"  {fname}: MAE={mae:.2f} R²={r2:.3f}")


print("=== parity XGBoost holistique 3-SP direct (style maison) ===")
make_panel(y_ox_true, y_ox_pred, C_RED_DARK,
           r"$\Delta G^\circ_{\mathrm{ox}}$ DFT (kJ/mol)",
           r"$\Delta G^\circ_{\mathrm{ox}}$ prédit (kJ/mol)",
           "parity_xgb_ox")
make_panel(y_red_true, y_red_pred, C_BLUE_DARK,
           r"$\Delta G^\circ_{\mathrm{red}}$ DFT (kJ/mol)",
           r"$\Delta G^\circ_{\mathrm{red}}$ prédit (kJ/mol)",
           "parity_xgb_red")
