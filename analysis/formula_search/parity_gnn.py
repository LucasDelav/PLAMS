#!/usr/bin/env python
"""
Parity plots des trois GNN équivariants (MACE rich, MACE minimal, DimeNet++)
sur ΔG_red et ΔG_ox, FORMAT IDENTIQUE aux figures multi-solvant de la P3
(parity_classical_ml_best.py, solvent_artifacts_p3.py) :
  - panneau unique par cible (figsize=FIGSIZE_DOUBLE_V)
  - points coloriés par solvant via viridis ordonné par ε croissant
  - helpers style.py (regression_with_band, stats_text, setup_grid)
  - légende solvant externalisée (solvent_legend.{png,pdf}, déjà existante)

Source : ML_Batt/checkpoints_aligned_{rich,minimal,dimenet}/oof_predictions.npz
Sorties : rapport/figures/parity_{mace_rich,mace_minimal,dimenet}_{ox,red}.{png,pdf}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, FIGSIZE_DOUBLE_V, C_RED_DARK, C_BLUE_DARK, C_GRAY)

apply_style()

ROOT = Path.home() / "Documents/Ecole/Fac/M2/ML_Batt"
META = ROOT / "data/aligned/dataset_index_meta.csv"
FIGDIR = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"

EPS = {"Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
       "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93, "Acetone": 20.7,
       "Methanol": 32.6, "Acetonitrile": 35.7, "DMSO": 46.7, "Water": 78.4}

solv_order = sorted(EPS.keys(), key=lambda s: EPS[s])
solv_color = {s: plt.cm.viridis(i / (len(solv_order) - 1))
              for i, s in enumerate(solv_order)}

# Ordre canonique du dataset PyG (sorted solvent_dir / sorted mol / sorted conf)
meta = pd.read_csv(META)
solvents_all = meta["solv"].to_numpy()

MODELS = {
    "mace_rich":    ("aligned_rich",    "MACE \\emph{rich}"),
    "mace_minimal": ("aligned_minimal", "MACE \\emph{minimal}"),
    "dimenet":      ("aligned_dimenet", "DimeNet++"),
}


def make_panel(ydft, ypred, color, xlab, ylab, fname, lims=None):
    """Patron identique à parity_classical_ml_best.make_panel."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    pt_colors = [solv_color[s] for s in solvents_all]
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


print("=== parity GNN équivariants (style maison) ===")
for slug, (run_tag, label) in MODELS.items():
    npz = ROOT / f"checkpoints_{run_tag}/oof_predictions.npz"
    if not npz.exists():
        print(f"  [skip] {npz} manquant")
        continue
    d = np.load(npz, allow_pickle=True)
    pred = d["pred"]
    target = d["target"]
    filled = d["filled"].astype(bool)
    # colonnes : 0..9 composantes + 10 dG_red + 11 dG_ox
    y_red_true = target[filled, 10]; y_red_pred = pred[filled, 10]
    y_ox_true  = target[filled, 11]; y_ox_pred  = pred[filled, 11]
    make_panel(y_ox_true, y_ox_pred, C_RED_DARK,
               r"$\Delta G^\circ_{\mathrm{ox}}$ DFT (kJ/mol)",
               r"$\Delta G^\circ_{\mathrm{ox}}$ prédit (kJ/mol)",
               f"parity_{slug}_ox")
    make_panel(y_red_true, y_red_pred, C_BLUE_DARK,
               r"$\Delta G^\circ_{\mathrm{red}}$ DFT (kJ/mol)",
               r"$\Delta G^\circ_{\mathrm{red}}$ prédit (kJ/mol)",
               f"parity_{slug}_red")
