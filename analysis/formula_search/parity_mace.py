#!/usr/bin/env python
"""
Parity plot d'une piste MACE (rich / minimal) sur ΔG_red et ΔG_ox.

Lit l'OOF nested-CV (5 folds GroupKFold-by-molecule, mêmes splits que la grille
classique) sauvegardé par scripts/train.py au format :
  results/<run_tag>/oof_predictions.npz   -> pred[N,12], target[N,12], filled[N]

Convention rapport : ox=ROUGE à gauche, red=BLEU à droite (cf. mémoire).
Points coloriés par SOLVANT (palette HSV muette). Mêmes choix que
parity_classical_ml_best.py pour comparaison visuelle directe.

Usage :
  python parity_mace.py <run_tag> [<fname_out>]
exemples :
  python parity_mace.py aligned_rich     # -> rapport/figures/parity_mace_rich.{pdf,png}
  python parity_mace.py aligned_minimal  # -> rapport/figures/parity_mace_minimal.{pdf,png}
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, C_RED_DARK, C_BLUE_DARK)

apply_style()

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)
RUN = sys.argv[1]
DEFAULT_TAG = {"aligned_rich": "rich", "aligned_minimal": "minimal"}.get(RUN, RUN)
FNAME = sys.argv[2] if len(sys.argv) >= 3 else f"parity_mace_{DEFAULT_TAG}"

ROOT = Path.home() / "Documents/Ecole/Fac/M2/ML_Batt"
# Le run_tag « aligned_rich » -> dossier checkpoints_aligned_rich/
NPZ = ROOT / f"checkpoints_{RUN}/oof_predictions.npz"
META = ROOT / "data/aligned/dataset_index_meta.csv"
FIGDIR = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"
MODEL_NAME = {"aligned_rich": "MACE (rich)", "aligned_minimal": "MACE (minimal)",
              "aligned_dimenet": "DimeNet++"}.get(RUN, RUN)

if not NPZ.exists():
    print(f"OOF file manquant : {NPZ}")
    sys.exit(2)

d = np.load(NPZ, allow_pickle=True)
pred = d["pred"]            # (N, 12) : 8 composantes + 2 résidus + dG_red + dG_ox
target = d["target"]
filled = d["filled"]
mask = filled.astype(bool)
y_red_true = target[mask, 10]
y_red_pred = pred[mask, 10]
y_ox_true = target[mask, 11]
y_ox_pred = pred[mask, 11]
N = mask.sum()
print(f"OOF couvert : {N}/{len(filled)}  ({100*N/len(filled):.0f}%)")

# Métadonnées : CSV exporté côté CRIANN après instanciation du dataset PyG
# (ordre canonique du RedoxDataset : sorted(solvent_dir)/sorted(mol)/sorted(conf)).
meta = pd.read_csv(META)
assert len(meta) == len(filled), f"Désynchro : meta={len(meta)} vs OOF={len(filled)}"
solvents = meta.loc[mask, "solv"].to_numpy()


def _muted(rgba):
    h, s, v = mcolors.rgb_to_hsv(np.array(rgba[:3]))
    return (*mcolors.hsv_to_rgb((h, s * 0.45, min(v, 0.78))), 1.0)


u_solv = sorted(set(solvents))
_cmap = plt.cm.gist_rainbow(np.linspace(0.02, 0.98, len(u_solv)))
PAL = {s: _muted(_cmap[i]) for i, s in enumerate(u_solv)}
colors = [PAL[s] for s in solvents]


def panel(ax, y_true, y_pred, regr_color, axis_label):
    sk = {k: v for k, v in SCATTER_KW.items() if k != "c"}
    ax.scatter(y_true, y_pred, c=colors, **sk)
    _, _, r2, _mae_regr, n = regression_with_band(ax, y_true, y_pred, color=regr_color)
    # MAE de PARITÉ (vs y=x), pas la MAE de régression — c'est la vraie erreur prédictive.
    mae = float(np.mean(np.abs(y_pred - y_true)))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
            color="#636363", alpha=0.5, lw=0.8, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"{axis_label} DFT (kJ/mol)")
    ax.set_ylabel(f"{axis_label} {MODEL_NAME} (kJ/mol)")
    setup_grid(ax)
    stats_text(ax, r2, mae, n)


fig, (ax_ox, ax_red) = plt.subplots(1, 2, figsize=(7.2, 4.4))
panel(ax_ox,  y_ox_true,  y_ox_pred,  C_RED_DARK,  r"$\Delta G_{\mathrm{ox}}$")
panel(ax_red, y_red_true, y_red_pred, C_BLUE_DARK, r"$\Delta G_{\mathrm{red}}$")
ax_ox.set_title("Oxydation")
ax_red.set_title("Réduction")

handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markeredgecolor="white", markeredgewidth=0.4,
                      markerfacecolor=PAL[s], label=s) for s in u_solv]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=11)
plt.tight_layout(rect=[0, 0.12, 1, 1])

out_pdf = f"{FIGDIR}/{FNAME}.pdf"
out_png = f"{FIGDIR}/{FNAME}.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight")
print(f"figure : {out_pdf}")
print(f"figure : {out_png}")

# Recap chiffres
from scipy.stats import linregress
for tag, yt, yp in [("ΔG_ox", y_ox_true, y_ox_pred),
                    ("ΔG_red", y_red_true, y_red_pred)]:
    s, i, r, p, _ = linregress(yt, yp)
    mae = np.mean(np.abs(yp - yt))
    print(f"  {tag}: MAE={mae:.2f}  R²={r**2:.3f}  pente={s:.3f}  ord.={i:.2f}")
