#!/usr/bin/env python3
"""Étape 2 — parity plots multi-solvant (10 solvants, couleur par solvant)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EH_TO_KJ = 2625.499638

SOLVENT_ORDER = [
    "Hexane", "Toluene", "Chloroform", "Tetrahydrofuran", "Dichloromethane",
    "Acetone", "Methanol", "Acetonitrile", "DMSO", "Water",
]


def metrics_kj(yt_ha, yp_ha):
    mask = np.isfinite(yp_ha)
    y = yt_ha[mask] * EH_TO_KJ
    yhat = yp_ha[mask] * EH_TO_KJ
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, r2, y, yhat


def main():
    z = np.load(RESULTS / "step2_predictions.npz", allow_pickle=True)
    solvent = z["solvent"]

    plots = [
        ("EA",          "EA"),
        ("IP",          "IP"),
        ("Erelax_red",  "E_relax red"),
        ("Erelax_ox",   "E_relax ox"),
        ("ddU_red",     "ΔΔU red"),
        ("ddU_ox",      "ΔΔU ox"),
        ("TdS_red",     "TΔS red"),
        ("TdS_ox",      "TΔS ox"),
        ("dG_red",      "ΔG red"),
        ("dG_ox",       "ΔG ox"),
    ]

    cmap = plt.cm.tab10
    color_for = {s: cmap(i) for i, s in enumerate(SOLVENT_ORDER)}

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    for ax, (key, title) in zip(axes.flat, plots):
        yt = z[f"true_{key}"]
        yp = z[f"pred_{key}"]
        mae, r2, y_kj, yp_kj = metrics_kj(yt, yp)
        for s in SOLVENT_ORDER:
            mask = (solvent == s) & np.isfinite(yp)
            if mask.sum() == 0:
                continue
            ax.scatter(yt[mask] * EH_TO_KJ, yp[mask] * EH_TO_KJ,
                       s=14, alpha=0.7, edgecolor="none",
                       color=color_for[s], label=s)
        if len(y_kj) > 0:
            lo, hi = min(y_kj.min(), yp_kj.min()), max(y_kj.max(), yp_kj.max())
            pad = (hi - lo) * 0.04
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{title}\nMAE={mae:.2f} kJ/mol  R²={r2:.3f}", fontsize=11)
        ax.set_xlabel("DFT (kJ/mol)")
        ax.set_ylabel("modèle (kJ/mol)")
        ax.grid(alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Étape 2 — Parity plots multi-solvant (Φ(ε)·f, OOF 5-fold GroupKFold)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_pdf = RESULTS / "step2_parity_plots.pdf"
    out_png = RESULTS / "step2_parity_plots.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
