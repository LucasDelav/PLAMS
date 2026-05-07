#!/usr/bin/env python3
"""Parity plots EA et IP (modèle Padé[2/1] reproduisant le rapport, step4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SOLVENT_ORDER = [
    "Hexane", "Toluene", "Chloroform", "Tetrahydrofuran", "Dichloromethane",
    "Acetone", "Methanol", "Acetonitrile", "DMSO", "Water",
]
SOLVENT_EPS = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}


def stats(yt, yp):
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def main():
    z = np.load(RESULTS / "step4_reproduce_predictions.npz", allow_pickle=True)
    solvent = z["solvent"]
    cmap = plt.cm.tab10
    color_for = {s: cmap(i) for i, s in enumerate(SOLVENT_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, target, title in [(axes[0], "EA", "EA"), (axes[1], "IP", "IP")]:
        yt = z[f"true_{target}"]
        yp = z[f"pred_{target}"]
        mae, rmse, r2 = stats(yt, yp)

        for s in SOLVENT_ORDER:
            mask = solvent == s
            if not mask.any():
                continue
            ax.scatter(yt[mask], yp[mask], s=22, alpha=0.75, edgecolor="none",
                       color=color_for[s],
                       label=f"{s} (ε={SOLVENT_EPS[s]:.1f})")

        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = (hi - lo) * 0.04
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{title}\nMAE = {mae:.2f} kJ/mol   RMSE = {rmse:.2f}   "
                     f"R² = {r2:.4f}   n = {len(yt)}", fontsize=12)
        ax.set_xlabel("DFT (kJ/mol)", fontsize=11)
        ax.set_ylabel("modèle Padé[2/1] (kJ/mol)", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Parity plots EA et IP — Padé[2/1](ε) sur (HOMO, LUMO) ≡ (μ, η)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_pdf = RESULTS / "step8_parity_EA_IP.pdf"
    out_png = RESULTS / "step8_parity_EA_IP.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
