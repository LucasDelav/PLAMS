#!/usr/bin/env python3
"""Parity plots EA/IP en reproduisant les filtres exacts du rapport.

Filtres rapport (cf. p1125 du rapport) :
  (i)  DEA Br/I + L-DOPA
  (ii) DFT non convergé (déjà NaN-droppés en amont)
  (iii) Conformères Hexane EA<0
  (iv) Sans LVC2 (déjà NaN-droppés)
  (v) |E_relax_LVC2| > 150 kJ/mol (LVC2 pathologique)

Cible attendue rapport : n=1411 conformères, MAE_EA≈3.7, MAE_IP≈5.2 kJ/mol.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
EH_TO_KJ = 2625.499638
EV_TO_KJ = 96.48533212
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
LVC2_PATHO_THRESHOLD_KJ = 150.0  # rapport p1125

EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}
SOLVENT_ORDER = list(EPSILON_R.keys())


def pade21_model(X, *params):
    eps = X[0]
    n_features = X.shape[0] - 1
    out = np.zeros_like(eps, dtype=float)
    for i in range(n_features):
        a0, a1, a2, b1 = params[i * 4: i * 4 + 4]
        out += (a0 + a1 * eps + a2 * eps ** 2) / (1.0 + b1 * eps) * X[1 + i]
    a0, a1, a2, b1 = params[n_features * 4: n_features * 4 + 4]
    out += (a0 + a1 * eps + a2 * eps ** 2) / (1.0 + b1 * eps)
    return out


def fit_pade21(eps, feats, y):
    nf = feats.shape[0]
    X = np.vstack([eps[None, :], feats])
    p0 = np.tile([1.0, 0.01, 0.0001, 0.01], nf + 1)
    lo = np.tile([-np.inf, -np.inf, -np.inf, 0.0], nf + 1)
    hi = np.tile([np.inf, np.inf, np.inf, np.inf], nf + 1)
    best = None
    for trial in [p0, p0 * 0.1, p0 * 10]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(pade21_model, X, y, p0=trial,
                                    bounds=(lo, hi), maxfev=20000, method="trf")
            yhat = pade21_model(X, *popt)
            if not np.all(np.isfinite(yhat)):
                continue
            mae = float(np.mean(np.abs(y - yhat)))
            if best is None or mae < best["mae"]:
                best = {"yhat": yhat, "mae": mae}
        except Exception:
            continue
    return best


def stats(yt, yp):
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def main():
    df = pd.read_parquet(DATA)
    print(f"Dataset initial: {len(df)} rows")
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    print(f"  après exclusion DEA + L-DOPA   : {len(df)}")

    # Filter (iii) Hexane EA<0 — already done globally in build_dataset (EA<0 dropped)
    # but rapport vire spécifiquement Hexane "EA<0" — chez nous EA est strictement >=0
    # (filtré au build_dataset) donc ce filtre n'agit pas ici.

    # Filter (v) LVC2 pathologique : |LVC2| > 150 kJ/mol pour red OU ox
    lvc2_red_kj = df.lvc2_red.to_numpy() * EH_TO_KJ
    lvc2_ox_kj = df.lvc2_ox.to_numpy() * EH_TO_KJ
    mask_patho = (np.abs(lvc2_red_kj) > LVC2_PATHO_THRESHOLD_KJ) | \
                 (np.abs(lvc2_ox_kj) > LVC2_PATHO_THRESHOLD_KJ)
    n_patho = int(mask_patho.sum())
    df = df.loc[~mask_patho].reset_index(drop=True)
    print(f"  après LVC2 pathologique (-{n_patho}): {len(df)}")
    print(f"\nObjectif rapport : 1411 conformères")
    print(f"σ(EA) = {df.EA.std() * EH_TO_KJ:.2f} kJ/mol")
    print(f"σ(IP) = {df.IP.std() * EH_TO_KJ:.2f} kJ/mol")

    eps = df["solvent"].map(EPSILON_R).to_numpy()
    mu_eV = df.mu.to_numpy()
    eta_eV = df.eta.to_numpy()
    HOMO_eV = mu_eV - eta_eV / 2.0
    LUMO_eV = mu_eV + eta_eV / 2.0
    IP_koop = -HOMO_eV * EV_TO_KJ
    EA_koop = -LUMO_eV * EV_TO_KJ
    EA = df.EA.to_numpy() * EH_TO_KJ
    IP = df.IP.to_numpy() * EH_TO_KJ
    feats = np.array([IP_koop, EA_koop])

    res_EA = fit_pade21(eps, feats, EA)
    res_IP = fit_pade21(eps, feats, IP)
    mae_ea, rmse_ea, r2_ea = stats(EA, res_EA["yhat"])
    mae_ip, rmse_ip, r2_ip = stats(IP, res_IP["yhat"])
    print(f"\nEA : MAE={mae_ea:.2f}  RMSE={rmse_ea:.2f}  R²={r2_ea:.4f}")
    print(f"IP : MAE={mae_ip:.2f}  RMSE={rmse_ip:.2f}  R²={r2_ip:.4f}")
    print(f"Rapport : EA MAE=3.7, IP MAE=5.2")

    # Plot
    cmap = plt.cm.tab10
    color_for = {s: cmap(i) for i, s in enumerate(SOLVENT_ORDER)}
    solvent = df["solvent"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, target_name, yt, yp, m, r, r2 in [
        (axes[0], "EA", EA, res_EA["yhat"], mae_ea, rmse_ea, r2_ea),
        (axes[1], "IP", IP, res_IP["yhat"], mae_ip, rmse_ip, r2_ip),
    ]:
        for s in SOLVENT_ORDER:
            mask = solvent == s
            if not mask.any():
                continue
            ax.scatter(yt[mask], yp[mask], s=22, alpha=0.75, edgecolor="none",
                       color=color_for[s],
                       label=f"{s} (ε={EPSILON_R[s]:.1f})")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = (hi - lo) * 0.04
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{target_name}\nMAE = {m:.2f} kJ/mol   RMSE = {r:.2f}   "
                     f"R² = {r2:.4f}   n = {len(yt)}", fontsize=12)
        ax.set_xlabel("DFT (kJ/mol)", fontsize=11)
        ax.set_ylabel("modèle Padé[2/1] (kJ/mol)", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Parity plots EA / IP — filtres rapport (DEA + L-DOPA + |LVC2|≤150 kJ/mol)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_pdf = RESULTS / "step8b_parity_EA_IP_rapport.pdf"
    out_png = RESULTS / "step8b_parity_EA_IP_rapport.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
