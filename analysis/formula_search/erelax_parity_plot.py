#!/usr/bin/env python3
"""Parity plots for E_relax (red, ox) — final model with Fukui correction.

Final model (per target, joint refit):
  y = (theta_mu*mu + theta_eta*eta + theta_mu2*mu^2 + theta_lvc*LVC2_paired)
        * (1 + a_corr * fukui_paired)

with fukui_paired = f+_max for Erelax_red, f-_max for Erelax_ox (both keep O).

Coefficients are loaded from results/erelax_best_fukui_models.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
PRED_DIR = REPO / "dataset_predict_dg"
MODELS = REPO / "analysis" / "formula_search" / "results" / "erelax_best_fukui_models.json"
OUT = REPO / "analysis" / "formula_search" / "results"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
DESCRIPTORS = ("mu", "eta", "mu2")
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
SOLVENT_ORDER = sorted(EPSILON_R, key=EPSILON_R.get)


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def load_fukui(df):
    rows = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        try:
            pred = json.load(open(path))
            ns = pred["states"]["neutral_sp"]
            cdft = ns["cdft"]
            syms = np.asarray(parse_xyz_symbols(ns["xyz"]))
            fp = np.asarray(cdft["fplus"], dtype=float)
            fm = np.asarray(cdft["fminus"], dtype=float)
            rows.append({"fplus_max": float(fp.max()), "fminus_max": float(fm.max())})
        except Exception:
            rows.append({"fplus_max": np.nan, "fminus_max": np.nan})
    return pd.DataFrame(rows, index=df.index)


def predict(df, target, model):
    suff = target.split("_")[1]
    cols = list(DESCRIPTORS) + [f"lvc2_{suff}"]
    X = df[cols].values.astype(float)
    theta = np.asarray(model["theta"])
    fcol = "fplus_max" if suff == "red" else "fminus_max"
    f = df[fcol].values
    return (X @ theta) * (1.0 + model["a_corr"] * f)


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].dropna(
        subset=["Erelax_red", "Erelax_ox", "mu", "eta", "lvc2_red", "lvc2_ox"]
    ).reset_index(drop=True)
    df["mu2"] = df["mu"] ** 2
    fk = load_fukui(df)
    df = df.drop(columns=[c for c in fk.columns if c in df.columns])
    df = pd.concat([df, fk], axis=1).dropna(subset=["fplus_max", "fminus_max"]).reset_index(drop=True)
    print(f"Plotting on n={len(df)} conformers, {df.molecule.nunique()} molecules,"
          f" {df.solvent.nunique()} solvents")

    models = json.load(open(MODELS))

    # Generate predictions
    preds = {}
    for tgt in ("Erelax_red", "Erelax_ox"):
        y_true = df[tgt].values * EH_TO_KJ
        y_pred = predict(df, tgt, models[tgt])
        preds[tgt] = (y_true, y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        print(f"  {tgt}: MAE={mae:.3f} RMSE={rmse:.3f} R²={r2:.3f}")

    # Plot
    palette = plt.get_cmap("tab10").colors
    solv_color = {s: palette[i % 10] for i, s in enumerate(SOLVENT_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    targets = [("Erelax_red", r"$E_{\rm relax}$ (red)"),
               ("Erelax_ox",  r"$E_{\rm relax}$ (ox)")]
    for ax, (tgt, title) in zip(axes, targets):
        y_true, y_pred = preds[tgt]
        for solv in SOLVENT_ORDER:
            mask = df.solvent.values == solv
            if mask.sum() == 0:
                continue
            ax.scatter(y_true[mask], y_pred[mask],
                       c=[solv_color[solv]], s=18, alpha=0.65,
                       edgecolor="none", label=f"{solv} (ε={EPSILON_R[solv]})")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.0, alpha=0.6)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        a = models[tgt]["a_corr"]
        ax.set_title(f"{title}  —  MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}\n"
                     f"a_corr={a:+.3f}", fontsize=11)
        ax.set_xlabel("DFT (kJ/mol)")
        ax.set_ylabel("Modèle (kJ/mol)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(r"$E_{\rm relax}$ — modèle linéaire (μ, η, μ², LVC2$_{\rm paired}$) "
                 r"× (1 + a · Fukui) — n=" + str(len(df)) + " conformères, 10 solvants",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    out_pdf = OUT / "erelax_parity.pdf"
    out_png = OUT / "erelax_parity.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"\nSaved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
