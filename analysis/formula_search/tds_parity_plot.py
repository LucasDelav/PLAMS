#!/usr/bin/env python3
"""Parity plots TΔS — pool (μ, η) × (1 + a · f²_paired)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
PRED_DIR = REPO / "dataset_predict_dg"
MODELS = REPO / "analysis" / "formula_search" / "results" / "tds_final_optuna.json"
OUT = REPO / "analysis" / "formula_search" / "results"

EH = 2625.499638
EXCLUDED = {"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
EPSILON_R = {"Hexane":1.88,"Toluene":2.38,"Chloroform":4.81,"Tetrahydrofuran":7.58,
             "Dichloromethane":8.93,"Acetone":20.7,"Methanol":32.6,"Acetonitrile":35.7,
             "DMSO":46.7,"Water":78.4}
SOLVENT_ORDER = sorted(EPSILON_R, key=EPSILON_R.get)


def load_dual(df):
    rows = []
    for _, row in df.iterrows():
        p = PRED_DIR/row.solvent/row.molecule/f"{row.molecule}_conf{int(row.conformer)}_prediction.json"
        try:
            d = np.asarray(json.load(open(p))["states"]["neutral_sp"]["cdft"]["dual"], dtype=float)
            rows.append({"f2_max": float(d.max()), "f2_min": float(d.min())})
        except Exception:
            rows.append({"f2_max": np.nan, "f2_min": np.nan})
    return pd.DataFrame(rows, index=df.index)


def predict(df, target, model):
    fcol = "f2_max" if target == "TdS_red" else "f2_min"
    return (model["theta_mu"]*df["mu"].values + model["theta_eta"]*df["eta"].values) \
           * (1.0 + model["a_corr"]*df[fcol].values)


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED)].dropna(subset=["TdS_red","TdS_ox","mu","eta"]).reset_index(drop=True)
    fk = load_dual(df)
    df = df.drop(columns=[c for c in fk.columns if c in df.columns])
    df = pd.concat([df, fk], axis=1).dropna().reset_index(drop=True)
    print(f"n={len(df)}")
    models = json.load(open(MODELS))

    palette = plt.get_cmap("tab10").colors
    solv_color = {s: palette[i % 10] for i, s in enumerate(SOLVENT_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    targets = [("TdS_red", r"$T\Delta S$ (red)"),
               ("TdS_ox",  r"$T\Delta S$ (ox)")]
    for ax, (tgt, title) in zip(axes, targets):
        y_true = df[tgt].values * EH
        y_pred = predict(df, tgt, models[tgt])
        for solv in SOLVENT_ORDER:
            mask = df.solvent.values == solv
            if mask.sum() == 0: continue
            ax.scatter(y_true[mask], y_pred[mask], c=[solv_color[solv]], s=18, alpha=0.65,
                       edgecolor="none", label=f"{solv} (ε={EPSILON_R[solv]})")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.0, alpha=0.6)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        ss_res = float(np.sum((y_true - y_pred)**2))
        ss_tot = float(np.sum((y_true - y_true.mean())**2))
        r2 = 1.0 - ss_res/ss_tot
        m = models[tgt]
        ax.set_title(f"{title}  —  MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}\n"
                     f"a_corr={m['a_corr']:+.3f}  ({m['fukui']})", fontsize=11)
        ax.set_xlabel("DFT (kJ/mol)")
        ax.set_ylabel("Modèle (kJ/mol)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(r"$T\Delta S$ — modèle (μ, η) const en ε × (1 + a · f²) "
                 + f"— n={len(df)} conformères, 10 solvants",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT/"tds_parity.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(OUT/"tds_parity.png", bbox_inches="tight", dpi=180)
    print(f"Saved -> {OUT/'tds_parity.pdf'}")
    print(f"Saved -> {OUT/'tds_parity.png'}")


if __name__ == "__main__":
    main()
