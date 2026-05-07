#!/usr/bin/env python3
"""Étape 1 — parity plots et MAE/RMSE en kJ/mol pour les 4 formes retenues.

Pour chaque paire :
  - reconstruit la forme top-BIC choisie (cf. step1_top30_*.csv)
  - prédiction OOF via 5-fold GroupKFold (groupes = molécule)
  - convertit MAE/RMSE Hartree -> kJ/mol
  - trace 8 parity plots (un par cible) + 2 plots dérivés (ΔG_red, ΔG_ox)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EH_TO_KJ = 2625.499638
N_FOLDS = 5
SOLVENT = "Acetonitrile"

PAIRED_MAP = {
    "lvc2_paired":        ("lvc2_red",     "lvc2_ox"),
    "fukui_paired_max":   ("fplus_max",    "fminus_max"),
    "fukui_paired_max_O": ("fplus_max_O",  "fminus_max_O"),
}

# Forme top-BIC retenue par paire (étape 1.3)
FORMS = {
    "EA_IP": {
        "pair": ("EA", "IP"),
        "label": "1 + γ + φ + μ + γ² + η·φ + η² + μ·γ + μ·φ",
        "monomials": [
            {},
            {"gamma": 1},
            {"fukui_paired_max": 1},
            {"mu": 1},
            {"gamma": 2},
            {"eta": 1, "fukui_paired_max": 1},
            {"eta": 2},
            {"mu": 1, "gamma": 1},
            {"mu": 1, "fukui_paired_max": 1},
        ],
    },
    "Erelax": {
        "pair": ("Erelax_red", "Erelax_ox"),
        "label": "1 + λ + Q + f₂·λ + Q·f₂ + φ·λ + φ·f₂",
        "monomials": [
            {},
            {"lvc2_paired": 1},
            {"qbader_max": 1},
            {"f2_max": 1, "lvc2_paired": 1},
            {"qbader_max": 1, "f2_max": 1},
            {"fukui_paired_max": 1, "lvc2_paired": 1},
            {"fukui_paired_max": 1, "f2_max": 1},
        ],
    },
    "ddU": {
        "pair": ("ddU_red", "ddU_ox"),
        "label": "1 + η + μ + φ² + μ·φ",
        "monomials": [
            {},
            {"eta": 1},
            {"mu": 1},
            {"fukui_paired_max": 2},
            {"mu": 1, "fukui_paired_max": 1},
        ],
    },
    "TdS": {
        "pair": ("TdS_red", "TdS_ox"),
        "label": "1 + φ² + μ²·φ",
        "monomials": [
            {},
            {"fukui_paired_max": 2},
            {"mu": 2, "fukui_paired_max": 1},
        ],
    },
}


def col(df: pd.DataFrame, name: str, side: str) -> np.ndarray:
    if name in PAIRED_MAP:
        red_col, ox_col = PAIRED_MAP[name]
        return df[red_col if side == "red" else ox_col].to_numpy()
    return df[name].to_numpy()


def design(df: pd.DataFrame, monomials: list[dict], side: str) -> np.ndarray:
    n = len(df)
    cols = []
    for m in monomials:
        v = np.ones(n)
        for name, exp in m.items():
            v = v * col(df, name, side) ** exp
        cols.append(v)
    return np.column_stack(cols)


def oof_predict(F: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    n = len(y)
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(F, y, groups):
        m = LinearRegression(fit_intercept=False).fit(F[tr], y[tr])
        pred[te] = m.predict(F[te])
    return pred


def fit_full(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    return LinearRegression(fit_intercept=False).fit(F, y).coef_


def metrics_kj(y_ha: np.ndarray, yhat_ha: np.ndarray) -> dict:
    y = y_ha * EH_TO_KJ
    yhat = yhat_ha * EH_TO_KJ
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae_kj": mae, "rmse_kj": rmse, "r2": r2,
            "y_kj": y, "yhat_kj": yhat}


def main() -> None:
    df_full = pd.read_parquet(DATA)
    df = df_full[df_full.solvent == SOLVENT].reset_index(drop=True)
    groups = df["molecule"].to_numpy()
    print(f"{SOLVENT}: {len(df)} rows, {df.molecule.nunique()} molecules")
    print()

    # OOF predictions for each target (Hartree -> kJ/mol)
    pred_ha: dict[str, np.ndarray] = {}
    true_ha: dict[str, np.ndarray] = {}
    summary_rows = []
    coefs_export = {}

    for pair_name, info in FORMS.items():
        name_red, name_ox = info["pair"]
        F_red = design(df, info["monomials"], "red")
        F_ox = design(df, info["monomials"], "ox")
        y_red = df[name_red].to_numpy()
        y_ox = df[name_ox].to_numpy()

        pred_red = oof_predict(F_red, y_red, groups)
        pred_ox = oof_predict(F_ox, y_ox, groups)

        m_red = metrics_kj(y_red, pred_red)
        m_ox = metrics_kj(y_ox, pred_ox)

        pred_ha[name_red], true_ha[name_red] = pred_red, y_red
        pred_ha[name_ox],  true_ha[name_ox]  = pred_ox,  y_ox

        coefs_export[pair_name] = {
            "label": info["label"],
            "monomials": info["monomials"],
            "theta_red": fit_full(F_red, y_red).tolist(),
            "theta_ox":  fit_full(F_ox,  y_ox).tolist(),
            "n_params": len(info["monomials"]),
            "mae_red_kj": m_red["mae_kj"], "mae_ox_kj": m_ox["mae_kj"],
            "rmse_red_kj": m_red["rmse_kj"], "rmse_ox_kj": m_ox["rmse_kj"],
            "r2_red": m_red["r2"], "r2_ox": m_ox["r2"],
        }

        summary_rows.append({
            "pair": pair_name, "target": name_red,
            "MAE_kJ/mol": m_red["mae_kj"], "RMSE_kJ/mol": m_red["rmse_kj"], "R2": m_red["r2"],
        })
        summary_rows.append({
            "pair": pair_name, "target": name_ox,
            "MAE_kJ/mol": m_ox["mae_kj"], "RMSE_kJ/mol": m_ox["rmse_kj"], "R2": m_ox["r2"],
        })

    # ΔG dérivés (sign convention from CLAUDE.md):
    #   ΔG_red = -EA + Erelax_red + ddU_red + TdS_red
    #   ΔG_ox  = +IP + Erelax_ox + ddU_ox + TdS_ox
    dg_red_true = -true_ha["EA"] + true_ha["Erelax_red"] + true_ha["ddU_red"] + true_ha["TdS_red"]
    dg_red_pred = -pred_ha["EA"] + pred_ha["Erelax_red"] + pred_ha["ddU_red"] + pred_ha["TdS_red"]
    dg_ox_true = +true_ha["IP"] + true_ha["Erelax_ox"] + true_ha["ddU_ox"] + true_ha["TdS_ox"]
    dg_ox_pred = +pred_ha["IP"] + pred_ha["Erelax_ox"] + pred_ha["ddU_ox"] + pred_ha["TdS_ox"]

    m_dg_red = metrics_kj(dg_red_true, dg_red_pred)
    m_dg_ox = metrics_kj(dg_ox_true, dg_ox_pred)
    summary_rows.append({"pair": "ΔG_red", "target": "ΔG_red",
                         "MAE_kJ/mol": m_dg_red["mae_kj"], "RMSE_kJ/mol": m_dg_red["rmse_kj"], "R2": m_dg_red["r2"]})
    summary_rows.append({"pair": "ΔG_ox", "target": "ΔG_ox",
                         "MAE_kJ/mol": m_dg_ox["mae_kj"], "RMSE_kJ/mol": m_dg_ox["rmse_kj"], "R2": m_dg_ox["r2"]})

    # Print and save summary
    sdf = pd.DataFrame(summary_rows)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sdf.to_csv(RESULTS / "step1_metrics_kJ.csv", index=False)
    with open(RESULTS / "step1_final_forms.json", "w") as f:
        json.dump(coefs_export, f, indent=2)
    print(f"\n[saved] step1_metrics_kJ.csv")
    print(f"[saved] step1_final_forms.json")

    # Parity plots (5 columns x 2 rows = 10)
    plots = [
        ("EA",          "EA",          true_ha["EA"],            pred_ha["EA"]),
        ("IP",          "IP",          true_ha["IP"],            pred_ha["IP"]),
        ("Erelax_red",  "E_relax red", true_ha["Erelax_red"],    pred_ha["Erelax_red"]),
        ("Erelax_ox",   "E_relax ox",  true_ha["Erelax_ox"],     pred_ha["Erelax_ox"]),
        ("ddU_red",     "ΔΔU red",     true_ha["ddU_red"],       pred_ha["ddU_red"]),
        ("ddU_ox",      "ΔΔU ox",      true_ha["ddU_ox"],        pred_ha["ddU_ox"]),
        ("TdS_red",     "TΔS red",     true_ha["TdS_red"],       pred_ha["TdS_red"]),
        ("TdS_ox",      "TΔS ox",      true_ha["TdS_ox"],        pred_ha["TdS_ox"]),
        ("dG_red",      "ΔG red",      dg_red_true,              dg_red_pred),
        ("dG_ox",       "ΔG ox",       dg_ox_true,               dg_ox_pred),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, (key, title, yt_ha, yp_ha) in zip(axes.flat, plots):
        m = metrics_kj(yt_ha, yp_ha)
        yt, yp = m["y_kj"], m["yhat_kj"]
        ax.scatter(yt, yp, s=14, alpha=0.7, edgecolor="none")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = (hi - lo) * 0.04
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{title}\nMAE={m['mae_kj']:.2f} kJ/mol  R²={m['r2']:.3f}", fontsize=11)
        ax.set_xlabel("DFT (kJ/mol)")
        ax.set_ylabel("modèle (kJ/mol)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Étape 1 — Parity plots ({SOLVENT}, OOF 5-fold GroupKFold)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = RESULTS / "step1_parity_plots.pdf"
    out_png = RESULTS / "step1_parity_plots.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[saved] {out_pdf.name}")
    print(f"[saved] {out_png.name}")


if __name__ == "__main__":
    main()
