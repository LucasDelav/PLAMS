#!/usr/bin/env python3
"""Étape 2 (pivot A) — chaque coefficient θ_k devient une fonction de ε.

Architecture :
    y = Σ_k θ_k(ε) · m_k(d)

avec θ_k(ε) développé sur la base de fonctions ε :
    θ_k(ε) = α_k · 1 + β_k · ε + γ_k · log(ε) + δ_k · (1/ε)
                     ↑ optionnellement Padé[1/1] non-linéaire

C'est la même architecture que le rapport `α(ε) + β(ε)·f₁ + γ(ε)·f₂`
sauf qu'on fait V1 (linéaire 4 features) ou V2 (Padé[1/1] non-linéaire).

Les monômes m_k sont fixés depuis step1_final_forms.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
EH_TO_KJ = 2625.499638
N_FOLDS = 5

EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}

PAIRED_MAP = {
    "lvc2_paired":        ("lvc2_red",     "lvc2_ox"),
    "fukui_paired_max":   ("fplus_max",    "fminus_max"),
    "fukui_paired_max_O": ("fplus_max_O",  "fminus_max_O"),
}


def col(df, name, side):
    if name in PAIRED_MAP:
        rc, oc = PAIRED_MAP[name]
        return df[rc if side == "red" else oc].to_numpy()
    return df[name].to_numpy()


def design_monomials(df, monomials, side):
    n = len(df)
    cols = []
    for m in monomials:
        v = np.ones(n)
        for k, e in m.items():
            v = v * col(df, k, side) ** e
        cols.append(v)
    return np.column_stack(cols)


def expand_with_eps_features(M, eps, features):
    """Given M (n, n_mono) and eps (n,), return F (n, n_mono * n_feat).

    features ⊂ {'const', 'eps', 'inv_eps', 'log_eps', 'eps2'}.
    """
    feat_vals = []
    for f in features:
        if f == "const":
            feat_vals.append(np.ones_like(eps))
        elif f == "eps":
            feat_vals.append(eps)
        elif f == "inv_eps":
            feat_vals.append(1.0 / eps)
        elif f == "log_eps":
            feat_vals.append(np.log(eps))
        elif f == "eps2":
            feat_vals.append(eps ** 2)
        else:
            raise ValueError(f"unknown feature: {f}")
    F = np.column_stack([M * fv[:, None] for fv in feat_vals])  # (n, n_mono * n_feat)
    return F


def metrics_kj(yt_ha, yp_ha):
    mask = np.isfinite(yp_ha)
    y = yt_ha[mask] * EH_TO_KJ
    yp = yp_ha[mask] * EH_TO_KJ
    err = y - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def cv_linear(F_red, F_ox, y_r, y_o, groups):
    n = len(y_r)
    pr = np.full(n, np.nan)
    po = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(F_red, y_r, groups):
        m_r = LinearRegression(fit_intercept=False).fit(F_red[tr], y_r[tr])
        m_o = LinearRegression(fit_intercept=False).fit(F_ox[tr], y_o[tr])
        pr[te] = m_r.predict(F_red[te])
        po[te] = m_o.predict(F_ox[te])
    return pr, po


def fit_full_train(F_red, F_ox, y_r, y_o):
    """Train-only fit (no CV) — to mirror the report's reporting style."""
    m_r = LinearRegression(fit_intercept=False).fit(F_red, y_r)
    m_o = LinearRegression(fit_intercept=False).fit(F_ox, y_o)
    return m_r.predict(F_red), m_o.predict(F_ox)


# Candidate ε-feature sets (tested per pair)
FEATURE_SETS = {
    "{1}":                ["const"],                                # baseline étape 1
    "{1, log ε}":         ["const", "log_eps"],
    "{1, ε}":             ["const", "eps"],
    "{1, 1/ε}":           ["const", "inv_eps"],
    "{1, ε, 1/ε}":        ["const", "eps", "inv_eps"],
    "{1, log ε, 1/ε}":    ["const", "log_eps", "inv_eps"],
    "{1, ε, ε²}":         ["const", "eps", "eps2"],
    "{1, ε, log ε, 1/ε}": ["const", "eps", "log_eps", "inv_eps"],
}


def main():
    df = pd.read_parquet(DATA).sort_values(
        ["solvent", "molecule", "conformer"]).reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    eps = df["eps"].to_numpy()
    groups = df["molecule"].to_numpy()
    print(f"dataset: {len(df)} rows, {df.molecule.nunique()} mols, "
          f"{df.solvent.nunique()} solvents")

    with open(RESULTS / "step1_final_forms.json") as f:
        step1 = json.load(f)

    pair_target = {
        "EA_IP":  ("EA", "IP"),
        "Erelax": ("Erelax_red", "Erelax_ox"),
        "ddU":    ("ddU_red", "ddU_ox"),
        "TdS":    ("TdS_red", "TdS_ox"),
    }

    summary_rows = []
    pred_by_target = {}
    true_by_target = {}
    best_per_pair = {}

    for pair_name, (name_r, name_o) in pair_target.items():
        info = step1[pair_name]
        monos = info["monomials"]
        n_mono = len(monos)
        M_red = design_monomials(df, monos, "red")
        M_ox = design_monomials(df, monos, "ox")
        y_r = df[name_r].to_numpy()
        y_o = df[name_o].to_numpy()

        sigma_r_kj = y_r.std() * EH_TO_KJ
        sigma_o_kj = y_o.std() * EH_TO_KJ
        print(f"\n=== {pair_name} === ({n_mono} monômes) "
              f"σ_red={sigma_r_kj:.1f} σ_ox={sigma_o_kj:.1f} kJ/mol")
        print(f"{'feature_set':25s} {'n_par':>6s} | {'MAE_r_cv':>8s} {'MAE_o_cv':>8s} {'R²_r':>7s} {'R²_o':>7s} | {'MAE_r_tr':>8s} {'MAE_o_tr':>8s}")

        results_for_pair = {}
        for feat_name, feats in FEATURE_SETS.items():
            F_red = expand_with_eps_features(M_red, eps, feats)
            F_ox = expand_with_eps_features(M_ox, eps, feats)
            n_par = F_red.shape[1] * 2  # red + ox

            pr_cv, po_cv = cv_linear(F_red, F_ox, y_r, y_o, groups)
            pr_tr, po_tr = fit_full_train(F_red, F_ox, y_r, y_o)

            mae_r_cv, _, r2_r_cv = metrics_kj(y_r, pr_cv)
            mae_o_cv, _, r2_o_cv = metrics_kj(y_o, po_cv)
            mae_r_tr, _, _ = metrics_kj(y_r, pr_tr)
            mae_o_tr, _, _ = metrics_kj(y_o, po_tr)

            results_for_pair[feat_name] = {
                "n_par": n_par,
                "mae_r_cv": mae_r_cv, "mae_o_cv": mae_o_cv,
                "r2_r_cv": r2_r_cv, "r2_o_cv": r2_o_cv,
                "mae_r_tr": mae_r_tr, "mae_o_tr": mae_o_tr,
                "pred_r_cv": pr_cv, "pred_o_cv": po_cv,
                "mae_total_cv": mae_r_cv + mae_o_cv,
            }
            print(f"{feat_name:25s} {n_par:6d} | {mae_r_cv:8.2f} {mae_o_cv:8.2f} "
                  f"{r2_r_cv:7.3f} {r2_o_cv:7.3f} | {mae_r_tr:8.2f} {mae_o_tr:8.2f}")

        best = min(results_for_pair, key=lambda k: results_for_pair[k]["mae_total_cv"])
        print(f"  -> meilleur jeu de features: {best}")
        best_per_pair[pair_name] = best

        info_best = results_for_pair[best]
        pred_by_target[name_r] = info_best["pred_r_cv"]
        pred_by_target[name_o] = info_best["pred_o_cv"]
        true_by_target[name_r] = y_r
        true_by_target[name_o] = y_o

        summary_rows.append({
            "pair": pair_name, "target": name_r, "feat_set": best,
            "n_par": info_best["n_par"],
            "MAE_kJ_cv": info_best["mae_r_cv"], "R2_cv": info_best["r2_r_cv"],
            "MAE_kJ_train": info_best["mae_r_tr"],
        })
        summary_rows.append({
            "pair": pair_name, "target": name_o, "feat_set": best,
            "n_par": info_best["n_par"],
            "MAE_kJ_cv": info_best["mae_o_cv"], "R2_cv": info_best["r2_o_cv"],
            "MAE_kJ_train": info_best["mae_o_tr"],
        })

    # Combined ΔG (CV pred)
    dg_red_t = (-true_by_target["EA"] + true_by_target["Erelax_red"]
                + true_by_target["ddU_red"] + true_by_target["TdS_red"])
    dg_red_p = (-pred_by_target["EA"] + pred_by_target["Erelax_red"]
                + pred_by_target["ddU_red"] + pred_by_target["TdS_red"])
    dg_ox_t = (+true_by_target["IP"] + true_by_target["Erelax_ox"]
               + true_by_target["ddU_ox"] + true_by_target["TdS_ox"])
    dg_ox_p = (+pred_by_target["IP"] + pred_by_target["Erelax_ox"]
               + pred_by_target["ddU_ox"] + pred_by_target["TdS_ox"])
    for nm, t, p in [("ΔG_red", dg_red_t, dg_red_p),
                     ("ΔG_ox", dg_ox_t, dg_ox_p)]:
        mae, _, r2 = metrics_kj(t, p)
        summary_rows.append({"pair": nm, "target": nm, "feat_set": "(combined)",
                             "n_par": np.nan, "MAE_kJ_cv": mae, "R2_cv": r2,
                             "MAE_kJ_train": np.nan})

    sdf = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("RÉSUMÉ ÉTAPE 2A (multi-solvant, OOF 5-fold GroupKFold, kJ/mol)")
    print("=" * 90)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sdf.to_csv(RESULTS / "step2A_metrics_kJ.csv", index=False)
    print(f"\n[saved] step2A_metrics_kJ.csv")

    with open(RESULTS / "step2A_best_feature_sets.json", "w") as f:
        json.dump(best_per_pair, f, indent=2)
    print(f"[saved] step2A_best_feature_sets.json")

    np.savez(RESULTS / "step2A_predictions.npz",
             solvent=df["solvent"].to_numpy(),
             molecule=df["molecule"].to_numpy(),
             **{f"true_{k}": v for k, v in true_by_target.items()},
             **{f"pred_{k}": v for k, v in pred_by_target.items()},
             true_dG_red=dg_red_t, pred_dG_red=dg_red_p,
             true_dG_ox=dg_ox_t, pred_dG_ox=dg_ox_p)
    print(f"[saved] step2A_predictions.npz")


if __name__ == "__main__":
    main()
