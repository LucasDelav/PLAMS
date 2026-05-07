#!/usr/bin/env python3
"""Test d'unification CDFT — prédire les 8 cibles avec UNIQUEMENT (μ, η).

Architecture identique à step4 : Y = α(ε)·μ + β(ε)·η + γ(ε), Padé[2/1](ε).
12 paramètres. Targets et filtres comme generate_section4.py.

Si la bijection (HOMO, LUMO) ↔ (μ, η) est complète, alors :
  - EA = -LUMO·EV_TO_KJ = -(μ + η/2)·EV_TO_KJ
  - IP = -HOMO·EV_TO_KJ = -(μ - η/2)·EV_TO_KJ
on retrouve (μ, η) comme une base linéaire équivalente.

Pour comparer, on inclut aussi (μ, η, ω = μ²/2η) pour évaluer si l'ajout du
descripteur dérivé ω apporte un gain marginal (en théorie redondant avec μ et η,
mais utilisable comme non-linéarité injectée).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
EH_TO_KJ = 2625.499638
EV_TO_KJ = 96.48533212
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}
FAMILY_NPP = {"pade_21": 4, "constant": 1}


def eval_coeff(family, eps, p):
    if family == "constant":
        return np.full_like(eps, p[0], dtype=float)
    if family == "pade_21":
        return (p[0] + p[1] * eps + p[2] * eps ** 2) / (1.0 + p[3] * eps)
    raise ValueError(family)


def make_model(n_features, family):
    npp = FAMILY_NPP[family]

    def model(X, *params):
        eps = X[0]
        out = np.zeros_like(eps, dtype=float)
        for i in range(n_features):
            off = i * npp
            out += eval_coeff(family, eps, params[off:off + npp]) * X[1 + i]
        off = n_features * npp
        out += eval_coeff(family, eps, params[off:off + npp])
        return out

    return model, (n_features + 1) * npp


def initial_guess(n_features, family):
    p0, lo, hi = [], [], []
    for _ in range(n_features + 1):
        if family == "constant":
            p0 += [0.0]
            lo += [-np.inf]
            hi += [np.inf]
        elif family == "pade_21":
            p0 += [1.0, 0.01, 0.0001, 0.01]
            lo += [-np.inf, -np.inf, -np.inf, 0.0]
            hi += [np.inf, np.inf, np.inf, np.inf]
    return np.array(p0), (np.array(lo), np.array(hi))


def fit_pade21(eps, feats, y, family="pade_21"):
    feats = np.atleast_2d(feats).astype(float)
    nf = feats.shape[0]
    model, total = make_model(nf, family)
    p0, bnd = initial_guess(nf, family)
    X = np.vstack([eps[None, :], feats])
    best = None
    for trial in [p0, p0 * 0.1, p0 * 10]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, X, y, p0=trial, bounds=bnd,
                                    maxfev=20000, method="trf")
            yhat = model(X, *popt)
            if np.any(np.isnan(yhat)) or np.any(np.isinf(yhat)):
                continue
            mae = float(np.mean(np.abs(y - yhat)))
            if best is None or mae < best["mae"]:
                best = {"params": popt, "yhat": yhat, "mae": mae,
                        "n_params": total}
        except Exception:
            continue
    return best


def metrics(y, yhat, np_):
    n = len(y)
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = (1.0 - (1.0 - r2) * (n - 1) / (n - np_ - 1)
              if n > np_ + 1 else r2)
    return mae, rmse, r2, r2_adj


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    print(f"Dataset: {len(df)} rows, {df.molecule.nunique()} mols")

    eps = df["eps"].to_numpy()
    mu_eV = df.mu.to_numpy()
    eta_eV = df.eta.to_numpy()
    gamma_eV = df.gamma.to_numpy()  # hyperhardness ∂³E/∂N³ (already in dataset, eV)
    omega_eV = mu_eV ** 2 / (2.0 * eta_eV)

    mu_kj = mu_eV * EV_TO_KJ
    eta_kj = eta_eV * EV_TO_KJ
    gamma_kj = gamma_eV * EV_TO_KJ
    omega_kj = omega_eV * EV_TO_KJ

    # LVC2 in kJ/mol — paired by direction (red→lvc2_red, ox→lvc2_ox)
    lvc2_red_kj = df.lvc2_red.to_numpy() * EH_TO_KJ
    lvc2_ox_kj = df.lvc2_ox.to_numpy() * EH_TO_KJ

    targets = {
        "EA":         (df.EA.to_numpy() * EH_TO_KJ, 3.7, 1.93),
        "IP":         (df.IP.to_numpy() * EH_TO_KJ, 5.2, 3.65),
        "Erelax_red": (df.Erelax_red.to_numpy() * EH_TO_KJ, 3.0, 3.83),
        "Erelax_ox":  (df.Erelax_ox.to_numpy() * EH_TO_KJ, 4.9, 5.43),
        "ddU_red":    (df.ddU_red.to_numpy() * EH_TO_KJ, 0.7, 0.69),
        "ddU_ox":     (df.ddU_ox.to_numpy() * EH_TO_KJ, 0.9, 0.93),
        "TdS_red":    (df.TdS_red.to_numpy() * EH_TO_KJ, 1.4, 1.38),
        "TdS_ox":     (df.TdS_ox.to_numpy() * EH_TO_KJ, 1.8, 1.80),
    }

    # Static feature sets (no LVC2)
    static_feature_sets = {
        "(μ, η)":         np.array([mu_kj, eta_kj]),
        "(μ, η, γ)":      np.array([mu_kj, eta_kj, gamma_kj]),
        "(μ, η, ω)":      np.array([mu_kj, eta_kj, omega_kj]),
        "(μ, η, γ, ω)":   np.array([mu_kj, eta_kj, gamma_kj, omega_kj]),
    }

    rows = []
    pred_unified = {}      # predictions with (μ, η) only
    pred_full = {}         # predictions with (μ, η, γ, ω, LVC2)
    true_kj = {}

    RED_TARGETS = {"EA", "Erelax_red", "ddU_red", "TdS_red"}
    # Cibles peu dépendantes du solvant : coefficients constants en ε
    CONSTANT_TARGETS = {"Erelax_red", "Erelax_ox", "TdS_red", "TdS_ox"}

    for tgt, (y, mae_rapport, mae_step4) in targets.items():
        true_kj[tgt] = y
        family = "constant" if tgt in CONSTANT_TARGETS else "pade_21"
        # LVC2 paired by direction
        lvc2 = lvc2_red_kj if tgt in RED_TARGETS else lvc2_ox_kj
        feature_sets = dict(static_feature_sets)
        feature_sets["(μ, η, LVC2)"] = np.array([mu_kj, eta_kj, lvc2])
        feature_sets["(μ, η, γ, LVC2)"] = np.array([mu_kj, eta_kj, gamma_kj, lvc2])
        feature_sets["(μ, η, γ, ω, LVC2)"] = np.array(
            [mu_kj, eta_kj, gamma_kj, omega_kj, lvc2])

        print(f"\n--- {tgt} ---  (family={family})")
        for fset_name, feats in feature_sets.items():
            res = fit_pade21(eps, feats, y, family=family)
            if res is None:
                print(f"   {fset_name}: FIT FAILED")
                continue
            m, r, r2, r2a = metrics(y, res["yhat"], res["n_params"])
            tag = "★" if m <= mae_rapport * 1.1 else "  "
            print(f"   {fset_name:22s}: MAE={m:6.2f}  RMSE={r:6.2f}  R²={r2:.4f}  "
                  f"n_par={res['n_params']:2d}  | rapport={mae_rapport:.1f}  step4={mae_step4:.2f}  {tag}")
            rows.append({
                "target": tgt, "features": fset_name, "family": family,
                "n_params": res["n_params"],
                "MAE": m, "RMSE": r, "R2": r2, "R2_adj": r2a,
                "MAE_rapport": mae_rapport, "MAE_step4_specific": mae_step4,
                "ratio_vs_rapport": m / mae_rapport,
            })
            if fset_name == "(μ, η)":
                pred_unified[tgt] = res["yhat"]
            if fset_name == "(μ, η, γ, ω, LVC2)":
                pred_full[tgt] = res["yhat"]

    # ΔG combined with (μ, η) only
    dg_red_t = -true_kj["EA"] + true_kj["Erelax_red"] + true_kj["ddU_red"] - true_kj["TdS_red"]
    dg_red_p_u = -pred_unified["EA"] + pred_unified["Erelax_red"] + pred_unified["ddU_red"] - pred_unified["TdS_red"]
    dg_ox_t = +true_kj["IP"] + true_kj["Erelax_ox"] + true_kj["ddU_ox"] - true_kj["TdS_ox"]
    dg_ox_p_u = +pred_unified["IP"] + pred_unified["Erelax_ox"] + pred_unified["ddU_ox"] - pred_unified["TdS_ox"]
    # ΔG combined with (μ, η, γ, ω, LVC2)
    dg_red_p_f = -pred_full["EA"] + pred_full["Erelax_red"] + pred_full["ddU_red"] - pred_full["TdS_red"]
    dg_ox_p_f = +pred_full["IP"] + pred_full["Erelax_ox"] + pred_full["ddU_ox"] - pred_full["TdS_ox"]

    for nm, t, p, label in [
        ("ΔG_red", dg_red_t, dg_red_p_u, "(combined μ,η)"),
        ("ΔG_ox",  dg_ox_t,  dg_ox_p_u,  "(combined μ,η)"),
        ("ΔG_red", dg_red_t, dg_red_p_f, "(combined μ,η,γ,ω,LVC2)"),
        ("ΔG_ox",  dg_ox_t,  dg_ox_p_f,  "(combined μ,η,γ,ω,LVC2)"),
    ]:
        m, r, r2, _ = metrics(t, p, 0)
        print(f"\n{nm} {label}: MAE={m:.2f}  RMSE={r:.2f}  R²={r2:.4f}")
        rows.append({"target": nm, "features": label, "n_params": np.nan,
                     "MAE": m, "RMSE": r, "R2": r2, "R2_adj": np.nan,
                     "MAE_rapport": np.nan, "MAE_step4_specific": np.nan,
                     "ratio_vs_rapport": np.nan})

    sdf = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("UNIFICATION CDFT — Padé[2/1] sur ε avec UNIQUEMENT (μ, η) [ou +ω]")
    print("=" * 100)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sdf.to_csv(RESULTS / "step5_unified_mu_eta_metrics.csv", index=False)
    print(f"\n[saved] step5_unified_mu_eta_metrics.csv")

    np.savez(RESULTS / "step5_unified_predictions.npz",
             solvent=df["solvent"].to_numpy(),
             molecule=df["molecule"].to_numpy(),
             **{f"true_{k}": v for k, v in true_kj.items()},
             **{f"pred_unified_{k}": v for k, v in pred_unified.items()},
             **{f"pred_full_{k}": v for k, v in pred_full.items()},
             true_dG_red=dg_red_t, pred_dG_red_unified=dg_red_p_u,
             true_dG_ox=dg_ox_t, pred_dG_ox_unified=dg_ox_p_u,
             pred_dG_red_full=dg_red_p_f, pred_dG_ox_full=dg_ox_p_f)


if __name__ == "__main__":
    main()
