#!/usr/bin/env python3
"""Reproduction exacte des résultats Padé[2/1] du rapport (analysis/generate_section4.py).

Architecture (identique au rapport):
    Y = Σ α_i(ε)·f_i + γ(ε)
avec α_i, γ ∈ Padé[2/1](ε) = (a₀ + a₁ε + a₂ε²)/(1 + b₁ε), b₁ ≥ 0.

Features choisies *exactement comme le rapport* :
    EA, IP : [IP_koop, EA_koop] = [-HOMO, -LUMO] en kJ/mol
    Erelax : best of [Koopmans, η+ω, HOMO+ω, η+μ]  (CDFT fallback, pas de LVC2)
    TdS    : [μ, η] en kJ/mol
    ddU    : [η, μ] en kJ/mol

Filtres : DEA Br/I + L-DOPA + EA<0 (déjà appliqué dans build_dataset.py).

Multi-start curve_fit (3 inits comme le rapport, pas Optuna) pour stabilité.
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
RESULTS.mkdir(exist_ok=True)

EH_TO_KJ = 2625.499638
EV_TO_KJ = 96.48533212

EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}

EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}

# Padé[2/1] : (a0 + a1·ε + a2·ε²)/(1 + b1·ε)
PADE_NPP = 4


def pade21(eps, params):
    a0, a1, a2, b1 = params
    return (a0 + a1 * eps + a2 * eps ** 2) / (1.0 + b1 * eps)


def make_model(n_features):
    """Y = Σ α_i(ε)·f_i + γ(ε), avec α_i et γ ∈ Padé[2/1]."""
    n_coeffs = n_features + 1  # +1 for γ intercept

    def model(X, *params):
        eps = X[0]
        result = np.zeros_like(eps, dtype=float)
        for i in range(n_features):
            offset = i * PADE_NPP
            result += pade21(eps, params[offset:offset + PADE_NPP]) * X[1 + i]
        offset = n_features * PADE_NPP
        result += pade21(eps, params[offset:offset + PADE_NPP])
        return result

    total = n_coeffs * PADE_NPP
    return model, total


def initial_guess(n_features):
    p0 = []
    lo = []
    hi = []
    for _ in range(n_features + 1):
        p0 += [1.0, 0.01, 0.0001, 0.01]
        lo += [-np.inf, -np.inf, -np.inf, 0.0]
        hi += [np.inf, np.inf, np.inf, np.inf]
    return np.array(p0), (np.array(lo), np.array(hi))


def fit_pade21(epsilon, features, y_target):
    """Multi-start curve_fit comme le rapport (p0, p0×0.1, p0×10)."""
    eps = np.asarray(epsilon, dtype=float)
    feats = np.atleast_2d(features).astype(float)
    n_features = feats.shape[0]
    model, total = make_model(n_features)
    p0, bounds = initial_guess(n_features)
    X = np.vstack([eps[None, :], feats])

    best = None
    for trial in [p0, p0 * 0.1, p0 * 10]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, X, y_target, p0=trial,
                                    bounds=bounds, maxfev=20000, method="trf")
            yhat = model(X, *popt)
            if np.any(np.isnan(yhat)) or np.any(np.isinf(yhat)):
                continue
            mae = float(np.mean(np.abs(y_target - yhat)))
            if best is None or mae < best["mae"]:
                best = {"params": popt, "yhat": yhat, "mae": mae,
                        "n_params": total}
        except Exception:
            continue
    return best


def metrics(y, yhat, n_params):
    n = len(y)
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1) if n > n_params + 1 else r2
    return mae, rmse, r2, r2_adj


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    print(f"After DEA/L-DOPA exclusion: {len(df)} rows, {df.molecule.nunique()} mols")

    # Targets in kJ/mol (data is in Hartree)
    EA = df.EA.to_numpy() * EH_TO_KJ
    IP = df.IP.to_numpy() * EH_TO_KJ
    Erelax_red = df.Erelax_red.to_numpy() * EH_TO_KJ
    Erelax_ox = df.Erelax_ox.to_numpy() * EH_TO_KJ
    ddU_red = df.ddU_red.to_numpy() * EH_TO_KJ
    ddU_ox = df.ddU_ox.to_numpy() * EH_TO_KJ
    TdS_red = df.TdS_red.to_numpy() * EH_TO_KJ
    TdS_ox = df.TdS_ox.to_numpy() * EH_TO_KJ

    eps = df["eps"].to_numpy()

    # Features (mu, eta in eV; HOMO=mu-eta/2, LUMO=mu+eta/2 in eV)
    mu_eV = df.mu.to_numpy()
    eta_eV = df.eta.to_numpy()
    HOMO_eV = mu_eV - eta_eV / 2.0
    LUMO_eV = mu_eV + eta_eV / 2.0
    omega_eV = mu_eV ** 2 / (2.0 * eta_eV)  # ω = μ²/(2η)

    IP_koop = -HOMO_eV * EV_TO_KJ      # in kJ/mol
    EA_koop = -LUMO_eV * EV_TO_KJ
    eta_kj = eta_eV * EV_TO_KJ
    mu_kj = mu_eV * EV_TO_KJ
    omega_kj = omega_eV * EV_TO_KJ

    # Target descriptions and feature sets (exact recipe from generate_section4.py)
    cases = [
        ("EA",          EA,         np.array([IP_koop, EA_koop]),  ["-HOMO", "-LUMO"], 3.7),
        ("IP",          IP,         np.array([IP_koop, EA_koop]),  ["-HOMO", "-LUMO"], 5.2),
        ("Erelax_red",  Erelax_red, None, None,                                          3.0),
        ("Erelax_ox",   Erelax_ox,  None, None,                                          4.9),
        ("ddU_red",     ddU_red,    np.array([eta_kj, mu_kj]),     ["η", "μ"],          0.7),
        ("ddU_ox",      ddU_ox,     np.array([eta_kj, mu_kj]),     ["η", "μ"],          0.9),
        ("TdS_red",     TdS_red,    np.array([mu_kj, eta_kj]),     ["μ", "η"],          1.4),
        ("TdS_ox",      TdS_ox,     np.array([mu_kj, eta_kj]),     ["μ", "η"],          1.8),
    ]

    rows = []
    pred = {}
    true = {}

    erelax_combos = [
        ("Koopmans", np.array([IP_koop, EA_koop]), ["-HOMO", "-LUMO"]),
        ("η+ω",      np.array([eta_kj, omega_kj]), ["η", "ω"]),
        ("HOMO+ω",   np.array([IP_koop, omega_kj]), ["-HOMO", "ω"]),
        ("η+μ",      np.array([eta_kj, mu_kj]),     ["η", "μ"]),
    ]

    for name, y, feats, fnames, mae_rapport in cases:
        if name.startswith("Erelax"):
            # Best combo for E_relax (CDFT fallback as in generate_section4.py)
            best_combo = None
            best_mae = np.inf
            for combo_name, fa, fn in erelax_combos:
                res = fit_pade21(eps, fa, y)
                if res is not None and res["mae"] < best_mae:
                    best_mae = res["mae"]
                    best_combo = (combo_name, fa, fn, res)
            combo_name, feats, fnames, res = best_combo
            print(f"\n--- {name} ---  best CDFT combo: {combo_name}  features={fnames}")
        else:
            print(f"\n--- {name} ---  features={fnames}")
            res = fit_pade21(eps, feats, y)

        if res is None:
            print(f"   FIT FAILED")
            continue
        m, r, r2, r2_adj = metrics(y, res["yhat"], res["n_params"])
        print(f"   MAE={m:.2f}  RMSE={r:.2f}  R²={r2:.4f}  R²_adj={r2_adj:.4f}  "
              f"n_par={res['n_params']}  | rapport: MAE={mae_rapport}")
        rows.append({
            "target": name, "features": "+".join(fnames),
            "n_params": res["n_params"],
            "MAE": m, "RMSE": r, "R2": r2, "R2_adj": r2_adj,
            "MAE_rapport": mae_rapport,
            "ratio": m / mae_rapport,
        })
        pred[name] = res["yhat"]
        true[name] = y

    # Combined ΔG
    dg_red_t = -true["EA"] + true["Erelax_red"] + true["ddU_red"] - true["TdS_red"]
    dg_red_p = -pred["EA"] + pred["Erelax_red"] + pred["ddU_red"] - pred["TdS_red"]
    dg_ox_t = +true["IP"] + true["Erelax_ox"] + true["ddU_ox"] - true["TdS_ox"]
    dg_ox_p = +pred["IP"] + pred["Erelax_ox"] + pred["ddU_ox"] - pred["TdS_ox"]

    for nm, t, p in [("ΔG_red", dg_red_t, dg_red_p), ("ΔG_ox", dg_ox_t, dg_ox_p)]:
        m, r, r2, _ = metrics(t, p, 0)
        rows.append({"target": nm, "features": "(combined)", "n_params": np.nan,
                     "MAE": m, "RMSE": r, "R2": r2, "R2_adj": np.nan,
                     "MAE_rapport": np.nan, "ratio": np.nan})

    sdf = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("REPRODUCTION RAPPORT — Padé[2/1] sur ε, features comme generate_section4.py")
    print("=" * 100)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sdf.to_csv(RESULTS / "step4_reproduce_rapport_metrics.csv", index=False)
    print(f"\n[saved] step4_reproduce_rapport_metrics.csv")

    np.savez(RESULTS / "step4_reproduce_predictions.npz",
             solvent=df["solvent"].to_numpy(),
             molecule=df["molecule"].to_numpy(),
             **{f"true_{k}": v for k, v in true.items()},
             **{f"pred_{k}": v for k, v in pred.items()},
             true_dG_red=dg_red_t, pred_dG_red=dg_red_p,
             true_dG_ox=dg_ox_t, pred_dG_ox=dg_ox_p)
    print(f"[saved] step4_reproduce_predictions.npz")


if __name__ == "__main__":
    main()
