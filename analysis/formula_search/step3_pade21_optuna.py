#!/usr/bin/env python3
"""Modèle Padé[2/1] sur (μ, η) avec coefficients Padé[2/1](ε), optimisé par Optuna.

Architecture (équivalente CDFT du rapport α(ε)+β(ε)·HOMO+γ(ε)·LUMO via la
bijection HOMO=μ-η/2, LUMO=μ+η/2) :

    y = α(ε) + B(ε)·μ + E(ε)·η

avec chacun de α, B, E ∈ Padé[2/1](ε_n) = (a + b·ε_n + c·ε_n²) / (1 + d·ε_n)
et ε_n = ε / ε_ref, ε_ref = 35.69 (Acetonitrile).

12 paramètres par signal × 2 signaux (red, ox) = 24 paramètres par paire.

Fit : Optuna propose un init pour les 12 params, curve_fit (Levenberg-Marquardt)
optimise localement, score = MAE_train (kJ/mol).
Reporting : MAE train + MAE CV 5-fold GroupKFold après re-fit dans chaque fold.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EH_TO_KJ = 2625.499638
N_FOLDS = 5
EPS_REF = 35.69  # Acetonitrile

EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}

PAIRS = [
    ("EA", "IP"),
    ("Erelax_red", "Erelax_ox"),
    ("ddU_red", "ddU_ox"),
    ("TdS_red", "TdS_ox"),
]

logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------- Padé[2/1] machinery ----------
def pade21(eps_n, a, b, c, d):
    """Padé[2/1]: (a + b*ε + c*ε²) / (1 + d*ε). Caller must guard against div-zero."""
    return (a + b * eps_n + c * eps_n ** 2) / (1.0 + d * eps_n)


def model_padded(eps_n, mu, eta, params):
    """y = α(ε) + B(ε)·μ + E(ε)·η, params shape (12,)."""
    alpha = pade21(eps_n, *params[0:4])
    B = pade21(eps_n, *params[4:8])
    E = pade21(eps_n, *params[8:12])
    return alpha + B * mu + E * eta


def model_for_curvefit(X, *params):
    """X: (3, n) -> (eps_n, mu, eta)."""
    eps_n, mu, eta = X[0], X[1], X[2]
    return model_padded(eps_n, mu, eta, np.asarray(params))


def safe_fit(X, y, p0, maxfev=20000):
    """curve_fit guarding against divergence; return params or None."""
    try:
        # Bound d in [-0.9/max(eps_n), 0.9/max(eps_n)] approx, but soft via large bounds.
        eps_max = float(X[0].max())
        d_lim = 0.95 / max(eps_max, 1e-6)
        lo = np.full(12, -np.inf)
        hi = np.full(12, np.inf)
        for k in (3, 7, 11):  # the d positions
            lo[k] = -d_lim
            hi[k] = d_lim
        popt, _ = curve_fit(model_for_curvefit, X, y, p0=p0,
                            bounds=(lo, hi), maxfev=maxfev)
        return popt
    except Exception:
        return None


def mae_kj(y_ha, yhat_ha):
    mask = np.isfinite(yhat_ha)
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_ha[mask] - yhat_ha[mask]) * EH_TO_KJ)))


def r2(y_ha, yhat_ha):
    mask = np.isfinite(yhat_ha)
    y = y_ha[mask] * EH_TO_KJ
    yh = yhat_ha[mask] * EH_TO_KJ
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(y_ha, yhat_ha):
    mask = np.isfinite(yhat_ha)
    return float(np.sqrt(np.mean(((y_ha[mask] - yhat_ha[mask]) * EH_TO_KJ) ** 2)))


def fit_one_signal(eps_n, mu, eta, y, p0, maxfev=20000):
    X = np.vstack([eps_n, mu, eta])
    popt = safe_fit(X, y, p0, maxfev=maxfev)
    if popt is None:
        return None, np.full_like(y, np.nan)
    yhat = model_for_curvefit(X, *popt)
    return popt, yhat


def cv_predict(eps_n, mu, eta, y, groups, p0):
    n = len(y)
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(eps_n, y, groups):
        popt, _ = fit_one_signal(eps_n[tr], mu[tr], eta[tr], y[tr], p0)
        if popt is None:
            continue
        X_te = np.vstack([eps_n[te], mu[te], eta[te]])
        pred[te] = model_for_curvefit(X_te, *popt)
    return pred


def optuna_search(eps_n, mu, eta, y, n_trials=500, scale_target=1.0):
    """Optuna search on 12-d Padé init; minimizes MAE_train (kJ/mol).

    scale_target estimates the target's typical magnitude to size the param ranges.
    """
    X = np.vstack([eps_n, mu, eta])
    s = scale_target  # rough scale
    s_mu = s / 3.0    # μ ~ -3 eV typical
    s_eta = s / 5.0   # η ~ 5 eV typical

    def objective(trial):
        p0 = np.array([
            trial.suggest_float("a_alpha", -3 * s, 3 * s),
            trial.suggest_float("b_alpha", -3 * s, 3 * s),
            trial.suggest_float("c_alpha", -3 * s, 3 * s),
            trial.suggest_float("d_alpha", -2.0, 2.0),
            trial.suggest_float("a_B", -3 * s_mu, 3 * s_mu),
            trial.suggest_float("b_B", -3 * s_mu, 3 * s_mu),
            trial.suggest_float("c_B", -3 * s_mu, 3 * s_mu),
            trial.suggest_float("d_B", -2.0, 2.0),
            trial.suggest_float("a_E", -3 * s_eta, 3 * s_eta),
            trial.suggest_float("b_E", -3 * s_eta, 3 * s_eta),
            trial.suggest_float("c_E", -3 * s_eta, 3 * s_eta),
            trial.suggest_float("d_E", -2.0, 2.0),
        ])
        popt = safe_fit(X, y, p0, maxfev=4000)
        if popt is None:
            return 1e6
        yhat = model_for_curvefit(X, *popt)
        return mae_kj(y, yhat)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=50)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    p0_best = np.array([best[k] for k in [
        "a_alpha", "b_alpha", "c_alpha", "d_alpha",
        "a_B", "b_B", "c_B", "d_B",
        "a_E", "b_E", "c_E", "d_E",
    ]])
    return p0_best, study.best_value


def main():
    df = pd.read_parquet(DATA).sort_values(
        ["solvent", "molecule", "conformer"]).reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    df["eps_n"] = df["eps"] / EPS_REF
    print(f"dataset: {len(df)} rows, {df.molecule.nunique()} mols, "
          f"{df.solvent.nunique()} solvents (after EA<0 filter)")

    eps_n = df["eps_n"].to_numpy()
    mu = df["mu"].to_numpy()
    eta = df["eta"].to_numpy()
    groups = df["molecule"].to_numpy()

    # Reference Rapport MAE (kJ/mol) — Padé[2/1] best
    rapport_mae = {
        "EA": 3.7, "IP": 5.2,
        "Erelax_red": 3.0, "Erelax_ox": 4.9,
        "ddU_red": 0.7, "ddU_ox": 0.9,
        "TdS_red": 1.4, "TdS_ox": 1.8,
    }

    summary = []
    pred_train = {}
    pred_cv = {}
    true_ha = {}
    final_params = {}

    for name_r, name_o in PAIRS:
        for tgt in (name_r, name_o):
            y = df[tgt].to_numpy()
            print(f"\n--- {tgt} ---")
            scale_target = float(y.std())  # target std in Hartree
            p0_best, best_train_mae = optuna_search(
                eps_n, mu, eta, y, n_trials=500, scale_target=scale_target,
            )

            # Final fit on full data
            popt_full, yhat_train = fit_one_signal(eps_n, mu, eta, y, p0_best)
            mae_t = mae_kj(y, yhat_train)
            r2_t = r2(y, yhat_train)
            rmse_t = rmse(y, yhat_train)

            # CV with init = optuna best
            yhat_cv = cv_predict(eps_n, mu, eta, y, groups, p0_best)
            mae_c = mae_kj(y, yhat_cv)
            r2_c = r2(y, yhat_cv)
            rmse_c = rmse(y, yhat_cv)

            print(f"   Optuna best (train): MAE={best_train_mae:.3f} kJ/mol")
            print(f"   final train: MAE={mae_t:.2f}  RMSE={rmse_t:.2f}  R²={r2_t:.4f}")
            print(f"   CV  (5-fold): MAE={mae_c:.2f}  RMSE={rmse_c:.2f}  R²={r2_c:.4f}")
            print(f"   rapport     : MAE={rapport_mae[tgt]:.1f} kJ/mol")

            pred_train[tgt] = yhat_train
            pred_cv[tgt] = yhat_cv
            true_ha[tgt] = y
            final_params[tgt] = (popt_full.tolist() if popt_full is not None else None)

            summary.append({
                "target": tgt,
                "MAE_train_kJ": mae_t, "RMSE_train_kJ": rmse_t, "R2_train": r2_t,
                "MAE_cv_kJ": mae_c, "RMSE_cv_kJ": rmse_c, "R2_cv": r2_c,
                "MAE_rapport": rapport_mae[tgt],
                "ratio_train_vs_rapport": mae_t / rapport_mae[tgt],
            })

    # Combined ΔG
    dg_red_t = (-true_ha["EA"] + true_ha["Erelax_red"]
                + true_ha["ddU_red"] + true_ha["TdS_red"])
    dg_red_p_tr = (-pred_train["EA"] + pred_train["Erelax_red"]
                   + pred_train["ddU_red"] + pred_train["TdS_red"])
    dg_red_p_cv = (-pred_cv["EA"] + pred_cv["Erelax_red"]
                   + pred_cv["ddU_red"] + pred_cv["TdS_red"])
    dg_ox_t = (+true_ha["IP"] + true_ha["Erelax_ox"]
               + true_ha["ddU_ox"] + true_ha["TdS_ox"])
    dg_ox_p_tr = (+pred_train["IP"] + pred_train["Erelax_ox"]
                  + pred_train["ddU_ox"] + pred_train["TdS_ox"])
    dg_ox_p_cv = (+pred_cv["IP"] + pred_cv["Erelax_ox"]
                  + pred_cv["ddU_ox"] + pred_cv["TdS_ox"])

    for nm, t, ptr, pcv in [
        ("ΔG_red", dg_red_t, dg_red_p_tr, dg_red_p_cv),
        ("ΔG_ox", dg_ox_t, dg_ox_p_tr, dg_ox_p_cv),
    ]:
        summary.append({
            "target": nm,
            "MAE_train_kJ": mae_kj(t, ptr), "RMSE_train_kJ": rmse(t, ptr), "R2_train": r2(t, ptr),
            "MAE_cv_kJ": mae_kj(t, pcv), "RMSE_cv_kJ": rmse(t, pcv), "R2_cv": r2(t, pcv),
            "MAE_rapport": np.nan, "ratio_train_vs_rapport": np.nan,
        })

    sdf = pd.DataFrame(summary)
    print("\n" + "=" * 100)
    print("RÉSUMÉ — Padé[2/1] sur (μ, η) avec coeffs Padé[2/1](ε), Optuna 150 trials")
    print("=" * 100)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sdf.to_csv(RESULTS / "step3_pade21_optuna_metrics.csv", index=False)
    print(f"\n[saved] step3_pade21_optuna_metrics.csv")

    with open(RESULTS / "step3_pade21_optuna_params.json", "w") as f:
        json.dump(final_params, f, indent=2)
    print(f"[saved] step3_pade21_optuna_params.json")

    np.savez(RESULTS / "step3_pade21_predictions.npz",
             solvent=df["solvent"].to_numpy(),
             molecule=df["molecule"].to_numpy(),
             **{f"true_{k}": v for k, v in true_ha.items()},
             **{f"pred_train_{k}": v for k, v in pred_train.items()},
             **{f"pred_cv_{k}": v for k, v in pred_cv.items()},
             true_dG_red=dg_red_t, pred_train_dG_red=dg_red_p_tr, pred_cv_dG_red=dg_red_p_cv,
             true_dG_ox=dg_ox_t, pred_train_dG_ox=dg_ox_p_tr, pred_cv_dG_ox=dg_ox_p_cv)
    print(f"[saved] step3_pade21_predictions.npz")


if __name__ == "__main__":
    main()
