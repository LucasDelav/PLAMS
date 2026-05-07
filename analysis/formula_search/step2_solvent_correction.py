#!/usr/bin/env python3
"""Étape 2 — correction solvant Φ(ε_r)·f(descripteurs) sur 1602 lignes.

Pour chaque paire, la forme f a été figée à l'étape 1 (step1_final_forms.json).
On teste plusieurs familles de Φ multiplicatif et on choisit la meilleure par
MAE total en 5-fold GroupKFold (groupes = molécule).

Le fit conjoint (θ_red, θ_ox, φ) est résolu par curve_fit en empilant
[y_red; y_ox] et en partageant φ entre les deux signaux.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
EH_TO_KJ = 2625.499638
N_FOLDS = 5

# Static dielectric constants used elsewhere in the project
EPSILON_R = {
    "Hexane": 1.88,
    "Toluene": 2.38,
    "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43,
    "Dichloromethane": 8.93,
    "Acetone": 20.49,
    "Methanol": 32.61,
    "Acetonitrile": 35.69,
    "DMSO": 46.83,
    "Water": 78.40,
}
EPS_REF = EPSILON_R["Acetonitrile"]  # forme f(d, θ) fittée à l'étape 1 sur ce solvant

PAIRED_MAP = {
    "lvc2_paired":        ("lvc2_red",     "lvc2_ox"),
    "fukui_paired_max":   ("fplus_max",    "fminus_max"),
    "fukui_paired_max_O": ("fplus_max_O",  "fminus_max_O"),
}


# ---------- Φ(ε_r) families (all normalized to Φ(ε_ref) = 1) ----------
def phi_constant(eps, *params):
    return np.ones_like(eps)


def phi_inv_eps(eps, a):
    """1 + a·(1/ε - 1/ε_ref), normalized => Φ(ε_ref)=1 with single param a."""
    return 1.0 + a * (1.0 / eps - 1.0 / EPS_REF)


def phi_log_eps(eps, a):
    return 1.0 + a * (np.log(eps) - np.log(EPS_REF))


def phi_onsager(eps, a):
    g = (eps - 1.0) / (2.0 * eps + 1.0)
    g_ref = (EPS_REF - 1.0) / (2.0 * EPS_REF + 1.0)
    return 1.0 + a * (g - g_ref)


def phi_born(eps, a):
    g = 1.0 - 1.0 / eps
    g_ref = 1.0 - 1.0 / EPS_REF
    return 1.0 + a * (g - g_ref)


def phi_pade(eps, a, b):
    """(1 + a·ε)/(1 + b·ε), shifted to be 1 at ε_ref by absorbing constant."""
    num = 1.0 + a * eps
    den = 1.0 + b * eps
    num_ref = 1.0 + a * EPS_REF
    den_ref = 1.0 + b * EPS_REF
    return (num / den) / (num_ref / den_ref)


PHI_FAMILIES = {
    "constant":      (phi_constant,  0, []),
    "inv_eps":       (phi_inv_eps,   1, [0.0]),
    "log_eps":       (phi_log_eps,   1, [0.0]),
    "onsager":       (phi_onsager,   1, [0.0]),
    "born":          (phi_born,      1, [0.0]),
    "pade2":         (phi_pade,      2, [0.0, 0.0]),
}


def col(df: pd.DataFrame, name: str, side: str) -> np.ndarray:
    if name in PAIRED_MAP:
        rc, oc = PAIRED_MAP[name]
        return df[rc if side == "red" else oc].to_numpy()
    return df[name].to_numpy()


def design(df: pd.DataFrame, monomials: list[dict], side: str) -> np.ndarray:
    n = len(df)
    cols = []
    for m in monomials:
        v = np.ones(n)
        for k, e in m.items():
            v = v * col(df, k, side) ** e
        cols.append(v)
    return np.column_stack(cols)


def joint_model_fn(F_red, F_ox, eps, phi_fn, n_params, n_phi):
    """Returns model(_x_dummy, *params) -> y_pred concatenated [red; ox]."""
    n = F_red.shape[0]

    def model(_x, *params):
        theta_red = np.array(params[:n_params])
        theta_ox = np.array(params[n_params:2 * n_params])
        phi_params = params[2 * n_params: 2 * n_params + n_phi]
        phi_vals = phi_fn(eps, *phi_params)
        y_red = phi_vals * (F_red @ theta_red)
        y_ox = phi_vals * (F_ox @ theta_ox)
        return np.concatenate([y_red, y_ox])
    return model


def fit_pair_family(F_red, F_ox, eps, y_red, y_ox, phi_fn, n_params, p0_phi,
                    theta_init_red, theta_init_ox):
    """Joint LM fit; returns (params, y_pred_red, y_pred_ox)."""
    n_phi = len(p0_phi)
    p0 = np.concatenate([theta_init_red, theta_init_ox, np.array(p0_phi)])
    y_stack = np.concatenate([y_red, y_ox])
    model = joint_model_fn(F_red, F_ox, eps, phi_fn, n_params, n_phi)
    try:
        popt, _ = curve_fit(model, np.zeros(len(y_stack)), y_stack, p0=p0, maxfev=20000)
    except Exception:
        return None, None, None
    pred_stack = model(None, *popt)
    n = F_red.shape[0]
    return popt, pred_stack[:n], pred_stack[n:]


def cv_fit_family(F_red, F_ox, eps, y_red, y_ox, groups, phi_fn, n_params,
                  p0_phi, theta_init_red, theta_init_ox):
    """5-fold GroupKFold; returns (oof_pred_red, oof_pred_ox, fold_phi_params)."""
    n = len(y_red)
    pr = np.full(n, np.nan)
    po = np.full(n, np.nan)
    n_phi = len(p0_phi)
    fold_phi = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(F_red, y_red, groups):
        popt, _, _ = fit_pair_family(
            F_red[tr], F_ox[tr], eps[tr], y_red[tr], y_ox[tr],
            phi_fn, n_params, p0_phi, theta_init_red, theta_init_ox,
        )
        if popt is None:
            return pr, po, fold_phi
        # Predict on test fold
        theta_r = popt[:n_params]
        theta_o = popt[n_params:2 * n_params]
        phi_p = popt[2 * n_params: 2 * n_params + n_phi]
        phi_te = phi_fn(eps[te], *phi_p)
        pr[te] = phi_te * (F_red[te] @ theta_r)
        po[te] = phi_te * (F_ox[te] @ theta_o)
        fold_phi.append(phi_p.tolist())
    return pr, po, fold_phi


def metrics_kj(y_ha, yhat_ha):
    mask = np.isfinite(yhat_ha)
    y = y_ha[mask] * EH_TO_KJ
    yhat = yhat_ha[mask] * EH_TO_KJ
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def main():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["solvent", "molecule", "conformer"]).reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    print(f"Multi-solvent dataset: {len(df)} rows, {df.molecule.nunique()} molecules, "
          f"{df.solvent.nunique()} solvents")

    with open(RESULTS / "step1_final_forms.json") as f:
        step1 = json.load(f)

    pair_target = {
        "EA_IP":   ("EA", "IP"),
        "Erelax":  ("Erelax_red", "Erelax_ox"),
        "ddU":     ("ddU_red", "ddU_ox"),
        "TdS":     ("TdS_red", "TdS_ox"),
    }

    summary_rows = []
    best_by_pair = {}
    pred_by_target = {}  # target -> oof pred
    true_by_target = {}

    for pair_name, (name_r, name_o) in pair_target.items():
        info = step1[pair_name]
        monos = info["monomials"]
        theta_init_red = np.array(info["theta_red"])
        theta_init_ox = np.array(info["theta_ox"])
        F_red = design(df, monos, "red")
        F_ox = design(df, monos, "ox")
        y_red = df[name_r].to_numpy()
        y_ox = df[name_o].to_numpy()
        groups = df["molecule"].to_numpy()
        eps = df["eps"].to_numpy()

        print(f"\n=== {pair_name} ===  ({len(monos)} monômes, {len(y_red)} lignes)")
        print(f"  σ(target) red={y_red.std()*EH_TO_KJ:.2f} kJ/mol  ox={y_ox.std()*EH_TO_KJ:.2f} kJ/mol")

        results_for_pair = {}
        for fam_name, (phi_fn, n_phi, p0_phi) in PHI_FAMILIES.items():
            pr, po, _ = cv_fit_family(
                F_red, F_ox, eps, y_red, y_ox, groups,
                phi_fn, len(monos), p0_phi, theta_init_red, theta_init_ox,
            )
            mae_r, rmse_r, r2_r = metrics_kj(y_red, pr)
            mae_o, rmse_o, r2_o = metrics_kj(y_ox, po)
            results_for_pair[fam_name] = {
                "mae_r": mae_r, "rmse_r": rmse_r, "r2_r": r2_r,
                "mae_o": mae_o, "rmse_o": rmse_o, "r2_o": r2_o,
                "mae_total": mae_r + mae_o,
                "n_phi": n_phi, "pred_r": pr, "pred_o": po,
            }
            print(f"  Φ={fam_name:12s} | MAE_red={mae_r:6.2f}  MAE_ox={mae_o:6.2f}  "
                  f"R²_red={r2_r:6.3f}  R²_ox={r2_o:6.3f}  total={mae_r+mae_o:6.2f}")

        best = min(results_for_pair, key=lambda k: results_for_pair[k]["mae_total"])
        print(f"  -> meilleure famille: {best}")
        best_by_pair[pair_name] = best
        pred_by_target[name_r] = results_for_pair[best]["pred_r"]
        pred_by_target[name_o] = results_for_pair[best]["pred_o"]
        true_by_target[name_r] = y_red
        true_by_target[name_o] = y_ox

        # Refit on full data (no CV) to export final coefficients
        info_best = results_for_pair[best]
        phi_fn, n_phi, p0_phi = PHI_FAMILIES[best]
        popt, _, _ = fit_pair_family(
            F_red, F_ox, eps, y_red, y_ox, phi_fn, len(monos), p0_phi,
            theta_init_red, theta_init_ox,
        )

        for tgt, mae, rmse, r2 in [
            (name_r, info_best["mae_r"], info_best["rmse_r"], info_best["r2_r"]),
            (name_o, info_best["mae_o"], info_best["rmse_o"], info_best["r2_o"]),
        ]:
            summary_rows.append({
                "pair": pair_name, "target": tgt, "phi_family": best,
                "MAE_kJ/mol": mae, "RMSE_kJ/mol": rmse, "R2": r2,
            })

        # Per-solvent MAE for the chosen family
        pred_r = info_best["pred_r"]
        pred_o = info_best["pred_o"]
        per_solv_rows = []
        for sname, sub in df.groupby("solvent"):
            idx = sub.index.to_numpy()
            mae_r, _, _ = metrics_kj(y_red[idx], pred_r[idx])
            mae_o, _, _ = metrics_kj(y_ox[idx], pred_o[idx])
            per_solv_rows.append({
                "pair": pair_name, "solvent": sname, "n": len(idx),
                "MAE_red": mae_r, "MAE_ox": mae_o,
            })
        pd.DataFrame(per_solv_rows).to_csv(
            RESULTS / f"step2_per_solvent_{pair_name}.csv", index=False, float_format="%.3f")

    # ΔG dérivés en kJ/mol
    dg_red_true = (-true_by_target["EA"] + true_by_target["Erelax_red"]
                   + true_by_target["ddU_red"] + true_by_target["TdS_red"])
    dg_red_pred = (-pred_by_target["EA"] + pred_by_target["Erelax_red"]
                   + pred_by_target["ddU_red"] + pred_by_target["TdS_red"])
    dg_ox_true = (+true_by_target["IP"] + true_by_target["Erelax_ox"]
                  + true_by_target["ddU_ox"] + true_by_target["TdS_ox"])
    dg_ox_pred = (+pred_by_target["IP"] + pred_by_target["Erelax_ox"]
                  + pred_by_target["ddU_ox"] + pred_by_target["TdS_ox"])

    for nm, t, p in [("ΔG_red", dg_red_true, dg_red_pred),
                     ("ΔG_ox", dg_ox_true, dg_ox_pred)]:
        mae, rmse, r2 = metrics_kj(t, p)
        summary_rows.append({"pair": nm, "target": nm, "phi_family": "(combined)",
                             "MAE_kJ/mol": mae, "RMSE_kJ/mol": rmse, "R2": r2})

    sdf = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("RÉSUMÉ ÉTAPE 2 (multi-solvant, 5-fold GroupKFold OOF, kJ/mol)")
    print("=" * 80)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sdf.to_csv(RESULTS / "step2_metrics_kJ.csv", index=False)
    print(f"\n[saved] step2_metrics_kJ.csv")
    with open(RESULTS / "step2_best_families.json", "w") as f:
        json.dump(best_by_pair, f, indent=2)
    print(f"[saved] step2_best_families.json")

    # Save predictions for parity plots
    np.savez(RESULTS / "step2_predictions.npz",
             solvent=df["solvent"].to_numpy(),
             molecule=df["molecule"].to_numpy(),
             **{f"true_{k}": v for k, v in true_by_target.items()},
             **{f"pred_{k}": v for k, v in pred_by_target.items()},
             true_dG_red=dg_red_true, pred_dG_red=dg_red_pred,
             true_dG_ox=dg_ox_true, pred_dG_ox=dg_ox_pred)
    print(f"[saved] step2_predictions.npz")


if __name__ == "__main__":
    main()
