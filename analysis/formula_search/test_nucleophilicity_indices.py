#!/usr/bin/env python3
"""Test nucleophilicity descriptors from CDFT for the IP model.

Model template:
    y(eps) = [Σ β_X(eps)·X for X in descriptors] × (1 + a_corr · fukui)
    β_X(eps) = (a0 + a1·g)/(1 + b1·g),  g = (eps-1)/(2*eps+1)
    intercept = 0
    fukui = f_min^dual_noO  (best Fukui from previous study)

Tests (all warm-started from the saved (μ+η+ω) IP fit):

  T1. Gázquez ω⁻ = I²/(2η) = (η/2 − μ)²/(2η)
      Replace ω → ω⁻.  3 descriptors, 10 params.

  T2. Roy N' = 10/ω⁻
      Replace ω → 10/ω⁻.  3 descriptors, 10 params.

  T3. (μ, η, ω, ω⁻) together
      4 descriptors, 13 params.

  T4. Local Domingo N·f⁻_max_noO as the Fukui term
      With N = −μ + η/2 (= E_HOMO with sign convention; constant offset folded
      into a_corr), the Fukui descriptor becomes (η/2−μ) · f⁻_max_noO.
      Same (μ, η, ω) descriptors, but Fukui term changes.

Reference: (μ, η, ω) × Padé[1/1] g_Born × (1 + a·f²_min_noO) → MAE 3.449, RMSE 6.403.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
PRED_DIR = REPO / "dataset_predict_dg"
COEF_PREV = REPO / "analysis" / "formula_search" / "results" / "fukui_correction_coeffs.csv"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
SEED = 42
N_TRIALS = 12000
STAB_TOL = 0.05
optuna.logging.set_verbosity(optuna.logging.WARNING)


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    return [lines[2 + i].split()[0] for i in range(int(lines[0]))]


def fukui_min_max_no_o(values, symbols):
    arr = np.asarray(values, dtype=float)
    syms = np.asarray(symbols)
    mask = syms != "O"
    if not mask.any():
        return float("nan"), float("nan")
    sub = arr[mask]
    return float(sub.min()), float(sub.max())


def load_fukui(df):
    rows = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        if not path.exists():
            rows.append((np.nan, np.nan)); continue
        try:
            with open(path) as f:
                pred = json.load(f)
            ns = pred["states"]["neutral_sp"]
            syms = parse_xyz_symbols(ns["xyz"])
            d_min, _ = fukui_min_max_no_o(ns["cdft"]["dual"], syms)
            _, fm_max = fukui_min_max_no_o(ns["cdft"]["fminus"], syms)
            rows.append((d_min, fm_max))
        except Exception:
            rows.append((np.nan, np.nan))
    return pd.DataFrame(rows, columns=["f2_min_noO", "fminus_max_noO"], index=df.index)


def predict(eps, descs_arr, fukui_term, params, descriptor_names):
    """descs_arr shape (n_descs, N). fukui_term shape (N,). params dict.
    Returns prediction or None if instability."""
    g = g_born(eps)
    out = np.zeros_like(eps)
    for i, d in enumerate(descriptor_names):
        a0, a1, b1 = params[f"a0_{d}"], params[f"a1_{d}"], params[f"b1_{d}"]
        den = 1.0 + b1 * g
        if np.any(np.abs(den) < STAB_TOL):
            return None
        out += (a0 + a1 * g) / den * descs_arr[i]
    return out * (1.0 + params["a_corr"] * fukui_term)


def warmstart(descriptor_names, fukui_label="f2_min_noO"):
    """Load saved (mu+eta+omega+f2_min_noO) IP fit as warm-start for new tests."""
    df = pd.read_csv(COEF_PREV)
    df = df[(df.target == "IP") & (df.fukui == fukui_label)].set_index("param")
    x0 = {}
    for d in descriptor_names:
        # Map omega-coefs to omega_minus / inv_omega_minus / new descriptors
        src = "omega" if d in ("omega_minus", "inv_omega_minus") else d
        if d in ("omega", "omega_minus", "inv_omega_minus"):
            for p in ("a0", "a1", "b1"):
                key = f"{p}_omega" if d != "omega" else f"{p}_omega"
                x0[f"{p}_{d}"] = float(df.loc[key, "value"])
        else:
            for p in ("a0", "a1", "b1"):
                x0[f"{p}_{d}"] = float(df.loc[f"{p}_{d}", "value"])
    x0["a_corr"] = float(df.loc["a_corr", "value"])
    return x0


def fit_test(name, df, descriptor_names, descs_arr, fukui_term, valid_mask,
             warmstart_label="f2_min_noO"):
    """Run joint refit and return result dict."""
    eps = df["epsilon"].values
    y = df["IP"].values * EH_TO_KJ

    eps_v = eps[valid_mask]
    descs_v = descs_arr[:, valid_mask]
    f_v = fukui_term[valid_mask]
    y_v = y[valid_mask]

    x0 = warmstart(descriptor_names, fukui_label=warmstart_label)

    PARAM_NAMES = ([f"{p}_{d}" for d in descriptor_names for p in ("a0", "a1", "b1")]
                   + ["a_corr"])

    scale_num = max(max(abs(v) for k, v in x0.items() if k.startswith(("a0_", "a1_"))), 50.0)
    bounds = {}
    for d in descriptor_names:
        bounds[f"a0_{d}"] = (-3 * scale_num, 3 * scale_num)
        bounds[f"a1_{d}"] = (-5 * scale_num, 5 * scale_num)
        bounds[f"b1_{d}"] = (-5.0, 5.0)
    bounds["a_corr"] = (-50.0, 50.0)

    def objective(trial):
        params = {n: trial.suggest_float(n, *bounds[n]) for n in PARAM_NAMES}
        pr = predict(eps_v, descs_v, f_v, params, descriptor_names)
        if pr is None or np.any(~np.isfinite(pr)):
            return 1e9
        return float(np.mean(np.abs(y_v - pr)))

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0, sigma0=10.0,
                                            n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(x0)
    for a_seed in (-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0):
        x0_a = dict(x0); x0_a["a_corr"] = a_seed
        study.enqueue_trial(x0_a)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {n: study.best_params[n] for n in PARAM_NAMES}
    pred_full = predict(eps, descs_arr, fukui_term, best, descriptor_names)
    pred_full = np.where(valid_mask, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))
    return {"name": name, "n_params": len(PARAM_NAMES), "MAE": mae, "RMSE": rmse,
            "params": best, "pred": pred_full,
            "descriptor_names": descriptor_names}


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["epsilon"] = df["solvent"].map(EPSILON_R)

    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    # I = η/2 − μ  (vertical IP, eV, from CDFT μ and η)
    I = df["eta"] / 2.0 - df["mu"]
    df["omega_minus"] = I ** 2 / (2.0 * df["eta"])     # Gázquez ω⁻
    df["inv_omega_minus"] = 10.0 / df["omega_minus"]   # Roy N' = 10/ω⁻

    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)
    print(f"\nDescriptor ranges:")
    for col in ("mu", "eta", "omega", "omega_minus", "inv_omega_minus"):
        print(f"  {col:18s}: [{df[col].min():+.3f}, {df[col].max():+.3f}]")

    fukui_df = load_fukui(df)
    df = pd.concat([df, fukui_df], axis=1)
    valid = (df[["f2_min_noO", "fminus_max_noO"]].notna().all(axis=1)).values
    print(f"\nFukui complete on {valid.sum()}/{len(df)} conformers", flush=True)

    eps = df["epsilon"].values

    # Reference (T0)
    ref = {"name": "REF: (μ,η,ω) × Padé g_Born × (1+a·f²_min_noO)",
           "n_params": 10, "MAE": 3.449, "RMSE": 6.403}

    # T1: Gázquez ω⁻
    descs_t1 = np.stack([df["mu"].values, df["eta"].values, df["omega_minus"].values])
    fukui_t1 = df["f2_min_noO"].values
    print("\n--- T1: (μ, η, ω⁻ Gázquez) × Padé g_Born × (1+a·f²_min_noO) ---", flush=True)
    t1 = fit_test("T1: μ,η,ω⁻ × f²_min", df, ["mu", "eta", "omega_minus"], descs_t1, fukui_t1, valid)
    print(f"  MAE = {t1['MAE']:.3f},  RMSE = {t1['RMSE']:.3f},  a_corr = {t1['params']['a_corr']:+.4f}")

    # T2: Roy N' = 10/ω⁻
    descs_t2 = np.stack([df["mu"].values, df["eta"].values, df["inv_omega_minus"].values])
    fukui_t2 = df["f2_min_noO"].values
    print("\n--- T2: (μ, η, 10/ω⁻ Roy N') × Padé g_Born × (1+a·f²_min_noO) ---", flush=True)
    t2 = fit_test("T2: μ,η,10/ω⁻ × f²_min", df, ["mu", "eta", "inv_omega_minus"], descs_t2, fukui_t2, valid)
    print(f"  MAE = {t2['MAE']:.3f},  RMSE = {t2['RMSE']:.3f},  a_corr = {t2['params']['a_corr']:+.4f}")

    # T3: (μ, η, ω, ω⁻) — both Parr and Gázquez
    descs_t3 = np.stack([df["mu"].values, df["eta"].values,
                         df["omega"].values, df["omega_minus"].values])
    fukui_t3 = df["f2_min_noO"].values
    print("\n--- T3: (μ, η, ω, ω⁻) × Padé g_Born × (1+a·f²_min_noO) ---", flush=True)
    t3 = fit_test("T3: μ,η,ω,ω⁻ × f²_min", df, ["mu", "eta", "omega", "omega_minus"], descs_t3, fukui_t3, valid)
    print(f"  MAE = {t3['MAE']:.3f},  RMSE = {t3['RMSE']:.3f},  a_corr = {t3['params']['a_corr']:+.4f}")

    # T4: Local Domingo N·f⁻_max_noO as Fukui term
    # With N = E_HOMO − E_HOMO(TCE) ; constant offset is absorbed by a_corr,
    # so we use N_proxy = −μ + η/2 (= E_HOMO_reversed in eV).
    # We use fminus_max_noO as the Fukui (since Domingo's local form is N · f⁻).
    descs_t4 = np.stack([df["mu"].values, df["eta"].values, df["omega"].values])
    N_proxy = -df["mu"].values + df["eta"].values / 2.0
    fukui_t4 = N_proxy * df["fminus_max_noO"].values
    print("\n--- T4: (μ, η, ω) × Padé g_Born × (1 + a·N_local·f⁻_max_noO) [Domingo local] ---", flush=True)
    t4 = fit_test("T4: μ,η,ω × N·f⁻_max", df, ["mu", "eta", "omega"], descs_t4, fukui_t4, valid,
                  warmstart_label="fminus_max_noO")
    print(f"  MAE = {t4['MAE']:.3f},  RMSE = {t4['RMSE']:.3f},  a_corr = {t4['params']['a_corr']:+.4f}")

    # Summary
    print("\n=== Summary ===\n", flush=True)
    rows = [
        ("REF: (μ,η,ω) × Padé g_Born × (1+a·f²_min_noO)", 10, 3.449, 6.403),
        (t1["name"], t1["n_params"], t1["MAE"], t1["RMSE"]),
        (t2["name"], t2["n_params"], t2["MAE"], t2["RMSE"]),
        (t3["name"], t3["n_params"], t3["MAE"], t3["RMSE"]),
        (t4["name"], t4["n_params"], t4["MAE"], t4["RMSE"]),
    ]
    print(f"  {'Model':<48s}  n_p  {'MAE':>6s}  {'RMSE':>6s}  Δ_MAE")
    for name, np_, m, r in rows:
        delta = m - 3.449
        marker = " <- best" if m == min(x[2] for x in rows) else ""
        print(f"  {name:<48s}  {np_:3d}  {m:6.3f}  {r:6.3f}  {delta:+.3f}{marker}")

    # Save coefficients
    out_csv = REPO / "analysis" / "formula_search" / "results" / "ip_nucleophilicity_indices.csv"
    rows_save = []
    for t in (t1, t2, t3, t4):
        for k, v in t["params"].items():
            rows_save.append({"test": t["name"], "n_params": t["n_params"],
                              "MAE": t["MAE"], "RMSE": t["RMSE"],
                              "param": k, "value": float(v)})
    pd.DataFrame(rows_save).to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
