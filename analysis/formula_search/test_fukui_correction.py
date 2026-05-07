#!/usr/bin/env python3
"""Test multiplicative Fukui corrections — joint refit of baseline + correction.

Baseline form: y_base(eps) = beta_mu(eps)·mu + beta_eta(eps)·eta + beta_omega(eps)·omega
with beta_X(eps) = (a0_X + a1_X * g)/(1 + b1_X * g),  g = g_Born(eps).
That's 9 baseline parameters per target.

Correction:        y = y_base × (1 + a · fukui)         + 1 parameter `a`.

All 10 parameters are refit jointly to minimise the MAE on 1336 conformers,
warm-started at the saved Padé[1/1] g_Born coefficients (from
final_solvent_models.json) and a=0. So the previous baseline is the
starting point of the optimisation, not a hard constraint.

4 separate tests:

  EA test 1:  EA_corr = EA_base × (1 + a · f_max^+_noO)
  EA test 2:  EA_corr = EA_base × (1 + a · f_max^dual_noO)
  IP test 1:  IP_corr = IP_base × (1 + a · f_max^-_noO)
  IP test 2:  IP_corr = IP_base × (1 + a · f_min^dual_noO)

"_noO" = max/min over ALL atoms whose symbol is not "O".
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
MODELS = REPO / "analysis" / "formula_search" / "results" / "final_solvent_models.json"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
DESCRIPTORS = ("mu", "eta", "omega")
SEED = 42
N_TRIALS = 12000
STAB_TOL = 0.05
optuna.logging.set_verbosity(optuna.logging.WARNING)


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def beta_pade11_g(eps, p):
    g = g_born(eps)
    return (p["a0"] + p["a1"] * g) / (1.0 + p["b1"] * g)


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def fukui_no_o(values, symbols):
    """Return (max_noO, min_noO). NaN if no non-O atom."""
    arr = np.asarray(values, dtype=float)
    syms = np.asarray(symbols)
    mask = syms != "O"
    if not mask.any():
        return float("nan"), float("nan")
    sub = arr[mask]
    return float(sub.max()), float(sub.min())


def load_fukui_noO(df):
    """For each conformer in df, read the predict-dg JSON and compute noO Fukui."""
    rows = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        if not path.exists():
            rows.append({"fplus_max_noO": np.nan, "fminus_max_noO": np.nan,
                         "f2_max_noO": np.nan, "f2_min_noO": np.nan})
            continue
        try:
            with open(path) as f:
                pred = json.load(f)
            ns = pred["states"]["neutral_sp"]
            cdft = ns["cdft"]
            syms = parse_xyz_symbols(ns["xyz"])
            fp_max, _ = fukui_no_o(cdft["fplus"], syms)
            fm_max, _ = fukui_no_o(cdft["fminus"], syms)
            d_max, d_min = fukui_no_o(cdft["dual"], syms)
            rows.append({
                "fplus_max_noO": fp_max,
                "fminus_max_noO": fm_max,
                "f2_max_noO": d_max,
                "f2_min_noO": d_min,
            })
        except (KeyError, IndexError, ValueError, json.JSONDecodeError):
            rows.append({"fplus_max_noO": np.nan, "fminus_max_noO": np.nan,
                         "f2_max_noO": np.nan, "f2_min_noO": np.nan})
    return pd.DataFrame(rows, index=df.index)


def baseline_predict(df, target, models):
    """Reconstruct y_pred for `target` using the saved Padé[1/1] g_Born model."""
    p_per = models["pade11_g"]["targets"][target]["params_per_descriptor"]
    eps = df["epsilon"].values
    pred = np.zeros(len(df))
    for d in DESCRIPTORS:
        pred += beta_pade11_g(eps, p_per[d]) * df[d].values
    return pred


# --- joint baseline + correction fit ---------------------------------------

PARAM_NAMES = [f"{p}_{d}" for d in DESCRIPTORS for p in ("a0", "a1", "b1")] + ["a_corr"]


def predict_full(eps, mu, eta, om, fukui, params):
    """Apply Padé[1/1] g_Born baseline × (1 + a_corr * fukui).
    `params` is dict keyed by PARAM_NAMES."""
    g = g_born(eps)
    pred = np.zeros_like(eps)
    for d, x in zip(DESCRIPTORS, (mu, eta, om)):
        a0, a1, b1 = params[f"a0_{d}"], params[f"a1_{d}"], params[f"b1_{d}"]
        den = 1.0 + b1 * g
        if np.any(np.abs(den) < STAB_TOL):
            return None
        pred += (a0 + a1 * g) / den * x
    return pred * (1.0 + params["a_corr"] * fukui)


def fit_correction(y, df, fukui_col, models, target, label):
    """Refit the 10 params (9 baseline + a_corr) jointly to minimise MAE.

    Warm-start: saved Padé[1/1] g_Born coefficients and a_corr = 0.
    """
    fukui = df[fukui_col].values
    valid = np.isfinite(fukui)
    if not valid.all():
        print(f"  ! {label}: dropping {int((~valid).sum())} rows with NaN fukui", flush=True)

    eps_v = df["epsilon"].values[valid]
    mu_v = df["mu"].values[valid]
    eta_v = df["eta"].values[valid]
    om_v = df["omega"].values[valid]
    f_v = fukui[valid]
    y_v = y[valid]

    # Warm-start values from saved JSON
    p_saved = models["pade11_g"]["targets"][target]["params_per_descriptor"]
    x0 = {f"a0_{d}": float(p_saved[d]["a0"]) for d in DESCRIPTORS}
    x0.update({f"a1_{d}": float(p_saved[d]["a1"]) for d in DESCRIPTORS})
    x0.update({f"b1_{d}": float(p_saved[d]["b1"]) for d in DESCRIPTORS})
    x0["a_corr"] = 0.0

    # Bounds: numerator scaled by warm-start magnitude × 5; denominator and a_corr modest
    scale = max(max(abs(v) for k, v in x0.items() if k.startswith(("a0_", "a1_"))), 50.0)
    bounds = {}
    for d in DESCRIPTORS:
        bounds[f"a0_{d}"] = (-3 * scale, 3 * scale)
        bounds[f"a1_{d}"] = (-5 * scale, 5 * scale)
        bounds[f"b1_{d}"] = (-5.0, 5.0)
    bounds["a_corr"] = (-50.0, 50.0)

    def objective(trial):
        params = {n: trial.suggest_float(n, *bounds[n]) for n in PARAM_NAMES}
        pred = predict_full(eps_v, mu_v, eta_v, om_v, f_v, params)
        if pred is None or np.any(~np.isfinite(pred)):
            return 1e9
        return float(np.mean(np.abs(y_v - pred)))

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0, sigma0=10.0,
                                            n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # Warm-start at a_corr = 0 plus several seeds with different a_corr signs
    # and magnitudes, so CMA-ES doesn't collapse to the baseline.
    study.enqueue_trial(x0)
    for a_seed in (-1.0, -0.3, -0.1, 0.1, 0.3, 1.0):
        x0_a = dict(x0)
        x0_a["a_corr"] = a_seed
        study.enqueue_trial(x0_a)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {n: study.best_params[n] for n in PARAM_NAMES}
    full_eps = df["epsilon"].values
    full_mu, full_eta, full_om = df["mu"].values, df["eta"].values, df["omega"].values
    pred_full = predict_full(full_eps, full_mu, full_eta, full_om, fukui, best)
    pred_full = np.where(valid, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))
    return best, pred_full, mae, rmse, valid


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)

    print("\nLoading per-conformer Fukui (excluding O atoms) from predict-dg JSONs...", flush=True)
    fukui_df = load_fukui_noO(df)
    df = pd.concat([df, fukui_df], axis=1)
    n_complete = df[["fplus_max_noO", "fminus_max_noO", "f2_max_noO", "f2_min_noO"]].dropna().shape[0]
    print(f"  Complete Fukui data on {n_complete}/{len(df)} conformers", flush=True)

    models = json.load(open(MODELS))

    tests = [
        ("EA", "fplus_max_noO",  "EA_corr = EA_base × (1 + a · f_max^+_noO)"),
        ("EA", "f2_max_noO",     "EA_corr = EA_base × (1 + a · f_max^dual_noO)"),
        ("IP", "fminus_max_noO", "IP_corr = IP_base × (1 + a · f_max^-_noO)"),
        ("IP", "f2_min_noO",     "IP_corr = IP_base × (1 + a · f_min^dual_noO)"),
    ]

    results = []
    for tgt, fukui_col, formula in tests:
        y = df[tgt].values * EH_TO_KJ
        base = baseline_predict(df, tgt, models)
        mae_base = float(np.mean(np.abs(y - base)))
        rmse_base = float(np.sqrt(np.mean((y - base) ** 2)))

        best, pred, mae_corr, rmse_corr, valid = fit_correction(
            y, df, fukui_col, models, tgt, label=f"{tgt}/{fukui_col}"
        )
        results.append({
            "target": tgt, "fukui": fukui_col, "formula": formula,
            "a_corr_opt": best["a_corr"],
            "params": best,
            "MAE_base": mae_base, "RMSE_base": rmse_base,
            "MAE_corr": mae_corr, "RMSE_corr": rmse_corr,
            "delta_MAE": mae_corr - mae_base, "delta_RMSE": rmse_corr - rmse_base,
            "pred_corr": pred, "y": y,
        })
        print(f"  done {tgt} / {fukui_col}: MAE {mae_base:.3f} -> {mae_corr:.3f}  (a_corr = {best['a_corr']:.4f})", flush=True)

    summary = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ("pred_corr", "y", "params")}
                            for r in results])
    print("\n=== Fukui-correction summary (joint refit of 10 params) ===\n", flush=True)
    print(summary[["target", "fukui", "a_corr_opt",
                   "MAE_base", "MAE_corr", "delta_MAE",
                   "RMSE_base", "RMSE_corr", "delta_RMSE"]]
          .to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # Save coefficients of every test
    coef_rows = []
    for r in results:
        for name, val in r["params"].items():
            coef_rows.append({"target": r["target"], "fukui": r["fukui"],
                              "param": name, "value": float(val)})
    coef_df = pd.DataFrame(coef_rows)
    coef_csv = REPO / "analysis" / "formula_search" / "results" / "fukui_correction_coeffs.csv"
    coef_df.to_csv(coef_csv, index=False)

    # Per-solvent breakdown for each correction
    print("\n=== Per-solvent MAE (kJ/mol): baseline vs corrected ===", flush=True)
    for r in results:
        tgt = r["target"]
        y = r["y"]
        base = baseline_predict(df, tgt, models)
        corr = r["pred_corr"]
        rows = []
        for solv, sub in df.groupby("solvent"):
            idx = sub.index
            mask = np.isfinite(corr[idx])
            if mask.sum() == 0:
                continue
            mae_b = float(np.mean(np.abs(y[idx] - base[idx])))
            mae_c = float(np.nanmean(np.abs(y[idx] - corr[idx])))
            rows.append({"solvent": solv, "epsilon": EPSILON_R[solv],
                         "MAE_base": mae_b, "MAE_corr": mae_c,
                         "delta": mae_c - mae_b, "n": int(mask.sum())})
        per_solv = pd.DataFrame(rows).sort_values("epsilon")
        print(f"\n--- {tgt} × (1 + a · {r['fukui']})   a_corr = {r['a_corr_opt']:.4f} ---")
        print(per_solv.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    out_csv = REPO / "analysis" / "formula_search" / "results" / "fukui_correction_tests.csv"
    summary.drop(columns=["formula"]).to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv.relative_to(REPO)}")
    print(f"Saved -> {coef_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
