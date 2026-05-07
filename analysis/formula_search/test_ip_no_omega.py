#!/usr/bin/env python3
"""IP model without omega: just (mu, eta) with Padé[1/1] g_Born + Fukui dual_min.

Same structure as the current best IP model but dropping omega.

Model:
    y(eps) = [beta_mu(eps)*mu + beta_eta(eps)*eta] * (1 + a_corr * f_min^dual_noO)

7 parameters total (3 per descriptor × 2 + 1 correction).

Comparison points:
  Padé[1/1] g_Born seul (mu, eta) — to be computed here as a sub-baseline
  Padé[1/1] g_Born × Fukui (mu, eta, omega) — current best: MAE 3.449, RMSE 6.403
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
DESCRIPTORS = ("mu", "eta")
SEED = 42
N_TRIALS = 12000
STAB_TOL = 0.05
optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAM_NAMES = [f"{p}_{d}" for d in DESCRIPTORS for p in ("a0", "a1", "b1")] + ["a_corr"]


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def fukui_dual_min_no_o(values, symbols):
    arr = np.asarray(values, dtype=float)
    syms = np.asarray(symbols)
    mask = syms != "O"
    if not mask.any():
        return float("nan")
    return float(arr[mask].min())


def load_fukui(df):
    rows = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        if not path.exists():
            rows.append(np.nan); continue
        try:
            with open(path) as f:
                pred = json.load(f)
            ns = pred["states"]["neutral_sp"]
            syms = parse_xyz_symbols(ns["xyz"])
            rows.append(fukui_dual_min_no_o(ns["cdft"]["dual"], syms))
        except Exception:
            rows.append(np.nan)
    return np.array(rows)


def predict(eps, mu, eta, fukui, params):
    g = g_born(eps)
    out = np.zeros_like(eps)
    for d, x in zip(DESCRIPTORS, (mu, eta)):
        a0, a1, b1 = params[f"a0_{d}"], params[f"a1_{d}"], params[f"b1_{d}"]
        den = 1.0 + b1 * g
        if np.any(np.abs(den) < STAB_TOL):
            return None
        out += (a0 + a1 * g) / den * x
    return out * (1.0 + params["a_corr"] * fukui)


def warmstart_from_omega():
    """Reuse the (mu, eta) coefs from the previous (mu+eta+omega) IP fit."""
    df = pd.read_csv(COEF_PREV)
    df = df[(df.target == "IP") & (df.fukui == "f2_min_noO")].set_index("param")
    x0 = {}
    for p in ("a0", "a1", "b1"):
        x0[f"{p}_mu"] = float(df.loc[f"{p}_mu", "value"])
        x0[f"{p}_eta"] = float(df.loc[f"{p}_eta", "value"])
    x0["a_corr"] = float(df.loc["a_corr", "value"])
    return x0


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)

    df["f2_min_noO"] = load_fukui(df)
    valid = np.isfinite(df.f2_min_noO.values)
    print(f"  Fukui complete on {valid.sum()}/{len(df)} conformers", flush=True)

    eps = df["epsilon"].values
    mu = df["mu"].values
    eta = df["eta"].values
    fukui = df["f2_min_noO"].values
    y = df["IP"].values * EH_TO_KJ

    eps_v, mu_v, eta_v, f_v, y_v = eps[valid], mu[valid], eta[valid], fukui[valid], y[valid]

    x0 = warmstart_from_omega()
    print(f"\nWarm-start (carries (μ, η) coefs from previous (μ+η+ω) fit):")
    for k, v in x0.items():
        print(f"  {k:10s} = {v:+.4f}")

    scale_num = max(max(abs(v) for k, v in x0.items() if k.startswith(("a0_", "a1_"))), 50.0)
    bounds = {}
    for d in DESCRIPTORS:
        bounds[f"a0_{d}"] = (-3 * scale_num, 3 * scale_num)
        bounds[f"a1_{d}"] = (-5 * scale_num, 5 * scale_num)
        bounds[f"b1_{d}"] = (-5.0, 5.0)
    bounds["a_corr"] = (-50.0, 50.0)

    def objective(trial):
        params = {n: trial.suggest_float(n, *bounds[n]) for n in PARAM_NAMES}
        pr = predict(eps_v, mu_v, eta_v, f_v, params)
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
    print(f"\nOptimising {N_TRIALS} trials...", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {n: study.best_params[n] for n in PARAM_NAMES}
    pred_full = predict(eps, mu, eta, fukui, best)
    pred_full = np.where(valid, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))

    # Also recompute the no-Fukui (mu,eta) Padé[1/1]g_Born baseline for reference
    # by setting a_corr=0 in the same params (can't beat the joint refit but gives
    # a reference of what the descriptor pool can do without correction).
    no_corr = dict(best); no_corr["a_corr"] = 0.0
    pred_noc = predict(eps, mu, eta, fukui, no_corr)
    pred_noc = np.where(valid, pred_noc, np.nan)
    mae_noc = float(np.nanmean(np.abs(y - pred_noc)))
    rmse_noc = float(np.sqrt(np.nanmean((y - pred_noc) ** 2)))

    print(f"\n=== IP without omega — joint refit ===")
    print(f"  Model: (μ, η) × Padé[1/1] g_Born × (1 + a·f²_min_noO),  7 params")
    print(f"  MAE  = {mae:.3f} kJ/mol")
    print(f"  RMSE = {rmse:.3f} kJ/mol")
    print(f"\n  (Same coeffs but without correction, a_corr=0):  MAE = {mae_noc:.3f}, RMSE = {rmse_noc:.3f}")

    print(f"\n=== Optimised coefficients ===")
    for k in PARAM_NAMES:
        print(f"  {k:10s} = {best[k]:+.4f}")

    print(f"\n=== Comparison with reference IP models ===")
    refs = [
        ("(μ, η, ω) × Padé g_Born × (1 + a·f²_min_noO)  [current best, 10 params]", 3.449, 6.403),
        ("(μ, η, ω) × Padé g_Born  [no Fukui, 9 params]",                              4.387, 7.306),
        ("(μ, η)    × Padé g_Born × (1 + a·f²_min_noO)  [this test, 7 params]",       mae,   rmse),
    ]
    print(f"  {'Model':<70s} MAE     RMSE")
    for name, m, r in refs:
        print(f"  {name:<70s} {m:6.3f}  {r:6.3f}")

    print(f"\n=== Per-solvent breakdown ===")
    rows = []
    for solv, sub in df.groupby("solvent"):
        idx = sub.index
        mask = np.isfinite(pred_full[idx])
        if mask.sum() == 0: continue
        e = float(np.nanmean(np.abs(y[idx] - pred_full[idx])))
        rows.append({"solvent": solv, "epsilon": EPSILON_R[solv],
                     "MAE": e, "n": int(mask.sum())})
    print(pd.DataFrame(rows).sort_values("epsilon").to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    out_csv = REPO / "analysis" / "formula_search" / "results" / "ip_no_omega_test.csv"
    pd.DataFrame([{"target": "IP", "model": "(mu+eta) × Padé[1/1]g_Born × (1+a·f²_min_noO)",
                   "MAE_kJmol": mae, "RMSE_kJmol": rmse,
                   "MAE_no_corr": mae_noc, "RMSE_no_corr": rmse_noc, **best}]).to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
