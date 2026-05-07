#!/usr/bin/env python3
"""Test IP model with nucleophilicity nu = 1/omega = 2*eta/mu^2 instead of omega.

Rationale: omega = mu^2/(2*eta) is Parr's electrophilicity, physically suited
to EA (electron acceptance). For IP (electron loss) the dual quantity 1/omega
is a candidate proxy for nucleophilicity.

Model:
    y(eps) = beta_mu(eps)*mu + beta_eta(eps)*eta + beta_nu(eps)*nu
           × (1 + a_corr * f_min^dual_noO)

with beta_X(eps) = (a0 + a1*g)/(1 + b1*g),  g = (eps-1)/(2*eps+1).

Joint refit of all 10 parameters via Optuna CMA-ES, multi-start on a_corr,
warm-started from the saved (mu+eta+omega) coefficients (with omega-coefs
re-used as starting points for nu-coefs — they're not the same descriptor
but the order of magnitude is comparable).

Compares to current best IP model: MAE 3.449 / RMSE 6.403.
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
COEF_PREV = REPO / "analysis" / "formula_search" / "results" / "fukui_correction_coeffs.csv"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
DESCRIPTORS = ("mu", "eta", "nu")  # nu replaces omega for IP
SEED = 42
N_TRIALS = 12000
STAB_TOL = 0.05
optuna.logging.set_verbosity(optuna.logging.WARNING)


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def fukui_no_o(values, symbols):
    arr = np.asarray(values, dtype=float)
    syms = np.asarray(symbols)
    mask = syms != "O"
    if not mask.any():
        return float("nan"), float("nan")
    sub = arr[mask]
    return float(sub.max()), float(sub.min())


def load_fukui_dual_min_noO(df):
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
            _, dmin = fukui_no_o(ns["cdft"]["dual"], syms)
            rows.append(dmin)
        except Exception:
            rows.append(np.nan)
    return np.array(rows)


PARAM_NAMES = [f"{p}_{d}" for d in DESCRIPTORS for p in ("a0", "a1", "b1")] + ["a_corr"]


def predict(eps, mu, eta, nu, fukui, params):
    g = g_born(eps)
    out = np.zeros_like(eps)
    for d, x in zip(DESCRIPTORS, (mu, eta, nu)):
        a0, a1, b1 = params[f"a0_{d}"], params[f"a1_{d}"], params[f"b1_{d}"]
        den = 1.0 + b1 * g
        if np.any(np.abs(den) < STAB_TOL):
            return None
        out += (a0 + a1 * g) / den * x
    return out * (1.0 + params["a_corr"] * fukui)


def warmstart_from_omega():
    """Take the previously-fitted IP coefs (mu+eta+omega + Fukui dual_min) and
    use them as the starting point — the omega coefs initialise the nu coefs."""
    df = pd.read_csv(COEF_PREV)
    df = df[(df.target == "IP") & (df.fukui == "f2_min_noO")].set_index("param")
    x0 = {}
    for p in ("a0", "a1", "b1"):
        x0[f"{p}_mu"] = float(df.loc[f"{p}_mu", "value"])
        x0[f"{p}_eta"] = float(df.loc[f"{p}_eta", "value"])
        x0[f"{p}_nu"] = float(df.loc[f"{p}_omega", "value"])
    x0["a_corr"] = float(df.loc["a_corr", "value"])
    return x0


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["nu"] = 1.0 / df["omega"]   # = 2*eta/mu^2
    df["epsilon"] = df["solvent"].map(EPSILON_R)

    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)
    print(f"omega range: [{df.omega.min():.3f}, {df.omega.max():.3f}] eV  (Parr electrophilicity)")
    print(f"nu    range: [{df.nu.min():.3f}, {df.nu.max():.3f}] 1/eV  (1/omega proposed nucleophilicity)")

    print("\nLoading f_min^dual_noO from predict-dg JSONs...", flush=True)
    df["f2_min_noO"] = load_fukui_dual_min_noO(df)
    valid = np.isfinite(df.f2_min_noO.values)
    print(f"  Complete on {valid.sum()}/{len(df)} conformers", flush=True)

    eps = df["epsilon"].values
    mu = df["mu"].values
    eta = df["eta"].values
    nu = df["nu"].values
    fukui = df["f2_min_noO"].values
    y = df["IP"].values * EH_TO_KJ

    eps_v, mu_v, eta_v, nu_v, f_v, y_v = (
        eps[valid], mu[valid], eta[valid], nu[valid], fukui[valid], y[valid]
    )

    x0 = warmstart_from_omega()
    print(f"\nWarm-start from previous (mu+eta+omega) IP fit:")
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
        pr = predict(eps_v, mu_v, eta_v, nu_v, f_v, params)
        if pr is None or np.any(~np.isfinite(pr)):
            return 1e9
        return float(np.mean(np.abs(y_v - pr)))

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0, sigma0=10.0,
                                            n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(x0)
    for a_seed in (-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0):
        x0_a = dict(x0)
        x0_a["a_corr"] = a_seed
        study.enqueue_trial(x0_a)
    print(f"\nOptimising {N_TRIALS} trials...", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {n: study.best_params[n] for n in PARAM_NAMES}
    pred_full = predict(eps, mu, eta, nu, fukui, best)
    pred_full = np.where(valid, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))

    # Reference: previous IP model with omega
    print(f"\n=== Comparison ===")
    print(f"  Previous IP (mu, eta, omega) × (1 + a·f²_min_noO):  MAE = 3.449,  RMSE = 6.403")
    print(f"  New      IP (mu, eta, nu=1/omega) × (1 + a·f²_min_noO):  MAE = {mae:.3f},  RMSE = {rmse:.3f}")
    print(f"  Δ MAE = {mae - 3.449:+.3f},  Δ RMSE = {rmse - 6.403:+.3f}")

    print(f"\n=== Optimised coefficients ({len(best)} params, intercept = 0) ===")
    for k in PARAM_NAMES:
        print(f"  {k:10s} = {best[k]:+.4f}")

    # Per-solvent breakdown
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

    out_csv = REPO / "analysis" / "formula_search" / "results" / "ip_nucleophilicity_test.csv"
    pd.DataFrame([{"target": "IP", "model": "(mu+eta+nu) × Padé[1/1]g_Born × (1+a·f²_min_noO)",
                   "MAE_kJmol": mae, "RMSE_kJmol": rmse, **best}]).to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
