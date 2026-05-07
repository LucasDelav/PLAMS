#!/usr/bin/env python3
"""Padé[m/n] solvent corrections for the (mu + eta + omega) model — v2.

Two-step approach (avoids fragile joint nonlinear optimisation):
  1. Per-solvent OLS-through-origin → (beta_mu, beta_eta, beta_omega) at each
     of the 10 epsilon_r values.
  2. Fit a Padé[m/n] to each coefficient series (10 points × 3 coefficients ×
     2 targets = 6 small 1D nonlinear fits per Padé order). Coefficients via
     Optuna CMA-ES (intercept = 0 at the *linear* level — i.e. no constant
     additive term in the original model; the Padé numerators include their
     own a_0).
  3. Evaluate the combined model y_pred = beta_mu(eps)·mu + ... on every
     conformer and report MAE in kJ/mol.

Filters: DEA + L-DOPA excluded; all 10 solvents kept.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}

EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
EPS_MAX = 78.4
PADE_ORDERS = [(m, n) for m in (1, 2, 3) for n in (1, 2, 3)]
N_TRIALS = 6000
SEED = 42
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)
DESCRIPTORS = ("mu", "eta", "omega")
TARGETS = ("IP", "EA")
STAB_TOL = 0.05

optuna.logging.set_verbosity(optuna.logging.WARNING)


def per_solvent_ols(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Fit OLS-through-origin per solvent. Returns DataFrame with eps and betas."""
    rows = []
    for solv, sub in df.groupby("solvent"):
        X = sub[list(DESCRIPTORS)].values
        y = sub[target].values * EH_TO_KJ
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append({"solvent": solv, "epsilon": EPSILON_R[solv],
                     "beta_mu": beta[0], "beta_eta": beta[1], "beta_omega": beta[2],
                     "n": len(sub)})
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


def pade_eval(eps_norm: np.ndarray, num_coefs: np.ndarray, den_coefs: np.ndarray
              ) -> tuple[np.ndarray, bool]:
    """Evaluate (a0 + a1·x + ...)/(1 + b1·x + ...). Returns (values, stable)."""
    num = sum(a * eps_norm ** j for j, a in enumerate(num_coefs))
    den = 1.0 + sum(b * eps_norm ** (j + 1) for j, b in enumerate(den_coefs))
    if np.any(np.abs(den) < STAB_TOL):
        return num / den, False
    return num / den, True


def fit_pade_1d(eps_values: np.ndarray, beta_values: np.ndarray, weights: np.ndarray,
                m: int, n: int, n_trials: int = N_TRIALS) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a Padé[m/n] to a 1D series (eps -> beta). Returns (num_coefs, den_coefs, mae_on_points)."""
    eps_norm = eps_values / EPS_MAX
    coef_scale = max(np.abs(beta_values).max(), 1.0)
    NUM_BND = max(3.0 * coef_scale, 100.0)
    DEN_BND = 5.0
    n_params = (m + 1) + n

    def objective(trial):
        params = []
        for j in range(m + 1):
            params.append(trial.suggest_float(f"a{j}", -NUM_BND, NUM_BND))
        for j in range(n):
            params.append(trial.suggest_float(f"b{j+1}", -DEN_BND, DEN_BND))
        a = np.array(params[: m + 1])
        b = np.array(params[m + 1 :])
        beta_pred, ok = pade_eval(eps_norm, a, b)
        if not ok:
            return 1e9
        # Weighted MSE — minimised by L2, robust then converted to MAE
        return float(np.sum(weights * (beta_values - beta_pred) ** 2) / weights.sum())

    # Warm-start: constant Padé at the weighted mean
    x0 = {f"a0": float(np.average(beta_values, weights=weights))}
    for j in range(1, m + 1):
        x0[f"a{j}"] = 0.0
    for j in range(1, n + 1):
        x0[f"b{j}"] = 0.0

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0, sigma0=NUM_BND / 4.0,
                                           n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(x0)

    # Polynomial warm starts (b = 0)
    if m >= 1:
        # Linear in eps_norm fit via OLS on the 10 points
        try:
            poly_coefs = np.polyfit(eps_norm, beta_values, deg=m, w=weights)[::-1]  # ascending
            x0_poly = {f"a{j}": float(poly_coefs[j]) if j < len(poly_coefs) else 0.0 for j in range(m + 1)}
            for j in range(1, n + 1):
                x0_poly[f"b{j}"] = 0.0
            study.enqueue_trial(x0_poly)
        except np.linalg.LinAlgError:
            pass

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    a = np.array([study.best_params[f"a{j}"] for j in range(m + 1)])
    b = np.array([study.best_params[f"b{j+1}"] for j in range(n)])
    beta_pred, _ = pade_eval(eps_norm, a, b)
    mae_pts = float(np.average(np.abs(beta_values - beta_pred), weights=weights))
    return a, b, mae_pts


def _fit_one(args):
    target, m, n, persol_df = args
    eps = persol_df["epsilon"].values
    weights = persol_df["n"].values.astype(float)
    fits = {}
    for desc in DESCRIPTORS:
        beta_vals = persol_df[f"beta_{desc}"].values
        a, b, mae_pts = fit_pade_1d(eps, beta_vals, weights, m, n)
        fits[desc] = {"a": a, "b": b, "mae_on_persolv_points_kJ": mae_pts}
    return {"target": target, "m": m, "n": n, "fits": fits}


def evaluate_full(df: pd.DataFrame, target: str, fits: dict) -> tuple[float, np.ndarray]:
    """Apply per-coefficient Padé fits to the full dataset."""
    eps_norm = (df["epsilon"].values / EPS_MAX)
    pred = np.zeros(len(df))
    for desc in DESCRIPTORS:
        a, b = fits[desc]["a"], fits[desc]["b"]
        beta, ok = pade_eval(eps_norm, a, b)
        if not ok:
            beta = np.where(np.abs(beta) > 1e6, 0.0, beta)  # safety net
        pred += beta * df[desc].values
    y = df[target].values * EH_TO_KJ
    return float(np.mean(np.abs(y - pred))), pred


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)

    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)

    persol_data = {tgt: per_solvent_ols(df, tgt) for tgt in TARGETS}

    print("\nPer-solvent OLS-through-origin coefficients (lower bound for Padé):", flush=True)
    for tgt in TARGETS:
        print(f"\n  --- {tgt} ---")
        print(persol_data[tgt].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    tasks = []
    for tgt in TARGETS:
        for (m, n) in PADE_ORDERS:
            tasks.append((tgt, m, n, persol_data[tgt]))

    print(f"\nFitting {len(tasks)} Padé orders on {N_WORKERS} workers, {N_TRIALS} trials each...", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_fit_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            mae_full, pred = evaluate_full(df, r["target"], r["fits"])
            r["MAE_full_kJmol"] = mae_full
            r["pred"] = pred
            results.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['target']:2s} Padé[{r['m']}/{r['n']}]  "
                  f"MAE_full = {mae_full:7.3f}", flush=True)

    # Summary table
    rows = []
    for (m, n) in PADE_ORDERS:
        row = {"Pade": f"[{m}/{n}]", "m": m, "n": n}
        for tgt in TARGETS:
            r = next(x for x in results if x["target"] == tgt and x["m"] == m and x["n"] == n)
            row[f"MAE_{tgt}_kJmol"] = r["MAE_full_kJmol"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary[["Pade", "m", "n", "MAE_IP_kJmol", "MAE_EA_kJmol"]]

    print("\n=== Global MAE per Padé order, model y = beta_mu(eps)·mu + beta_eta(eps)·eta + beta_omega(eps)·omega (intercept=0) ===\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # Per-solvent breakdown for best Padé per target
    print("\nPer-solvent MAE for the best Padé per target:\n")
    for tgt in TARGETS:
        best = min((r for r in results if r["target"] == tgt), key=lambda r: r["MAE_full_kJmol"])
        y = df[tgt].values * EH_TO_KJ
        residuals = np.abs(y - best["pred"])
        agg = pd.DataFrame({
            "solvent": df["solvent"].values,
            "epsilon": df["epsilon"].values,
            "abs_err": residuals,
        }).groupby(["solvent", "epsilon"]).agg(
            MAE_kJmol=("abs_err", "mean"),
            n=("abs_err", "size"),
        ).reset_index().sort_values("epsilon")
        print(f"=== {tgt}  best Padé[{best['m']}/{best['n']}]  (global MAE = {best['MAE_full_kJmol']:.3f} kJ/mol) ===")
        print(agg.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        print()

    # Save
    out_dir = REPO / "analysis" / "formula_search" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "test_pade_solvent_v2_MAE.csv", index=False)

    coef_rows = []
    for r in results:
        m, n = r["m"], r["n"]
        for desc, fit in r["fits"].items():
            for j, av in enumerate(fit["a"]):
                coef_rows.append({"target": r["target"], "Pade": f"[{m}/{n}]",
                                  "descriptor": desc, "coef": f"a_{j}", "value": float(av)})
            for j, bv in enumerate(fit["b"], start=1):
                coef_rows.append({"target": r["target"], "Pade": f"[{m}/{n}]",
                                  "descriptor": desc, "coef": f"b_{j}", "value": float(bv)})
    pd.DataFrame(coef_rows).to_csv(out_dir / "test_pade_solvent_v2_coeffs.csv", index=False)
    print(f"Saved -> {(out_dir / 'test_pade_solvent_v2_MAE.csv').relative_to(REPO)}")
    print(f"Saved -> {(out_dir / 'test_pade_solvent_v2_coeffs.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
