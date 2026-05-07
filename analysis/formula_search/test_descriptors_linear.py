#!/usr/bin/env python3
"""Test mono/bi/tri/quadri linear regressions of IP/EA from {mu, eta, gamma, omega}.

Intercept fixed at 0. Coefficients optimised via Optuna (TPE) minimising MAE.
MAE reported in kJ/mol on the Acetonitrile subset of the mixed dataset
(merged_dataset.parquet) after the standard downstream filter excluding
DEA molecules + L-DOPA.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

N_WORKERS = max(1, (os.cpu_count() or 1) - 1)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"

EH_TO_KJ = 2625.499638
DESCRIPTORS = ["mu", "eta", "gamma", "omega"]
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}

N_TRIALS = 4000
SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def fit_optuna(X: np.ndarray, y: np.ndarray, n_trials: int = N_TRIALS) -> tuple[np.ndarray, float]:
    """Find coefficients minimising MAE (intercept = 0) via TPE.

    Warm-start with the OLS-through-origin solution, then explore a tight
    neighbourhood + a wider band so TPE can refine without losing the optimum.
    """
    ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    bounds = [max(50.0, 5.0 * abs(c)) for c in ols]

    def objective(trial: optuna.trial.Trial) -> float:
        beta = np.array([
            trial.suggest_float(f"b{i}", -bounds[i], bounds[i])
            for i in range(X.shape[1])
        ])
        return float(np.mean(np.abs(y - X @ beta)))

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Warm-start with OLS so the search starts from a reasonable basin
    study.enqueue_trial({f"b{i}": float(ols[i]) for i in range(X.shape[1])})
    # Tight neighbourhood seeds (jittered around OLS)
    rng = np.random.default_rng(SEED)
    for _ in range(10):
        jitter = rng.normal(0.0, 0.05, size=X.shape[1])
        seed_pt = ols * (1.0 + jitter)
        seed_pt = np.clip(seed_pt, [-b for b in bounds], bounds)
        study.enqueue_trial({f"b{i}": float(seed_pt[i]) for i in range(X.shape[1])})

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    beta = np.array([study.best_params[f"b{i}"] for i in range(X.shape[1])])
    return beta, study.best_value


def _fit_task(args: tuple) -> dict:
    """Worker entry point — must be picklable (top-level function)."""
    k, combo, tgt_name, X, y = args
    beta, mae = fit_optuna(X, y)
    row = {"k": k, "model": " + ".join(combo), "target": tgt_name, "MAE_kJmol": mae}
    for d, b in zip(combo, beta):
        row[f"coef_{d}"] = b
    return row


def main() -> None:
    df = pd.read_parquet(DATA)
    df = df[df.solvent == "Acetonitrile"].reset_index(drop=True)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])

    y_ip = df["IP"].values * EH_TO_KJ
    y_ea = df["EA"].values * EH_TO_KJ

    tasks = []
    for k in (1, 2, 3, 4):
        for combo in combinations(DESCRIPTORS, k):
            X = df[list(combo)].values
            for tgt_name, y in [("IP", y_ip), ("EA", y_ea)]:
                tasks.append((k, combo, tgt_name, X, y))

    print(f"Running {len(tasks)} fits on {N_WORKERS} workers, {N_TRIALS} trials each...")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_fit_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i:2d}/{len(tasks)}] k={r['k']} {r['target']:2s} {r['model']:30s} MAE={r['MAE_kJmol']:7.3f}")

    res = pd.DataFrame(rows)

    # Combined table: one row per (k, model), MAE_IP and MAE_EA columns
    pivot_mae = res.pivot_table(
        index=["k", "model"], columns="target", values="MAE_kJmol"
    ).reset_index()
    pivot_mae = pivot_mae.rename(columns={"IP": "MAE_IP_kJmol", "EA": "MAE_EA_kJmol"})
    pivot_mae = pivot_mae[["k", "model", "MAE_IP_kJmol", "MAE_EA_kJmol"]]
    pivot_mae = pivot_mae.sort_values(["k", "MAE_IP_kJmol"]).reset_index(drop=True)

    print(f"\nDataset: Acetonitrile, after filter DEA+L-DOPA")
    print(f"  n = {len(df)} conformers, {df.molecule.nunique()} molecules")
    print(f"Linear regression through origin, coefficients via Optuna TPE ({N_TRIALS} trials, seed={SEED})")
    print(f"Descriptors in eV; targets in kJ/mol\n")
    print(pivot_mae.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # Detailed coefficients per (k, model, target)
    coef_cols = [f"coef_{d}" for d in DESCRIPTORS]
    detail = res[["k", "model", "target", "MAE_kJmol"] + coef_cols].copy()
    detail = detail.sort_values(["k", "model", "target"]).reset_index(drop=True)

    print("\n\nCoefficients (intercept = 0):")
    print(detail.to_string(
        index=False,
        float_format=lambda x: f"{x:9.4f}" if pd.notna(x) else "       --",
        na_rep="       --",
    ))

    out_dir = REPO / "analysis" / "formula_search" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot_mae.to_csv(out_dir / "test_mu_eta_gamma_omega_acetonitrile_MAE.csv", index=False)
    detail.to_csv(out_dir / "test_mu_eta_gamma_omega_acetonitrile_coeffs.csv", index=False)
    print(f"\nSaved -> {(out_dir / 'test_mu_eta_gamma_omega_acetonitrile_MAE.csv').relative_to(REPO)}")
    print(f"Saved -> {(out_dir / 'test_mu_eta_gamma_omega_acetonitrile_coeffs.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
