#!/usr/bin/env python3
"""Test Padé[m/n] corrections in epsilon_r for the (mu + eta + omega) model.

Model:
    y(eps) = beta_mu(eps) * mu + beta_eta(eps) * eta + beta_omega(eps) * omega

where each beta_X(eps) is a Padé fraction:
    beta_X(eps) = (a0 + a1*eps + ... + am*eps^m) / (1 + b1*eps + ... + bn*eps^n)

Intercept fixed at 0 (no constant term independent of descriptors).
Coefficients optimised via Optuna TPE (warm-start from per-solvent OLS),
parallelised across (Padé order, target) tasks.

Filters: DEA molecules + L-DOPA excluded; all 10 solvents kept.
MAE reported in kJ/mol (global + per solvent for the best Padé per target).
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

PADE_ORDERS = [(m, n) for m in (1, 2, 3) for n in (1, 2, 3)]
N_TRIALS = 8000
EPS_MAX = 78.4  # Water — used to normalise epsilon to [0, 1] for stable Padé fits
SEED = 42
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)
DESCRIPTORS = ("mu", "eta", "omega")
STAB_TOL = 0.05  # |denom| > STAB_TOL required everywhere

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _per_solvent_ols(df: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    """OLS-through-origin coefficients (mu, eta, omega) per solvent.

    Used to seed the Optuna search and to set sensible parameter bounds.
    """
    out = {}
    for solv, sub in df.groupby("solvent"):
        X = sub[list(DESCRIPTORS)].values
        y = sub[target].values * EH_TO_KJ
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[solv] = beta
    return out


def _build_eps_powers(eps: np.ndarray, m_max: int, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute powers of eps once per fit. Returns (num_basis, den_basis).
    num_basis[j] = eps**j   for j in 0..m_max  (shape (m+1, N))
    den_basis[j] = eps**(j+1) for j in 0..n_max-1  (shape (n, N))
    """
    num_basis = np.stack([eps ** j for j in range(m_max + 1)], axis=0)
    den_basis = np.stack([eps ** j for j in range(1, n_max + 1)], axis=0)
    return num_basis, den_basis


def _pade_predict_fast(num_basis: np.ndarray, den_basis: np.ndarray,
                       descs: np.ndarray, params: np.ndarray, m: int, n: int) -> np.ndarray | None:
    """Vectorised Padé prediction.
    descs: (3, N) stacked (mu, eta, omega)
    params: 3*(m+n+1) flat — for each descriptor: a_0..a_m then b_1..b_n
    """
    n_per = m + n + 1
    P = params.reshape(3, n_per)
    A = P[:, : m + 1]            # (3, m+1)
    B = P[:, m + 1 :]            # (3, n)
    num = A @ num_basis          # (3, N)
    den = 1.0 + B @ den_basis    # (3, N)
    if np.any(np.abs(den) < STAB_TOL):
        return None
    return (num / den * descs).sum(axis=0)


def _fit_one(args: tuple) -> dict:
    target, m, n, X_eps, X_mu, X_eta, X_om, y, ols_per_solv = args
    n_per = m + n + 1
    n_params = 3 * n_per

    descs = np.stack([X_mu, X_eta, X_om], axis=0)
    eps_norm = X_eps / EPS_MAX  # all powers in [0, 1]
    num_basis, den_basis = _build_eps_powers(eps_norm, m, n)

    # With normalised eps, bounds are simple and uniform across powers
    coef_scale = max(abs(ols_per_solv["Acetonitrile"]).max(), 1.0)
    NUM_BND = max(3.0 * coef_scale, 200.0)
    DEN_BND = 5.0
    num_bounds = [NUM_BND] * (m + 1)
    den_bounds = [DEN_BND] * n

    def objective(trial: optuna.trial.Trial) -> float:
        params = np.empty(n_params)
        idx = 0
        for i in range(3):
            for j in range(m + 1):
                params[idx] = trial.suggest_float(f"a{i}_{j}", -num_bounds[j], num_bounds[j])
                idx += 1
            for j in range(n):
                params[idx] = trial.suggest_float(f"b{i}_{j+1}", -den_bounds[j], den_bounds[j])
                idx += 1
        pred = _pade_predict_fast(num_basis, den_basis, descs, params, m, n)
        if pred is None:
            return 1e9
        return float(np.mean(np.abs(y - pred)))

    # CMA-ES is much faster than TPE for continuous, smooth-ish objectives
    # in moderate dimension (~21 params for [3/3]). Warm-start mean = OLS-mean
    # constant Padé so the search starts in the right basin.
    mean_const = np.mean([ols_per_solv[s] for s in ols_per_solv], axis=0)
    x0 = {}
    for i in range(3):
        x0[f"a{i}_0"] = float(mean_const[i])
        for j in range(1, m + 1):
            x0[f"a{i}_{j}"] = 0.0
        for j in range(1, n + 1):
            x0[f"b{i}_{j}"] = 0.0
    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0, sigma0=NUM_BND / 6.0,
                                           n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Also enqueue the warm-start point explicitly (and a few jitters) so the
    # very first trial has a good baseline regardless of sampler behaviour.
    study.enqueue_trial(x0)
    rng = np.random.default_rng(SEED)
    ace_const = ols_per_solv.get("Acetonitrile", mean_const)
    x0_ace = dict(x0)
    for i in range(3):
        x0_ace[f"a{i}_0"] = float(ace_const[i])
    study.enqueue_trial(x0_ace)
    for _ in range(4):
        jitter = rng.normal(0.0, 0.1, size=3)
        seed_pt = dict(x0)
        for i in range(3):
            seed_pt[f"a{i}_0"] = float(mean_const[i] * (1.0 + jitter[i]))
        study.enqueue_trial(seed_pt)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = np.array([study.best_params.get(k, 0.0) for k in
                            [f"a{i}_{j}" for i in range(3) for j in range(m + 1)]
                            + [f"b{i}_{j}" for i in range(3) for j in range(1, n + 1)]])
    # The above ordering is not identical to the predictor ordering (which
    # interleaves a_i then b_i per descriptor). Reorder to match the predictor.
    coeffs_in_order = []
    for i in range(3):
        for j in range(m + 1):
            coeffs_in_order.append(study.best_params[f"a{i}_{j}"])
        for j in range(1, n + 1):
            coeffs_in_order.append(study.best_params[f"b{i}_{j}"])
    best_params = np.array(coeffs_in_order)

    pred = _pade_predict_fast(num_basis, den_basis, descs, best_params, m, n)
    return {
        "target": target,
        "m": m,
        "n": n,
        "MAE_global": float(np.mean(np.abs(y - pred))),
        "params": best_params,
        "pred": pred,
        "y": y,
    }


def main() -> None:
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)

    n_per_solv = df.solvent.value_counts().sort_index()
    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules")
    print(f"Per-solvent counts:\n{n_per_solv.to_string()}\n")

    eps = df["epsilon"].values.astype(float)
    mu = df["mu"].values
    eta = df["eta"].values
    om = df["omega"].values

    tasks = []
    for target in ("IP", "EA"):
        y = df[target].values * EH_TO_KJ
        ols_ps = _per_solvent_ols(df, target)
        for m, n in PADE_ORDERS:
            tasks.append((target, m, n, eps, mu, eta, om, y, ols_ps))

    print(f"Running {len(tasks)} Padé fits on {N_WORKERS} workers, {N_TRIALS} trials each...")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_fit_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['target']:2s} Padé[{r['m']}/{r['n']}]  MAE = {r['MAE_global']:7.3f}", flush=True)

    # Summary table per Padé order
    rows = []
    for (m, n) in PADE_ORDERS:
        row = {"m": m, "n": n, "Pade": f"[{m}/{n}]"}
        for tgt in ("IP", "EA"):
            r = next(x for x in results if x["target"] == tgt and x["m"] == m and x["n"] == n)
            row[f"MAE_{tgt}_kJmol"] = r["MAE_global"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary[["Pade", "m", "n", "MAE_IP_kJmol", "MAE_EA_kJmol"]]

    print("\nGlobal MAE per Padé order (intercept = 0, fit over all 10 solvents):\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # Per-solvent MAE for the best Padé order per target
    print("\nPer-solvent MAE (kJ/mol) — best Padé per target:\n")
    for tgt in ("IP", "EA"):
        best = min((r for r in results if r["target"] == tgt), key=lambda r: r["MAE_global"])
        residuals = np.abs(best["y"] - best["pred"])
        per_solv = pd.DataFrame({
            "solvent": df["solvent"].values,
            "epsilon": eps,
            "abs_err": residuals,
        })
        agg = per_solv.groupby("solvent").agg(
            MAE_kJmol=("abs_err", "mean"),
            n=("abs_err", "size"),
        ).reset_index().sort_values("MAE_kJmol")
        print(f"=== {tgt}  best Padé[{best['m']}/{best['n']}]  (global MAE = {best['MAE_global']:.3f} kJ/mol) ===")
        print(agg.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        print()

    out_dir = REPO / "analysis" / "formula_search" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "test_pade_solvent_MAE.csv", index=False)

    # Also persist coefficients for reproducibility
    coef_rows = []
    for r in results:
        m, n = r["m"], r["n"]
        labels = []
        for desc in DESCRIPTORS:
            for j in range(m + 1):
                labels.append(f"a_{desc}_{j}")
            for j in range(1, n + 1):
                labels.append(f"b_{desc}_{j}")
        for lbl, val in zip(labels, r["params"]):
            coef_rows.append({"target": r["target"], "Pade": f"[{m}/{n}]",
                              "coef": lbl, "value": float(val)})
    pd.DataFrame(coef_rows).to_csv(out_dir / "test_pade_solvent_coeffs.csv", index=False)

    print(f"Saved -> {(out_dir / 'test_pade_solvent_MAE.csv').relative_to(REPO)}")
    print(f"Saved -> {(out_dir / 'test_pade_solvent_coeffs.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
