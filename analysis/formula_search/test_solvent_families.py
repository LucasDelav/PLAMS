#!/usr/bin/env python3
"""Benchmark 5 solvent-correction families for the (mu+eta+omega) model.

Families (all expressed as beta_X(eps), to be applied per-descriptor):
    born     : beta(eps) = a + b * (eps-1)/(2eps+1)         [2 params]
    onsager  : beta(eps) = a + b * (eps-1)/(eps+2)          [2 params]
    log      : beta(eps) = a + b * ln(eps)                  [2 params]
    power    : beta(eps) = a + b * eps**(-c)                [3 params]
    pade11_g : beta(eps) = (a0 + a1*g)/(1 + b1*g), g=Born   [3 params]
    pade33   : Padé[3/3] in eps_norm, reference from v2     [7 params]

Two fitting protocols per family:
  (A) per-coefficient 1D fit on the 10 per-solvent OLS values (CMA-ES, MSE)
  (B) joint fit minimising the global MAE on 1336 conformers (CMA-ES, MAE)

Reports global MAE_IP and MAE_EA in kJ/mol for each (family, protocol)
combination, plus per-solvent breakdown for the best (family, protocol) per target.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DESCRIPTORS = ("mu", "eta", "omega")
TARGETS = ("IP", "EA")
SEED = 42
N_TRIALS_1D = 4000        # per-coefficient 1D fit
N_TRIALS_JOINT = 8000     # joint global-MAE fit
STAB_TOL = 0.05
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FAMILIES = ["born", "onsager", "log", "power", "pade11_g", "pade33"]


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)
def g_onsager(eps): return (eps - 1.0) / (eps + 2.0)


# Each family: (n_params, eval(eps, params)->beta or None on instability,
#               x0(beta_mean)->dict of param init,
#               param spec [(name, lo, hi)])
def make_family(name: str, beta_scale: float):
    """Return dict with 'n_params', 'eval', 'x0', 'spec'."""
    sc = max(beta_scale, 50.0)
    if name == "born":
        return {
            "n_params": 2, "spec": [("a", -3*sc, 3*sc), ("b", -3*sc, 3*sc)],
            "eval": lambda eps, p: p[0] + p[1] * g_born(eps),
            "x0": lambda mean: {"a": float(mean), "b": 0.0},
        }
    if name == "onsager":
        return {
            "n_params": 2, "spec": [("a", -3*sc, 3*sc), ("b", -3*sc, 3*sc)],
            "eval": lambda eps, p: p[0] + p[1] * g_onsager(eps),
            "x0": lambda mean: {"a": float(mean), "b": 0.0},
        }
    if name == "log":
        return {
            "n_params": 2, "spec": [("a", -3*sc, 3*sc), ("b", -3*sc, 3*sc)],
            "eval": lambda eps, p: p[0] + p[1] * np.log(eps),
            "x0": lambda mean: {"a": float(mean), "b": 0.0},
        }
    if name == "power":
        # beta(eps) = a + b * eps^(-c); c in (0.05, 5) to avoid degeneracy
        return {
            "n_params": 3,
            "spec": [("a", -3*sc, 3*sc), ("b", -5*sc, 5*sc), ("c", 0.05, 5.0)],
            "eval": lambda eps, p: p[0] + p[1] * (eps ** (-p[2])),
            "x0": lambda mean: {"a": float(mean), "b": 0.0, "c": 1.0},
        }
    if name == "pade11_g":
        # beta(eps) = (a0 + a1*g) / (1 + b1*g),  g in [0, 0.49] (Born saturated)
        def _eval(eps, p):
            g = g_born(eps)
            num = p[0] + p[1] * g
            den = 1.0 + p[2] * g
            if np.any(np.abs(den) < STAB_TOL):
                return None
            return num / den
        return {
            "n_params": 3,
            "spec": [("a0", -3*sc, 3*sc), ("a1", -5*sc, 5*sc), ("b1", -5.0, 5.0)],
            "eval": _eval,
            "x0": lambda mean: {"a0": float(mean), "a1": 0.0, "b1": 0.0},
        }
    if name == "pade33":
        # Padé[3/3] in eps_norm = eps / EPS_MAX, reference baseline
        def _eval(eps, p):
            x = eps / EPS_MAX
            num = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3
            den = 1.0 + p[4]*x + p[5]*x**2 + p[6]*x**3
            if np.any(np.abs(den) < STAB_TOL):
                return None
            return num / den
        return {
            "n_params": 7,
            "spec": ([("a0", -3*sc, 3*sc), ("a1", -5*sc, 5*sc),
                      ("a2", -5*sc, 5*sc), ("a3", -5*sc, 5*sc),
                      ("b1", -5.0, 5.0), ("b2", -5.0, 5.0), ("b3", -5.0, 5.0)]),
            "eval": _eval,
            "x0": lambda mean: {"a0": float(mean), "a1": 0.0, "a2": 0.0, "a3": 0.0,
                                "b1": 0.0, "b2": 0.0, "b3": 0.0},
        }
    raise ValueError(name)


# === Per-coefficient 1D fit (protocol A) ==================================

def per_solvent_ols(df, target):
    rows = []
    for solv, sub in df.groupby("solvent"):
        X = sub[list(DESCRIPTORS)].values
        y = sub[target].values * EH_TO_KJ
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append({"solvent": solv, "epsilon": EPSILON_R[solv],
                     "beta_mu": beta[0], "beta_eta": beta[1], "beta_omega": beta[2],
                     "n": len(sub)})
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


def fit_1d(eps, betas, weights, family, beta_scale, n_trials=N_TRIALS_1D):
    fam = make_family(family, beta_scale)
    spec = fam["spec"]
    mean_b = float(np.average(betas, weights=weights))

    def objective(trial):
        params = np.array([trial.suggest_float(name, lo, hi) for (name, lo, hi) in spec])
        pred = fam["eval"](eps, params)
        if pred is None or np.any(~np.isfinite(pred)):
            return 1e9
        return float(np.sum(weights * (betas - pred) ** 2) / weights.sum())

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=fam["x0"](mean_b),
                                            sigma0=beta_scale, n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(fam["x0"](mean_b))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = np.array([study.best_params[name] for (name, _, _) in spec])
    return p, fam["eval"]


def evaluate_full(df, target, fits):
    """fits = {desc: (params, eval_fn)}"""
    eps = df["epsilon"].values
    pred = np.zeros(len(df))
    for desc in DESCRIPTORS:
        p, eval_fn = fits[desc]
        beta = eval_fn(eps, p)
        pred += beta * df[desc].values
    y = df[target].values * EH_TO_KJ
    return float(np.mean(np.abs(y - pred))), pred


# === Joint global-MAE fit (protocol B) ====================================

def fit_joint(df, target, family, beta_scales, n_trials=N_TRIALS_JOINT):
    """Optimise all 3 descriptor families jointly to minimise global MAE."""
    fam_specs = [make_family(family, sc) for sc in beta_scales]
    eps = df["epsilon"].values
    mu = df["mu"].values
    eta = df["eta"].values
    om = df["omega"].values
    descs = (mu, eta, om)
    y = df[target].values * EH_TO_KJ

    # Build full param spec — prefix per descriptor index
    full_spec = []
    for i, fam in enumerate(fam_specs):
        for (name, lo, hi) in fam["spec"]:
            full_spec.append((f"d{i}_{name}", lo, hi))
    n_params = len(full_spec)

    # Warm start from per-coefficient 1D solution
    persol = per_solvent_ols(df, target)
    epsv = persol["epsilon"].values
    weights = persol["n"].values.astype(float)
    x0 = {}
    p_per_desc = []
    for i, desc in enumerate(DESCRIPTORS):
        p1d, _ = fit_1d(epsv, persol[f"beta_{desc}"].values, weights, family,
                        beta_scales[i], n_trials=N_TRIALS_1D)
        p_per_desc.append(p1d)
        for j, (name, _, _) in enumerate(fam_specs[i]["spec"]):
            x0[f"d{i}_{name}"] = float(p1d[j])

    def split_params(flat):
        out = []
        idx = 0
        for fam in fam_specs:
            k = fam["n_params"]
            out.append(flat[idx:idx + k])
            idx += k
        return out

    def predict(flat):
        parts = split_params(flat)
        pred = np.zeros(len(df))
        for fam, p, x in zip(fam_specs, parts, descs):
            beta = fam["eval"](eps, p)
            if beta is None or np.any(~np.isfinite(beta)):
                return None
            pred = pred + beta * x
        return pred

    def objective(trial):
        flat = np.array([trial.suggest_float(name, lo, hi) for (name, lo, hi) in full_spec])
        pred = predict(flat)
        if pred is None:
            return 1e9
        return float(np.mean(np.abs(y - pred)))

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0, sigma0=10.0,
                                            n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(x0)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    flat = np.array([study.best_params[name] for (name, _, _) in full_spec])
    pred = predict(flat)
    mae = float(np.mean(np.abs(y - pred)))
    return mae, pred, flat, p_per_desc


# === Worker entry points ===================================================

def _task_protA(args):
    target, family, df_pkl = args
    df = pd.read_parquet(df_pkl) if isinstance(df_pkl, (str, Path)) else df_pkl
    persol = per_solvent_ols(df, target)
    eps = persol["epsilon"].values
    weights = persol["n"].values.astype(float)
    fits = {}
    for desc in DESCRIPTORS:
        b = persol[f"beta_{desc}"].values
        scale = max(np.abs(b).max(), 1.0)
        p, eval_fn = fit_1d(eps, b, weights, family, scale)
        fits[desc] = (p, eval_fn)
    mae, pred = evaluate_full(df, target, fits)
    return {"target": target, "family": family, "protocol": "A",
            "MAE_kJmol": mae, "pred": pred,
            "params": {desc: list(fits[desc][0]) for desc in DESCRIPTORS}}


def _task_protB(args):
    target, family, df_pkl = args
    df = pd.read_parquet(df_pkl) if isinstance(df_pkl, (str, Path)) else df_pkl
    persol = per_solvent_ols(df, target)
    scales = [max(np.abs(persol[f"beta_{d}"].values).max(), 1.0) for d in DESCRIPTORS]
    mae, pred, flat, _ = fit_joint(df, target, family, scales)
    return {"target": target, "family": family, "protocol": "B",
            "MAE_kJmol": mae, "pred": pred, "params_flat": list(flat)}


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)

    # Save filtered df to a temp parquet for workers
    tmp_parq = REPO / "analysis" / "formula_search" / "data" / "_tmp_filtered.parquet"
    df.to_parquet(tmp_parq, index=False)

    tasks = []
    for tgt in TARGETS:
        for fam in FAMILIES:
            tasks.append((tgt, fam, tmp_parq))

    print(f"\n=== Protocol A: per-coefficient 1D fit (MSE on 10 per-solvent OLS values) ===", flush=True)
    print(f"Running {len(tasks)} fits on {N_WORKERS} workers, {N_TRIALS_1D} trials each...", flush=True)

    results_A = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_task_protA, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results_A.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['target']:2s} {r['family']:10s}  MAE = {r['MAE_kJmol']:7.3f}", flush=True)

    print(f"\n=== Protocol B: joint global-MAE fit (warm-started from A) ===", flush=True)
    print(f"Running {len(tasks)} fits on {N_WORKERS} workers, {N_TRIALS_JOINT} trials each...", flush=True)

    results_B = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_task_protB, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results_B.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['target']:2s} {r['family']:10s}  MAE = {r['MAE_kJmol']:7.3f}", flush=True)

    tmp_parq.unlink(missing_ok=True)

    # Summary table
    print("\n=== Global MAE summary (kJ/mol) ===\n")
    rows = []
    for fam in FAMILIES:
        row = {"family": fam}
        for tgt in TARGETS:
            rA = next(r for r in results_A if r["target"] == tgt and r["family"] == fam)
            rB = next(r for r in results_B if r["target"] == tgt and r["family"] == fam)
            row[f"{tgt}_protA"] = rA["MAE_kJmol"]
            row[f"{tgt}_protB"] = rB["MAE_kJmol"]
        rows.append(row)
    summary = pd.DataFrame(rows)[["family", "IP_protA", "IP_protB", "EA_protA", "EA_protB"]]
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\n  protA = per-coef 1D fit on 10 OLS values (MSE objective)")
    print("  protB = joint fit on 1336 conformers (MAE objective, warm-started from A)")
    print(f"  Lower bound (per-solvent OLS, weighted): IP=4.50, EA=1.91 kJ/mol")
    print(f"  No-eps baseline (global OLS):           IP=23.84, EA=24.25 kJ/mol")

    # Best per target
    print("\n=== Per-solvent breakdown (best (family, protocol) per target) ===\n")
    for tgt in TARGETS:
        cands = [r for r in results_A + results_B if r["target"] == tgt]
        best = min(cands, key=lambda r: r["MAE_kJmol"])
        y = df[tgt].values * EH_TO_KJ
        per_solv = pd.DataFrame({
            "solvent": df["solvent"].values,
            "epsilon": df["epsilon"].values,
            "abs_err": np.abs(y - best["pred"]),
        }).groupby(["solvent", "epsilon"]).agg(
            MAE_kJmol=("abs_err", "mean"), n=("abs_err", "size"),
        ).reset_index().sort_values("epsilon")
        print(f"--- {tgt}  best = {best['family']} (protocol {best['protocol']})  global MAE = {best['MAE_kJmol']:.3f} ---")
        print(per_solv.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        print()

    out_dir = REPO / "analysis" / "formula_search" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "test_solvent_families_MAE.csv", index=False)
    print(f"Saved -> {(out_dir / 'test_solvent_families_MAE.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
