#!/usr/bin/env python3
"""Test multiplicative Fukui corrections on E_relax baseline.

Baseline (Step 2): y = sum_X theta_X * X with X in {mu, eta, mu2, lvc2_paired}.
No solvent dependence ('const' family).

Correction:        y = y_base * (1 + a_corr * fukui)

Joint refit (5 params total: 4 thetas + a_corr) by minimising MAE on the
1336-conformer / 10-solvent dataset, warm-started at the const baseline.

Tests Fukui variants (with and without oxygen atoms):
  Erelax_red : f+_max [_noO], f2_max [_noO]      (reduction populates LUMO)
  Erelax_ox  : f-_max [_noO], f2_min [_noO]      (oxidation depopulates HOMO)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
PRED_DIR = REPO / "dataset_predict_dg"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
DESCRIPTORS = ("mu", "eta", "mu2")  # +lvc2_paired added per-target


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def fukui_stats(values, symbols, drop_O):
    """Return (max, min) over atoms (excluding O if drop_O)."""
    arr = np.asarray(values, dtype=float)
    syms = np.asarray(symbols)
    mask = (syms != "O") if drop_O else np.ones(len(syms), dtype=bool)
    if not mask.any():
        return float("nan"), float("nan")
    sub = arr[mask]
    return float(sub.max()), float(sub.min())


def load_fukui(df):
    """For each conformer, read predict-dg JSON and compute the relevant Fukui descriptors."""
    rows = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        if not path.exists():
            rows.append({k: np.nan for k in (
                "fplus_max", "fplus_max_noO",
                "fminus_max", "fminus_max_noO",
                "f2_max", "f2_max_noO",
                "f2_min", "f2_min_noO")})
            continue
        try:
            pred = json.load(open(path))
            ns = pred["states"]["neutral_sp"]
            cdft = ns["cdft"]
            syms = parse_xyz_symbols(ns["xyz"])
            fp_max, _ = fukui_stats(cdft["fplus"], syms, drop_O=False)
            fp_max_noO, _ = fukui_stats(cdft["fplus"], syms, drop_O=True)
            fm_max, _ = fukui_stats(cdft["fminus"], syms, drop_O=False)
            fm_max_noO, _ = fukui_stats(cdft["fminus"], syms, drop_O=True)
            d_max, d_min = fukui_stats(cdft["dual"], syms, drop_O=False)
            d_max_noO, d_min_noO = fukui_stats(cdft["dual"], syms, drop_O=True)
            rows.append({
                "fplus_max": fp_max, "fplus_max_noO": fp_max_noO,
                "fminus_max": fm_max, "fminus_max_noO": fm_max_noO,
                "f2_max": d_max, "f2_max_noO": d_max_noO,
                "f2_min": d_min, "f2_min_noO": d_min_noO,
            })
        except (KeyError, IndexError, ValueError, json.JSONDecodeError):
            rows.append({k: np.nan for k in (
                "fplus_max", "fplus_max_noO",
                "fminus_max", "fminus_max_noO",
                "f2_max", "f2_max_noO",
                "f2_min", "f2_min_noO")})
    return pd.DataFrame(rows, index=df.index)


def predict_baseline(df, target, params):
    """const-family baseline: y = sum theta_X * X."""
    suff = target.split("_")[1]
    cols = list(DESCRIPTORS) + [f"lvc2_{suff}"]
    X = df[cols].values.astype(float)
    return X @ np.asarray(params)


def fit_baseline(df, target):
    """Refit the const baseline on the full dataset (warm-start for correction)."""
    suff = target.split("_")[1]
    cols = list(DESCRIPTORS) + [f"lvc2_{suff}"]
    X = df[cols].values.astype(float)
    y = df[target].values * EH_TO_KJ
    def obj(p): return np.mean(np.abs(y - X @ p))
    best, bestv = None, np.inf
    rng = np.random.default_rng(42)
    for s in range(8):
        x0 = np.zeros(X.shape[1]); x0[0] = 1.0
        if s > 0: x0 = x0 + rng.normal(0, 0.5, size=X.shape[1])
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 20000, "fatol": 1e-6})
        if r.fun < bestv: bestv, best = r.fun, r.x
    return best, bestv


def fit_correction(df, target, fukui_col, theta0):
    """Joint refit of (theta_mu, theta_eta, theta_mu2, theta_lvc, a_corr).

    Warm-started at theta0 (baseline) and a_corr=0, plus a few seeds.
    """
    suff = target.split("_")[1]
    cols = list(DESCRIPTORS) + [f"lvc2_{suff}"]
    X = df[cols].values.astype(float)
    y = df[target].values * EH_TO_KJ
    f = df[fukui_col].values
    valid = np.isfinite(f)
    Xv, yv, fv = X[valid], y[valid], f[valid]

    n_theta = X.shape[1]
    def predict(p):
        theta, a = p[:n_theta], p[n_theta]
        return (Xv @ theta) * (1.0 + a * fv)
    def obj(p): return np.mean(np.abs(yv - predict(p)))

    seeds = [(0.0,), (-1.0,), (-0.3,), (-0.1,), (0.1,), (0.3,), (1.0,)]
    best, bestv = None, np.inf
    rng = np.random.default_rng(42)
    for i, (a_seed,) in enumerate(seeds):
        x0 = np.concatenate([theta0, [a_seed]])
        if i > 4: x0[:n_theta] += rng.normal(0, 0.1*np.abs(theta0).max(), size=n_theta)
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 50000, "fatol": 1e-7, "xatol": 1e-7})
        if r.fun < bestv: bestv, best = r.fun, r.x

    pred_full = (X @ best[:n_theta]) * (1.0 + best[n_theta] * f)
    pred_full = np.where(valid, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))
    return best, pred_full, mae, rmse, valid


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].dropna(
        subset=["Erelax_red", "Erelax_ox", "mu", "eta", "lvc2_red", "lvc2_ox"]
    ).reset_index(drop=True)
    df["mu2"] = df["mu"] ** 2
    print(f"Dataset: n={len(df)}, n_mol={df.molecule.nunique()}, "
          f"n_solv={df.solvent.nunique()}", flush=True)

    print("\nLoading Fukui descriptors from predict-dg JSONs...", flush=True)
    fukui_df = load_fukui(df)
    # drop any pre-existing columns that would collide with the recomputed Fukui set
    df = df.drop(columns=[c for c in fukui_df.columns if c in df.columns])
    df = pd.concat([df, fukui_df], axis=1)
    print(f"  Fukui complete on {fukui_df.dropna().shape[0]}/{len(df)} conformers")

    # Fit baselines
    baselines = {}
    for tgt in ("Erelax_red", "Erelax_ox"):
        theta0, _ = fit_baseline(df, tgt)
        pred_base = predict_baseline(df, tgt, theta0)
        y = df[tgt].values * EH_TO_KJ
        mae_b = float(np.mean(np.abs(y - pred_base)))
        rmse_b = float(np.sqrt(np.mean((y - pred_base) ** 2)))
        baselines[tgt] = {"theta": theta0, "mae": mae_b, "rmse": rmse_b}
        print(f"\nBaseline {tgt}: theta={theta0.round(3).tolist()}  "
              f"MAE={mae_b:.3f} RMSE={rmse_b:.3f}")

    # Fukui-correction tests
    tests = {
        "Erelax_red": [
            ("fplus_max", "f+_max"),
            ("fplus_max_noO", "f+_max_noO"),
            ("f2_max", "f2_max"),
            ("f2_max_noO", "f2_max_noO"),
        ],
        "Erelax_ox": [
            ("fminus_max", "f-_max"),
            ("fminus_max_noO", "f-_max_noO"),
            ("f2_min", "f2_min"),
            ("f2_min_noO", "f2_min_noO"),
        ],
    }

    results = []
    for tgt, vlist in tests.items():
        theta0 = baselines[tgt]["theta"]
        for fcol, label in vlist:
            print(f"  fitting {tgt} × (1+a·{label})...", flush=True)
            best, pred, mae, rmse, valid = fit_correction(df, tgt, fcol, theta0)
            n_theta = len(theta0)
            results.append({
                "target": tgt, "fukui": fcol, "label": label,
                "theta": best[:n_theta].tolist(),
                "a_corr": float(best[n_theta]),
                "mae_base": baselines[tgt]["mae"],
                "rmse_base": baselines[tgt]["rmse"],
                "mae_corr": mae, "rmse_corr": rmse,
                "delta_mae": mae - baselines[tgt]["mae"],
            })
            print(f"    MAE: {baselines[tgt]['mae']:.2f} -> {mae:.2f}   "
                  f"a_corr={best[n_theta]:.4f}", flush=True)

    summary = pd.DataFrame([{
        "target": r["target"], "fukui": r["label"],
        "a_corr": r["a_corr"], "MAE_base": r["mae_base"],
        "MAE_corr": r["mae_corr"], "delta_MAE": r["delta_mae"],
        "RMSE_base": r["rmse_base"], "RMSE_corr": r["rmse_corr"],
    } for r in results])
    print("\n=== Fukui-correction summary on E_relax (joint refit, n_params=5) ===\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # save
    out = REPO / "analysis" / "formula_search" / "results"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "erelax_fukui_correction_tests.csv", index=False)
    coef_rows = []
    for r in results:
        for d, v in zip(list(DESCRIPTORS) + [f"lvc2_{r['target'].split('_')[1]}"], r["theta"]):
            coef_rows.append({"target": r["target"], "fukui": r["fukui"],
                              "param": f"theta_{d}", "value": float(v)})
        coef_rows.append({"target": r["target"], "fukui": r["fukui"],
                          "param": "a_corr", "value": float(r["a_corr"])})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(out / "erelax_fukui_correction_coeffs.csv", index=False)

    # Save the best per target
    best_per_tgt = {}
    for tgt in ("Erelax_red", "Erelax_ox"):
        sub = [r for r in results if r["target"] == tgt]
        best = min(sub, key=lambda r: r["mae_corr"])
        best_per_tgt[tgt] = best
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in best_per_tgt.items()},
              open(out / "erelax_best_fukui_models.json", "w"), indent=2)
    print(f"\nSaved -> {out / 'erelax_fukui_correction_tests.csv'}")
    print(f"Saved -> {out / 'erelax_fukui_correction_coeffs.csv'}")
    print(f"Saved -> {out / 'erelax_best_fukui_models.json'}")
    for tgt, b in best_per_tgt.items():
        print(f"  Best for {tgt}: {b['label']}  MAE={b['mae_corr']:.2f}  a_corr={b['a_corr']:.4f}")


if __name__ == "__main__":
    main()
