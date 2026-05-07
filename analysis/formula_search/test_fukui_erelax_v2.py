#!/usr/bin/env python3
"""E_relax v2 — pool (mu, eta, omega, LVC2_paired), const en eps,
correction × (1 + a · fukui_paired) avec 4 variantes Fukui :
  - f±_max          : max sur tous les atomes
  - f±_max(O)       : max restreint aux oxygènes
  - f±_max(hetero)  : max restreint aux hétéroatomes {N,O,S,P,F,Cl,Br,I}
  - f±_max_noO      : max sur tous les atomes sauf O
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
DESCRIPTORS = ("mu", "eta", "omega")
HETEROATOMS = {"N", "O", "S", "P", "F", "Cl", "Br", "I"}


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def fukui_max_subset(values, symbols, mode):
    """mode in {'all','O','hetero','noO'}"""
    arr = np.asarray(values, dtype=float)
    syms = np.asarray(symbols)
    if mode == "all":
        mask = np.ones(len(syms), dtype=bool)
    elif mode == "O":
        mask = (syms == "O")
    elif mode == "hetero":
        mask = np.array([s in HETEROATOMS for s in syms])
    elif mode == "noO":
        mask = (syms != "O")
    else:
        raise ValueError(mode)
    if not mask.any():
        return float("nan")
    return float(arr[mask].max())


def load_fukui(df):
    rows = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        out = {f"{kind}_max_{mode}": np.nan
               for kind in ("fplus", "fminus") for mode in ("all", "O", "hetero", "noO")}
        if path.exists():
            try:
                pred = json.load(open(path))
                ns = pred["states"]["neutral_sp"]
                cdft = ns["cdft"]
                syms = parse_xyz_symbols(ns["xyz"])
                for mode in ("all", "O", "hetero", "noO"):
                    out[f"fplus_max_{mode}"] = fukui_max_subset(cdft["fplus"], syms, mode)
                    out[f"fminus_max_{mode}"] = fukui_max_subset(cdft["fminus"], syms, mode)
            except (KeyError, IndexError, ValueError, json.JSONDecodeError):
                pass
        rows.append(out)
    return pd.DataFrame(rows, index=df.index)


def fit_baseline(df, target):
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
    suff = target.split("_")[1]
    cols = list(DESCRIPTORS) + [f"lvc2_{suff}"]
    X = df[cols].values.astype(float)
    y = df[target].values * EH_TO_KJ
    f = df[fukui_col].values
    valid = np.isfinite(f)
    Xv, yv, fv = X[valid], y[valid], f[valid]
    n_theta = X.shape[1]

    def predict(p):
        return (Xv @ p[:n_theta]) * (1.0 + p[n_theta] * fv)

    def obj(p): return np.mean(np.abs(yv - predict(p)))

    seeds = [0.0, -1.0, -0.3, -0.1, 0.1, 0.3, 1.0]
    best, bestv = None, np.inf
    rng = np.random.default_rng(42)
    for i, a_seed in enumerate(seeds):
        x0 = np.concatenate([theta0, [a_seed]])
        if i > 4: x0[:n_theta] += rng.normal(0, 0.1*np.abs(theta0).max(), size=n_theta)
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 50000, "fatol": 1e-7, "xatol": 1e-7})
        if r.fun < bestv: bestv, best = r.fun, r.x

    pred_full = (X @ best[:n_theta]) * (1.0 + best[n_theta] * f)
    pred_full = np.where(valid, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))
    n_used = int(valid.sum())
    return best, pred_full, mae, rmse, n_used


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].dropna(
        subset=["Erelax_red", "Erelax_ox", "mu", "eta", "lvc2_red", "lvc2_ox"]
    ).reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    print(f"Dataset: n={len(df)}, n_mol={df.molecule.nunique()}, "
          f"n_solv={df.solvent.nunique()}", flush=True)

    print("\nLoading Fukui descriptors (4 modes)...", flush=True)
    fk = load_fukui(df)
    df = df.drop(columns=[c for c in fk.columns if c in df.columns])
    df = pd.concat([df, fk], axis=1)
    coverage = {c: int(df[c].notna().sum()) for c in fk.columns}
    print(f"  Coverage examples: fplus_max_O={coverage['fplus_max_O']}, "
          f"fplus_max_hetero={coverage['fplus_max_hetero']}, "
          f"fminus_max_hetero={coverage['fminus_max_hetero']}", flush=True)

    # Baselines
    baselines = {}
    for tgt in ("Erelax_red", "Erelax_ox"):
        theta0, _ = fit_baseline(df, tgt)
        suff = tgt.split("_")[1]
        cols = list(DESCRIPTORS) + [f"lvc2_{suff}"]
        X = df[cols].values
        y = df[tgt].values * EH_TO_KJ
        pred_b = X @ theta0
        mae_b = float(np.mean(np.abs(y - pred_b)))
        rmse_b = float(np.sqrt(np.mean((y - pred_b) ** 2)))
        baselines[tgt] = {"theta": theta0, "mae": mae_b, "rmse": rmse_b}
        print(f"\nBaseline {tgt} (mu, eta, omega, lvc2_{suff}): "
              f"theta={theta0.round(4).tolist()}  MAE={mae_b:.3f} RMSE={rmse_b:.3f}")

    # 4 Fukui variants per target
    tests = {
        "Erelax_red": [
            ("fplus_max_all",    "f+_max"),
            ("fplus_max_O",      "f+_max(O)"),
            ("fplus_max_hetero", "f+_max(hetero)"),
            ("fplus_max_noO",    "f+_max_noO"),
        ],
        "Erelax_ox": [
            ("fminus_max_all",    "f-_max"),
            ("fminus_max_O",      "f-_max(O)"),
            ("fminus_max_hetero", "f-_max(hetero)"),
            ("fminus_max_noO",    "f-_max_noO"),
        ],
    }

    results = []
    for tgt, vlist in tests.items():
        theta0 = baselines[tgt]["theta"]
        print(f"\n--- {tgt} ---", flush=True)
        for fcol, label in vlist:
            best, pred, mae, rmse, n_used = fit_correction(df, tgt, fcol, theta0)
            n_theta = len(theta0)
            results.append({
                "target": tgt, "fukui": fcol, "label": label, "n_used": n_used,
                "theta": best[:n_theta].tolist(),
                "a_corr": float(best[n_theta]),
                "mae_base": baselines[tgt]["mae"],
                "rmse_base": baselines[tgt]["rmse"],
                "mae_corr": mae, "rmse_corr": rmse,
                "delta_mae": mae - baselines[tgt]["mae"],
            })
            print(f"  {label:18}  n={n_used:4d}  MAE {baselines[tgt]['mae']:.2f} -> {mae:.2f}  "
                  f"RMSE {baselines[tgt]['rmse']:.2f} -> {rmse:.2f}  a_corr={best[n_theta]:+.4f}",
                  flush=True)

    summary = pd.DataFrame([{
        "target": r["target"], "fukui": r["label"], "n_used": r["n_used"],
        "a_corr": r["a_corr"],
        "MAE_base": r["mae_base"], "MAE_corr": r["mae_corr"], "delta_MAE": r["delta_mae"],
        "RMSE_base": r["rmse_base"], "RMSE_corr": r["rmse_corr"],
    } for r in results])
    print("\n=== Fukui-correction summary on E_relax v2 (pool with omega) ===\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    out = REPO / "analysis" / "formula_search" / "results"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "erelax_v2_fukui_correction_tests.csv", index=False)
    coef_rows = []
    for r in results:
        suff = r["target"].split("_")[1]
        cols = list(DESCRIPTORS) + [f"lvc2_{suff}"]
        for d, v in zip(cols, r["theta"]):
            coef_rows.append({"target": r["target"], "fukui": r["fukui"],
                              "param": f"theta_{d}", "value": float(v)})
        coef_rows.append({"target": r["target"], "fukui": r["fukui"],
                          "param": "a_corr", "value": float(r["a_corr"])})
    pd.DataFrame(coef_rows).to_csv(out / "erelax_v2_fukui_correction_coeffs.csv", index=False)

    best_per_tgt = {}
    for tgt in ("Erelax_red", "Erelax_ox"):
        sub = [r for r in results if r["target"] == tgt]
        best = min(sub, key=lambda r: r["mae_corr"])
        best_per_tgt[tgt] = best
    json.dump(best_per_tgt, open(out / "erelax_v2_best_fukui_models.json", "w"), indent=2)
    print(f"\nSaved -> {out / 'erelax_v2_fukui_correction_tests.csv'}")
    print(f"Saved -> {out / 'erelax_v2_fukui_correction_coeffs.csv'}")
    print(f"Saved -> {out / 'erelax_v2_best_fukui_models.json'}")
    for tgt, b in best_per_tgt.items():
        print(f"  Best for {tgt}: {b['label']}  MAE={b['mae_corr']:.3f}  a_corr={b['a_corr']:+.4f}")


if __name__ == "__main__":
    main()
