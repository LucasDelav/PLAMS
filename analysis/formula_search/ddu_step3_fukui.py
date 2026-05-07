#!/usr/bin/env python3
"""ddU Step 3: Fukui multiplicative correction (all-atoms only).

Baseline (Step 2 winner): pool (mu, eta) with the best ε-correction family.
The exact baseline form is loaded from `results/ddu_step2_best_per_target.json`
which Step 2's analysis script writes once we pick the gagnant.

Correction:  y = y_base * (1 + a_corr * fukui)

Tested Fukui variants (all-atoms only):
  ddU_red : f+_max, f²_max
  ddU_ox  : f-_max, f²_min

Joint refit (n_baseline + 1 params) via Nelder-Mead with multi-start.
Parallel over (target, fukui) pairs with ProcessPoolExecutor.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
PRED_DIR = REPO / "dataset_predict_dg"
OUT = REPO / "analysis" / "formula_search" / "results"
EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {"Hexane":1.88,"Toluene":2.38,"Chloroform":4.81,"Tetrahydrofuran":7.58,
             "Dichloromethane":8.93,"Acetone":20.7,"Methanol":32.6,"Acetonitrile":35.7,
             "DMSO":46.7,"Water":78.4}
STAB_TOL = 0.02


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def load_fukui(df):
    rows = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        out = {"fplus_max": np.nan, "fminus_max": np.nan,
               "f2_max": np.nan, "f2_min": np.nan}
        try:
            pred = json.load(open(path))
            cdft = pred["states"]["neutral_sp"]["cdft"]
            out["fplus_max"] = float(np.asarray(cdft["fplus"]).max())
            out["fminus_max"] = float(np.asarray(cdft["fminus"]).max())
            out["f2_max"] = float(np.asarray(cdft["dual"]).max())
            out["f2_min"] = float(np.asarray(cdft["dual"]).min())
        except Exception:
            pass
        rows.append(out)
    return pd.DataFrame(rows, index=df.index)


# Family registry — must mirror Step 2
def make_family(name):
    if name == "const":      return (1, lambda eps, g, p: np.full_like(eps, p[0]))
    if name == "affine_eps": return (2, lambda eps, g, p: p[0] + p[1]*eps)
    if name == "onsager":    return (2, lambda eps, g, p: p[0] + p[1]*(eps-1.0)/(eps+2.0))
    if name == "born_lin":   return (2, lambda eps, g, p: p[0] + p[1]*g)
    if name == "log_eps":    return (2, lambda eps, g, p: p[0] + p[1]*np.log(eps))
    if name == "pade11_g":   return (3, lambda eps, g, p: (p[0] + p[1]*g)/(1.0 + p[2]*g))
    if name == "pade11_eps": return (3, lambda eps, g, p: (p[0] + p[1]*eps)/(1.0 + p[2]*eps))
    if name == "pade21_eps": return (4, lambda eps, g, p: (p[0] + p[1]*eps + p[2]*eps**2)/(1.0 + p[3]*eps))
    if name == "pade22_eps": return (5, lambda eps, g, p: (p[0] + p[1]*eps + p[2]*eps**2)/(1.0 + p[3]*eps + p[4]*eps**2))
    raise ValueError(name)


def safe_pred(eps, g, mu, eta, fam_mu, fam_eta, params):
    n_mu = make_family(fam_mu)[0]
    p_mu, p_eta = params[:n_mu], params[n_mu:]
    # denominator stability checks
    if fam_mu == "pade11_g":
        if np.any(np.abs(1.0 + p_mu[2]*g) < STAB_TOL): return None
    elif fam_mu == "pade11_eps":
        if np.any(np.abs(1.0 + p_mu[2]*eps) < STAB_TOL): return None
    elif fam_mu == "pade21_eps":
        if np.any(np.abs(1.0 + p_mu[3]*eps) < STAB_TOL): return None
    elif fam_mu == "pade22_eps":
        if np.any(np.abs(1.0 + p_mu[3]*eps + p_mu[4]*eps**2) < STAB_TOL): return None
    if fam_eta == "pade11_g":
        if np.any(np.abs(1.0 + p_eta[2]*g) < STAB_TOL): return None
    elif fam_eta == "pade11_eps":
        if np.any(np.abs(1.0 + p_eta[2]*eps) < STAB_TOL): return None
    elif fam_eta == "pade21_eps":
        if np.any(np.abs(1.0 + p_eta[3]*eps) < STAB_TOL): return None
    elif fam_eta == "pade22_eps":
        if np.any(np.abs(1.0 + p_eta[3]*eps + p_eta[4]*eps**2) < STAB_TOL): return None
    bmu = make_family(fam_mu)[1](eps, g, p_mu)
    beta = make_family(fam_eta)[1](eps, g, p_eta)
    return bmu*mu + beta*eta


def fit_correction_one(args):
    """Worker: fit a single (target, fukui) joint refit."""
    target, fukui_col, fam_mu, fam_eta, theta0_baseline, df_dict = args
    df = pd.DataFrame(df_dict)
    eps, g = df["epsilon"].values, df["g"].values
    mu, eta = df["mu"].values, df["eta"].values
    f = df[fukui_col].values
    valid = np.isfinite(f)
    eps_v, g_v, mu_v, eta_v = eps[valid], g[valid], mu[valid], eta[valid]
    f_v = f[valid]
    y = df[target].values * EH_TO_KJ
    y_v = y[valid]
    n_base = len(theta0_baseline)

    def predict(p):
        base = safe_pred(eps_v, g_v, mu_v, eta_v, fam_mu, fam_eta, p[:n_base])
        if base is None: return None
        return base * (1.0 + p[n_base] * f_v)

    def obj(p):
        pp = predict(p)
        if pp is None: return 1e9
        return float(np.mean(np.abs(y_v - pp)))

    rng = np.random.default_rng(42)
    best, bv = None, np.inf
    for i, a_seed in enumerate([0.0, -1.0, -0.3, -0.1, 0.1, 0.3, 1.0]):
        x0 = np.concatenate([theta0_baseline, [a_seed]])
        if i > 4: x0[:n_base] += rng.normal(0, 0.1*max(1.0, np.abs(theta0_baseline).max()), size=n_base)
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 50000, "fatol": 1e-7, "xatol": 1e-7})
        if r.fun < bv: bv, best = r.fun, r.x

    # full-array predict for final metrics
    pred_full = safe_pred(eps, g, mu, eta, fam_mu, fam_eta, best[:n_base])
    pred_full = pred_full * (1.0 + best[n_base] * f) if pred_full is not None else np.full_like(y, np.nan)
    pred_full = np.where(valid, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))
    return {
        "target": target, "fukui": fukui_col,
        "theta_base": best[:n_base].tolist(),
        "a_corr": float(best[n_base]),
        "mae": mae, "rmse": rmse, "n_used": int(valid.sum()),
    }


def main():
    print(f"[ddU Step 3] starting...", flush=True)
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].dropna(
        subset=["ddU_red", "ddU_ox", "mu", "eta"]
    ).reset_index(drop=True)
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    df["g"] = (df["epsilon"] - 1.0) / (2.0 * df["epsilon"] + 1.0)
    print(f"  Loaded {len(df)} conformers, {df.molecule.nunique()} molecules, "
          f"{df.solvent.nunique()} solvents", flush=True)

    fk = load_fukui(df)
    df = df.drop(columns=[c for c in fk.columns if c in df.columns])
    df = pd.concat([df, fk], axis=1)
    print(f"  Fukui complete on {fk.dropna().shape[0]}/{len(df)} conformers", flush=True)

    # Load Step 2 baseline (best per target)
    step2_best = json.load(open(OUT / "ddu_step2_best_per_target.json"))
    print(f"  Loaded Step 2 baselines: "
          f"red→{step2_best['ddU_red']['fam_mu']}/{step2_best['ddU_red']['fam_eta']}, "
          f"ox→{step2_best['ddU_ox']['fam_mu']}/{step2_best['ddU_ox']['fam_eta']}",
          flush=True)

    tests = [
        ("ddU_red", "fplus_max"),
        ("ddU_red", "f2_max"),
        ("ddU_ox", "fminus_max"),
        ("ddU_ox", "f2_min"),
    ]
    df_dict = df[["epsilon", "g", "mu", "eta", "ddU_red", "ddU_ox",
                  "fplus_max", "fminus_max", "f2_max", "f2_min"]].to_dict("list")
    args = []
    for tgt, fcol in tests:
        b = step2_best[tgt]
        args.append((tgt, fcol, b["fam_mu"], b["fam_eta"], np.asarray(b["params"]), df_dict))

    n_workers = min(len(args), os.cpu_count() or 4)
    print(f"  Running {len(args)} fits in parallel on {n_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(fit_correction_one, args))

    print("\n=== ddU Step 3 — Fukui correction summary ===\n", flush=True)
    print(f"{'Target':<10} {'Fukui':<14} {'a_corr':>10} {'MAE':>8} {'RMSE':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['target']:<10} {r['fukui']:<14} {r['a_corr']:+10.4f} "
              f"{r['mae']:8.3f} {r['rmse']:8.3f}", flush=True)

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "theta_base"} for r in results])
    summary.to_csv(OUT / "ddu_step3_fukui_tests.csv", index=False)

    # Best per target → JSON
    best_per = {}
    for tgt in ("ddU_red", "ddU_ox"):
        sub = [r for r in results if r["target"] == tgt]
        bb = min(sub, key=lambda r: r["mae"])
        bb["fam_mu"] = step2_best[tgt]["fam_mu"]
        bb["fam_eta"] = step2_best[tgt]["fam_eta"]
        best_per[tgt] = bb
    json.dump(best_per, open(OUT / "ddu_step3_best.json", "w"), indent=2)
    print(f"\nSaved -> {OUT / 'ddu_step3_fukui_tests.csv'}")
    print(f"Saved -> {OUT / 'ddu_step3_best.json'}")


if __name__ == "__main__":
    main()
