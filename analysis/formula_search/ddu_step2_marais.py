#!/usr/bin/env python3
"""ddU Step 2 (marais-ready): combinatoric ε-correction families on pool (mu, eta).

Optimisations vs version précédente:
  - ProcessPoolExecutor sur tous les cores
  - Nelder-Mead allégé (maxiter 2000, fatol 1e-4) — convergence MAE @ ~0.001 kJ/mol suffit
  - 3 multi-starts par fit CV (au lieu de 4), 5 pour full (au lieu de 8)
  - Fonction objective vectorisée (numpy pur)
  - Pool réduit à 7 familles (retire affine_eps et pade11_eps redondantes)
  - Logs flushés par worker

Familles testées: const, onsager, born_lin, log_eps, pade11_g, pade21_eps, pade22_eps
Pool: (mu, eta), score: MAE_red + MAE_ox sur joint refit.

Lancer:
  ssh marais "cd ~/data/PlamsV5/analysis/formula_search && \
              .venv/bin/python ddu_step2_marais.py 2>&1 | tee results/ddu_step2.log"
"""
from __future__ import annotations
import json, os, sys, time
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
OUT = REPO / "analysis" / "formula_search" / "results"

EH = 2625.499638
EXCLUDED = {"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
EPSILON_R = {"Hexane":1.88,"Toluene":2.38,"Chloroform":4.81,"Tetrahydrofuran":7.58,
             "Dichloromethane":8.93,"Acetone":20.7,"Methanol":32.6,"Acetonitrile":35.7,
             "DMSO":46.7,"Water":78.4}
STAB_TOL = 0.02

# Param counts per family
FAM_NPARAM = {
    "const": 1, "onsager": 2, "born_lin": 2, "log_eps": 2,
    "pade11_g": 3, "pade21_eps": 4, "pade22_eps": 5,
}
F_NAMES = list(FAM_NPARAM.keys())


def fam_eval(name, eps, g, p):
    """Vectorised family evaluation. Returns array (or None if denominator unstable)."""
    if name == "const":      return np.full_like(eps, p[0])
    if name == "onsager":    return p[0] + p[1]*(eps-1.0)/(eps+2.0)
    if name == "born_lin":   return p[0] + p[1]*g
    if name == "log_eps":    return p[0] + p[1]*np.log(eps)
    if name == "pade11_g":
        den = 1.0 + p[2]*g
        if (np.abs(den) < STAB_TOL).any(): return None
        return (p[0] + p[1]*g) / den
    if name == "pade21_eps":
        den = 1.0 + p[3]*eps
        if (np.abs(den) < STAB_TOL).any(): return None
        return (p[0] + p[1]*eps + p[2]*eps**2) / den
    if name == "pade22_eps":
        den = 1.0 + p[3]*eps + p[4]*eps**2
        if (np.abs(den) < STAB_TOL).any(): return None
        return (p[0] + p[1]*eps + p[2]*eps**2) / den
    raise ValueError(name)


def predict(eps, g, mu, eta, fam_mu, fam_eta, params):
    n_mu = FAM_NPARAM[fam_mu]
    bmu = fam_eval(fam_mu, eps, g, params[:n_mu])
    if bmu is None: return None
    beta = fam_eval(fam_eta, eps, g, params[n_mu:])
    if beta is None: return None
    return bmu*mu + beta*eta


def fit_one(eps, g, mu, eta, y, fam_mu, fam_eta, n_starts=3, seed=42):
    n = FAM_NPARAM[fam_mu] + FAM_NPARAM[fam_eta]
    rng = np.random.default_rng(seed)
    best_p, best_v = None, np.inf
    for s in range(n_starts):
        x0 = np.zeros(n)
        x0[0] = 1.0; x0[FAM_NPARAM[fam_mu]] = 1.0
        if s > 0: x0 = x0 + rng.normal(0, 0.5, size=n)
        def obj(p):
            pp = predict(eps, g, mu, eta, fam_mu, fam_eta, p)
            return 1e9 if pp is None else float(np.mean(np.abs(y - pp)))
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 2000, "fatol": 1e-4, "xatol": 1e-4})
        if r.fun < best_v: best_v, best_p = r.fun, r.x
    return best_p, best_v


def cv_then_full(eps, g, mu, eta, y, groups, fam_mu, fam_eta):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(np.zeros(len(y)), y, groups):
        p, _ = fit_one(eps[tr], g[tr], mu[tr], eta[tr], y[tr], fam_mu, fam_eta, n_starts=3)
        if p is None:
            pred[te] = np.nan
        else:
            pp = predict(eps[te], g[te], mu[te], eta[te], fam_mu, fam_eta, p)
            pred[te] = pp if pp is not None else np.nan
    valid = np.isfinite(pred)
    if valid.sum() < 0.9 * len(y):
        return None
    mae_cv = float(np.mean(np.abs(y[valid] - pred[valid])))
    rmse_cv = float(np.sqrt(np.mean((y[valid] - pred[valid])**2)))
    p_full, _ = fit_one(eps, g, mu, eta, y, fam_mu, fam_eta, n_starts=5)
    pp_full = predict(eps, g, mu, eta, fam_mu, fam_eta, p_full)
    mae_fit = float(np.mean(np.abs(y - pp_full))) if pp_full is not None else np.nan
    return {"mae_cv": mae_cv, "rmse_cv": rmse_cv, "mae_fit": mae_fit,
            "params": p_full.tolist(), "n_params": len(p_full)}


# Worker
_DATA = None  # populated in worker init

def _init(eps, g, mu, eta, y_red, y_ox, groups):
    global _DATA
    _DATA = (eps, g, mu, eta, y_red, y_ox, groups)


def _job(combo):
    target, fam_mu, fam_eta = combo
    eps, g, mu, eta, y_red, y_ox, groups = _DATA
    y = y_red if target == "ddU_red" else y_ox
    t0 = time.time()
    r = cv_then_full(eps, g, mu, eta, y, groups, fam_mu, fam_eta)
    dt = time.time() - t0
    if r is None:
        return {"target": target, "fam_mu": fam_mu, "fam_eta": fam_eta,
                "MAE_cv": np.nan, "RMSE_cv": np.nan, "MAE_fit": np.nan,
                "n_params": np.nan, "params": None, "dt": dt, "skipped": True}
    return {"target": target, "fam_mu": fam_mu, "fam_eta": fam_eta,
            "MAE_cv": r["mae_cv"], "RMSE_cv": r["rmse_cv"], "MAE_fit": r["mae_fit"],
            "n_params": r["n_params"], "params": r["params"], "dt": dt, "skipped": False}


def main():
    print(f"[ddU Step 2 / marais] starting at {time.strftime('%H:%M:%S')}", flush=True)
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED)].dropna(
        subset=["ddU_red", "ddU_ox", "mu", "eta"]
    ).reset_index(drop=True)
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    df["g"] = (df["epsilon"]-1.0) / (2.0*df["epsilon"]+1.0)
    print(f"  n={len(df)}, n_mol={df.molecule.nunique()}, "
          f"n_solv={df.solvent.nunique()}", flush=True)

    eps = df["epsilon"].values.astype(np.float64)
    g = df["g"].values.astype(np.float64)
    mu = df["mu"].values.astype(np.float64)
    eta = df["eta"].values.astype(np.float64)
    y_red = (df["ddU_red"].values * EH).astype(np.float64)
    y_ox = (df["ddU_ox"].values * EH).astype(np.float64)
    groups = df["molecule"].values

    combos = []
    for tgt in ("ddU_red", "ddU_ox"):
        for fmu, feta in product(F_NAMES, F_NAMES):
            combos.append((tgt, fmu, feta))
    n_total = len(combos)
    n_workers = min(n_total, os.cpu_count() or 4)
    print(f"  Combos: {n_total} (7 fams × 7 × 2 targets), workers={n_workers}", flush=True)

    t_global = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init,
                              initargs=(eps, g, mu, eta, y_red, y_ox, groups)) as ex:
        futures = {ex.submit(_job, c): c for c in combos}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            results.append(r)
            tag = "OK " if not r["skipped"] else "SKIP"
            print(f"  [{done}/{n_total}] {tag} {r['target']:8} {r['fam_mu']:11}/{r['fam_eta']:11} "
                  f"k={r['n_params']:>4} MAE_cv={r['MAE_cv']:6.3f} dt={r['dt']:5.1f}s",
                  flush=True)
    print(f"  Total: {time.time()-t_global:.1f}s", flush=True)

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in results])
    res.to_csv(OUT / "ddu_step2_families.csv", index=False)

    # Top 10 per target
    print("\n=== ddU_red — Top 10 by MAE_cv ===", flush=True)
    print(res[res.target=="ddU_red"].nsmallest(10, "MAE_cv")
          [["fam_mu","fam_eta","n_params","MAE_cv","RMSE_cv","MAE_fit","dt"]]
          .to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\n=== ddU_ox — Top 10 by MAE_cv ===", flush=True)
    print(res[res.target=="ddU_ox"].nsmallest(10, "MAE_cv")
          [["fam_mu","fam_eta","n_params","MAE_cv","RMSE_cv","MAE_fit","dt"]]
          .to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # Best UNIFORM family per target (for downstream Step 3)
    best_per_target = {}
    for tgt in ("ddU_red", "ddU_ox"):
        sub = [r for r in results if r["target"] == tgt and not r["skipped"]]
        bb = min(sub, key=lambda r: r["MAE_cv"])
        best_per_target[tgt] = {
            "fam_mu": bb["fam_mu"], "fam_eta": bb["fam_eta"],
            "n_params": int(bb["n_params"]),
            "params": bb["params"],
            "MAE_cv": bb["MAE_cv"], "RMSE_cv": bb["RMSE_cv"], "MAE_fit": bb["MAE_fit"],
        }
    json.dump(best_per_target, open(OUT / "ddu_step2_best_per_target.json", "w"), indent=2)
    print(f"\nBest red: {best_per_target['ddU_red']['fam_mu']}/{best_per_target['ddU_red']['fam_eta']} "
          f"MAE={best_per_target['ddU_red']['MAE_cv']:.3f}", flush=True)
    print(f"Best ox:  {best_per_target['ddU_ox']['fam_mu']}/{best_per_target['ddU_ox']['fam_eta']} "
          f"MAE={best_per_target['ddU_ox']['MAE_cv']:.3f}", flush=True)
    print(f"\nSaved -> {OUT / 'ddu_step2_families.csv'}", flush=True)
    print(f"Saved -> {OUT / 'ddu_step2_best_per_target.json'}", flush=True)


if __name__ == "__main__":
    main()
