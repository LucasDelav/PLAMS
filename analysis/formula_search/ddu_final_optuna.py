#!/usr/bin/env python3
"""ddU final model — Optuna CMA-ES optimisation.

Form (homogène red/ox):
  y = (theta_mu·mu + theta_eta·eta + theta_dEta·Δη_paired) × (1 + a · f²_paired)

with:
  - Δη_paired = η(reduced_sp) − η(neutral_sp) for ddU_red
                η(oxidized_sp) − η(neutral_sp) for ddU_ox
  - f²_paired = max(dual) for ddU_red, min(dual) for ddU_ox
  - All from vertical SP on neutral geometry. NO ionic-relaxation data.

Steps run:
  - Step 2 (re)confirm: test 4 ε-families on the new pool to confirm `const` wins
  - Step 3 (re)confirm: 4 Fukui variants (f+_max, f-_max, f²_max, f²_min) all-atoms
  - Final: Optuna CMA-ES joint fit, n_trials=12000

Outputs:
  - results/ddu_final_step2.csv  (family comparison)
  - results/ddu_final_step3.csv  (fukui comparison)
  - results/ddu_final_optuna.json (final model + coefficients)
"""
from __future__ import annotations
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "ddu_delta_descriptors.parquet"
OUT = REPO / "analysis" / "formula_search" / "results"
EH = 2625.499638
SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)

EPSILON_R = {"Hexane":1.88,"Toluene":2.38,"Chloroform":4.81,"Tetrahydrofuran":7.58,
             "Dichloromethane":8.93,"Acetone":20.7,"Methanol":32.6,"Acetonitrile":35.7,
             "DMSO":46.7,"Water":78.4}


# ---------- Family registry ----------
def fam_eval(name, eps, g, p):
    if name == "const":      return np.full_like(eps, p[0])
    if name == "onsager":    return p[0] + p[1]*(eps-1.0)/(eps+2.0)
    if name == "born_lin":   return p[0] + p[1]*g
    if name == "pade21_eps":
        den = 1.0 + p[3]*eps
        if (np.abs(den) < 0.02).any(): return None
        return (p[0] + p[1]*eps + p[2]*eps**2) / den
    raise ValueError(name)

NPAR = {"const":1, "onsager":2, "born_lin":2, "pade21_eps":4}


def predict_family(eps, g, mu, eta, dEta, fukui, fam_mu, fam_eta, fam_dEta, params):
    """y = (β_μ·μ + β_η·η + β_dEta·dEta) × (1 + a·fukui)"""
    n_mu, n_eta = NPAR[fam_mu], NPAR[fam_eta]
    n_dE = NPAR[fam_dEta]
    p_mu  = params[:n_mu]
    p_eta = params[n_mu:n_mu+n_eta]
    p_dE  = params[n_mu+n_eta:n_mu+n_eta+n_dE]
    a_corr = params[n_mu+n_eta+n_dE]
    bmu = fam_eval(fam_mu, eps, g, p_mu)
    if bmu is None: return None
    beta = fam_eval(fam_eta, eps, g, p_eta)
    if beta is None: return None
    bdE = fam_eval(fam_dEta, eps, g, p_dE)
    if bdE is None: return None
    return (bmu*mu + beta*eta + bdE*dEta) * (1.0 + a_corr*fukui)


# ---------- Worker for Step 2 family search ----------
_DATA = None

def _init_step2(eps, g, mu, eta, dEta_red, dEta_ox, f2max, f2min, y_red, y_ox, groups):
    global _DATA
    _DATA = (eps, g, mu, eta, dEta_red, dEta_ox, f2max, f2min, y_red, y_ox, groups)

def _job_step2(args):
    target, fam_mu, fam_eta, fam_dEta = args
    eps, g, mu, eta, dEta_red, dEta_ox, f2max, f2min, y_red, y_ox, groups = _DATA
    if target == "ddU_red":
        dEta = dEta_red; fukui = f2max; y = y_red
    else:
        dEta = dEta_ox;  fukui = f2min; y = y_ox

    n_total = NPAR[fam_mu] + NPAR[fam_eta] + NPAR[fam_dEta] + 1
    rng = np.random.default_rng(SEED)
    pred = np.zeros_like(y)

    def fit(eps, g, mu, eta, dEta, fukui, y, n_starts=3):
        best_p, best_v = None, np.inf
        for s in range(n_starts):
            x0 = np.zeros(n_total)
            x0[0] = 1.0
            i = NPAR[fam_mu]; x0[i] = 1.0
            i = NPAR[fam_mu] + NPAR[fam_eta]; x0[i] = 1.0
            if s > 0: x0 = x0 + rng.normal(0, 0.5, size=n_total)
            def obj(p):
                pp = predict_family(eps, g, mu, eta, dEta, fukui, fam_mu, fam_eta, fam_dEta, p)
                if pp is None: return 1e9
                return float(np.mean(np.abs(y - pp)))
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 3000, "fatol": 1e-5})
            if r.fun < best_v: best_v, best_p = r.fun, r.x
        return best_p, best_v

    # CV
    for tr, te in GroupKFold(5).split(y, y, groups):
        p, _ = fit(eps[tr], g[tr], mu[tr], eta[tr], dEta[tr], fukui[tr], y[tr], n_starts=3)
        if p is None:
            pred[te] = np.nan
        else:
            pp = predict_family(eps[te], g[te], mu[te], eta[te], dEta[te], fukui[te],
                                fam_mu, fam_eta, fam_dEta, p)
            pred[te] = pp if pp is not None else np.nan
    valid = np.isfinite(pred)
    if valid.sum() < 0.9 * len(y):
        return {"target": target, "fam_mu": fam_mu, "fam_eta": fam_eta, "fam_dEta": fam_dEta,
                "MAE_cv": np.nan, "RMSE_cv": np.nan, "skipped": True}
    mae_cv = float(np.mean(np.abs(y[valid] - pred[valid])))
    rmse_cv = float(np.sqrt(np.mean((y[valid] - pred[valid])**2)))
    p_full, _ = fit(eps, g, mu, eta, dEta, fukui, y, n_starts=5)
    return {"target": target, "fam_mu": fam_mu, "fam_eta": fam_eta, "fam_dEta": fam_dEta,
            "MAE_cv": mae_cv, "RMSE_cv": rmse_cv, "n_params": n_total,
            "params": p_full.tolist(), "skipped": False}


def step2_family_search(df):
    print("\n=== Step 2: ε-family search on pool (μ, η, Δη_paired) ===", flush=True)
    eps = df["epsilon"].values
    g = df["g"].values
    mu = df["mu"].values
    eta = df["eta"].values
    dEta_red = df["deta_red"].values
    dEta_ox  = df["deta_ox"].values
    f2max = df["f2_max"].values
    f2min = df["f2_min"].values
    y_red = (df["ddU_red"].values * EH)
    y_ox  = (df["ddU_ox"].values * EH)
    groups = df["molecule"].values

    # Test only uniform families to keep it simple and homogeneous
    fams = ["const", "onsager", "born_lin", "pade21_eps"]
    combos = []
    for tgt in ("ddU_red", "ddU_ox"):
        for f in fams:
            combos.append((tgt, f, f, f))  # uniform
    n_workers = min(len(combos), os.cpu_count() or 4)
    print(f"  Combos: {len(combos)} (uniform only), workers={n_workers}", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init_step2,
                              initargs=(eps, g, mu, eta, dEta_red, dEta_ox, f2max, f2min,
                                        y_red, y_ox, groups)) as ex:
        for r in ex.map(_job_step2, combos):
            results.append(r)
            tag = "OK" if not r.get("skipped") else "SK"
            print(f"  [{tag}] {r['target']:8} {r['fam_mu']:11} k={r.get('n_params','-')} "
                  f"MAE={r.get('MAE_cv', float('nan')):6.3f}", flush=True)
    print(f"  step 2 total: {time.time()-t0:.1f}s", flush=True)

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in results])
    res.to_csv(OUT / "ddu_final_step2.csv", index=False)
    print("\n  Step 2 family ranking per target:", flush=True)
    for tgt in ("ddU_red", "ddU_ox"):
        sub = res[(res.target == tgt) & ~res.skipped].sort_values("MAE_cv")
        print(f"\n  -- {tgt} --")
        print(sub[["fam_mu", "n_params", "MAE_cv", "RMSE_cv"]].to_string(
            index=False, float_format=lambda x: f"{x:7.3f}"))
    return res


# ---------- Step 3: Fukui variants ----------
def step3_fukui_search(df):
    print("\n=== Step 3: Fukui variant comparison (all-atoms only) ===", flush=True)
    # Re-compute fukui variants from JSON
    PRED = REPO / "dataset_predict_dg"
    fp, fm = [], []
    for _, row in df.iterrows():
        p = PRED / row.solvent / row.molecule / f"{row.molecule}_conf{int(row.conformer)}_prediction.json"
        try:
            d = json.load(open(p))
            ns = d["states"]["neutral_sp"]["cdft"]
            fp.append(float(np.asarray(ns["fplus"]).max()))
            fm.append(float(np.asarray(ns["fminus"]).max()))
        except Exception:
            fp.append(np.nan); fm.append(np.nan)
    df = df.copy()
    df["fplus_max"] = fp
    df["fminus_max"] = fm
    df = df.dropna(subset=["fplus_max", "fminus_max"]).reset_index(drop=True)

    mu = df["mu"].values; eta = df["eta"].values
    dEta_red = df["deta_red"].values; dEta_ox = df["deta_ox"].values
    y_red = (df["ddU_red"].values * EH)
    y_ox  = (df["ddU_ox"].values * EH)

    def fit_const(target, dEta, fukui, y, n_starts=15):
        # const family: y = (a*mu + b*eta + c*dEta) * (1 + a_corr * fukui)
        rng = np.random.default_rng(SEED)
        best, bv = None, np.inf
        def predict(p):
            return (p[0]*mu + p[1]*eta + p[2]*dEta) * (1.0 + p[3]*fukui)
        def obj(p): return np.mean(np.abs(y - predict(p)))
        for s in range(n_starts):
            x0 = np.array([1.0, 1.0, 1.0, 0.0]) + (rng.normal(0, 0.5, 4) if s > 0 else 0.0)
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 30000, "fatol": 1e-7, "xatol": 1e-7})
            if r.fun < bv: bv, best = r.fun, r.x
        pred = predict(best)
        mae = float(np.mean(np.abs(y - pred)))
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        return best, mae, rmse, r2

    tests = [
        ("ddU_red", "fplus_max",  "f+_max"),
        ("ddU_red", "f2_max",     "f2_max"),
        ("ddU_ox",  "fminus_max", "f-_max"),
        ("ddU_ox",  "f2_min",     "f2_min"),
    ]
    results = []
    for tgt, fcol, label in tests:
        dEta = dEta_red if tgt == "ddU_red" else dEta_ox
        y = y_red if tgt == "ddU_red" else y_ox
        f = df[fcol].values
        best, mae, rmse, r2 = fit_const(tgt, dEta, f, y)
        results.append({"target": tgt, "fukui": label, "MAE": mae, "RMSE": rmse, "R2": r2,
                        "params": best.tolist()})
        print(f"  {tgt:<10} {label:<10}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}", flush=True)

    pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in results]
                ).to_csv(OUT / "ddu_final_step3.csv", index=False)
    return results


# ---------- Step 4-prep: final Optuna CMA-ES ----------
def final_optuna_fit(df, n_trials=12000):
    print(f"\n=== Final Optuna CMA-ES fit (n_trials={n_trials} per target) ===", flush=True)
    mu = df["mu"].values; eta = df["eta"].values
    dEta_red = df["deta_red"].values; dEta_ox = df["deta_ox"].values
    f2max = df["f2_max"].values; f2min = df["f2_min"].values
    y_red = (df["ddU_red"].values * EH)
    y_ox  = (df["ddU_ox"].values * EH)
    groups = df["molecule"].values

    PARAM_NAMES = ["theta_mu", "theta_eta", "theta_dEta", "a_corr"]

    def predict_arr(p, mu_, eta_, dEta_, fukui_):
        return (p[0]*mu_ + p[1]*eta_ + p[2]*dEta_) * (1.0 + p[3]*fukui_)

    bounds = {"theta_mu": (-100, 100), "theta_eta": (-100, 100),
              "theta_dEta": (-100, 100), "a_corr": (-50, 50)}

    def fit_target(tag, dEta, fukui, y):
        def objective(trial):
            p = np.array([trial.suggest_float(n, *bounds[n]) for n in PARAM_NAMES])
            pred = predict_arr(p, mu, eta, dEta, fukui)
            return float(np.mean(np.abs(y - pred)))

        # Warm-start from a Nelder-Mead minimum
        rng = np.random.default_rng(SEED)
        x0 = np.array([1.0, 1.0, 1.0, 0.0])
        best_x0, best_v0 = None, np.inf
        for s in range(8):
            xi = x0 + (rng.normal(0, 0.5, 4) if s > 0 else 0.0)
            r = minimize(lambda p: float(np.mean(np.abs(y - predict_arr(p, mu, eta, dEta, fukui)))),
                         xi, method="Nelder-Mead", options={"maxiter": 30000, "fatol": 1e-8})
            if r.fun < best_v0: best_v0, best_x0 = r.fun, r.x

        x_init = {n: float(best_x0[i]) for i, n in enumerate(PARAM_NAMES)}
        sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x_init, sigma0=2.0,
                                                n_startup_trials=0)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.enqueue_trial(x_init)
        # plus quelques seeds variés
        for delta in (-1.0, -0.3, 0.3, 1.0):
            study.enqueue_trial({n: x_init[n] + delta for n in PARAM_NAMES})
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = np.array([study.best_params[n] for n in PARAM_NAMES])
        pred = predict_arr(best, mu, eta, dEta, fukui)
        mae = float(np.mean(np.abs(y - pred)))
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        # CV-MAE
        pred_cv = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(y, y, groups):
            def obj_tr(p):
                return float(np.mean(np.abs(y[tr] - predict_arr(p, mu[tr], eta[tr], dEta[tr], fukui[tr]))))
            r = minimize(obj_tr, best, method="Nelder-Mead",
                         options={"maxiter": 20000, "fatol": 1e-7})
            pred_cv[te] = predict_arr(r.x, mu[te], eta[te], dEta[te], fukui[te])
        mae_cv = float(np.mean(np.abs(y - pred_cv)))
        rmse_cv = float(np.sqrt(np.mean((y - pred_cv) ** 2)))
        # intra capture
        df2 = df[["molecule"]].copy(); df2["p"] = pred; df2["y"] = y
        agg = df2.groupby("molecule").agg(t=("y","std"), p=("p","std"))
        cap = float(agg.p.mean() / agg.t.mean())
        return best, mae, rmse, r2, mae_cv, rmse_cv, cap

    out = {}
    for tag, dEta, fukui, y, fcol in [("ddU_red", dEta_red, f2max, y_red, "f2_max"),
                                        ("ddU_ox", dEta_ox, f2min, y_ox, "f2_min")]:
        t0 = time.time()
        best, mae, rmse, r2, mae_cv, rmse_cv, cap = fit_target(tag, dEta, fukui, y)
        out[tag] = {
            "fukui": fcol,
            "delta_eta_state": "reduced_sp" if tag == "ddU_red" else "oxidized_sp",
            "theta_mu": float(best[0]), "theta_eta": float(best[1]),
            "theta_dEta": float(best[2]), "a_corr": float(best[3]),
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "MAE_cv": mae_cv, "RMSE_cv": rmse_cv,
            "intra_capture": cap, "n": len(df),
        }
        print(f"  {tag}: θ_μ={best[0]:+.4f}  θ_η={best[1]:+.4f}  "
              f"θ_Δη={best[2]:+.4f}  a_corr={best[3]:+.4f}  "
              f"MAE={mae:.3f} R²={r2:.3f} CV-MAE={mae_cv:.3f} intra={cap:.0%}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    json.dump(out, open(OUT / "ddu_final_optuna.json", "w"), indent=2)
    print(f"\n  Saved -> {OUT / 'ddu_final_optuna.json'}", flush=True)
    return out


def main():
    print(f"[ddU final / marais] {time.strftime('%H:%M:%S')}", flush=True)
    df = pd.read_parquet(DATA)
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    df["g"] = (df["epsilon"]-1.0) / (2.0*df["epsilon"]+1.0)
    print(f"  n={len(df)} conformers, {df.molecule.nunique()} molecules,"
          f" {df.solvent.nunique()} solvents", flush=True)

    step2_family_search(df)
    step3_fukui_search(df)
    final_optuna_fit(df, n_trials=12000)


if __name__ == "__main__":
    main()
