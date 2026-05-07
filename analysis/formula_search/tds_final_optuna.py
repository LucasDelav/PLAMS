#!/usr/bin/env python3
"""TΔS final model — Steps 2/3/4 in one shot.

Form (homogène red/ox):
  y = (theta_mu·mu + theta_eta·eta) × (1 + a · f²_paired)

Pool: (μ, η) — minimal, k=3 par cible (θ_μ, θ_η, a).
Step 2: confirm `const` family.
Step 3: 4 Fukui variants all-atoms.
Step 4: Optuna CMA-ES final fit.

Aucun calcul ionique relaxé requis.
"""
from __future__ import annotations
import json, os, time
from concurrent.futures import ProcessPoolExecutor
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
EXCLUDED = {"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
EPSILON_R = {"Hexane":1.88,"Toluene":2.38,"Chloroform":4.81,"Tetrahydrofuran":7.58,
             "Dichloromethane":8.93,"Acetone":20.7,"Methanol":32.6,"Acetonitrile":35.7,
             "DMSO":46.7,"Water":78.4}


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


def predict_family(eps, g, mu, eta, fukui, fam_mu, fam_eta, params):
    n_mu = NPAR[fam_mu]
    p_mu = params[:n_mu]; p_eta = params[n_mu:n_mu+NPAR[fam_eta]]
    a_corr = params[n_mu+NPAR[fam_eta]]
    bmu = fam_eval(fam_mu, eps, g, p_mu)
    if bmu is None: return None
    beta = fam_eval(fam_eta, eps, g, p_eta)
    if beta is None: return None
    return (bmu*mu + beta*eta) * (1.0 + a_corr*fukui)


# === Step 2 family search ===
_DATA = None
def _init_step2(eps, g, mu, eta, f2max, f2min, y_red, y_ox, groups):
    global _DATA
    _DATA = (eps, g, mu, eta, f2max, f2min, y_red, y_ox, groups)

def _job_step2(args):
    target, fam_mu, fam_eta = args
    eps, g, mu, eta, f2max, f2min, y_red, y_ox, groups = _DATA
    fukui = f2max if target == "TdS_red" else f2min
    y = y_red if target == "TdS_red" else y_ox
    n_total = NPAR[fam_mu] + NPAR[fam_eta] + 1
    rng = np.random.default_rng(SEED)

    def fit(eps_s, g_s, mu_s, eta_s, fukui_s, y_s, n_starts=3):
        best_p, best_v = None, np.inf
        for s in range(n_starts):
            x0 = np.zeros(n_total); x0[0]=1.0; x0[NPAR[fam_mu]]=1.0
            if s>0: x0 = x0 + rng.normal(0, 0.5, size=n_total)
            def obj(p):
                pp = predict_family(eps_s, g_s, mu_s, eta_s, fukui_s, fam_mu, fam_eta, p)
                if pp is None: return 1e9
                return float(np.mean(np.abs(y_s - pp)))
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 3000, "fatol": 1e-5})
            if r.fun < best_v: best_v, best_p = r.fun, r.x
        return best_p, best_v

    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(y, y, groups):
        p, _ = fit(eps[tr], g[tr], mu[tr], eta[tr], fukui[tr], y[tr], 3)
        if p is None: pred[te] = np.nan
        else:
            pp = predict_family(eps[te], g[te], mu[te], eta[te], fukui[te], fam_mu, fam_eta, p)
            pred[te] = pp if pp is not None else np.nan
    valid = np.isfinite(pred)
    if valid.sum() < 0.9*len(y):
        return {"target":target,"fam_mu":fam_mu,"fam_eta":fam_eta,
                "MAE_cv":np.nan,"RMSE_cv":np.nan,"skipped":True}
    mae_cv = float(np.mean(np.abs(y[valid]-pred[valid])))
    rmse_cv = float(np.sqrt(np.mean((y[valid]-pred[valid])**2)))
    p_full, _ = fit(eps, g, mu, eta, fukui, y, 5)
    return {"target":target,"fam_mu":fam_mu,"fam_eta":fam_eta,
            "MAE_cv":mae_cv,"RMSE_cv":rmse_cv,"n_params":n_total,
            "params":p_full.tolist(),"skipped":False}


def step2(df):
    print("\n=== Step 2: ε-family search on (μ, η) ===", flush=True)
    eps = df["epsilon"].values; g = df["g"].values
    mu = df["mu"].values; eta = df["eta"].values
    f2max = df["f2_max"].values; f2min = df["f2_min"].values
    y_red = df["TdS_red"].values * EH
    y_ox  = df["TdS_ox"].values * EH
    groups = df["molecule"].values
    fams = ["const","onsager","born_lin","pade21_eps"]
    combos = [(t, f, f) for t in ("TdS_red","TdS_ox") for f in fams]
    n_workers = min(len(combos), os.cpu_count() or 4)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_step2,
                              initargs=(eps,g,mu,eta,f2max,f2min,y_red,y_ox,groups)) as ex:
        for r in ex.map(_job_step2, combos):
            results.append(r)
            print(f"  [OK] {r['target']:8} {r['fam_mu']:11} k={r.get('n_params','-')} "
                  f"MAE_cv={r.get('MAE_cv',float('nan')):6.3f}", flush=True)
    print(f"  step 2 total: {time.time()-t0:.1f}s", flush=True)
    res = pd.DataFrame([{k:v for k,v in r.items() if k!="params"} for r in results])
    res.to_csv(OUT/"tds_final_step2.csv", index=False)
    print("\n  Family ranking:")
    for tgt in ("TdS_red","TdS_ox"):
        sub = res[(res.target==tgt) & ~res.skipped].sort_values("MAE_cv")
        print(f"\n  -- {tgt} --")
        print(sub[["fam_mu","n_params","MAE_cv","RMSE_cv"]].to_string(
            index=False, float_format=lambda x: f"{x:7.3f}"))
    return res


# === Step 3 Fukui search ===
def step3(df):
    print("\n=== Step 3: Fukui variants (all-atoms) ===", flush=True)
    PRED = REPO/"dataset_predict_dg"
    fp, fm = [], []
    for _, row in df.iterrows():
        p = PRED/row.solvent/row.molecule/f"{row.molecule}_conf{int(row.conformer)}_prediction.json"
        try:
            d = json.load(open(p))
            cdft = d["states"]["neutral_sp"]["cdft"]
            fp.append(float(np.asarray(cdft["fplus"]).max()))
            fm.append(float(np.asarray(cdft["fminus"]).max()))
        except Exception:
            fp.append(np.nan); fm.append(np.nan)
    df = df.copy(); df["fplus_max"]=fp; df["fminus_max"]=fm
    df = df.dropna(subset=["fplus_max","fminus_max"]).reset_index(drop=True)
    mu = df["mu"].values; eta = df["eta"].values
    y_red = df["TdS_red"].values * EH
    y_ox  = df["TdS_ox"].values * EH

    def fit_const(fukui, y, n_starts=15):
        rng = np.random.default_rng(SEED)
        best, bv = None, np.inf
        def predict(p): return (p[0]*mu + p[1]*eta) * (1.0 + p[2]*fukui)
        def obj(p): return np.mean(np.abs(y - predict(p)))
        for s in range(n_starts):
            x0 = np.array([1.0,1.0,0.0]) + (rng.normal(0,0.5,3) if s>0 else 0.0)
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter":30000,"fatol":1e-7,"xatol":1e-7})
            if r.fun < bv: bv, best = r.fun, r.x
        pred = predict(best)
        mae = float(np.mean(np.abs(y-pred)))
        rmse = float(np.sqrt(np.mean((y-pred)**2)))
        ss = np.sum((y-pred)**2); st = np.sum((y-y.mean())**2)
        return best, mae, rmse, 1-ss/st

    tests = [
        ("TdS_red", "fplus_max",  "f+_max"),
        ("TdS_red", "f2_max",     "f2_max"),
        ("TdS_ox",  "fminus_max", "f-_max"),
        ("TdS_ox",  "f2_min",     "f2_min"),
    ]
    results = []
    for tgt, fcol, label in tests:
        y = y_red if tgt=="TdS_red" else y_ox
        f = df[fcol].values
        best, mae, rmse, r2 = fit_const(f, y)
        results.append({"target":tgt,"fukui":label,"MAE":mae,"RMSE":rmse,"R2":r2,
                        "params":best.tolist()})
        print(f"  {tgt:<8}  {label:<10}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}", flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!="params"} for r in results]
                ).to_csv(OUT/"tds_final_step3.csv", index=False)
    return results


# === Step 4 final Optuna fit ===
def step4_final(df, n_trials=12000):
    print(f"\n=== Step 4: Optuna CMA-ES (n_trials={n_trials} per target) ===", flush=True)
    PRED = REPO/"dataset_predict_dg"
    fp, fm = [], []
    for _, row in df.iterrows():
        p = PRED/row.solvent/row.molecule/f"{row.molecule}_conf{int(row.conformer)}_prediction.json"
        try:
            d = json.load(open(p))
            cdft = d["states"]["neutral_sp"]["cdft"]
            fp.append(float(np.asarray(cdft["fplus"]).max()))
            fm.append(float(np.asarray(cdft["fminus"]).max()))
        except Exception:
            fp.append(np.nan); fm.append(np.nan)
    df = df.copy(); df["fplus_max"]=fp; df["fminus_max"]=fm
    df = df.dropna(subset=["fplus_max","fminus_max"]).reset_index(drop=True)

    mu = df["mu"].values; eta = df["eta"].values
    f2max = df["f2_max"].values; f2min = df["f2_min"].values
    y_red = df["TdS_red"].values * EH
    y_ox  = df["TdS_ox"].values * EH
    groups = df["molecule"].values
    PARAM_NAMES = ["theta_mu", "theta_eta", "a_corr"]
    bounds = {"theta_mu":(-100,100), "theta_eta":(-100,100), "a_corr":(-50,50)}

    def predict_arr(p, mu_, eta_, fukui_):
        return (p[0]*mu_ + p[1]*eta_) * (1.0 + p[2]*fukui_)

    def fit_target(tag, fukui, y):
        def objective(trial):
            p = np.array([trial.suggest_float(n, *bounds[n]) for n in PARAM_NAMES])
            return float(np.mean(np.abs(y - predict_arr(p, mu, eta, fukui))))
        rng = np.random.default_rng(SEED)
        x0 = np.array([1.0, 1.0, 0.0])
        best_x0, best_v0 = None, np.inf
        for s in range(8):
            xi = x0 + (rng.normal(0, 0.5, 3) if s>0 else 0.0)
            r = minimize(lambda p: float(np.mean(np.abs(y-predict_arr(p, mu, eta, fukui)))),
                         xi, method="Nelder-Mead", options={"maxiter":30000,"fatol":1e-8})
            if r.fun < best_v0: best_v0, best_x0 = r.fun, r.x
        x_init = {n: float(best_x0[i]) for i,n in enumerate(PARAM_NAMES)}
        sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x_init, sigma0=2.0,
                                                n_startup_trials=0)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.enqueue_trial(x_init)
        for delta in (-1.0,-0.3,0.3,1.0):
            study.enqueue_trial({n: x_init[n]+delta for n in PARAM_NAMES})
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = np.array([study.best_params[n] for n in PARAM_NAMES])
        pred = predict_arr(best, mu, eta, fukui)
        mae = float(np.mean(np.abs(y-pred)))
        rmse = float(np.sqrt(np.mean((y-pred)**2)))
        ss = np.sum((y-pred)**2); st = np.sum((y-y.mean())**2)
        r2 = 1.0 - ss/st
        pred_cv = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(y,y,groups):
            def obj_tr(p):
                return float(np.mean(np.abs(y[tr]-predict_arr(p, mu[tr], eta[tr], fukui[tr]))))
            r = minimize(obj_tr, best, method="Nelder-Mead",
                         options={"maxiter":20000,"fatol":1e-7})
            pred_cv[te] = predict_arr(r.x, mu[te], eta[te], fukui[te])
        mae_cv = float(np.mean(np.abs(y-pred_cv)))
        rmse_cv = float(np.sqrt(np.mean((y-pred_cv)**2)))
        df2 = df[["molecule"]].copy(); df2["p"]=pred; df2["y"]=y
        agg = df2.groupby("molecule").agg(t=("y","std"),p=("p","std"))
        cap = float(agg.p.mean()/agg.t.mean())
        return best, mae, rmse, r2, mae_cv, rmse_cv, cap

    out = {}
    for tag, fukui, y, fcol in [("TdS_red", f2max, y_red, "f2_max"),
                                 ("TdS_ox", f2min, y_ox, "f2_min")]:
        t0 = time.time()
        best, mae, rmse, r2, mae_cv, rmse_cv, cap = fit_target(tag, fukui, y)
        out[tag] = {
            "fukui": fcol,
            "theta_mu": float(best[0]), "theta_eta": float(best[1]),
            "a_corr": float(best[2]),
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "MAE_cv": mae_cv, "RMSE_cv": rmse_cv,
            "intra_capture": cap, "n": len(df),
        }
        print(f"  {tag}: θ_μ={best[0]:+.4f}  θ_η={best[1]:+.4f}  a_corr={best[2]:+.4f}  "
              f"MAE={mae:.3f} R²={r2:.3f} CV-MAE={mae_cv:.3f} intra={cap:.0%}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    json.dump(out, open(OUT/"tds_final_optuna.json","w"), indent=2)
    print(f"\nSaved -> {OUT/'tds_final_optuna.json'}", flush=True)
    return out


def main():
    print(f"[TΔS final / marais] {time.strftime('%H:%M:%S')}", flush=True)
    df = pd.read_parquet(DATA)
    # Add TdS columns from merged_dataset
    m = pd.read_parquet(REPO/"analysis/formula_search/data/merged_dataset.parquet")
    m = m[~m.molecule.isin(EXCLUDED)]
    df = df.merge(m[['molecule','solvent','conformer','TdS_red','TdS_ox']],
                  on=['molecule','solvent','conformer'], how='left')
    df = df.dropna(subset=['TdS_red','TdS_ox','mu','eta','f2_max','f2_min']).reset_index(drop=True)
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    df["g"] = (df["epsilon"]-1.0)/(2.0*df["epsilon"]+1.0)
    print(f"  n={len(df)}, n_mol={df.molecule.nunique()}, n_solv={df.solvent.nunique()}",
          flush=True)
    print(f"  TdS amplitudes (kJ/mol): "
          f"red |·|_mean={(df['TdS_red']*EH).abs().mean():.2f}  "
          f"ox |·|_mean={(df['TdS_ox']*EH).abs().mean():.2f}", flush=True)
    step2(df)
    step3(df)
    step4_final(df, n_trials=12000)


if __name__ == "__main__":
    main()
