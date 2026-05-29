#!/usr/bin/env python
"""Screening multi-solvant des fonctions de correction de solvant (partie 3 fold-in).
Formes ACTUELLES de la partie 3 ; chaque coefficient β_X(ε) suit une famille testée.
Protocole identique à la partie 3 : objectif MAE, CMA-ES warm-start Nelder-Mead,
out-of-fold GroupKFold-5 par molécule.

Familles testées (appliquées simultanément aux coefficients linéaires) :
  const, lin_g (Born), onsager, pade11, pade21, log, exp, sigmoid, tanh, power
"""
import sys, json, time, warnings, argparse
import numpy as np, pandas as pd, optuna
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)

EH = 2625.499638; SEED = 0
EXCL = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPS = {"Hexane":1.88,"Toluene":2.38,"Chloroform":4.81,"Tetrahydrofuran":7.58,
       "Dichloromethane":8.93,"Acetone":20.7,"Methanol":32.6,"Acetonitrile":35.7,
       "DMSO":46.7,"Water":78.4}

# ---------- familles de dépendance solvant β(ε) ----------
# chaque famille : (n_params, fonction(p_slice, eps, g, gons) -> coeff par ligne)
def _pade11(p, eps, g, gons):
    den = 1.0 + p[2]*g
    return (p[0] + p[1]*g) / den
def _pade21(p, eps, g, gons):
    den = 1.0 + p[3]*g
    return (p[0] + p[1]*g + p[2]*g*g) / den
FAMILIES = {
    "const":   (1, lambda p,eps,g,gons: p[0] + 0*g),
    "lin_g":   (2, lambda p,eps,g,gons: p[0] + p[1]*g),
    "onsager": (2, lambda p,eps,g,gons: p[0] + p[1]*gons),
    "pade11":  (3, _pade11),
    "pade21":  (4, _pade21),
    "log":     (2, lambda p,eps,g,gons: p[0] + p[1]*np.log(eps)),
    "exp":     (3, lambda p,eps,g,gons: p[0] + p[1]*(1.0 - np.exp(-eps/p[2]))),
    "sigmoid": (4, lambda p,eps,g,gons: p[0] + p[1]/(1.0 + np.exp(-(eps-p[2])/p[3]))),
    "tanh":    (4, lambda p,eps,g,gons: p[0] + p[1]*np.tanh((eps-p[2])/p[3])),
    "power":   (3, lambda p,eps,g,gons: p[0] + p[1]*np.power(eps, p[2])),
}
# bornes par type de paramètre (index dans le slice de la famille)
def fam_bounds(fam):
    n = FAMILIES[fam][0]
    b = [(-2000.0, 2000.0)]*n
    if fam == "pade11": b[2] = (-1.8, 30.0)
    if fam == "pade21": b[3] = (-1.8, 30.0)
    if fam == "exp":    b[2] = (0.5, 200.0)             # tau
    if fam == "sigmoid": b[2] = (1.0, 80.0); b[3] = (0.5, 60.0)   # eps0, s
    if fam == "tanh":    b[2] = (1.0, 80.0); b[3] = (0.5, 60.0)
    if fam == "power":   b[2] = (-2.0, 2.0)             # exposant
    return b

# ---------- données ----------
def load():
    m = pd.read_parquet("data/merged_dataset.parquet")
    dd = pd.read_parquet("data/ddu_delta_descriptors.parquet")
    lv = pd.read_parquet("data/lvc_multisolvent_thr100.parquet")
    m = m[~m.molecule.isin(EXCL)].copy(); dd = dd[~dd.molecule.isin(EXCL)].copy()
    df = m.merge(dd[["molecule","solvent","conformer","f2_min"]],
                 on=["molecule","solvent","conformer"], how="inner")
    df = df.merge(lv, on=["molecule","solvent","conformer"], how="inner").reset_index(drop=True)
    df["eps"] = df.solvent.map(EPS)
    df["g"]   = (df.eps-1.0)/(2.0*df.eps+1.0)
    df["gons"]= (df.eps-1.0)/(df.eps+2.0)
    df["omega_plus"]   = (df.eta + 2*df.mu)**2/(8*df.eta)
    df["inv_omega_roy"]= 2*df.eta/(df.eta/2 - df.mu)**2
    return df

# ---------- modèle : sum_d beta_d(eps) * X_d  x (1 + a*f) ----------
def make_predict(fam, ncoef, use_fukui):
    nfp = FAMILIES[fam][0]; fn = FAMILIES[fam][1]
    def predict(p, X, f, eps, g, gons):
        base = np.zeros(X.shape[0])
        for d in range(ncoef):
            pc = p[d*nfp:(d+1)*nfp]
            base = base + fn(pc, eps, g, gons) * X[:, d]
        if use_fukui:
            base = base * (1.0 + p[ncoef*nfp]*f)
        return base
    npar = ncoef*nfp + (1 if use_fukui else 0)
    return predict, npar

def penalty(fam, p, ncoef, g):
    # garde-fou dénominateur Padé : 1+b*g > 0.05 sur tous les solvants
    if fam not in ("pade11","pade21"): return 0.0
    nfp = FAMILIES[fam][0]; bi = 2 if fam=="pade11" else 3
    bad = 0.0
    for d in range(ncoef):
        b = p[d*nfp+bi]
        den = 1.0 + b*g
        if den.min() < 0.05: bad += (0.05 - den.min())*1e4
    return bad

def fit_family(df, cols, fcol, tgt, fam, use_fukui, n_trials, seed=SEED):
    y = (df[tgt].values*EH).astype(float)
    Xr = df[cols].values.astype(float); sc = Xr.std(0); sc[sc==0]=1; X = Xr/sc
    f = df[fcol].values.astype(float) if (use_fukui and fcol) else np.zeros(len(y))
    eps, g, gons = df.eps.values, df.g.values, df.gons.values
    groups = df.molecule.values
    ncoef = len(cols)
    predict, npar = make_predict(fam, ncoef, use_fukui)
    fb = fam_bounds(fam)
    bounds = []
    for d in range(ncoef): bounds += fb
    if use_fukui: bounds += [(-50.0, 50.0)]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    def mae(p, Xs, fs, es, gs, gos, ys):
        return float(np.mean(np.abs(ys - predict(p, Xs, fs, es, gs, gos)))) + penalty(fam, p, ncoef, g)
    # warm start Nelder-Mead
    rng = np.random.default_rng(seed)
    # x0 : coef constants ~1 (standardisés), shape params au milieu des bornes
    x0 = np.clip(np.array([(l+h)/2 if (abs(l)>100 or abs(h)>100) else (l+h)/2 for l,h in bounds]), lo, hi)
    # mettre les "a0" de chaque coef à ~ écart-type cible / ncoef, le reste 0
    x0[:] = 0.0
    nfp = FAMILIES[fam][0]
    for d in range(ncoef): x0[d*nfp] = 1.0
    for i,(l,h) in enumerate(bounds):
        if l>0.4: x0[i] = (l+h)/2  # tau, s, eps0 etc.
    x0 = np.clip(x0, lo, hi)
    bx, bv = x0.copy(), np.inf
    for s in range(8):
        xi = np.clip(x0 + (rng.normal(0,0.5,npar) if s else 0), lo, hi)
        r = minimize(lambda p: mae(p,X,f,eps,g,gons,y), xi, method="Nelder-Mead",
                     options={"maxiter":30000,"fatol":1e-8,"xatol":1e-8})
        if r.fun < bv: bv, bx = r.fun, r.x
    # CMA-ES
    if npar >= 2:
        names = [f"p{i}" for i in range(npar)]
        xi = {n: float(np.clip(bx[i], lo[i], hi[i])) for i,n in enumerate(names)}
        st = optuna.create_study(direction="minimize",
              sampler=optuna.samplers.CmaEsSampler(seed=seed, x0=xi, sigma0=1.0, n_startup_trials=0))
        st.enqueue_trial(xi)
        st.optimize(lambda t: mae(np.array([t.suggest_float(n, lo[i], hi[i]) for i,n in enumerate(names)]),
                                  X,f,eps,g,gons,y), n_trials=n_trials)
        best = np.array([st.best_params[n] for n in names])
        if bv < mae(best,X,f,eps,g,gons,y): best = bx
    else:
        best = bx
    # OOF GroupKFold-5 (refit Nelder-Mead par fold)
    pc = np.zeros_like(y)
    for trn, te in GroupKFold(5).split(y, y, groups):
        r = minimize(lambda p: mae(p,X[trn],f[trn],eps[trn],g[trn],gons[trn],y[trn]),
                     best, method="Nelder-Mead", options={"maxiter":20000,"fatol":1e-7})
        pc[te] = predict(r.x, X[te], f[te], eps[te], g[te], gons[te])
    mae_cv = float(np.mean(np.abs(y - pc)))
    r2 = 1 - np.sum((y-pc)**2)/np.sum((y-y.mean())**2)
    n = len(y); k = npar
    aic = n*np.log(np.mean((y-pc)**2)) + 2*k
    return dict(fam=fam, mae=mae_cv, rmse=float(np.sqrt(np.mean((y-pc)**2))),
                r2=r2, npar=k, aic=aic, pred=pc, best=best)

# spec des cibles (forme partie 3)
TARGETS = {
    "EA":        (["mu","eta","omega_plus"],    "f2_max", True),
    "IP":        (["mu","eta","inv_omega_roy"], "f2_min", True),
    "Erelax_red":(["eta","lvc_red_100"],        "f2_max", True),
    "Erelax_ox": (["eta","lvc_ox_100"],         "f2_min", True),
    "ddU_red":   (["mu","eta"],                 "f2_max", True),
    "ddU_ox":    (["mu","eta"],                 "f2_min", True),
    "TdS_red":   (["mu","eta"],                 None,     False),
    "TdS_ox":    (["mu","eta"],                 None,     False),
}

def trials_for(npar):
    return int(min(12000, max(6000, 700*npar)))

def fit_one(arg):
    import os
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
    tgt, fam = arg
    df = load()
    cols, fcol, uf = TARGETS[tgt]
    npar = FAMILIES[fam][0]*len(cols) + (1 if uf else 0)
    t0 = time.time()
    r = fit_family(df, cols, fcol, tgt, fam, uf, trials_for(npar))
    r["target"] = tgt; r["sec"] = time.time()-t0
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="chronométrer un fit IP pade11")
    ap.add_argument("--full", action="store_true", help="screen complet parallélisé")
    ap.add_argument("--families", default="const,lin_g,onsager,pade11,pade21,log,exp,sigmoid,tanh,power")
    ap.add_argument("--trials", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    if args.bench:
        df = load()
        print(f"[data] n={len(df)}", flush=True)
        cols, fcol, uf = TARGETS["IP"]
        t0 = time.time(); r = fit_family(df, cols, fcol, "IP", "pade11", uf, args.trials)
        print(f"[bench] IP pade11 ({r['npar']}p,{args.trials}t): MAE={r['mae']:.2f} R2={r['r2']:.3f} --> {time.time()-t0:.1f}s", flush=True)
        sys.exit(0)
    if args.full:
        import os
        from concurrent.futures import ProcessPoolExecutor
        fams = args.families.split(",")
        jobs = [(t, f) for t in TARGETS for f in fams]
        nw = args.workers or min(os.cpu_count() or 4, 10)
        print(f"[full] {len(jobs)} fits ({len(TARGETS)} cibles × {len(fams)} familles), {nw} workers", flush=True)
        t0 = time.time(); rows = []; preds = {}
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for r in ex.map(fit_one, jobs):
                rows.append({k: r[k] for k in ["target","fam","mae","rmse","r2","npar","aic","sec"]})
                preds[f"{r['target']}|{r['fam']}"] = r["pred"]
                print(f"  {r['target']:11s} {r['fam']:8s} MAE={r['mae']:6.2f} R2={r['r2']:+.3f} "
                      f"AIC={r['aic']:7.0f} ({r['npar']}p,{r['sec']:.0f}s)", flush=True)
        tab = pd.DataFrame(rows).sort_values(["target","mae"])
        tab.to_csv("results/solvent_screen_p3.csv", index=False)
        np.savez_compressed("results/solvent_screen_p3_preds.npz", **preds)
        print(f"\n[done] {len(jobs)} fits en {time.time()-t0:.0f}s → results/solvent_screen_p3.csv", flush=True)
        print("\n=== meilleure famille par cible (CV-MAE) ===", flush=True)
        for t in TARGETS:
            sub = tab[tab.target==t].sort_values("mae")
            best = sub.iloc[0]
            print(f"  {t:11s} → {best.fam:8s} MAE={best.mae:.2f} (const={sub[sub.fam=='const'].mae.iloc[0]:.2f})", flush=True)
        sys.exit(0)
