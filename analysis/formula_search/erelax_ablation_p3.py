#!/usr/bin/env python
"""
Ablation systématique du modèle E_relax = (pool linéaire)·(1+a·f²).
5 ingrédients : μ, η, μ², LVC (pool linéaire) + Fukui (correction multiplicative).
On retire 1, 2, 3 ingrédients (toutes combinaisons, ≥1 terme linéaire conservé),
fit MAE-objectif (Optuna CMA-ES warm-start Nelder-Mead) + CV GroupKFold-5 par molécule.
But : classer l'importance de chaque terme (Δ CV-MAE quand on le retire).
"""
import sys; sys.path.insert(0,"/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/analysis/formula_search")
import numpy as np, pandas as pd, optuna, warnings
from itertools import combinations
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
EH=2625.499638; SEED=0; EXCL={"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
m=pd.read_parquet("data/merged_dataset.parquet"); dd=pd.read_parquet("data/ddu_delta_descriptors.parquet")
m=m[(m.solvent=="Acetonitrile")&(~m.molecule.isin(EXCL))].copy()
dd=dd[(dd.solvent=="Acetonitrile")&(~dd.molecule.isin(EXCL))].copy()
df=m.merge(dd[["molecule","solvent","conformer","f2_min"]],on=["molecule","solvent","conformer"]).reset_index(drop=True)
lvc=pd.read_parquet("data/lvc_acetonitrile_thr100.parquet")  # déjà en kJ/mol
df=df.merge(lvc[["molecule","solvent","conformer","lvc_ox_100","lvc_red_100"]],on=["molecule","solvent","conformer"]).reset_index(drop=True)
df["mu2"]=df.mu**2; g=df.molecule.values
LIN=["mu","eta","mu2","lvc"]; ALL=LIN+["fukui"]

def fitm(lincols, fukui_on, y, fcol, nt=2500):
    """Fit (X@theta)*(1+a f) MAE-objectif + CV-MAE GroupKFold-5. lincols non vide."""
    Xr=df[lincols].values.astype(float); sc=Xr.std(0); sc[sc==0]=1; X=Xr/sc
    f=df[fcol].values; k=X.shape[1]
    npar=k+(1 if fukui_on else 0)
    def pred(p,Xx,ff):
        v=Xx@p[:k]
        return v*(1+p[k]*ff) if fukui_on else v
    def mae(p,Xx,ff,yy): return float(np.mean(np.abs(yy-pred(p,Xx,ff))))
    rng=np.random.default_rng(SEED); x0=np.r_[np.ones(k),([0.] if fukui_on else [])]; bx,bv=x0,np.inf
    for s in range(6):
        xi=x0+(rng.normal(0,.5,npar) if s else 0)
        rr=minimize(lambda p:mae(p,X,f,y),xi,method="Nelder-Mead",options={"maxiter":20000,"fatol":1e-8})
        if rr.fun<bv: bv,bx=rr.fun,rr.x
    names=[f"t{i}" for i in range(k)]+(["a"] if fukui_on else [])
    bnd={n:(-1000,1000) for n in names[:k]};
    if fukui_on: bnd["a"]=(-50,50)
    if npar >= 2:  # CMA-ES exige >=2 dimensions
        xi={n:float(np.clip(bx[i],bnd[n][0],bnd[n][1])) for i,n in enumerate(names)}  # warm-start dans les bornes
        st=optuna.create_study(direction="minimize",sampler=optuna.samplers.CmaEsSampler(seed=SEED,x0=xi,sigma0=2.0,n_startup_trials=0))
        st.enqueue_trial(xi)
        st.optimize(lambda t:mae(np.array([t.suggest_float(n,*bnd[n]) for n in names]),X,f,y),n_trials=nt)
        best=np.array([st.best_params[n] for n in names])
        if bv<mae(best,X,f,y): best=bx
    else:  # 1 paramètre : Nelder-Mead multi-start suffit
        best=bx
    pc=np.zeros_like(y)
    for tr,te in GroupKFold(5).split(y,y,g):
        rr=minimize(lambda p:mae(p,X[tr],f[tr],y[tr]),best,method="Nelder-Mead",options={"maxiter":15000,"fatol":1e-7})
        pc[te]=pred(rr.x,X[te],f[te])
    return float(np.mean(np.abs(y-pc)))

def run(target, lvc_col, fcol):
    y=df[target].values*EH
    dfn={"lvc":lvc_col}  # map ingrédient -> colonne df
    def cols_of(subset):
        lin=[("lvc_col" if c=="lvc" else c) for c in subset if c in LIN]
        return [ (lvc_col if c=="lvc" else c) for c in subset if c in LIN], ("fukui" in subset)
    full_lin=[ (lvc_col if c=="lvc" else c) for c in LIN]
    full=fitm(full_lin, True, y, fcol)
    print(f"\n########## {target} ##########")
    print(f"MODÈLE COMPLET (μ,η,μ²,LVC,Fukui) : CV-MAE = {full:.3f} kJ/mol")
    res={}
    for nrem in (1,2,3):
        print(f"\n--- retrait de {nrem} ingrédient(s) ---")
        rows=[]
        for rem in combinations(ALL, nrem):
            keep=[c for c in ALL if c not in rem]
            lin=[ (lvc_col if c=="lvc" else c) for c in keep if c in LIN]
            if not lin: continue
            mae=fitm(lin, ("fukui" in keep), y, fcol)
            rows.append((", ".join(rem), mae, mae-full))
        for name,mae,d in sorted(rows,key=lambda x:x[1]):
            print(f"   retiré: {name:28s} CV-MAE={mae:6.3f}  Δ={d:+.3f}")
        res[nrem]=rows
    # importance par retrait simple
    print(f"\n>>> Importance (retrait simple, Δ CV-MAE décroissant) :")
    for name,mae,d in sorted(res[1],key=lambda x:-x[2]):
        print(f"     {name:8s} : Δ={d:+.3f}")
    return full

run("Erelax_ox","lvc_ox_100","f2_min")
run("Erelax_red","lvc_red_100","f2_max")
print("\n=== FIN ===")
