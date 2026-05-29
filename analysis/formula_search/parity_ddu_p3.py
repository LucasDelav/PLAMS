#!/usr/bin/env python
"""
Parity plots ΔΔU (partie 3, acétonitrile, 154 conf) — style maison.
Modèle retenu : red (μ,η)×(1+a·f²), ox (μ,η) linéaire (Fukui dégénéré en ox).
Variante +RRHO : + λ·ΔU_RRHO + γ (ΔU_RRHO = Σ U_vib(ω_ion_sp) − Σ U_vib(ω_neutre_sp)),
pour DÉMONTRER le gain qu'apporterait le terme physique (au prix des Hessiens ioniques).
"""
import sys, json; sys.path.insert(0,"/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
import numpy as np, pandas as pd, optuna, warnings
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, FIGSIZE_DOUBLE_V, C_RED_DARK, C_BLUE_DARK, C_GRAY)
import matplotlib.pyplot as plt
apply_style()
EH=2625.499638; SEED=0; EXCL={"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
FIG="/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"
PRED=Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/dataset_predict_dg/Acetonitrile")
HCKJ=0.011962566; KT_CM=207.22
def Uvib(modes):
    s=0.0
    for mo in modes:
        w=mo["frequency_cm1"]
        if w<=10: continue
        s+=HCKJ*w*(0.5+1.0/np.expm1(w/KT_CM))
    return s
m=pd.read_parquet("data/merged_dataset.parquet"); dd=pd.read_parquet("data/ddu_delta_descriptors.parquet")
m=m[(m.solvent=="Acetonitrile")&(~m.molecule.isin(EXCL))].copy()
dd=dd[(dd.solvent=="Acetonitrile")&(~dd.molecule.isin(EXCL))].copy()
df=m.merge(dd[["molecule","solvent","conformer","f2_min"]],on=["molecule","solvent","conformer"]).reset_index(drop=True)
rr=[];ro=[];keep=[]
for _,r in df.iterrows():
    p=PRED/r.molecule/f"{r.molecule}_conf{int(r.conformer)}_prediction.json"
    if not p.exists(): keep.append(False); continue
    j=json.load(open(p))["states"]
    try:
        Un=Uvib(j["neutral_sp"]["vibrations"]["modes"])
        rr.append(Uvib(j["reduced_sp"]["vibrations"]["modes"])-Un)
        ro.append(Uvib(j["oxidized_sp"]["vibrations"]["modes"])-Un); keep.append(True)
    except Exception: keep.append(False)
df=df[keep].reset_index(drop=True); df["rrho_red"]=rr; df["rrho_ox"]=ro
g=df.molecule.values
def _muted(c):
    h,s,v=mcolors.rgb_to_hsv(np.array(c[:3])); return (*mcolors.hsv_to_rgb((h,s*0.45,min(v,0.78))),1.0)
mols=sorted(df.molecule.unique()); PAL={mo:_muted(c) for mo,c in zip(mols,plt.cm.gist_rainbow(np.linspace(0,1,len(mols))))}
pcols=[PAL[mo] for mo in df.molecule]
def oof(tgt,fcol,rrho_col,nt=6000):
    y=df[tgt].values*EH; Xr=df[["mu","eta"]].values.astype(float); sc=Xr.std(0); X=Xr/sc
    f=df[fcol].values if fcol else np.zeros(len(y))
    use_f=fcol is not None; use_r=rrho_col is not None
    if use_r: R=df[rrho_col].values; Rsc=R.std(); Rn=R/Rsc
    def pred(p,X,f,Rn):
        b=(X@p[:2]); 
        if use_f: b=b*(1+p[2]*f)
        if use_r: b=b+p[-2]*Rn+p[-1]
        return b
    names=["t0","t1"]+(["a"] if use_f else [])+(["lam","gam"] if use_r else [])
    bnd={"t0":(-1000,1000),"t1":(-1000,1000),"a":(-50,50),"lam":(-1000,1000),"gam":(-1000,1000)}
    npar=len(names); rng=np.random.default_rng(SEED)
    x0=np.r_[1,1,([0.] if use_f else []),([0.,0.] if use_r else [])]; bx,bv=x0,np.inf
    def mae(p,X,f,Rn,y): return float(np.mean(np.abs(y-pred(p,X,f,Rn))))
    for s in range(8):
        xi=x0+(rng.normal(0,.5,npar) if s else 0)
        r_=minimize(lambda p:mae(p,X,f,(Rn if use_r else None),y),xi,method="Nelder-Mead",options={"maxiter":30000,"fatol":1e-9})
        if r_.fun<bv:bv,bx=r_.fun,r_.x
    if npar>=2:
        xi={n:float(np.clip(bx[i],*bnd[n])) for i,n in enumerate(names)}
        st=optuna.create_study(direction="minimize",sampler=optuna.samplers.CmaEsSampler(seed=SEED,x0=xi,sigma0=2.0,n_startup_trials=0))
        st.enqueue_trial(xi); st.optimize(lambda t:mae(np.array([t.suggest_float(n,*bnd[n]) for n in names]),X,f,(Rn if use_r else None),y),n_trials=nt)
        best=np.array([st.best_params[n] for n in names])
        if bv<mae(best,X,f,(Rn if use_r else None),y): best=bx
    else: best=bx
    pc=np.zeros_like(y)
    for tr,te in GroupKFold(5).split(y,y,g):
        r_=minimize(lambda p:mae(p,X[tr],f[tr],(Rn[tr] if use_r else None),y[tr]),best,method="Nelder-Mead",options={"maxiter":20000,"fatol":1e-8})
        pc[te]=pred(r_.x,X[te],f[te],(Rn[te] if use_r else None))
    return y,pc
def panel(tgt,fcol,rrho_col,color,fname):
    y,pc=oof(tgt,fcol,rrho_col)
    fig,ax=plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    scat=dict(SCATTER_KW); scat.pop("c",None)
    ax.scatter(y,pc,c=pcols,**scat); regression_with_band(ax,y,pc,color=color)
    allv=np.concatenate([y,pc]); mar=(allv.max()-allv.min())*0.12; lims=[allv.min()-mar,allv.max()+mar]
    ax.plot(lims,lims,"--",color=C_GRAY,lw=0.8,alpha=0.5,zorder=0); ax.set_xlim(lims); ax.set_ylim(lims)
    r2=1-np.sum((y-pc)**2)/np.sum((y-y.mean())**2)
    stats_text(ax,r2,np.mean(np.abs(pc-y)),len(y))
    ax.set_xlabel(r"$\Delta\Delta U^{\mathrm{DFT}}$ (kJ/mol)"); ax.set_ylabel(r"$\Delta\Delta U$ prédit (kJ/mol)"); setup_grid(ax)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}.png"); fig.savefig(f"{FIG}/{fname}.pdf"); plt.close()
    print(f"  {fname}: MAE={np.mean(np.abs(pc-y)):.3f} R2={r2:.3f}")
# modèle retenu (sans RRHO) : forme symétrique red (μ,η)x(1+a·f²_max), ox (μ,η)x(1+a·f²_min)
panel("ddU_red","f2_max",None,C_BLUE_DARK,"parity_ddu_red")
panel("ddU_ox","f2_min",None,C_RED_DARK,"parity_ddu_ox")
# variante +RRHO (preuve du gain)
panel("ddU_red","f2_max","rrho_red",C_BLUE_DARK,"parity_ddu_red_rrho")
panel("ddU_ox","f2_min","rrho_ox",C_RED_DARK,"parity_ddu_ox_rrho")
