#!/usr/bin/env python
"""TΔS (acétonitrile,154) : (μ,η)×(1+a·f²) + test ΔS_RRHO (échoue, contrairement à ΔΔU)
+ baselines. Puis ΔG combiné par composante + parity plots (style maison)."""
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
apply_style(); EH=2625.499638; SEED=0; EXCL={"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
FIG="/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"
PRED=Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/dataset_predict_dg/Acetonitrile")
KT_CM=207.22; RTkJ=2.47880  # R·T à 298.15 K en kJ/mol
def TSvib(modes):
    s=0.0
    for mo in modes:
        w=mo["frequency_cm1"]
        if w<=10: continue
        u=w/KT_CM; s+=u/np.expm1(u)-np.log1p(-np.exp(-u))
    return RTkJ*s
m=pd.read_parquet("data/merged_dataset.parquet"); dd=pd.read_parquet("data/ddu_delta_descriptors.parquet")
m=m[(m.solvent=="Acetonitrile")&(~m.molecule.isin(EXCL))].copy()
dd=dd[(dd.solvent=="Acetonitrile")&(~dd.molecule.isin(EXCL))].copy()
df=m.merge(dd[["molecule","solvent","conformer","f2_min"]],on=["molecule","solvent","conformer"]).reset_index(drop=True)
lvc=pd.read_parquet("data/lvc_acetonitrile_thr100.parquet")
df=df.merge(lvc[["molecule","solvent","conformer","lvc_ox_100","lvc_red_100"]],on=["molecule","solvent","conformer"]).reset_index(drop=True)
# ΔTS_vib RRHO (SP verticaux)
tr=[];to=[];keep=[]
for _,r in df.iterrows():
    p=PRED/r.molecule/f"{r.molecule}_conf{int(r.conformer)}_prediction.json"
    if not p.exists(): keep.append(False); continue
    j=json.load(open(p))["states"]
    try:
        Sn=TSvib(j["neutral_sp"]["vibrations"]["modes"])
        tr.append(TSvib(j["reduced_sp"]["vibrations"]["modes"])-Sn)
        to.append(TSvib(j["oxidized_sp"]["vibrations"]["modes"])-Sn); keep.append(True)
    except Exception: keep.append(False)
df=df[keep].reset_index(drop=True); df["dtsvib_red"]=tr; df["dtsvib_ox"]=to
df["inv_omega_roy"]=2*df.eta/(df.eta/2-df.mu)**2  # 1/ω⁻ Gázquez (IP)
df["omega_plus"]=(df.eta+2*df.mu)**2/(8*df.eta)   # ω⁺ Gázquez (EA, forme du rapport)
g=df.molecule.values
def fit(cols,fcol,tgt,add=None,nt=6000,ret_pred=False):
    y=df[tgt].values*EH; Xr=df[cols].values.astype(float); sc=Xr.std(0); sc[sc==0]=1; X=Xr/sc
    f=df[fcol].values if fcol else np.zeros(len(y)); k=X.shape[1]
    uf=fcol is not None; ur=add is not None
    if ur: A=df[add].values; Asc=A.std(); An=A/Asc
    def pr(p,X,f,An):
        b=X@p[:k]
        if uf: b=b*(1+p[k]*f)
        if ur: b=b+p[-2]*An+p[-1]
        return b
    names=[f"t{i}" for i in range(k)]+(["a"] if uf else [])+(["lam","gam"] if ur else [])
    bnd={n:(-1000,1000) for n in names}; 
    if uf: bnd["a"]=(-50,50)
    npar=len(names); rng=np.random.default_rng(SEED)
    x0=np.r_[np.ones(k),([0.] if uf else []),([0.,0.] if ur else [])]; bx,bv=x0,np.inf
    mae=lambda p,X,f,An,y: float(np.mean(np.abs(y-pr(p,X,f,An))))
    for s in range(8):
        xi=x0+(rng.normal(0,.5,npar) if s else 0)
        rr=minimize(lambda p:mae(p,X,f,(An if ur else None),y),xi,method="Nelder-Mead",options={"maxiter":30000,"fatol":1e-9})
        if rr.fun<bv:bv,bx=rr.fun,rr.x
    if npar>=2:
        xi={n:float(np.clip(bx[i],*bnd[n])) for i,n in enumerate(names)}
        st=optuna.create_study(direction="minimize",sampler=optuna.samplers.CmaEsSampler(seed=SEED,x0=xi,sigma0=2.,n_startup_trials=0))
        st.enqueue_trial(xi); st.optimize(lambda t:mae(np.array([t.suggest_float(n,*bnd[n]) for n in names]),X,f,(An if ur else None),y),n_trials=nt)
        best=np.array([st.best_params[n] for n in names])
        if bv<mae(best,X,f,(An if ur else None),y):best=bx
    else: best=bx
    pc=np.zeros_like(y)
    for trn,te in GroupKFold(5).split(y,y,g):
        rr=minimize(lambda p:mae(p,X[trn],f[trn],(An[trn] if ur else None),y[trn]),best,method="Nelder-Mead",options={"maxiter":20000,"fatol":1e-8})
        pc[te]=pr(rr.x,X[te],f[te],(An[te] if ur else None))
    r2=1-np.sum((y-pc)**2)/np.sum((y-y.mean())**2)
    if ret_pred: return pc
    return np.mean(np.abs(y-pc)),r2,(best[:-1]/sc if uf and not ur else None),(best[k] if uf else None)
print("### TΔS stats ###")
for t in ["TdS_red","TdS_ox"]:
    y=df[t].values*EH; print(f"  {t}: moy={y.mean():.2f} std={y.std():.2f} | MAE(const)={np.mean(np.abs(y-y.mean())):.2f}")
print("### TΔS modèles ###")
for tgt,fk,rc in [("TdS_red","f2_max","dtsvib_red"),("TdS_ox","f2_min","dtsvib_ox")]:
    mc,r2,th,a=fit(["mu","eta"],fk,tgt); 
    print(f"  {tgt} (μ,η)×(1+f²): MAE={mc:.2f} R2={r2:.3f} θμ={th[0]:+.3f} θη={th[1]:+.3f} a={a:+.3f}")
    cr=np.corrcoef(df[rc],df[tgt]*EH)[0,1]
    mcr,r2r,_,_=fit(["mu","eta"],fk,tgt,add=rc)
    print(f"      +ΔS_RRHO (corr={cr:+.2f}): MAE={mcr:.2f} R2={r2r:.3f}  (Δ={mcr-mc:+.2f})")
# parity TΔS
def _muted(c): h,s,v=mcolors.rgb_to_hsv(np.array(c[:3])); return (*mcolors.hsv_to_rgb((h,s*0.45,min(v,0.78))),1.0)
mols=sorted(df.molecule.unique()); PAL={mo:_muted(c) for mo,c in zip(mols,plt.cm.gist_rainbow(np.linspace(0,1,len(mols))))}
pcols=[PAL[mo] for mo in df.molecule]
def panel(ydft,ypred,color,xl,yl,fname):
    fig,ax=plt.subplots(figsize=FIGSIZE_DOUBLE_V); sc=dict(SCATTER_KW); sc.pop("c",None)
    ax.scatter(ydft,ypred,c=pcols,**sc); regression_with_band(ax,ydft,ypred,color=color)
    allv=np.concatenate([ydft,ypred]); mar=(allv.max()-allv.min())*0.12; lims=[allv.min()-mar,allv.max()+mar]
    ax.plot(lims,lims,"--",color=C_GRAY,lw=0.8,alpha=0.5,zorder=0); ax.set_xlim(lims); ax.set_ylim(lims)
    r2=1-np.sum((ydft-ypred)**2)/np.sum((ydft-ydft.mean())**2); stats_text(ax,r2,np.mean(np.abs(ypred-ydft)),len(ydft))
    ax.set_xlabel(xl); ax.set_ylabel(yl); setup_grid(ax); fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}.png"); fig.savefig(f"{FIG}/{fname}.pdf"); plt.close()
for tgt,col,fn in [("TdS_ox",C_RED_DARK,"parity_tds_ox"),("TdS_red",C_BLUE_DARK,"parity_tds_red")]:
    pc=fit(["mu","eta"],None,tgt,ret_pred=True); panel(df[tgt].values*EH,pc,col,r"$T\Delta S^{\mathrm{DFT}}$ (kJ/mol)",r"$T\Delta S$ prédit (kJ/mol)",fn)
# ΔG combiné
print("### ΔG combiné ###")
EA=fit(["mu","eta","omega_plus"],"f2_max","EA",ret_pred=True); IP=fit(["mu","eta","inv_omega_roy"],"f2_min","IP",ret_pred=True)
Er=fit(["eta","lvc_red_100"],"f2_max","Erelax_red",ret_pred=True); Eo=fit(["eta","lvc_ox_100"],"f2_min","Erelax_ox",ret_pred=True)
dUr=fit(["mu","eta"],"f2_max","ddU_red",ret_pred=True); dUo=fit(["mu","eta"],"f2_min","ddU_ox",ret_pred=True)
Tr=fit(["mu","eta"],None,"TdS_red",ret_pred=True); To=fit(["mu","eta"],None,"TdS_ox",ret_pred=True)
D={k:df[k].values*EH for k in["EA","IP","Erelax_red","Erelax_ox","ddU_red","ddU_ox","TdS_red","TdS_ox"]}
dGr=-EA+Er+dUr-Tr; dGo=IP+Eo+dUo-To
dGrD=-D["EA"]+D["Erelax_red"]+D["ddU_red"]-D["TdS_red"]; dGoD=D["IP"]+D["Erelax_ox"]+D["ddU_ox"]-D["TdS_ox"]
for lab,p,d in [("ΔG_red",dGr,dGrD),("ΔG_ox",dGo,dGoD)]:
    r2=1-np.sum((d-p)**2)/np.sum((d-d.mean())**2); print(f"  {lab}: MAE={np.mean(np.abs(p-d)):.2f} R2={r2:.3f}")
panel(dGoD,dGo,C_RED_DARK,r"$\Delta G^\circ_{\mathrm{ox}}$ DFT (kJ/mol)",r"$\Delta G^\circ_{\mathrm{ox}}$ prédit (kJ/mol)","parity_dg_ox")
panel(dGrD,dGr,C_BLUE_DARK,r"$\Delta G^\circ_{\mathrm{red}}$ DFT (kJ/mol)",r"$\Delta G^\circ_{\mathrm{red}}$ prédit (kJ/mol)","parity_dg_red")
print("DONE")
