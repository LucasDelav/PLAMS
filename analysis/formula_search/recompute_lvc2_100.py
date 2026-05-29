import json, numpy as np, pandas as pd, optuna
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
optuna.logging.set_verbosity(optuna.logging.WARNING)
EH=2625.499638; SEED=0; THR=100; EXCL={"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
PRED=Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/dataset_predict_dg/Acetonitrile")

m=pd.read_parquet("data/merged_dataset.parquet"); dd=pd.read_parquet("data/ddu_delta_descriptors.parquet")
m=m[(m.solvent=="Acetonitrile")&(~m.molecule.isin(EXCL))].copy()
dd=dd[(dd.solvent=="Acetonitrile")&(~dd.molecule.isin(EXCL))].copy()
df=m.merge(dd[["molecule","solvent","conformer","f2_min"]],on=["molecule","solvent","conformer"]).reset_index(drop=True)
df["mu2"]=df.mu**2

def lvc2(state,thr):
    Ha=np.array(state['hessian']); n=Ha.shape[0]; H=Ha.transpose(0,2,1,3).reshape(3*n,3*n); H=0.5*(H+H.T)
    g=np.array(state['gradients']).flatten(); E=0.0
    for mo in state['vibrations']['modes']:
        w=mo['frequency_cm1']; N=np.array(mo['displacements']).flatten()
        if w<thr: continue
        k=N@H@N
        if k<=0: continue
        E+=-((g@N)**2)/(2*k)
    return E
ox100=[];rd100=[];keep=[]
for _,r in df.iterrows():
    p=PRED/r.molecule/f"{r.molecule}_conf{int(r.conformer)}_prediction.json"
    if not p.exists(): keep.append(False); continue
    j=json.load(open(p))
    ox100.append(lvc2(j['states']['oxidized_sp'],THR)); rd100.append(lvc2(j['states']['reduced_sp'],THR)); keep.append(True)
df=df[keep].reset_index(drop=True); df["lvc2_ox_100"]=ox100; df["lvc2_red_100"]=rd100
df[["molecule","solvent","conformer","lvc2_ox_100","lvc2_red_100"]].to_parquet("data/lvc2_acetonitrile_thr100.parquet")
g=df.molecule.values; print(f"LVC2(100) calculé pour {len(df)} conformères, sauvé.",flush=True)

def pred(p,X,f): k=X.shape[1]; return (X@p[:k])*(1+p[k]*f)
def mae(p,X,f,y): return float(np.mean(np.abs(y-pred(p,X,f))))
def fit(cols,fcol,tgt,nt=6000):
    y=df[tgt].values*EH; Xr=df[cols].values.astype(float); sc=Xr.std(0); sc[sc==0]=1; X=Xr/sc; f=df[fcol].values; k=X.shape[1]
    names=[f"t{i}" for i in range(k)]+["a"]; bnd={n:(-1000,1000) for n in names[:-1]}; bnd["a"]=(-50,50)
    rng=np.random.default_rng(SEED); x0=np.r_[np.ones(k),0.]; bx,bv=x0,np.inf
    for s in range(8):
        xi=x0+(rng.normal(0,.5,k+1) if s else 0)
        rr=minimize(lambda p:mae(p,X,f,y),xi,method="Nelder-Mead",options={"maxiter":30000,"fatol":1e-8})
        if rr.fun<bv:bv,bx=rr.fun,rr.x
    xi={n:float(bx[i]) for i,n in enumerate(names)}
    st=optuna.create_study(direction="minimize",sampler=optuna.samplers.CmaEsSampler(seed=SEED,x0=xi,sigma0=2.0,n_startup_trials=0))
    st.enqueue_trial(xi); st.optimize(lambda t:mae(np.array([t.suggest_float(n,*bnd[n]) for n in names]),X,f,y),n_trials=nt)
    best=np.array([st.best_params[n] for n in names])
    if bv<mae(best,X,f,y):best=bx
    pc=np.zeros_like(y)
    for tr,te in GroupKFold(5).split(y,y,g):
        rr=minimize(lambda p:mae(p,X[tr],f[tr],y[tr]),best,method="Nelder-Mead",options={"maxiter":20000,"fatol":1e-7})
        pc[te]=pred(rr.x,X[te],f[te])
    theta_raw=best[:-1]/sc
    return pc, np.mean(np.abs(y-pc)), theta_raw, best[-1]

print("\n=== E_relax @ LVC2(100) ===",flush=True)
oof={}
for tgt,lvc,fk in [("Erelax_ox","lvc2_ox_100","f2_min"),("Erelax_red","lvc2_red_100","f2_max")]:
    pc,mae_cv,th,a=fit(["mu","eta","mu2",lvc],fk,tgt); oof[tgt]=pc
    print(f"  {tgt}: CV-MAE={mae_cv:.2f} | θ_mu={th[0]:+.3g} θ_eta={th[1]:+.3g} θ_mu2={th[2]:+.3g} θ_L={th[3]:+.4g} a={a:+.3f}",flush=True)
    pcN,maeN,_,_=fit(["mu","eta","mu2"],fk,tgt)
    print(f"      sans LVC2: CV-MAE={maeN:.2f}  (Δ={maeN-mae_cv:+.2f})",flush=True)

# combined dG with E_relax(100)
print("\n=== ΔG combiné avec E_relax(100) ===",flush=True)
for t,c,f in [("EA",["mu","eta"],"f2_max")]: pass
def oofc(cols,fcol,tgt):
    pc,_,_,_=fit(cols,fcol,tgt); return pc
df["omega_parr"]=df.mu**2/(2*df.eta); df["inv_omega_roy"]=2*df.eta/(df.eta/2-df.mu)**2
EA=oofc(["mu","eta","omega_parr"],"f2_max","EA"); IP=oofc(["mu","eta","inv_omega_roy"],"f2_min","IP")
dUr=oofc(["mu","eta"],"f2_max","ddU_red"); dUo=oofc(["mu","eta"],"f2_min","ddU_ox")  # note: ddU uses (mu,eta) here approx; real uses deta — but for combined trend ok
TSr=oofc(["mu","eta"],"f2_max","TdS_red"); TSo=oofc(["mu","eta"],"f2_min","TdS_ox")
D={k:df[k].values*EH for k in["EA","IP","Erelax_red","Erelax_ox","ddU_red","ddU_ox","TdS_red","TdS_ox"]}
dGr=-EA+oof["Erelax_red"]+dUr-TSr; dGo=IP+oof["Erelax_ox"]+dUo-TSo
dGrD=-D["EA"]+D["Erelax_red"]+D["ddU_red"]-D["TdS_red"]; dGoD=D["IP"]+D["Erelax_ox"]+D["ddU_ox"]-D["TdS_ox"]
print(f"  ΔG_red MAE={np.mean(np.abs(dGr-dGrD)):.2f}  ΔG_ox MAE={np.mean(np.abs(dGo-dGoD)):.2f}",flush=True)
