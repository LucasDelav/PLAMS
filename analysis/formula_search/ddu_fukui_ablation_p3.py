#!/usr/bin/env python
"""
Ablation Fukui dual sur ddU — modèle du rapport (μ,η)×(1+a·f²), objectif MAE Optuna.

Compare, pour ddU_red et ddU_ox, le modèle SANS correction de Fukui (μ,η linéaire)
au modèle AVEC Fukui dual (μ,η)×(1+a·f²) :
  ddU_red : f²_max   |   ddU_ox : f²_min

Méthodo identique à optuna_acetonitrile_p3.py :
  - descripteurs ÷std sans centrage (pas d'intercept)
  - warm-start Nelder-Mead (8 départs) + Optuna CMA-ES, on garde le meilleur des deux
  - CV-MAE en GroupKFold-5 par molécule (refit Nelder-Mead par pli)
Évalué sur ACÉTONITRILE (n=154, 68 mol) puis MULTI-SOLVANT (n=1336, 68 mol).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
EH = 2625.499638
SEED = 0
N_TRIALS = 4000
EXCL = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}


def load(slice_name):
    m = pd.read_parquet("data/merged_dataset.parquet")
    d = pd.read_parquet("data/ddu_delta_descriptors.parquet")
    m = m[~m.molecule.isin(EXCL)].copy()
    d = d[~d.molecule.isin(EXCL)].copy()
    if slice_name == "Acetonitrile":
        m = m[m.solvent == "Acetonitrile"].copy()
        d = d[d.solvent == "Acetonitrile"].copy()
    df = m.merge(d[["molecule", "solvent", "conformer", "f2_min"]],
                 on=["molecule", "solvent", "conformer"], how="inner").reset_index(drop=True)
    return df


def predict(p, X, fukui):
    k = X.shape[1]
    if fukui is None:
        return X @ p[:k]
    return (X @ p[:k]) * (1.0 + p[k] * fukui)


def mae_of(p, X, fukui, y):
    return float(np.mean(np.abs(y - predict(p, X, fukui))))


def fit(df, target, cols, fcol):
    """fcol=None -> modèle linéaire (μ,η) ; sinon (μ,η)×(1+a·fcol)."""
    groups = df.molecule.values
    y = df[target].values * EH
    Xr = df[cols].values.astype(float)
    scale = Xr.std(axis=0)
    X = Xr / scale
    fukui = None if fcol is None else df[fcol].values.astype(float)
    k = X.shape[1]
    npar = k + (0 if fukui is None else 1)
    names = [f"t{i}" for i in range(k)] + ([] if fukui is None else ["a"])

    # warm-start Nelder-Mead (8 départs)
    rng = np.random.default_rng(SEED)
    x0 = np.concatenate([np.ones(k), ([] if fukui is None else [0.0])])
    best_x, best_v = x0, np.inf
    for s in range(8):
        xi = x0 + (rng.normal(0, 0.5, npar) if s > 0 else 0.0)
        r = minimize(lambda p: mae_of(p, X, fukui, y), xi, method="Nelder-Mead",
                     options={"maxiter": 30000, "fatol": 1e-8})
        if r.fun < best_v:
            best_v, best_x = r.fun, r.x

    x_init = {n: float(best_x[i]) for i, n in enumerate(names)}
    bounds = {n: (-1000.0, 1000.0) for n in names}
    if "a" in bounds:
        bounds["a"] = (-50.0, 50.0)

    def objective(trial):
        p = np.array([trial.suggest_float(n, *bounds[n]) for n in names])
        return mae_of(p, X, fukui, y)

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x_init, sigma0=2.0,
                                           n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(x_init)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = np.array([study.best_params[n] for n in names])
    if best_v < mae_of(best, X, fukui, y):
        best = best_x
    mae = mae_of(best, X, fukui, y)
    pred = predict(best, X, fukui)
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    # CV-MAE GroupKFold-5 par molécule
    pred_cv = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(y, y, groups):
        fk_tr = None if fukui is None else fukui[tr]
        r = minimize(lambda p: mae_of(p, X[tr], fk_tr, y[tr]), best,
                     method="Nelder-Mead", options={"maxiter": 20000, "fatol": 1e-7})
        fk_te = None if fukui is None else fukui[te]
        pred_cv[te] = predict(r.x, X[te], fk_te)
    mae_cv = float(np.mean(np.abs(y - pred_cv)))

    theta_unscaled = best[:k] / scale
    a = None if fukui is None else float(best[-1])
    return dict(mae=mae, mae_cv=mae_cv, r2=r2, theta=theta_unscaled.tolist(), a=a)


for slice_name in ["Acetonitrile", "Multi-solvant (10)"]:
    df = load("Acetonitrile" if slice_name == "Acetonitrile" else "ALL")
    print("=" * 76)
    print(f"{slice_name} — {len(df)} conformères, {df.molecule.nunique()} molécules")
    print("=" * 76)
    for tgt, fcol in [("ddU_red", "f2_max"), ("ddU_ox", "f2_min")]:
        no = fit(df, tgt, ["mu", "eta"], None)
        ye = fit(df, tgt, ["mu", "eta"], fcol)
        gain = no["mae_cv"] - ye["mae_cv"]
        pct = 100.0 * gain / no["mae_cv"]
        print(f"\n  {tgt}   (Fukui dual = {fcol})")
        print(f"    sans Fukui  (μ,η)          : CV-MAE={no['mae_cv']:.3f}  "
              f"MAE={no['mae']:.3f}  R²={no['r2']:.3f}  "
              f"[θμ={no['theta'][0]:+.3f} θη={no['theta'][1]:+.3f}]")
        print(f"    avec Fukui  (μ,η)×(1+a·f²) : CV-MAE={ye['mae_cv']:.3f}  "
              f"MAE={ye['mae']:.3f}  R²={ye['r2']:.3f}  "
              f"[θμ={ye['theta'][0]:+.3f} θη={ye['theta'][1]:+.3f} a={ye['a']:+.3f}]")
        print(f"    --> gain CV-MAE = {gain:+.3f} kJ/mol ({pct:+.1f}%)")
    print()
