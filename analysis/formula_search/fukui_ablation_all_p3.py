#!/usr/bin/env python
"""
Ablation de la correction de Fukui dual sur les 4 paires de termes thermodynamiques.

Pour chaque cible, on compare le modèle SANS Fukui (base linéaire) au modèle
AVEC Fukui dual  base × (1 + a·f²) :
  red-type -> f²_max   |   ox-type -> f²_min

Formes de base = celles RETENUES dans le rapport (partie 3) :
  EA          : (μ, η, ω⁺_Parr)
  IP          : (μ, η, 1/ω⁻_Roy)
  Erelax_red  : (η, LVC_red)        LVC = Hessien neutre, seuil 100 cm⁻¹
  Erelax_ox   : (η, LVC_ox)
  ddU_red/ox  : (μ, η)
  TdS_red/ox  : (μ, η)

Méthodo identique à optuna_acetonitrile_p3.py (objectif MAE, ÷std sans intercept,
warm-start Nelder-Mead + Optuna CMA-ES, CV-MAE GroupKFold-5 par molécule).
Évalué sur ACÉTONITRILE (154) et MULTI-SOLVANT (1336).
NB EA/IP multi-solvant : ici à coefficients CONSTANTS (le modèle final du rapport
utilise une dépendance Padé[1/1]-Born en ε ; le gain Fukui en % reste indicatif).
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

SPECS = [
    ("EA",         ["mu", "eta", "omega_plus"],    "f2_max", "EA/IP"),
    ("IP",         ["mu", "eta", "inv_omega_roy"], "f2_min", "EA/IP"),
    ("Erelax_red", ["eta", "lvc_red_100"],         "f2_max", "E_relax"),
    ("Erelax_ox",  ["eta", "lvc_ox_100"],          "f2_min", "E_relax"),
    ("ddU_red",    ["mu", "eta"],                  "f2_max", "ddU"),
    ("ddU_ox",     ["mu", "eta"],                  "f2_min", "ddU"),
    ("TdS_red",    ["mu", "eta"],                  "f2_max", "TdS"),
    ("TdS_ox",     ["mu", "eta"],                  "f2_min", "TdS"),
]


def load(slice_name):
    m = pd.read_parquet("data/merged_dataset.parquet")
    d = pd.read_parquet("data/ddu_delta_descriptors.parquet")
    lvc = pd.read_parquet("data/lvc_multisolvent_thr100.parquet")
    m = m[~m.molecule.isin(EXCL)].copy()
    d = d[~d.molecule.isin(EXCL)].copy()
    if slice_name == "Acetonitrile":
        m = m[m.solvent == "Acetonitrile"]
        d = d[d.solvent == "Acetonitrile"]
    df = m.merge(d[["molecule", "solvent", "conformer", "f2_min"]],
                 on=["molecule", "solvent", "conformer"], how="inner")
    df = df.merge(lvc[["molecule", "solvent", "conformer", "lvc_ox_100", "lvc_red_100"]],
                  on=["molecule", "solvent", "conformer"], how="inner").reset_index(drop=True)
    df["omega_plus"] = (df.eta + 2 * df.mu) ** 2 / (8 * df.eta)
    df["inv_omega_roy"] = 2 * df.eta / (df.eta / 2 - df.mu) ** 2
    return df


def predict(p, X, fukui):
    k = X.shape[1]
    if fukui is None:
        return X @ p[:k]
    return (X @ p[:k]) * (1.0 + p[k] * fukui)


def mae_of(p, X, fukui, y):
    return float(np.mean(np.abs(y - predict(p, X, fukui))))


def fit(df, target, cols, fcol):
    groups = df.molecule.values
    y = df[target].values * EH
    Xr = df[cols].values.astype(float)
    scale = Xr.std(axis=0)
    scale[scale == 0] = 1.0
    X = Xr / scale
    fukui = None if fcol is None else df[fcol].values.astype(float)
    k = X.shape[1]
    npar = k + (0 if fukui is None else 1)
    names = [f"t{i}" for i in range(k)] + ([] if fukui is None else ["a"])

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

    pred = predict(best, X, fukui)
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    pred_cv = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(y, y, groups):
        fk_tr = None if fukui is None else fukui[tr]
        r = minimize(lambda p: mae_of(p, X[tr], fk_tr, y[tr]), best,
                     method="Nelder-Mead", options={"maxiter": 20000, "fatol": 1e-7})
        fk_te = None if fukui is None else fukui[te]
        pred_cv[te] = predict(r.x, X[te], fk_te)
    mae_cv = float(np.mean(np.abs(y - pred_cv)))
    a = None if fukui is None else float(best[-1])
    return dict(mae_cv=mae_cv, r2=r2, a=a)


for slice_name in ["Acetonitrile", "Multi-solvant"]:
    df = load(slice_name)
    print("=" * 84)
    print(f"{slice_name} — {len(df)} conformères, {df.molecule.nunique()} molécules")
    print("=" * 84)
    print(f"  {'cible':12s} {'base':22s} {'sansF':>7s} {'avecF':>7s} {'gain':>8s} {'%':>7s}  a")
    for tgt, cols, fcol, _grp in SPECS:
        no = fit(df, tgt, cols, None)
        ye = fit(df, tgt, cols, fcol)
        gain = no["mae_cv"] - ye["mae_cv"]
        pct = 100.0 * gain / no["mae_cv"]
        print(f"  {tgt:12s} ({','.join(cols):20s}) "
              f"{no['mae_cv']:7.3f} {ye['mae_cv']:7.3f} {gain:+8.3f} {pct:+6.1f}%  a={ye['a']:+.2f}")
    print()
