#!/usr/bin/env python
"""
ddU symétrique (μ,η)×(1+a·f²) red+ox : coefficients non-scalés + correction
du sur-apprentissage acétonitrile de ddU_ox.

- Fit multi-solvant (n=1336) : coefficients bien conditionnés (référence du modèle final).
- Fit acétonitrile (n=154)   : sweep d'une pénalité L2 sur `a` pour mater le a=-20.
Objectif MAE, ÷std sans intercept, warm-start Nelder-Mead + Optuna CMA-ES,
CV-MAE GroupKFold-5 par molécule.
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
    if slice_name == "acn":
        m = m[m.solvent == "Acetonitrile"]
        d = d[d.solvent == "Acetonitrile"]
    df = m.merge(d[["molecule", "solvent", "conformer", "f2_min"]],
                 on=["molecule", "solvent", "conformer"], how="inner").reset_index(drop=True)
    return df


def predict(p, X, fukui):
    return (X @ p[:2]) * (1.0 + p[2] * fukui)


def obj(p, X, fukui, y, lam):
    return float(np.mean(np.abs(y - predict(p, X, fukui))) + lam * p[2] ** 2)


def fit(df, target, fcol, lam=0.0):
    groups = df.molecule.values
    y = df[target].values * EH
    Xr = df[["mu", "eta"]].values.astype(float)
    scale = Xr.std(axis=0)
    X = Xr / scale
    fukui = df[fcol].values.astype(float)
    names = ["t0", "t1", "a"]

    rng = np.random.default_rng(SEED)
    x0 = np.array([1.0, 1.0, 0.0])
    best_x, best_v = x0, np.inf
    for s in range(8):
        xi = x0 + (rng.normal(0, 0.5, 3) if s > 0 else 0.0)
        r = minimize(lambda p: obj(p, X, fukui, y, lam), xi, method="Nelder-Mead",
                     options={"maxiter": 30000, "fatol": 1e-9})
        if r.fun < best_v:
            best_v, best_x = r.fun, r.x
    x_init = {n: float(best_x[i]) for i, n in enumerate(names)}
    bounds = {"t0": (-1000, 1000), "t1": (-1000, 1000), "a": (-50, 50)}

    def objective(trial):
        p = np.array([trial.suggest_float(n, *bounds[n]) for n in names])
        return obj(p, X, fukui, y, lam)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.CmaEsSampler(seed=SEED, x0=x_init, sigma0=2.0, n_startup_trials=0))
    study.enqueue_trial(x_init)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = np.array([study.best_params[n] for n in names])
    if best_v < obj(best, X, fukui, y, lam):
        best = best_x

    mae = float(np.mean(np.abs(y - predict(best, X, fukui))))
    pred_cv = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(y, y, groups):
        r = minimize(lambda p: obj(p, X[tr], fukui[tr], y[tr], lam), best,
                     method="Nelder-Mead", options={"maxiter": 20000, "fatol": 1e-8})
        pred_cv[te] = predict(r.x, X[te], fukui[te])
    mae_cv = float(np.mean(np.abs(y - pred_cv)))
    theta_un = best[:2] / scale
    return dict(tmu=theta_un[0], teta=theta_un[1], a=best[2], mae=mae, mae_cv=mae_cv)


print("### Multi-solvant (n=1336) — coefficients du modèle final (lam=0) ###")
dfm = load("multi")
for tgt, fcol in [("ddU_red", "f2_max"), ("ddU_ox", "f2_min")]:
    r = fit(dfm, tgt, fcol, lam=0.0)
    print(f"  {tgt}: θμ={r['tmu']:+.3f}  θη={r['teta']:+.3f}  a={r['a']:+.3f}  "
          f"MAE={r['mae']:.3f}  CV-MAE={r['mae_cv']:.3f}")

print("\n### Acétonitrile (n=154) — ddU_ox : sweep pénalité L2 sur `a` ###")
dfa = load("acn")
print(f"  {'lam':>7s} {'θμ':>8s} {'θη':>8s} {'a':>8s} {'MAE':>7s} {'CV-MAE':>7s}")
for lam in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]:
    r = fit(dfa, "ddU_ox", "f2_min", lam=lam)
    print(f"  {lam:7.3f} {r['tmu']:+8.3f} {r['teta']:+8.3f} {r['a']:+8.3f} "
          f"{r['mae']:7.3f} {r['mae_cv']:7.3f}")
print("\n  (réf. acn ddU_ox linéaire sans Fukui : CV-MAE=1.178)")

print("\n### Acétonitrile (n=154) — ddU_red (lam=0, pour la table) ###")
r = fit(dfa, "ddU_red", "f2_max", lam=0.0)
print(f"  ddU_red: θμ={r['tmu']:+.3f}  θη={r['teta']:+.3f}  a={r['a']:+.3f}  "
      f"MAE={r['mae']:.3f}  CV-MAE={r['mae_cv']:.3f}")
