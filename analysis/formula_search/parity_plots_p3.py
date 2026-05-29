#!/usr/bin/env python
"""
Parity plots (partie 3, acétonitrile) — STYLE MAISON (rapport/style.py).
2 panneaux Oxydation (rouge) / Réduction (bleu) : régression + bande 95 %,
boîte stats (R²/MAE/n), police serif. Points coloriés par MOLÉCULE (68 couleurs).
Prédictions out-of-fold (CMA-ES MAE + refit Nelder-Mead par fold), comme le rapport.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import optuna

sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, FIGSIZE_DOUBLE_V, C_RED_DARK, C_BLUE_DARK, C_GRAY)
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
apply_style()
EH = 2625.499638
SEED = 0
EXCL = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
FIGDIR = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"

m = pd.read_parquet("data/merged_dataset.parquet")
d = pd.read_parquet("data/ddu_delta_descriptors.parquet")
m = m[(m.solvent == "Acetonitrile") & (~m.molecule.isin(EXCL))].copy()
d = d[(d.solvent == "Acetonitrile") & (~d.molecule.isin(EXCL))].copy()
df = m.merge(d[["molecule", "solvent", "conformer", "f2_min"]],
             on=["molecule", "solvent", "conformer"], how="inner").reset_index(drop=True)
# LVC (Hessien neutre, seuil 100 cm-1) — descripteur vibronique retenu (remplace LVC2)
_lvc = pd.read_parquet("data/lvc_acetonitrile_thr100.parquet")
df = df.merge(_lvc[["molecule", "solvent", "conformer", "lvc_ox_100", "lvc_red_100"]],
              on=["molecule", "solvent", "conformer"], how="inner").reset_index(drop=True)
df["omega_plus"] = (df.eta + 2 * df.mu)**2 / (8 * df.eta)   # pouvoir électro-accepteur de Gázquez (EA)
df["inv_omega_roy"] = 2 * df.eta / (df.eta / 2 - df.mu)**2  # 1/ω⁻ Gázquez (IP)
groups = df.molecule.values

import matplotlib.colors as mcolors

def _muted(rgba):
    """Rendu terne/professionnel : saturation réduite, luminosité plafonnée (HSV)."""
    h, s, v = mcolors.rgb_to_hsv(np.array(rgba[:3]))
    s *= 0.45
    v = min(v, 0.78)
    r, g, b = mcolors.hsv_to_rgb((h, s, v))
    return (r, g, b, 1.0)

mols = sorted(df.molecule.unique())
palette = [_muted(c) for c in plt.cm.gist_rainbow(np.linspace(0, 1, len(mols)))]
MCOL = {mol: palette[i] for i, mol in enumerate(mols)}
pt_colors = [MCOL[mol] for mol in df.molecule]


def predict(p, X, f):
    k = X.shape[1]
    return (X @ p[:k]) * (1.0 + p[k] * f)


def mae(p, X, f, y):
    return float(np.mean(np.abs(y - predict(p, X, f))))


def oof_predict(target, cols, fcol, n_trials=6000):
    y = df[target].values * EH
    Xr = df[cols].values.astype(float)
    scale = Xr.std(axis=0)
    X = Xr / scale
    f = df[fcol].values.astype(float)
    k = X.shape[1]
    names = [f"t{i}" for i in range(k)] + ["a"]
    bnd = {n: (-1000, 1000) for n in names[:-1]}
    bnd["a"] = (-50, 50)
    rng = np.random.default_rng(SEED)
    x0 = np.concatenate([np.ones(k), [0.0]])
    bx, bv = x0, np.inf
    for s in range(8):
        xi = x0 + (rng.normal(0, 0.5, k + 1) if s else 0.0)
        r = minimize(lambda p: mae(p, X, f, y), xi, method="Nelder-Mead",
                     options={"maxiter": 30000, "fatol": 1e-8})
        if r.fun < bv:
            bv, bx = r.fun, r.x
    xi = {n: float(bx[i]) for i, n in enumerate(names)}
    smp = optuna.samplers.CmaEsSampler(seed=SEED, x0=xi, sigma0=2.0, n_startup_trials=0)
    st = optuna.create_study(direction="minimize", sampler=smp)
    st.enqueue_trial(xi)
    st.optimize(lambda t: mae(np.array([t.suggest_float(n, *bnd[n]) for n in names]), X, f, y),
                n_trials=n_trials)
    best = np.array([st.best_params[n] for n in names])
    if bv < mae(best, X, f, y):
        best = bx
    pred_cv = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(y, y, groups):
        r = minimize(lambda p: mae(p, X[tr], f[tr], y[tr]), best,
                     method="Nelder-Mead", options={"maxiter": 20000, "fatol": 1e-7})
        pred_cv[te] = predict(r.x, X[te], f[te])
    return y, pred_cv, float(np.mean(np.abs(y - pred_cv)))


# Limites d'axes partagées (mêmes échelles Koopmans / Politzer / CDFT)
IP_LIMS = (505, 715)
EA_LIMS = (40, 432)

def make_panel(target, cols, fcol, color, label, fname, lims=None):
    """Un panneau parity, patron identique aux figures existantes (plot_fmaxO_parity)
    sauf : couleur des points par molécule + labels d'axes adaptés."""
    ydft, ypred, mae_cv = oof_predict(target, cols, fcol)
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    scat = dict(SCATTER_KW); scat.pop("c", None)
    ax.scatter(ydft, ypred, c=pt_colors, **scat)          # <- couleur par molécule
    regression_with_band(ax, ydft, ypred, color=color)
    if lims is None:
        all_vals = np.concatenate([ydft, ypred])
        margin = (all_vals.max() - all_vals.min()) * 0.12
        lims = [all_vals.min() - margin, all_vals.max() + margin]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2 = 1 - np.sum((ydft - ypred)**2) / np.sum((ydft - ydft.mean())**2)
    stats_text(ax, r2, np.mean(np.abs(ypred - ydft)), len(ydft))
    ax.set_xlabel(f"${label}^{{\\mathrm{{DFT}}}}$ (kJ/mol)")
    ax.set_ylabel(f"${label}$ prédit (kJ/mol)")
    setup_grid(ax)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/{fname}.png")
    plt.savefig(f"{FIGDIR}/{fname}.pdf")
    plt.close()
    print(f"  {fname}: MAE={mae_cv:.2f}  R2={r2:.3f}")


if __name__ == "__main__":
    make_panel("IP", ["mu", "eta", "inv_omega_roy"], "f2_min", C_RED_DARK,
               "\\mathrm{IP}", "parity_ip", lims=IP_LIMS)
    make_panel("EA", ["mu", "eta", "omega_plus"], "f2_max", C_BLUE_DARK,
               "\\mathrm{EA}", "parity_ea", lims=EA_LIMS)
    make_panel("Erelax_ox", ["eta", "lvc_ox_100"], "f2_min", C_RED_DARK,
               "E_{\\mathrm{relax}}", "parity_erelax_ox")
    make_panel("Erelax_red", ["eta", "lvc_red_100"], "f2_max", C_BLUE_DARK,
               "E_{\\mathrm{relax}}", "parity_erelax_red")
    # Version sans le terme vibronique LVC (descripteur frontière seul)
    make_panel("Erelax_ox", ["eta"], "f2_min", C_RED_DARK,
               "E_{\\mathrm{relax}}", "parity_erelax_ox_nolvc")
    make_panel("Erelax_red", ["eta"], "f2_max", C_BLUE_DARK,
               "E_{\\mathrm{relax}}", "parity_erelax_red_nolvc")
