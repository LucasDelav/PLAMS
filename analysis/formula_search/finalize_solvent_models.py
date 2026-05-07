#!/usr/bin/env python3
"""Finalise the (mu+eta+omega) × g(eps) models for IP and EA.

Two models, both joint-MAE fitted on the 1336 conformers (protocol B):
  - born     : beta_X(eps) = a_X + b_X * (eps-1)/(2eps+1)
  - pade11_g : beta_X(eps) = (a0_X + a1_X * g_Born) / (1 + b1_X * g_Born)

Saves coefficients to JSON and generates two parity plots
(EA / IP subfigures), one per correction.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
OUT_DIR = REPO / "analysis" / "formula_search" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
DESCRIPTORS = ("mu", "eta", "omega")
TARGETS = ("IP", "EA")
SEED = 42
N_TRIALS_1D = 5000
N_TRIALS_JOINT = 12000
STAB_TOL = 0.05
optuna.logging.set_verbosity(optuna.logging.WARNING)


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def beta_born(eps, p):
    """beta(eps) = a + b * g_Born(eps).  p = [a, b]."""
    return p[0] + p[1] * g_born(eps)


def beta_pade11_g(eps, p):
    """beta(eps) = (a0 + a1 * g) / (1 + b1 * g),  g = g_Born.  p = [a0, a1, b1]."""
    g = g_born(eps)
    den = 1.0 + p[2] * g
    if np.any(np.abs(den) < STAB_TOL):
        return None
    return (p[0] + p[1] * g) / den


FAMILY = {
    "born":     {"n": 2, "names": ["a", "b"],
                 "eval": beta_born,
                 "x0_const": lambda mean: {"a": float(mean), "b": 0.0}},
    "pade11_g": {"n": 3, "names": ["a0", "a1", "b1"],
                 "eval": beta_pade11_g,
                 "x0_const": lambda mean: {"a0": float(mean), "a1": 0.0, "b1": 0.0}},
}


def per_solvent_ols(df, target):
    rows = []
    for solv, sub in df.groupby("solvent"):
        X = sub[list(DESCRIPTORS)].values
        y = sub[target].values * EH_TO_KJ
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append({"solvent": solv, "epsilon": EPSILON_R[solv],
                     "beta_mu": beta[0], "beta_eta": beta[1], "beta_omega": beta[2],
                     "n": len(sub)})
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


def fit_1d(eps, betas, weights, family, scale, n_trials=N_TRIALS_1D):
    fam = FAMILY[family]
    sc = max(scale, 50.0)
    bounds = {"a": (-3*sc, 3*sc), "b": (-3*sc, 3*sc),
              "a0": (-3*sc, 3*sc), "a1": (-5*sc, 5*sc), "b1": (-5.0, 5.0)}
    mean_b = float(np.average(betas, weights=weights))

    def objective(trial):
        p = np.array([trial.suggest_float(name, *bounds[name]) for name in fam["names"]])
        pred = fam["eval"](eps, p)
        if pred is None or np.any(~np.isfinite(pred)):
            return 1e9
        return float(np.sum(weights * (betas - pred) ** 2) / weights.sum())

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=fam["x0_const"](mean_b),
                                            sigma0=sc, n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(fam["x0_const"](mean_b))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return np.array([study.best_params[name] for name in fam["names"]])


def fit_joint(df, target, family, n_trials=N_TRIALS_JOINT):
    fam = FAMILY[family]
    eps = df["epsilon"].values
    descs = [df[d].values for d in DESCRIPTORS]
    y = df[target].values * EH_TO_KJ

    # Warm-start: per-coefficient 1D fit
    persol = per_solvent_ols(df, target)
    epsv = persol["epsilon"].values
    weights = persol["n"].values.astype(float)
    p_init = []
    scales = []
    for d in DESCRIPTORS:
        b = persol[f"beta_{d}"].values
        sc = max(np.abs(b).max(), 1.0)
        scales.append(sc)
        p_init.append(fit_1d(epsv, b, weights, family, sc))

    sc_max = max(scales)
    bounds = {"a": (-3*sc_max, 3*sc_max), "b": (-3*sc_max, 3*sc_max),
              "a0": (-3*sc_max, 3*sc_max), "a1": (-5*sc_max, 5*sc_max), "b1": (-5.0, 5.0)}

    full_names = [f"d{i}_{name}" for i in range(3) for name in fam["names"]]
    full_bounds = [bounds[name] for _ in range(3) for name in fam["names"]]
    x0_dict = {f"d{i}_{name}": float(p_init[i][j])
               for i in range(3) for j, name in enumerate(fam["names"])}

    def predict(flat):
        pred = np.zeros(len(df))
        for i in range(3):
            p = flat[i * fam["n"]:(i + 1) * fam["n"]]
            beta = fam["eval"](eps, p)
            if beta is None or np.any(~np.isfinite(beta)):
                return None
            pred = pred + beta * descs[i]
        return pred

    def objective(trial):
        flat = np.array([trial.suggest_float(n, lo, hi)
                         for n, (lo, hi) in zip(full_names, full_bounds)])
        pred = predict(flat)
        if pred is None:
            return 1e9
        return float(np.mean(np.abs(y - pred)))

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0_dict, sigma0=10.0,
                                            n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(x0_dict)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    flat = np.array([study.best_params[n] for n in full_names])

    # Decompose into per-descriptor params
    params = {DESCRIPTORS[i]: dict(zip(fam["names"], flat[i*fam["n"]:(i+1)*fam["n"]].tolist()))
              for i in range(3)}
    pred = predict(flat)
    return params, pred, float(np.mean(np.abs(y - pred)))


def parity_plot(df, target_pred: dict[str, np.ndarray], family_label: str, out_path: Path):
    """One figure with EA + IP subplots, points coloured by solvent."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    solvents = sorted(df.solvent.unique(), key=lambda s: EPSILON_R[s])
    cmap = plt.get_cmap("viridis")
    colours = {s: cmap(i / max(len(solvents) - 1, 1)) for i, s in enumerate(solvents)}

    for ax, target in zip(axes, ("EA", "IP")):
        y = df[target].values * EH_TO_KJ
        pred = target_pred[target]
        for s in solvents:
            mask = (df.solvent == s).values
            ax.scatter(y[mask], pred[mask], s=18, alpha=0.7,
                       color=colours[s], label=f"{s} (ε={EPSILON_R[s]:g})")
        lo = min(y.min(), pred.min())
        hi = max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        mae = float(np.mean(np.abs(y - pred)))
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        ax.set_xlabel(f"{target} référence (kJ/mol)")
        ax.set_ylabel(f"{target} prédit (kJ/mol)")
        ax.set_title(f"{target}  —  MAE = {mae:.2f} kJ/mol,  RMSE = {rmse:.2f} kJ/mol")
        ax.grid(alpha=0.3)
        if target == "EA":
            ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)

    fig.suptitle(f"Parité — modèle (μ + η + ω) × {family_label}(ε)  "
                 f"[n={len(df)}, 10 solvants, 68 molécules]", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)

    final_models = {}
    for family in ("born", "pade11_g"):
        print(f"\n=== Fitting family: {family} ===", flush=True)
        family_block = {"family": family,
                        "form": ("beta_X(eps) = a_X + b_X * (eps-1)/(2eps+1)" if family == "born"
                                 else "beta_X(eps) = (a0_X + a1_X * g) / (1 + b1_X * g),  g = (eps-1)/(2eps+1)"),
                        "intercept": 0.0,
                        "descriptors": list(DESCRIPTORS),
                        "fit_protocol": "joint global-MAE on 1336 conformers (CMA-ES, warm-started from per-coef 1D MSE fit)",
                        "n_conformers": int(len(df)),
                        "n_molecules": int(df.molecule.nunique()),
                        "solvents": sorted(df.solvent.unique()),
                        "filters": "exclude Br-DA, Br-HMC, I-DA, I-HMC, L-DOPA; EA<0 dropped at parquet build",
                        "targets": {}}
        target_pred = {}
        for tgt in TARGETS:
            params, pred, mae = fit_joint(df, tgt, family)
            target_pred[tgt] = pred
            family_block["targets"][tgt] = {
                "params_per_descriptor": params,
                "MAE_kJmol": mae,
                "RMSE_kJmol": float(np.sqrt(np.mean(((df[tgt].values * EH_TO_KJ) - pred) ** 2))),
            }
            print(f"  {tgt}: MAE = {mae:.3f} kJ/mol", flush=True)
            for d, p in params.items():
                print(f"    beta_{d}(eps): {p}", flush=True)

        final_models[family] = family_block

        out_png = OUT_DIR / f"parity_{family}.png"
        parity_plot(df, target_pred, "Born" if family == "born" else "Padé[1/1] en g_Born", out_png)
        print(f"  Saved parity plot -> {out_png.relative_to(REPO)}")

    out_json = OUT_DIR / "final_solvent_models.json"
    with open(out_json, "w") as f:
        json.dump(final_models, f, indent=2, ensure_ascii=False)
    print(f"\nSaved coefficients -> {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
