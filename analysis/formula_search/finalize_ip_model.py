#!/usr/bin/env python3
"""Final IP model: replace omega by 1/omega_minus (Roy nucleophilicity, no factor)
and use f²_min over ALL atoms (latest convention) as the Fukui correction.

Model:
    IP(eps) = [β_μ(ε)·μ + β_η(ε)·η + β_n(ε)·n] × (1 + a_corr · f²_min_all)
    n = 1/ω⁻ = 2η / (η/2 − μ)²       [eV⁻¹, Roy/Gázquez nucleophilicity]
    β_X(ε) = (a0 + a1·g)/(1 + b1·g),  g = (ε−1)/(2ε+1)
    intercept = 0

Joint refit of 10 parameters via Optuna CMA-ES, multi-start on a_corr,
warm-started from the previous (μ+η+ω, f²_min_noO) coefficients.

Saves coefficients to JSON and parity plot.
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
PRED_DIR = REPO / "dataset_predict_dg"
COEF_PREV = REPO / "analysis" / "formula_search" / "results" / "fukui_correction_coeffs.csv"
OUT_DIR = REPO / "analysis" / "formula_search" / "results"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
DESCRIPTORS = ("mu", "eta", "n")  # n = 1/ω⁻
SEED = 42
N_TRIALS = 12000
STAB_TOL = 0.05
optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAM_NAMES = [f"{p}_{d}" for d in DESCRIPTORS for p in ("a0", "a1", "b1")] + ["a_corr"]


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    return [lines[2 + i].split()[0] for i in range(int(lines[0]))]


def load_f2_min_all(df):
    """Min of dual Fukui over ALL atoms (no exclusion)."""
    out = []
    for _, row in df.iterrows():
        path = (PRED_DIR / row.solvent / row.molecule
                / f"{row.molecule}_conf{int(row.conformer)}_prediction.json")
        if not path.exists():
            out.append(np.nan); continue
        try:
            with open(path) as f:
                pred = json.load(f)
            dual = pred["states"]["neutral_sp"]["cdft"]["dual"]
            out.append(float(min(dual)))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def predict(eps, mu, eta, n_desc, fukui, params):
    g = g_born(eps)
    out = np.zeros_like(eps)
    for d, x in zip(DESCRIPTORS, (mu, eta, n_desc)):
        a0, a1, b1 = params[f"a0_{d}"], params[f"a1_{d}"], params[f"b1_{d}"]
        den = 1.0 + b1 * g
        if np.any(np.abs(den) < STAB_TOL):
            return None
        out += (a0 + a1 * g) / den * x
    return out * (1.0 + params["a_corr"] * fukui)


def warmstart():
    """Use previous (μ+η+ω) IP fit as starting point — ω-coefs map to n-coefs."""
    df = pd.read_csv(COEF_PREV)
    df = df[(df.target == "IP") & (df.fukui == "f2_min_noO")].set_index("param")
    x0 = {}
    for p in ("a0", "a1", "b1"):
        x0[f"{p}_mu"] = float(df.loc[f"{p}_mu", "value"])
        x0[f"{p}_eta"] = float(df.loc[f"{p}_eta", "value"])
        x0[f"{p}_n"] = float(df.loc[f"{p}_omega", "value"])
    x0["a_corr"] = float(df.loc["a_corr", "value"])
    return x0


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    I = df["eta"] / 2.0 - df["mu"]
    df["n"] = 2.0 * df["eta"] / (I ** 2)   # 1/ω⁻ = 2η/I²
    print(f"Dataset: 10 solvents, n={len(df)} conformers, {df.molecule.nunique()} molecules", flush=True)
    print(f"\n  n = 1/ω⁻ range: [{df.n.min():.3f}, {df.n.max():.3f}] eV⁻¹")

    print(f"\nLoading f²_min over ALL atoms from predict-dg JSONs...", flush=True)
    df["f2_min_all"] = load_f2_min_all(df)
    valid = np.isfinite(df.f2_min_all.values)
    print(f"  Complete on {valid.sum()}/{len(df)} conformers", flush=True)

    eps = df["epsilon"].values
    mu = df["mu"].values
    eta = df["eta"].values
    n_desc = df["n"].values
    fukui = df["f2_min_all"].values
    y = df["IP"].values * EH_TO_KJ

    eps_v, mu_v, eta_v, n_v, f_v, y_v = (
        eps[valid], mu[valid], eta[valid], n_desc[valid], fukui[valid], y[valid]
    )

    x0 = warmstart()

    scale_num = max(max(abs(v) for k, v in x0.items() if k.startswith(("a0_", "a1_"))), 50.0)
    bounds = {}
    for d in DESCRIPTORS:
        bounds[f"a0_{d}"] = (-3 * scale_num, 3 * scale_num)
        bounds[f"a1_{d}"] = (-5 * scale_num, 5 * scale_num)
        bounds[f"b1_{d}"] = (-5.0, 5.0)
    bounds["a_corr"] = (-50.0, 50.0)

    def objective(trial):
        params = {nm: trial.suggest_float(nm, *bounds[nm]) for nm in PARAM_NAMES}
        pr = predict(eps_v, mu_v, eta_v, n_v, f_v, params)
        if pr is None or np.any(~np.isfinite(pr)):
            return 1e9
        return float(np.mean(np.abs(y_v - pr)))

    sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=x0, sigma0=10.0,
                                            n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(x0)
    for a_seed in (-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0):
        x0_a = dict(x0); x0_a["a_corr"] = a_seed
        study.enqueue_trial(x0_a)
    print(f"\nOptimising {N_TRIALS} trials...", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {nm: study.best_params[nm] for nm in PARAM_NAMES}
    pred_full = predict(eps, mu, eta, n_desc, fukui, best)
    pred_full = np.where(valid, pred_full, np.nan)
    mae = float(np.nanmean(np.abs(y - pred_full)))
    rmse = float(np.sqrt(np.nanmean((y - pred_full) ** 2)))

    print(f"\n=== Final IP model ===")
    print(f"  Form: [β_μ(ε)·μ + β_η(ε)·η + β_n(ε)·n] × (1 + a_corr · f²_min_all)")
    print(f"  with n = 1/ω⁻ = 2η/(η/2−μ)²")
    print(f"  MAE  = {mae:.3f} kJ/mol")
    print(f"  RMSE = {rmse:.3f} kJ/mol")

    print(f"\n=== Optimised coefficients (10 params, intercept = 0) ===")
    for k in PARAM_NAMES:
        print(f"  {k:10s} = {best[k]:+.6f}")

    print(f"\n=== Per-solvent breakdown ===")
    rows = []
    for solv, sub in df.groupby("solvent"):
        idx = sub.index
        mask = np.isfinite(pred_full[idx])
        if mask.sum() == 0: continue
        e = float(np.nanmean(np.abs(y[idx] - pred_full[idx])))
        re = float(np.sqrt(np.nanmean((y[idx] - pred_full[idx]) ** 2)))
        rows.append({"solvent": solv, "epsilon": EPSILON_R[solv],
                     "MAE": e, "RMSE": re, "n": int(mask.sum())})
    persol = pd.DataFrame(rows).sort_values("epsilon")
    print(persol.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    print(f"\n=== Comparison with previous IP models ===")
    refs = [
        ("(μ,η,ω) Padé g_Born seul (sans Fukui)",                              4.387, 7.306, 9),
        ("(μ,η,ω) × (1+a·f²_min_noO)                       [previous]",         3.449, 6.403, 10),
        ("(μ,η,ω) × (1+a·f²_min_all)                       [updated]",          3.397, 6.005, 10),
        ("(μ,η,1/ω⁻) × (1+a·f²_min_noO)                    [Roy + noO]",        3.182, 6.132, 10),
        ("(μ,η,1/ω⁻) × (1+a·f²_min_all)                    [THIS]",             mae,   rmse, 10),
        ("Rapport (Padé[2/1] en ε)                          [external]",        5.300, 8.600, 12),
    ]
    print(f"  {'Model':<60s} {'MAE':>6s} {'RMSE':>6s} {'n_p':>3s}")
    best_mae = min(r[1] for r in refs)
    for name, m, r, np_ in refs:
        flag = " ← best" if m == best_mae else ""
        print(f"  {name:<60s} {m:6.3f} {r:6.3f} {np_:3d}{flag}")

    # Save
    final_model = {
        "form": "[β_μ(ε)·μ + β_η(ε)·η + β_n(ε)·n] × (1 + a_corr · f²_min_all)",
        "n_definition": "n = 1/ω⁻ = 2*eta / (eta/2 − mu)² = 2η/I²  (Roy nucleophilicity, no scaling factor)",
        "fukui_definition": "f²_min over all atoms (no exclusion) of f² = f⁺ − f⁻",
        "g_born": "g(ε) = (ε−1)/(2ε+1)",
        "beta_X": "β_X(ε) = (a0_X + a1_X·g)/(1 + b1_X·g)",
        "intercept": 0.0,
        "n_conformers": int(valid.sum()),
        "n_molecules": int(df.molecule.nunique()),
        "solvents": sorted(df.solvent.unique()),
        "filters": "exclude Br-DA, Br-HMC, I-DA, I-HMC, L-DOPA; EA<0 dropped at parquet build",
        "MAE_kJmol": mae,
        "RMSE_kJmol": rmse,
        "params": {
            "mu": {"a0": best["a0_mu"], "a1": best["a1_mu"], "b1": best["b1_mu"]},
            "eta": {"a0": best["a0_eta"], "a1": best["a1_eta"], "b1": best["b1_eta"]},
            "n": {"a0": best["a0_n"], "a1": best["a1_n"], "b1": best["b1_n"]},
            "a_corr": best["a_corr"],
        },
    }
    with open(OUT_DIR / "final_ip_model.json", "w") as f:
        json.dump(final_model, f, indent=2)
    print(f"\nSaved -> {(OUT_DIR / 'final_ip_model.json').relative_to(REPO)}")

    # Parity plot
    fig, ax = plt.subplots(figsize=(7, 6))
    solvents_sorted = sorted(df.solvent.unique(), key=lambda s: EPSILON_R[s])
    cmap = plt.get_cmap("viridis")
    colours = {s: cmap(i / max(len(solvents_sorted) - 1, 1)) for i, s in enumerate(solvents_sorted)}
    for s in solvents_sorted:
        mask = (df.solvent == s).values & valid
        ax.scatter(y[mask], pred_full[mask], s=18, alpha=0.7,
                   color=colours[s], label=f"{s} (ε={EPSILON_R[s]:g})")
    lo = min(np.nanmin(y), np.nanmin(pred_full))
    hi = max(np.nanmax(y), np.nanmax(pred_full))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("IP référence (kJ/mol)")
    ax.set_ylabel("IP prédit (kJ/mol)")
    ax.set_title(f"IP — (μ, η, 1/ω⁻) × Padé[1/1] g_Born × (1 + a·f²_min_all)\n"
                 f"MAE = {mae:.2f} kJ/mol,  RMSE = {rmse:.2f} kJ/mol  (n = {int(valid.sum())})")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_ip_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {(OUT_DIR / 'parity_ip_final.png').relative_to(REPO)}")


if __name__ == "__main__":
    main()
