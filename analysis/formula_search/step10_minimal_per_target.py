#!/usr/bin/env python3
"""Étape 10 — sous-pool minimal nécessaire par cible.

Pour chacune des 8 cibles, teste tous les 15 sous-ensembles non-vides du pool
{μ, μ², η, LVC2} et identifie le sous-pool minimal qui prédit aussi bien que
le pool complet à 4 features.

Architecture : Padé[2/1](ε) pour EA/IP/ddU, constant pour Erelax/TdS, LVC2
paired par direction.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
EH_TO_KJ = 2625.499638
EV_TO_KJ = 96.48533212
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}
FAMILY_NPP = {"pade_21": 4, "constant": 1}


def eval_coeff(family, eps, p):
    if family == "constant":
        return np.full_like(eps, p[0], dtype=float)
    return (p[0] + p[1] * eps + p[2] * eps ** 2) / (1.0 + p[3] * eps)


def make_model(nf, family):
    npp = FAMILY_NPP[family]

    def model(X, *params):
        eps = X[0]
        out = np.zeros_like(eps, dtype=float)
        for i in range(nf):
            off = i * npp
            out += eval_coeff(family, eps, params[off:off + npp]) * X[1 + i]
        off = nf * npp
        out += eval_coeff(family, eps, params[off:off + npp])
        return out

    return model, (nf + 1) * npp


def initial_guess(nf, family):
    p0, lo, hi = [], [], []
    for _ in range(nf + 1):
        if family == "constant":
            p0 += [0.0]; lo += [-np.inf]; hi += [np.inf]
        else:
            p0 += [1.0, 0.01, 0.0001, 0.01]
            lo += [-np.inf, -np.inf, -np.inf, 0.0]
            hi += [np.inf, np.inf, np.inf, np.inf]
    return np.array(p0), (np.array(lo), np.array(hi))


def fit(eps, feats, y, family):
    feats = np.atleast_2d(feats).astype(float)
    f_means = feats.mean(axis=1, keepdims=True)
    f_stds = feats.std(axis=1, keepdims=True)
    f_stds[f_stds == 0] = 1.0
    feats_n = (feats - f_means) / f_stds
    nf = feats_n.shape[0]
    model, total = make_model(nf, family)
    p0, bnd = initial_guess(nf, family)
    X = np.vstack([eps[None, :], feats_n])
    best_mae = np.inf
    for trial in [p0, p0 * 0.1, p0 * 10]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, X, y, p0=trial, bounds=bnd,
                                    maxfev=5000, method="trf")
            yhat = model(X, *popt)
            if not np.all(np.isfinite(yhat)):
                continue
            mae = float(np.mean(np.abs(y - yhat)))
            if mae < best_mae:
                best_mae = mae
        except Exception:
            continue
    return best_mae if np.isfinite(best_mae) else None


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    eps = df["solvent"].map(EPSILON_R).to_numpy()
    print(f"Dataset: {len(df)} rows, {df.molecule.nunique()} mols")

    mu = df.mu.to_numpy() * EV_TO_KJ
    eta = df.eta.to_numpy() * EV_TO_KJ
    mu2 = mu ** 2
    lvc2_red = df.lvc2_red.to_numpy() * EH_TO_KJ
    lvc2_ox = df.lvc2_ox.to_numpy() * EH_TO_KJ

    targets = {
        "EA": (df.EA.to_numpy() * EH_TO_KJ, "pade_21", "red"),
        "IP": (df.IP.to_numpy() * EH_TO_KJ, "pade_21", "ox"),
        "Erelax_red": (df.Erelax_red.to_numpy() * EH_TO_KJ, "constant", "red"),
        "Erelax_ox":  (df.Erelax_ox.to_numpy() * EH_TO_KJ, "constant", "ox"),
        "ddU_red":    (df.ddU_red.to_numpy() * EH_TO_KJ, "pade_21", "red"),
        "ddU_ox":     (df.ddU_ox.to_numpy() * EH_TO_KJ, "pade_21", "ox"),
        "TdS_red":    (df.TdS_red.to_numpy() * EH_TO_KJ, "constant", "red"),
        "TdS_ox":     (df.TdS_ox.to_numpy() * EH_TO_KJ, "constant", "ox"),
    }

    pool_names = ["μ", "μ²", "η", "LVC2"]
    pool_arrays = {"μ": mu, "μ²": mu2, "η": eta}

    rows = []
    for tgt_name, (y, family, direction) in targets.items():
        lvc2_arr = lvc2_red if direction == "red" else lvc2_ox
        full_arrays = dict(pool_arrays)
        full_arrays["LVC2"] = lvc2_arr

        # Reference : full pool of 4
        feats_full = np.array([full_arrays[n] for n in pool_names])
        mae_ref = fit(eps, feats_full, y, family)

        # Test all 15 non-empty subsets
        for size in range(1, 5):
            for combo in combinations(pool_names, size):
                feats = np.array([full_arrays[n] for n in combo])
                mae = fit(eps, feats, y, family)
                if mae is None:
                    continue
                rows.append({
                    "target": tgt_name,
                    "subset": "+".join(combo),
                    "size": size,
                    "MAE": mae,
                    "MAE_full4": mae_ref,
                    "Δ_vs_full": mae - mae_ref,
                    "ratio_vs_full": mae / mae_ref,
                })

    rdf = pd.DataFrame(rows)
    rdf.to_csv(ROOT / "results" / "step10_subsets.csv", index=False)
    print(f"\n[saved] step10_subsets.csv ({len(rdf)} fits)")

    # --- Synthèse par cible : pour chaque taille, le meilleur sous-pool
    print("\n" + "=" * 110)
    print("SOUS-POOL MINIMAL PAR CIBLE (best subset à chaque taille)")
    print("=" * 110)
    summary = []
    for tgt in targets:
        sub = rdf[rdf.target == tgt].sort_values("MAE")
        full = sub[sub["size"] == 4].iloc[0]
        print(f"\n--- {tgt}  (full4 MAE = {full['MAE']:.3f}) ---")
        for size in (1, 2, 3, 4):
            best = rdf[(rdf.target == tgt) & (rdf["size"] == size)].sort_values("MAE").iloc[0]
            tag = ""
            if best["Δ_vs_full"] <= 0.05:
                tag = "★ identique au full"
            elif best["Δ_vs_full"] <= 0.20:
                tag = "≈ acceptable (<0.2)"
            print(f"   k={size}  {best['subset']:18s}  MAE={best['MAE']:6.3f}  "
                  f"Δ vs full4={best['Δ_vs_full']:+.3f}  {tag}")
            summary.append({
                "target": tgt, "size": size,
                "best_subset": best["subset"], "MAE": best["MAE"],
                "Δ_vs_full": best["Δ_vs_full"],
            })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(ROOT / "results" / "step10_minimal_summary.csv", index=False)
    print(f"\n[saved] step10_minimal_summary.csv")

    # --- Recommendation : sous-pool minimal acceptable par cible
    print("\n" + "=" * 110)
    print("RECOMMANDATION : sous-pool minimal avec Δ ≤ 0.10 kJ/mol vs pool 4-features")
    print("=" * 110)
    print(f"{'target':14s} {'sous-pool minimal':25s} {'k':>2s}  {'MAE':>7s}  {'Δ_vs_full':>10s}")
    print("-" * 70)
    final = []
    for tgt in targets:
        sub = rdf[rdf.target == tgt].sort_values(["size", "MAE"])
        # Find smallest size achieving Δ ≤ 0.10
        chosen = None
        for size in (1, 2, 3, 4):
            best = sub[sub["size"] == size].iloc[0]
            if best["Δ_vs_full"] <= 0.10:
                chosen = best
                break
        if chosen is None:
            chosen = sub[sub["size"] == 4].iloc[0]
        print(f"{tgt:14s} {chosen['subset']:25s} {int(chosen['size']):2d}  "
              f"{chosen['MAE']:7.3f}  {chosen['Δ_vs_full']:+10.3f}")
        final.append({"target": tgt, "min_subset": chosen["subset"],
                      "k": int(chosen["size"]), "MAE": chosen["MAE"],
                      "Δ_vs_full": chosen["Δ_vs_full"]})
    pd.DataFrame(final).to_csv(ROOT / "results" / "step10_recommendation.csv", index=False)
    print(f"\n[saved] step10_recommendation.csv")


if __name__ == "__main__":
    main()
