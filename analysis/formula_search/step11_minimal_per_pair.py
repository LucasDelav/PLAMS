#!/usr/bin/env python3
"""Étape 11 — sous-pool minimal partagé par paire (red+ox).

Contrainte : la même combinaison de descripteurs doit être utilisée pour les
deux cibles d'une paire. Seuil de tolérance : Δ(somme_MAE_paire) ≤ 0.10 kJ/mol
vs pool complet à 4 features.

Le sous-pool peut différer entre paires (EA/IP vs Erelax vs ddU vs TdS).
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

    mu = df.mu.to_numpy() * EV_TO_KJ
    eta = df.eta.to_numpy() * EV_TO_KJ
    mu2 = mu ** 2
    lvc2_red = df.lvc2_red.to_numpy() * EH_TO_KJ
    lvc2_ox = df.lvc2_ox.to_numpy() * EH_TO_KJ

    pairs = {
        "EA/IP":     (("EA", "pade_21", df.EA.to_numpy() * EH_TO_KJ),
                      ("IP", "pade_21", df.IP.to_numpy() * EH_TO_KJ)),
        "Erelax":    (("Erelax_red", "constant", df.Erelax_red.to_numpy() * EH_TO_KJ),
                      ("Erelax_ox",  "constant", df.Erelax_ox.to_numpy() * EH_TO_KJ)),
        "ddU":       (("ddU_red", "pade_21", df.ddU_red.to_numpy() * EH_TO_KJ),
                      ("ddU_ox",  "pade_21", df.ddU_ox.to_numpy() * EH_TO_KJ)),
        "TdS":       (("TdS_red", "constant", df.TdS_red.to_numpy() * EH_TO_KJ),
                      ("TdS_ox",  "constant", df.TdS_ox.to_numpy() * EH_TO_KJ)),
    }
    pool_names = ["μ", "μ²", "η", "LVC2"]
    pool_static = {"μ": mu, "μ²": mu2, "η": eta}

    rows = []
    for pair_name, ((rname, rfam, ry), (oname, ofam, oy)) in pairs.items():
        for size in range(1, 5):
            for combo in combinations(pool_names, size):
                feats_red = []
                feats_ox = []
                for n in combo:
                    if n == "LVC2":
                        feats_red.append(lvc2_red); feats_ox.append(lvc2_ox)
                    else:
                        feats_red.append(pool_static[n]); feats_ox.append(pool_static[n])
                mae_r = fit(eps, np.array(feats_red), ry, rfam)
                mae_o = fit(eps, np.array(feats_ox), oy, ofam)
                if mae_r is None or mae_o is None:
                    continue
                rows.append({
                    "pair": pair_name, "subset": "+".join(combo),
                    "k": size,
                    f"MAE_{rname}": mae_r,
                    f"MAE_{oname}": mae_o,
                    "MAE_pair_sum": mae_r + mae_o,
                })

    rdf = pd.DataFrame(rows)

    # ΔG combined per pair (just take per-pair sum)
    print("\n" + "=" * 100)
    print("MEILLEUR SOUS-POOL PARTAGÉ PAR PAIRE (à chaque taille k)")
    print("=" * 100)
    summary = []
    for pair_name in pairs:
        sub = rdf[rdf.pair == pair_name].copy()
        full = sub[sub["k"] == 4].iloc[0]
        full_sum = full["MAE_pair_sum"]
        print(f"\n--- {pair_name}  (full4 sum_MAE = {full_sum:.3f}) ---")
        for size in (1, 2, 3, 4):
            best = sub[sub["k"] == size].sort_values("MAE_pair_sum").iloc[0]
            delta = best["MAE_pair_sum"] - full_sum
            tag = ""
            if delta <= 0.10:
                tag = "★ acceptable"
            elif delta <= 0.30:
                tag = "≈ proche"
            print(f"   k={size}  {best['subset']:18s}  "
                  f"sum={best['MAE_pair_sum']:6.3f}  "
                  f"Δ={delta:+.3f}  {tag}")

        # Find smallest k achieving Δ ≤ 0.10
        chosen = None
        for size in (1, 2, 3, 4):
            best = sub[sub["k"] == size].sort_values("MAE_pair_sum").iloc[0]
            if best["MAE_pair_sum"] - full_sum <= 0.10:
                chosen = best
                break
        if chosen is None:
            chosen = full
        summary.append({
            "pair": pair_name,
            "min_subset": chosen["subset"],
            "k": int(chosen["k"]),
            **{c: chosen[c] for c in chosen.index if c.startswith("MAE_")},
            "Δ_vs_full": chosen["MAE_pair_sum"] - full_sum,
        })

    print("\n" + "=" * 100)
    print("RECOMMANDATION FINALE — sous-pool partagé par paire (Δ ≤ 0.10 kJ/mol)")
    print("=" * 100)
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sdf.to_csv(ROOT / "results" / "step11_pair_recommendation.csv", index=False)
    rdf.to_csv(ROOT / "results" / "step11_pair_subsets.csv", index=False)
    print(f"\n[saved] step11_pair_recommendation.csv, step11_pair_subsets.csv")


if __name__ == "__main__":
    main()
