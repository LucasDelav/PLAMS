#!/usr/bin/env python3
"""Étape 6 — Ablation des descripteurs CDFT.

Part du pool {μ, η, γ, ω, LVC2} et retire à chaque tour le descripteur dont
l'absence dégrade le moins le modèle global.

Architecture identique à step5 :
  - EA, IP, ddU : Padé[2/1](ε), 4 params/coef
  - Erelax, TdS : constant en ε,    1 param/coef
  - LVC2 paired par direction (red/ox)
  - Filtres : DEA Br/I + L-DOPA + EA<0
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
RESULTS = ROOT / "results"
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
    if family == "pade_21":
        return (p[0] + p[1] * eps + p[2] * eps ** 2) / (1.0 + p[3] * eps)


def make_model(n_features, family):
    npp = FAMILY_NPP[family]

    def model(X, *params):
        eps = X[0]
        out = np.zeros_like(eps, dtype=float)
        for i in range(n_features):
            off = i * npp
            out += eval_coeff(family, eps, params[off:off + npp]) * X[1 + i]
        off = n_features * npp
        out += eval_coeff(family, eps, params[off:off + npp])
        return out

    return model, (n_features + 1) * npp


def initial_guess(n_features, family):
    p0, lo, hi = [], [], []
    for _ in range(n_features + 1):
        if family == "constant":
            p0 += [0.0]
            lo += [-np.inf]
            hi += [np.inf]
        else:
            p0 += [1.0, 0.01, 0.0001, 0.01]
            lo += [-np.inf, -np.inf, -np.inf, 0.0]
            hi += [np.inf, np.inf, np.inf, np.inf]
    return np.array(p0), (np.array(lo), np.array(hi))


def fit_model(eps, feats, y, family):
    feats = np.atleast_2d(feats).astype(float)
    nf = feats.shape[0]
    model, total = make_model(nf, family)
    p0, bnd = initial_guess(nf, family)
    X = np.vstack([eps[None, :], feats])
    best = None
    for trial in [p0, p0 * 0.1, p0 * 10]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, X, y, p0=trial, bounds=bnd,
                                    maxfev=20000, method="trf")
            yhat = model(X, *popt)
            if np.any(np.isnan(yhat)) or np.any(np.isinf(yhat)):
                continue
            mae = float(np.mean(np.abs(y - yhat)))
            if best is None or mae < best["mae"]:
                best = {"yhat": yhat, "mae": mae, "n_params": total}
        except Exception:
            continue
    return best


def metrics_simple(y, yhat):
    err = y - yhat
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def evaluate_pool(pool_names, all_features, eps, targets, families):
    """Fit every target with the given subset of descriptors. Return per-target
    MAE + ΔG combined MAE."""
    feats_pool = np.array([all_features[k] for k in pool_names])
    yhats = {}
    maes = {}
    for tgt, y in targets.items():
        feats = feats_pool.copy()
        # paired LVC2: replace generic LVC2 column with red/ox-specific values
        if "LVC2" in pool_names:
            idx = pool_names.index("LVC2")
            if tgt in {"EA", "Erelax_red", "ddU_red", "TdS_red"}:
                feats[idx] = all_features["LVC2_red"]
            else:
                feats[idx] = all_features["LVC2_ox"]
        res = fit_model(eps, feats, y, families[tgt])
        if res is None:
            return None
        yhats[tgt] = res["yhat"]
        maes[tgt] = res["mae"]
    # Combined ΔG
    dg_red_t = -targets["EA"] + targets["Erelax_red"] + targets["ddU_red"] - targets["TdS_red"]
    dg_red_p = -yhats["EA"] + yhats["Erelax_red"] + yhats["ddU_red"] - yhats["TdS_red"]
    dg_ox_t = +targets["IP"] + targets["Erelax_ox"] + targets["ddU_ox"] - targets["TdS_ox"]
    dg_ox_p = +yhats["IP"] + yhats["Erelax_ox"] + yhats["ddU_ox"] - yhats["TdS_ox"]
    mae_dg_red, _ = metrics_simple(dg_red_t, dg_red_p)
    mae_dg_ox, _ = metrics_simple(dg_ox_t, dg_ox_p)
    return {
        "maes": maes,
        "sum_mae_8": sum(maes.values()),
        "mae_dG_red": mae_dg_red,
        "mae_dG_ox": mae_dg_ox,
        "mae_dG_total": mae_dg_red + mae_dg_ox,
        "n_targets": len(maes),
    }


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    print(f"Dataset: {len(df)} rows, {df.molecule.nunique()} mols")

    eps = df["eps"].to_numpy()
    mu_eV = df.mu.to_numpy()
    eta_eV = df.eta.to_numpy()
    gamma_eV = df.gamma.to_numpy()
    omega_eV = mu_eV ** 2 / (2.0 * eta_eV)

    all_features = {
        "μ":      mu_eV * EV_TO_KJ,
        "η":      eta_eV * EV_TO_KJ,
        "γ":      gamma_eV * EV_TO_KJ,
        "ω":      omega_eV * EV_TO_KJ,
        "LVC2":   df.lvc2_red.to_numpy() * EH_TO_KJ,  # placeholder, paired in evaluate_pool
        "LVC2_red": df.lvc2_red.to_numpy() * EH_TO_KJ,
        "LVC2_ox":  df.lvc2_ox.to_numpy() * EH_TO_KJ,
    }

    targets = {
        "EA":         df.EA.to_numpy() * EH_TO_KJ,
        "IP":         df.IP.to_numpy() * EH_TO_KJ,
        "Erelax_red": df.Erelax_red.to_numpy() * EH_TO_KJ,
        "Erelax_ox":  df.Erelax_ox.to_numpy() * EH_TO_KJ,
        "ddU_red":    df.ddU_red.to_numpy() * EH_TO_KJ,
        "ddU_ox":     df.ddU_ox.to_numpy() * EH_TO_KJ,
        "TdS_red":    df.TdS_red.to_numpy() * EH_TO_KJ,
        "TdS_ox":     df.TdS_ox.to_numpy() * EH_TO_KJ,
    }

    families = {
        "EA": "pade_21", "IP": "pade_21",
        "Erelax_red": "constant", "Erelax_ox": "constant",
        "ddU_red": "pade_21", "ddU_ox": "pade_21",
        "TdS_red": "constant", "TdS_ox": "constant",
    }

    pool = ["μ", "η", "γ", "ω", "LVC2"]

    # Reference: full pool
    ref = evaluate_pool(pool, all_features, eps, targets, families)
    print(f"\n=== POOL DE RÉFÉRENCE: {pool} ===")
    print(f"   sum MAE 8 cibles = {ref['sum_mae_8']:.3f} kJ/mol")
    print(f"   ΔG_red MAE = {ref['mae_dG_red']:.3f} kJ/mol")
    print(f"   ΔG_ox  MAE = {ref['mae_dG_ox']:.3f} kJ/mol")
    print(f"   ΔG total   = {ref['mae_dG_total']:.3f} kJ/mol")
    print("   Per-target MAE:")
    for tgt, m in ref["maes"].items():
        print(f"      {tgt:12s}: {m:.3f}")

    history = [{"pool": list(pool), "ref": ref}]

    # Iterate ablation
    while len(pool) > 2:
        print(f"\n{'='*70}\nABLATION du pool {pool} (taille {len(pool)})\n{'='*70}")
        rows = []
        for d in pool:
            sub_pool = [x for x in pool if x != d]
            res = evaluate_pool(sub_pool, all_features, eps, targets, families)
            if res is None:
                continue
            row = {
                "removed": d,
                "remaining": "+".join(sub_pool),
                "sum_mae_8": res["sum_mae_8"],
                "mae_dG_red": res["mae_dG_red"],
                "mae_dG_ox": res["mae_dG_ox"],
                "mae_dG_total": res["mae_dG_total"],
                "Δsum_mae_8": res["sum_mae_8"] - ref["sum_mae_8"],
                "Δmae_dG_total": res["mae_dG_total"] - ref["mae_dG_total"],
            }
            rows.append(row)
        rdf = pd.DataFrame(rows).sort_values("Δmae_dG_total")
        print(rdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        best_to_remove = rdf.iloc[0]["removed"]
        new_pool = [x for x in pool if x != best_to_remove]
        new_ref = evaluate_pool(new_pool, all_features, eps, targets, families)
        print(f"\n→ Retrait du moins informatif : {best_to_remove}")
        print(f"   nouveau pool = {new_pool}")
        print(f"   sum MAE 8 cibles = {new_ref['sum_mae_8']:.3f} kJ/mol "
              f"(Δ = +{new_ref['sum_mae_8'] - ref['sum_mae_8']:.3f})")
        print(f"   ΔG_red MAE = {new_ref['mae_dG_red']:.3f}  "
              f"(Δ = {new_ref['mae_dG_red'] - ref['mae_dG_red']:+.3f})")
        print(f"   ΔG_ox  MAE = {new_ref['mae_dG_ox']:.3f}  "
              f"(Δ = {new_ref['mae_dG_ox'] - ref['mae_dG_ox']:+.3f})")
        print("   Per-target MAE:")
        for tgt, m in new_ref["maes"].items():
            delta = m - ref["maes"][tgt]
            print(f"      {tgt:12s}: {m:6.3f}   (Δ {delta:+.3f})")

        history.append({"pool": new_pool, "ref": new_ref,
                        "removed": best_to_remove,
                        "ablation_table": rdf})
        pool = new_pool
        ref = new_ref

    # Save
    summary_rows = []
    for h in history:
        r = h["ref"]
        summary_rows.append({
            "pool": "+".join(h["pool"]),
            "size": len(h["pool"]),
            "sum_mae_8": r["sum_mae_8"],
            "mae_dG_red": r["mae_dG_red"],
            "mae_dG_ox": r["mae_dG_ox"],
            "mae_dG_total": r["mae_dG_total"],
            **{f"MAE_{k}": v for k, v in r["maes"].items()},
        })
    sdf = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("HISTORIQUE DES POOLS APRÈS ABLATION SUCCESSIVE")
    print("=" * 90)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sdf.to_csv(RESULTS / "step6_ablation_history.csv", index=False)
    print(f"\n[saved] step6_ablation_history.csv")


if __name__ == "__main__":
    main()
