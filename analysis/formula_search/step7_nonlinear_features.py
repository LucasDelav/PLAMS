#!/usr/bin/env python3
"""Étape 7 — explorer combinaisons non-linéaires de (μ, η) au-delà de ω.

(A) ADD : pour chaque candidat X = f(μ, η), tester (μ, η, X, LVC2) et voir
    si X bat ω = μ²/(2η) sur le ΔG total.
(B) REPLACE : tester si remplacer μ ou η par une non-linéarité aide.
(C) DUO : essayer deux features non-linéaires en plus de (μ, η, LVC2).
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
    if not np.all(np.isfinite(feats)) or np.std(feats) == 0:
        return None
    # Z-score normalize features for numerical stability with non-linearities
    f_means = feats.mean(axis=1, keepdims=True)
    f_stds = feats.std(axis=1, keepdims=True)
    f_stds[f_stds == 0] = 1.0
    feats_n = (feats - f_means) / f_stds
    nf = feats_n.shape[0]
    model, total = make_model(nf, family)
    p0, bnd = initial_guess(nf, family)
    X = np.vstack([eps[None, :], feats_n])
    best = None
    for trial in [p0, p0 * 0.1, p0 * 10]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, X, y, p0=trial, bounds=bnd,
                                    maxfev=3000, method="trf")
            yhat = model(X, *popt)
            if not (np.all(np.isfinite(yhat))):
                continue
            mae = float(np.mean(np.abs(y - yhat)))
            if best is None or mae < best["mae"]:
                best = {"yhat": yhat, "mae": mae}
        except Exception:
            continue
    return best


def evaluate(features_dict, eps, targets, families, lvc2_red, lvc2_ox):
    """features_dict = ordered dict of name->vector. LVC2 if present is paired."""
    yhats = {}
    maes = {}
    for tgt, y in targets.items():
        feats = []
        for name, vec in features_dict.items():
            if name == "LVC2":
                feats.append(lvc2_red if tgt in {"EA", "Erelax_red", "ddU_red", "TdS_red"} else lvc2_ox)
            else:
                feats.append(vec)
        feats = np.array(feats)
        res = fit(eps, feats, y, families[tgt])
        if res is None:
            return None
        yhats[tgt] = res["yhat"]
        maes[tgt] = res["mae"]
    dg_red_t = -targets["EA"] + targets["Erelax_red"] + targets["ddU_red"] - targets["TdS_red"]
    dg_red_p = -yhats["EA"] + yhats["Erelax_red"] + yhats["ddU_red"] - yhats["TdS_red"]
    dg_ox_t = +targets["IP"] + targets["Erelax_ox"] + targets["ddU_ox"] - targets["TdS_ox"]
    dg_ox_p = +yhats["IP"] + yhats["Erelax_ox"] + yhats["ddU_ox"] - yhats["TdS_ox"]
    return {
        "maes": maes,
        "sum_mae_8": sum(maes.values()),
        "dG_red": float(np.mean(np.abs(dg_red_t - dg_red_p))),
        "dG_ox": float(np.mean(np.abs(dg_ox_t - dg_ox_p))),
        "dG_total": float(np.mean(np.abs(dg_red_t - dg_red_p))) + float(np.mean(np.abs(dg_ox_t - dg_ox_p))),
    }


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    eps = df["eps"].to_numpy()
    print(f"Dataset: {len(df)} rows, {df.molecule.nunique()} mols")

    mu = df.mu.to_numpy() * EV_TO_KJ
    eta = df.eta.to_numpy() * EV_TO_KJ
    lvc2_red = df.lvc2_red.to_numpy() * EH_TO_KJ
    lvc2_ox = df.lvc2_ox.to_numpy() * EH_TO_KJ

    targets = {
        "EA": df.EA.to_numpy() * EH_TO_KJ,
        "IP": df.IP.to_numpy() * EH_TO_KJ,
        "Erelax_red": df.Erelax_red.to_numpy() * EH_TO_KJ,
        "Erelax_ox": df.Erelax_ox.to_numpy() * EH_TO_KJ,
        "ddU_red": df.ddU_red.to_numpy() * EH_TO_KJ,
        "ddU_ox": df.ddU_ox.to_numpy() * EH_TO_KJ,
        "TdS_red": df.TdS_red.to_numpy() * EH_TO_KJ,
        "TdS_ox": df.TdS_ox.to_numpy() * EH_TO_KJ,
    }
    families = {
        "EA": "pade_21", "IP": "pade_21",
        "Erelax_red": "constant", "Erelax_ox": "constant",
        "ddU_red": "pade_21", "ddU_ox": "pade_21",
        "TdS_red": "constant", "TdS_ox": "constant",
    }

    # === Library of nonlinear feature candidates ===
    safe_div = lambda a, b: np.where(np.abs(b) > 1e-9, a / b, 0.0)
    candidates = {
        "ω = μ²/2η":         mu ** 2 / (2 * eta),
        "μ²":                 mu ** 2,
        "η²":                 eta ** 2,
        "1/η (softness)":     safe_div(1.0, eta),
        "1/μ":                safe_div(1.0, mu),
        "μ·η":                mu * eta,
        "μ/η":                safe_div(mu, eta),
        "η/μ":                safe_div(eta, mu),
        "μ²/η":               safe_div(mu ** 2, eta),
        "μ/η²":               safe_div(mu, eta ** 2),
        "μ²·η":               mu ** 2 * eta,
        "μ²/η²":              safe_div(mu ** 2, eta ** 2),
        "√η":                 np.sqrt(np.maximum(eta, 1e-9)),
        "√|μ|":               np.sqrt(np.abs(mu)),
        "log(η)":             np.log(np.maximum(eta, 1e-9)),
        "log|μ|":             np.log(np.maximum(np.abs(mu), 1e-9)),
        "μ³":                 mu ** 3,
        "η³":                 eta ** 3,
        "μ³/η²":              safe_div(mu ** 3, eta ** 2),
        "(μ-η/2)² (HOMO²)":   (mu - eta / 2) ** 2,
        "(μ+η/2)² (LUMO²)":   (mu + eta / 2) ** 2,
        "μ²+η²":              mu ** 2 + eta ** 2,
        "μ²-η²":              mu ** 2 - eta ** 2,
    }

    # === Reference results ===
    base = {"μ": mu, "η": eta, "LVC2": "PAIRED"}
    ref_2 = evaluate({"μ": mu, "η": eta, "LVC2": "PAIRED"}, eps, targets, families, lvc2_red, lvc2_ox)
    ref_omega = evaluate({"μ": mu, "η": eta, "ω": mu ** 2 / (2 * eta), "LVC2": "PAIRED"},
                         eps, targets, families, lvc2_red, lvc2_ox)
    print(f"\nBaseline (μ,η,LVC2)            : ΔG_tot={ref_2['dG_total']:.3f}  sum8={ref_2['sum_mae_8']:.3f}")
    print(f"Réf      (μ,η,ω=μ²/2η,LVC2)    : ΔG_tot={ref_omega['dG_total']:.3f}  sum8={ref_omega['sum_mae_8']:.3f}")

    # === (A) ADD a single nonlinear feature to (μ, η, LVC2) ===
    print("\n" + "=" * 80)
    print("(A) AJOUT d'UNE feature non-linéaire à (μ, η, LVC2)")
    print("=" * 80)
    rows_A = []
    for name, vec in candidates.items():
        feats = {"μ": mu, "η": eta, name: vec, "LVC2": "PAIRED"}
        res = evaluate(feats, eps, targets, families, lvc2_red, lvc2_ox)
        if res is None:
            continue
        rows_A.append({
            "feature": name,
            "ΔG_red": res["dG_red"], "ΔG_ox": res["dG_ox"],
            "ΔG_total": res["dG_total"], "sum8": res["sum_mae_8"],
            "Δvs_ω": res["dG_total"] - ref_omega["dG_total"],
        })
    Adf = pd.DataFrame(rows_A).sort_values("ΔG_total")
    print(Adf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # === (B) REPLACE μ or η by a nonlinear function ===
    print("\n" + "=" * 80)
    print("(B) REMPLACER μ ou η par une non-linéarité (3 features + LVC2)")
    print("=" * 80)
    rows_B = []
    for slot, base_name, base_vec in [("μ", "η", eta), ("η", "μ", mu)]:
        for name, vec in candidates.items():
            if name in ("μ²", "η²", "ω = μ²/2η"):  # already known
                pass
            feats = {name: vec, base_name: base_vec, "LVC2": "PAIRED"}
            res = evaluate(feats, eps, targets, families, lvc2_red, lvc2_ox)
            if res is None:
                continue
            rows_B.append({
                "replaces": slot,
                "by": name,
                "kept": base_name,
                "ΔG_red": res["dG_red"], "ΔG_ox": res["dG_ox"],
                "ΔG_total": res["dG_total"], "sum8": res["sum_mae_8"],
                "Δvs_ω": res["dG_total"] - ref_omega["dG_total"],
            })
    Bdf = pd.DataFrame(rows_B).sort_values("ΔG_total")
    print(Bdf.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # === (C) DUO: ajouter DEUX non-linéarités à (μ, η, LVC2) ===
    print("\n" + "=" * 80)
    print("(C) AJOUT de DEUX non-linéarités à (μ, η, LVC2)")
    print("=" * 80)
    rows_C = []
    cand_keys = list(candidates.keys())
    for k1, k2 in combinations(cand_keys, 2):
        feats = {"μ": mu, "η": eta, k1: candidates[k1], k2: candidates[k2], "LVC2": "PAIRED"}
        res = evaluate(feats, eps, targets, families, lvc2_red, lvc2_ox)
        if res is None:
            continue
        rows_C.append({
            "extras": f"{k1} + {k2}",
            "ΔG_total": res["dG_total"], "sum8": res["sum_mae_8"],
            "Δvs_ω": res["dG_total"] - ref_omega["dG_total"],
        })
    Cdf = pd.DataFrame(rows_C).sort_values("ΔG_total")
    print(Cdf.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    Adf.to_csv(RESULTS / "step7_addone.csv", index=False)
    Bdf.to_csv(RESULTS / "step7_replace.csv", index=False)
    Cdf.to_csv(RESULTS / "step7_addtwo.csv", index=False)
    print(f"\n[saved] step7_addone.csv, step7_replace.csv, step7_addtwo.csv")


if __name__ == "__main__":
    main()
