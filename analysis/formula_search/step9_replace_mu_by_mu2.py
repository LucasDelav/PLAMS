#!/usr/bin/env python3
"""Test : peut-on remplacer μ par μ² ? Pool minimal (μ², η, LVC2) ?

Comparaison de 3 pools de descripteurs avec architecture habituelle :
  - EA, IP, ddU : Padé[2/1](ε)
  - Erelax, TdS : coefficients constants en ε
  - LVC2 paired (red/ox)
"""
from __future__ import annotations

import warnings
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
    best = None
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
            if best is None or mae < best["mae"]:
                best = {"yhat": yhat, "mae": mae}
        except Exception:
            continue
    return best


def evaluate(name, descriptor_funcs, eps, targets, families, lvc2_red, lvc2_ox):
    """descriptor_funcs is dict {name: array (or 'LVC2_PAIRED' sentinel)}."""
    yhats = {}
    maes = {}
    for tgt, y in targets.items():
        feats = []
        for d_name, d_val in descriptor_funcs.items():
            if d_val is None:  # LVC2 paired
                feats.append(lvc2_red if tgt in {"EA", "Erelax_red", "ddU_red", "TdS_red"} else lvc2_ox)
            else:
                feats.append(d_val)
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
    mae_dg_red = float(np.mean(np.abs(dg_red_t - dg_red_p)))
    mae_dg_ox = float(np.mean(np.abs(dg_ox_t - dg_ox_p)))
    return {
        "name": name,
        "maes": maes,
        "sum_mae_8": sum(maes.values()),
        "mae_dG_red": mae_dg_red,
        "mae_dG_ox": mae_dg_ox,
        "mae_dG_total": mae_dg_red + mae_dg_ox,
    }


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

    pools = [
        ("(μ, η, LVC2)",        {"μ": mu, "η": eta, "LVC2": None}),
        ("(μ², η, LVC2)",       {"μ²": mu2, "η": eta, "LVC2": None}),
        ("(μ, η, μ², LVC2)",    {"μ": mu, "η": eta, "μ²": mu2, "LVC2": None}),
        ("(μ², η)",             {"μ²": mu2, "η": eta}),
        ("(μ, η)",              {"μ": mu, "η": eta}),
    ]

    rows = []
    print("\n" + "=" * 90)
    print(f"{'Pool':25s} {'#feat':>5s}  {'EA':>6s} {'IP':>6s} {'Eo_r':>6s} {'Eo_o':>6s} "
          f"{'ddU_r':>6s} {'ddU_o':>6s} {'TdS_r':>6s} {'TdS_o':>6s}  {'ΔG_r':>6s} {'ΔG_o':>6s} {'sum8':>6s}")
    print("=" * 90)
    for name, fdict in pools:
        nf = len(fdict)
        res = evaluate(name, fdict, eps, targets, families, lvc2_red, lvc2_ox)
        if res is None:
            print(f"{name:25s} FAIL")
            continue
        m = res["maes"]
        print(f"{name:25s} {nf:5d}  "
              f"{m['EA']:6.2f} {m['IP']:6.2f} {m['Erelax_red']:6.2f} {m['Erelax_ox']:6.2f} "
              f"{m['ddU_red']:6.2f} {m['ddU_ox']:6.2f} {m['TdS_red']:6.2f} {m['TdS_ox']:6.2f}  "
              f"{res['mae_dG_red']:6.2f} {res['mae_dG_ox']:6.2f} {res['sum_mae_8']:6.2f}")
        rows.append({
            "pool": name, "n_features": nf,
            **{f"MAE_{k}": v for k, v in m.items()},
            "MAE_dG_red": res["mae_dG_red"],
            "MAE_dG_ox": res["mae_dG_ox"],
            "MAE_dG_total": res["mae_dG_total"],
            "sum_mae_8": res["sum_mae_8"],
        })

    pd.DataFrame(rows).to_csv(ROOT / "results" / "step9_replace_mu.csv", index=False)
    print("\n[saved] results/step9_replace_mu.csv")


if __name__ == "__main__":
    main()
