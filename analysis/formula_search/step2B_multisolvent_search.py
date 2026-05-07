#!/usr/bin/env python3
"""Étape 2B — recherche exhaustive de forme f directement sur multi-solvant.

Au lieu de reprendre la forme f figée à l'étape 1 (Acetonitrile-optimal),
on relance une recherche exhaustive polynômes sur les sous-pools validés,
mais en multi-solvant et avec design = monômes × {1, 1/ε} (architecture 2A).

Pour chaque paire :
    y = Σ_k [α_k + β_k/ε] · m_k(d)

où la combinaison de monômes m_k est cherchée exhaustivement.

Score = MAE_total kJ/mol en 5-fold GroupKFold (groupes = molécule).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forms import enumerate_polynomial_forms, PolynomialForm

DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_WORKERS = int(os.environ.get("FORMULA_SEARCH_WORKERS", "8"))
N_FOLDS = 5
EH_TO_KJ = 2625.499638

EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.40,
}

PAIRED_MAP = {
    "lvc2_paired":        ("lvc2_red",     "lvc2_ox"),
    "fukui_paired_max":   ("fplus_max",    "fminus_max"),
    "fukui_paired_max_O": ("fplus_max_O",  "fminus_max_O"),
}

SUBPOOLS: dict[str, list[str]] = {
    "EA_IP":   ["mu", "eta", "fukui_paired_max", "gamma"],
    "Erelax":  ["fukui_paired_max", "qbader_max", "qbader_max_O", "f2_max", "lvc2_paired"],
    "ddU":     ["mu", "eta", "fukui_paired_max", "gamma"],
    "TdS":     ["mu", "eta", "fukui_paired_max", "f2_max", "gamma"],
}

PAIRS: dict[str, tuple[str, str]] = {
    "EA_IP":  ("EA", "IP"),
    "Erelax": ("Erelax_red", "Erelax_ox"),
    "ddU":    ("ddU_red", "ddU_ox"),
    "TdS":    ("TdS_red", "TdS_ox"),
}

# ε-feature set (best from étape 2A)
EPS_FEATURES = ["const", "inv_eps"]


def resolve_pool(df, pool, side):
    cols = []
    for name in pool:
        if name in PAIRED_MAP:
            rc, oc = PAIRED_MAP[name]
            cols.append(df[rc if side == "red" else oc].to_numpy())
        else:
            cols.append(df[name].to_numpy())
    return np.column_stack(cols)


def expand_with_eps(M, eps, features):
    feat_vals = []
    for f in features:
        if f == "const":
            feat_vals.append(np.ones_like(eps))
        elif f == "inv_eps":
            feat_vals.append(1.0 / eps)
        elif f == "log_eps":
            feat_vals.append(np.log(eps))
        elif f == "eps":
            feat_vals.append(eps)
        else:
            raise ValueError(f"unknown feat: {f}")
    return np.column_stack([M * fv[:, None] for fv in feat_vals])


@dataclass
class Result:
    pair: str
    k: int
    n_monomials: int
    n_par_total: int
    descriptors: str
    label: str
    mae_red_cv_kj: float
    mae_ox_cv_kj: float
    mae_total_kj: float
    r2_red_cv: float
    r2_ox_cv: float
    bic_total: float


def metrics_kj(y_ha, yhat_ha):
    mask = np.isfinite(yhat_ha)
    y = y_ha[mask] * EH_TO_KJ
    yhat = yhat_ha[mask] * EH_TO_KJ
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, r2


def bic(y_ha, yhat_ha, n_params):
    mask = np.isfinite(yhat_ha)
    y = y_ha[mask] * EH_TO_KJ
    yhat = yhat_ha[mask] * EH_TO_KJ
    n = len(y)
    if n == 0:
        return float("inf")
    rss = float(np.sum((y - yhat) ** 2))
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + n_params * np.log(n)


_WS: dict = {}


def _init(pair, pool, X_r, X_o, eps, y_r, y_o, groups):
    _WS.update({"pair": pair, "pool": pool, "X_r": X_r, "X_o": X_o,
                "eps": eps, "y_r": y_r, "y_o": y_o, "groups": groups})


def _eval(form: PolynomialForm) -> dict | None:
    s = _WS
    M_r = form.design(s["X_r"])
    M_o = form.design(s["X_o"])
    F_r = expand_with_eps(M_r, s["eps"], EPS_FEATURES)
    F_o = expand_with_eps(M_o, s["eps"], EPS_FEATURES)
    n = F_r.shape[0]
    pr = np.full(n, np.nan)
    po = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(F_r, s["y_r"], s["groups"]):
        m_r = LinearRegression(fit_intercept=False).fit(F_r[tr], s["y_r"][tr])
        m_o = LinearRegression(fit_intercept=False).fit(F_o[tr], s["y_o"][tr])
        pr[te] = m_r.predict(F_r[te])
        po[te] = m_o.predict(F_o[te])
    mae_r, r2_r = metrics_kj(s["y_r"], pr)
    mae_o, r2_o = metrics_kj(s["y_o"], po)
    if not (np.isfinite(mae_r) and np.isfinite(mae_o)):
        return None
    n_par_signal = F_r.shape[1]
    return asdict(Result(
        pair=s["pair"], k=form.k, n_monomials=form.n_params,
        n_par_total=2 * n_par_signal,
        descriptors=",".join(s["pool"][i] for i in form.descriptor_idx),
        label=form.label(s["pool"]),
        mae_red_cv_kj=mae_r, mae_ox_cv_kj=mae_o,
        mae_total_kj=mae_r + mae_o,
        r2_red_cv=r2_r, r2_ox_cv=r2_o,
        bic_total=bic(s["y_r"], pr, n_par_signal) + bic(s["y_o"], po, n_par_signal),
    ))


def run_pair(pair_name, top_n=30):
    df = pd.read_parquet(DATA).sort_values(
        ["solvent", "molecule", "conformer"]).reset_index(drop=True)
    df["eps"] = df["solvent"].map(EPSILON_R)
    pool = SUBPOOLS[pair_name]
    name_red, name_ox = PAIRS[pair_name]

    X_r = resolve_pool(df, pool, "red")
    X_o = resolve_pool(df, pool, "ox")
    y_r = df[name_red].to_numpy()
    y_o = df[name_ox].to_numpy()
    groups = df["molecule"].to_numpy()
    eps = df["eps"].to_numpy()

    print(f"\n=== {pair_name} ===  pool={pool}  rows={len(df)}")
    print(f"   ε-features={EPS_FEATURES}  (× {len(EPS_FEATURES)} per monôme)")

    n_d = len(pool)
    k_vals = tuple(range(1, n_d + 1))
    forms = enumerate_polynomial_forms(
        n_d, k_values=k_vals, max_total_degree=3, max_subset_count=15,
    )
    print(f"   polynomial forms: {len(forms):,}  (workers={N_WORKERS})")

    init_args = (pair_name, pool, X_r, X_o, eps, y_r, y_o, groups)
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init, initargs=init_args) as pool_x:
        for rec in pool_x.map(_eval, forms, chunksize=200):
            if rec is not None:
                results.append(rec)
    print(f"   poly sweep: {time.perf_counter()-t0:.1f}s | valid: {len(results):,}")

    df_res = pd.DataFrame(results).sort_values("mae_total_kj").reset_index(drop=True)
    out = RESULTS / f"step2B_top{top_n}_{pair_name}.csv"
    df_res.head(top_n).to_csv(out, index=False, float_format="%.4g")
    print(f"   -> {out.name}")
    return df_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None, choices=list(PAIRS) + [None])
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()
    pairs = [args.pair] if args.pair else list(PAIRS)
    all_rows = []
    for p in pairs:
        df_res = run_pair(p, top_n=args.top_n)
        all_rows.append(df_res)
    if len(all_rows) > 1:
        full = pd.concat(all_rows, ignore_index=True)
        full.to_parquet(RESULTS / "step2B_all_results.parquet", index=False)
        print(f"\n[saved] step2B_all_results.parquet ({len(full):,} rows)")


if __name__ == "__main__":
    main()
