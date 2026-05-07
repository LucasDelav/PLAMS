#!/usr/bin/env python3
"""Étape 1.3 — recherche exhaustive polynômes/Padé sur sous-pool, par paire.

Sur Acetonitrile uniquement, pour chaque paire (red, ox) :
  - le pool est restreint au sous-pool validé en étape 1.2
  - les descripteurs paired (lvc2_paired, fukui_paired_max, fukui_paired_max_O)
    sont résolus dynamiquement : red ↔ {lvc2_red, fplus_max, fplus_max_O},
                                   ox  ↔ {lvc2_ox,  fminus_max, fminus_max_O}
  - la forme (jeu de monômes ou Padé) est partagée red/ox, coefficients distincts
  - score = MAE_red + MAE_ox en 5-fold GroupKFold (groups=molecule)

Sortie : results/step1_top<N>_<pair>.csv et results/step1_final_form_<pair>.json
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
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forms import PadeForm, PolynomialForm, enumerate_pade_forms, enumerate_polynomial_forms

DATA = ROOT / "data" / "merged_dataset.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_WORKERS = int(os.environ.get("FORMULA_SEARCH_WORKERS", "20"))
N_FOLDS = 5
SOLVENT = "Acetonitrile"

PAIRED_MAP = {
    "lvc2_paired":        ("lvc2_red",     "lvc2_ox"),
    "fukui_paired_max":   ("fplus_max",    "fminus_max"),
    "fukui_paired_max_O": ("fplus_max_O",  "fminus_max_O"),
}

# Sub-pools validated in étape 1.2
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


def resolve_pool(df: pd.DataFrame, pool: list[str], side: str) -> np.ndarray:
    """Build a (n, k) matrix for `pool` resolving paired descriptors to red or ox columns."""
    cols = []
    for name in pool:
        if name in PAIRED_MAP:
            red_col, ox_col = PAIRED_MAP[name]
            cols.append(df[red_col if side == "red" else ox_col].to_numpy())
        else:
            cols.append(df[name].to_numpy())
    return np.column_stack(cols)


@dataclass
class Result:
    pair: str
    form_kind: str
    k: int
    n_params: int
    descriptors: str
    label_red: str
    label_ox: str
    mae_red_cv: float
    mae_ox_cv: float
    mae_total: float
    mae_total_norm: float
    r2_red_cv: float
    r2_ox_cv: float
    rmse_red_cv: float
    rmse_ox_cv: float
    bic_total: float


def metrics(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def bic(y: np.ndarray, yhat: np.ndarray, n_params: int) -> float:
    mask = np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    n = len(y)
    if n == 0:
        return float("inf")
    rss = float(np.sum((y - yhat) ** 2))
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + n_params * np.log(n)


def cv_polynomial(form: PolynomialForm,
                  X_r: np.ndarray, X_o: np.ndarray,
                  y_r: np.ndarray, y_o: np.ndarray,
                  groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two independent linear fits (same monomials), red on X_r, ox on X_o."""
    F_r = form.design(X_r)
    F_o = form.design(X_o)
    n = X_r.shape[0]
    pr = np.full(n, np.nan)
    po = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(F_r, y_r, groups):
        m_r = LinearRegression(fit_intercept=False).fit(F_r[tr], y_r[tr])
        m_o = LinearRegression(fit_intercept=False).fit(F_o[tr], y_o[tr])
        pr[te] = m_r.predict(F_r[te])
        po[te] = m_o.predict(F_o[te])
    return pr, po


def cv_pade(form: PadeForm,
            X_r: np.ndarray, X_o: np.ndarray,
            y_r: np.ndarray, y_o: np.ndarray,
            groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X_r.shape[0]
    fn = form.model_fn()
    p0 = np.zeros(form.n_params)
    p0[0] = float(np.mean(y_r))
    pr = np.full(n, np.nan)
    po = np.full(n, np.nan)
    Xs_r = X_r[:, list(form.descriptor_idx)]
    Xs_o = X_o[:, list(form.descriptor_idx)]
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(Xs_r, y_r, groups):
        x_tr_r = Xs_r[tr].reshape(-1)
        x_te_r = Xs_r[te].reshape(-1)
        x_tr_o = Xs_o[tr].reshape(-1)
        x_te_o = Xs_o[te].reshape(-1)
        try:
            popt_r, _ = curve_fit(fn, x_tr_r, y_r[tr], p0=p0, maxfev=3000)
            popt_o, _ = curve_fit(fn, x_tr_o, y_o[tr], p0=p0, maxfev=3000)
            pr[te] = fn(x_te_r, *popt_r)
            po[te] = fn(x_te_o, *popt_o)
        except Exception:
            return pr, po  # NaNs -> skipped
    return pr, po


_WS: dict = {}


def _worker_init(pair_name, pool, X_r, X_o, y_r, y_o, groups, std_r, std_o):
    _WS.update({
        "pair": pair_name, "pool": pool,
        "X_r": X_r, "X_o": X_o, "y_r": y_r, "y_o": y_o,
        "groups": groups, "std_r": std_r, "std_o": std_o,
    })


def _eval(form_kind: tuple) -> dict | None:
    form, kind = form_kind
    s = _WS
    if kind == "poly":
        pr, po = cv_polynomial(form, s["X_r"], s["X_o"], s["y_r"], s["y_o"], s["groups"])
    else:
        pr, po = cv_pade(form, s["X_r"], s["X_o"], s["y_r"], s["y_o"], s["groups"])
    mae_r, rmse_r, r2_r = metrics(s["y_r"], pr)
    mae_o, rmse_o, r2_o = metrics(s["y_o"], po)
    if not (np.isfinite(mae_r) and np.isfinite(mae_o)):
        return None
    pool = s["pool"]
    desc_names = [pool[i] for i in form.descriptor_idx]
    return asdict(Result(
        pair=s["pair"], form_kind=kind, k=form.k, n_params=form.n_params,
        descriptors=",".join(desc_names),
        label_red=form.label(pool), label_ox=form.label(pool),
        mae_red_cv=mae_r, mae_ox_cv=mae_o,
        mae_total=mae_r + mae_o,
        mae_total_norm=mae_r / s["std_r"] + mae_o / s["std_o"],
        r2_red_cv=r2_r, r2_ox_cv=r2_o,
        rmse_red_cv=rmse_r, rmse_ox_cv=rmse_o,
        bic_total=bic(s["y_r"], pr, form.n_params) + bic(s["y_o"], po, form.n_params),
    ))


def run_pair(pair_name: str, top_n: int = 30) -> pd.DataFrame:
    df_full = pd.read_parquet(DATA)
    df = df_full[df_full.solvent == SOLVENT].reset_index(drop=True)
    pool = SUBPOOLS[pair_name]
    name_red, name_ox = PAIRS[pair_name]

    X_r = resolve_pool(df, pool, "red")
    X_o = resolve_pool(df, pool, "ox")
    y_r = df[name_red].to_numpy()
    y_o = df[name_ox].to_numpy()
    groups = df["molecule"].to_numpy()
    std_r, std_o = float(y_r.std()), float(y_o.std())

    print(f"\n=== {pair_name}  ({name_red}/{name_ox}) ===")
    print(f"pool ({len(pool)}): {pool}")
    print(f"rows: {len(df)} | molecules: {df.molecule.nunique()}")
    print(f"target std: red={std_r:.4f} | ox={std_o:.4f}")

    n_d = len(pool)
    k_vals = tuple(range(1, n_d + 1))
    forms_poly = enumerate_polynomial_forms(
        n_d, k_values=k_vals, max_total_degree=3, max_subset_count=15,
    )
    forms_pade = enumerate_pade_forms(n_d, k_values=k_vals)
    print(f"polynomial forms: {len(forms_poly):,} | Padé forms: {len(forms_pade):,} | workers: {N_WORKERS}")

    init_args = (pair_name, pool, X_r, X_o, y_r, y_o, groups, std_r, std_o)
    results = []

    t0 = time.perf_counter()
    work = [(f, "poly") for f in forms_poly]
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_worker_init, initargs=init_args) as pool_x:
        for rec in pool_x.map(_eval, work, chunksize=200):
            if rec is not None:
                results.append(rec)
    print(f"  poly sweep: {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    work = [(f, "pade") for f in forms_pade]
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_worker_init, initargs=init_args) as pool_x:
        for rec in pool_x.map(_eval, work, chunksize=10):
            if rec is not None:
                results.append(rec)
    print(f"  Padé sweep: {time.perf_counter()-t0:.1f}s")
    print(f"  valid results: {len(results)}")

    df_res = pd.DataFrame(results).sort_values("mae_total").reset_index(drop=True)
    out = RESULTS / f"step1_top{top_n}_{pair_name}.csv"
    df_res.head(top_n).to_csv(out, index=False, float_format="%.6g")
    print(f"  -> {out.name}")
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
        all_rows.append(df_res.assign(pair=p))
    if len(all_rows) > 1:
        full = pd.concat(all_rows, ignore_index=True)
        full.to_parquet(RESULTS / "step1_all_results.parquet", index=False)
        print(f"\n[saved] step1_all_results.parquet ({len(full):,} rows)")


if __name__ == "__main__":
    main()
