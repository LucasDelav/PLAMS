#!/usr/bin/env python3
"""Phase 3: search for the best joint (red, ox) form per pair on Acetonitrile.

For each of 4 pairs (EA/IP, Erelax_red/ox, ddU_red/ox, TdS_red/ox):
  - Iterate over the form library (polynomial subsets up to degree 3, +Padé).
  - Fit two coefficient vectors (theta_red, theta_ox) sharing the same form.
  - Score by 5-fold GroupKFold CV (groups = molecule) -> avoid leakage.
  - Aggregate MAE and BIC; rank.

Outputs:
  results/<pair>_top.csv : top forms ranked by joint MAE
  results/all_results.parquet : every form evaluated (all 4 pairs)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

N_WORKERS = int(os.environ.get("FORMULA_SEARCH_WORKERS", "20"))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))
from forms import (  # noqa: E402
    PadeForm, PolynomialForm,
    enumerate_pade_forms, enumerate_polynomial_forms,
)

DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
RESULTS = REPO / "analysis" / "formula_search" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DESCRIPTORS = [
    "mu", "eta", "gamma",
    "fplus_max", "fplus_max_O",
    "fminus_max", "fminus_max_O",
    "f2_max", "f2_max_O",
    "qbader_max", "qbader_max_O",
    "lr_diag_max",
    "lvc2_red", "lvc2_ox",
]
PAIRS = [("EA", "IP"), ("Erelax_red", "Erelax_ox"),
         ("ddU_red", "ddU_ox"), ("TdS_red", "TdS_ox")]
PAIR_BY_NAME = {f"{a}_{b}": (a, b) for a, b in PAIRS}
N_FOLDS = 5
SOLVENT_PHASE1 = "Acetonitrile"


@dataclass
class Result:
    pair: str
    form_kind: str          # "poly" | "pade"
    k: int
    n_params: int
    descriptors: str        # comma-separated names
    label: str              # human-readable formula
    mae_red_cv: float
    mae_ox_cv: float
    mae_total: float
    mae_total_norm: float   # MAE_red/std_red + MAE_ox/std_ox
    r2_red_cv: float
    r2_ox_cv: float
    rmse_red_cv: float
    rmse_ox_cv: float
    bic_total: float


def cv_score_polynomial(form: PolynomialForm, X: np.ndarray,
                        y_red: np.ndarray, y_ox: np.ndarray,
                        groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold predictions for red and ox."""
    F = form.design(X)
    n = X.shape[0]
    pred_r = np.full(n, np.nan)
    pred_o = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(F, y_red, groups):
        m_r = LinearRegression(fit_intercept=False).fit(F[tr], y_red[tr])
        m_o = LinearRegression(fit_intercept=False).fit(F[tr], y_ox[tr])
        pred_r[te] = m_r.predict(F[te])
        pred_o[te] = m_o.predict(F[te])
    return pred_r, pred_o


def cv_score_pade(form: PadeForm, X: np.ndarray,
                  y_red: np.ndarray, y_ox: np.ndarray,
                  groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    fn = form.model_fn()
    p0 = np.zeros(form.n_params)
    p0[0] = float(np.mean(y_red))
    pred_r = np.full(n, np.nan)
    pred_o = np.full(n, np.nan)
    X_sub = X[:, list(form.descriptor_idx)]
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X_sub, y_red, groups):
        x_tr = X_sub[tr].reshape(-1)
        x_te = X_sub[te].reshape(-1)
        try:
            popt_r, _ = curve_fit(fn, x_tr, y_red[tr], p0=p0, maxfev=3000)
            popt_o, _ = curve_fit(fn, x_tr, y_ox[tr], p0=p0, maxfev=3000)
            pred_r[te] = fn(x_te, *popt_r)
            pred_o[te] = fn(x_te, *popt_o)
        except Exception:
            return pred_r, pred_o  # NaNs propagate -> skipped downstream
    return pred_r, pred_o


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


_WORKER_STATE: dict = {}


def _worker_init(X, y_r, y_o, groups, std_r, std_o, pair_label):
    _WORKER_STATE.update({
        "X": X, "y_r": y_r, "y_o": y_o, "groups": groups,
        "std_r": std_r, "std_o": std_o, "pair_label": pair_label,
    })


def _eval_one(form_and_kind):
    form, kind = form_and_kind
    s = _WORKER_STATE
    if kind == "poly":
        pred_r, pred_o = cv_score_polynomial(form, s["X"], s["y_r"], s["y_o"], s["groups"])
    else:
        pred_r, pred_o = cv_score_pade(form, s["X"], s["y_r"], s["y_o"], s["groups"])
    mae_r, rmse_r, r2_r = metrics(s["y_r"], pred_r)
    mae_o, rmse_o, r2_o = metrics(s["y_o"], pred_o)
    if not (np.isfinite(mae_r) and np.isfinite(mae_o)):
        return None
    return {
        "pair": s["pair_label"], "form_kind": kind, "k": form.k, "n_params": form.n_params,
        "descriptors": ",".join(DESCRIPTORS[i] for i in form.descriptor_idx),
        "label": form.label(DESCRIPTORS),
        "mae_red_cv": mae_r, "mae_ox_cv": mae_o,
        "mae_total": mae_r + mae_o,
        "mae_total_norm": mae_r / s["std_r"] + mae_o / s["std_o"],
        "r2_red_cv": r2_r, "r2_ox_cv": r2_o,
        "rmse_red_cv": rmse_r, "rmse_ox_cv": rmse_o,
        "bic_total": bic(s["y_r"], pred_r, form.n_params) + bic(s["y_o"], pred_o, form.n_params),
    }


def evaluate_pair(pair: tuple[str, str], df: pd.DataFrame,
                  forms_poly: list[PolynomialForm],
                  forms_pade: list[PadeForm]) -> list[Result]:
    name_red, name_ox = pair
    pair_label = f"{name_red}/{name_ox}"
    X = df[DESCRIPTORS].to_numpy()
    y_r = df[name_red].to_numpy()
    y_o = df[name_ox].to_numpy()
    groups = df["molecule"].to_numpy()
    std_r, std_o = float(y_r.std()), float(y_o.std())

    print(f"\n[{pair_label}] target stats: "
          f"red mean={y_r.mean():.4f} std={std_r:.4f} | "
          f"ox mean={y_o.mean():.4f} std={std_o:.4f}")
    print(f"[{pair_label}] polynomial forms: {len(forms_poly)} | Padé forms: {len(forms_pade)}"
          f" | workers: {N_WORKERS}")

    results: list[Result] = []
    init_args = (X, y_r, y_o, groups, std_r, std_o, pair_label)

    work = [(f, "poly") for f in forms_poly]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_worker_init, initargs=init_args) as pool:
        for i, rec in enumerate(pool.map(_eval_one, work, chunksize=200), 1):
            if rec is not None:
                results.append(Result(**rec))
            if i % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i}/{len(work)} polynomial forms ({elapsed:.1f}s)")
    print(f"  polynomial sweep done: {time.perf_counter()-t0:.1f}s")

    work = [(f, "pade") for f in forms_pade]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_worker_init, initargs=init_args) as pool:
        for rec in pool.map(_eval_one, work, chunksize=10):
            if rec is not None:
                results.append(Result(**rec))
    print(f"  Padé sweep done: {time.perf_counter()-t0:.1f}s")
    print(f"[{pair_label}] valid results: {len(results)}")
    return results


def main(top_n: int = 30, only_pair: str | None = None) -> None:
    df_full = pd.read_parquet(DATA)
    df = df_full[df_full.solvent == SOLVENT_PHASE1].reset_index(drop=True)
    print(f"Phase 1 dataset ({SOLVENT_PHASE1}): {len(df)} rows, {df.molecule.nunique()} molecules")

    n_d = len(DESCRIPTORS)
    forms_poly = enumerate_polynomial_forms(
        n_d, k_values=(1, 2, 3), max_total_degree=3, max_subset_count=12,
    )
    forms_pade = enumerate_pade_forms(n_d, k_values=(1, 2, 3))
    print(f"Total polynomial forms: {len(forms_poly):,}")
    print(f"Total Padé forms: {len(forms_pade):,}")

    pairs_to_run = PAIRS
    if only_pair is not None:
        if only_pair not in PAIR_BY_NAME:
            raise SystemExit(f"--pair must be one of {list(PAIR_BY_NAME)}")
        pairs_to_run = [PAIR_BY_NAME[only_pair]]

    all_rows: list[Result] = []
    for pair in pairs_to_run:
        rows = evaluate_pair(pair, df, forms_poly, forms_pade)
        all_rows.extend(rows)

        df_pair = pd.DataFrame([r.__dict__ for r in rows])
        df_pair = df_pair.sort_values("mae_total").reset_index(drop=True)
        out = RESULTS / f"{pair[0]}_{pair[1]}_top{top_n}.csv"
        df_pair.head(top_n).to_csv(out, index=False, float_format="%.6g")
        print(f"  -> top {top_n} written to {out.name}")

    df_all = pd.DataFrame([r.__dict__ for r in all_rows])
    suffix = f"_{only_pair}" if only_pair else ""
    out_pq = RESULTS / f"all_results{suffix}.parquet"
    df_all.to_parquet(out_pq, index=False)
    print(f"\nAll {len(df_all):,} evaluated forms written to {out_pq}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None,
                    help=f"restrict to one pair: {list(PAIR_BY_NAME)}")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()
    main(top_n=args.top_n, only_pair=args.pair)
