#!/usr/bin/env python3
"""Step 1b — exhaustive brute-force search per target on the reduced pool
(top-K from step1_freq_correlations + the baseline (μ, η, Δη_paired)) on
Acetonitrile only.

Per-target search (independent fits for ddU_red, ddU_ox, TdS_red, TdS_ox):
  - Polynomial forms: subsets of size k=1..5, monomial subsets up to degree 3
    (capped via max_subset_count=12 for tractable enumeration at k≥3)
  - Padé(1,1) forms: subsets of size k=1..5
  - Scoring: 5-fold GroupKFold by molecule, MAE in kJ/mol

Outputs:
    results/step1_freq_top<N>_<target>.csv — top N ranked forms per target
    results/step1_freq_all_<target>.parquet — every evaluated form
"""
from __future__ import annotations

import argparse
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))
from forms import (  # noqa: E402
    PadeForm, PolynomialForm,
    enumerate_pade_forms, enumerate_polynomial_forms,
)

DATA_DIR = REPO / "analysis" / "formula_search" / "data"
RESULTS = REPO / "analysis" / "formula_search" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MERGED = DATA_DIR / "merged_dataset.parquet"
DELTA = DATA_DIR / "ddu_delta_descriptors.parquet"
FREQ = DATA_DIR / "freq_descriptors_full.parquet"

EH_TO_KJ = 2625.499638
SOLVENT = "Acetonitrile"
N_FOLDS = 5
N_WORKERS = int(os.environ.get("FORMULA_SEARCH_WORKERS", str(min(os.cpu_count() or 4, 16))))
TARGETS = ["ddU_red", "ddU_ox", "TdS_red", "TdS_ox"]
DEFAULT_TOP_N = 50


@dataclass
class Result:
    target: str
    form_kind: str
    k: int
    n_params: int
    descriptors: str
    label: str
    mae_cv: float
    rmse_cv: float
    r2_cv: float
    mae_total_norm: float
    bic_total: float


def load_data(targets: list[str]) -> pd.DataFrame:
    merged = pd.read_parquet(MERGED)
    delta = pd.read_parquet(DELTA)
    freq = pd.read_parquet(FREQ)

    df = merged[merged.solvent == SOLVENT].copy()
    for c in targets:
        df[c] = df[c] * EH_TO_KJ

    delta_ace = delta[delta.solvent == SOLVENT][[
        "solvent", "molecule", "conformer", "deta_red", "deta_ox"
    ]]
    df = df.merge(delta_ace, on=["solvent", "molecule", "conformer"], how="inner")

    freq_ace = freq[freq.solvent == SOLVENT]
    df = df.merge(freq_ace, on=["solvent", "molecule", "conformer"], how="inner")
    return df


def load_pool(target: str, top_k: int = 50) -> list[str]:
    """Return the descriptor pool for a target: top-K from correlations
    plus baseline (μ, η, Δη_paired). Reads any matching step1_freq_top*.txt
    and truncates to top_k."""
    candidates = sorted(RESULTS.glob(f"step1_freq_top*_{target}.txt"))
    if not candidates:
        raise FileNotFoundError(f"No step1_freq_top*_{target}.txt — run step1_freq_correlations.py first")
    # Pick the file with the largest K available
    pool = candidates[-1].read_text().strip().splitlines()[:top_k]
    deta_paired = "deta_red" if target.endswith("_red") else "deta_ox"
    for d in ("mu", "eta", deta_paired):
        if d not in pool:
            pool.append(d)
    return list(dict.fromkeys(pool))  # de-dup keeping order


# ---------------------------------------------------------------------------
# CV scoring (single-target)
# ---------------------------------------------------------------------------
_WS: dict = {}


def _init_worker(X, y, groups, std_y, target, descriptor_names):
    _WS.update({"X": X, "y": y, "groups": groups, "std_y": std_y,
                "target": target, "descriptor_names": list(descriptor_names)})


def _cv_poly(form: PolynomialForm) -> tuple[np.ndarray, int]:
    F = form.design(_WS["X"])
    n = F.shape[0]
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(F, _WS["y"], _WS["groups"]):
        m = LinearRegression(fit_intercept=False).fit(F[tr], _WS["y"][tr])
        pred[te] = m.predict(F[te])
    return pred, form.n_params


def _cv_pade(form: PadeForm) -> tuple[np.ndarray, int]:
    fn = form.model_fn()
    X = _WS["X"][:, list(form.descriptor_idx)]
    y = _WS["y"]
    groups = _WS["groups"]
    n = X.shape[0]
    pred = np.full(n, np.nan)
    p0 = np.zeros(form.n_params)
    p0[0] = float(np.mean(y))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        try:
            popt, _ = curve_fit(fn, X[tr].reshape(-1), y[tr], p0=p0, maxfev=3000)
            pred[te] = fn(X[te].reshape(-1), *popt)
        except Exception:
            return pred, form.n_params  # NaNs propagate
    return pred, form.n_params


def _eval_one(form_and_kind):
    form, kind = form_and_kind
    try:
        pred, n_params = _cv_poly(form) if kind == "poly" else _cv_pade(form)
    except Exception:
        return None
    mask = np.isfinite(pred)
    if mask.sum() < 0.9 * len(pred):
        return None
    y, pred = _WS["y"][mask], pred[mask]
    err = y - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = len(y)
    bic_v = n * np.log(ss_res / n) + n_params * np.log(n) if ss_res > 0 else float("inf")
    return {
        "target": _WS["target"], "form_kind": kind,
        "k": form.k, "n_params": n_params,
        "descriptors": ",".join(_WS["descriptor_names"][i] for i in form.descriptor_idx),
        "label": form.label(_WS["descriptor_names"]),
        "mae_cv": mae, "rmse_cv": rmse, "r2_cv": r2,
        "mae_total_norm": mae / _WS["std_y"],
        "bic_total": bic_v,
    }


def search_target(target: str, df: pd.DataFrame, pool: list[str],
                  k_values=(1, 2, 3, 4, 5), top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    X = df[pool].to_numpy()
    y = df[target].to_numpy()
    groups = df["molecule"].to_numpy()
    std_y = float(y.std())

    forms_poly = enumerate_polynomial_forms(
        len(pool), k_values=k_values, max_total_degree=3, max_subset_count=12,
    )
    forms_pade = enumerate_pade_forms(len(pool), k_values=k_values)

    print(f"\n[{target}] pool size: {len(pool)} | poly forms: {len(forms_poly):,} | "
          f"Padé forms: {len(forms_pade):,} | workers: {N_WORKERS}", flush=True)

    init_args = (X, y, groups, std_y, target, list(pool))
    results: list[dict] = []

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker, initargs=init_args) as pool_exec:
        n_done = 0
        for rec in pool_exec.map(_eval_one,
                                  ((f, "poly") for f in forms_poly),
                                  chunksize=400):
            if rec is not None:
                results.append(rec)
            n_done += 1
            if n_done % 50000 == 0:
                print(f"  poly {n_done}/{len(forms_poly):,} ({time.perf_counter()-t0:.1f}s)",
                      flush=True)
    print(f"  poly sweep done in {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker, initargs=init_args) as pool_exec:
        n_done = 0
        for rec in pool_exec.map(_eval_one,
                                  ((f, "pade") for f in forms_pade),
                                  chunksize=20):
            if rec is not None:
                results.append(rec)
            n_done += 1
            if n_done % 5000 == 0:
                print(f"  pade {n_done}/{len(forms_pade):,} ({time.perf_counter()-t0:.1f}s)",
                      flush=True)
    print(f"  Padé sweep done in {time.perf_counter()-t0:.1f}s", flush=True)

    df_res = pd.DataFrame(results).sort_values("mae_cv").reset_index(drop=True)
    print(f"  valid results: {len(df_res):,} | best MAE: {df_res.mae_cv.min():.4f}")
    return df_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None,
                    help=f"restrict to one target: {TARGETS}")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--k-max", type=int, default=5)
    args = ap.parse_args()

    targets = [args.target] if args.target else TARGETS
    if args.target and args.target not in TARGETS:
        raise SystemExit(f"--target must be one of {TARGETS}")

    df = load_data(targets)
    print(f"Acetonitrile dataset: {df.shape}, {df.molecule.nunique()} molecules")

    for tgt in targets:
        pool = load_pool(tgt, top_k=args.top_k)
        print(f"\n=== {tgt} | pool ({len(pool)}): {pool[:6]}...")
        df_res = search_target(tgt, df, pool, k_values=tuple(range(1, args.k_max + 1)))
        out_top = RESULTS / f"step1_freq_top{args.top_n}_{tgt}.csv"
        df_res.head(args.top_n).to_csv(out_top, index=False, float_format="%.6g")
        out_all = RESULTS / f"step1_freq_all_{tgt}.parquet"
        df_res.to_parquet(out_all, index=False)
        print(f"  -> {out_top.name} ({args.top_n} rows)")
        print(f"  -> {out_all.name} ({len(df_res):,} rows)")


if __name__ == "__main__":
    main()
