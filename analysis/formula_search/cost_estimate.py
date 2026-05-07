#!/usr/bin/env python3
"""Estimate cost of exhaustive vs Optuna-guided formula search.

Produces a table comparing:
  (A) exhaustive: every subset of size k in {1,2,3} from 12 descriptors,
      crossed with form types {linear, poly2, poly3, pade1, pade2}
  (C) Optuna-guided: ~N trials sampling k, descriptors, form, monomial subset

Wall time per fit is benchmarked on real Acetonitrile data, then extrapolated.
"""
from __future__ import annotations

import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"

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

N_FOLDS = 5
N_DESC = len(DESCRIPTORS)


def n_monomials(k: int, total_degree: int) -> int:
    """Number of monomials in k variables with total degree <= total_degree (excluding the constant)."""
    return math.comb(k + total_degree, k) - 1


def n_pade_params(k: int, num_deg: int = 1, den_deg: int = 1) -> int:
    """Number of free parameters in a Padé approximant: numerator + denom (minus 1 for normalization)."""
    return n_monomials(k, num_deg) + 1 + n_monomials(k, den_deg)


def make_poly_features(X: np.ndarray, total_degree: int) -> np.ndarray:
    """Return [1, x_i, x_i x_j (i<=j), ...] up to total_degree (no normalization).

    Returns the pure polynomial feature matrix (constant included as first column)."""
    n, k = X.shape
    feats = [np.ones(n)]
    if total_degree >= 1:
        for i in range(k):
            feats.append(X[:, i])
    if total_degree >= 2:
        for i in range(k):
            for j in range(i, k):
                feats.append(X[:, i] * X[:, j])
    if total_degree >= 3:
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    feats.append(X[:, i] * X[:, j] * X[:, l])
    return np.column_stack(feats)


def fit_linear_form(X: np.ndarray, y_red: np.ndarray, y_ox: np.ndarray,
                    groups: np.ndarray, total_degree: int) -> tuple[float, float]:
    """K-fold CV. Returns (mae_red_cv, mae_ox_cv) in same units as targets."""
    F = make_poly_features(X, total_degree)
    gkf = GroupKFold(n_splits=N_FOLDS)
    mae_r, mae_o = [], []
    for tr, te in gkf.split(F, y_red, groups):
        m_r = LinearRegression(fit_intercept=False).fit(F[tr], y_red[tr])
        m_o = LinearRegression(fit_intercept=False).fit(F[tr], y_ox[tr])
        mae_r.append(np.mean(np.abs(m_r.predict(F[te]) - y_red[te])))
        mae_o.append(np.mean(np.abs(m_o.predict(F[te]) - y_ox[te])))
    return float(np.mean(mae_r)), float(np.mean(mae_o))


def fit_pade_form(X: np.ndarray, y_red: np.ndarray, y_ox: np.ndarray,
                  groups: np.ndarray) -> tuple[float, float]:
    """Padé degree (1,1) per descriptor: y = (a0 + sum a_i d_i) / (1 + sum b_i d_i)."""
    n, k = X.shape

    def pade(X_flat, *theta):
        Xa = X_flat.reshape(-1, k)
        a = theta[: 1 + k]
        b = theta[1 + k:]
        num = a[0] + Xa @ np.asarray(a[1:])
        den = 1.0 + Xa @ np.asarray(b)
        return num / den

    p0 = np.zeros(1 + 2 * k)
    p0[0] = 0.1
    gkf = GroupKFold(n_splits=N_FOLDS)
    mae_r, mae_o = [], []
    for tr, te in gkf.split(X, y_red, groups):
        Xtr = X[tr].reshape(-1)
        Xte = X[te].reshape(-1)
        try:
            popt_r, _ = curve_fit(pade, Xtr, y_red[tr], p0=p0, maxfev=2000)
            popt_o, _ = curve_fit(pade, Xtr, y_ox[tr], p0=p0, maxfev=2000)
            mae_r.append(np.mean(np.abs(pade(Xte, *popt_r) - y_red[te])))
            mae_o.append(np.mean(np.abs(pade(Xte, *popt_o) - y_ox[te])))
        except Exception:
            return float("nan"), float("nan")
    return float(np.mean(mae_r)), float(np.mean(mae_o))


def benchmark(df: pd.DataFrame) -> dict:
    """Benchmark single-fit time for each form type."""
    df_a = df[df.solvent == "Acetonitrile"].reset_index(drop=True)
    groups = df_a["molecule"].values
    times = {}

    for k in (1, 2, 3):
        sub = DESCRIPTORS[:k]
        X = df_a[sub].values
        y_r = df_a["EA"].values
        y_o = df_a["IP"].values

        for deg in (1, 2, 3):
            label = f"poly_k{k}_deg{deg}"
            t0 = time.perf_counter()
            for _ in range(20):
                fit_linear_form(X, y_r, y_o, groups, deg)
            times[label] = (time.perf_counter() - t0) / 20

        if k <= 2:
            label = f"pade_k{k}"
            t0 = time.perf_counter()
            for _ in range(20):
                fit_pade_form(X, y_r, y_o, groups)
            times[label] = (time.perf_counter() - t0) / 20

    return times


def count_exhaustive() -> dict:
    """Number of (subset, form) combinations to fit exhaustively (per pair)."""
    counts = {}
    total = 0
    for k in (1, 2, 3):
        n_sub = math.comb(N_DESC, k)
        for deg in (1, 2, 3):
            n_p = n_monomials(k, deg)
            counts[f"poly_k{k}_deg{deg}"] = (n_sub, n_p)
            total += n_sub
        if k <= 2:
            n_p = n_pade_params(k)
            counts[f"pade_k{k}"] = (n_sub, n_p)
            total += n_sub
    counts["TOTAL_per_pair"] = total
    return counts


def main() -> None:
    df = pd.read_parquet(DATA)
    print(f"Dataset: {len(df)} rows ({df.molecule.nunique()} mols, {df.solvent.nunique()} solvs)")
    print(f"Phase 1 (Acetonitrile): {len(df[df.solvent=='Acetonitrile'])} rows\n")

    print("=" * 72)
    print("Form space per pair (subsets × form types)")
    print("=" * 72)
    counts = count_exhaustive()
    for k, v in counts.items():
        if k.startswith("TOTAL"):
            continue
        n_sub, n_p = v
        print(f"  {k:18s}  subsets={n_sub:4d}  free_params={n_p}")
    total = counts["TOTAL_per_pair"]
    print(f"  TOTAL per pair: {total} formulas (×4 pairs = {4*total} formula evaluations)")

    print("\n" + "=" * 72)
    print("Benchmarking single-fit time on Acetonitrile (5-fold CV, joint red+ox)")
    print("=" * 72)
    times = benchmark(df)
    for k, t in times.items():
        print(f"  {k:18s}  {t*1000:7.2f} ms/fit")

    print("\n" + "=" * 72)
    print("(A) EXHAUSTIVE — Acetonitrile only, 4 pairs")
    print("=" * 72)
    total_a = 0.0
    for k in (1, 2, 3):
        n_sub = math.comb(N_DESC, k)
        for deg in (1, 2, 3):
            label = f"poly_k{k}_deg{deg}"
            total_a += n_sub * times[label]
        if k <= 2:
            label = f"pade_k{k}"
            total_a += n_sub * times[label]
    total_a *= 4
    print(f"  Wall time (A): {total_a:.1f} s = {total_a/60:.2f} min")

    print("\n" + "=" * 72)
    print("(C) OPTUNA-guided — Acetonitrile only, 4 pairs")
    print("=" * 72)
    avg_t = np.mean(list(times.values()))
    for n_trials in (500, 1000, 5000):
        wall = 4 * n_trials * avg_t
        print(f"  {n_trials:5d} trials/pair  ->  {wall:7.1f} s = {wall/60:6.2f} min")

    print("\n" + "=" * 72)
    print("Recommendation")
    print("=" * 72)
    if total_a < 600:
        print(f"  Exhaustive (A) takes {total_a/60:.1f} min — small enough to run in full.")
        print("  -> recommend (A): provides total coverage and a complete ranking.")
    else:
        print(f"  Exhaustive (A) takes {total_a/60:.1f} min — non-trivial but tractable.")
        print("  -> still recommend (A); Optuna only needed if we extend to higher degrees.")


if __name__ == "__main__":
    main()
