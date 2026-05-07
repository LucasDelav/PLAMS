#!/usr/bin/env python3
"""Phase 4: take top-N forms per pair from Phase 3 and add a g(eps_r) correction.

Workflow:
  1. Load top-K (default 3) forms per pair from Phase 3 results.
  2. For each form, refit on the full multi-solvent dataset (1604 rows) WITHOUT
     any solvent term -- this is the baseline.
  3. Then refit y = f(d; theta) + g(eps_r; phi) jointly, where g is one of:
        Born     :  alpha * (eps_r - 1) / (2*eps_r + 1)
        Onsager  :  alpha * (eps_r - 1) / (eps_r + 2)
        Log      :  alpha * log(eps_r)
        Pade1    :  (alpha + beta * eps_r) / (1 + gamma * eps_r)
  4. Score baseline vs corrected by GroupKFold CV on molecule (5-fold) on all
     10 solvents. Report MAE per pair, per solvent, with/without correction.

Outputs:
  results/phase4_baseline.csv   : per-pair, per-form baseline MAE
  results/phase4_corrected.csv  : per-pair, per-form, per-correction-family MAE
  results/phase4_summary.csv    : best correction family per (pair, form)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))
from forms import (  # noqa: E402
    Monomial, PadeForm, PolynomialForm,
    all_monomials,
)

DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
RESULTS = REPO / "analysis" / "formula_search" / "results"

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
TOP_K_PER_PAIR = 3

# Static dielectric constants at 298 K (CRC Handbook values).
EPSILON_R = {
    "Hexane": 1.88,
    "Toluene": 2.38,
    "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58,
    "Dichloromethane": 8.93,
    "Acetone": 20.7,
    "Methanol": 32.6,
    "Acetonitrile": 35.7,
    "DMSO": 46.7,
    "Water": 78.4,
}


# --- correction families ---

def g_born(eps: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * (eps - 1.0) / (2.0 * eps + 1.0)


def g_onsager(eps: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * (eps - 1.0) / (eps + 2.0)


def g_log(eps: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * np.log(eps)


def g_pade(eps: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    return (alpha + beta * eps) / (1.0 + gamma * eps)


CORRECTIONS = {
    "none": (lambda eps: np.zeros_like(eps), 0),
    "born": (g_born, 1),
    "onsager": (g_onsager, 1),
    "log": (g_log, 1),
    "pade": (g_pade, 3),
}


# --- form parsing from Phase 3 CSV ---

def parse_polynomial_form(label: str, descriptor_str: str) -> PolynomialForm:
    """Reconstruct a PolynomialForm from its CSV label.

    The label is a sum of monomials; descriptor_str is comma-separated names."""
    desc_names = descriptor_str.split(",")
    desc_idx = tuple(DESCRIPTORS.index(n) for n in desc_names)
    monos: list[Monomial] = []
    for term in label.split(" + "):
        term = term.strip()
        if term == "1":
            monos.append(Monomial((0,) * len(desc_idx)))
            continue
        exps = [0] * len(desc_idx)
        for factor in term.split("*"):
            if "^" in factor:
                v, e = factor.split("^")
                exps[desc_names.index(v)] = int(e)
            else:
                exps[desc_names.index(factor)] += 1
        monos.append(Monomial(tuple(exps)))
    return PolynomialForm(desc_idx, tuple(monos))


# --- joint fit (form + correction) ---

def fit_polynomial_with_correction(
    form: PolynomialForm,
    X: np.ndarray, y: np.ndarray, eps: np.ndarray,
    correction: str,
):
    """Fit y = F @ theta + g(eps; phi) by alternating LSQ + 1D nonlinear."""
    F = form.design(X)
    g_fn, n_phi = CORRECTIONS[correction]
    if n_phi == 0:
        m = LinearRegression(fit_intercept=False).fit(F, y)
        return m.coef_, np.array([])

    # Initialize: fit theta on raw y first
    theta = LinearRegression(fit_intercept=False).fit(F, y).coef_
    if correction == "pade":
        phi = np.array([0.0, 0.0, 0.01])
    else:
        phi = np.array([0.0])

    for _ in range(50):
        # Update phi given theta
        residual = y - F @ theta
        try:
            if n_phi == 1:
                popt, _ = curve_fit(g_fn, eps, residual, p0=phi, maxfev=2000)
            else:
                popt, _ = curve_fit(g_fn, eps, residual, p0=phi, maxfev=2000)
            phi_new = popt
        except Exception:
            phi_new = phi
        # Update theta given phi
        adj_y = y - g_fn(eps, *phi_new)
        theta_new = LinearRegression(fit_intercept=False).fit(F, adj_y).coef_
        if (np.linalg.norm(theta_new - theta) < 1e-9
                and np.linalg.norm(phi_new - phi) < 1e-9):
            theta, phi = theta_new, phi_new
            break
        theta, phi = theta_new, phi_new
    return theta, phi


def predict_polynomial(form: PolynomialForm, theta: np.ndarray,
                       phi: np.ndarray, X: np.ndarray, eps: np.ndarray,
                       correction: str) -> np.ndarray:
    F = form.design(X)
    g_fn, _ = CORRECTIONS[correction]
    base = F @ theta
    extra = g_fn(eps, *phi) if len(phi) > 0 else 0.0
    return base + extra


def cv_evaluate_polynomial(form: PolynomialForm, df: pd.DataFrame,
                           target_red: str, target_ox: str,
                           correction: str) -> dict:
    X = df[DESCRIPTORS].to_numpy()
    y_r = df[target_red].to_numpy()
    y_o = df[target_ox].to_numpy()
    eps = df["eps_r"].to_numpy()
    groups = df["molecule"].to_numpy()

    n = len(df)
    pred_r = np.full(n, np.nan)
    pred_o = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y_r, groups):
        try:
            theta_r, phi_r = fit_polynomial_with_correction(
                form, X[tr], y_r[tr], eps[tr], correction)
            theta_o, phi_o = fit_polynomial_with_correction(
                form, X[tr], y_o[tr], eps[tr], correction)
            pred_r[te] = predict_polynomial(form, theta_r, phi_r, X[te], eps[te], correction)
            pred_o[te] = predict_polynomial(form, theta_o, phi_o, X[te], eps[te], correction)
        except Exception:
            continue

    mask = np.isfinite(pred_r) & np.isfinite(pred_o)
    err_r = y_r[mask] - pred_r[mask]
    err_o = y_o[mask] - pred_o[mask]
    out = {
        "mae_red": float(np.mean(np.abs(err_r))) if len(err_r) else float("nan"),
        "mae_ox": float(np.mean(np.abs(err_o))) if len(err_o) else float("nan"),
        "rmse_red": float(np.sqrt(np.mean(err_r ** 2))) if len(err_r) else float("nan"),
        "rmse_ox": float(np.sqrt(np.mean(err_o ** 2))) if len(err_o) else float("nan"),
    }
    out["mae_total"] = out["mae_red"] + out["mae_ox"]

    # per-solvent breakdown
    per_solv = {}
    for solv, sub in df.iloc[mask.nonzero()[0]].groupby("solvent"):
        idx = sub.index
        e_r = y_r[idx] - pred_r[idx]
        e_o = y_o[idx] - pred_o[idx]
        per_solv[solv] = (float(np.mean(np.abs(e_r))), float(np.mean(np.abs(e_o))))
    out["per_solvent"] = per_solv
    return out


def main(top_k: int = TOP_K_PER_PAIR) -> None:
    df = pd.read_parquet(DATA)
    df["eps_r"] = df["solvent"].map(EPSILON_R)
    if df["eps_r"].isna().any():
        missing = df.loc[df["eps_r"].isna(), "solvent"].unique()
        raise RuntimeError(f"Missing eps_r for: {missing}")
    print(f"Phase 4 dataset: {len(df)} rows over {df.solvent.nunique()} solvents")

    rows_baseline = []
    rows_corrected = []
    for pair in PAIRS:
        pair_label = f"{pair[0]}/{pair[1]}"
        top_csv = RESULTS / f"{pair[0]}_{pair[1]}_top30.csv"
        if not top_csv.exists():
            print(f"  SKIP {pair_label}: {top_csv.name} not found yet")
            continue
        top = pd.read_csv(top_csv).head(top_k)
        for _, row in top.iterrows():
            if row["form_kind"] != "poly":
                continue  # Padé multi-solvent fit not handled here yet
            try:
                form = parse_polynomial_form(row["label"], row["descriptors"])
            except Exception as e:
                print(f"  parse error for {row['label']!r}: {e}")
                continue

            for corr_name in CORRECTIONS:
                m = cv_evaluate_polynomial(form, df, pair[0], pair[1], corr_name)
                rec = {
                    "pair": pair_label,
                    "form_label": row["label"],
                    "descriptors": row["descriptors"],
                    "n_params": int(row["n_params"]),
                    "correction": corr_name,
                    "mae_red_5sol_cv": m["mae_red"],
                    "mae_ox_5sol_cv": m["mae_ox"],
                    "mae_total_cv": m["mae_total"],
                }
                for solv, (mae_r, mae_o) in m["per_solvent"].items():
                    rec[f"mae_red_{solv}"] = mae_r
                    rec[f"mae_ox_{solv}"] = mae_o
                if corr_name == "none":
                    rows_baseline.append(rec)
                else:
                    rows_corrected.append(rec)
                print(f"  [{pair_label}] {row['label'][:50]:50s} {corr_name:8s}"
                      f" MAE_total={m['mae_total']:.5f}")

    pd.DataFrame(rows_baseline).to_csv(RESULTS / "phase4_baseline.csv",
                                       index=False, float_format="%.6g")
    pd.DataFrame(rows_corrected).to_csv(RESULTS / "phase4_corrected.csv",
                                        index=False, float_format="%.6g")

    # Best correction per (pair, form)
    summary = []
    for pair_form, sub in pd.DataFrame(rows_corrected).groupby(["pair", "form_label"]):
        baseline = next((r for r in rows_baseline
                         if r["pair"] == pair_form[0] and r["form_label"] == pair_form[1]),
                        None)
        if baseline is None:
            continue
        best_idx = sub["mae_total_cv"].idxmin()
        best = sub.loc[best_idx]
        summary.append({
            "pair": pair_form[0],
            "form_label": pair_form[1],
            "descriptors": best["descriptors"],
            "n_params": int(best["n_params"]),
            "best_correction": best["correction"],
            "mae_baseline": baseline["mae_total_cv"],
            "mae_corrected": best["mae_total_cv"],
            "improvement_pct": 100 * (baseline["mae_total_cv"] - best["mae_total_cv"]) / baseline["mae_total_cv"],
        })
    pd.DataFrame(summary).to_csv(RESULTS / "phase4_summary.csv",
                                 index=False, float_format="%.6g")
    print(f"\nWrote {len(summary)} summary rows to phase4_summary.csv")


if __name__ == "__main__":
    main()
