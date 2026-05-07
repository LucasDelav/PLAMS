#!/usr/bin/env python3
"""Step 2 — Fukui multiplicative correction on the Step-1 retained forms.

For each target, take the top-M forms from Step 1 and refit them with a
multiplicative Fukui correction:

    y_corrected = base_form(X; θ) × (1 + a · F)

where F is one of:
    f+_max, f-_max, f2_max, f2_min, f2_dual_paired, f+_max_noO, etc.

The correction is fit jointly with θ (linear via least squares because the
correction is a simple scalar factor on the design matrix).

Outputs:
    results/step2_fukui_<target>.csv — best form × Fukui combinations
    results/step2_final_form_<target>.json — selected form (lowest CV MAE)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))

DATA_DIR = REPO / "analysis" / "formula_search" / "data"
RESULTS = REPO / "analysis" / "formula_search" / "results"

MERGED = DATA_DIR / "merged_dataset.parquet"
DELTA = DATA_DIR / "ddu_delta_descriptors.parquet"
FREQ = DATA_DIR / "freq_descriptors_full.parquet"

EH_TO_KJ = 2625.499638
SOLVENT = "Acetonitrile"
N_FOLDS = 5
TARGETS = ["ddU_red", "ddU_ox", "TdS_red", "TdS_ox"]

# Fukui correction variants (column names in merged_dataset)
FUKUI_VARIANTS = {
    "none": None,                      # no correction baseline
    "fplus_max": "fplus_max",          # max f+ all atoms
    "fminus_max": "fminus_max",        # max f- all atoms
    "fplus_max_noO": "fplus_max_noO",  # all-atoms minus O
    "fminus_max_noO": "fminus_max_noO",
    "f2_max": "f2_max",                # dual descriptor max
    "f2_max_noO": "f2_max_noO",
    "f2_paired": "f2_paired",          # f2_max for red, f2_min for ox (computed below)
    "f2_paired_noO": "f2_paired_noO",
}

# Targets where the f2_paired equals f2_max (red) vs the absolute |f2| at the
# minimum (ox). The merged dataset has f2_max and f2_max_O — but no f2_min.
# We build f2_min = -f2_max if not provided, but the existing rapport uses
# f2_min via min of f²_neg (from Bader). For now: compute f²_paired manually.


def parse_label_to_design(label: str, descriptor_names: list[str], X: np.ndarray) -> np.ndarray:
    """Build the design matrix from a comma-separated descriptor list and an
    expression label such as '1 + mu + dRr_A_dU_total*mu + mu^3'.
    """
    # Label is space-padded sum of monomials. Parse each monomial.
    name_to_idx = {n: i for i, n in enumerate(descriptor_names)}
    cols: list[np.ndarray] = []
    for term in [t.strip() for t in label.split("+")]:
        if term == "1":
            cols.append(np.ones(X.shape[0]))
            continue
        # Term like "mu^2*deta_red" or "dRr_A_dU_total*mu"
        col = np.ones(X.shape[0])
        for atom in term.split("*"):
            if "^" in atom:
                name, exp = atom.split("^")
                exp = int(exp)
            else:
                name, exp = atom, 1
            if name not in name_to_idx:
                raise ValueError(f"Unknown descriptor in label: {name!r} (label={label!r})")
            col = col * (X[:, name_to_idx[name]] ** exp if exp != 1 else X[:, name_to_idx[name]])
        cols.append(col)
    return np.column_stack(cols)


def cv_fit_with_correction(D: np.ndarray, F: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Fit y = (D @ θ) × (1 + a F) by alternating linear least-squares between
    θ (closed-form for fixed a) and a (1-D scalar minimization).

    Returns out-of-fold predictions and the fitted (θ, a) for the full dataset.
    """
    n = D.shape[0]
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D, y, groups):
        try:
            theta, a = _fit_one(D[tr], F[tr], y[tr])
            pred[te] = (D[te] @ theta) * (1.0 + a * F[te])
        except Exception:
            return pred, np.zeros(D.shape[1]), 0.0
    # Full fit
    theta, a = _fit_one(D, F, y)
    return pred, theta, a


def cv_fit_no_correction(D: np.ndarray, y: np.ndarray, groups: np.ndarray):
    n = D.shape[0]
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D, y, groups):
        m = LinearRegression(fit_intercept=False).fit(D[tr], y[tr])
        pred[te] = m.predict(D[te])
    m = LinearRegression(fit_intercept=False).fit(D, y)
    return pred, m.coef_


def _fit_one(D: np.ndarray, F: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Alternating optimisation: θ from least squares with weighted design,
    then a from 1-D minimisation of residual MSE. Repeated 8 sweeps.
    """
    a = 0.0
    theta = np.linalg.lstsq(D, y, rcond=None)[0]
    for _ in range(8):
        # Fix a, refit θ via weighted lstsq:
        # y = D θ (1 + a F) → y / (1 + a F) = D θ
        denom = 1.0 + a * F
        if np.any(np.abs(denom) < 1e-9):
            break
        theta = np.linalg.lstsq(D / denom[:, None], y / denom, rcond=None)[0]
        # Fix θ, fit a:  y = (Dθ)(1 + aF) ↔ a = ⟨F·(Dθ), y - Dθ⟩ / ⟨F·(Dθ), F·(Dθ)⟩
        Dt = D @ theta
        num = np.sum(F * Dt * (y - Dt))
        den = np.sum((F * Dt) ** 2)
        if den < 1e-12:
            break
        a_new = num / den
        if abs(a_new - a) < 1e-8:
            a = a_new
            break
        a = a_new
    return theta, a


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    mask = np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    if len(y) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_data() -> tuple[pd.DataFrame, list[str]]:
    merged = pd.read_parquet(MERGED)
    delta = pd.read_parquet(DELTA)
    freq = pd.read_parquet(FREQ)

    df = merged[merged.solvent == SOLVENT].copy()
    for c in TARGETS:
        df[c] = df[c] * EH_TO_KJ

    delta_ace = delta[delta.solvent == SOLVENT][[
        "solvent", "molecule", "conformer", "deta_red", "deta_ox", "f2_max", "f2_min"
    ]].rename(columns={"f2_max": "f2_max_delta", "f2_min": "f2_min"})
    df = df.merge(delta_ace, on=["solvent", "molecule", "conformer"], how="inner")

    freq_ace = freq[freq.solvent == SOLVENT]
    df = df.merge(freq_ace, on=["solvent", "molecule", "conformer"], how="inner")

    # Fukui paired columns. Convention from existing baselines:
    #   red → f²_max (most-nucleophilic site dual descriptor)
    #   ox  → f²_min (most-electrophilic site dual descriptor)
    # f2_max is in merged_dataset; f2_min comes from ddu_delta_descriptors.
    df["f2_paired_red"] = df["f2_max"]
    df["f2_paired_ox"] = df["f2_min"]
    # noO variants: use f2_max_O (oxygen-excluded). f2_min_noO is unavailable in
    # the merged set, so we mirror with -f2_max_O as a proxy.
    df["f2_paired_red_noO"] = df["f2_max_O"]
    df["f2_paired_ox_noO"] = -df["f2_max_O"]
    df["fplus_max_noO"] = df["fplus_max_O"] if "fplus_max_O" in df.columns else df["fplus_max"]
    df["fminus_max_noO"] = df["fminus_max_O"] if "fminus_max_O" in df.columns else df["fminus_max"]
    df["f2_max_noO_col"] = df["f2_max_O"] if "f2_max_O" in df.columns else df["f2_max"]

    # Build the descriptor name list aligned with X column order
    return df, ["solvent", "molecule", "conformer"]


def run_target(target: str, df: pd.DataFrame, top_m: int, top_n: int) -> pd.DataFrame:
    print(f"\n=== {target} ===")
    csv = RESULTS / f"step1_freq_top50_{target}.csv"  # may not exist yet
    candidates = sorted(RESULTS.glob(f"step1_freq_top*_{target}.csv"))
    if not candidates:
        raise FileNotFoundError(f"No step1_freq_top*_{target}.csv — run Step 1 first")
    s1 = pd.read_csv(candidates[-1]).head(top_m)
    print(f"  loaded top {len(s1)} forms from {candidates[-1].name}")

    y = df[target].to_numpy()
    groups = df["molecule"].to_numpy()

    # Pull out which descriptors are referenced in the candidate forms
    needed = set()
    for desc_str in s1["descriptors"]:
        needed.update(d.strip() for d in desc_str.split(","))
    needed.update(["mu", "eta",
                   "deta_red" if target.endswith("_red") else "deta_ox"])
    needed = [d for d in needed if d in df.columns]

    X = df[needed].to_numpy()
    name_to_idx = {n: i for i, n in enumerate(needed)}

    rows: list[dict] = []
    for _, base in s1.iterrows():
        try:
            D = parse_label_to_design(base["label"], needed, X)
        except Exception as exc:
            print(f"  ! skip form (parse error): {exc}")
            continue
        # No correction baseline
        pred_nc, _ = cv_fit_no_correction(D, y, groups)
        m_nc = metrics(y, pred_nc)
        rows.append({
            "label": base["label"],
            "descriptors": base["descriptors"],
            "k_base": int(base["k"]),
            "n_params_base": int(base["n_params"]),
            "fukui": "none",
            "n_params": int(base["n_params"]),
            **{f"{k}_cv": v for k, v in m_nc.items()},
            "a_corr": 0.0,
        })
        # Fukui-corrected variants
        for fname, _ in FUKUI_VARIANTS.items():
            if fname == "none":
                continue
            if fname == "f2_paired":
                col = f"f2_paired_red" if target.endswith("_red") else "f2_paired_ox"
            elif fname == "f2_paired_noO":
                col = f"f2_paired_red_noO" if target.endswith("_red") else "f2_paired_ox_noO"
            elif fname == "f2_max_noO":
                col = "f2_max_noO_col"
            else:
                col = fname
            if col not in df.columns:
                continue
            F = df[col].to_numpy()
            pred, theta, a = cv_fit_with_correction(D, F, y, groups)
            m_c = metrics(y, pred)
            rows.append({
                "label": base["label"],
                "descriptors": base["descriptors"],
                "k_base": int(base["k"]),
                "n_params_base": int(base["n_params"]),
                "fukui": fname,
                "n_params": int(base["n_params"]) + 1,
                **{f"{k}_cv": v for k, v in m_c.items()},
                "a_corr": float(a),
            })

    df_res = pd.DataFrame(rows).sort_values("mae_cv").reset_index(drop=True)
    out = RESULTS / f"step2_fukui_{target}.csv"
    df_res.head(top_n).to_csv(out, index=False, float_format="%.6g")
    full = RESULTS / f"step2_fukui_all_{target}.parquet"
    df_res.to_parquet(full, index=False)
    print(f"  best MAE (no corr): {df_res[df_res.fukui == 'none'].mae_cv.min():.4f}")
    print(f"  best MAE (with corr): {df_res[df_res.fukui != 'none'].mae_cv.min():.4f}")
    print(f"  -> {out.name}")

    # Save the very best as final
    best = df_res.iloc[0]
    final = {
        "target": target,
        "label": best["label"],
        "descriptors": best["descriptors"],
        "fukui": best["fukui"],
        "a_corr": best["a_corr"],
        "n_params": int(best["n_params"]),
        "mae_cv": float(best["mae_cv"]),
        "rmse_cv": float(best["rmse_cv"]),
        "r2_cv": float(best["r2_cv"]),
    }
    with open(RESULTS / f"step2_final_form_{target}.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"  -> step2_final_form_{target}.json")
    return df_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None,
                    help=f"restrict to one target: {TARGETS}")
    ap.add_argument("--top-m", type=int, default=20,
                    help="number of Step-1 forms to evaluate")
    ap.add_argument("--top-n", type=int, default=50,
                    help="rows kept in step2_fukui_<target>.csv")
    args = ap.parse_args()

    df, _ = load_data()
    print(f"Acetonitrile data: {df.shape}, {df.molecule.nunique()} molecules")

    targets = [args.target] if args.target else TARGETS
    for tgt in targets:
        run_target(tgt, df, top_m=args.top_m, top_n=args.top_n)


if __name__ == "__main__":
    main()
