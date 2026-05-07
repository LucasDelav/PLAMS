#!/usr/bin/env python3
"""Step 1a — correlation metrics for the 770 frequency-distribution descriptors
against the four ΔΔU/TΔS targets, on Acetonitrile only.

For each (descriptor, target) pair compute:
    - Pearson r  (linear)
    - Spearman rho (monotonic)
    - distance correlation (any non-linear dependence)
    - mutual information (kernel-based, sklearn KSG)
    - polynomial-CV score: best 5-fold MAE of degree ≤ 3 with that descriptor
      alone (the "polynomial correlation" — captures non-monotonic but smooth
      relationships).

Composite ranking per target = mean of normalised (Pearson abs, Spearman abs,
dCor, MI, 1 - poly_MAE/std). Top-K descriptors per target are written for the
brute-force search step.

Outputs:
    results/step1_freq_correlations.csv — full table
    results/step1_freq_top<K>_<target>.txt — descriptor list per target
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import dcor
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "analysis" / "formula_search" / "data"
RESULTS = REPO / "analysis" / "formula_search" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MERGED = DATA_DIR / "merged_dataset.parquet"
DELTA = DATA_DIR / "ddu_delta_descriptors.parquet"
FREQ = DATA_DIR / "freq_descriptors_full.parquet"

SOLVENT = "Acetonitrile"
EH_TO_KJ = 2625.499638
TARGETS = ["ddU_red", "ddU_ox", "TdS_red", "TdS_ox"]
N_FOLDS = 5
TOP_K = 50

META = {"solvent", "molecule", "conformer", "n_modes", "n_imag_n", "n_imag_r", "n_imag_o", "_pairB_error"}


def load_data() -> pd.DataFrame:
    """Load and merge frequency descriptors with targets (Acetonitrile only)."""
    merged = pd.read_parquet(MERGED)
    delta = pd.read_parquet(DELTA)
    freq = pd.read_parquet(FREQ)

    print(f"merged: {merged.shape}, delta: {delta.shape}, freq: {freq.shape}")

    # Targets in Hartree -> kJ/mol
    df = merged[merged.solvent == SOLVENT].copy()
    for c in TARGETS:
        df[c] = df[c] * EH_TO_KJ

    # Add Δη_paired (depends on target, but we compute both and let user query)
    delta_ace = delta[delta.solvent == SOLVENT][
        ["solvent", "molecule", "conformer", "deta_red", "deta_ox"]
    ]
    df = df.merge(delta_ace, on=["solvent", "molecule", "conformer"], how="inner")

    # Frequency descriptors
    freq_ace = freq[freq.solvent == SOLVENT]
    df = df.merge(freq_ace, on=["solvent", "molecule", "conformer"], how="inner")

    print(f"merged Acetonitrile dataset: {df.shape}, {df.molecule.nunique()} molecules")
    return df


def poly_cv_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  max_degree: int = 3) -> float:
    """5-fold GroupKFold CV MAE of best polynomial of degree ≤ max_degree."""
    best_mae = np.inf
    n = len(x)
    for deg in range(1, max_degree + 1):
        F = np.column_stack([x ** d for d in range(deg + 1)])
        pred = np.full(n, np.nan)
        try:
            for tr, te in GroupKFold(n_splits=N_FOLDS).split(F, y, groups):
                m = LinearRegression(fit_intercept=False).fit(F[tr], y[tr])
                pred[te] = m.predict(F[te])
            mae = np.mean(np.abs(y - pred))
            if mae < best_mae:
                best_mae = mae
        except Exception:
            pass
    return float(best_mae)


def compute_metrics(df: pd.DataFrame, descriptors: list[str]) -> pd.DataFrame:
    groups = df["molecule"].to_numpy()
    rows = []
    t0 = time.perf_counter()
    for i, d in enumerate(descriptors):
        x = df[d].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            continue
        if x.std() < 1e-12:
            continue  # constant column
        for tgt in TARGETS:
            y = df[tgt].to_numpy(dtype=float)
            if not np.isfinite(y).all():
                continue
            std_y = float(y.std())
            try:
                r_pear, _ = pearsonr(x, y)
                r_spear, _ = spearmanr(x, y)
                r_dcor = float(dcor.distance_correlation(x, y))
                mi = float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])
                pmae = poly_cv_score(x, y, groups)
                pmae_norm = pmae / std_y if std_y > 0 else np.nan
                rows.append({
                    "descriptor": d,
                    "target": tgt,
                    "pearson": float(r_pear),
                    "spearman": float(r_spear),
                    "dcor": r_dcor,
                    "mi": mi,
                    "poly_mae": pmae,
                    "poly_mae_norm": pmae_norm,
                    "abs_pearson": abs(float(r_pear)),
                    "abs_spearman": abs(float(r_spear)),
                    "poly_skill": max(0.0, 1.0 - pmae_norm) if np.isfinite(pmae_norm) else 0.0,
                })
            except Exception:
                continue
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(descriptors)} descriptors ({elapsed:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    df = load_data()
    descriptors = [c for c in df.columns
                   if c not in META
                   and not c.startswith(("ddU_", "TdS_", "EA", "IP", "Erelax_",
                                          "Ee_", "HOMO", "LUMO"))
                   and df[c].dtype != "object"]
    print(f"Candidate descriptors: {len(descriptors)}")
    # Always include the baseline 3 (μ, η, Δη_paired_red and _ox available as deta_red/ox)
    for d in ("mu", "eta", "deta_red", "deta_ox"):
        if d not in descriptors and d in df.columns:
            descriptors.append(d)
    print(f"Total with baseline: {len(descriptors)}")

    print("\nComputing correlation metrics ...")
    metrics_df = compute_metrics(df, descriptors)
    out_csv = RESULTS / "step1_freq_correlations.csv"
    metrics_df.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"+ Saved {out_csv} ({metrics_df.shape})")

    # Composite ranking per target: mean of (abs_pearson, abs_spearman, dcor,
    # MI normalised within target, poly_skill).
    print("\nTop descriptors per target (composite rank):")
    rankings = {}
    for tgt in TARGETS:
        sub = metrics_df[metrics_df.target == tgt].copy()
        # Normalise each metric to [0,1]
        for col in ("abs_pearson", "abs_spearman", "dcor", "mi", "poly_skill"):
            v = sub[col].to_numpy()
            v_max = v.max()
            sub[f"{col}_norm"] = (v / v_max) if v_max > 0 else 0.0
        sub["composite"] = sub[[f"{c}_norm" for c in
                                ("abs_pearson", "abs_spearman", "dcor",
                                 "mi", "poly_skill")]].mean(axis=1)
        sub = sub.sort_values("composite", ascending=False).reset_index(drop=True)
        rankings[tgt] = sub.head(TOP_K).descriptor.tolist()
        print(f"\n  --- {tgt} ---")
        for r in sub.head(15).itertuples():
            print(f"    {r.descriptor:35s}  pear={r.pearson:+.3f}  "
                  f"sp={r.spearman:+.3f}  dCor={r.dcor:.3f}  "
                  f"MI={r.mi:.3f}  polyMAE={r.poly_mae:.3f}  comp={r.composite:.3f}")

    for tgt, lst in rankings.items():
        out = RESULTS / f"step1_freq_top{TOP_K}_{tgt}.txt"
        out.write_text("\n".join(lst))
        print(f"  -> {out.name} ({len(lst)} descriptors)")


if __name__ == "__main__":
    main()
