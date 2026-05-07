#!/usr/bin/env python3
"""Étape 1 — sélection de descripteurs sur Acetonitrile uniquement.

Pour chaque (descripteur ∈ pool 13, cible ∈ 8 targets), calcule 5 métriques :
    - Pearson  (linéaire)
    - Spearman (monotone)
    - dCor     (non-linéaire)
    - MI       (mutual information)
    - PolyCV   (R² 5-fold CV groupée par molécule du meilleur poly degré ≤ 3)

Le pool expose `lvc2_paired` (et non `lvc2_red`/`lvc2_ox`) : pour les cibles red
(EA, Erelax_red, ddU_red, TdS_red) il vaut `lvc2_red` ; pour les ox il vaut `lvc2_ox`.

Sorties :
    results/step1_correlations_long.csv        (format long descriptor × target × metric)
    results/step1_correlations_pivot_<m>.csv   (un pivot par métrique, format 13 × 8)
    results/step1_correlations_score.csv       (score consolidé descripteur × paire)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import dcor
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "merged_dataset.parquet"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

DESCRIPTORS = [
    "mu", "eta", "gamma",
    "fukui_paired_max", "fukui_paired_max_O",
    "f2_max", "f2_max_O",
    "qbader_max", "qbader_max_O",
    "lr_diag_max",
    "lvc2_paired",
]

TARGETS_RED = ["EA", "Erelax_red", "ddU_red", "TdS_red"]
TARGETS_OX = ["IP", "Erelax_ox", "ddU_ox", "TdS_ox"]
TARGETS = TARGETS_RED + TARGETS_OX

PAIRS = {
    "EA_IP": ("EA", "IP"),
    "Erelax": ("Erelax_red", "Erelax_ox"),
    "ddU": ("ddU_red", "ddU_ox"),
    "TdS": ("TdS_red", "TdS_ox"),
}

N_FOLDS = 5
RNG = np.random.default_rng(42)


PAIRED_MAP = {
    "lvc2_paired":      ("lvc2_red",      "lvc2_ox"),       # red↔red, ox↔ox
    "fukui_paired_max":   ("fplus_max",     "fminus_max"),    # red↔f+ (LUMO), ox↔f- (HOMO)
    "fukui_paired_max_O": ("fplus_max_O",   "fminus_max_O"),
}


def descriptor_series(df: pd.DataFrame, desc: str, target: str) -> pd.Series:
    if desc in PAIRED_MAP:
        red_col, ox_col = PAIRED_MAP[desc]
        return df[red_col] if target in TARGETS_RED else df[ox_col]
    return df[desc]


def poly_cv_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray, max_deg: int = 3) -> float:
    """Best 5-fold (GroupKFold) CV R² of univariate polynomial of degree ≤ max_deg.

    Returns the max over degrees 1..max_deg of the mean fold R² (test).
    """
    if len(np.unique(groups)) < N_FOLDS:
        return float("nan")
    gkf = GroupKFold(n_splits=N_FOLDS)
    best = -np.inf
    for deg in range(1, max_deg + 1):
        # Build polynomial features [x, x^2, ..., x^deg] (no bias - LR has intercept)
        X = np.vander(x, deg + 1, increasing=True)[:, 1:]  # cols 1..deg
        fold_r2 = []
        for tr, te in gkf.split(X, y, groups=groups):
            try:
                model = LinearRegression().fit(X[tr], y[tr])
                pred = model.predict(X[te])
                fold_r2.append(r2_score(y[te], pred))
            except Exception:
                fold_r2.append(np.nan)
        mean_r2 = np.nanmean(fold_r2)
        if mean_r2 > best:
            best = mean_r2
    return float(best) if np.isfinite(best) else float("nan")


def compute_metrics(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Compute 5 association metrics for descriptor x and target y."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[mask], y[mask], groups[mask]
    if len(x) < 10 or np.std(x) == 0:
        return {k: float("nan") for k in ("pearson", "spearman", "dcor", "mi", "poly_cv_r2")}

    pearson = abs(pearsonr(x, y).statistic)
    spearman = abs(spearmanr(x, y).statistic)
    dc = float(dcor.distance_correlation(x, y))
    mi = float(mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0])
    poly = poly_cv_r2(x, y, g)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "dcor": dc,
        "mi": mi,
        "poly_cv_r2": poly,
    }


def main() -> None:
    df = pd.read_parquet(DATA)
    df = df[df.solvent == "Acetonitrile"].reset_index(drop=True)
    print(f"Acetonitrile rows: {len(df)}")
    print(f"Unique molecules: {df.molecule.nunique()}")

    rows = []
    for target in TARGETS:
        y = df[target].to_numpy()
        groups = df["molecule"].to_numpy()
        for desc in DESCRIPTORS:
            x = descriptor_series(df, desc, target).to_numpy()
            metrics = compute_metrics(x, y, groups)
            rows.append({"descriptor": desc, "target": target, **metrics})

    long = pd.DataFrame(rows)
    long.to_csv(OUT / "step1_correlations_long.csv", index=False)
    print(f"\n[saved] {OUT / 'step1_correlations_long.csv'}")

    # Pivot per metric for human reading
    metric_names = ["pearson", "spearman", "dcor", "mi", "poly_cv_r2"]
    for m in metric_names:
        piv = long.pivot(index="descriptor", columns="target", values=m)
        piv = piv.reindex(index=DESCRIPTORS, columns=TARGETS)
        path = OUT / f"step1_correlations_pivot_{m}.csv"
        piv.round(3).to_csv(path)
        print(f"[saved] {path}")

    # Consolidated score: per-pair, average metric ranks (averaged across red/ox of the pair)
    # Higher rank = better; we report mean of normalized ranks as a quick sort key.
    score_rows = []
    for pair_name, (t_red, t_ox) in PAIRS.items():
        for desc in DESCRIPTORS:
            row = {"pair": pair_name, "descriptor": desc}
            for m in metric_names:
                rv = long.query("descriptor == @desc and target == @t_red")[m].iloc[0]
                ov = long.query("descriptor == @desc and target == @t_ox")[m].iloc[0]
                row[f"{m}_red"] = rv
                row[f"{m}_ox"] = ov
                row[f"{m}_avg"] = np.nanmean([rv, ov])
            # Composite: normalize each metric's _avg to [0,1] within the pair, then mean.
            score_rows.append(row)
    score = pd.DataFrame(score_rows)

    for pair_name in PAIRS:
        sub = score[score.pair == pair_name].copy()
        for m in metric_names:
            col = f"{m}_avg"
            vals = sub[col].to_numpy()
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax > vmin:
                sub[f"{m}_norm"] = (vals - vmin) / (vmax - vmin)
            else:
                sub[f"{m}_norm"] = 0.0
        norm_cols = [f"{m}_norm" for m in metric_names]
        sub["composite_score"] = sub[norm_cols].mean(axis=1)
        sub = sub.sort_values("composite_score", ascending=False)
        path = OUT / f"step1_correlations_score_{pair_name}.csv"
        sub.round(4).to_csv(path, index=False)
        print(f"[saved] {path}")

    # Pretty-print summary tables to stdout (one per pair: rank by composite)
    print("\n" + "=" * 80)
    print("SCORE CONSOLIDÉ PAR PAIRE (descripteurs triés par score composite)")
    print("=" * 80)
    for pair_name, (t_red, t_ox) in PAIRS.items():
        path = OUT / f"step1_correlations_score_{pair_name}.csv"
        sub = pd.read_csv(path)
        print(f"\n--- Paire: {pair_name}  ({t_red} / {t_ox}) ---")
        cols_show = [
            "descriptor",
            "pearson_avg", "spearman_avg", "dcor_avg", "mi_avg", "poly_cv_r2_avg",
            "composite_score",
        ]
        print(sub[cols_show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
