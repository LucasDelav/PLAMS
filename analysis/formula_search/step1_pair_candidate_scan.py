#!/usr/bin/env python3
"""Step 1 — homogeneous-pair candidate scan over physically-motivated 4th
descriptors.

Background: in the previous (μ, η, Δη_paired, dRr_dU_total) search, the
4th descriptor was incorrectly fed identical for red and ox (using only
the red-pair value). Fixed here: every paired descriptor maps to its own
red column AND ox column.

For each candidate X_paired, fit the homogeneous form
    y_state = (β_μ·μ + β_η·η + β_Δη·Δη_paired + β_X·X_paired_state) × (1 + a·F_state)
and rank candidates by MAE_red + MAE_ox (5-fold CV by molecule).

The scan covers:
  - Total paired RRHO sums (dRr*/dRo* dU/dUq/dTS/dTSq)
  - Window-restricted RRHO sums (low / mid / high frequency bands)
  - Δω-distribution descriptors (RMS, max abs, mode-softening counts...)
  - Pairing variants (A = energy order, B = eigenvector overlap)

Output: results/step1_candidate_scan_<pair>.csv
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))
from step2_fukui_correction import _fit_one, metrics  # noqa: E402

DATA_DIR = REPO / "analysis" / "formula_search" / "data"
RESULTS = REPO / "analysis" / "formula_search" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EH_TO_KJ = 2625.499638
SOLVENT = "Acetonitrile"
N_FOLDS = 5


@dataclass(frozen=True)
class Paired:
    """A descriptor with state-specific column names. For red the value is
    `red_col`, for ox the value is `ox_col`."""
    name: str         # logical name, used in labels
    red_col: str
    ox_col: str

    def values(self, df: pd.DataFrame, state: str) -> np.ndarray:
        col = self.red_col if state == "red" else self.ox_col
        return df[col].to_numpy(dtype=float)


# Baseline 3 descriptors (always present in the pool)
BASE_DESCRIPTORS = [
    Paired("mu", "mu", "mu"),
    Paired("eta", "eta", "eta"),
    Paired("deta_paired", "deta_red", "deta_ox"),
]

# Candidate 4th descriptors X_paired (red_col, ox_col)
# Each pair uses descriptors computed from the relevant frequency set:
#   - red target uses dRr_*/dwr_* (red ion vs neutral)
#   - ox target  uses dRo_*/dwo_* (ox ion vs neutral)
def _candidates_ddU() -> list[Paired]:
    cands = []
    # Total RRHO sums (standard and qRRHO), both pairings
    for pairing in ("A", "B"):
        for kind in ("dU", "dUq"):
            cands.append(Paired(f"dRrr_{pairing}_{kind}_total",
                                f"dRr_{pairing}_{kind}_total",
                                f"dRo_{pairing}_{kind}_total"))
    # Window-restricted RRHO sums for ΔU
    windows = ["0_500", "0_1000", "0_1500", "0_2000",
               "500_800", "800_1200", "1200_1600",
               "1500_inf", "2000_inf", "3000_inf"]
    for pairing in ("A", "B"):
        for win in windows:
            for kind in ("dU", "dUq"):
                cands.append(Paired(f"dRrr_{pairing}_{kind}_{win}",
                                    f"dRr_{pairing}_{kind}_{win}",
                                    f"dRo_{pairing}_{kind}_{win}"))
    # Δω-distribution descriptors
    dw_stats = ["dmean", "dstd", "dabs_mean", "dabs_max",
                "dmax_signed", "dmin_signed", "drms",
                "dsoft", "dhard",
                "dN_abs_gt_25", "dN_abs_gt_50",
                "dN_abs_gt_100", "dN_abs_gt_200"]
    for pairing in ("A", "B"):
        for stat in dw_stats:
            cands.append(Paired(f"dw_{pairing}_{stat}",
                                f"dwr_{pairing}_{stat}",
                                f"dwo_{pairing}_{stat}"))
    return cands


def _candidates_TdS() -> list[Paired]:
    cands = []
    for pairing in ("A", "B"):
        for kind in ("dTS", "dTSq"):
            cands.append(Paired(f"dRrr_{pairing}_{kind}_total",
                                f"dRr_{pairing}_{kind}_total",
                                f"dRo_{pairing}_{kind}_total"))
    windows = ["0_300", "0_500", "0_1000", "0_1500", "0_2000",
               "100_200", "200_300", "300_500"]
    for pairing in ("A", "B"):
        for win in windows:
            for kind in ("dTS", "dTSq"):
                cands.append(Paired(f"dRrr_{pairing}_{kind}_{win}",
                                    f"dRr_{pairing}_{kind}_{win}",
                                    f"dRo_{pairing}_{kind}_{win}"))
    # Same Δω descriptors (low-freq variant might matter for entropy)
    dw_stats = ["dmean", "dstd", "drms", "dabs_max",
                "dN_abs_gt_25", "dN_abs_gt_50", "dN_abs_gt_100",
                "dsoft", "dhard"]
    for pairing in ("A", "B"):
        for stat in dw_stats:
            cands.append(Paired(f"dw_{pairing}_{stat}",
                                f"dwr_{pairing}_{stat}",
                                f"dwo_{pairing}_{stat}"))
    return cands


def load_data() -> pd.DataFrame:
    merged = pd.read_parquet(DATA_DIR / "merged_dataset.parquet")
    delta = pd.read_parquet(DATA_DIR / "ddu_delta_descriptors.parquet")
    freq = pd.read_parquet(DATA_DIR / "freq_descriptors_full.parquet")

    df = merged[merged.solvent == SOLVENT].copy()
    for c in ("ddU_red", "ddU_ox", "TdS_red", "TdS_ox"):
        df[c] = df[c] * EH_TO_KJ
    delta_ace = delta[delta.solvent == SOLVENT][[
        "solvent", "molecule", "conformer",
        "deta_red", "deta_ox", "f2_max", "f2_min",
    ]].rename(columns={"f2_max": "f2max_corr", "f2_min": "f2min_corr"})
    df = df.merge(delta_ace, on=["solvent", "molecule", "conformer"], how="inner")
    df = df.merge(freq, on=["solvent", "molecule", "conformer"], how="inner")
    return df


def cv_fit_paired(df, descriptors_paired, fukui_red_col, fukui_ox_col,
                   y_red, y_ox, with_fukui=True):
    """Fit and CV-score the form on red and ox with the given paired descriptors."""
    n = len(df)
    groups = df["molecule"].to_numpy()
    F_red = df[fukui_red_col].to_numpy()
    F_ox = df[fukui_ox_col].to_numpy()

    D_red = np.column_stack([d.values(df, "red") for d in descriptors_paired])
    D_ox = np.column_stack([d.values(df, "ox") for d in descriptors_paired])

    pred_r = np.full(n, np.nan)
    pred_o = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D_red, y_red, groups):
        if with_fukui:
            theta_r, a_r = _fit_one(D_red[tr], F_red[tr], y_red[tr])
            theta_o, a_o = _fit_one(D_ox[tr], F_ox[tr], y_ox[tr])
            pred_r[te] = (D_red[te] @ theta_r) * (1.0 + a_r * F_red[te])
            pred_o[te] = (D_ox[te] @ theta_o) * (1.0 + a_o * F_ox[te])
        else:
            theta_r = np.linalg.lstsq(D_red[tr], y_red[tr], rcond=None)[0]
            theta_o = np.linalg.lstsq(D_ox[tr], y_ox[tr], rcond=None)[0]
            pred_r[te] = D_red[te] @ theta_r
            pred_o[te] = D_ox[te] @ theta_o
    return pred_r, pred_o


def scan_pair(label: str, candidates: list[Paired], df: pd.DataFrame,
              targets, fukui_red_col, fukui_ox_col,
              with_fukui: bool = True) -> pd.DataFrame:
    name_red, name_ox = targets
    y_red = df[name_red].to_numpy()
    y_ox = df[name_ox].to_numpy()

    rows = []
    # Baseline (no X): the rapport's pool (μ, η, Δη_paired)
    pred_r, pred_o = cv_fit_paired(df, BASE_DESCRIPTORS,
                                     fukui_red_col, fukui_ox_col,
                                     y_red, y_ox, with_fukui)
    m_r, m_o = metrics(y_red, pred_r), metrics(y_ox, pred_o)
    rows.append({
        "candidate_X": "(none — baseline)",
        "n_total_unrelated_to_X": 3,
        "mae_red": m_r["mae"], "mae_ox": m_o["mae"],
        "mae_total": m_r["mae"] + m_o["mae"],
        "r2_red": m_r["r2"], "r2_ox": m_o["r2"],
    })

    # Each X candidate: pool = baseline + X
    for X in candidates:
        # Skip if any required column is missing
        if X.red_col not in df.columns or X.ox_col not in df.columns:
            continue
        # Skip if column is constant
        if df[X.red_col].std() < 1e-12 or df[X.ox_col].std() < 1e-12:
            continue
        descs = list(BASE_DESCRIPTORS) + [X]
        pred_r, pred_o = cv_fit_paired(df, descs,
                                         fukui_red_col, fukui_ox_col,
                                         y_red, y_ox, with_fukui)
        m_r = metrics(y_red, pred_r)
        m_o = metrics(y_ox, pred_o)
        rows.append({
            "candidate_X": X.name,
            "X_red_col": X.red_col,
            "X_ox_col": X.ox_col,
            "mae_red": m_r["mae"], "mae_ox": m_o["mae"],
            "mae_total": m_r["mae"] + m_o["mae"],
            "r2_red": m_r["r2"], "r2_ox": m_o["r2"],
        })
    df_res = pd.DataFrame(rows).sort_values("mae_total").reset_index(drop=True)
    out = RESULTS / f"step1_candidate_scan_{label}.csv"
    df_res.to_csv(out, index=False, float_format="%.6g")
    print(f"  -> {out.name} ({len(df_res)} candidates)")
    return df_res


def main():
    df = load_data()
    print(f"Acetonitrile dataset: {df.shape}, {df.molecule.nunique()} molecules")

    print("\n=== ΔΔU pair candidate scan (with Fukui multiplicative)")
    cands = _candidates_ddU()
    print(f"  {len(cands)} candidate descriptors")
    df_ddu = scan_pair("ddU", cands, df,
                        targets=("ddU_red", "ddU_ox"),
                        fukui_red_col="f2max_corr",
                        fukui_ox_col="f2min_corr",
                        with_fukui=True)
    print("\n  Top 15 (Fukui ON):")
    print(df_ddu.head(15).to_string(index=False))

    print("\n=== TΔS pair candidate scan (with Fukui multiplicative)")
    cands = _candidates_TdS()
    print(f"  {len(cands)} candidate descriptors")
    df_tds = scan_pair("TdS", cands, df,
                        targets=("TdS_red", "TdS_ox"),
                        fukui_red_col="f2max_corr",
                        fukui_ox_col="f2min_corr",
                        with_fukui=True)
    print("\n  Top 15 (Fukui ON):")
    print(df_tds.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
