#!/usr/bin/env python3
"""Step 1 — search for a physically motivated 4th descriptor to REPLACE
Δη_paired in the (μ, η, Δη_paired, dRr_dU_0_1500_paired) pool.

The premise: Δη_paired is a vertical electronic property (CDFT), not directly
related to vibrational reorganization. Its presence in the rapport baseline
is a correlation with the underlying chemistry, not a causal physical link.
We hypothesise that swapping it out for either:
  (a) nothing (k=3 model: μ, η, dRr_paired)
  (b) a vibrational distribution descriptor (Δω-based moment / count)
  (c) a coupling/vibronic descriptor (LVC2)
  (d) a higher-order CDFT term (γ, ω electrophilicity)
might preserve or improve the fit while giving a cleaner physical story.

Form: y = (β·d) × (1 + a·F_paired) on Acetonitrile, 5-fold CV by molecule.

Outputs:
    results/step1_replace_deta.csv (full ranking)
    results/step1_replace_deta_summary.md
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    name: str
    red_col: str
    ox_col: str
    category: str  # short physical category for grouping in output

    def values(self, df, state):
        return df[self.red_col if state == "red" else self.ox_col].to_numpy(dtype=float)


# -- Anchor (always in the pool): vibrational direct + minimal electronic
ANCHOR = [
    Paired("mu", "mu", "mu", "CDFT"),
    Paired("eta", "eta", "eta", "CDFT"),
    Paired("dRr_dU_0_1500", "dRr_A_dU_0_1500", "dRo_A_dU_0_1500", "vib-direct"),
]

# -- Candidates for the 4th descriptor X (categorised by physical meaning)
CANDIDATES = [
    # (a) electronic CDFT higher-order
    Paired("deta_paired", "deta_red", "deta_ox", "CDFT-vertical (rapport)"),
    Paired("gamma", "gamma", "gamma", "CDFT γ=∂³E/∂N³"),

    # (b) Bader / linear response (electronic, local)
    Paired("qbader_max", "qbader_max", "qbader_max", "QTAIM"),
    Paired("qbader_max_O", "qbader_max_O", "qbader_max_O", "QTAIM_noO"),
    Paired("lr_diag_max", "lr_diag_max", "lr_diag_max", "Lin. response"),

    # (c) Vibronic coupling (LVC2 — frontier-orbital × normal mode coupling)
    Paired("lvc2_paired", "lvc2_red", "lvc2_ox", "Vibronic LVC2"),

    # (d) Δω-distribution descriptors (truly vibrational, no electronic content)
    Paired("dw_drms_A", "dwr_A_drms", "dwo_A_drms", "Δω RMS"),
    Paired("dw_dstd_A", "dwr_A_dstd", "dwo_A_dstd", "Δω std"),
    Paired("dw_dabs_max_A", "dwr_A_dabs_max", "dwo_A_dabs_max", "Δω |max|"),
    Paired("dw_dabs_mean_A", "dwr_A_dabs_mean", "dwo_A_dabs_mean", "Δω |mean|"),
    Paired("dw_dsoft_A", "dwr_A_dsoft", "dwo_A_dsoft", "# softening modes"),
    Paired("dw_dhard_A", "dwr_A_dhard", "dwo_A_dhard", "# hardening modes"),
    Paired("dw_dN_abs_gt_50_A", "dwr_A_dN_abs_gt_50", "dwo_A_dN_abs_gt_50", "# Δω>50"),
    Paired("dw_dN_abs_gt_100_A", "dwr_A_dN_abs_gt_100", "dwo_A_dN_abs_gt_100", "# Δω>100"),
    Paired("dw_dmean_A", "dwr_A_dmean", "dwo_A_dmean", "mean signed Δω"),
    Paired("dw_dmax_signed_A", "dwr_A_dmax_signed", "dwo_A_dmax_signed", "max(Δω)"),
    Paired("dw_dmin_signed_A", "dwr_A_dmin_signed", "dwo_A_dmin_signed", "min(Δω)"),

    # Eigenvector-overlap pairing variants
    Paired("dw_drms_B", "dwr_B_drms", "dwo_B_drms", "Δω RMS (overlap)"),
    Paired("dw_dabs_max_B", "dwr_B_dabs_max", "dwo_B_dabs_max", "Δω |max| (overlap)"),

    # (e) Frequency-distribution properties of the neutral or ion alone
    #     (purely vibrational, single-state descriptors)
    Paired("wr_kurt", "wn_cut0_kurt", "wn_cut0_kurt", "ω_neut kurt (state-shared)"),
    # ω_red q50 / wo_q50 to see if the median frequency varies meaningfully
    Paired("wstate_q50", "wr_q50", "wo_q50", "ω_state q50"),
    Paired("wstate_q10", "wr_q10", "wo_q10", "ω_state q10"),

    # (f) Other: ZPE, TS_vib of neutral state
    #     Already implicitly captured by ω_neutre statistics; skipped to focus
    #     on candidates that genuinely add info.

    # (g) ω as a 4th descriptor — present in Erelax model, "frequency scale"
    Paired("omega_neutral_q50", "wn_cut0_q50", "wn_cut0_q50", "ω_neutre q50 (size proxy)"),
]


def load_data() -> pd.DataFrame:
    merged = pd.read_parquet(DATA_DIR / "merged_dataset.parquet")
    delta = pd.read_parquet(DATA_DIR / "ddu_delta_descriptors.parquet")
    freq = pd.read_parquet(DATA_DIR / "freq_descriptors_full.parquet")

    df = merged[merged.solvent == SOLVENT].copy()
    for c in ("ddU_red", "ddU_ox", "TdS_red", "TdS_ox"):
        df[c] = df[c] * EH_TO_KJ
    df = df.merge(
        delta[["solvent", "molecule", "conformer",
               "deta_red", "deta_ox", "f2_max", "f2_min"]]
            .rename(columns={"f2_max": "f2max_corr", "f2_min": "f2min_corr"}),
        on=["solvent", "molecule", "conformer"], how="inner",
    )
    df = df.merge(freq, on=["solvent", "molecule", "conformer"], how="inner")
    return df


def cv_fit_form(descriptors_paired, df, fukui_red_col, fukui_ox_col, y_red, y_ox):
    n = len(df)
    groups = df["molecule"].to_numpy()
    F_red = df[fukui_red_col].to_numpy()
    F_ox = df[fukui_ox_col].to_numpy()
    D_red = np.column_stack([d.values(df, "red") for d in descriptors_paired])
    D_ox = np.column_stack([d.values(df, "ox") for d in descriptors_paired])
    pred_r = np.full(n, np.nan)
    pred_o = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D_red, y_red, groups):
        theta_r, a_r = _fit_one(D_red[tr], F_red[tr], y_red[tr])
        theta_o, a_o = _fit_one(D_ox[tr], F_ox[tr], y_ox[tr])
        pred_r[te] = (D_red[te] @ theta_r) * (1.0 + a_r * F_red[te])
        pred_o[te] = (D_ox[te] @ theta_o) * (1.0 + a_o * F_ox[te])
    return pred_r, pred_o


def main():
    df = load_data()
    print(f"Acetonitrile: {df.shape}, {df.molecule.nunique()} molecules")
    y_red = df["ddU_red"].to_numpy()
    y_ox = df["ddU_ox"].to_numpy()

    rows = []

    # --- 3-monome baseline (no X) ---
    pred_r, pred_o = cv_fit_form(ANCHOR, df, "f2max_corr", "f2min_corr", y_red, y_ox)
    m_r = metrics(y_red, pred_r); m_o = metrics(y_ox, pred_o)
    rows.append({
        "X_name": "(none — k=3 anchor)",
        "X_category": "—",
        "k": 3, "n_total": 8,  # 3 βred + 3 βox + a_red + a_ox
        "mae_red": m_r["mae"], "mae_ox": m_o["mae"],
        "mae_total": m_r["mae"] + m_o["mae"],
        "r2_red": m_r["r2"], "r2_ox": m_o["r2"],
    })

    # --- For each X candidate: anchor + X ---
    skipped = []
    for X in CANDIDATES:
        if X.red_col not in df.columns or X.ox_col not in df.columns:
            skipped.append((X.name, "missing column"))
            continue
        if df[X.red_col].std() < 1e-12 or df[X.ox_col].std() < 1e-12:
            skipped.append((X.name, "constant column"))
            continue
        pool = list(ANCHOR) + [X]
        pred_r, pred_o = cv_fit_form(pool, df, "f2max_corr", "f2min_corr",
                                       y_red, y_ox)
        m_r = metrics(y_red, pred_r); m_o = metrics(y_ox, pred_o)
        rows.append({
            "X_name": X.name,
            "X_category": X.category,
            "k": 4, "n_total": 10,
            "X_red_col": X.red_col, "X_ox_col": X.ox_col,
            "mae_red": m_r["mae"], "mae_ox": m_o["mae"],
            "mae_total": m_r["mae"] + m_o["mae"],
            "r2_red": m_r["r2"], "r2_ox": m_o["r2"],
        })

    df_res = pd.DataFrame(rows).sort_values("mae_total").reset_index(drop=True)
    out = RESULTS / "step1_replace_deta.csv"
    df_res.to_csv(out, index=False, float_format="%.6g")
    print(f"\n+ Saved {out.name}\n")
    print(f"Anchor: {[d.name for d in ANCHOR]} + Fukui f²_paired")
    print(f"Tested {len(rows) - 1} X-replacement candidates ({len(skipped)} skipped)")

    print("\nFull ranking:")
    cols = ["X_name", "X_category", "k", "mae_red", "mae_ox", "mae_total",
            "r2_red", "r2_ox"]
    print(df_res[cols].to_string(index=False))

    if skipped:
        print(f"\nSkipped: {skipped}")


if __name__ == "__main__":
    main()
