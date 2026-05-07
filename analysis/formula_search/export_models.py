#!/usr/bin/env python3
"""Export final fitted models for the best form per pair.

Loads top forms from Phase 3 / Phase 4, refits on the full Phase 1 (Acetonitrile,
single solvent, no correction) and Phase 2 (10 solvents, with best correction)
data, and writes a JSON with theta_red / theta_ox coefficients per form.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))
from forms import Monomial, PolynomialForm  # noqa: E402
from solvent_correction import (  # noqa: E402
    CORRECTIONS, EPSILON_R, fit_polynomial_with_correction, parse_polynomial_form,
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


def export_phase1() -> dict:
    df_full = pd.read_parquet(DATA)
    df = df_full[df_full.solvent == "Acetonitrile"].reset_index(drop=True)
    X = df[DESCRIPTORS].to_numpy()

    out = {}
    for pair in PAIRS:
        top_csv = RESULTS / f"{pair[0]}_{pair[1]}_top30.csv"
        if not top_csv.exists():
            continue
        top_row = pd.read_csv(top_csv).iloc[0]
        if top_row["form_kind"] != "poly":
            continue
        form = parse_polynomial_form(top_row["label"], top_row["descriptors"])
        F = form.design(X)
        theta_r = LinearRegression(fit_intercept=False).fit(F, df[pair[0]]).coef_
        theta_o = LinearRegression(fit_intercept=False).fit(F, df[pair[1]]).coef_

        out[f"{pair[0]}/{pair[1]}"] = {
            "form_label": top_row["label"],
            "descriptors": top_row["descriptors"].split(","),
            "monomials": [list(m.exps) for m in form.monomials],
            "theta_red": theta_r.tolist(),
            "theta_ox": theta_o.tolist(),
            "mae_red_cv_acetonitrile": float(top_row["mae_red_cv"]),
            "mae_ox_cv_acetonitrile": float(top_row["mae_ox_cv"]),
        }
    return out


def export_phase2() -> dict:
    summary_csv = RESULTS / "phase4_summary.csv"
    if not summary_csv.exists():
        return {}
    summary = pd.read_csv(summary_csv)
    df = pd.read_parquet(DATA)
    df["eps_r"] = df["solvent"].map(EPSILON_R)
    X = df[DESCRIPTORS].to_numpy()
    eps = df["eps_r"].to_numpy()

    out = {}
    for pair_label, sub in summary.groupby("pair"):
        target_red, target_ox = pair_label.split("/")
        # pick the form with lowest mae_corrected
        best = sub.loc[sub["mae_corrected"].idxmin()]
        form = parse_polynomial_form(best["form_label"], best["descriptors"])
        corr = best["best_correction"]
        theta_r, phi_r = fit_polynomial_with_correction(
            form, X, df[target_red].to_numpy(), eps, corr)
        theta_o, phi_o = fit_polynomial_with_correction(
            form, X, df[target_ox].to_numpy(), eps, corr)

        out[pair_label] = {
            "form_label": best["form_label"],
            "monomials": [list(m.exps) for m in form.monomials],
            "best_correction": corr,
            "theta_red": theta_r.tolist(),
            "theta_ox": theta_o.tolist(),
            "phi_red": phi_r.tolist(),
            "phi_ox": phi_o.tolist(),
            "mae_baseline": float(best["mae_baseline"]),
            "mae_corrected": float(best["mae_corrected"]),
            "improvement_pct": float(best["improvement_pct"]),
        }
    return out


def main() -> None:
    p1 = export_phase1()
    p2 = export_phase2()
    out = {"phase1_acetonitrile": p1, "phase2_multisolvent": p2}
    out_path = RESULTS / "final_models.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
