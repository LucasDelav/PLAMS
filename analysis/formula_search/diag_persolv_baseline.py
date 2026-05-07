#!/usr/bin/env python3
"""Baseline diagnostics: per-solvent OLS-through-origin and global OLS.

Establishes the lower bound (per-solvent OLS) and the no-correction baseline
(single global linear fit) that the Padé[m/n](eps_r) must beat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
DESCRIPTORS = ["mu", "eta", "omega"]


def main() -> None:
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)

    for tgt in ("IP", "EA"):
        y = df[tgt].values * EH_TO_KJ
        X_all = df[DESCRIPTORS].values

        # Global OLS (one set of coefficients)
        beta_global, *_ = np.linalg.lstsq(X_all, y, rcond=None)
        mae_global = float(np.mean(np.abs(y - X_all @ beta_global)))

        print(f"\n=== {tgt} — global OLS-through-origin (no eps dependence) ===")
        print(f"  beta = {dict(zip(DESCRIPTORS, [f'{b:8.3f}' for b in beta_global]))}")
        print(f"  global MAE = {mae_global:7.3f} kJ/mol")

        # Per-solvent OLS (lower bound)
        rows = []
        for solv, sub in df.groupby("solvent"):
            X = sub[DESCRIPTORS].values
            ys = sub[tgt].values * EH_TO_KJ
            beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
            mae = float(np.mean(np.abs(ys - X @ beta)))
            rows.append({
                "solvent": solv, "epsilon_r": EPSILON_R[solv],
                "beta_mu": beta[0], "beta_eta": beta[1], "beta_omega": beta[2],
                "MAE": mae, "n": len(sub),
            })
        per = pd.DataFrame(rows).sort_values("epsilon_r")
        # Weighted average MAE = lower bound for any global model
        wmae = float((per.MAE * per.n).sum() / per.n.sum())
        print(f"\n=== {tgt} — per-solvent OLS-through-origin (lower bound) ===")
        print(per.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
        print(f"\n  weighted-average MAE = {wmae:7.3f} kJ/mol  (lower bound for any Padé(eps))")


if __name__ == "__main__":
    main()
