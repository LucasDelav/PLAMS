#!/usr/bin/env python3
"""Rank conformers by absolute IP error under the Padé[1/1] g_Born model.

Loads the saved final_solvent_models.json, reconstructs IP predictions on the
1336-conformer dataset, then sorts and reports:
  - Top 30 worst individual conformers
  - Per-molecule mean absolute error (worst 20)
  - Per-solvent breakdown (with std and max)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
MODELS = REPO / "analysis" / "formula_search" / "results" / "final_solvent_models.json"

EH_TO_KJ = 2625.499638
EXCLUDED_MOLS = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58, "Dichloromethane": 8.93,
    "Acetone": 20.7, "Methanol": 32.6, "Acetonitrile": 35.7,
    "DMSO": 46.7, "Water": 78.4,
}
DESCRIPTORS = ("mu", "eta", "omega")
STAB_TOL = 0.05


def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def beta_pade11_g(eps, p):
    g = g_born(eps)
    return (p["a0"] + p["a1"] * g) / (1.0 + p["b1"] * g)


def main():
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED_MOLS)].reset_index(drop=True)
    df["omega"] = df["mu"] ** 2 / (2.0 * df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)

    models = json.load(open(MODELS))
    p_ip = models["pade11_g"]["targets"]["IP"]["params_per_descriptor"]

    eps = df["epsilon"].values
    pred = np.zeros(len(df))
    for d in DESCRIPTORS:
        pred += beta_pade11_g(eps, p_ip[d]) * df[d].values

    y = df["IP"].values * EH_TO_KJ
    df["IP_kJmol"] = y
    df["IP_pred_kJmol"] = pred
    df["abs_err"] = np.abs(y - pred)
    df["signed_err"] = pred - y

    mae = df.abs_err.mean()
    rmse = float(np.sqrt(np.mean(df.signed_err ** 2)))
    print(f"Padé[1/1] in g_Born — IP — n={len(df)}")
    print(f"  MAE  = {mae:.3f} kJ/mol")
    print(f"  RMSE = {rmse:.3f} kJ/mol")
    print(f"  RMSE/MAE = {rmse / mae:.2f}  (Gaussian ~1.25)")
    print(f"  Max |err| = {df.abs_err.max():.2f} kJ/mol")
    print(f"  P50 = {df.abs_err.median():.2f}, P90 = {df.abs_err.quantile(0.90):.2f}, "
          f"P95 = {df.abs_err.quantile(0.95):.2f}, P99 = {df.abs_err.quantile(0.99):.2f}")

    # Top 30 worst conformers
    worst = df.sort_values("abs_err", ascending=False).head(30)
    print("\n=== Top 30 worst conformers (IP, signed = pred - true) ===")
    print(worst[["molecule", "conformer", "solvent", "IP_kJmol",
                 "IP_pred_kJmol", "signed_err", "abs_err"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    # Per-molecule average
    by_mol = df.groupby("molecule").agg(
        mean_abs_err=("abs_err", "mean"),
        max_abs_err=("abs_err", "max"),
        n=("abs_err", "size"),
    ).sort_values("mean_abs_err", ascending=False)
    print("\n=== Top 20 worst molecules by mean |err| (averaged over conformers/solvents) ===")
    print(by_mol.head(20).to_string(float_format=lambda x: f"{x:7.2f}"))

    # Per-solvent
    by_solv = df.groupby("solvent").agg(
        MAE=("abs_err", "mean"),
        RMSE=("signed_err", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        max_err=("abs_err", "max"),
        n=("abs_err", "size"),
    ).reset_index()
    by_solv["epsilon"] = by_solv["solvent"].map(EPSILON_R)
    by_solv = by_solv.sort_values("epsilon")[["solvent", "epsilon", "MAE", "RMSE", "max_err", "n"]]
    print("\n=== Per-solvent breakdown ===")
    print(by_solv.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    # Save full per-conformer error table
    out_csv = REPO / "analysis" / "formula_search" / "results" / "ip_per_conformer_error.csv"
    df_save = df[["solvent", "epsilon", "molecule", "conformer",
                   "IP_kJmol", "IP_pred_kJmol", "signed_err", "abs_err"]] \
                .sort_values("abs_err", ascending=False)
    df_save.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
