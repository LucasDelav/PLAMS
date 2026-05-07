#!/usr/bin/env python3
"""Merge dataset/ (DFT targets) and dataset_predict_dg/ (descriptors).

Output: parquet with one row per (solvent, molecule, conformer).

Descriptors (12, all extracted from `states.neutral_sp` of the predict-dg JSON):
    mu, eta, gamma                     - global ConceptualDFT scalars
    fplus_max,  fplus_max_O            - max of f+ over all atoms / over O atoms
    fminus_max, fminus_max_O           - max of f- over all atoms / over O atoms
    f2_max,     f2_max_O               - max of dual descriptor (f+ - f-)
    qbader_max, qbader_max_O           - max of |bader charge|
    lr_diag_max                        - max diagonal entry of linear response

`gamma` = ∂³E/∂N³ = ∂²μ/∂N² computed from the three Koopmans chemical potentials
of the predict-dg pipeline: gamma = μ(reduced_sp) − 2·μ(neutral_sp) + μ(oxidized_sp)
(centered finite-difference, exact for the 3-point dataset). Same eV unit as μ/η.

Targets (8, all in Hartree, from `derived` of the DFT JSON):
    EA, IP                              - vertical electron affinity / ionization potential
    Erelax_red, Erelax_ox               - deformation energies (memory: name renamed from Edef)
    ddU_red,    ddU_ox                  - internal energy correction
    TdS_red,    TdS_ox                  - entropy term

Water/Chloroform dedup: drop Water rows whose Ee_neutral matches the Chloroform
row of the same (mol, conf) within 1e-8 Ha (known datapoint duplication bug).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DFT_DIR = REPO / "dataset"
PRED_DIR = REPO / "dataset_predict_dg"
OUT_DIR = REPO / "analysis" / "formula_search" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTOR_COLS = [
    "mu", "eta", "gamma",
    "fplus_max", "fplus_max_O",
    "fminus_max", "fminus_max_O",
    "f2_max", "f2_max_O",
    "qbader_max", "qbader_max_O",
    "lr_diag_max",
    "lvc2_red", "lvc2_ox",
]

TARGET_COLS = [
    "EA", "IP",
    "Erelax_red", "Erelax_ox",
    "ddU_red", "ddU_ox",
    "TdS_red", "TdS_ox",
]


def parse_xyz_symbols(xyz: str) -> list[str]:
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2 + i].split()[0] for i in range(n)]


def reduce_per_atom(values: list[float], symbols: list[str], target: str = "O") -> tuple[float, float]:
    """Return (max global, max over atoms with `symbol == target`).

    For lr_diag (negative-valued), we use absolute value.
    Otherwise return raw max (signed)."""
    arr = np.asarray(values, dtype=float)
    syms = np.asarray(symbols)
    glob = float(arr.max())
    mask = syms == target
    sub = float(arr[mask].max()) if mask.any() else float("nan")
    return glob, sub


def reduce_abs(values: list[float], symbols: list[str], target: str = "O") -> tuple[float, float]:
    """Return (max |x| global, max |x| over O atoms)."""
    arr = np.abs(np.asarray(values, dtype=float))
    syms = np.asarray(symbols)
    glob = float(arr.max())
    mask = syms == target
    sub = float(arr[mask].max()) if mask.any() else float("nan")
    return glob, sub


def load_pair(pred_json: Path, dft_json: Path) -> dict | None:
    try:
        with open(pred_json) as f:
            pred = json.load(f)
        with open(dft_json) as f:
            dft = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ! skip {pred_json.parent.name}: {e}", file=sys.stderr)
        return None

    try:
        ns = pred["states"]["neutral_sp"]
        cdft = ns["cdft"]
        symbols = parse_xyz_symbols(ns["xyz"])

        fplus_g, fplus_O = reduce_per_atom(cdft["fplus"], symbols)
        fminus_g, fminus_O = reduce_per_atom(cdft["fminus"], symbols)
        f2_g, f2_O = reduce_per_atom(cdft["dual"], symbols)
        qb_g, qb_O = reduce_abs(ns["qtaim"]["bader_charges"], symbols)

        lr = ns["matrices"]["linear_response"]
        lr_diag = [lr[i][i] for i in range(len(lr))]
        lr_diag_max = float(np.max(np.abs(lr_diag)))

        # gamma = mu(reduced) - 2*mu(neutral) + mu(oxidized) -- 3rd derivative
        # of E vs N (centered finite difference, exact on the 3-point grid).
        mu_neutral = cdft["mu"]
        mu_red = pred["states"]["reduced_sp"]["cdft"]["mu"]
        mu_ox = pred["states"]["oxidized_sp"]["cdft"]["mu"]
        gamma_val = mu_red - 2.0 * mu_neutral + mu_ox

        d = {
            "mu": mu_neutral,
            "eta": cdft["eta"],
            "gamma": gamma_val,
            "fplus_max": fplus_g, "fplus_max_O": fplus_O,
            "fminus_max": fminus_g, "fminus_max_O": fminus_O,
            "f2_max": f2_g, "f2_max_O": f2_O,
            "qbader_max": qb_g, "qbader_max_O": qb_O,
            "lr_diag_max": lr_diag_max,
            "Ee_neutral_pred": ns["thermo"]["Ee"],
        }

        dft_neutral = dft["states"].get("neutral_dzp") or dft["states"].get("neutral")
        d["Ee_neutral_dft"] = dft_neutral["thermo"]["Ee"] if dft_neutral else float("nan")

        pred_derived = pred.get("derived") or {}
        d["lvc2_red"] = (pred_derived.get("E_relax_LVC2_red") or {}).get("ha", float("nan"))
        d["lvc2_ox"] = (pred_derived.get("E_relax_LVC2_ox") or {}).get("ha", float("nan"))

        # Composite TZ2P targets (cohérence de base TZ2P, cf. rapport p1125):
        #   EA = Ee(neutral_tz2p) - Ee(reduced_sp)        [reduced_sp = SP TZ2P @ neutre geom]
        #   IP = Ee(oxidized_sp) - Ee(neutral_tz2p)
        #   Erelax = Ee(ion_tz2p) - Ee(ion_sp)            [TZ2P/TZ2P pure deformation]
        states = dft["states"]
        Ee_n_tz = states["neutral_tz2p"]["thermo"]["Ee"]
        Ee_r_sp = states["reduced_sp"]["thermo"]["Ee"]
        Ee_o_sp = states["oxidized_sp"]["thermo"]["Ee"]
        Ee_r_tz = states["reduced_tz2p"]["thermo"]["Ee"]
        Ee_o_tz = states["oxidized_tz2p"]["thermo"]["Ee"]
        d["EA"] = Ee_n_tz - Ee_r_sp
        d["IP"] = Ee_o_sp - Ee_n_tz
        d["Erelax_red"] = Ee_r_tz - Ee_r_sp
        d["Erelax_ox"] = Ee_o_tz - Ee_o_sp

        # ddU = pure thermal-correction differential: (ZPE+thermal)_ion - (ZPE+thermal)_neutral.
        # The JSON 'derived["ddU"]' is (U-Ee)_diff which collapses to -ΔEe + ΔU_thermal,
        # i.e. dominated by -ΔEe ≈ EA/IP. The thermo-only piece is just U_ion - U_neutral.
        U_n = states["neutral_dzp"]["thermo"].get("U")
        U_ro = states["reduced_opt"]["thermo"].get("U")
        U_oo = states["oxidized_opt"]["thermo"].get("U")
        if U_n is not None and U_ro is not None and U_oo is not None:
            d["ddU_red"] = U_ro - U_n
            d["ddU_ox"] = U_oo - U_n
        else:
            d["ddU_red"] = float("nan")
            d["ddU_ox"] = float("nan")

        # TdS already has the correct thermo-only convention in derived
        derived = dft["derived"]
        d["TdS_red"] = derived["TdS_red"]["ha"]
        d["TdS_ox"] = derived["TdS_ox"]["ha"]

    except (KeyError, IndexError, ValueError) as e:
        print(f"  ! skip {pred_json.parent.name}: missing field {e}", file=sys.stderr)
        return None

    return d


def main() -> None:
    rows = []
    for solvent_dir in sorted(DFT_DIR.iterdir()):
        if not solvent_dir.is_dir():
            continue
        solvent = solvent_dir.name
        pred_solvent = PRED_DIR / solvent
        if not pred_solvent.is_dir():
            continue

        for mol_dir in sorted(solvent_dir.iterdir()):
            if not mol_dir.is_dir():
                continue
            mol = mol_dir.name
            pred_mol = pred_solvent / mol
            if not pred_mol.is_dir():
                continue

            for dft_json in sorted(mol_dir.glob(f"{mol}_conf*_DFT.json")):
                conf = dft_json.stem.replace(f"{mol}_conf", "").replace("_DFT", "")
                pred_json = pred_mol / f"{mol}_conf{conf}_prediction.json"
                if not pred_json.exists():
                    continue
                d = load_pair(pred_json, dft_json)
                if d is None:
                    continue
                d["solvent"] = solvent
                d["molecule"] = mol
                d["conformer"] = int(conf)
                rows.append(d)

        print(f"  {solvent}: {sum(1 for r in rows if r['solvent'] == solvent)} rows")

    df = pd.DataFrame(rows)
    cols = (
        ["solvent", "molecule", "conformer", "Ee_neutral_dft", "Ee_neutral_pred"]
        + DESCRIPTOR_COLS + TARGET_COLS
    )
    df = df[cols]
    print(f"\nBefore dedup: {len(df)} rows")

    # Water == Chloroform bug filter: half of Water DFT targets are byte-identical
    # copies of Chloroform's. Detect via DFT-side Ee_neutral and drop Water rows.
    if {"Water", "Chloroform"} <= set(df.solvent.unique()):
        wc = df[df.solvent.isin(["Water", "Chloroform"])].pivot_table(
            index=["molecule", "conformer"], columns="solvent",
            values="Ee_neutral_dft", aggfunc="first",
        )
        dup = wc[(wc["Water"] - wc["Chloroform"]).abs() < 1e-8].dropna().index
        if len(dup) > 0:
            keys = df.set_index(["molecule", "conformer"]).index
            mask = ~((df.solvent == "Water") & keys.isin(dup))
            df = df.loc[mask].reset_index(drop=True)
            print(f"After Water==Chloroform dedup: {len(df)} rows ({len(dup)} Water entries dropped)")

    df = df.dropna(subset=DESCRIPTOR_COLS + TARGET_COLS).reset_index(drop=True)
    print(f"After NaN drop:    {len(df)} rows")

    # Filter EA < 0 (vertical electron affinity going negative is unphysical:
    # the implicit solvent fails to stabilize the anion). Cf. rapport p1125.
    n_neg = (df.EA < 0).sum()
    if n_neg > 0:
        df = df[df.EA >= 0].reset_index(drop=True)
        print(f"After EA<0 drop:  {len(df)} rows ({n_neg} dropped)")

    # Filter EA < 0 (vertical electron affinity going negative is unphysical:
    # the implicit solvent fails to stabilize the anion). Cf. rapport p1125.
    n_neg = (df.EA < 0).sum()
    if n_neg > 0:
        df = df[df.EA >= 0].reset_index(drop=True)
        print(f"After EA<0 drop:   {len(df)} rows ({n_neg} dropped)")

    # HOMO/LUMO via bijection with (mu, eta): mu = (H+L)/2, eta = L-H
    df["HOMO"] = df["mu"] - df["eta"] / 2.0
    df["LUMO"] = df["mu"] + df["eta"] / 2.0

    out_pq = OUT_DIR / "merged_dataset.parquet"
    out_csv = OUT_DIR / "merged_dataset.csv"
    df.to_parquet(out_pq, index=False)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_pq}")
    print(f"Saved -> {out_csv}")
    print(f"\nShape: {df.shape}")
    print(f"Solvents: {sorted(df.solvent.unique())}")
    print(f"Molecules: {df.molecule.nunique()}")
    print(f"Conformers per (mol,solv): mean={df.groupby(['molecule', 'solvent']).size().mean():.1f}")


if __name__ == "__main__":
    main()
