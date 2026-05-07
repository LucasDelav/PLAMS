#!/usr/bin/env python3
"""Build the frequency-distribution descriptor parquet from predict-dg JSONs.

Reads dataset_predict_dg/<solvent>/<molecule>/<mol>_conf<n>_prediction.json
for all (solvent, molecule, conformer) and computes the full descriptor
banks A-G (see freq_descriptors_lib.py).

Output:
    analysis/formula_search/data/freq_descriptors_full.parquet

One row per (solvent, molecule, conformer). Designed to be merged onto
merged_dataset.parquet via the (solvent, molecule, conformer) key.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freq_descriptors_lib import (  # noqa: E402
    descriptors_freqset,
    descriptors_delta,
    descriptors_paired_rrho,
    pair_by_energy,
    pair_by_overlap,
)

REPO = Path(__file__).resolve().parents[2]
PRED_DIR = REPO / "dataset_predict_dg"
OUT_DIR = REPO / "analysis" / "formula_search" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "freq_descriptors_full.parquet"


def _extract_modes(state_modes):
    """Pull signed frequency array (cm⁻¹) and displacement list.

    Imaginary modes are stored with a negative frequency; we keep them so
    that the 3N−6 mode count stays constant across the three states. The
    distribution descriptors use |ω| for RRHO sums (Gaussian/AMS default
    convention) and expose a separate `n_imag` counter.
    """
    om, disp = [], []
    for m in state_modes:
        f = m.get("frequency_cm1")
        if f is None:
            continue
        om.append(float(f))
        disp.append(m.get("displacements"))
    return np.array(om, dtype=float), disp


def process_json(json_path: Path) -> dict | None:
    try:
        with open(json_path) as fh:
            d = json.load(fh)
    except Exception:
        return None

    meta = d.get("metadata", {})
    states = d.get("states", {})
    needed = ("neutral_sp", "reduced_sp", "oxidized_sp")
    if not all(s in states for s in needed):
        return None

    try:
        om_n, disp_n = _extract_modes(states["neutral_sp"]["vibrations"]["modes"])
        om_r, disp_r = _extract_modes(states["reduced_sp"]["vibrations"]["modes"])
        om_o, disp_o = _extract_modes(states["oxidized_sp"]["vibrations"]["modes"])
    except (KeyError, TypeError):
        return None
    if om_n.size == 0 or om_r.size == 0 or om_o.size == 0:
        return None

    # Use |ω| for distribution and RRHO descriptors (treats imaginary as a soft
    # vibrational mode; standard convention). Keep signed arrays for pairing
    # by overlap of eigenvectors and for the imaginary-count diagnostic.
    am_n, am_r, am_o = np.abs(om_n), np.abs(om_r), np.abs(om_o)

    row = {
        "solvent": meta.get("solvent"),
        "molecule": meta.get("molecule"),
        "conformer": int(meta.get("conformer", 0)),
        "n_modes": int(om_n.size),
        "n_imag_n": int((om_n < 0).sum()),
        "n_imag_r": int((om_r < 0).sum()),
        "n_imag_o": int((om_o < 0).sum()),
    }

    # Single-set descriptors (use |ω|)
    row.update(descriptors_freqset(am_n, "wn"))
    row.update(descriptors_freqset(am_r, "wr"))
    row.update(descriptors_freqset(am_o, "wo"))

    # Pairing A: by energy order on |ω|
    if am_n.size == am_r.size:
        d_red, on_p, or_p = pair_by_energy(am_n, am_r)
        row.update(descriptors_delta(d_red, "dwr_A"))
        row.update(descriptors_paired_rrho(on_p, or_p, "dRr_A"))
    if am_n.size == am_o.size:
        d_ox, on_p, oo_p = pair_by_energy(am_n, am_o)
        row.update(descriptors_delta(d_ox, "dwo_A"))
        row.update(descriptors_paired_rrho(on_p, oo_p, "dRo_A"))

    # Pairing B: eigenvector overlap (Hungarian); pair on |ω|
    can_pair_B = (
        om_n.size == om_r.size == om_o.size
        and all(d is not None for d in disp_n + disp_r + disp_o)
        and len(disp_n[0]) > 0
    )
    if can_pair_B:
        try:
            d_red_B, on_p, or_p, _ = pair_by_overlap(disp_n, am_n, disp_r, am_r)
            row.update(descriptors_delta(d_red_B, "dwr_B"))
            row.update(descriptors_paired_rrho(on_p, or_p, "dRr_B"))
            d_ox_B, on_p, oo_p, _ = pair_by_overlap(disp_n, am_n, disp_o, am_o)
            row.update(descriptors_delta(d_ox_B, "dwo_B"))
            row.update(descriptors_paired_rrho(on_p, oo_p, "dRo_B"))
        except Exception as e:
            row["_pairB_error"] = str(e)[:80]

    return row


def _worker(path_str):
    return process_json(Path(path_str))


def main():
    json_paths = sorted(PRED_DIR.glob("*/*/*_conf*_prediction.json"))
    print(f"Found {len(json_paths)} predict-dg JSONs in {PRED_DIR}")

    t0 = time.time()
    n_proc = min(cpu_count(), 16)
    rows = []
    skipped = 0
    with Pool(n_proc) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, [str(p) for p in json_paths], chunksize=8)):
            if row is None:
                skipped += 1
            else:
                rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"  processed {i + 1}/{len(json_paths)} (kept {len(rows)})", flush=True)

    print(f"\nProcessed {len(json_paths)} files in {time.time() - t0:.1f}s")
    print(f"  Kept:    {len(rows)}")
    print(f"  Skipped: {skipped}")

    df = pd.DataFrame(rows).sort_values(["solvent", "molecule", "conformer"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH)
    print(f"\n+ Saved {OUT_PATH}")
    print(f"  shape: {df.shape}")

    # Family summary
    families = {
        "metadata": ["solvent", "molecule", "conformer", "n_modes_n", "n_modes_r", "n_modes_o"],
        "wn_*": [c for c in df.columns if c.startswith("wn_")],
        "wr_*": [c for c in df.columns if c.startswith("wr_")],
        "wo_*": [c for c in df.columns if c.startswith("wo_")],
        "dwr_A_*": [c for c in df.columns if c.startswith("dwr_A_")],
        "dwo_A_*": [c for c in df.columns if c.startswith("dwo_A_")],
        "dwr_B_*": [c for c in df.columns if c.startswith("dwr_B_")],
        "dwo_B_*": [c for c in df.columns if c.startswith("dwo_B_")],
        "dRr_A_* (paired RRHO red, energy)": [c for c in df.columns if c.startswith("dRr_A_")],
        "dRo_A_* (paired RRHO ox, energy)":  [c for c in df.columns if c.startswith("dRo_A_")],
        "dRr_B_* (paired RRHO red, overlap)": [c for c in df.columns if c.startswith("dRr_B_")],
        "dRo_B_* (paired RRHO ox, overlap)":  [c for c in df.columns if c.startswith("dRo_B_")],
    }
    print("\nColumn families:")
    for k, v in families.items():
        print(f"  {k:50s} : {len(v)}")

    print("\nPer-solvent counts:")
    print(df.groupby("solvent").size())


if __name__ == "__main__":
    main()
