#!/usr/bin/env amspython
"""
Extract LVC2 E_def values for the 4 DEA molecules from lvc_sp_dea/ RKF files.

Computes Boltzmann-averaged LVC2 for reduction and oxidation, then prints
results for CSV integration.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmanalysis.lvc_analysis import compute_lvc2_edef_from_rkf

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LVC_SP_DEA_DIR = PROJECT_ROOT / "lvc_sp_dea"

DEA_MOLECULES = {
    'Br-DA':  {'n_conf': 4, 'weights': {1: 0.252160, 2: 0.314381, 3: 0.201866, 4: 0.231593}},
    'Br-HMC': {'n_conf': 3, 'weights': {1: 0.455717, 2: 0.240447, 3: 0.303837}},
    'I-DA':   {'n_conf': 3, 'weights': {1: 0.591242, 2: 0.118179, 3: 0.214579}},
    'I-HMC':  {'n_conf': 2, 'weights': {1: 0.334684, 2: 0.665316}},
}


def find_sp_freq_rkf(workdir, mol_name, conf_num, state):
    """Find the sp_freq RKF file for a given conformer and state."""
    plams_dir = workdir / "redox" / "calculations" / "plams_workdir"
    dirname = f"{mol_name}_conf_{conf_num}_{state}_sp_freq"

    # Check ams.results/adf.rkf first (new format)
    rkf = plams_dir / dirname / "ams.results" / "adf.rkf"
    if rkf.exists():
        return rkf
    # Fallback to adf.rkf directly
    rkf = plams_dir / dirname / "adf.rkf"
    if rkf.exists():
        return rkf
    return None


def main():
    results = {}

    for mol_name, info in DEA_MOLECULES.items():
        workdir = LVC_SP_DEA_DIR / f"{mol_name}_workdir"
        weights = info['weights']

        print(f"\n{'='*60}")
        print(f"  {mol_name}")
        print(f"{'='*60}")

        lvc2_red_per_conf = {}
        lvc2_ox_per_conf = {}

        for conf_num in sorted(weights.keys()):
            w = weights[conf_num]

            # Reduction
            rkf_red = find_sp_freq_rkf(workdir, mol_name, conf_num, 'reduced')
            if rkf_red:
                res_red = compute_lvc2_edef_from_rkf(rkf_red)
                if res_red['success']:
                    lvc2_red_per_conf[conf_num] = res_red['E_def_lvc2_kj']
                    print(f"  conf_{conf_num} (w={w:.4f}): LVC2_red = {res_red['E_def_lvc2_kj']:.3f} kJ/mol "
                          f"(n_imag={res_red['n_imag']}, n_modes={res_red['n_modes_used']})")
                else:
                    print(f"  conf_{conf_num}: FAILED reduction - {res_red['error']}")
            else:
                print(f"  conf_{conf_num}: reduced_sp_freq RKF not found")

            # Oxidation
            rkf_ox = find_sp_freq_rkf(workdir, mol_name, conf_num, 'oxidized')
            if rkf_ox:
                res_ox = compute_lvc2_edef_from_rkf(rkf_ox)
                if res_ox['success']:
                    lvc2_ox_per_conf[conf_num] = res_ox['E_def_lvc2_kj']
                    print(f"  conf_{conf_num} (w={w:.4f}): LVC2_ox  = {res_ox['E_def_lvc2_kj']:.3f} kJ/mol "
                          f"(n_imag={res_ox['n_imag']}, n_modes={res_ox['n_modes_used']})")
                else:
                    print(f"  conf_{conf_num}: FAILED oxidation - {res_ox['error']}")
            else:
                print(f"  conf_{conf_num}: oxidized_sp_freq RKF not found")

        # Boltzmann average
        avg_red = None
        if lvc2_red_per_conf:
            total_w = sum(weights[c] for c in lvc2_red_per_conf)
            avg_red = sum(weights[c] * lvc2_red_per_conf[c] for c in lvc2_red_per_conf) / total_w

        avg_ox = None
        if lvc2_ox_per_conf:
            total_w = sum(weights[c] for c in lvc2_ox_per_conf)
            avg_ox = sum(weights[c] * lvc2_ox_per_conf[c] for c in lvc2_ox_per_conf) / total_w

        results[mol_name] = {
            'lvc2_red': avg_red,
            'lvc2_ox': avg_ox,
            'per_conf_red': lvc2_red_per_conf,
            'per_conf_ox': lvc2_ox_per_conf,
        }

        print(f"\n  Boltzmann-averaged LVC2:")
        print(f"    Reduction: {avg_red:.3f} kJ/mol" if avg_red else "    Reduction: N/A")
        print(f"    Oxidation: {avg_ox:.3f} kJ/mol" if avg_ox else "    Oxidation: N/A")

    # Output JSON for easy integration
    print(f"\n{'='*60}")
    print("  JSON OUTPUT (for CSV integration)")
    print(f"{'='*60}")

    json_out = {}
    for mol, data in results.items():
        json_out[mol] = {
            'lvc2_red': data['lvc2_red'],
            'lvc2_ox': data['lvc2_ox'],
        }

    print(json.dumps(json_out, indent=2))

    # Save to file
    out_path = PROJECT_ROOT / "rapport" / "data" / "dea_lvc2_values.json"
    with open(out_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
