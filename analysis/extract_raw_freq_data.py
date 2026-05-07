#!/usr/bin/env amspython
"""
Extract raw frequency + IR intensity data from all neutral RKF files
and save to a JSON file for downstream analysis.
"""

import json
import re
from pathlib import Path
import numpy as np
from scm.plams import KFReader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = _PROJECT_ROOT / "data" / "1594174" / "dir_workdir"
MOLECULES = [
    "Al-CA", "CA", "Cl-DA", "Cl-HMC",
    "DA", "DHI", "F-DA", "F-HMC", "H2C-CA", "HMC", "HO-CA",
]
OUTPUT = _PROJECT_ROOT / "data" / "raw_freq_data.json"


def read_boltzmann_weights(mol_name):
    """Read Boltzmann weights from conformers_summary.txt."""
    summary = BASE / f"{mol_name}_workdir" / "conformers" / "results" / "conformers_summary.txt"
    if not summary.exists():
        return {}
    weights = {}
    in_table = False
    for line in summary.read_text().splitlines():
        if "ΔE [kJ/mol]" in line and "Poids Boltzmann" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("---"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    conf_num = int(parts[0])
                    weight = float(parts[2])
                    weights[conf_num] = weight
                except ValueError:
                    in_table = False
            else:
                in_table = False
    return weights


def extract_from_rkf(rkf_path):
    """Extract frequencies, IR intensities from a single RKF."""
    kf = KFReader(str(rkf_path))
    freqs = list(np.array(kf.read("Vibrations", "Frequencies[cm-1]"), dtype=float))

    ir = None
    for key in ["Intensities[km/mol]", "Intensities", "IR Intensities"]:
        try:
            ir = list(np.array(kf.read("Vibrations", key), dtype=float))
            break
        except Exception:
            continue

    return {"frequencies": freqs, "ir_intensities": ir}


data = {}
for mol in MOLECULES:
    plams_dir = BASE / f"{mol}_workdir" / "redox" / "calculations" / "plams_workdir"
    if not plams_dir.exists():
        print(f"  SKIP {mol}: no plams_workdir")
        continue

    pat = re.compile(rf"^{re.escape(mol)}_conf_(\d+)_neutral(?:\.\d+)?$")
    conformers = {}
    for d in sorted(plams_dir.iterdir()):
        m = pat.match(d.name)
        if m:
            conf_num = int(m.group(1))
            rkf = d / "adf.rkf"
            if rkf.exists():
                conformers[conf_num] = str(rkf)

    weights = read_boltzmann_weights(mol)

    mol_data = {"conformers": {}, "boltzmann_weights": {str(k): v for k, v in weights.items()}}
    for cn, rkf_path in sorted(conformers.items()):
        try:
            mol_data["conformers"][str(cn)] = extract_from_rkf(rkf_path)
            print(f"  {mol} conf_{cn}: {len(mol_data['conformers'][str(cn)]['frequencies'])} modes")
        except Exception as e:
            print(f"  ERROR {mol} conf_{cn}: {e}")

    data[mol] = mol_data

OUTPUT.write_text(json.dumps(data, indent=1))
print(f"\nSaved to {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
