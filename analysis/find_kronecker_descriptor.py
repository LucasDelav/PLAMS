#!/usr/bin/env amspython
"""
Analyse des orbitales virtuelles pour trouver le meilleur descripteur Kronecker
séparant les molécules DEA (rupture C-X) des molécules stables.

Candidats testés :
  1. LUMO_hal_frac             : fraction halogène dans la LUMO seule
  2. max_hal_frac_5            : max fraction halogène dans LUMO..LUMO+4
  3. sigma_star_energy_gap     : E(sigma*_CX) - E(LUMO) en eV
  4. sigma_star_hal_frac       : fraction halogène de l'orbitale sigma*(C-X)
  5. weighted_hal_accessibility: hal_frac / (1 + delta_E) pour l'orbitale sigma*

Pour chaque descripteur, on détermine le seuil optimal qui sépare
parfaitement DEA de stable (si possible).
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.constants import EH_TO_EV

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Acetonitrile_marais"

DEA_SET = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}
HALOGEN_Z = {9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}
N_VIRTUAL_SCAN = 15  # Scan LUMO through LUMO+14


def extract_virtual_orbital_composition(rkf_path, n_virtual=N_VIRTUAL_SCAN):
    """Extract element-wise composition of the first n_virtual unoccupied MOs.

    Returns list of dicts, one per virtual MO, with keys:
      mo_index, energy_eV, O_frac, C_frac, hal_frac, N_frac, H_frac,
      per_atom_fracs (list), halogen_atom_indices
    """
    from scm.plams import KFReader
    kf = KFReader(str(rkf_path))

    n_mo = kf.read('A', 'nmo_A')
    occupations = np.array(kf.read('A', 'froc_A'), dtype=float)
    energies = np.array(kf.read('A', 'eps_A'), dtype=float)
    n_occ = int(np.sum(occupations > 0.5))

    n_atoms = kf.read('Molecule', 'nAtoms')
    atomic_numbers = list(kf.read('Molecule', 'AtomicNumbers'))

    n_sfo = kf.read('SFOs', 'number')
    sfo_fragments = list(kf.read('SFOs', 'fragment'))  # 1-indexed

    n_contrib = list(kf.read('SFO popul', 'number of contributions'))
    sfo_pop_idx = list(kf.read('SFO popul', 'sfo_index'))
    sfo_pop_val = list(kf.read('SFO popul', 'sfo_pop'))

    # Find halogen atom indices
    halogen_atom_indices = [i for i, z in enumerate(atomic_numbers) if z in HALOGEN_Z]

    results = []
    lumo_eV = energies[n_occ] * EH_TO_EV

    # Walk through all MOs to find offsets
    offset = 0
    for mo_idx in range(n_mo):
        nc = n_contrib[mo_idx]

        # Only process virtual MOs from LUMO to LUMO+(n_virtual-1)
        virt_idx = mo_idx - n_occ  # 0 = LUMO, 1 = LUMO+1, etc.
        if 0 <= virt_idx < n_virtual:
            atom_weights = np.zeros(n_atoms)
            idx_slice = sfo_pop_idx[offset:offset + nc]
            val_slice = sfo_pop_val[offset:offset + nc]

            for sfo_idx, pop_val in zip(idx_slice, val_slice):
                if 1 <= sfo_idx <= n_sfo:
                    atom_idx = sfo_fragments[sfo_idx - 1] - 1
                    if 0 <= atom_idx < n_atoms:
                        atom_weights[atom_idx] += abs(pop_val)

            total = np.sum(atom_weights)
            if total > 0:
                fracs = atom_weights / total
            else:
                fracs = atom_weights

            O_frac = sum(fracs[i] for i in range(n_atoms) if atomic_numbers[i] == 8)
            C_frac = sum(fracs[i] for i in range(n_atoms) if atomic_numbers[i] == 6)
            hal_frac = sum(fracs[i] for i in range(n_atoms) if atomic_numbers[i] in HALOGEN_Z)
            N_frac = sum(fracs[i] for i in range(n_atoms) if atomic_numbers[i] == 7)
            H_frac = sum(fracs[i] for i in range(n_atoms) if atomic_numbers[i] == 1)

            mo_eV = energies[mo_idx] * EH_TO_EV

            results.append({
                'virt_index': virt_idx,  # 0=LUMO, 1=LUMO+1, ...
                'energy_eV': mo_eV,
                'delta_eV': mo_eV - lumo_eV,  # gap from LUMO
                'O_frac': O_frac,
                'C_frac': C_frac,
                'hal_frac': hal_frac,
                'N_frac': N_frac,
                'H_frac': H_frac,
            })

        offset += nc

    return results, lumo_eV, halogen_atom_indices, atomic_numbers


def find_best_conformer_rkf(workdir, mol_name):
    """Find the neutral RKF of the conformer with highest Boltzmann weight.

    Searches directly in conformers/calculations/03_dft_opt/ to avoid
    dependency on neutral_rkf_paths.json (which may have stale paths).
    """
    import re

    # Find all neutral RKFs
    dft_dir = workdir / "conformers" / "calculations" / "03_dft_opt"
    neutral_rkfs = {}

    if dft_dir.exists():
        for conf_dir in dft_dir.iterdir():
            if not conf_dir.is_dir():
                continue
            m = re.match(r'conf_(\d+)', conf_dir.name)
            if m:
                conf_num = int(m.group(1))
                rkf = conf_dir / "plams_workdir" / f"conf_{conf_num}_dft_opt" / "adf.rkf"
                if rkf.exists():
                    neutral_rkfs[conf_num] = rkf

    if not neutral_rkfs:
        return None

    # Read Boltzmann weights
    summary_path = workdir / "conformers" / "results" / "conformers_summary.txt"
    weights = {}
    if summary_path.exists():
        with open(summary_path) as f:
            for line in f:
                m = re.match(r'\s+(\d+)\s+[\d.]+\s+([\d.]+)', line)
                if m:
                    weights[int(m.group(1))] = float(m.group(2))

    best = max(neutral_rkfs.keys(), key=lambda c: weights.get(c, 0))
    return neutral_rkfs[best]


def compute_descriptors(mol_name, virt_orbitals, halogen_indices):
    """Compute candidate Kronecker descriptors from virtual orbital data."""
    has_halogen = len(halogen_indices) > 0

    # Descriptor 1: LUMO hal_frac
    lumo_hal = virt_orbitals[0]['hal_frac'] if virt_orbitals else 0.0

    # Descriptor 2: max hal_frac in LUMO..LUMO+4
    max_hal_5 = max((v['hal_frac'] for v in virt_orbitals[:5]), default=0.0)

    # Descriptor 3: max hal_frac in LUMO..LUMO+9
    max_hal_10 = max((v['hal_frac'] for v in virt_orbitals[:10]), default=0.0)

    # Descriptor 4: sigma*(C-X) identification
    # Find the virtual MO with highest halogen character
    sigma_star = None
    if has_halogen:
        best_hal = max(virt_orbitals, key=lambda v: v['hal_frac'])
        if best_hal['hal_frac'] > 0.05:
            sigma_star = best_hal

    sigma_star_gap = sigma_star['delta_eV'] if sigma_star else float('inf')
    sigma_star_hal = sigma_star['hal_frac'] if sigma_star else 0.0
    sigma_star_idx = sigma_star['virt_index'] if sigma_star else -1

    # Descriptor 5: weighted accessibility = hal_frac / (1 + delta_E)
    if sigma_star and sigma_star['delta_eV'] >= 0:
        weighted_access = sigma_star_hal / (1.0 + sigma_star['delta_eV'])
    elif sigma_star:
        weighted_access = sigma_star_hal * (1.0 + abs(sigma_star['delta_eV']))
    else:
        weighted_access = 0.0

    # Descriptor 6: cumulative hal_frac weighted by 1/(1+delta_E) across all virtual MOs
    cumul_weighted_hal = 0.0
    for v in virt_orbitals:
        if v['delta_eV'] >= 0:
            cumul_weighted_hal += v['hal_frac'] / (1.0 + v['delta_eV'])
        else:
            cumul_weighted_hal += v['hal_frac'] * (1.0 + abs(v['delta_eV']))

    return {
        'molecule': mol_name,
        'has_halogen': has_halogen,
        'is_dea': mol_name in DEA_SET,
        'lumo_hal_frac': lumo_hal,
        'max_hal_5': max_hal_5,
        'max_hal_10': max_hal_10,
        'sigma_star_gap_eV': sigma_star_gap,
        'sigma_star_hal_frac': sigma_star_hal,
        'sigma_star_virt_idx': sigma_star_idx,
        'weighted_access': weighted_access,
        'cumul_weighted_hal': cumul_weighted_hal,
    }


def find_optimal_threshold(values, labels, higher_means_dea=True):
    """Find the threshold that best separates DEA (True) from stable (False).

    Returns (threshold, perfect_separation, n_misclassified).
    """
    combined = sorted(zip(values, labels), key=lambda x: x[0])
    n = len(combined)
    best_threshold = None
    best_errors = n

    for i in range(n + 1):
        if i == 0:
            t = combined[0][0] - 0.001
        elif i == n:
            t = combined[-1][0] + 0.001
        else:
            t = (combined[i-1][0] + combined[i][0]) / 2

        if higher_means_dea:
            errors = sum(1 for v, l in combined if (v >= t) != l)
        else:
            errors = sum(1 for v, l in combined if (v <= t) != l)

        if errors < best_errors:
            best_errors = errors
            best_threshold = t

    return best_threshold, best_errors == 0, best_errors


def main():
    # Find all catechol workdirs
    quinones = {
        'CQ', 'HMQ', 'Al-CQ', 'HO-CQ', 'H2C-CQ',
        'Cl-HMQ', 'F-HMQ', 'Br-HMQ', 'I-HMQ',
        'DOPA', 'Cl-DOPA', 'F-DOPA', 'Br-DOPA', 'I-DOPA', 'L-DOPA',
        'O-IQ', 'P-IQ', 'S-IQ', 'IQ',
    }

    molecules = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.endswith('_workdir'):
            mol = d.name.replace('_workdir', '')
            if mol not in quinones:
                molecules.append((mol, d))

    print(f"Found {len(molecules)} catechol molecules\n")

    # Extract orbital data
    all_descriptors = []

    for mol_name, workdir in molecules:
        rkf = find_best_conformer_rkf(workdir, mol_name)
        if rkf is None:
            print(f"  [!] {mol_name}: no neutral RKF found")
            continue

        try:
            virt_orbs, lumo_eV, hal_idx, at_nums = extract_virtual_orbital_composition(rkf)
        except Exception as e:
            print(f"  [!] {mol_name}: extraction failed - {e}")
            continue

        halogens = [HALOGEN_Z[at_nums[i]] for i in hal_idx]
        desc = compute_descriptors(mol_name, virt_orbs, hal_idx)
        desc['lumo_eV'] = lumo_eV
        desc['halogens'] = ','.join(halogens) if halogens else '-'
        all_descriptors.append(desc)

        tag = "DEA" if desc['is_dea'] else "   "
        print(f"  {tag} {mol_name:>10}  hal={desc['halogens']:>4}  "
              f"LUMO_hal={desc['lumo_hal_frac']:.3f}  "
              f"max5={desc['max_hal_5']:.3f}  max10={desc['max_hal_10']:.3f}  "
              f"σ*_gap={desc['sigma_star_gap_eV']:+.2f}eV  "
              f"σ*_hal={desc['sigma_star_hal_frac']:.3f}  "
              f"σ*_idx=LUMO+{desc['sigma_star_virt_idx']}  "
              f"w_access={desc['weighted_access']:.3f}  "
              f"cumul={desc['cumul_weighted_hal']:.3f}")

    # Analyse each descriptor
    print(f"\n{'='*70}")
    print("ANALYSE DES DESCRIPTEURS KRONECKER")
    print(f"{'='*70}\n")

    descriptors_to_test = [
        ('lumo_hal_frac', 'LUMO hal_frac', True),
        ('max_hal_5', 'Max hal_frac (LUMO..+4)', True),
        ('max_hal_10', 'Max hal_frac (LUMO..+9)', True),
        ('sigma_star_hal_frac', 'σ*(C-X) hal_frac', True),
        ('sigma_star_gap_eV', 'σ*(C-X) gap from LUMO (eV)', False),  # lower gap = more DEA
        ('weighted_access', 'Weighted accessibility', True),
        ('cumul_weighted_hal', 'Cumulative weighted hal', True),
    ]

    for key, label, higher_is_dea in descriptors_to_test:
        values = [d[key] for d in all_descriptors]
        labels = [d['is_dea'] for d in all_descriptors]

        # Replace inf with a large value for threshold search
        max_finite = max((v for v in values if v != float('inf')), default=100)
        values_clean = [v if v != float('inf') else max_finite + 10 for v in values]

        threshold, perfect, n_errors = find_optimal_threshold(
            values_clean, labels, higher_means_dea=higher_is_dea)

        status = "PARFAIT" if perfect else f"{n_errors} erreur(s)"
        direction = ">=" if higher_is_dea else "<="

        print(f"  {label:35s}  seuil={threshold:.4f}  {direction}  [{status}]")

        # Show values for DEA vs stable
        dea_vals = [v for v, d in zip(values_clean, all_descriptors) if d['is_dea']]
        stable_vals = [v for v, d in zip(values_clean, all_descriptors) if not d['is_dea']]
        print(f"    DEA   : {sorted(dea_vals)}")
        print(f"    Stable: {sorted(stable_vals)}")
        if dea_vals and stable_vals:
            gap = min(dea_vals) - max(stable_vals) if higher_is_dea else min(stable_vals) - max(dea_vals)
            print(f"    Marge de séparation: {gap:.4f}")
        print()

    # Save results
    out_path = PROJECT_ROOT / "rapport" / "data" / "kronecker_descriptors.json"
    with open(out_path, 'w') as f:
        json.dump(all_descriptors, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
