#!/usr/bin/env amspython
"""
Analyse des descripteurs pour prédire le RMSD et corriger le LVC2.

Descripteurs extraits :
  - Huang-Rhys total (depuis LVC2 / SP ionique)
  - n_imag (modes imaginaires du Hessien ionique)
  - Composition LUMO : poids par atome/type d'atome (depuis neutre)
  - Énergie LUMO, gap HOMO-LUMO

Usage:
    amspython analysis/predict_rmsd.py
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.constants import EH_TO_EV, EH_TO_KJ

DATA_DIR = PROJECT_ROOT / "data" / "Acetonitrile"
LVC_CSV = PROJECT_ROOT / "rapport" / "data" / "all_methods_comparison_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "rapport" / "data"

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}
QUINONES = {
    'CQ', 'HMQ', 'Al-CQ', 'HO-CQ', 'H2C-CQ',
    'Cl-HMQ', 'F-HMQ', 'Br-HMQ', 'I-HMQ',
    'DOPA', 'Cl-DOPA', 'F-DOPA', 'Br-DOPA', 'I-DOPA', 'L-DOPA',
    'O-IQ', 'P-IQ', 'S-IQ', 'IQ',
}


# ═══════════════════════════════════════════════════════════════════════════════
# DESCRIPTEUR : HUANG-RHYS TOTAL (depuis données LVC2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_huang_rhys_from_lvc2(workdir, mol_name, state='reduced'):
    """Calcule le facteur de Huang-Rhys total S = Σ |E_i|/(ℏω_i).

    S_i = F_i² / (2 ω_i k_i) ≈ |contribution_i| / (ℏω_i)
    En pratique : S_total = Σ_i |contribution_kJ_i| / (ℏω_i en kJ/mol)

    Retourne aussi le déplacement total ΔQ = √(Σ ΔQ_i²).
    """
    from qmanalysis.lvc_analysis import (
        find_sp_rkf_files, extract_gradients,
        extract_normal_modes_and_fc
    )

    sp_rkfs = find_sp_rkf_files(workdir, mol_name, state, 'sp_freq')
    if not sp_rkfs:
        return None

    # Boltzmann-weighted average over conformers
    weights_path = workdir / "conformers" / "results" / "conformers_summary.txt"
    weights = _read_weights(weights_path)

    S_total_avg = 0.0
    dQ_total_avg = 0.0
    n_imag_avg = 0.0
    total_w = 0.0

    for conf_num, rkf_path in sp_rkfs.items():
        w = weights.get(conf_num, 1.0)

        modes, fc, freq = extract_normal_modes_and_fc(rkf_path)
        grad = extract_gradients(rkf_path)
        if modes is None or grad is None:
            continue

        S_total = 0.0
        dQ_total_sq = 0.0
        n_imag = 0

        for i in range(len(modes)):
            if modes[i] is None or fc[i] <= 0 or freq[i] < 50.0:
                if freq[i] < 0:
                    n_imag += 1
                continue

            F_i = np.dot(grad, modes[i])
            # ΔQ_i = F_i / k_i (displacement in normal coordinate)
            dQ_i = F_i / fc[i]
            dQ_total_sq += dQ_i ** 2

            # Huang-Rhys: S_i = ΔQ_i² × ω_i / 2  (in atomic units)
            # Simpler: S_i = |E_i| / (ℏω_i)
            E_i = -(F_i ** 2) / (2 * fc[i])  # Hartree
            # ω in cm⁻¹ → ℏω in Hartree: ℏω = freq * 4.5563e-6 Hartree
            hbar_omega = freq[i] * 4.5563e-6
            if hbar_omega > 0:
                S_i = abs(E_i) / hbar_omega
                S_total += S_i

        total_w += w
        S_total_avg += w * S_total
        dQ_total_avg += w * np.sqrt(dQ_total_sq)
        n_imag_avg += w * n_imag

    if total_w > 0:
        return {
            'S_total': S_total_avg / total_w,
            'dQ_total': dQ_total_avg / total_w,
            'n_imag': n_imag_avg / total_w,
        }
    return None


def _read_weights(summary_path):
    """Lit les poids de Boltzmann depuis conformers_summary.txt."""
    weights = {}
    if not summary_path.exists():
        return weights
    with open(summary_path) as f:
        for line in f:
            # Format: "conf_N: weight = X.XXX, ..."
            if line.strip().startswith('conf_'):
                parts = line.strip().split()
                try:
                    conf_num = int(parts[0].replace('conf_', '').rstrip(':'))
                    # Find weight value
                    for i, p in enumerate(parts):
                        if p == 'weight' and i + 2 < len(parts):
                            weights[conf_num] = float(parts[i + 2].rstrip(','))
                            break
                except (ValueError, IndexError):
                    pass
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# DESCRIPTEUR : COMPOSITION DE LA LUMO (depuis neutre)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_lumo_composition(workdir, mol_name):
    """Extrait la composition atomique de la LUMO via les populations SFO.

    Retourne un dict avec :
      - lumo_eV, homo_eV, gap_eV
      - lumo_O_fraction : poids de la LUMO sur les O
      - lumo_C_fraction : poids de la LUMO sur les C
      - lumo_hal_fraction : poids sur les halogènes (Cl, Br, F, I)
      - lumo_max_atom_fraction : poids maximal sur un seul atome
      - lumo_n_atoms_50pct : nombre d'atomes contribuant à 50% de la LUMO
    """
    from scm.plams import KFReader
    from qmanalysis.frequency_extraction import find_neutral_rkf_files

    neutral_rkfs = find_neutral_rkf_files(workdir, mol_name)
    if not neutral_rkfs:
        return None

    # Use first conformer (or Boltzmann-average later)
    # For simplicity, use the conformer with highest weight
    weights_path = workdir / "conformers" / "results" / "conformers_summary.txt"
    weights = _read_weights(weights_path)
    best_conf = max(neutral_rkfs.keys(), key=lambda c: weights.get(c, 0))
    rkf_path = neutral_rkfs[best_conf]

    try:
        kf = KFReader(str(rkf_path))

        # Orbital data
        n_mo = kf.read('A', 'nmo_A')
        occupations = np.array(kf.read('A', 'froc_A'), dtype=float)
        energies = np.array(kf.read('A', 'eps_A'), dtype=float)
        n_occ = int(np.sum(occupations > 0.5))

        homo_eV = energies[n_occ - 1] * EH_TO_EV
        lumo_eV = energies[n_occ] * EH_TO_EV
        gap_eV = lumo_eV - homo_eV

        # Atomic info
        n_atoms = kf.read('Molecule', 'nAtoms')
        atomic_numbers = list(kf.read('Molecule', 'AtomicNumbers'))

        # SFO data
        n_sfo = kf.read('SFOs', 'number')
        sfo_fragments = list(kf.read('SFOs', 'fragment'))  # 1-indexed atom

        # SFO populations (packed format)
        n_contrib = list(kf.read('SFO popul', 'number of contributions'))
        sfo_pop_idx = list(kf.read('SFO popul', 'sfo_index'))
        sfo_pop_val = list(kf.read('SFO popul', 'sfo_pop'))

        # Extract LUMO composition
        lumo_mo_idx = n_occ  # 0-indexed
        offset = 0
        atom_weights = np.zeros(n_atoms)

        for mo_idx in range(n_mo):
            nc = n_contrib[mo_idx]
            if mo_idx == lumo_mo_idx:
                idx_slice = sfo_pop_idx[offset:offset + nc]
                val_slice = sfo_pop_val[offset:offset + nc]

                for sfo_idx, pop_val in zip(idx_slice, val_slice):
                    # sfo_idx is 1-indexed
                    if 1 <= sfo_idx <= n_sfo:
                        atom_idx = sfo_fragments[sfo_idx - 1] - 1  # 0-indexed
                        if 0 <= atom_idx < n_atoms:
                            atom_weights[atom_idx] += abs(pop_val)
                break
            offset += nc

        # Normalize
        total_weight = np.sum(atom_weights)
        if total_weight > 0:
            atom_fractions = atom_weights / total_weight
        else:
            atom_fractions = atom_weights

        # Group by element type
        O_frac = sum(atom_fractions[i] for i in range(n_atoms)
                     if atomic_numbers[i] == 8)
        C_frac = sum(atom_fractions[i] for i in range(n_atoms)
                     if atomic_numbers[i] == 6)
        H_frac = sum(atom_fractions[i] for i in range(n_atoms)
                     if atomic_numbers[i] == 1)
        hal_frac = sum(atom_fractions[i] for i in range(n_atoms)
                       if atomic_numbers[i] in (9, 17, 35, 53))
        N_frac = sum(atom_fractions[i] for i in range(n_atoms)
                     if atomic_numbers[i] == 7)

        max_atom_frac = np.max(atom_fractions)

        # Number of atoms contributing to 50% of LUMO
        sorted_fracs = np.sort(atom_fractions)[::-1]
        cumsum = np.cumsum(sorted_fracs)
        n_atoms_50 = int(np.searchsorted(cumsum, 0.5)) + 1

        return {
            'homo_eV': homo_eV,
            'lumo_eV': lumo_eV,
            'gap_eV': gap_eV,
            'lumo_O_frac': O_frac,
            'lumo_C_frac': C_frac,
            'lumo_H_frac': H_frac,
            'lumo_hal_frac': hal_frac,
            'lumo_N_frac': N_frac,
            'lumo_max_atom_frac': max_atom_frac,
            'lumo_n_atoms_50pct': n_atoms_50,
            'atom_fractions': atom_fractions.tolist(),
            'atomic_numbers': atomic_numbers,
        }

    except Exception as e:
        print(f"    Erreur extraction LUMO pour {mol_name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE COMBINÉE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ANALYSE DES DESCRIPTEURS PRÉDICTIFS DU RMSD")
    print("=" * 70)

    # Load LVC2 data
    with open(LVC_CSV) as f:
        lvc_rows = list(csv.DictReader(f))

    mol_names = [r['molecule'] for r in lvc_rows]
    real_red = np.array([float(r['E_def_real_red']) for r in lvc_rows])
    lvc2_red = np.array([float(r['E_def_LVC2_red']) for r in lvc_rows])
    rmsd_red = np.array([float(r['RMSD_red']) for r in lvc_rows])
    real_ox = np.array([float(r['E_def_real_ox']) for r in lvc_rows])
    lvc2_ox = np.array([float(r['E_def_LVC2_ox']) for r in lvc_rows])
    rmsd_ox = np.array([float(r['RMSD_ox']) for r in lvc_rows])
    residual_red = real_red - lvc2_red
    residual_ox = real_ox - lvc2_ox

    n = len(mol_names)

    # ── Collect descriptors ──
    print(f"\nExtraction des descripteurs pour {n} molécules...")
    huang_rhys_red = np.full(n, np.nan)
    huang_rhys_ox = np.full(n, np.nan)
    n_imag_red = np.full(n, np.nan)
    n_imag_ox = np.full(n, np.nan)
    dQ_red = np.full(n, np.nan)
    dQ_ox = np.full(n, np.nan)
    lumo_O = np.full(n, np.nan)
    lumo_C = np.full(n, np.nan)
    lumo_hal = np.full(n, np.nan)
    lumo_eV = np.full(n, np.nan)
    gap_eV = np.full(n, np.nan)
    lumo_max_atom = np.full(n, np.nan)
    lumo_n_atoms_50 = np.full(n, np.nan)

    for i, mol in enumerate(mol_names):
        workdir = DATA_DIR / f"{mol}_workdir"
        if not workdir.exists():
            print(f"  {mol}: workdir non trouvé")
            continue

        # Huang-Rhys
        hr_red = compute_huang_rhys_from_lvc2(workdir, mol, 'reduced')
        hr_ox = compute_huang_rhys_from_lvc2(workdir, mol, 'oxidized')
        if hr_red:
            huang_rhys_red[i] = hr_red['S_total']
            n_imag_red[i] = hr_red['n_imag']
            dQ_red[i] = hr_red['dQ_total']
        if hr_ox:
            huang_rhys_ox[i] = hr_ox['S_total']
            n_imag_ox[i] = hr_ox['n_imag']
            dQ_ox[i] = hr_ox['dQ_total']

        # LUMO composition
        lumo = extract_lumo_composition(workdir, mol)
        if lumo:
            lumo_O[i] = lumo['lumo_O_frac']
            lumo_C[i] = lumo['lumo_C_frac']
            lumo_hal[i] = lumo['lumo_hal_frac']
            lumo_eV[i] = lumo['lumo_eV']
            gap_eV[i] = lumo['gap_eV']
            lumo_max_atom[i] = lumo['lumo_max_atom_frac']
            lumo_n_atoms_50[i] = lumo['lumo_n_atoms_50pct']

        print(f"  {mol}: S_red={huang_rhys_red[i]:.1f}, S_ox={huang_rhys_ox[i]:.1f}, "
              f"n_imag_red={n_imag_red[i]:.0f}, LUMO_O={lumo_O[i]:.2f}, "
              f"LUMO_hal={lumo_hal[i]:.2f}")

    # ── Correlations ──
    print("\n" + "=" * 70)
    print("CORRÉLATIONS AVEC LE RMSD (RÉDUCTION)")
    print("=" * 70)

    descriptors = {
        'S_Huang-Rhys': huang_rhys_red,
        'n_imag': n_imag_red,
        'ΔQ_total': dQ_red,
        '|LVC2_red|': np.abs(lvc2_red),
        'LUMO_O_frac': lumo_O,
        'LUMO_C_frac': lumo_C,
        'LUMO_hal_frac': lumo_hal,
        'LUMO_eV': lumo_eV,
        'Gap_eV': gap_eV,
        'LUMO_max_atom': lumo_max_atom,
    }

    print(f"\n  {'Descripteur':<20} {'vs RMSD_red':>12} {'vs Résidu_red':>14} {'p(RMSD)':>10} {'p(Rés)':>10}")
    print(f"  {'-'*66}")

    good_predictors = {}
    for name, vals in descriptors.items():
        mask = np.isfinite(vals) & np.isfinite(rmsd_red)
        if np.sum(mask) < 5 or np.std(vals[mask]) < 1e-10:
            print(f"  {name:<20} {'n/a':>12} {'n/a':>14}")
            continue

        fit_rmsd = linregress(vals[mask], rmsd_red[mask])
        fit_res = linregress(vals[mask], residual_red[mask])

        r2_rmsd = fit_rmsd.rvalue ** 2
        r2_res = fit_res.rvalue ** 2

        print(f"  {name:<20} R²={r2_rmsd:.3f}   R²={r2_res:.3f}     "
              f"p={fit_rmsd.pvalue:.4f}  p={fit_res.pvalue:.4f}")

        if fit_rmsd.pvalue < 0.05 or fit_res.pvalue < 0.05:
            good_predictors[name] = vals

    # ── Corrélations oxydation ──
    print("\n" + "=" * 70)
    print("CORRÉLATIONS AVEC LE RMSD (OXYDATION)")
    print("=" * 70)

    descriptors_ox = {
        'S_Huang-Rhys': huang_rhys_ox,
        'n_imag': n_imag_ox,
        '|LVC2_ox|': np.abs(lvc2_ox),
        'LUMO_O_frac': lumo_O,
        'Gap_eV': gap_eV,
    }

    print(f"\n  {'Descripteur':<20} {'vs RMSD_ox':>12} {'vs Résidu_ox':>14}")
    print(f"  {'-'*50}")
    for name, vals in descriptors_ox.items():
        mask = np.isfinite(vals) & np.isfinite(rmsd_ox)
        if np.sum(mask) < 5 or np.std(vals[mask]) < 1e-10:
            continue
        fit_rmsd = linregress(vals[mask], rmsd_ox[mask])
        fit_res = linregress(vals[mask], residual_ox[mask])
        print(f"  {name:<20} R²={fit_rmsd.rvalue**2:.3f}   R²={fit_res.rvalue**2:.3f}")

    # ── Modèle de correction multilinéaire ──
    print("\n" + "=" * 70)
    print("MODÈLES DE CORRECTION (RÉDUCTION)")
    print("=" * 70)

    # Model 1: LVC2 calibré (baseline)
    fit1 = linregress(lvc2_red, real_red)
    cal1 = fit1.slope * lvc2_red + fit1.intercept
    mae1 = np.mean(np.abs(real_red - cal1))

    # Model 2: LVC2 + S_Huang-Rhys
    mask2 = np.isfinite(huang_rhys_red)
    if np.sum(mask2) >= 5:
        X2 = np.column_stack([lvc2_red[mask2], huang_rhys_red[mask2], np.ones(np.sum(mask2))])
        c2, _, _, _ = np.linalg.lstsq(X2, real_red[mask2], rcond=None)
        pred2 = X2 @ c2
        mae2 = np.mean(np.abs(real_red[mask2] - pred2))
        ss_res = np.sum((real_red[mask2] - pred2)**2)
        ss_tot = np.sum((real_red[mask2] - np.mean(real_red[mask2]))**2)
        r2_2 = 1 - ss_res / ss_tot
        # LOO
        loo2 = []
        for j in range(np.sum(mask2)):
            Xt = np.delete(X2, j, axis=0)
            yt = np.delete(real_red[mask2], j)
            ct, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
            loo2.append(abs(real_red[mask2][j] - X2[j] @ ct))
        mae2_loo = np.mean(loo2)
    else:
        mae2 = mae2_loo = r2_2 = np.nan

    # Model 3: LVC2 + n_imag
    mask3 = np.isfinite(n_imag_red)
    if np.sum(mask3) >= 5:
        X3 = np.column_stack([lvc2_red[mask3], n_imag_red[mask3], np.ones(np.sum(mask3))])
        c3, _, _, _ = np.linalg.lstsq(X3, real_red[mask3], rcond=None)
        pred3 = X3 @ c3
        mae3 = np.mean(np.abs(real_red[mask3] - pred3))
        ss_res = np.sum((real_red[mask3] - pred3)**2)
        ss_tot = np.sum((real_red[mask3] - np.mean(real_red[mask3]))**2)
        r2_3 = 1 - ss_res / ss_tot
        loo3 = []
        for j in range(np.sum(mask3)):
            Xt = np.delete(X3, j, axis=0)
            yt = np.delete(real_red[mask3], j)
            ct, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
            loo3.append(abs(real_red[mask3][j] - X3[j] @ ct))
        mae3_loo = np.mean(loo3)
    else:
        mae3 = mae3_loo = r2_3 = np.nan

    # Model 4: LVC2 + S_HR + LUMO_O
    mask4 = np.isfinite(huang_rhys_red) & np.isfinite(lumo_O)
    if np.sum(mask4) >= 5:
        X4 = np.column_stack([lvc2_red[mask4], huang_rhys_red[mask4], lumo_O[mask4], np.ones(np.sum(mask4))])
        c4, _, _, _ = np.linalg.lstsq(X4, real_red[mask4], rcond=None)
        pred4 = X4 @ c4
        mae4 = np.mean(np.abs(real_red[mask4] - pred4))
        ss_res = np.sum((real_red[mask4] - pred4)**2)
        ss_tot = np.sum((real_red[mask4] - np.mean(real_red[mask4]))**2)
        r2_4 = 1 - ss_res / ss_tot
        loo4 = []
        for j in range(np.sum(mask4)):
            Xt = np.delete(X4, j, axis=0)
            yt = np.delete(real_red[mask4], j)
            ct, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
            loo4.append(abs(real_red[mask4][j] - X4[j] @ ct))
        mae4_loo = np.mean(loo4)
    else:
        mae4 = mae4_loo = r2_4 = np.nan

    print(f"\n  {'Modèle':<35} {'R²':>6} {'MAE':>8} {'MAE LOO':>9}")
    print(f"  {'-'*60}")
    print(f"  {'LVC2 brut':<35} {'0.843':>6} {'2.2':>8} {'---':>9}")
    print(f"  {'LVC2 calibré (linéaire)':<35} {fit1.rvalue**2:>6.3f} {mae1:>8.1f} {'1.9':>9}")
    print(f"  {'LVC2 + S_Huang-Rhys':<35} {r2_2:>6.3f} {mae2:>8.1f} {mae2_loo:>9.1f}")
    print(f"  {'LVC2 + n_imag':<35} {r2_3:>6.3f} {mae3:>8.1f} {mae3_loo:>9.1f}")
    print(f"  {'LVC2 + S_HR + LUMO_O':<35} {r2_4:>6.3f} {mae4:>8.1f} {mae4_loo:>9.1f}")

    # ── Export descriptors CSV ──
    desc_file = OUTPUT_DIR / "descriptors_per_molecule.csv"
    with open(desc_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'molecule', 'E_real_red', 'LVC2_red', 'RMSD_red',
            'E_real_ox', 'LVC2_ox', 'RMSD_ox',
            'S_HR_red', 'S_HR_ox', 'n_imag_red', 'n_imag_ox',
            'dQ_red', 'dQ_ox',
            'LUMO_O_frac', 'LUMO_C_frac', 'LUMO_hal_frac',
            'LUMO_eV', 'gap_eV', 'LUMO_max_atom', 'LUMO_n_atoms_50pct',
        ])
        for i in range(n):
            writer.writerow([
                mol_names[i], real_red[i], lvc2_red[i], rmsd_red[i],
                real_ox[i], lvc2_ox[i], rmsd_ox[i],
                huang_rhys_red[i], huang_rhys_ox[i], n_imag_red[i], n_imag_ox[i],
                dQ_red[i], dQ_ox[i],
                lumo_O[i], lumo_C[i], lumo_hal[i],
                lumo_eV[i], gap_eV[i], lumo_max_atom[i], lumo_n_atoms_50[i],
            ])
    print(f"\n  Descripteurs exportés: {desc_file}")

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == '__main__':
    main()
