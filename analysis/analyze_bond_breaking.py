#!/usr/bin/env amspython
"""
Analyse de rupture de liaison lors de la réduction.

Compare les géométries neutres et réduites optimisées pour identifier
les élongations et ruptures de liaison, en particulier les liaisons C-X
dans les dérivés halogénés (attachement électronique dissociatif, DEA).
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

# ── Constantes (importées depuis common.bond_detection) ──────────────────────

from common.bond_detection import (
    COVALENT_RADII, BOND_TOLERANCE, BREAKING_THRESHOLD, ELONGATION_THRESHOLD,
    BOHR_TO_ANG, Z_TO_SYMBOL, detect_bonds, bond_type_label,
)
from analysis import MOLECULES_11

HALOGENS = {'F', 'Cl', 'Br', 'I'}


# ── Extraction de données RKF ────────────────────────────────────────────────

def extract_geometry(rkf_path):
    """Extrait symboles atomiques et coordonnées depuis un fichier adf.rkf.

    Returns:
        (symbols, coords) : (list[str], np.ndarray shape (n,3) en Å)
    """
    from scm.plams import KFReader
    kf = KFReader(str(rkf_path))

    n_atoms = kf.read('Molecule', 'nAtoms')
    atomic_numbers = kf.read('Molecule', 'AtomicNumbers')
    coords_bohr = np.array(kf.read('Molecule', 'Coords'), dtype=float)

    symbols = [Z_TO_SYMBOL.get(z, f'Z{z}') for z in atomic_numbers]
    coords = coords_bohr.reshape(n_atoms, 3) * BOHR_TO_ANG

    return symbols, coords


# ── Analyse d'un conformère ───────────────────────────────────────────────────

def analyze_conformer(neutral_rkf, reduced_rkf):
    """Compare la géométrie neutre et réduite pour un conformère.

    Returns:
        list of dicts avec info par liaison
    """
    sym_n, coords_n = extract_geometry(neutral_rkf)
    sym_r, coords_r = extract_geometry(reduced_rkf)

    assert len(sym_n) == len(sym_r), "Nombre d'atomes différent"
    assert sym_n == sym_r, "Symboles atomiques différents"

    bonds = detect_bonds(sym_n, coords_n)
    results = []

    for i, j in bonds:
        d_n = np.linalg.norm(coords_n[i] - coords_n[j])
        d_r = np.linalg.norm(coords_r[i] - coords_r[j])
        elongation = d_r - d_n
        bt = bond_type_label(sym_n[i], sym_n[j])

        if elongation > BREAKING_THRESHOLD:
            status = 'broken'
        elif elongation > ELONGATION_THRESHOLD:
            status = 'elongated'
        else:
            status = 'intact'

        results.append({
            'atom1_idx': i + 1,  # 1-indexed
            'atom1_symbol': sym_n[i],
            'atom2_idx': j + 1,
            'atom2_symbol': sym_n[j],
            'bond_type': bt,
            'd_neutral': round(d_n, 3),
            'd_reduced': round(d_r, 3),
            'elongation': round(elongation, 3),
            'status': status,
        })

    return results


# ── Recherche des fichiers RKF ────────────────────────────────────────────────

def find_rkf_pairs(workdir, mol_name):
    """Trouve les paires (neutre, réduit_opt) de fichiers RKF.

    Returns:
        list of (conf_num, neutral_rkf, reduced_rkf)
    """
    plams_dir = Path(workdir) / 'redox' / 'calculations' / 'plams_workdir'
    if not plams_dir.is_dir():
        return []

    pairs = []
    pattern = re.compile(rf'^{re.escape(mol_name)}_conf_(\d+)_neutral$')

    for entry in sorted(plams_dir.iterdir()):
        m = pattern.match(entry.name)
        if not m:
            continue
        conf_num = int(m.group(1))
        neutral_rkf = entry / 'adf.rkf'
        reduced_rkf = plams_dir / f'{mol_name}_conf_{conf_num}_reduced_opt' / 'adf.rkf'

        if neutral_rkf.is_file() and reduced_rkf.is_file():
            pairs.append((conf_num, neutral_rkf, reduced_rkf))

    return pairs


def read_weights(workdir, mol_name):
    """Lit les poids de Boltzmann depuis le CSV redox."""
    csv_path = Path(workdir) / 'redox' / 'results' / f'{mol_name}_redox_conformers.csv'
    weights = {}
    if csv_path.is_file():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                weights[int(row['conformer'])] = float(row['weight'])
    return weights


def read_edef(workdir, mol_name):
    """Lit les Edef_red depuis le CSV redox."""
    csv_path = Path(workdir) / 'redox' / 'results' / f'{mol_name}_redox_conformers.csv'
    edefs = {}
    if csv_path.is_file():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                edefs[int(row['conformer'])] = float(row['Edef_red_kJ'])
    return edefs


# ── Analyse batch ─────────────────────────────────────────────────────────────

def analyze_molecule(batch_dir, mol_name):
    """Analyse complète d'une molécule (tous conformères)."""
    workdir = Path(batch_dir) / f'{mol_name}_workdir'
    if not workdir.is_dir():
        return None

    pairs = find_rkf_pairs(workdir, mol_name)
    if not pairs:
        return None

    weights = read_weights(workdir, mol_name)
    edefs = read_edef(workdir, mol_name)

    all_bonds = []
    for conf_num, neutral_rkf, reduced_rkf in pairs:
        try:
            bonds = analyze_conformer(neutral_rkf, reduced_rkf)
            w = weights.get(conf_num, 1.0 / len(pairs))
            edef = edefs.get(conf_num, float('nan'))
            for b in bonds:
                b['molecule'] = mol_name
                b['conformer'] = conf_num
                b['weight'] = w
                b['Edef_red_kJ'] = edef
            all_bonds.extend(bonds)
        except Exception as e:
            print(f"  ⚠ {mol_name} conf {conf_num}: {e}")

    return all_bonds


def summarize_molecule(all_bonds, mol_name):
    """Résumé par molécule : liaison la plus élongée, liaisons C-X."""
    if not all_bonds:
        return None

    # Trouver la liaison la plus élongée (moyenne pondérée par Boltzmann)
    # Regrouper par (atom1_idx, atom2_idx)
    bond_map = {}
    for b in all_bonds:
        key = (b['atom1_idx'], b['atom2_idx'])
        if key not in bond_map:
            bond_map[key] = []
        bond_map[key].append(b)

    # Pour chaque liaison, calculer l'élongation moyenne Boltzmann
    best = None
    cx_bonds = []

    for key, entries in bond_map.items():
        w_total = sum(e['weight'] for e in entries)
        avg_d_n = sum(e['weight'] * e['d_neutral'] for e in entries) / w_total
        avg_d_r = sum(e['weight'] * e['d_reduced'] for e in entries) / w_total
        avg_elong = sum(e['weight'] * e['elongation'] for e in entries) / w_total

        ref = entries[0]  # pour les labels
        is_halogen = ref['atom1_symbol'] in HALOGENS or ref['atom2_symbol'] in HALOGENS

        info = {
            'molecule': mol_name,
            'bond_type': ref['bond_type'],
            'atom1_idx': ref['atom1_idx'],
            'atom1_symbol': ref['atom1_symbol'],
            'atom2_idx': ref['atom2_idx'],
            'atom2_symbol': ref['atom2_symbol'],
            'd_neutral': round(avg_d_n, 3),
            'd_reduced': round(avg_d_r, 3),
            'elongation': round(avg_elong, 3),
            'status': 'broken' if avg_elong > BREAKING_THRESHOLD else
                      'elongated' if avg_elong > ELONGATION_THRESHOLD else 'intact',
            'is_halogen_bond': is_halogen,
        }

        # Track C-X halogen bonds
        if is_halogen and ('C' in [ref['atom1_symbol'], ref['atom2_symbol']]):
            cx_bonds.append(info)

        if best is None or abs(avg_elong) > abs(best['elongation']):
            best = info

    return {
        'max_elongation_bond': best,
        'cx_bonds': cx_bonds,
    }


# ── Sortie ────────────────────────────────────────────────────────────────────

def write_detail_csv(all_data, output_dir):
    """Écrit le CSV détaillé par liaison par conformère."""
    path = output_dir / 'bond_breaking_analysis.csv'
    fields = ['molecule', 'conformer', 'weight', 'atom1_idx', 'atom1_symbol',
              'atom2_idx', 'atom2_symbol', 'bond_type', 'd_neutral', 'd_reduced',
              'elongation', 'status', 'Edef_red_kJ']

    # Ne garder que les liaisons non-intactes + toutes les C-X
    filtered = [b for b in all_data
                if b['status'] != 'intact' or
                (b['atom1_symbol'] in HALOGENS or b['atom2_symbol'] in HALOGENS)]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(filtered)

    print(f"  → {path} ({len(filtered)} liaisons)")


def write_summary(summaries, output_dir):
    """Écrit le rapport texte formaté."""
    path = output_dir / 'bond_breaking_report.txt'

    lines = []
    lines.append('=' * 100)
    lines.append('ANALYSE DE RUPTURE DE LIAISON EN RÉDUCTION')
    lines.append('=' * 100)
    lines.append('')

    # ── Section 1 : Résumé pour toutes les molécules ──
    lines.append('--- Liaison la plus élongée par molécule ---')
    lines.append(f'{"Molécule":<12} {"Liaison":<8} {"d(neutre)":<10} {"d(réduit)":<10} '
                 f'{"Δd (Å)":<10} {"Statut":<12}')
    lines.append('-' * 70)

    for mol in MOLECULES_11:
        s = summaries.get(mol)
        if not s or not s['max_elongation_bond']:
            lines.append(f'{mol:<12} pas de données')
            continue
        b = s['max_elongation_bond']
        lines.append(f'{mol:<12} {b["bond_type"]:<8} {b["d_neutral"]:>9.3f} '
                     f'{b["d_reduced"]:>9.3f} {b["elongation"]:>+9.3f} {b["status"]:<12}')

    # ── Section 2 : Liaisons C-X des halogénés ──
    lines.append('')
    lines.append('--- Liaisons C-X dans les dérivés halogénés ---')
    lines.append(f'{"Molécule":<12} {"Liaison":<8} {"d(neutre)":<10} {"d(réduit)":<10} '
                 f'{"Δd (Å)":<10} {"Rupture?":<10}')
    lines.append('-' * 70)

    halogen_mols = ['F-DA', 'Cl-DA', 'F-HMC', 'Cl-HMC']
    for mol in halogen_mols:
        s = summaries.get(mol)
        if not s:
            continue
        for cx in s['cx_bonds']:
            rupture = 'Oui' if cx['status'] == 'broken' else 'Non'
            lines.append(f'{mol:<12} {cx["bond_type"]:<8} {cx["d_neutral"]:>9.3f} '
                         f'{cx["d_reduced"]:>9.3f} {cx["elongation"]:>+9.3f} {rupture:<10}')

    # ── Section 3 : Classification des mécanismes d'échec ──
    lines.append('')
    lines.append('--- Classification des mécanismes d\'échec LVC en réduction ---')
    lines.append('')

    dea_mols = []
    conf_mols = []
    harmonic_mols = []

    for mol in MOLECULES_11:
        s = summaries.get(mol)
        if not s or not s['max_elongation_bond']:
            continue
        b = s['max_elongation_bond']
        if b['status'] == 'broken':
            dea_mols.append(mol)
        elif b['status'] == 'elongated':
            conf_mols.append(mol)
        else:
            if b['elongation'] > 0.1:
                conf_mols.append(mol)
            else:
                harmonic_mols.append(mol)

    lines.append(f'  Harmonique (Δd_max < 0.1 Å) : {", ".join(harmonic_mols)}')
    lines.append(f'  Réarrangement conformationnel : {", ".join(conf_mols)}')
    lines.append(f'  Dissociation de liaison (DEA) : {", ".join(dea_mols)}')

    # ── Section 4 : Tableau LaTeX prêt à copier ──
    lines.append('')
    lines.append('--- Tableau LaTeX (C-X halogenés) ---')
    lines.append('')

    for mol in halogen_mols:
        s = summaries.get(mol)
        if not s or not s['cx_bonds']:
            continue
        cx = s['cx_bonds'][0]  # une seule C-X par molécule
        rupture = 'Oui' if cx['status'] == 'broken' else 'Non'
        lines.append(f'{mol:<8} & C--{cx["atom2_symbol"] if cx["atom1_symbol"] == "C" else cx["atom1_symbol"]} '
                     f'& {cx["d_neutral"]:.3f} & {cx["d_reduced"]:.3f} & {cx["elongation"]:+.3f} '
                     f'& {rupture} \\\\')

    text = '\n'.join(lines) + '\n'
    with open(path, 'w') as f:
        f.write(text)
    print(f"  → {path}")

    # Aussi imprimer
    print()
    print(text)


def write_summary_csv(summaries, output_dir):
    """Écrit le CSV résumé par molécule."""
    path = output_dir / 'bond_breaking_summary.csv'
    fields = ['molecule', 'max_bond_type', 'max_d_neutral', 'max_d_reduced',
              'max_elongation', 'max_status', 'has_cx_bond', 'cx_bond_type',
              'cx_d_neutral', 'cx_d_reduced', 'cx_elongation', 'cx_breaks']

    rows = []
    for mol in MOLECULES_11:
        s = summaries.get(mol)
        if not s:
            continue
        b = s['max_elongation_bond']
        row = {
            'molecule': mol,
            'max_bond_type': b['bond_type'],
            'max_d_neutral': b['d_neutral'],
            'max_d_reduced': b['d_reduced'],
            'max_elongation': b['elongation'],
            'max_status': b['status'],
        }
        if s['cx_bonds']:
            cx = s['cx_bonds'][0]
            row['has_cx_bond'] = True
            row['cx_bond_type'] = cx['bond_type']
            row['cx_d_neutral'] = cx['d_neutral']
            row['cx_d_reduced'] = cx['d_reduced']
            row['cx_elongation'] = cx['elongation']
            row['cx_breaks'] = cx['status'] == 'broken'
        else:
            row['has_cx_bond'] = False
            row['cx_bond_type'] = ''
            row['cx_d_neutral'] = ''
            row['cx_d_reduced'] = ''
            row['cx_elongation'] = ''
            row['cx_breaks'] = ''
        rows.append(row)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse de rupture de liaison lors de la réduction")
    _root = str(Path(__file__).resolve().parent.parent)
    parser.add_argument('--batch-dir', default=_root + '/data/1594174/dir_workdir',
                        help='Répertoire contenant les workdirs')
    parser.add_argument('--output-dir', default=_root + '/results/bond_breaking',
                        help='Répertoire de sortie')
    parser.add_argument('--molecules', nargs='*', default=None,
                        help='Molécules spécifiques (défaut : toutes)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    molecules = args.molecules or MOLECULES_11

    print('=' * 60)
    print('ANALYSE DE RUPTURE DE LIAISON EN RÉDUCTION')
    print('=' * 60)
    print()

    all_data = []
    summaries = {}

    for mol in molecules:
        print(f'  {mol}...', end=' ', flush=True)
        bonds = analyze_molecule(args.batch_dir, mol)
        if bonds is None:
            print('pas de données')
            continue

        n_broken = sum(1 for b in bonds if b['status'] == 'broken')
        n_elongated = sum(1 for b in bonds if b['status'] == 'elongated')
        all_data.extend(bonds)

        summary = summarize_molecule(bonds, mol)
        summaries[mol] = summary

        max_b = summary['max_elongation_bond']
        print(f'{len(bonds)} liaisons, max Δd = {max_b["elongation"]:+.3f} Å '
              f'({max_b["bond_type"]})', end='')
        if n_broken:
            print(f' — {n_broken} ROMPUE(S)', end='')
        if n_elongated:
            print(f' — {n_elongated} élongée(s)', end='')
        print()

    print()
    write_detail_csv(all_data, output_dir)
    write_summary_csv(summaries, output_dir)
    write_summary(summaries, output_dir)


if __name__ == '__main__':
    main()
