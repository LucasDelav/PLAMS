#!/usr/bin/env amspython
"""
Analyse de changement conformationnel lors de la réduction et oxydation.

Pour chaque molécule ayant plusieurs conformères, compare la géométrie
réduite/oxydée optimisée de chaque conformère N à TOUTES les géométries
neutres pour détecter si l'optimisation provoque un changement de conformère.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from scm.plams import KFReader, Molecule

# ── Constantes ───────────────────────────────────────────────────────────────

BOHR_TO_ANG = 0.529177249

Z_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl',
    13: 'Al', 35: 'Br', 53: 'I',
}

from analysis import MOLECULES_11


# ── RMSD (Kabsch) ───────────────────────────────────────────────────────────

def kabsch_rmsd(coords1, coords2):
    """Calcule le RMSD après alignement optimal de Kabsch.

    Args:
        coords1, coords2: np.ndarray shape (n, 3)

    Returns:
        float: RMSD en Angstrom
    """
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    H = c1.T @ c2
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    c1_aligned = c1 @ R
    diff = c1_aligned - c2
    return np.sqrt((diff ** 2).sum() / len(coords1))


# ── Extraction de géométries ────────────────────────────────────────────────

def extract_coords_from_rkf(rkf_path):
    """Extrait les coordonnées atomiques depuis un fichier adf.rkf.

    Returns:
        np.ndarray shape (n, 3) en Angstrom
    """
    kf = KFReader(str(rkf_path))
    n_atoms = kf.read('Molecule', 'nAtoms')
    coords_bohr = np.array(kf.read('Molecule', 'Coords'), dtype=float)
    return coords_bohr.reshape(n_atoms, 3) * BOHR_TO_ANG


def extract_coords_from_xyz(xyz_path):
    """Extrait les coordonnées atomiques depuis un fichier XYZ.

    Returns:
        np.ndarray shape (n, 3) en Angstrom
    """
    mol = Molecule(str(xyz_path))
    return np.array([atom.coords for atom in mol.atoms])


# ── Recherche des fichiers ──────────────────────────────────────────────────

def find_neutral_xyz(workdir, mol_name):
    """Trouve les fichiers XYZ des conformères neutres.

    Returns:
        dict: {conf_num: Path}
    """
    results_dir = Path(workdir) / 'conformers' / 'results'
    if not results_dir.is_dir():
        return {}

    neutrals = {}
    for f in sorted(results_dir.glob('conf_*.xyz')):
        m = re.match(r'conf_(\d+)\.xyz', f.name)
        if m:
            neutrals[int(m.group(1))] = f
    return neutrals


def find_charged_rkf(workdir, mol_name, state='reduced_opt'):
    """Trouve les fichiers RKF des conformères optimisés (réduit ou oxydé).

    Args:
        state: 'reduced_opt' ou 'oxidized_opt'

    Returns:
        dict: {conf_num: Path}
    """
    plams_dir = Path(workdir) / 'redox' / 'calculations' / 'plams_workdir'
    if not plams_dir.is_dir():
        return {}

    pattern = re.compile(rf'^{re.escape(mol_name)}_conf_(\d+)_{state}$')
    charged = {}
    for entry in sorted(plams_dir.iterdir()):
        m = pattern.match(entry.name)
        if m:
            rkf = entry / 'adf.rkf'
            if rkf.is_file():
                charged[int(m.group(1))] = rkf
    return charged


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


# ── Analyse d'une molécule ──────────────────────────────────────────────────

def analyze_molecule(batch_dir, mol_name):
    """Analyse conformationnelle complète pour une molécule.

    Returns:
        dict avec 'reduction' et 'oxidation', chacun contenant :
        - 'matrix': dict {(conf_charged, conf_neutral): RMSD}
        - 'neutral_confs': list des numéros de conformères neutres
        - 'charged_confs': list des numéros de conformères chargés
        - 'details': list of dicts par conformère chargé
    """
    workdir = Path(batch_dir) / f'{mol_name}_workdir'
    if not workdir.is_dir():
        return None

    neutrals = find_neutral_xyz(workdir, mol_name)
    if len(neutrals) < 1:
        return None

    weights = read_weights(workdir, mol_name)

    # Charger toutes les coordonnées neutres
    neutral_coords = {}
    for conf_num, xyz_path in neutrals.items():
        try:
            neutral_coords[conf_num] = extract_coords_from_xyz(xyz_path)
        except Exception as e:
            print(f"  Warning: {mol_name} neutral conf {conf_num}: {e}")

    if not neutral_coords:
        return None

    results = {}

    for state_label, state_suffix in [('reduction', 'reduced_opt'), ('oxidation', 'oxidized_opt')]:
        charged = find_charged_rkf(workdir, mol_name, state_suffix)
        if not charged:
            continue

        matrix = {}
        details = []

        for conf_c, rkf_path in sorted(charged.items()):
            try:
                coords_c = extract_coords_from_rkf(rkf_path)
            except Exception as e:
                print(f"  Warning: {mol_name} {state_label} conf {conf_c}: {e}")
                continue

            # Calculer RMSD vers tous les neutres
            rmsds = {}
            for conf_n, coords_n in sorted(neutral_coords.items()):
                if coords_c.shape != coords_n.shape:
                    continue
                rmsds[conf_n] = kabsch_rmsd(coords_c, coords_n)
                matrix[(conf_c, conf_n)] = rmsds[conf_n]

            if not rmsds:
                continue

            rmsd_self = rmsds.get(conf_c, float('nan'))
            closest_conf = min(rmsds, key=lambda k: rmsds[k])
            rmsd_closest = rmsds[closest_conf]

            # Changement conformationnel : le plus proche n'est pas soi-même
            # ET le RMSD vers ce conformère est nettement plus petit
            change_detected = (closest_conf != conf_c and
                               not np.isnan(rmsd_self) and
                               rmsd_closest < rmsd_self * 0.8)

            # Partial change: closest is different but RMSD ratio not definitive
            partial_change = (closest_conf != conf_c and
                              not np.isnan(rmsd_self) and
                              not change_detected and
                              rmsd_closest < rmsd_self * 0.95)

            detail = {
                'molecule': mol_name,
                'state': state_label,
                'conf_charged': conf_c,
                'weight': weights.get(conf_c, 1.0 / len(charged)),
                'rmsd_self': round(rmsd_self, 4) if not np.isnan(rmsd_self) else None,
                'closest_neutral_conf': closest_conf,
                'rmsd_closest': round(rmsd_closest, 4),
                'change_detected': change_detected,
                'partial_change': partial_change,
                'all_rmsds': {k: round(v, 4) for k, v in rmsds.items()},
            }
            details.append(detail)

        results[state_label] = {
            'matrix': matrix,
            'neutral_confs': sorted(neutral_coords.keys()),
            'charged_confs': sorted(charged.keys()),
            'details': details,
        }

    return results if results else None


# ── Sortie ──────────────────────────────────────────────────────────────────

def write_rmsd_matrices(all_results, output_dir):
    """Ecrit les matrices RMSD par molécule en CSV."""
    matrices_dir = output_dir / 'rmsd_matrices'
    matrices_dir.mkdir(exist_ok=True)

    for mol_name, results in sorted(all_results.items()):
        for state_label, data in results.items():
            if len(data['neutral_confs']) < 2:
                continue

            path = matrices_dir / f'rmsd_{mol_name}_{state_label}.csv'
            neutral_confs = data['neutral_confs']

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['charged_conf'] + [f'neutral_{n}' for n in neutral_confs]
                writer.writerow(header)

                for conf_c in data['charged_confs']:
                    row = [f'conf_{conf_c}']
                    for conf_n in neutral_confs:
                        val = data['matrix'].get((conf_c, conf_n), '')
                        if val != '':
                            row.append(f'{val:.4f}')
                        else:
                            row.append('')
                    writer.writerow(row)


def write_summary_csv(all_results, output_dir):
    """Ecrit le CSV résumé par molécule/conformère."""
    path = output_dir / 'conformational_changes_summary.csv'
    fields = ['molecule', 'state', 'conformer', 'weight', 'rmsd_self',
              'closest_neutral_conf', 'rmsd_closest', 'change_detected',
              'partial_change', 'n_neutral_confs']

    rows = []
    for mol_name in MOLECULES_11:
        if mol_name not in all_results:
            continue
        for state_label in ['reduction', 'oxidation']:
            data = all_results[mol_name].get(state_label)
            if not data:
                continue
            n_neut = len(data['neutral_confs'])
            for detail in data['details']:
                rows.append({
                    'molecule': mol_name,
                    'state': state_label,
                    'conformer': detail['conf_charged'],
                    'weight': detail['weight'],
                    'rmsd_self': detail['rmsd_self'],
                    'closest_neutral_conf': detail['closest_neutral_conf'],
                    'rmsd_closest': detail['rmsd_closest'],
                    'change_detected': detail['change_detected'],
                    'partial_change': detail.get('partial_change', False),
                    'n_neutral_confs': n_neut,
                })

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path} ({len(rows)} entries)")


def write_report(all_results, output_dir):
    """Ecrit le rapport formaté."""
    path = output_dir / 'conformational_changes_report.txt'

    lines = []
    lines.append('=' * 100)
    lines.append('ANALYSE DE CHANGEMENT CONFORMATIONNEL EN REDUCTION ET OXYDATION')
    lines.append('=' * 100)
    lines.append('')

    for state_label in ['reduction', 'oxidation']:
        lines.append(f'--- {state_label.upper()} ---')
        lines.append('')
        lines.append(f'{"Molecule":<12} {"Conf":<6} {"Weight":<8} {"RMSD_self":<12} '
                     f'{"Closest":<10} {"RMSD_closest":<14} {"Change?":<10}')
        lines.append('-' * 80)

        any_change = False
        for mol_name in MOLECULES_11:
            if mol_name not in all_results:
                continue
            data = all_results[mol_name].get(state_label)
            if not data:
                continue
            if len(data['neutral_confs']) < 2:
                # Single conformer — skip
                for detail in data['details']:
                    lines.append(f'{mol_name:<12} {detail["conf_charged"]:<6} '
                                 f'{detail["weight"]:<8.3f} '
                                 f'{detail["rmsd_self"]:<12} '
                                 f'{"—":<10} {"—":<14} {"N/A (1 conf)"}')
                continue

            for detail in data['details']:
                rmsd_s = f'{detail["rmsd_self"]:.4f}' if detail["rmsd_self"] is not None else 'N/A'
                if detail['change_detected']:
                    change_str = 'OUI'
                elif detail.get('partial_change', False):
                    change_str = '~partiel'
                else:
                    change_str = 'non'

                lines.append(f'{mol_name:<12} {detail["conf_charged"]:<6} '
                             f'{detail["weight"]:<8.3f} '
                             f'{rmsd_s:<12} '
                             f'{detail["closest_neutral_conf"]:<10} '
                             f'{detail["rmsd_closest"]:<14.4f} '
                             f'{change_str}')

        lines.append('')

    # ── Classification globale ──
    lines.append('')
    lines.append('--- CLASSIFICATION PAR MOLECULE (REDUCTION) ---')
    lines.append('')

    for mol_name in MOLECULES_11:
        if mol_name not in all_results:
            lines.append(f'  {mol_name:<12} : pas de donnees')
            continue
        data = all_results[mol_name].get('reduction')
        if not data:
            lines.append(f'  {mol_name:<12} : pas de donnees reduction')
            continue

        n_confs = len(data['neutral_confs'])
        if n_confs < 2:
            lines.append(f'  {mol_name:<12} : 1 conformere (pas d\'analyse possible)')
            continue

        details = data['details']
        n_changes = sum(1 for d in details if d['change_detected'])
        n_partial = sum(1 for d in details if d.get('partial_change', False))
        avg_rmsd_self = np.mean([d['rmsd_self'] for d in details if d['rmsd_self'] is not None])

        if n_changes > 0:
            label = f'CHANGEMENT CONF. ({n_changes}/{len(details)} conf.)'
            if n_partial > 0:
                label += f' + {n_partial} partiel(s)'
        elif n_partial > 0:
            label = f'changement partiel ({n_partial}/{len(details)} conf.), avg RMSD={avg_rmsd_self:.3f}'
        elif avg_rmsd_self > 0.3:
            label = f'RMSD eleve sans changement conf. (avg={avg_rmsd_self:.3f})'
        else:
            label = f'stable (avg RMSD={avg_rmsd_self:.3f})'

        lines.append(f'  {mol_name:<12} : {n_confs} conf., {label}')

    # ── Matrices RMSD les plus interessantes ──
    lines.append('')
    lines.append('--- MATRICES RMSD (molecules a changement ou RMSD > 0.3) ---')
    lines.append('')

    for mol_name in MOLECULES_11:
        if mol_name not in all_results:
            continue
        data = all_results[mol_name].get('reduction')
        if not data or len(data['neutral_confs']) < 2:
            continue

        details = data['details']
        has_change = any(d['change_detected'] for d in details)
        avg_rmsd = np.mean([d['rmsd_self'] for d in details if d['rmsd_self'] is not None])

        if not has_change and avg_rmsd < 0.3:
            continue

        neutral_confs = data['neutral_confs']
        lines.append(f'{mol_name} (reduction):')
        header = f'  {"charged":<12} ' + ' '.join(f'{"neut_" + str(n):<10}' for n in neutral_confs)
        lines.append(header)

        for detail in details:
            row = f'  conf_{detail["conf_charged"]:<8} '
            for n in neutral_confs:
                val = detail['all_rmsds'].get(n)
                if val is not None:
                    marker = ' <-' if n == detail['closest_neutral_conf'] and detail['change_detected'] else ''
                    star = '*' if n == detail['conf_charged'] else ' '
                    row += f'{val:.4f}{star:<5} '
                else:
                    row += f'{"N/A":<10} '
            lines.append(row)
        lines.append('')

    text = '\n'.join(lines) + '\n'
    with open(path, 'w') as f:
        f.write(text)
    print(f"  -> {path}")
    print()
    print(text)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse de changement conformationnel en reduction/oxydation")
    _root = str(Path(__file__).resolve().parent.parent)
    parser.add_argument('--batch-dir', default=_root + '/data/1594174/dir_workdir',
                        help='Repertoire contenant les workdirs')
    parser.add_argument('--output-dir', default=_root + '/results/conformational_changes',
                        help='Repertoire de sortie')
    parser.add_argument('--molecules', nargs='*', default=None,
                        help='Molecules specifiques (defaut : toutes)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    molecules = args.molecules or MOLECULES_11

    print('=' * 60)
    print('ANALYSE DE CHANGEMENT CONFORMATIONNEL')
    print('=' * 60)
    print()

    all_results = {}

    for mol in molecules:
        print(f'  {mol}...', end=' ', flush=True)
        result = analyze_molecule(args.batch_dir, mol)
        if result is None:
            print('pas de donnees')
            continue

        all_results[mol] = result

        # Summary line
        for state in ['reduction', 'oxidation']:
            data = result.get(state)
            if not data:
                continue
            n_changes = sum(1 for d in data['details'] if d['change_detected'])
            n_partial = sum(1 for d in data['details'] if d.get('partial_change', False))
            n_confs = len(data['details'])
            if n_changes > 0 or n_partial > 0:
                parts = []
                if n_changes > 0:
                    parts.append(f'{n_changes}/{n_confs} change')
                if n_partial > 0:
                    parts.append(f'{n_partial}/{n_confs} partiel')
                print(f'{state}: {", ".join(parts)}', end='  ')
            else:
                print(f'{state}: stable', end='  ')
        print()

    print()
    print('Ecriture des resultats...')
    write_rmsd_matrices(all_results, output_dir)
    write_summary_csv(all_results, output_dir)
    write_report(all_results, output_dir)
    print('Termine.')


if __name__ == '__main__':
    main()
