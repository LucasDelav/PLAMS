#!/usr/bin/env amspython
"""
Analyse ALIE de Politzer & Murray : calcul de Ī_S,min et comparaison IP/EA.

Ce script :
  1. Découvre les fichiers RKF neutres de chaque molécule/conformère
  2. Lance le calcul ALIE (DENSF post-traitement) pour chacun
  3. Charge les données redox (IP/EA) et orbitales (HOMO/LUMO) existantes
  4. Calcule les prédictions :
     - IP_Koopmans = -HOMO × EH_TO_KJ
     - IP_Politzer = I_S,min × EH_TO_KJ
     - EA_Koopmans = -LUMO × EH_TO_KJ
  5. Effectue la moyenne de Boltzmann par molécule
  6. Exporte les résultats en CSV
  7. Appelle la visualisation (parity plots)

Usage:
    ./politzer_analysis.py --batch-dir 1594174/dir_workdir --output-dir politzer_results
    ./politzer_analysis.py --workdirs mol1_workdir mol2_workdir --output-dir results
    ./politzer_analysis.py --batch-dir workdirs/ --no-plots
"""

import sys
import argparse
import csv
from pathlib import Path

import numpy as np

# Ajouter la racine du projet et le répertoire analysis/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import DEA_EXCLUDED
from common.constants import EH_TO_KJ, EH_TO_EV
from common.utils import print_header, print_section, print_success, print_error, print_warning
from common.multi_molecule_data import read_boltzmann_weights
from qmanalysis.frequency_extraction import find_neutral_rkf_files
from qmanalysis.alie_extraction import compute_alie_for_conformer


# ============================================================================
# LECTEURS CSV SANS PANDAS (amspython n'a pas pandas)
# ============================================================================

def _read_csv_as_dicts(csv_path):
    """Lit un CSV et retourne une liste de dicts avec conversion numérique."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def load_conformer_redox_data_csv(workdir, molecule_name):
    """
    Lit les données redox per-conformère directement depuis le CSV (sans pandas).

    Returns:
        Liste de dicts avec conformer, weight, IP_kJ, EA_kJ, etc.
    """
    csv_file = workdir / "redox" / "results" / f"{molecule_name}_redox_conformers.csv"
    rows = _read_csv_as_dicts(csv_file)
    for row in rows:
        if 'conformer' in row:
            row['conformer'] = int(row['conformer'])
    return rows


def load_conformer_orbital_data_csv(workdir, molecule_name):
    """
    Lit les données orbitales per-conformère directement depuis le CSV (sans pandas).

    Returns:
        {conf_num: {HOMO_eV, LUMO_eV, gap_eV}}
    """
    csv_file = workdir / "redox" / "results" / f"{molecule_name}_orbital_data.csv"
    rows = _read_csv_as_dicts(csv_file)
    result = {}
    for row in rows:
        conf_num = int(row.get('conformer', 0))
        result[conf_num] = {
            'HOMO_eV': row.get('HOMO_eV', float('nan')),
            'LUMO_eV': row.get('LUMO_eV', float('nan')),
            'gap_eV': row.get('gap_eV', float('nan')),
        }
    return result


def discover_workdirs(batch_dir: Path):
    """Découvre les workdirs dans un répertoire batch."""
    return sorted([d for d in batch_dir.glob("*_workdir") if d.is_dir()])


def process_molecule(
    workdir: Path,
    output_dir: Path,
    grid_quality: str = "medium",
    verbose: bool = False,
):
    """
    Traite une molécule : calcule l'ALIE pour tous ses conformères neutres.

    Args:
        workdir: Répertoire de travail de la molécule
        output_dir: Répertoire de sortie pour les fichiers DENSF
        grid_quality: Qualité de la grille DENSF
        verbose: Mode verbeux

    Returns:
        Dict[int, Dict] : {conf_num: alie_descriptors}
    """
    mol_name = workdir.name.replace('_workdir', '')

    # Trouver les RKF neutres
    rkf_files = find_neutral_rkf_files(workdir, mol_name)
    if not rkf_files:
        print_warning(f"Aucun RKF neutre trouvé pour {mol_name}")
        return {}

    alie_dir = output_dir / "alie_calculations" / mol_name
    alie_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for conf_num in sorted(rkf_files.keys()):
        rkf_path = rkf_files[conf_num]
        name = f"{mol_name}_conf_{conf_num}"
        work_subdir = str(alie_dir / f"conf_{conf_num}")

        if verbose:
            print(f"\n  Conformère {conf_num}:")

        descriptors = compute_alie_for_conformer(
            str(rkf_path), work_subdir, name=name,
            grid_quality=grid_quality, verbose=verbose,
        )

        if descriptors.get('success', False):
            results[conf_num] = descriptors
            if not verbose:
                print_success(
                    f"  {mol_name} conf_{conf_num}: "
                    f"I_S,min = {descriptors['I_S_min_eV']:.2f} eV, "
                    f"V_S,min = {descriptors['V_S_min_eV']:.2f} eV, "
                    f"{descriptors['n_surface_points']} pts"
                )
        else:
            print_error(
                f"  {mol_name} conf_{conf_num}: "
                f"ALIE échoué ({descriptors.get('error', '?')})"
            )

    return results


def build_per_conformer_table(
    workdirs,
    alie_results: dict,
):
    """
    Construit le tableau per-conformère avec toutes les données.

    Colonnes : molecule_name, conformer, weight,
               IP_kJ, EA_kJ, HOMO_eV, LUMO_eV,
               IP_Koopmans_kJ, EA_Koopmans_kJ,
               I_S_min_Ha, I_S_min_eV, I_S_min_kJ (= IP_Politzer_kJ),
               V_S_min_Ha, V_S_min_eV, V_S_min_kJ,
               n_surface_points

    Returns:
        List[Dict] : lignes du tableau
    """
    rows = []

    for workdir in workdirs:
        workdir = Path(workdir)
        mol_name = workdir.name.replace('_workdir', '')

        # Données redox
        redox_data = load_conformer_redox_data_csv(workdir, mol_name)
        redox_by_conf = {r['conformer']: r for r in redox_data}

        # Données orbitales
        orbital_data = load_conformer_orbital_data_csv(workdir, mol_name)

        # Poids de Boltzmann
        weights = read_boltzmann_weights(workdir)
        if weights is None:
            weights = {}

        # Données ALIE
        mol_alie = alie_results.get(mol_name, {})

        # Fusionner
        all_confs = set(redox_by_conf.keys()) | set(mol_alie.keys())
        for conf_num in sorted(all_confs):
            row = {
                'molecule_name': mol_name,
                'conformer': conf_num,
            }

            # Poids
            w = weights.get(f"conf_{conf_num}", np.nan)
            row['weight'] = w

            # Redox
            redox = redox_by_conf.get(conf_num, {})
            row['IP_kJ'] = redox.get('IP_kJ', np.nan)
            row['EA_kJ'] = redox.get('EA_kJ', np.nan)

            # Orbitales (note: HOMO_eV dans orbital_data est en Hartree malgré le nom)
            orb = orbital_data.get(conf_num, {})
            homo_ha = orb.get('HOMO_eV', np.nan)
            lumo_ha = orb.get('LUMO_eV', np.nan)
            row['HOMO_Ha'] = homo_ha
            row['LUMO_Ha'] = lumo_ha
            row['HOMO_eV'] = homo_ha * EH_TO_EV if not np.isnan(homo_ha) else np.nan
            row['LUMO_eV'] = lumo_ha * EH_TO_EV if not np.isnan(lumo_ha) else np.nan

            # Koopmans predictions
            # IP_Koopmans = -HOMO (Hartree) → kJ/mol
            row['IP_Koopmans_kJ'] = -homo_ha * EH_TO_KJ if not np.isnan(homo_ha) else np.nan
            # EA_Koopmans = -LUMO (Hartree) → kJ/mol
            row['EA_Koopmans_kJ'] = -lumo_ha * EH_TO_KJ if not np.isnan(lumo_ha) else np.nan

            # ALIE descriptors
            alie = mol_alie.get(conf_num, {})
            row['I_S_min_Ha'] = alie.get('I_S_min_Ha', np.nan)
            row['I_S_min_eV'] = alie.get('I_S_min_eV', np.nan)
            row['I_S_min_kJ'] = alie.get('I_S_min_kJ', np.nan)
            row['IP_Politzer_kJ'] = alie.get('I_S_min_kJ', np.nan)

            row['V_S_min_Ha'] = alie.get('V_S_min_Ha', np.nan)
            row['V_S_min_eV'] = alie.get('V_S_min_eV', np.nan)
            row['V_S_min_kJ'] = alie.get('V_S_min_kJ', np.nan)

            row['I_S_mean_Ha'] = alie.get('I_S_mean_Ha', np.nan)
            row['I_S_max_Ha'] = alie.get('I_S_max_Ha', np.nan)
            row['I_S_std_Ha'] = alie.get('I_S_std_Ha', np.nan)
            row['n_surface_points'] = alie.get('n_surface_points', 0)

            rows.append(row)

    return rows


def compute_boltzmann_averages(per_conf_rows):
    """
    Calcule les moyennes de Boltzmann par molécule.

    Args:
        per_conf_rows: Liste de dicts per-conformère

    Returns:
        List[Dict] : une ligne par molécule
    """
    # Grouper par molécule
    by_mol = {}
    for row in per_conf_rows:
        mol = row['molecule_name']
        if mol not in by_mol:
            by_mol[mol] = []
        by_mol[mol].append(row)

    avg_cols = [
        'IP_kJ', 'EA_kJ', 'HOMO_Ha', 'LUMO_Ha', 'HOMO_eV', 'LUMO_eV',
        'IP_Koopmans_kJ', 'EA_Koopmans_kJ',
        'I_S_min_Ha', 'I_S_min_eV', 'I_S_min_kJ', 'IP_Politzer_kJ',
        'V_S_min_Ha', 'V_S_min_eV', 'V_S_min_kJ',
        'I_S_mean_Ha', 'I_S_max_Ha', 'I_S_std_Ha',
    ]

    mol_rows = []
    for mol_name in sorted(by_mol.keys()):
        confs = by_mol[mol_name]
        mol_row = {'molecule_name': mol_name, 'n_conformers': len(confs)}

        # Extraire les poids
        weights = np.array([c.get('weight', np.nan) for c in confs])
        valid_w = ~np.isnan(weights)
        if not np.any(valid_w):
            # Poids uniformes
            weights = np.ones(len(confs)) / len(confs)
            valid_w = np.ones(len(confs), dtype=bool)

        w_sum = np.sum(weights[valid_w])
        if w_sum == 0:
            w_sum = 1.0

        for col in avg_cols:
            vals = np.array([c.get(col, np.nan) for c in confs])
            mask = valid_w & ~np.isnan(vals)
            if np.any(mask):
                mol_row[col] = float(np.sum(vals[mask] * weights[mask]) / np.sum(weights[mask]))
            else:
                mol_row[col] = np.nan

        mol_rows.append(mol_row)

    return mol_rows


def export_csv(rows, output_path, fieldnames=None):
    """Exporte une liste de dicts en CSV."""
    if not rows:
        return

    if fieldnames is None:
        # Collecter toutes les clés dans l'ordre
        seen = set()
        fieldnames = []
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Analyse ALIE de Politzer & Murray : calcul de I_S,min et comparaison IP/EA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemples:
  ./politzer_analysis.py --batch-dir 1594174/dir_workdir --output-dir politzer_results
  ./politzer_analysis.py --workdirs DHI_workdir O-DHI_workdir --output-dir results
  ./politzer_analysis.py --batch-dir workdirs/ --no-plots --grid fine
""",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--batch-dir', type=str,
                        help="Répertoire contenant les *_workdir")
    source.add_argument('--workdirs', type=str, nargs='+',
                        help="Liste de répertoires de travail")

    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'politzer'),
                        help="Répertoire de sortie (défaut: results/politzer)")
    parser.add_argument('--grid', type=str, default='medium',
                        choices=['coarse', 'medium', 'fine'],
                        help="Qualité de grille DENSF (défaut: medium)")
    parser.add_argument('--no-plots', action='store_true',
                        help="Désactiver la génération de graphiques")
    parser.add_argument('--verbose', action='store_true',
                        help="Mode verbeux (détails DENSF)")

    args = parser.parse_args()

    # --- Résoudre les workdirs ---
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            print_error(f"Répertoire batch non trouvé: {batch_dir}")
            sys.exit(1)
        workdirs = discover_workdirs(batch_dir)
    else:
        workdirs = [Path(w) for w in args.workdirs]

    if not workdirs:
        print_error("Aucun workdir trouvé")
        sys.exit(1)

    # Exclure les molécules DEA (rupture C-X en réduction)
    n_before = len(workdirs)
    workdirs = [w for w in workdirs if w.name.replace('_workdir', '') not in DEA_EXCLUDED]
    n_excluded = n_before - len(workdirs)
    if n_excluded > 0:
        print(f"  Exclu {n_excluded} molécule(s) DEA: {', '.join(sorted(DEA_EXCLUDED))}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Header ---
    print_header("ANALYSE ALIE DE POLITZER & MURRAY")
    print(f"  Molécules:    {len(workdirs)}")
    print(f"  Grille DENSF: {args.grid}")
    print(f"  Sortie:       {output_dir}")
    print()

    # --- Initialiser PLAMS ---
    from scm.plams import init, finish
    init()

    try:
        # --- Phase 1 : Calcul ALIE par molécule/conformère ---
        print_section("Phase 1 : Calcul ALIE (DENSF post-traitement)")

        all_alie = {}  # {mol_name: {conf_num: descriptors}}
        for workdir in workdirs:
            mol_name = workdir.name.replace('_workdir', '')
            print(f"\n{mol_name}:")

            results = process_molecule(
                workdir, output_dir,
                grid_quality=args.grid,
                verbose=args.verbose,
            )
            all_alie[mol_name] = results

        # --- Phase 2 : Construction du tableau per-conformère ---
        print_section("Phase 2 : Consolidation des données")

        per_conf = build_per_conformer_table(workdirs, all_alie)
        print(f"  {len(per_conf)} conformères chargés")

        # Export per-conformer CSV
        conf_csv = output_dir / "politzer_per_conformer.csv"
        conf_fields = [
            'molecule_name', 'conformer', 'weight',
            'IP_kJ', 'EA_kJ', 'HOMO_eV', 'LUMO_eV',
            'IP_Koopmans_kJ', 'EA_Koopmans_kJ',
            'I_S_min_Ha', 'I_S_min_eV', 'IP_Politzer_kJ',
            'V_S_min_Ha', 'V_S_min_eV', 'V_S_min_kJ',
            'I_S_mean_Ha', 'I_S_max_Ha', 'I_S_std_Ha',
            'n_surface_points',
        ]
        export_csv(per_conf, conf_csv, fieldnames=conf_fields)
        print_success(f"  Per-conformère: {conf_csv}")

        # --- Phase 3 : Moyennes de Boltzmann ---
        print_section("Phase 3 : Moyennes de Boltzmann par molécule")

        mol_rows = compute_boltzmann_averages(per_conf)
        print(f"  {len(mol_rows)} molécules")

        mol_csv = output_dir / "politzer_per_molecule.csv"
        mol_fields = [
            'molecule_name', 'n_conformers',
            'IP_kJ', 'EA_kJ', 'HOMO_eV', 'LUMO_eV',
            'IP_Koopmans_kJ', 'EA_Koopmans_kJ',
            'I_S_min_eV', 'IP_Politzer_kJ',
            'V_S_min_eV', 'V_S_min_kJ',
        ]
        export_csv(mol_rows, mol_csv, fieldnames=mol_fields)
        print_success(f"  Per-molécule:   {mol_csv}")

        # Afficher le résumé
        print_section("Résumé des prédictions IP")
        print(f"  {'Molécule':<20s} {'IP_DFT':>10s} {'IP_Koopmans':>12s} {'IP_Politzer':>12s}")
        print(f"  {'':20s} {'(kJ/mol)':>10s} {'(kJ/mol)':>12s} {'(kJ/mol)':>12s}")
        print(f"  {'-'*56}")
        for row in mol_rows:
            ip_dft = row.get('IP_kJ', np.nan)
            ip_k = row.get('IP_Koopmans_kJ', np.nan)
            ip_p = row.get('IP_Politzer_kJ', np.nan)
            print(f"  {row['molecule_name']:<20s} {ip_dft:>10.1f} {ip_k:>12.1f} {ip_p:>12.1f}")

        # --- Phase 4 : Visualisation ---
        if not args.no_plots:
            print_section("Phase 4 : Génération des graphiques")
            try:
                from generate_ip_ea_comparison import generate_all_plots
                generate_all_plots(
                    conf_csv=str(conf_csv),
                    mol_csv=str(mol_csv),
                    output_dir=str(output_dir),
                )
            except ImportError:
                print_warning(
                    "generate_ip_ea_comparison.py non trouvé. "
                    "Exécutez-le séparément :\n"
                    f"  python3 generate_ip_ea_comparison.py "
                    f"--conf-csv {conf_csv} --mol-csv {mol_csv} "
                    f"--output-dir {output_dir}"
                )
            except Exception as e:
                print_warning(f"Erreur lors de la visualisation: {e}")

        print_header("TERMINÉ")
        print_success(f"Résultats dans: {output_dir}/")

    finally:
        finish()


if __name__ == '__main__':
    main()
