#!/usr/bin/env amspython
"""
Extraction des descripteurs fréquentiels depuis les fichiers RKF neutres.
Sauvegarde un CSV simple (sans pandas) utilisable ensuite par python3.

Usage:
    source ~/ams2025.102/amsbashrc.sh
    amspython extract_freq_descriptors.py \
        --batch-dir 1594174/dir_workdir \
        --output-csv rapport_erelax_correlations/freq_descriptors.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmanalysis.frequency_extraction import find_neutral_rkf_files, extract_all_frequency_data
from qmanalysis.frequency_descriptors import compute_frequency_descriptors


def read_boltzmann_weights_simple(workdir):
    """Lecture des poids de Boltzmann sans pandas."""
    summary_file = Path(workdir) / "conformers" / "results" / "conformers_summary.txt"
    if not summary_file.exists():
        return None

    weights = {}
    in_table = False
    with open(summary_file, 'r') as f:
        for line in f:
            if "CONFORMÈRES FINAUX" in line or "FINAL CONFORMERS" in line:
                in_table = True
                continue
            if line.strip().startswith("===") or line.strip().startswith("---"):
                continue
            if in_table and line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        conf_num = int(parts[0])
                        weight = float(parts[2])
                        weights[f"conf_{conf_num}"] = weight
                    except (ValueError, IndexError):
                        pass
    return weights if weights else None


def extract_freq_for_molecule(workdir, molecule_name):
    """Extrait et moyenne (Boltzmann) les descripteurs fréquentiels."""
    workdir = Path(workdir)
    rkf_files = find_neutral_rkf_files(workdir, molecule_name)
    if not rkf_files:
        return None

    weights = read_boltzmann_weights_simple(workdir)

    all_descriptors = []
    all_weights = []

    for conf_num, rkf_path in sorted(rkf_files.items()):
        try:
            freq_data = extract_all_frequency_data(rkf_path)
            if freq_data is None:
                continue
            descriptors = compute_frequency_descriptors(freq_data)
            if descriptors is None:
                continue

            w = 1.0
            if weights:
                w = weights.get(f"conf_{conf_num}", 0.0)
                if w == 0:
                    continue

            all_descriptors.append(descriptors)
            all_weights.append(w)
        except Exception as e:
            print(f"    ⚠ conf_{conf_num}: {e}")

    if not all_descriptors:
        return None

    total_weight = sum(all_weights)
    averaged = {}
    for key in all_descriptors[0]:
        values = [d.get(key, float('nan')) for d in all_descriptors]
        if any(np.isnan(v) for v in values):
            averaged[key] = float('nan')
        else:
            averaged[key] = sum(v * w for v, w in zip(values, all_weights)) / total_weight

    return averaged


def extract_freq_per_conformer(workdir, molecule_name):
    """Extrait les descripteurs fréquentiels par conformère (sans moyenner)."""
    workdir = Path(workdir)
    rkf_files = find_neutral_rkf_files(workdir, molecule_name)
    if not rkf_files:
        return []

    results = []
    for conf_num, rkf_path in sorted(rkf_files.items()):
        try:
            freq_data = extract_all_frequency_data(rkf_path)
            if freq_data is None:
                continue
            descriptors = compute_frequency_descriptors(freq_data)
            if descriptors is None:
                continue

            descriptors['molecule_name'] = molecule_name
            descriptors['conformer'] = conf_num
            results.append(descriptors)
        except Exception as e:
            print(f"    ⚠ conf_{conf_num}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extraction des descripteurs frequentiels depuis les RKF"
    )
    parser.add_argument('--batch-dir', type=str, required=True)
    parser.add_argument('--output-csv', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'erelax_correlations' / 'freq_descriptors.csv'))
    parser.add_argument('--per-conformer', action='store_true', default=False,
                        help="Ecrire une ligne par conformere (sans moyenner)")
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXTRACTION DES DESCRIPTEURS FREQUENTIELS")
    if args.per_conformer:
        print("  Mode: per-conformere (pas de moyenne Boltzmann)")
    print("=" * 80)

    batch_dir = Path(args.batch_dir)
    workdirs = sorted([d for d in batch_dir.glob("*_workdir") if d.is_dir()])
    print(f"Workdirs: {len(workdirs)}")

    results = []
    for workdir in workdirs:
        mol_name = workdir.name.replace('_workdir', '')
        print(f"\n  {mol_name}:", end=" ")

        if args.per_conformer:
            conf_results = extract_freq_per_conformer(workdir, mol_name)
            if conf_results:
                results.extend(conf_results)
                print(f"✓ ({len(conf_results)} conformeres)")
            else:
                print("⚠ pas de donnees")
        else:
            descriptors = extract_freq_for_molecule(workdir, mol_name)
            if descriptors:
                descriptors['molecule_name'] = mol_name
                results.append(descriptors)
                print(f"✓ ({len(descriptors) - 1} descripteurs)")
            else:
                print("⚠ pas de donnees")

    if not results:
        print("\nERREUR: aucun descripteur extrait")
        sys.exit(1)

    # Écrire le CSV sans pandas
    if args.per_conformer:
        id_keys = ['molecule_name', 'conformer']
    else:
        id_keys = ['molecule_name']
    desc_keys = sorted([k for k in results[0] if k not in id_keys])
    all_keys = id_keys + desc_keys

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in all_keys})

    if args.per_conformer:
        print(f"\n✓ {len(results)} conformeres extraits -> {output_csv}")
    else:
        print(f"\n✓ {len(results)} molecules extraites -> {output_csv}")
    print(f"  {len(desc_keys)} descripteurs")


if __name__ == "__main__":
    main()
