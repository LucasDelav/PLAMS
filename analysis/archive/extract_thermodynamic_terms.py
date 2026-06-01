#!/usr/bin/env amspython
"""
Extraction des termes thermodynamiques (TΔS) depuis les fichiers RKF redox.

Pour chaque molécule, lit TS (en Hartree) depuis les RKF des états neutres,
réduit optimisé et oxydé optimisé. Calcule TΔS par conformère puis moyenne
par poids de Boltzmann.

Usage:
    source ~/ams2025.102/amsbashrc.sh
    amspython extract_thermodynamic_terms.py \
        --batch-dir 1594174/dir_workdir \
        --output-csv rapport_full_decomposition/thermodynamic_terms.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scm.plams import KFReader

EH_TO_KJ = 2625.499638  # Hartree -> kJ/mol


def read_boltzmann_weights_simple(workdir):
    """Lecture des poids de Boltzmann depuis conformers_summary.txt."""
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


def find_conformer_numbers(workdir, molecule_name):
    """Trouve les numéros de conformères depuis le répertoire redox."""
    plams_dir = Path(workdir) / "redox" / "calculations" / "plams_workdir"
    if not plams_dir.exists():
        return []

    conf_nums = set()
    for d in plams_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"{molecule_name}_conf_") and d.name.endswith("_neutral"):
            num_str = d.name.replace(f"{molecule_name}_conf_", "").replace("_neutral", "")
            try:
                conf_nums.add(int(num_str))
            except ValueError:
                pass
    return sorted(conf_nums)


def read_ts_from_rkf(rkf_path):
    """Lit TS (en Hartree) depuis un fichier RKF."""
    try:
        kf = KFReader(str(rkf_path))
        ts = kf.read("Thermodynamics", "TS")
        return ts
    except Exception as e:
        print(f"    ⚠ Cannot read TS from {rkf_path}: {e}")
        return None


def extract_tds_for_molecule(workdir, molecule_name):
    """Extrait TΔS pour réduction et oxydation, moyenné par Boltzmann."""
    workdir = Path(workdir)
    plams_dir = workdir / "redox" / "calculations" / "plams_workdir"

    conf_nums = find_conformer_numbers(workdir, molecule_name)
    if not conf_nums:
        return None

    weights = read_boltzmann_weights_simple(workdir)

    all_tds_red = []
    all_tds_ox = []
    all_weights = []

    for conf_num in conf_nums:
        neutral_rkf = plams_dir / f"{molecule_name}_conf_{conf_num}_neutral" / "adf.rkf"
        reduced_rkf = plams_dir / f"{molecule_name}_conf_{conf_num}_reduced_opt" / "adf.rkf"
        oxidized_rkf = plams_dir / f"{molecule_name}_conf_{conf_num}_oxidized_opt" / "adf.rkf"

        if not all(p.exists() for p in [neutral_rkf, reduced_rkf, oxidized_rkf]):
            print(f"    ⚠ conf_{conf_num}: missing RKF files")
            continue

        ts_neutral = read_ts_from_rkf(neutral_rkf)
        ts_reduced = read_ts_from_rkf(reduced_rkf)
        ts_oxidized = read_ts_from_rkf(oxidized_rkf)

        if any(x is None for x in [ts_neutral, ts_reduced, ts_oxidized]):
            continue

        tds_red = (ts_reduced - ts_neutral) * EH_TO_KJ
        tds_ox = (ts_oxidized - ts_neutral) * EH_TO_KJ

        w = 1.0
        if weights:
            w = weights.get(f"conf_{conf_num}", 0.0)
            if w == 0:
                continue

        all_tds_red.append(tds_red)
        all_tds_ox.append(tds_ox)
        all_weights.append(w)

    if not all_tds_red:
        return None

    total_w = sum(all_weights)
    avg_tds_red = sum(v * w for v, w in zip(all_tds_red, all_weights)) / total_w
    avg_tds_ox = sum(v * w for v, w in zip(all_tds_ox, all_weights)) / total_w

    return {
        'TdS_red_kJ': avg_tds_red,
        'TdS_ox_kJ': avg_tds_ox,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraction de TΔS depuis les fichiers RKF redox"
    )
    parser.add_argument('--batch-dir', type=str, required=True,
                        help="Répertoire contenant les workdirs")
    parser.add_argument('--output-csv', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'full_decomposition' / 'thermodynamic_terms.csv'),
                        help="Fichier CSV de sortie")
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXTRACTION DES TERMES THERMODYNAMIQUES (TΔS)")
    print("=" * 80)

    batch_dir = Path(args.batch_dir)
    workdirs = sorted([d for d in batch_dir.glob("*_workdir") if d.is_dir()])
    print(f"Workdirs: {len(workdirs)}")

    results = []
    for workdir in workdirs:
        mol_name = workdir.name.replace('_workdir', '')
        print(f"\n  {mol_name}:", end=" ")
        tds = extract_tds_for_molecule(workdir, mol_name)
        if tds:
            tds['molecule_name'] = mol_name
            results.append(tds)
            print(f"✓ TΔS_red={tds['TdS_red_kJ']:.2f}, TΔS_ox={tds['TdS_ox_kJ']:.2f} kJ/mol")
        else:
            print("⚠ pas de donnees")

    if not results:
        print("\nERREUR: aucun terme extrait")
        sys.exit(1)

    fieldnames = ['molecule_name', 'TdS_red_kJ', 'TdS_ox_kJ']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    print(f"\n✓ {len(results)} molecules extraites -> {output_csv}")


if __name__ == "__main__":
    main()
