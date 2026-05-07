"""
Module d'analyse pour le pipeline conformères.

Ce module permet de ré-analyser des conformères existants sans refaire
les calculs DFT coûteux. Utile pour ajuster les paramètres de filtrage
ou régénérer les rapports.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from scm.plams import KFReader

from common.utils import boltzmann_weights
from common.constants import EH_TO_KJ


def extract_conf_number(filename: str) -> Optional[int]:
    """
    Extrait le numéro de conformère depuis un nom de fichier.

    Args:
        filename: Nom de fichier (ex: "conf_1.xyz", "molecule_conf_2_neutral.xyz")

    Returns:
        int: Numéro de conformère, ou None si non trouvé

    Examples:
        >>> extract_conf_number("conf_1.xyz")
        1
        >>> extract_conf_number("molecule_conf_12_neutral.xyz")
        12
    """
    import re

    # Pattern: chercher "conf" suivi de "_" et un ou plusieurs chiffres
    match = re.search(r'conf[_\s]+(\d+)', filename, re.IGNORECASE)

    if match:
        return int(match.group(1))

    return None


def load_conformers_from_workdir(
    workdir: Path,
    subfolder: str = "boltzmann_filtered"
) -> Dict[int, Dict[str, Path]]:
    """
    Charge les conformères depuis un calcul précédent.

    Cette fonction recherche les fichiers XYZ et RKF associés dans le répertoire
    de calculs. Elle permet de ré-analyser sans refaire les calculs DFT.

    Args:
        workdir: Répertoire de travail principal
        subfolder: Sous-dossier contenant les conformères (défaut: "boltzmann_filtered")

    Returns:
        dict: {conf_num: {'xyz': Path, 'rkf': Path, 'energy': float}}

    Example:
        >>> conformers = load_conformers_from_workdir(Path("molecule_workdir"))
        >>> print(conformers[1]['energy'])
        -156.123456  # Hartree
    """
    workdir = Path(workdir)
    calc_dir = workdir / "conformers" / "calculations" / subfolder

    if not calc_dir.exists():
        print(f"⚠ Répertoire de calcul introuvable: {calc_dir}")
        return {}

    conformers = {}

    # Chercher tous les fichiers XYZ
    for xyz_file in calc_dir.glob("*.xyz"):
        conf_num = extract_conf_number(xyz_file.stem)

        if conf_num is None:
            print(f"⚠ Impossible d'extraire le numéro de conformère: {xyz_file.name}")
            continue

        # Chercher le fichier RKF associé - essayer plusieurs emplacements
        rkf_path = None

        # Option 1: Dans subfolder/plams_workdir/ (cas normal)
        plams_dir = calc_dir / "plams_workdir"
        if plams_dir.exists():
            rkf_files = list(plams_dir.glob(f"*conf*{conf_num}*/adf.rkf"))
            if rkf_files:
                rkf_path = rkf_files[0]

        # Option 2: Dans 05_freq/conf_X/plams_workdir/ (après vérification fréquences)
        if rkf_path is None:
            freq_dir = workdir / "conformers" / "calculations" / "05_freq" / f"conf_{conf_num}" / "plams_workdir"
            if freq_dir.exists():
                rkf_files = list(freq_dir.glob(f"*freq*/adf.rkf"))
                if rkf_files:
                    rkf_path = rkf_files[0]

        # Option 3: Dans 04_filter/plams_workdir/ (après filtrage RMSD)
        if rkf_path is None:
            filter_dir = workdir / "conformers" / "calculations" / "04_filter" / "plams_workdir"
            if filter_dir.exists():
                rkf_files = list(filter_dir.glob(f"*conf*{conf_num}*/adf.rkf"))
                if rkf_files:
                    rkf_path = rkf_files[0]

        # Option 4: Dans 03_score/plams_workdir/ (après scoring DFT)
        if rkf_path is None:
            score_dir = workdir / "conformers" / "calculations" / "03_score" / "plams_workdir"
            if score_dir.exists():
                rkf_files = list(score_dir.glob(f"**/Score_{conf_num - 1}/adf.rkf"))  # Score indexé à 0
                if not rkf_files:
                    rkf_files = list(score_dir.glob(f"**/Score_{conf_num}/adf.rkf"))
                if rkf_files:
                    rkf_path = rkf_files[0]

        if rkf_path is None:
            print(f"⚠ RKF non trouvé pour conformère {conf_num} (cherché dans subfolder, 05_freq, 04_filter, 03_score)")
            continue

        # Lire l'énergie depuis le RKF
        try:
            kf_reader = KFReader(str(rkf_path))
            energy = kf_reader.read("Energy", "Bond Energy")

            conformers[conf_num] = {
                'xyz': xyz_file,
                'rkf': rkf_path,
                'energy': energy
            }

        except Exception as e:
            print(f"⚠ Erreur lecture RKF pour conformère {conf_num}: {e}")
            continue

    print(f"✓ {len(conformers)} conformère(s) chargé(s) depuis {subfolder}/")

    return conformers


def analyze_existing_conformers(
    workdir: Path,
    temperature: float = 298.15,
    subfolder: str = "boltzmann_filtered"
) -> Dict:
    """
    Recalcule les poids de Boltzmann sur des conformères existants.

    Cette fonction permet de ré-analyser des conformères sans refaire les calculs.
    Utile pour changer la température ou ajuster les paramètres.

    Args:
        workdir: Répertoire de travail principal
        temperature: Température en K (défaut: 298.15)
        subfolder: Sous-dossier source (défaut: "boltzmann_filtered")

    Returns:
        dict: {
            'conformers': {conf_num: {'xyz', 'rkf', 'energy'}},
            'relative_energies_kj': {conf_num: float},
            'weights': {conf_num: float},
            'partition_function': float,
            'temperature': float
        }

    Example:
        >>> results = analyze_existing_conformers(Path("molecule_workdir"))
        >>> print(results['weights'])
        {1: 0.9755, 2: 0.0245}
    """
    workdir = Path(workdir)

    # Charger les conformères
    conformers = load_conformers_from_workdir(workdir, subfolder)

    if not conformers:
        raise ValueError("Aucun conformère trouvé!")

    # Extraire les énergies
    conf_nums = sorted(conformers.keys())
    energies = [conformers[num]['energy'] for num in conf_nums]

    # Calculer énergies relatives en kJ/mol
    min_energy = min(energies)
    relative_energies_hartree = [e - min_energy for e in energies]
    relative_energies_kj = [e * EH_TO_KJ for e in relative_energies_hartree]

    # Calculer poids de Boltzmann
    weights_array, Z = boltzmann_weights(relative_energies_kj, temperature)

    # Convertir en dictionnaires
    relative_energies_dict = {
        conf_num: energy
        for conf_num, energy in zip(conf_nums, relative_energies_kj)
    }

    weights_dict = {
        conf_num: float(weight)
        for conf_num, weight in zip(conf_nums, weights_array)
    }

    return {
        'conformers': conformers,
        'relative_energies_kj': relative_energies_dict,
        'weights': weights_dict,
        'partition_function': float(Z),
        'temperature': temperature
    }


def export_conformers_summary(
    results: Dict,
    output_file: Path,
    molecule_name: str = "molecule"
) -> None:
    """
    Exporte un résumé des conformères analysés.

    Args:
        results: Résultats de analyze_existing_conformers()
        output_file: Fichier de sortie
        molecule_name: Nom de la molécule
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ANALYSE DES CONFORMÈRES - {molecule_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Température: {results['temperature']:.2f} K\n")
        f.write(f"Fonction de partition: Z = {results['partition_function']:.6e}\n")
        f.write(f"Nombre de conformères: {len(results['conformers'])}\n\n")

        f.write("-" * 80 + "\n")
        f.write("CONFORMÈRES FINAUX\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'#':<5} {'ΔE [kJ/mol]':<15} {'Poids':<12} {'%':<10}\n")
        f.write("-" * 80 + "\n")

        for conf_num in sorted(results['weights'].keys()):
            energy = results['relative_energies_kj'][conf_num]
            weight = results['weights'][conf_num]
            percentage = weight * 100

            f.write(f"{conf_num:<5} {energy:>10.3f}     {weight:>10.6f}  {percentage:>8.2f}%\n")

        f.write("-" * 80 + "\n")
        f.write(f"Z = {results['partition_function']:.6e}\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Résumé exporté: {output_file}")


def export_conformers_csv(
    results: Dict,
    output_file: Path
) -> None:
    """
    Exporte les données des conformères en CSV.

    Args:
        results: Résultats de analyze_existing_conformers()
        output_file: Fichier CSV de sortie
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        # Header
        f.write("conformer,energy_rel_kJ,boltzmann_weight,xyz_file\n")

        # Data rows
        for conf_num in sorted(results['weights'].keys()):
            energy = results['relative_energies_kj'][conf_num]
            weight = results['weights'][conf_num]
            xyz_file = results['conformers'][conf_num]['xyz'].name

            f.write(f"{conf_num},{energy:.6f},{weight:.8f},{xyz_file}\n")

    print(f"✓ CSV exporté: {output_file}")
