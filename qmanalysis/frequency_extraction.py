"""
Extraction des données de fréquences vibrationnelles depuis les fichiers RKF.

Ce module fournit des fonctions pour localiser les fichiers RKF des calculs neutres
et extraire les données de fréquences, intensités IR, et propriétés thermodynamiques.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

from .common import print_warning, hartree_to_kj


def find_neutral_rkf_files(workdir, molecule_name: str) -> Dict[int, Path]:
    """
    Localise les fichiers adf.rkf des calculs neutres (DZP opt+freq).

    Cherche d'abord dans redox/calculations/plams_workdir/<mol>_conf_N_neutral/,
    puis dans conformers/results/neutral_rkf_paths.json (pipeline conformères,
    où le neutre n'est pas recalculé dans le pipeline redox).

    Args:
        workdir (str or Path): Répertoire de travail principal de la molécule
        molecule_name (str): Nom de la molécule

    Returns:
        Dict[int, Path]: Dictionnaire {conf_num: chemin_rkf}
    """
    import json
    workdir = Path(workdir)
    rkf_files = {}

    # 1. Chercher dans redox/calculations/plams_workdir/ (ancien pattern)
    plams_workdir = workdir / "redox" / "calculations" / "plams_workdir"

    if plams_workdir.exists():
        neutral_pattern = re.compile(
            rf"^{re.escape(molecule_name)}_conf_(\d+)_neutral(?:\.\d+)?$"
        )
        for job_dir in plams_workdir.iterdir():
            if not job_dir.is_dir():
                continue
            match = neutral_pattern.match(job_dir.name)
            if match:
                conf_num = int(match.group(1))
                rkf_path = job_dir / "adf.rkf"
                if rkf_path.exists():
                    if conf_num not in rkf_files or job_dir.name > rkf_files[conf_num].parent.name:
                        rkf_files[conf_num] = rkf_path

    # 2. Fallback : neutral_rkf_paths.json (pipeline conformères)
    if not rkf_files:
        json_path = workdir / "conformers" / "results" / "neutral_rkf_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            for conf_str, rkf_str in paths.items():
                conf_num = int(conf_str)
                rkf_path = Path(rkf_str)
                if rkf_path.exists():
                    rkf_files[conf_num] = rkf_path

    if not rkf_files:
        print_warning(f"Aucun RKF neutre trouvé pour {molecule_name}")

    return rkf_files


def extract_ir_intensities(kf_reader) -> Optional[np.ndarray]:
    """
    Extrait les intensités IR depuis un fichier RKF.

    Args:
        kf_reader: Objet KFReader ouvert sur un fichier adf.rkf

    Returns:
        np.ndarray: Tableau des intensités IR (km/mol), ou None si non disponible
    """
    # Essayer différentes clés possibles pour les intensités IR
    possible_keys = [
        "Intensities[km/mol]",  # Format AMS 2025
        "Intensities",          # Format ancien
        "IR Intensities",
    ]

    for key in possible_keys:
        try:
            intensities = np.array(kf_reader.read("Vibrations", key), dtype=float)
            return intensities
        except (KeyError, Exception):
            continue

    return None


def extract_thermodynamic_data(kf_reader) -> Dict[str, float]:
    """
    Extrait les données thermodynamiques depuis un fichier RKF.

    Args:
        kf_reader: Objet KFReader ouvert sur un fichier adf.rkf

    Returns:
        Dict contenant: H, G, U, TS, T, P (ou valeurs None si non disponibles)
    """
    thermo = {}

    keys = [
        ("H", "Thermodynamics", "Enthalpy"),
        ("G", "Thermodynamics", "Gibbs free Energy"),
        ("TS", "Thermodynamics", "TS"),
        ("T", "Thermodynamics", "Temperature"),
        ("P", "Thermodynamics", "Pressure"),
    ]

    for name, section, key in keys:
        try:
            value = kf_reader.read(section, key)
            thermo[name] = float(value) if value is not None else None
        except (KeyError, Exception):
            thermo[name] = None

    # Calculer U si possible
    try:
        thermo["U"] = kf_reader.read("Thermodynamics", "Internal Energy total")
    except (KeyError, Exception):
        # Fallback: U = H - RT
        if thermo.get("H") is not None and thermo.get("T") is not None:
            from .common import R
            R_ha = R / 4.184 / 627.509  # R en Hartree/K
            thermo["U"] = thermo["H"] - R_ha * thermo["T"]
        else:
            thermo["U"] = None

    return thermo


def extract_all_frequency_data(rkf_path) -> Dict[str, Any]:
    """
    Extrait toutes les données de fréquences et propriétés associées depuis un RKF.

    Args:
        rkf_path (str or Path): Chemin vers le fichier adf.rkf

    Returns:
        Dict contenant:
            - 'frequencies': np.ndarray des fréquences en cm⁻¹
            - 'ir_intensities': np.ndarray des intensités IR (km/mol) ou None
            - 'zpe_hartree': Énergie de point zéro en Hartree
            - 'zpe_kj': Énergie de point zéro en kJ/mol
            - 'n_modes': Nombre de modes vibrationnels
            - 'thermodynamics': Dict des données thermodynamiques
            - 'success': True si l'extraction a réussi

    Example:
        >>> data = extract_all_frequency_data("path/to/adf.rkf")
        >>> if data['success']:
        ...     print(f"ZPE = {data['zpe_kj']:.2f} kJ/mol")
        ...     print(f"N modes = {data['n_modes']}")
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))

        # Extraire les fréquences
        frequencies = np.array(kf.read("Vibrations", "Frequencies[cm-1]"), dtype=float)

        # Extraire les intensités IR
        ir_intensities = extract_ir_intensities(kf)

        # Extraire la ZPE si disponible, sinon calculer depuis les fréquences
        try:
            zpe_hartree = kf.read("Vibrations", "ZeroPointEnergy")
        except (KeyError, Exception):
            # Calcul manuel: ZPE = Σ (ν_i / 2) * h * c * N_A
            # En Hartree: 1 cm⁻¹ = 4.5563e-6 Hartree
            CM_TO_HA = 4.5563e-6
            # Ne prendre que les fréquences positives pour ZPE
            pos_freqs = frequencies[frequencies > 0]
            zpe_hartree = np.sum(pos_freqs) * 0.5 * CM_TO_HA

        zpe_kj = hartree_to_kj(zpe_hartree)

        # Extraire les données thermodynamiques
        thermodynamics = extract_thermodynamic_data(kf)

        return {
            'frequencies': frequencies,
            'ir_intensities': ir_intensities,
            'zpe_hartree': zpe_hartree,
            'zpe_kj': zpe_kj,
            'n_modes': len(frequencies),
            'thermodynamics': thermodynamics,
            'success': True
        }

    except Exception as e:
        print_warning(f"Erreur extraction depuis {rkf_path}: {e}")
        return {
            'frequencies': np.array([]),
            'ir_intensities': None,
            'zpe_hartree': None,
            'zpe_kj': None,
            'n_modes': 0,
            'thermodynamics': {},
            'success': False
        }


def extract_normal_mode_vectors(rkf_path, mode_numbers: Optional[list] = None) -> Dict[int, np.ndarray]:
    """
    Extrait les vecteurs de modes normaux depuis un fichier RKF.

    Args:
        rkf_path (str or Path): Chemin vers le fichier adf.rkf
        mode_numbers (list, optional): Liste des numéros de modes à extraire.
                                       Si None, extrait tous les modes.

    Returns:
        Dict[int, np.ndarray]: Dictionnaire {mode_number: displacement_vector}
                               où displacement_vector a la forme [x1,y1,z1,x2,y2,z2,...]
    """
    modes = {}

    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))

        # Nombre total de modes
        frequencies = kf.read("Vibrations", "Frequencies[cm-1]")
        n_modes = len(frequencies)

        if mode_numbers is None:
            mode_numbers = list(range(1, n_modes + 1))

        for mode_num in mode_numbers:
            if 1 <= mode_num <= n_modes:
                try:
                    displacements = np.array(
                        kf.read("Vibrations", f"NoWeightNormalMode({mode_num})"),
                        dtype=float
                    )
                    modes[mode_num] = displacements
                except (KeyError, Exception):
                    pass

    except Exception as e:
        print_warning(f"Erreur extraction modes normaux depuis {rkf_path}: {e}")

    return modes
