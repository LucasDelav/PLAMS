"""
Analyse LVC (Linear Vibronic Coupling) pour le calcul de l'énergie de déformation.

Ce module implémente l'approche LVC qui permet de calculer l'énergie de déformation
(Edef) à partir de la théorie harmonique, sans corrélations empiriques.

Formule LVC:
    E_def = -sum_i (F_i^2 / (2 * k_i))

Où:
    F_i = Gradients(ion) · NoWeightNormalMode(i)(neutral)
    k_i = Constante de force du mode i (depuis le neutre)

Sources des données dans les RKF:
    - Gradients: AMSResults/Gradients dans *_reduced_sp/adf.rkf ou *_oxidized_sp/adf.rkf
    - Force constants: Vibrations/ForceConstants dans *_neutral/adf.rkf
    - Normal modes: Vibrations/NoWeightNormalMode(i) dans *_neutral/adf.rkf
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .common import (
    print_warning, print_error, print_success, print_section,
    hartree_to_kj, EH_TO_KJ
)


def extract_gradients(rkf_path: Path) -> Optional[np.ndarray]:
    """
    Extrait les gradients depuis AMSResults/Gradients d'un fichier RKF.

    Les gradients représentent la force sur chaque atome dans un calcul single-point.
    Ils sont stockés en Hartree/Bohr.

    Args:
        rkf_path (Path): Chemin vers le fichier adf.rkf (typiquement *_reduced_sp ou *_oxidized_sp)

    Returns:
        np.ndarray: Vecteur des gradients de dimension 3N (Hartree/Bohr), ou None si erreur
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))

        # Les gradients sont stockés dans AMSResults/Gradients
        gradients = np.array(kf.read("AMSResults", "Gradients"), dtype=float)

        return gradients

    except KeyError:
        print_warning(f"Gradients non trouvés dans {rkf_path}")
        return None
    except Exception as e:
        print_error(f"Erreur extraction gradients depuis {rkf_path}: {e}")
        return None


def extract_atomic_masses(rkf_path: Path) -> Optional[np.ndarray]:
    """
    Extrait les masses atomiques depuis un RKF et retourne un vecteur 3N.

    Chaque masse est répétée 3 fois (pour x, y, z) pour permettre
    la pondération mass-weighted des gradients et modes normaux.

    Returns:
        np.ndarray de dimension 3N avec masses en amu, ou None si erreur
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))
        masses = np.array(kf.read("Molecule", "AtomMasses"), dtype=float)
        # Répéter chaque masse 3 fois (x, y, z)
        return np.repeat(masses, 3)
    except Exception as e:
        print_warning(f"Impossible d'extraire les masses depuis {rkf_path}: {e}")
        return None


def extract_normal_modes_and_fc(rkf_path: Path) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extrait tous les modes normaux et les constantes de force depuis le neutre.

    Args:
        rkf_path (Path): Chemin vers le fichier adf.rkf du calcul neutre

    Returns:
        Tuple contenant:
            - normal_modes (List[np.ndarray]): Liste des vecteurs de modes normaux (3N chacun)
            - force_constants (np.ndarray): Constantes de force en Hartree/Bohr²
            - frequencies (np.ndarray): Fréquences en cm⁻¹

    Note:
        Les modes normaux sont les NoWeightNormalMode, normalisés à 1.
        Les constantes de force sont en unités Hartree/Bohr².
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))

        # Extraire les fréquences pour connaître le nombre de modes
        frequencies = np.array(kf.read("Vibrations", "Frequencies[cm-1]"), dtype=float)
        n_modes = len(frequencies)

        # Extraire les constantes de force
        force_constants = np.array(kf.read("Vibrations", "ForceConstants"), dtype=float)

        # Extraire tous les modes normaux (NoWeightNormalMode)
        normal_modes = []
        for i in range(1, n_modes + 1):
            try:
                mode = np.array(kf.read("Vibrations", f"NoWeightNormalMode({i})"), dtype=float)
                normal_modes.append(mode)
            except (KeyError, Exception) as e:
                print_warning(f"Mode {i} non trouvé: {e}")
                normal_modes.append(None)

        return normal_modes, force_constants, frequencies

    except KeyError as e:
        print_warning(f"Données vibrationnelles manquantes dans {rkf_path}: {e}")
        return None, None, None
    except Exception as e:
        print_error(f"Erreur extraction modes normaux depuis {rkf_path}: {e}")
        return None, None, None


def compute_lvc_edef(
    gradients: np.ndarray,
    normal_modes: List[np.ndarray],
    force_constants: np.ndarray,
    frequencies: np.ndarray,
    freq_threshold: float = 50.0,
    masses_3n: Optional[np.ndarray] = None
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Calcule E_def par la méthode LVC (Linear Vibronic Coupling).

    Formule: E_def = -sum_i (F_i^2 / (2 * k_i))

    Où F_i est la projection du gradient ionique sur le mode normal i.

    Si masses_3n est fourni, la projection est mass-weighted :
        F_i = (m · g) · N_i
    Ceci est nécessaire quand les modes et les gradients viennent de RKF
    différents (LVC ordre 1), car les ForceConstants sont en espace
    mass-weighted (k_i = ω_i²) tandis que NoWeightNormalMode est en
    espace cartésien.

    Si masses_3n est None, projection cartésienne directe (LVC2, modes
    et FC du même Hessien — la cohérence interne rend la mass-weighting
    inutile).

    Args:
        gradients (np.ndarray): Gradients de l'ion (Hartree/Bohr), dimension 3N
        normal_modes (List[np.ndarray]): Modes normaux, chacun de dimension 3N
        force_constants (np.ndarray): Constantes de force (Hartree/Bohr²)
        frequencies (np.ndarray): Fréquences en cm⁻¹ (pour filtrage)
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)
        masses_3n (np.ndarray, optional): Masses atomiques répétées 3x (amu),
            dimension 3N. Si fourni, active la projection mass-weighted.

    Returns:
        Tuple (E_def_total_kj, contributions)
    """
    contributions = []
    E_def_total_hartree = 0.0

    n_modes = len(normal_modes)

    for i in range(n_modes):
        mode = normal_modes[i]
        freq = frequencies[i]
        k_i = force_constants[i]

        # Ignorer les modes invalides ou avec fréquence trop basse
        if mode is None:
            continue
        if freq < freq_threshold:
            continue
        if k_i <= 0:
            continue

        # Projection du gradient sur le mode normal
        if masses_3n is not None:
            # Projection mass-weighted pour cohérence avec ForceConstants
            F_i = np.dot(masses_3n * gradients, mode)
        else:
            F_i = np.dot(gradients, mode)

        # Contribution à E_def: -F_i² / (2 * k_i)
        contribution_hartree = -(F_i ** 2) / (2.0 * k_i)
        contribution_kj = contribution_hartree * EH_TO_KJ

        E_def_total_hartree += contribution_hartree

        contributions.append({
            'mode_num': i + 1,  # 1-indexé
            'frequency': freq,
            'F_i': F_i,
            'k_i': k_i,
            'contribution_hartree': contribution_hartree,
            'contribution_kj': contribution_kj,
        })

    E_def_total_kj = E_def_total_hartree * EH_TO_KJ

    # Trier les contributions par valeur absolue (plus grande contribution en premier)
    contributions.sort(key=lambda x: abs(x['contribution_kj']), reverse=True)

    return E_def_total_kj, contributions


def compute_lvc2_edef_from_rkf(
    ion_sp_rkf: Path,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Calcule E_def par la méthode LVC d'ordre 2 (modes ioniques du SP).

    Utilise les modes normaux et constantes de force de l'ion calculés à la
    géométrie neutre (extraits du fichier RKF du single-point ionique).
    Ceci est le "vrai ordre 2" du développement de Taylor de V_ion, car
    on utilise le Hessien de la PES ionique évalué à la géométrie neutre.

    Formule: E_def = -sum_j (F'_j² / (2 * k'_j))

    Où F'_j = gradient · mode_ion_j (projection du gradient sur le mode ionique j)
    et k'_j = constante de force du mode ionique j, toutes deux extraites du SP.

    Avantage: ne nécessite AUCUN calcul supplémentaire (les données sont déjà
    dans le fichier RKF du single-point ionique).

    Args:
        ion_sp_rkf: Chemin vers le RKF du single-point ionique (reduced_sp ou oxidized_sp)
        freq_threshold: Seuil minimal de fréquence (cm⁻¹). Les modes imaginaires
                       (fréquence < 0) sont automatiquement exclus.
        top_n_modes: Nombre de modes dominants à retourner

    Returns:
        Dict avec 'success', 'E_def_lvc2_kj', 'n_modes_used', 'n_imag', etc.
    """
    result = {
        'success': False,
        'E_def_lvc2_kj': None,
        'E_def_lvc2_hartree': None,
        'n_modes_used': 0,
        'n_modes_total': 0,
        'n_imag': 0,
        'top_contributions': [],
        'all_contributions': [],
        'error': None,
    }

    # Extraire gradient et modes/FC depuis le même fichier SP
    gradients = extract_gradients(ion_sp_rkf)
    if gradients is None:
        result['error'] = f"Impossible d'extraire les gradients depuis {ion_sp_rkf}"
        return result

    ion_modes, ion_fc, ion_freq = extract_normal_modes_and_fc(ion_sp_rkf)
    if ion_modes is None or ion_fc is None:
        result['error'] = f"Impossible d'extraire les modes/FC ioniques depuis {ion_sp_rkf}"
        return result

    result['n_modes_total'] = len(ion_freq)
    result['n_imag'] = int(np.sum(ion_freq < 0))

    # Calculer LVC2 avec les modes ioniques du SP
    E_def_kj, contributions = compute_lvc_edef(
        gradients, ion_modes, ion_fc, ion_freq,
        freq_threshold=freq_threshold
    )

    result['success'] = True
    result['E_def_lvc2_kj'] = E_def_kj
    result['E_def_lvc2_hartree'] = E_def_kj / EH_TO_KJ
    result['n_modes_used'] = len(contributions)
    result['top_contributions'] = contributions[:top_n_modes]
    result['all_contributions'] = contributions

    return result


def compute_lvc_edef_from_rkf(
    neutral_rkf: Path,
    ion_sp_rkf: Path,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Calcule E_def LVC à partir des fichiers RKF.

    Cette fonction est le point d'entrée principal pour calculer E_def par LVC.
    Elle extrait automatiquement les données nécessaires et effectue le calcul.

    Args:
        neutral_rkf (Path): Chemin vers le RKF du calcul neutre optimisé
        ion_sp_rkf (Path): Chemin vers le RKF du calcul single-point de l'ion
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)
        top_n_modes (int): Nombre de modes dominants à retourner

    Returns:
        Dict contenant:
            - 'success': True si le calcul a réussi
            - 'E_def_lvc_kj': Énergie de déformation LVC totale (kJ/mol)
            - 'E_def_lvc_hartree': Énergie de déformation LVC totale (Hartree)
            - 'n_modes_used': Nombre de modes utilisés
            - 'n_modes_total': Nombre total de modes vibrationnels
            - 'top_contributions': Liste des top_n_modes contributions dominantes
            - 'all_contributions': Liste de toutes les contributions par mode
            - 'error': Message d'erreur si success=False
    """
    result = {
        'success': False,
        'E_def_lvc_kj': None,
        'E_def_lvc_hartree': None,
        'n_modes_used': 0,
        'n_modes_total': 0,
        'top_contributions': [],
        'all_contributions': [],
        'error': None,
    }

    # 1. Extraire les gradients de l'ion
    gradients = extract_gradients(ion_sp_rkf)
    if gradients is None:
        result['error'] = f"Impossible d'extraire les gradients depuis {ion_sp_rkf}"
        return result

    # 2. Extraire les modes normaux et constantes de force du neutre
    normal_modes, force_constants, frequencies = extract_normal_modes_and_fc(neutral_rkf)
    if normal_modes is None or force_constants is None:
        result['error'] = f"Impossible d'extraire les données vibrationnelles depuis {neutral_rkf}"
        return result

    # 3. Vérifier la cohérence des dimensions
    n_atoms_grad = len(gradients) // 3
    n_atoms_mode = len(normal_modes[0]) // 3 if normal_modes[0] is not None else 0

    if n_atoms_grad != n_atoms_mode:
        result['error'] = f"Incohérence dimensions: {n_atoms_grad} atomes (gradient) vs {n_atoms_mode} atomes (modes)"
        return result

    # 4. Calculer E_def LVC (projection cartésienne, pas de mass-weighting)
    E_def_kj, contributions = compute_lvc_edef(
        gradients, normal_modes, force_constants, frequencies,
        freq_threshold=freq_threshold
    )

    # 5. Remplir les résultats
    result['success'] = True
    result['E_def_lvc_kj'] = E_def_kj
    result['E_def_lvc_hartree'] = E_def_kj / EH_TO_KJ
    result['n_modes_used'] = len(contributions)
    result['n_modes_total'] = len(normal_modes)
    result['top_contributions'] = contributions[:top_n_modes]
    result['all_contributions'] = contributions

    return result


def find_sp_rkf_files(workdir: Path, molecule_name: str, state: str,
                      suffix: str = 'sp_freq') -> Dict[int, Path]:
    """
    Localise les fichiers RKF des calculs single-point.

    Cherche d'abord les répertoires *_{state}_{suffix} (ex: _reduced_sp_grad),
    puis retombe sur l'ancien pattern *_{state}_sp si rien n'est trouvé.

    Args:
        workdir (Path): Répertoire de travail principal
        molecule_name (str): Nom de la molécule
        state (str): 'reduced' ou 'oxidized'
        suffix (str): Suffixe du répertoire SP ('sp_grad', 'sp_freq', ou 'sp')

    Returns:
        Dict[int, Path]: {conf_num: chemin_rkf}
    """
    workdir = Path(workdir)
    plams_workdir = workdir / "redox" / "calculations" / "plams_workdir"

    if not plams_workdir.exists():
        print_warning(f"Répertoire plams_workdir non trouvé: {plams_workdir}")
        return {}

    def _scan(pattern_suffix):
        found = {}
        sp_pattern = re.compile(
            rf"^{re.escape(molecule_name)}_conf_(\d+)_{state}_{pattern_suffix}(?:\.\d+)?$"
        )
        for job_dir in plams_workdir.iterdir():
            if not job_dir.is_dir():
                continue
            match = sp_pattern.match(job_dir.name)
            if match:
                conf_num = int(match.group(1))
                rkf_path = job_dir / "adf.rkf"
                if not rkf_path.exists():
                    rkf_path = job_dir / "ams.results" / "adf.rkf"
                if rkf_path.exists():
                    if conf_num not in found or job_dir.name > found[conf_num].parent.name:
                        found[conf_num] = rkf_path
        return found

    rkf_files = _scan(suffix)

    # Fallback sur l'ancien pattern '_sp' si rien trouvé
    if not rkf_files and suffix != 'sp':
        rkf_files = _scan('sp')

    return rkf_files


def run_lvc_analysis_for_molecule(
    workdir: Path,
    molecule_name: str,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Exécute l'analyse LVC complète pour une molécule.

    Args:
        workdir (Path): Répertoire de travail de la molécule
        molecule_name (str): Nom de la molécule
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)
        top_n_modes (int): Nombre de modes dominants à rapporter

    Returns:
        Dict contenant les résultats LVC pour réduction et oxydation par conformère
    """
    from .frequency_extraction import find_neutral_rkf_files

    workdir = Path(workdir)

    # Localiser les fichiers RKF
    neutral_rkfs = find_neutral_rkf_files(workdir, molecule_name)
    # LVC et LVC2 : sp_freq contient gradients + modes DZP (base cohérente avec neutre)
    reduced_sp_freq_rkfs = find_sp_rkf_files(workdir, molecule_name, 'reduced', 'sp_freq')
    oxidized_sp_freq_rkfs = find_sp_rkf_files(workdir, molecule_name, 'oxidized', 'sp_freq')

    results = {
        'molecule': molecule_name,
        'conformers': {},
        'summary': {},
    }

    # Traiter chaque conformère
    for conf_num in sorted(neutral_rkfs.keys()):
        neutral_rkf = neutral_rkfs.get(conf_num)
        # LVC utilise gradients DZP (sp_freq) pour cohérence de base avec neutre DZP
        reduced_sp_rkf = reduced_sp_freq_rkfs.get(conf_num)
        oxidized_sp_rkf = oxidized_sp_freq_rkfs.get(conf_num)
        reduced_sp_freq_rkf = reduced_sp_freq_rkfs.get(conf_num)
        oxidized_sp_freq_rkf = oxidized_sp_freq_rkfs.get(conf_num)

        conf_results = {
            'reduction': None,
            'oxidation': None,
            'reduction_lvc2': None,
            'oxidation_lvc2': None,
        }

        # LVC pour réduction (gradients TZ2P + modes neutres)
        if neutral_rkf and reduced_sp_rkf:
            lvc_red = compute_lvc_edef_from_rkf(
                neutral_rkf, reduced_sp_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            conf_results['reduction'] = lvc_red

        # LVC2 pour réduction (modes ioniques DZP)
        if reduced_sp_freq_rkf:
            lvc2_red = compute_lvc2_edef_from_rkf(
                reduced_sp_freq_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            conf_results['reduction_lvc2'] = lvc2_red

        # LVC pour oxydation (gradients TZ2P + modes neutres)
        if neutral_rkf and oxidized_sp_rkf:
            lvc_ox = compute_lvc_edef_from_rkf(
                neutral_rkf, oxidized_sp_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            conf_results['oxidation'] = lvc_ox

        # LVC2 pour oxydation (modes ioniques DZP)
        if oxidized_sp_freq_rkf:
            lvc2_ox = compute_lvc2_edef_from_rkf(
                oxidized_sp_freq_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            conf_results['oxidation_lvc2'] = lvc2_ox

        results['conformers'][conf_num] = conf_results

    return results


def load_real_edef(workdir: Path, molecule_name: str) -> Dict[int, Dict[str, float]]:
    """
    Charge les Edef réels (E_opt - E_sp) depuis les résultats redox.

    Args:
        workdir (Path): Répertoire de travail
        molecule_name (str): Nom de la molécule

    Returns:
        Dict[int, Dict]: {conf_num: {'Edef_red_kJ': float, 'Edef_ox_kJ': float}}
    """
    from .correlation_analysis import load_edef_data
    return load_edef_data(workdir, molecule_name)


def extract_coordinates_from_rkf(rkf_path: Path) -> Optional[np.ndarray]:
    """
    Extrait les coordonnées atomiques depuis un fichier RKF.

    Args:
        rkf_path (Path): Chemin vers le fichier adf.rkf

    Returns:
        np.ndarray: Coordonnées en Angstrom, shape (n_atoms, 3), ou None si erreur
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))

        # Les coordonnées sont en Bohr dans le RKF
        coords_bohr = np.array(kf.read("Molecule", "Coords"), dtype=float)
        n_atoms = kf.read("Molecule", "nAtoms")

        # Conversion Bohr -> Angstrom
        bohr_to_ang = 0.529177249
        coords_ang = coords_bohr.reshape(n_atoms, 3) * bohr_to_ang

        return coords_ang

    except Exception as e:
        print_warning(f"Erreur extraction coordonnées depuis {rkf_path}: {e}")
        return None


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calcule le RMSD (Root Mean Square Deviation) entre deux géométries.

    Le RMSD est calculé après alignement optimal (minimisation par rotation/translation).

    Args:
        coords1 (np.ndarray): Coordonnées de référence, shape (n_atoms, 3)
        coords2 (np.ndarray): Coordonnées à comparer, shape (n_atoms, 3)

    Returns:
        float: RMSD en Angstrom
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"Dimensions incompatibles: {coords1.shape} vs {coords2.shape}")

    # Centrer les deux structures
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    # Alignement optimal par SVD (Kabsch algorithm)
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)

    # Gérer les réflexions
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])

    R = Vt.T @ D @ U.T
    c2_aligned = c2 @ R

    # Calculer le RMSD
    diff = c1 - c2_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


def find_opt_rkf_files(workdir: Path, molecule_name: str, state: str) -> Dict[int, Path]:
    """
    Localise les fichiers RKF des calculs d'optimisation (reduced_opt ou oxidized_opt).

    Args:
        workdir (Path): Répertoire de travail principal
        molecule_name (str): Nom de la molécule
        state (str): 'reduced' ou 'oxidized'

    Returns:
        Dict[int, Path]: {conf_num: chemin_rkf}
    """
    workdir = Path(workdir)
    rkf_files = {}

    plams_workdir = workdir / "redox" / "calculations" / "plams_workdir"

    if not plams_workdir.exists():
        return rkf_files

    # Pattern pour les calculs OPT (exclure les _OLD)
    opt_pattern = re.compile(rf"^{re.escape(molecule_name)}_conf_(\d+)_{state}_opt$")

    for job_dir in plams_workdir.iterdir():
        if not job_dir.is_dir():
            continue
        if "_OLD" in job_dir.name:
            continue

        match = opt_pattern.match(job_dir.name)
        if match:
            conf_num = int(match.group(1))
            rkf_path = job_dir / "adf.rkf"

            if rkf_path.exists():
                rkf_files[conf_num] = rkf_path

    return rkf_files


def compute_rmsd_for_molecule(
    workdir: Path,
    molecule_name: str
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Calcule le RMSD entre géométries neutre et réduites/oxydées pour tous les conformères.

    Args:
        workdir (Path): Répertoire de travail
        molecule_name (str): Nom de la molécule

    Returns:
        Dict[int, Dict]: {conf_num: {'rmsd_red': float, 'rmsd_ox': float}}
            - rmsd_red: RMSD entre neutral_opt et reduced_opt
            - rmsd_ox: RMSD entre neutral_opt et oxidized_opt
    """
    from .frequency_extraction import find_neutral_rkf_files

    workdir = Path(workdir)

    # Localiser les fichiers RKF
    neutral_rkfs = find_neutral_rkf_files(workdir, molecule_name)
    reduced_opt_rkfs = find_opt_rkf_files(workdir, molecule_name, 'reduced')
    oxidized_opt_rkfs = find_opt_rkf_files(workdir, molecule_name, 'oxidized')

    results = {}

    for conf_num in sorted(neutral_rkfs.keys()):
        neutral_rkf = neutral_rkfs.get(conf_num)
        reduced_opt_rkf = reduced_opt_rkfs.get(conf_num)
        oxidized_opt_rkf = oxidized_opt_rkfs.get(conf_num)

        conf_results = {
            'rmsd_red': None,
            'rmsd_ox': None,
        }

        # Extraire les coordonnées du neutre
        if neutral_rkf:
            coords_neutral = extract_coordinates_from_rkf(neutral_rkf)
        else:
            coords_neutral = None

        # RMSD neutre vs réduit
        if coords_neutral is not None and reduced_opt_rkf:
            coords_reduced = extract_coordinates_from_rkf(reduced_opt_rkf)
            if coords_reduced is not None:
                try:
                    conf_results['rmsd_red'] = compute_rmsd(coords_neutral, coords_reduced)
                except Exception as e:
                    print_warning(f"Erreur RMSD réduction conf {conf_num}: {e}")

        # RMSD neutre vs oxydé
        if coords_neutral is not None and oxidized_opt_rkf:
            coords_oxidized = extract_coordinates_from_rkf(oxidized_opt_rkf)
            if coords_oxidized is not None:
                try:
                    conf_results['rmsd_ox'] = compute_rmsd(coords_neutral, coords_oxidized)
                except Exception as e:
                    print_warning(f"Erreur RMSD oxydation conf {conf_num}: {e}")

        results[conf_num] = conf_results

    return results
