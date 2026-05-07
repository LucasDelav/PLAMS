"""
Analyse QVC (Quadratic Vibronic Coupling) pour le calcul de l'énergie de déformation.

Ce module implémente l'approche QVC qui ajoute des corrections quadratiques au LVC
pour tenir compte des changements de fréquence entre états.

Formules QVC:
    λ_QVC = λ_LVC × (1 + correction_quadratique)

    correction = Σᵢ (Δωᵢ/ωᵢ) × (λᵢ/λ_total)

Estimation des changements de fréquence:
    Sans calculer les modes de l'ion, on estime Δω via:
    Δωᵢ/ωᵢ ≈ α × |Fᵢ|/√kᵢ

    où α est un paramètre empirique (~0.1-0.5) déterminé par fitting.

Comparaison des méthodes:
    - LVC: -Σ Fᵢ²/2kᵢ              (linéaire uniquement)
    - QVC: LVC × (1 + corr)         (linéaire + quadratique)
    - Duschinsky: Σ ωᵢDᵢ²/2         (déplacement direct)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .common import (
    print_warning, print_error,
    EH_TO_KJ
)
from .lvc_analysis import (
    extract_gradients,
    extract_normal_modes_and_fc,
    compute_lvc_edef,
    find_sp_rkf_files,
)
from .frequency_extraction import find_neutral_rkf_files


# ============================================================================
# ESTIMATION DES CHANGEMENTS DE FRÉQUENCE
# ============================================================================

def estimate_frequency_shift_ratio(
    F_i: float,
    k_i: float,
    omega_i: float,
    alpha: float = 0.3
) -> float:
    """
    Estime le ratio Δω_i/ω_i depuis le gradient et la constante de force.

    L'approximation utilisée est:
        Δωᵢ/ωᵢ ≈ α × |Fᵢ|/√kᵢ

    Cette formule est dérivée de l'observation que les modes avec de grands
    gradients tendent à avoir des fréquences modifiées proportionnellement.

    Args:
        F_i (float): Projection du gradient sur le mode i (Hartree/Bohr)
        k_i (float): Constante de force du mode i (Hartree/Bohr²)
        omega_i (float): Fréquence du mode i en cm⁻¹
        alpha (float): Paramètre empirique (défaut: 0.3, plage typique 0.1-0.5)

    Returns:
        float: Ratio Δω_i/ω_i (adimensionnel, peut être positif ou négatif)

    Note:
        Le signe du ratio indique si la fréquence augmente (positif) ou
        diminue (négatif) dans l'état ionique.
    """
    if k_i <= 0 or omega_i <= 0:
        return 0.0

    # |F_i|/√k_i a des unités de √(Hartree) = √Eh
    # Le ratio Δω/ω est adimensionnel
    ratio = alpha * np.abs(F_i) / np.sqrt(k_i)

    # Le signe de F_i peut indiquer la direction du changement
    # mais pour une estimation simple, on utilise le module
    # car la relation entre gradient et changement de fréquence est complexe

    return ratio


def estimate_all_frequency_shifts(
    gradients: np.ndarray,
    normal_modes: List[np.ndarray],
    force_constants: np.ndarray,
    frequencies: np.ndarray,
    alpha: float = 0.3,
    freq_threshold: float = 50.0
) -> np.ndarray:
    """
    Estime les ratios Δω_i/ω_i pour tous les modes.

    Args:
        gradients (np.ndarray): Gradients de l'ion (Hartree/Bohr), dimension 3N
        normal_modes (List[np.ndarray]): Modes normaux du neutre
        force_constants (np.ndarray): Constantes de force (Hartree/Bohr²)
        frequencies (np.ndarray): Fréquences en cm⁻¹
        alpha (float): Paramètre empirique
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)

    Returns:
        np.ndarray: Ratios Δω_i/ω_i pour chaque mode (n_modes,)
    """
    n_modes = len(normal_modes)
    shifts = np.zeros(n_modes)

    for i in range(n_modes):
        mode = normal_modes[i]
        freq = frequencies[i]
        k_i = force_constants[i]

        if mode is None or freq < freq_threshold or k_i <= 0:
            continue

        # Projection du gradient sur le mode
        F_i = np.dot(gradients, mode)

        # Estimer le ratio de changement de fréquence
        shifts[i] = estimate_frequency_shift_ratio(F_i, k_i, freq, alpha)

    return shifts


# ============================================================================
# CALCUL DE LA CORRECTION QVC
# ============================================================================

def compute_qvc_correction(
    lvc_contributions: np.ndarray,
    frequency_shifts: np.ndarray,
    frequencies: np.ndarray,
    freq_threshold: float = 50.0
) -> Tuple[float, np.ndarray]:
    """
    Calcule le facteur de correction QVC.

    Formule:
        correction = Σᵢ (Δωᵢ/ωᵢ) × (λᵢ/λ_total)

    Le facteur multiplicatif final est (1 + correction).

    Args:
        lvc_contributions (np.ndarray): Contributions LVC λ_i par mode (kJ/mol)
        frequency_shifts (np.ndarray): Ratios Δω_i/ω_i estimés
        frequencies (np.ndarray): Fréquences en cm⁻¹
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)

    Returns:
        Tuple contenant:
            - correction_factor (float): Facteur multiplicatif (1 + correction)
            - mode_corrections (np.ndarray): Correction par mode
    """
    # Masquer les modes invalides
    mask = frequencies >= freq_threshold

    # Contributions LVC valides (en valeur absolue car elles sont négatives)
    valid_contributions = np.abs(lvc_contributions[mask])
    valid_shifts = frequency_shifts[mask]

    # Total LVC
    lambda_total = np.sum(valid_contributions)

    if lambda_total <= 0:
        return 1.0, np.zeros_like(lvc_contributions)

    # Poids de chaque mode
    weights = valid_contributions / lambda_total

    # Correction totale
    correction = np.sum(valid_shifts * weights)

    # Correction par mode (pour diagnostic)
    mode_corrections = np.zeros_like(lvc_contributions)
    mode_corrections[mask] = valid_shifts * weights

    # Facteur multiplicatif
    correction_factor = 1.0 + correction

    # Limiter le facteur pour éviter des corrections extrêmes
    correction_factor = max(0.5, min(2.0, correction_factor))

    return correction_factor, mode_corrections


# ============================================================================
# FONCTION PRINCIPALE QVC
# ============================================================================

def compute_qvc_edef(
    neutral_rkf: Path,
    ion_sp_rkf: Path,
    alpha: float = 0.3,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Calcule E_def avec corrections QVC.

    Cette méthode applique une correction quadratique au LVC pour tenir compte
    des changements de fréquence estimés entre l'état neutre et l'état ionique.

    Args:
        neutral_rkf (Path): Chemin vers le RKF du calcul neutre optimisé
        ion_sp_rkf (Path): Chemin vers le RKF du calcul single-point de l'ion
        alpha (float): Paramètre empirique pour l'estimation des Δω (défaut: 0.3)
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)
        top_n_modes (int): Nombre de modes dominants à retourner

    Returns:
        Dict contenant:
            - 'success': True si le calcul a réussi
            - 'E_def_LVC_kJ': Énergie de déformation LVC (kJ/mol)
            - 'E_def_QVC_kJ': Énergie de déformation QVC corrigée (kJ/mol)
            - 'correction_factor': Facteur multiplicatif appliqué
            - 'frequency_shifts': Ratios Δω_i/ω_i estimés
            - 'alpha': Paramètre α utilisé
            - 'error': Message d'erreur si success=False
    """
    result = {
        'success': False,
        'E_def_LVC_kJ': None,
        'E_def_QVC_kJ': None,
        'correction_factor': None,
        'frequency_shifts': None,
        'alpha': alpha,
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

    # 4. Calculer E_def LVC (méthode de base)
    E_def_LVC_kJ, contributions = compute_lvc_edef(
        gradients, normal_modes, force_constants, frequencies,
        freq_threshold=freq_threshold
    )

    # 5. Convertir les contributions en array
    n_modes = len(frequencies)
    lvc_contributions = np.zeros(n_modes)
    for c in contributions:
        mode_idx = c['mode_num'] - 1  # Convertir de 1-indexé à 0-indexé
        lvc_contributions[mode_idx] = c['contribution_kj']

    # 6. Estimer les changements de fréquence
    frequency_shifts = estimate_all_frequency_shifts(
        gradients, normal_modes, force_constants, frequencies,
        alpha=alpha, freq_threshold=freq_threshold
    )

    # 7. Calculer la correction QVC
    correction_factor, mode_corrections = compute_qvc_correction(
        lvc_contributions, frequency_shifts, frequencies,
        freq_threshold=freq_threshold
    )

    # 8. Appliquer la correction
    E_def_QVC_kJ = E_def_LVC_kJ * correction_factor

    # 9. Remplir les résultats
    result['success'] = True
    result['E_def_LVC_kJ'] = E_def_LVC_kJ
    result['E_def_QVC_kJ'] = E_def_QVC_kJ
    result['correction_factor'] = correction_factor
    result['frequency_shifts'] = frequency_shifts
    result['mode_corrections'] = mode_corrections
    result['n_modes_used'] = len(contributions)
    result['n_modes_total'] = n_modes

    # 10. Ajouter les contributions dominantes avec info QVC
    dominant_modes = []
    for c in contributions[:top_n_modes]:
        mode_idx = c['mode_num'] - 1
        dominant_modes.append({
            'mode_num': c['mode_num'],
            'frequency': c['frequency'],
            'F_i': c['F_i'],
            'k_i': c['k_i'],
            'lvc_contribution_kj': c['contribution_kj'],
            'freq_shift_ratio': frequency_shifts[mode_idx],
            'mode_correction': mode_corrections[mode_idx],
        })

    result['dominant_modes'] = dominant_modes

    return result


# ============================================================================
# FONCTION D'ANALYSE PAR MOLÉCULE
# ============================================================================

def run_qvc_analysis_for_molecule(
    workdir: Path,
    molecule_name: str,
    alpha: float = 0.3,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Exécute l'analyse QVC complète pour une molécule.

    Args:
        workdir (Path): Répertoire de travail de la molécule
        molecule_name (str): Nom de la molécule
        alpha (float): Paramètre empirique pour l'estimation des Δω
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)
        top_n_modes (int): Nombre de modes dominants à rapporter

    Returns:
        Dict contenant les résultats QVC pour réduction et oxydation par conformère
    """
    workdir = Path(workdir)

    # Localiser les fichiers RKF
    neutral_rkfs = find_neutral_rkf_files(workdir, molecule_name)
    reduced_sp_rkfs = find_sp_rkf_files(workdir, molecule_name, 'reduced')
    oxidized_sp_rkfs = find_sp_rkf_files(workdir, molecule_name, 'oxidized')

    results = {
        'molecule': molecule_name,
        'alpha': alpha,
        'conformers': {},
    }

    # Traiter chaque conformère
    for conf_num in sorted(neutral_rkfs.keys()):
        neutral_rkf = neutral_rkfs.get(conf_num)
        reduced_sp_rkf = reduced_sp_rkfs.get(conf_num)
        oxidized_sp_rkf = oxidized_sp_rkfs.get(conf_num)

        conf_results = {
            'reduction': None,
            'oxidation': None,
        }

        # QVC pour réduction
        if neutral_rkf and reduced_sp_rkf:
            qvc_red = compute_qvc_edef(
                neutral_rkf, reduced_sp_rkf,
                alpha=alpha, freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            conf_results['reduction'] = qvc_red

        # QVC pour oxydation
        if neutral_rkf and oxidized_sp_rkf:
            qvc_ox = compute_qvc_edef(
                neutral_rkf, oxidized_sp_rkf,
                alpha=alpha, freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            conf_results['oxidation'] = qvc_ox

        results['conformers'][conf_num] = conf_results

    return results


# ============================================================================
# OPTIMISATION DU PARAMÈTRE ALPHA
# ============================================================================

def optimize_alpha(
    workdirs: List[Path],
    molecule_names: List[str],
    real_edef: Dict[str, Dict[int, Dict[str, float]]],
    process: str = 'reduction',
    freq_threshold: float = 50.0,
    alpha_range: Tuple[float, float] = (0.1, 1.0)
) -> Tuple[float, float]:
    """
    Optimise le paramètre α par minimisation de l'erreur.

    Args:
        workdirs (List[Path]): Liste des répertoires de travail
        molecule_names (List[str]): Liste des noms de molécules
        real_edef (Dict): Dict imbriqué {mol_name: {conf_num: {'Edef_red_kJ': ..., ...}}}
        process (str): 'reduction' ou 'oxidation'
        freq_threshold (float): Seuil minimal de fréquence (cm⁻¹)
        alpha_range (Tuple): Plage de recherche pour α

    Returns:
        Tuple contenant:
            - alpha_optimal (float): Valeur optimale de α
            - min_error (float): Erreur minimale (MAE) en kJ/mol
    """
    from scipy.optimize import minimize_scalar

    edef_key = 'Edef_red_kJ' if process == 'reduction' else 'Edef_ox_kJ'

    def compute_error(alpha):
        """Calcule la MAE pour un α donné."""
        errors = []

        for workdir, mol_name in zip(workdirs, molecule_names):
            mol_real = real_edef.get(mol_name, {})

            neutral_rkfs = find_neutral_rkf_files(workdir, mol_name)
            sp_rkfs = find_sp_rkf_files(workdir, mol_name,
                                         'reduced' if process == 'reduction' else 'oxidized')

            for conf_num in neutral_rkfs.keys():
                if conf_num not in sp_rkfs:
                    continue
                if conf_num not in mol_real:
                    continue

                real_val = mol_real[conf_num].get(edef_key)
                if real_val is None or np.isnan(real_val):
                    continue

                qvc_result = compute_qvc_edef(
                    neutral_rkfs[conf_num], sp_rkfs[conf_num],
                    alpha=alpha, freq_threshold=freq_threshold
                )

                if qvc_result['success']:
                    qvc_val = qvc_result['E_def_QVC_kJ']
                    errors.append(abs(qvc_val - real_val))

        if not errors:
            return float('inf')

        return np.mean(errors)

    # Optimisation
    result = minimize_scalar(compute_error, bounds=alpha_range, method='bounded')

    return result.x, result.fun
