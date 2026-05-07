"""
Analyse Duschinsky pour le calcul de l'énergie de déformation.

Ce module implémente deux approches cohérentes en unités pour estimer E_def
à partir du déplacement géométrique entre états neutre et ionique :

1. **Displacement (modes neutres)** : validation du LVC
   E_def = -1/2 * sum_i k_i * dQ_i^2
   avec dQ_i = (1/mu_i) * sum_a(m_a * N_i_a * delta_R_a)
   Utilise les modes normaux et constantes de force du neutre.
   Doit donner ~= LVC pour un PES harmonique.

2. **Duschinsky (modes ioniques)** : approche au-delà du LVC
   E_def = -1/2 * sum_j k'_j * dQ_j^2
   avec dQ_j = (1/mu_j) * sum_a(m_a * N'_j_a * delta_R_a)
   Utilise les modes normaux et constantes de force de l'état ionique optimisé.
   Inclut implicitement la rotation des modes (matrice de Duschinsky)
   et le changement de fréquences entre les deux états.

Convention d'unités (identique au LVC) :
    - Coordonnées : Bohr (lues directement depuis Molecule/Coords dans les RKF)
    - Modes normaux : NoWeightNormalMode(i), normalisés à 1 (sans masse)
    - Constantes de force : Hartree/Bohr² (depuis Vibrations/ForceConstants)
    - Masses atomiques : AMU (depuis Molecule/AtomMasses)
    - Résultats : Hartree, convertis en kJ/mol

IMPORTANT: La projection du déplacement sur les modes normaux doit être
    mass-weighted car les NoWeightNormalMode sont orthogonaux en espace
    mass-weighted, PAS en espace cartésien. La projection correcte est :
        dQ_i = (1/mu_i) * sum_a(m_a * N_i_a * delta_R_a)
    où mu_i = sum_a(m_a * N_i_a^2) est la masse réduite du mode i.
    Le simple produit scalaire N_i . delta_R surestime E_def d'un facteur ~3x
    car les modes non-orthogonaux en cartésien surestiment les projections.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .common import (
    print_warning, print_error, print_success,
    EH_TO_KJ,
)
from .lvc_analysis import (
    extract_normal_modes_and_fc,
    extract_coordinates_from_rkf,
    extract_gradients,
    find_opt_rkf_files,
    find_sp_rkf_files,
)
from .frequency_extraction import find_neutral_rkf_files


# ============================================================================
# CONSTANTES
# ============================================================================

BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


# ============================================================================
# EXTRACTION DES COORDONNÉES EN BOHR
# ============================================================================

def extract_coordinates_bohr(rkf_path: Path) -> Optional[np.ndarray]:
    """
    Extrait les coordonnées atomiques en Bohr depuis un fichier RKF.

    Les coordonnées sont stockées en Bohr dans Molecule/Coords.
    Cette fonction les lit sans conversion (contrairement à
    extract_coordinates_from_rkf qui convertit en Angstrom).

    Args:
        rkf_path: Chemin vers le fichier adf.rkf

    Returns:
        Coordonnées en Bohr, shape (3N,), ou None si erreur.
        Le vecteur est aplati (3N) pour être directement compatible
        avec les modes normaux NoWeightNormalMode.
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))

        coords_bohr = np.array(kf.read("Molecule", "Coords"), dtype=float)
        n_atoms = kf.read("Molecule", "nAtoms")

        # Retourner en vecteur plat (3N,) pour compatibilité avec les modes
        return coords_bohr.reshape(n_atoms * 3)

    except Exception as e:
        print_warning(f"Erreur extraction coordonnées Bohr depuis {rkf_path}: {e}")
        return None


def extract_atomic_masses(rkf_path: Path) -> Optional[np.ndarray]:
    """
    Extrait les masses atomiques depuis Molecule/AtomMasses d'un fichier RKF.

    Args:
        rkf_path: Chemin vers le fichier adf.rkf

    Returns:
        Masses atomiques en u.m.a. (n_atoms,), ou None si erreur.
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))
        masses = np.array(kf.read("Molecule", "AtomMasses"), dtype=float)
        return masses

    except KeyError:
        try:
            from scm.plams import PeriodicTable, KFReader
            kf = KFReader(str(rkf_path))
            atom_symbols = kf.read("Molecule", "AtomSymbols")
            if isinstance(atom_symbols, str):
                symbols = atom_symbols.split()
            else:
                symbols = list(atom_symbols)

            pt = PeriodicTable()
            masses = np.array([pt.get_mass(s.strip()) for s in symbols], dtype=float)
            return masses

        except Exception as e:
            print_warning(f"Masses atomiques non trouvées dans {rkf_path}: {e}")
            return None

    except Exception as e:
        print_error(f"Erreur extraction masses depuis {rkf_path}: {e}")
        return None


# ============================================================================
# CALCUL E_DEF PAR DÉPLACEMENT (MODES NEUTRES) — VALIDATION DU LVC
# ============================================================================

def compute_displacement_edef_neutral_modes(
    neutral_rkf: Path,
    ion_opt_rkf: Path,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Calcule E_def par projection mass-weighted du déplacement sur les modes du neutre.

    Formule : E_def = -1/2 * sum_i k_i * dQ_i^2
    avec dQ_i = (1/mu_i) * sum_a(m_a * N_i_a * delta_R_a)

    La projection mass-weighted est nécessaire car les NoWeightNormalMode
    sont orthogonaux en espace mass-weighted, pas en espace cartésien.

    Pour un PES harmonique parfait, E_def_displacement = E_def_LVC.
    L'écart entre les deux mesure l'anharmonicité le long de chaque mode.

    Args:
        neutral_rkf: RKF du calcul neutre optimisé (avec fréquences)
        ion_opt_rkf: RKF de l'ion optimisé (pour les coordonnées)
        freq_threshold: Seuil minimal de fréquence (cm^-1)
        top_n_modes: Nombre de modes dominants à retourner

    Returns:
        Dict avec 'success', 'E_def_kJ', 'E_def_hartree',
        'contributions', 'n_modes_used', 'n_modes_total', 'error'
    """
    result = {
        'success': False,
        'E_def_kJ': None,
        'E_def_hartree': None,
        'contributions': [],
        'n_modes_used': 0,
        'n_modes_total': 0,
        'error': None,
    }

    # 1. Coordonnées en Bohr
    coords_neutral = extract_coordinates_bohr(neutral_rkf)
    coords_ion = extract_coordinates_bohr(ion_opt_rkf)
    if coords_neutral is None:
        result['error'] = f"Coordonnées non extraites depuis {neutral_rkf}"
        return result
    if coords_ion is None:
        result['error'] = f"Coordonnées non extraites depuis {ion_opt_rkf}"
        return result
    if coords_neutral.shape != coords_ion.shape:
        result['error'] = f"Dimensions incompatibles: {coords_neutral.shape} vs {coords_ion.shape}"
        return result

    # 2. Modes normaux et constantes de force du neutre
    normal_modes, force_constants, frequencies = extract_normal_modes_and_fc(neutral_rkf)
    if normal_modes is None:
        result['error'] = f"Modes normaux non extraits depuis {neutral_rkf}"
        return result

    # 3. Masses atomiques (pour projection mass-weighted)
    masses = extract_atomic_masses(neutral_rkf)
    if masses is None:
        result['error'] = f"Masses atomiques non extraites depuis {neutral_rkf}"
        return result
    masses_3N = np.repeat(masses, 3)  # (3N,) en AMU

    # 4. Déplacement en Bohr
    delta_R = coords_ion - coords_neutral  # (3N,) en Bohr

    # 5. Projection mass-weighted et calcul E_def
    #
    # Les NoWeightNormalMode sont orthogonaux en espace mass-weighted,
    # PAS en espace cartésien. La projection correcte est :
    #   dQ_i = (1/mu_i) * sum_a(m_a * N_i_a * delta_R_a)
    # où mu_i = sum_a(m_a * N_i_a²) est la masse réduite du mode i.
    #
    # Ceci est cohérent avec la formule LVC E = -sum F_i²/(2k_i)
    # car sum(N_i N_i^T / k_i) = H^{-1} exactement.
    E_def_total_hartree = 0.0
    contributions = []

    for i in range(len(normal_modes)):
        mode = normal_modes[i]
        freq = frequencies[i]
        k_i = force_constants[i]

        if mode is None or freq < freq_threshold or k_i <= 0:
            continue

        # Masse réduite du mode i
        mu_i = np.sum(masses_3N * mode ** 2)  # AMU

        # Projection mass-weighted du déplacement sur le mode
        W_i = np.sum(masses_3N * mode * delta_R)  # AMU·Bohr
        delta_Q_i = W_i / mu_i  # Bohr

        # E_def_i = -1/2 * k_i * delta_Q_i^2  (en Hartree, négatif car relaxation)
        contribution_hartree = -0.5 * k_i * delta_Q_i ** 2
        contribution_kj = contribution_hartree * EH_TO_KJ

        E_def_total_hartree += contribution_hartree

        contributions.append({
            'mode_num': i + 1,
            'frequency': freq,
            'delta_Q': delta_Q_i,
            'k_i': k_i,
            'contribution_hartree': contribution_hartree,
            'contribution_kj': contribution_kj,
        })

    # Trier par contribution absolue décroissante
    contributions.sort(key=lambda x: abs(x['contribution_kj']), reverse=True)

    result['success'] = True
    result['E_def_kJ'] = E_def_total_hartree * EH_TO_KJ
    result['E_def_hartree'] = E_def_total_hartree
    result['contributions'] = contributions
    result['top_contributions'] = contributions[:top_n_modes]
    result['n_modes_used'] = len(contributions)
    result['n_modes_total'] = len(normal_modes)

    return result


# ============================================================================
# CALCUL E_DEF DUSCHINSKY (MODES IONIQUES)
# ============================================================================

def compute_duschinsky_edef(
    neutral_rkf: Path,
    ion_opt_rkf: Path,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Calcule E_def par la méthode Duschinsky (modes de l'état ionique).

    Formule : E_def = -1/2 * sum_j k'_j * dQ_j^2
    avec dQ_j = (1/mu_j) * sum_a(m_a * N'_j_a * delta_R_a)

    Utilise les modes normaux (N'_j) et constantes de force (k'_j) de l'état
    ionique optimisé avec projection mass-weighted.

    L'écart entre cette méthode et le LVC mesure l'effet combiné de la rotation
    des modes et du changement de fréquences. L'écart résiduel avec le E_def réel
    mesure l'anharmonicité pure (termes cubiques et au-delà).

    Args:
        neutral_rkf: RKF du calcul neutre optimisé (pour les coordonnées neutres)
        ion_opt_rkf: RKF de l'ion optimisé (coordonnées, modes, constantes de force)
        freq_threshold: Seuil minimal de fréquence (cm^-1)
        top_n_modes: Nombre de modes dominants à retourner

    Returns:
        Dict avec 'success', 'E_def_kJ', 'E_def_hartree',
        'contributions', 'n_modes_used', 'n_modes_total', 'error'
    """
    result = {
        'success': False,
        'E_def_kJ': None,
        'E_def_hartree': None,
        'contributions': [],
        'n_modes_used': 0,
        'n_modes_total': 0,
        'error': None,
    }

    # 1. Coordonnées en Bohr
    coords_neutral = extract_coordinates_bohr(neutral_rkf)
    coords_ion = extract_coordinates_bohr(ion_opt_rkf)
    if coords_neutral is None:
        result['error'] = f"Coordonnées non extraites depuis {neutral_rkf}"
        return result
    if coords_ion is None:
        result['error'] = f"Coordonnées non extraites depuis {ion_opt_rkf}"
        return result
    if coords_neutral.shape != coords_ion.shape:
        result['error'] = f"Dimensions incompatibles: {coords_neutral.shape} vs {coords_ion.shape}"
        return result

    # 2. Modes normaux et constantes de force de l'ION
    ion_modes, ion_fc, ion_freq = extract_normal_modes_and_fc(ion_opt_rkf)
    if ion_modes is None:
        result['error'] = f"Modes normaux non extraits depuis {ion_opt_rkf}"
        return result

    # 3. Masses atomiques de l'ion (pour projection mass-weighted)
    masses = extract_atomic_masses(ion_opt_rkf)
    if masses is None:
        result['error'] = f"Masses atomiques non extraites depuis {ion_opt_rkf}"
        return result
    masses_3N = np.repeat(masses, 3)  # (3N,) en AMU

    # 4. Déplacement en Bohr
    # delta_R = R_neutre - R_ion : déplacement DE l'ion VERS la géométrie neutre
    # (le "point de départ" sur la PES de l'ion)
    delta_R = coords_neutral - coords_ion  # (3N,) en Bohr

    # 5. Projection mass-weighted et calcul E_def (même correction que displacement)
    E_def_total_hartree = 0.0
    contributions = []

    for j in range(len(ion_modes)):
        mode = ion_modes[j]
        freq = ion_freq[j]
        k_j = ion_fc[j]

        if mode is None or freq < freq_threshold or k_j <= 0:
            continue

        # Masse réduite du mode j (ion)
        mu_j = np.sum(masses_3N * mode ** 2)  # AMU

        # Projection mass-weighted du déplacement sur le mode ionique
        W_j = np.sum(masses_3N * mode * delta_R)  # AMU·Bohr
        delta_Q_j = W_j / mu_j  # Bohr

        # E_def_j = -1/2 * k'_j * delta_Q_j^2  (Hartree, négatif)
        contribution_hartree = -0.5 * k_j * delta_Q_j ** 2
        contribution_kj = contribution_hartree * EH_TO_KJ

        E_def_total_hartree += contribution_hartree

        contributions.append({
            'mode_num': j + 1,
            'frequency': freq,
            'delta_Q': delta_Q_j,
            'k_j': k_j,
            'contribution_hartree': contribution_hartree,
            'contribution_kj': contribution_kj,
        })

    contributions.sort(key=lambda x: abs(x['contribution_kj']), reverse=True)

    result['success'] = True
    result['E_def_kJ'] = E_def_total_hartree * EH_TO_KJ
    result['E_def_hartree'] = E_def_total_hartree
    result['contributions'] = contributions
    result['top_contributions'] = contributions[:top_n_modes]
    result['n_modes_used'] = len(contributions)
    result['n_modes_total'] = len(ion_modes)

    return result


# ============================================================================
# CORRECTION DE MORSE POUR DUSCHINSKY (RÉGIME ANHARMONIQUE)
# ============================================================================

def solve_morse_a(
    k: float,
    dQ: float,
    F_abs: float,
    a_min: float = 1e-6,
    a_max: float = 50.0
) -> Optional[float]:
    """
    Résout F = (k/a)(e^{-a|dQ|} - e^{-2a|dQ|}) pour le paramètre de Morse a.

    Pour un potentiel de Morse V(q) = D(1 - e^{-aq})², le gradient est :
        F(dQ) = (k/a)(e^{-a|dQ|} - e^{-2a|dQ|})
    où D = k/(2a²).

    Quand a → 0, F → k*dQ (limite harmonique).
    Quand a → ∞, F → 0 (dissociation, gradient nul).

    Args:
        k: Constante de force (Hartree/Bohr²)
        dQ: Déplacement mass-weighted (Bohr)
        F_abs: Valeur absolue du gradient projeté (Hartree/Bohr)
        a_min: Borne inférieure pour a
        a_max: Borne supérieure pour a

    Returns:
        a (> 0) ou None si pas de solution (mode quasi-harmonique ou données invalides).
    """
    from scipy.optimize import brentq

    dQ_abs = abs(dQ)

    if dQ_abs < 1e-10 or F_abs < 1e-15 or k <= 0:
        return None

    # Vérifier que F < k*dQ (sinon pas de solution Morse côté dissociatif)
    F_harmonic = k * dQ_abs
    if F_abs >= F_harmonic * 0.99:
        return None

    def func(a):
        u = np.exp(-a * dQ_abs)
        return (k / a) * (u - u**2) - F_abs

    try:
        f_low = func(a_min)
        f_high = func(a_max)
        if f_low * f_high > 0:
            f_high = func(100.0)
            if f_low * f_high > 0:
                return None
            a_max = 100.0
        return brentq(func, a_min, a_max, xtol=1e-10)
    except (ValueError, RuntimeError):
        return None


def morse_energy(k: float, dQ: float, a: float) -> float:
    """
    Calcule l'énergie Morse : D(1 - e^{-a|dQ|})² avec D = k/(2a²).

    Pour petit dQ, ceci tend vers ½ k dQ² (harmonique).
    Pour grand dQ, ceci sature à D (énergie de dissociation).

    Args:
        k: Constante de force (Hartree/Bohr²)
        dQ: Déplacement mass-weighted (Bohr)
        a: Paramètre de Morse (Bohr⁻¹)

    Returns:
        Énergie en Hartree (toujours positive).
    """
    D = k / (2 * a**2)
    return D * (1 - np.exp(-a * abs(dQ)))**2


def compute_morse_duschinsky_edef(
    neutral_rkf: Path,
    ion_opt_rkf: Path,
    ion_sp_rkf: Path,
    freq_threshold: float = 50.0,
    dQ_threshold: float = 0.05,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Calcule E_def par Duschinsky avec correction de Morse pour l'anharmonicité.

    Pour chaque mode j de l'ion :
    1. Calcule dQ_j par projection mass-weighted du déplacement
    2. Calcule F_j par projection du gradient SP sur le mode ionique
    3. Évalue l'harmonicité h_j = F_j / (-k_j * dQ_j)
       - h ≈ 1 : mode harmonique, utilise E = ½ k dQ²
       - h ≪ 1 : mode anharmonique, ajuste un potentiel de Morse
    4. Pour les modes anharmoniques avec |dQ| > dQ_threshold :
       - Résout F = (k/a)(e^{-aΔQ} - e^{-2aΔQ}) pour a
       - E_Morse = D(1 - e^{-aΔQ})² avec D = k/(2a²)

    Args:
        neutral_rkf: RKF du calcul neutre optimisé
        ion_opt_rkf: RKF de l'ion optimisé (modes, FC, coordonnées)
        ion_sp_rkf: RKF du single-point ionique sur géométrie neutre (gradients)
        freq_threshold: Seuil minimal de fréquence (cm⁻¹)
        dQ_threshold: Seuil de |dQ| pour appliquer la correction Morse (Bohr)
        top_n_modes: Nombre de modes dominants à retourner

    Returns:
        Dict avec 'success', 'E_def_kJ', 'E_def_hartree',
        'E_def_harmonic_kJ' (pour comparaison), 'contributions',
        'n_modes_used', 'n_modes_total', 'n_modes_morse', 'error'
    """
    result = {
        'success': False,
        'E_def_kJ': None,
        'E_def_hartree': None,
        'E_def_harmonic_kJ': None,
        'contributions': [],
        'n_modes_used': 0,
        'n_modes_total': 0,
        'n_modes_morse': 0,
        'error': None,
    }

    # 1. Coordonnées en Bohr
    coords_neutral = extract_coordinates_bohr(neutral_rkf)
    coords_ion = extract_coordinates_bohr(ion_opt_rkf)
    if coords_neutral is None:
        result['error'] = f"Coordonnées non extraites depuis {neutral_rkf}"
        return result
    if coords_ion is None:
        result['error'] = f"Coordonnées non extraites depuis {ion_opt_rkf}"
        return result
    if coords_neutral.shape != coords_ion.shape:
        result['error'] = f"Dimensions incompatibles: {coords_neutral.shape} vs {coords_ion.shape}"
        return result

    # 2. Modes normaux et constantes de force de l'ION
    ion_modes, ion_fc, ion_freq = extract_normal_modes_and_fc(ion_opt_rkf)
    if ion_modes is None:
        result['error'] = f"Modes normaux non extraits depuis {ion_opt_rkf}"
        return result

    # 3. Masses atomiques de l'ion
    masses = extract_atomic_masses(ion_opt_rkf)
    if masses is None:
        result['error'] = f"Masses atomiques non extraites depuis {ion_opt_rkf}"
        return result
    masses_3N = np.repeat(masses, 3)

    # 4. Gradient SP (ion PES à géométrie neutre)
    gradients = extract_gradients(ion_sp_rkf)
    if gradients is None:
        result['error'] = f"Gradients non extraits depuis {ion_sp_rkf}"
        return result

    # 5. Déplacement neutre → ion
    delta_R = coords_neutral - coords_ion  # (3N,) en Bohr

    # 6. Projection et calcul E_def avec correction Morse
    E_def_total_hartree = 0.0
    E_def_harmonic_hartree = 0.0
    contributions = []
    n_modes_morse = 0

    for j in range(len(ion_modes)):
        mode = ion_modes[j]
        freq = ion_freq[j]
        k_j = ion_fc[j]

        if mode is None or freq < freq_threshold or k_j <= 0:
            continue

        # Masse réduite et projection mass-weighted
        mu_j = np.sum(masses_3N * mode ** 2)
        W_j = np.sum(masses_3N * mode * delta_R)
        dQ_j = W_j / mu_j

        # Gradient projeté sur mode ionique
        F_j = np.dot(gradients, mode)

        # Énergie harmonique (toujours calculée pour comparaison)
        E_harm_j = -0.5 * k_j * dQ_j ** 2

        # Harmonicité
        h_j = F_j / (-k_j * dQ_j) if abs(dQ_j) > 1e-10 else 1.0

        # Correction Morse
        a_j = solve_morse_a(k_j, dQ_j, abs(F_j))

        if a_j is not None and abs(dQ_j) > dQ_threshold:
            E_morse_j = -morse_energy(k_j, dQ_j, a_j)
            n_modes_morse += 1
        else:
            E_morse_j = E_harm_j  # fallback harmonique

        E_def_total_hartree += E_morse_j
        E_def_harmonic_hartree += E_harm_j

        contributions.append({
            'mode_num': j + 1,
            'frequency': freq,
            'delta_Q': dQ_j,
            'k_j': k_j,
            'F_j': F_j,
            'harmonicity': h_j,
            'a_morse': a_j,
            'contribution_hartree': E_morse_j,
            'contribution_kj': E_morse_j * EH_TO_KJ,
            'contribution_harmonic_kj': E_harm_j * EH_TO_KJ,
        })

    contributions.sort(key=lambda x: abs(x['contribution_kj']), reverse=True)

    result['success'] = True
    result['E_def_kJ'] = E_def_total_hartree * EH_TO_KJ
    result['E_def_hartree'] = E_def_total_hartree
    result['E_def_harmonic_kJ'] = E_def_harmonic_hartree * EH_TO_KJ
    result['contributions'] = contributions
    result['top_contributions'] = contributions[:top_n_modes]
    result['n_modes_used'] = len(contributions)
    result['n_modes_total'] = len(ion_modes)
    result['n_modes_morse'] = n_modes_morse

    return result


# ============================================================================
# MATRICE DE DUSCHINSKY
# ============================================================================

def compute_duschinsky_matrix(
    neutral_rkf: Path,
    ion_opt_rkf: Path,
    freq_threshold: float = 50.0
) -> Dict[str, Any]:
    """
    Calcule la matrice de Duschinsky J_ij = N'_i . N_j.

    Cette matrice quantifie le mélange des modes normaux entre les deux états.
    J est unitaire si les modes sont identiques (pas de rotation).
    Des éléments hors-diagonaux importants indiquent un mélange de modes.

    Args:
        neutral_rkf: RKF du calcul neutre
        ion_opt_rkf: RKF de l'ion optimisé

    Returns:
        Dict avec 'success', 'J_matrix' (np.ndarray), 'freq_neutral',
        'freq_ion', 'diag_norm' (norme des éléments diagonaux),
        'off_diag_norm' (norme des éléments hors-diagonaux), 'error'
    """
    result = {
        'success': False,
        'J_matrix': None,
        'freq_neutral': None,
        'freq_ion': None,
        'diag_norm': None,
        'off_diag_norm': None,
        'error': None,
    }

    # Modes du neutre
    neutral_modes, _, freq_neutral = extract_normal_modes_and_fc(neutral_rkf)
    if neutral_modes is None:
        result['error'] = f"Modes non extraits depuis {neutral_rkf}"
        return result

    # Modes de l'ion
    ion_modes, _, freq_ion = extract_normal_modes_and_fc(ion_opt_rkf)
    if ion_modes is None:
        result['error'] = f"Modes non extraits depuis {ion_opt_rkf}"
        return result

    # Filtrer par seuil de fréquence
    mask_n = freq_neutral >= freq_threshold
    mask_i = freq_ion >= freq_threshold

    # Indices des modes retenus
    idx_n = np.where(mask_n)[0]
    idx_i = np.where(mask_i)[0]

    n_n = len(idx_n)
    n_i = len(idx_i)

    # Construire la matrice J (n_ion x n_neutral)
    J = np.zeros((n_i, n_n))
    for ii, i in enumerate(idx_i):
        for jj, j in enumerate(idx_n):
            if ion_modes[i] is not None and neutral_modes[j] is not None:
                J[ii, jj] = np.dot(ion_modes[i], neutral_modes[j])

    # Diagnostics
    min_dim = min(n_i, n_n)
    diag_elements = np.array([J[k, k] for k in range(min_dim)])
    diag_norm = np.sqrt(np.sum(diag_elements ** 2)) / min_dim if min_dim > 0 else 0

    off_diag_sum = np.sum(J ** 2) - np.sum(diag_elements ** 2)
    total_elements = n_i * n_n - min_dim
    off_diag_norm = np.sqrt(off_diag_sum / total_elements) if total_elements > 0 else 0

    result['success'] = True
    result['J_matrix'] = J
    result['freq_neutral'] = freq_neutral[mask_n]
    result['freq_ion'] = freq_ion[mask_i]
    result['diag_norm'] = diag_norm
    result['off_diag_norm'] = off_diag_norm

    return result


# ============================================================================
# FONCTION D'ANALYSE PAR MOLÉCULE
# ============================================================================

def run_duschinsky_analysis_for_molecule(
    workdir: Path,
    molecule_name: str,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Exécute l'analyse Duschinsky complète pour une molécule.

    Pour chaque conformère, calcule :
    - E_def par displacement (modes neutres) — pour validation vs LVC
    - E_def par Duschinsky harmonique (modes ioniques)
    - E_def par Duschinsky-Morse (modes ioniques + correction anharmonique)
    - Matrice de Duschinsky (diagnostic)

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule
        freq_threshold: Seuil minimal de fréquence (cm^-1)
        top_n_modes: Nombre de modes dominants à rapporter

    Returns:
        Dict contenant les résultats par conformère pour réduction et oxydation
    """
    workdir = Path(workdir)

    # Localiser les fichiers RKF
    neutral_rkfs = find_neutral_rkf_files(workdir, molecule_name)
    reduced_opt_rkfs = find_opt_rkf_files(workdir, molecule_name, 'reduced')
    oxidized_opt_rkfs = find_opt_rkf_files(workdir, molecule_name, 'oxidized')
    reduced_sp_rkfs = find_sp_rkf_files(workdir, molecule_name, 'reduced')
    oxidized_sp_rkfs = find_sp_rkf_files(workdir, molecule_name, 'oxidized')

    results = {
        'molecule': molecule_name,
        'conformers': {},
    }

    for conf_num in sorted(neutral_rkfs.keys()):
        neutral_rkf = neutral_rkfs.get(conf_num)
        reduced_opt_rkf = reduced_opt_rkfs.get(conf_num)
        oxidized_opt_rkf = oxidized_opt_rkfs.get(conf_num)
        reduced_sp_rkf = reduced_sp_rkfs.get(conf_num)
        oxidized_sp_rkf = oxidized_sp_rkfs.get(conf_num)

        conf_results = {
            'reduction': None,
            'oxidation': None,
            'reduction_displacement': None,
            'oxidation_displacement': None,
            'reduction_morse': None,
            'oxidation_morse': None,
            'reduction_duschinsky_matrix': None,
            'oxidation_duschinsky_matrix': None,
        }

        # Réduction
        if neutral_rkf and reduced_opt_rkf:
            # Duschinsky harmonique (modes ioniques) — méthode principale
            conf_results['reduction'] = compute_duschinsky_edef(
                neutral_rkf, reduced_opt_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            # Displacement (modes neutres) — validation
            conf_results['reduction_displacement'] = compute_displacement_edef_neutral_modes(
                neutral_rkf, reduced_opt_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            # Morse-Duschinsky (correction anharmonique)
            if reduced_sp_rkf:
                conf_results['reduction_morse'] = compute_morse_duschinsky_edef(
                    neutral_rkf, reduced_opt_rkf, reduced_sp_rkf,
                    freq_threshold=freq_threshold, top_n_modes=top_n_modes
                )
            # Matrice de Duschinsky
            conf_results['reduction_duschinsky_matrix'] = compute_duschinsky_matrix(
                neutral_rkf, reduced_opt_rkf,
                freq_threshold=freq_threshold
            )

        # Oxydation
        if neutral_rkf and oxidized_opt_rkf:
            conf_results['oxidation'] = compute_duschinsky_edef(
                neutral_rkf, oxidized_opt_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            conf_results['oxidation_displacement'] = compute_displacement_edef_neutral_modes(
                neutral_rkf, oxidized_opt_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )
            # Morse-Duschinsky (correction anharmonique)
            if oxidized_sp_rkf:
                conf_results['oxidation_morse'] = compute_morse_duschinsky_edef(
                    neutral_rkf, oxidized_opt_rkf, oxidized_sp_rkf,
                    freq_threshold=freq_threshold, top_n_modes=top_n_modes
                )
            conf_results['oxidation_duschinsky_matrix'] = compute_duschinsky_matrix(
                neutral_rkf, oxidized_opt_rkf,
                freq_threshold=freq_threshold
            )

        results['conformers'][conf_num] = conf_results

    return results


# ============================================================================
# FONCTIONS HUANG-RHYS LEGACY (DEPRECATED)
# ============================================================================

# Les fonctions ci-dessous sont conservées pour compatibilité mais ne doivent
# plus être utilisées. Elles souffrent d'un bug d'unités : les coordonnées
# en Angstrom sont mélangées avec la constante HUANG_RHYS_FACTOR qui suppose
# une normalisation mass-weighted spécifique.

HUANG_RHYS_FACTOR = 5.8065e-3  # DEPRECATED

def compute_displacement_vector(
    coords_neutral: np.ndarray,
    coords_ion: np.ndarray,
    masses: np.ndarray,
    normal_modes: list
) -> np.ndarray:
    """DEPRECATED: Utiliser compute_displacement_edef_neutral_modes() à la place."""
    import warnings
    warnings.warn(
        "compute_displacement_vector() is deprecated due to unit inconsistencies. "
        "Use compute_displacement_edef_neutral_modes() instead.",
        DeprecationWarning, stacklevel=2
    )
    n_atoms = len(masses)
    delta_r = coords_ion - coords_neutral
    delta_r_flat = delta_r.flatten()
    sqrt_masses_3N = np.repeat(np.sqrt(masses), 3)
    delta_r_mw = delta_r_flat * sqrt_masses_3N
    D = np.zeros(len(normal_modes))
    for i, mode in enumerate(normal_modes):
        if mode is not None:
            D[i] = np.dot(delta_r_mw, mode)
    return D


def compute_huang_rhys_factors(
    displacement: np.ndarray,
    frequencies: np.ndarray
) -> np.ndarray:
    """DEPRECATED: Utiliser compute_duschinsky_edef() à la place."""
    import warnings
    warnings.warn(
        "compute_huang_rhys_factors() is deprecated due to unit inconsistencies. "
        "Use compute_duschinsky_edef() instead.",
        DeprecationWarning, stacklevel=2
    )
    return HUANG_RHYS_FACTOR * frequencies * (displacement ** 2)


def compute_reorganization_energy(
    huang_rhys: np.ndarray,
    frequencies: np.ndarray
) -> tuple:
    """DEPRECATED: Utiliser compute_duschinsky_edef() à la place."""
    import warnings
    warnings.warn(
        "compute_reorganization_energy() is deprecated due to unit inconsistencies. "
        "Use compute_duschinsky_edef() instead.",
        DeprecationWarning, stacklevel=2
    )
    from .common import CM_TO_KJ
    hbar_omega_kj = frequencies * CM_TO_KJ
    lambda_i = huang_rhys * hbar_omega_kj
    return np.sum(lambda_i), lambda_i
