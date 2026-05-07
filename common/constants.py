"""
Constantes physiques et facteurs de conversion.

Ce module centralise toutes les constantes physiques et les facteurs
de conversion d'unités utilisés dans les différents pipelines.
"""

# ============================================================================
# CONSTANTES PHYSIQUES
# ============================================================================

# Constante de Faraday [C/mol]
F = 96485.33212

# Constante des gaz parfaits [J/(mol·K)]
R = 8.314462618

# Température par défaut [K]
TEMPERATURE = 298.15

# Unité d'énergie interne par défaut
UNIT = "kJ/mol"

# ============================================================================
# FACTEURS DE CONVERSION D'ÉNERGIE
# ============================================================================

# Conversion Hartree → eV
EH_TO_EV = 27.211386245988

# Conversion Hartree → kJ/mol
EH_TO_KJ = 2625.499638

# Conversion eV → kJ/mol
EV_TO_KJ = 96.485

# Conversion kJ/mol → eV
KJ_TO_EV = 1.0 / 96.485

# Conversion kcal/mol → kJ/mol
KCAL_TO_KJ = 4.184

# ============================================================================
# SEUILS ET PARAMÈTRES PAR DÉFAUT
# ============================================================================

# Seuil RMSD pour détecter un changement conformationnel significatif [Å]
# Utilisé dans l'analyse redox pour identifier les changements structuraux
RMSD_CONFORMATIONAL_CHANGE_THRESHOLD = 0.3

# Poids de Boltzmann minimal pour filtrer les conformères [sans unité]
# Les conformères avec un poids inférieur sont considérés négligeables
BOLTZMANN_MIN_WEIGHT_DEFAULT = 0.10

# Seuil pour détecter les fréquences imaginaires [cm⁻¹]
# Une fréquence < seuil indique un point selle (non-minimum)
# Défaut: 0.0 cm⁻¹ (aucune fréquence négative tolérée)
FREQUENCY_IMAGINARY_THRESHOLD = 0.0

# Nombre maximum de tentatives de correction de fréquences imaginaires
# Utilisé dans la correction itérative des géométries
MAX_FREQUENCY_CORRECTION_ATTEMPTS = 3

# Facteur d'échelle par défaut pour la perturbation géométrique [Å]
# Utilisé pour déplacer les atomes le long des modes normaux
PERTURBATION_SCALE_DEFAULT = 0.1

# Perturbation de base pour correction de fréquences [Å]
# Première tentative de déplacement lors de la correction
PERTURBATION_SCALE_FREQUENCY_CORRECTION = 0.2

# ============================================================================
# CLUSTER-CONTINUUM
# ============================================================================

# Seuil de charge Mulliken pour détecter les hydrogènes labiles [e]
LABILE_H_CHARGE_THRESHOLD = 0.10

# Distance par défaut pour le placement des molécules de solvant [Å]
H_BOND_DISTANCE_DEFAULT = 1.9

# ============================================================================
# FONCTIONS DE CONVERSION
# ============================================================================

def hartree_to_ev(energy_ha):
    """
    Convertit une énergie de Hartree en eV.

    Args:
        energy_ha (float): Énergie en Hartree

    Returns:
        float: Énergie en eV
    """
    return energy_ha * EH_TO_EV


def hartree_to_kj(energy_ha):
    """
    Convertit une énergie de Hartree en kJ/mol.

    Args:
        energy_ha (float): Énergie en Hartree

    Returns:
        float: Énergie en kJ/mol
    """
    return energy_ha * EH_TO_KJ


def ev_to_volt(energy_ev):
    """
    Convertit une énergie en eV en potentiel en Volts (vs vacuum).
    Pour un transfert mono-électronique : E(V) = -E(eV)

    Args:
        energy_ev (float): Énergie en eV

    Returns:
        float: Potentiel en V
    """
    return -energy_ev


def kj_to_volt(energy_kj, n_electrons=1):
    """
    Convertit une énergie de réaction en kJ/mol en potentiel électrochimique en Volts.
    E(V) = -ΔG(kJ/mol) / (n·F) où F = 96.485 kJ/(V·mol)

    Args:
        energy_kj (float): Énergie en kJ/mol
        n_electrons (int): Nombre d'électrons transférés (défaut: 1)

    Returns:
        float: Potentiel en V
    """
    return -energy_kj / (n_electrons * EV_TO_KJ)


def kcal_to_kj(energy_kcal):
    """
    Convertit une énergie de kcal/mol en kJ/mol.

    Args:
        energy_kcal (float): Énergie en kcal/mol

    Returns:
        float: Énergie en kJ/mol
    """
    return energy_kcal * KCAL_TO_KJ
