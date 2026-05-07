"""
Configuration spécifique et constantes pour le module d'analyse fréquence-Edef.

Ce module définit les constantes de plages de fréquences et ré-exporte les
utilitaires du module common principal.
"""

# Ré-exports du module common principal
from common.constants import (
    R, F, TEMPERATURE, UNIT,
    EH_TO_EV, EH_TO_KJ, EV_TO_KJ, KJ_TO_EV, KCAL_TO_KJ,
    hartree_to_kj, hartree_to_ev
)

from common.utils import (
    boltzmann_weights,
    print_header,
    print_section,
    print_success,
    print_warning,
    print_error,
    format_parameter_summary
)

# ============================================================================
# CONSTANTES POUR L'ANALYSE DE FRÉQUENCES
# ============================================================================

# Plages de fréquences vibrationnelles (cm⁻¹)
# Ces plages sont utilisées pour catégoriser les modes normaux

FREQUENCY_RANGES = {
    'low': (0, 500),           # Modes basse fréquence (flexibilité moléculaire)
    'medium_low': (500, 1000), # Bending modes
    'medium': (1000, 2000),    # Stretching modes (C-C, C-O, C=C)
    'high': (2000, 4000),      # X-H stretches (C-H, O-H, N-H)
}

# Descripteurs ciblés pour les corrélations focalisées ΔΔU ↔ hautes fréquences
# Les modes haute fréquence dominent la ZPE et l'énergie thermique vibrationnelle
HIGH_FREQ_DESCRIPTORS = [
    'n_modes_high', 'stiffness_index', 'sum_freq_high', 'mean_freq_high',
    'zpe_kj', 'freq_max', 'sum_ir_high', 'mean_ir_high', 'frac_ir_high',
    'freq_mean', 'freq_harmonic_mean', 'mean_freq_q4',
]

# Descripteurs ciblés pour les corrélations focalisées TΔS ↔ basses fréquences
# Les modes basse fréquence contribuent le plus à l'entropie vibrationnelle
LOW_FREQ_DESCRIPTORS = [
    'n_modes_low', 'flexibility_index', 'sum_freq_low', 'mean_freq_low',
    'freq_min', 'freq_1', 'freq_2', 'freq_3',
    'ratio_low_high', 'ratio_low_medium',
    'sum_ir_low', 'mean_ir_low', 'frac_ir_low', 'frac_bending', 'mean_freq_q1',
]

# Facteur de conversion pour ZPE : cm⁻¹ → kJ/mol
# ZPE = Σ (hν/2) = Σ (ν / 2) * h * c * N_A
# h * c * N_A = 1.196266e-2 kJ/mol/cm⁻¹
CM_TO_KJ = 1.196266e-2  # kJ/mol per cm⁻¹

# ============================================================================
# DESCRIPTEURS POUR L'ANALYSE DENSITE QTAIM vs EA/IP
# ============================================================================

# Descripteurs haute densite (correles a la rigidite/IP)
HIGH_DENSITY_DESCRIPTORS = [
    'rho_heavy_mean', 'Ts_heavy_mean', 'rho_BCP_mean', 'rho_BCP_CC_mean',
    'Ts_over_rho_heavy_mean', 'rho_C_mean', 'ELF_heavy_mean',
]

# Descripteurs basse densite / charge (correles a la delocalisation/EA)
CHARGE_DESCRIPTORS = [
    'charge_span', 'sum_abs_charge', 'charge_O_mean', 'charge_C_mean',
    'laplacian_heavy_mean', 'laplacian_over_rho_heavy_mean',
]

# ============================================================================
# FONCTIONS UTILITAIRES SPÉCIFIQUES
# ============================================================================

def cm_to_kj(wavenumber_cm):
    """
    Convertit une fréquence vibrationnelle (cm⁻¹) en énergie (kJ/mol).

    Args:
        wavenumber_cm (float): Fréquence en cm⁻¹

    Returns:
        float: Énergie en kJ/mol
    """
    return wavenumber_cm * CM_TO_KJ


def count_modes_in_range(frequencies, low, high):
    """
    Compte le nombre de modes dans une plage de fréquences donnée.

    Args:
        frequencies (array-like): Tableau des fréquences en cm⁻¹
        low (float): Limite inférieure de la plage
        high (float): Limite supérieure de la plage

    Returns:
        int: Nombre de modes dans la plage [low, high[
    """
    import numpy as np
    freqs = np.array(frequencies, dtype=float)
    return int(np.sum((freqs >= low) & (freqs < high)))
