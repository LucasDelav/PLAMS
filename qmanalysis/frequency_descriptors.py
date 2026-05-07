"""
Calcul des descripteurs de fréquences vibrationnelles.

Ce module calcule divers descripteurs statistiques et physiques à partir
des données de fréquences vibrationnelles extraites des fichiers RKF.
"""

from typing import Dict, Any, Optional

import numpy as np

from .common import FREQUENCY_RANGES, count_modes_in_range, CM_TO_KJ


def compute_zpe(frequencies: np.ndarray) -> float:
    """
    Calcule l'énergie de point zéro (ZPE) à partir des fréquences.

    La ZPE est calculée comme: ZPE = Σ (ν_i / 2) en unités d'énergie

    Args:
        frequencies (np.ndarray): Tableau des fréquences en cm⁻¹

    Returns:
        float: ZPE en kJ/mol

    Note:
        Seules les fréquences positives sont prises en compte.
        Les fréquences imaginaires (négatives) sont ignorées.
    """
    # Ne prendre que les fréquences positives
    pos_freqs = frequencies[frequencies > 0]

    if len(pos_freqs) == 0:
        return 0.0

    # ZPE = Σ (ν_i / 2) * facteur_de_conversion
    zpe_kj = np.sum(pos_freqs) * 0.5 * CM_TO_KJ

    return float(zpe_kj)


def compute_frequency_statistics(frequencies: np.ndarray) -> Dict[str, float]:
    """
    Calcule les statistiques de base des fréquences.

    Args:
        frequencies (np.ndarray): Tableau des fréquences en cm⁻¹

    Returns:
        Dict avec statistiques étendues
    """
    from scipy.stats import skew, kurtosis

    # Ne considérer que les fréquences positives pour les statistiques
    pos_freqs = frequencies[frequencies > 0]

    if len(pos_freqs) == 0:
        return {
            'freq_max': np.nan,
            'freq_min': np.nan,
            'freq_mean': np.nan,
            'freq_std': np.nan,
            'freq_median': np.nan,
            'freq_spread': np.nan,
            'freq_q25': np.nan,
            'freq_q75': np.nan,
            'freq_iqr': np.nan,
            'freq_skewness': np.nan,
            'freq_kurtosis': np.nan,
            'freq_cv': np.nan,
            'freq_harmonic_mean': np.nan,
            'freq_geometric_mean': np.nan,
            'freq_1': np.nan,
            'freq_2': np.nan,
            'freq_3': np.nan,
            'freq_4': np.nan,
            'freq_5': np.nan,
            'sum_freq_low': np.nan,
            'sum_freq_medium': np.nan,
            'sum_freq_high': np.nan,
            'mean_freq_low': np.nan,
            'mean_freq_medium': np.nan,
            'mean_freq_high': np.nan,
            # Moyennes par quartile
            'mean_freq_q1': np.nan,
            'mean_freq_q2': np.nan,
            'mean_freq_q3': np.nan,
            'mean_freq_q4': np.nan,
        }

    sorted_freqs = np.sort(pos_freqs)
    q25 = float(np.percentile(pos_freqs, 25))
    q50 = float(np.percentile(pos_freqs, 50))
    q75 = float(np.percentile(pos_freqs, 75))
    mean_val = float(np.mean(pos_freqs))
    std_val = float(np.std(pos_freqs))

    # Fréquences dans chaque plage spectrale
    low_freqs = pos_freqs[(pos_freqs >= 0) & (pos_freqs < 500)]
    medium_freqs = pos_freqs[(pos_freqs >= 500) & (pos_freqs < 2000)]
    high_freqs = pos_freqs[pos_freqs >= 2000]

    # Fréquences par quartile (basé sur la distribution, pas les plages spectrales)
    freqs_q1 = sorted_freqs[sorted_freqs <= q25]  # 25% modes les plus mous
    freqs_q2 = sorted_freqs[(sorted_freqs > q25) & (sorted_freqs <= q50)]  # 25-50%
    freqs_q3 = sorted_freqs[(sorted_freqs > q50) & (sorted_freqs <= q75)]  # 50-75%
    freqs_q4 = sorted_freqs[sorted_freqs > q75]  # 25% modes les plus rigides

    # Moyennes harmonique et géométrique
    harmonic_mean = float(len(pos_freqs) / np.sum(1.0 / pos_freqs)) if np.all(pos_freqs > 0) else np.nan
    geometric_mean = float(np.exp(np.mean(np.log(pos_freqs)))) if np.all(pos_freqs > 0) else np.nan

    return {
        'freq_max': float(np.max(pos_freqs)),
        'freq_min': float(np.min(pos_freqs)),
        'freq_mean': mean_val,
        'freq_std': std_val,
        'freq_median': float(np.median(pos_freqs)),
        'freq_spread': float(np.max(pos_freqs) - np.min(pos_freqs)),
        'freq_q25': q25,
        'freq_q75': q75,
        'freq_iqr': q75 - q25,
        'freq_skewness': float(skew(pos_freqs)),
        'freq_kurtosis': float(kurtosis(pos_freqs)),
        'freq_cv': std_val / mean_val if mean_val > 0 else np.nan,
        'freq_harmonic_mean': harmonic_mean,
        'freq_geometric_mean': geometric_mean,
        # 5 premières fréquences (les plus basses)
        'freq_1': float(sorted_freqs[0]) if len(sorted_freqs) > 0 else np.nan,
        'freq_2': float(sorted_freqs[1]) if len(sorted_freqs) > 1 else np.nan,
        'freq_3': float(sorted_freqs[2]) if len(sorted_freqs) > 2 else np.nan,
        'freq_4': float(sorted_freqs[3]) if len(sorted_freqs) > 3 else np.nan,
        'freq_5': float(sorted_freqs[4]) if len(sorted_freqs) > 4 else np.nan,
        # Sommes par plage
        'sum_freq_low': float(np.sum(low_freqs)) if len(low_freqs) > 0 else 0.0,
        'sum_freq_medium': float(np.sum(medium_freqs)) if len(medium_freqs) > 0 else 0.0,
        'sum_freq_high': float(np.sum(high_freqs)) if len(high_freqs) > 0 else 0.0,
        # Moyennes par plage spectrale
        'mean_freq_low': float(np.mean(low_freqs)) if len(low_freqs) > 0 else np.nan,
        'mean_freq_medium': float(np.mean(medium_freqs)) if len(medium_freqs) > 0 else np.nan,
        'mean_freq_high': float(np.mean(high_freqs)) if len(high_freqs) > 0 else np.nan,
        # Moyennes par quartile de la distribution
        'mean_freq_q1': float(np.mean(freqs_q1)) if len(freqs_q1) > 0 else np.nan,
        'mean_freq_q2': float(np.mean(freqs_q2)) if len(freqs_q2) > 0 else np.nan,
        'mean_freq_q3': float(np.mean(freqs_q3)) if len(freqs_q3) > 0 else np.nan,
        'mean_freq_q4': float(np.mean(freqs_q4)) if len(freqs_q4) > 0 else np.nan,
    }


def compute_mode_counts(frequencies: np.ndarray) -> Dict[str, int]:
    """
    Compte le nombre de modes dans chaque plage de fréquences.

    Args:
        frequencies (np.ndarray): Tableau des fréquences en cm⁻¹

    Returns:
        Dict avec: n_modes_low, n_modes_medium_low, n_modes_medium, n_modes_high, n_modes_total
    """
    # Ne considérer que les fréquences positives
    pos_freqs = frequencies[frequencies > 0]

    counts = {
        'n_modes_total': len(pos_freqs),
    }

    for range_name, (low, high) in FREQUENCY_RANGES.items():
        key = f'n_modes_{range_name}'
        counts[key] = count_modes_in_range(pos_freqs, low, high)

    return counts


def compute_flexibility_indices(mode_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calcule les indices de flexibilité et rigidité moléculaire.

    Args:
        mode_counts (Dict): Comptages de modes par plage (depuis compute_mode_counts)

    Returns:
        Dict avec indices et ratios étendus
    """
    n_total = mode_counts.get('n_modes_total', 0)
    n_low = mode_counts.get('n_modes_low', 0)
    n_medium_low = mode_counts.get('n_modes_medium_low', 0)
    n_medium = mode_counts.get('n_modes_medium', 0)
    n_high = mode_counts.get('n_modes_high', 0)

    if n_total == 0:
        return {
            'flexibility_index': np.nan,
            'stiffness_index': np.nan,
            'ratio_low_high': np.nan,
            'ratio_low_medium': np.nan,
            'ratio_medium_high': np.nan,
            'frac_bending': np.nan,
            'frac_stretching': np.nan,
        }

    return {
        'flexibility_index': float(n_low) / float(n_total),
        'stiffness_index': float(n_high) / float(n_total),
        'ratio_low_high': float(n_low) / float(n_high) if n_high > 0 else np.nan,
        'ratio_low_medium': float(n_low) / float(n_medium) if n_medium > 0 else np.nan,
        'ratio_medium_high': float(n_medium) / float(n_high) if n_high > 0 else np.nan,
        'frac_bending': float(n_medium_low) / float(n_total),
        'frac_stretching': float(n_medium) / float(n_total),
    }


def compute_ir_descriptors(
    frequencies: np.ndarray,
    ir_intensities: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Calcule les descripteurs liés aux intensités IR.

    Args:
        frequencies (np.ndarray): Tableau des fréquences en cm⁻¹
        ir_intensities (np.ndarray, optional): Tableau des intensités IR (km/mol)

    Returns:
        Dict avec statistiques IR complètes (ou NaN si IR non disponible)
    """
    # Valeurs par défaut (NaN) si pas de données IR
    nan_result = {
        # Statistiques de base
        'ir_max': np.nan,
        'ir_min': np.nan,
        'ir_mean': np.nan,
        'ir_std': np.nan,
        'ir_median': np.nan,
        'ir_sum': np.nan,
        'ir_spread': np.nan,
        # Percentiles
        'ir_q25': np.nan,
        'ir_q75': np.nan,
        'ir_iqr': np.nan,
        # Moyennes par quartile
        'mean_ir_q1': np.nan,
        'mean_ir_q2': np.nan,
        'mean_ir_q3': np.nan,
        'mean_ir_q4': np.nan,
        # Top 5 intensités (les plus fortes)
        'ir_top1': np.nan,
        'ir_top2': np.nan,
        'ir_top3': np.nan,
        'ir_top4': np.nan,
        'ir_top5': np.nan,
        # Fréquence du max IR
        'freq_of_max_ir': np.nan,
        # Fréquence pondérée par IR
        'freq_ir_weighted': np.nan,
        # IR par plage de fréquence
        'sum_ir_low': np.nan,
        'sum_ir_medium': np.nan,
        'sum_ir_high': np.nan,
        'mean_ir_low': np.nan,
        'mean_ir_medium': np.nan,
        'mean_ir_high': np.nan,
        # Fraction IR par plage
        'frac_ir_low': np.nan,
        'frac_ir_medium': np.nan,
        'frac_ir_high': np.nan,
    }

    if ir_intensities is None or len(ir_intensities) == 0:
        return nan_result

    # Ne considérer que les modes avec fréquences positives
    pos_mask = frequencies > 0
    pos_freqs = frequencies[pos_mask]

    if len(ir_intensities) == len(frequencies):
        pos_ir = ir_intensities[pos_mask]
    else:
        pos_ir = ir_intensities

    if len(pos_ir) == 0:
        return nan_result

    # Statistiques de base
    ir_max = float(np.max(pos_ir))
    ir_min = float(np.min(pos_ir))
    ir_mean = float(np.mean(pos_ir))
    ir_std = float(np.std(pos_ir))
    ir_sum = float(np.sum(pos_ir))

    # Percentiles
    ir_q25 = float(np.percentile(pos_ir, 25))
    ir_q50 = float(np.percentile(pos_ir, 50))
    ir_q75 = float(np.percentile(pos_ir, 75))

    # Moyennes par quartile d'intensité
    sorted_ir = np.sort(pos_ir)
    ir_q1_vals = sorted_ir[sorted_ir <= ir_q25]
    ir_q2_vals = sorted_ir[(sorted_ir > ir_q25) & (sorted_ir <= ir_q50)]
    ir_q3_vals = sorted_ir[(sorted_ir > ir_q50) & (sorted_ir <= ir_q75)]
    ir_q4_vals = sorted_ir[sorted_ir > ir_q75]

    # Top 5 intensités (triées décroissantes)
    sorted_ir_desc = np.sort(pos_ir)[::-1]

    # Fréquence du max IR
    max_idx = np.argmax(pos_ir)
    freq_of_max_ir = float(pos_freqs[max_idx]) if max_idx < len(pos_freqs) else np.nan

    # Fréquence moyenne pondérée par intensité IR
    if ir_sum > 0:
        freq_ir_weighted = float(np.sum(pos_freqs * pos_ir) / ir_sum)
    else:
        freq_ir_weighted = np.nan

    # IR par plage de fréquence
    low_mask = (pos_freqs >= 0) & (pos_freqs < 500)
    medium_mask = (pos_freqs >= 500) & (pos_freqs < 2000)
    high_mask = pos_freqs >= 2000

    ir_low = pos_ir[low_mask]
    ir_medium = pos_ir[medium_mask]
    ir_high = pos_ir[high_mask]

    sum_ir_low = float(np.sum(ir_low)) if len(ir_low) > 0 else 0.0
    sum_ir_medium = float(np.sum(ir_medium)) if len(ir_medium) > 0 else 0.0
    sum_ir_high = float(np.sum(ir_high)) if len(ir_high) > 0 else 0.0

    return {
        # Statistiques de base
        'ir_max': ir_max,
        'ir_min': ir_min,
        'ir_mean': ir_mean,
        'ir_std': ir_std,
        'ir_median': ir_q50,
        'ir_sum': ir_sum,
        'ir_spread': ir_max - ir_min,
        # Percentiles
        'ir_q25': ir_q25,
        'ir_q75': ir_q75,
        'ir_iqr': ir_q75 - ir_q25,
        # Moyennes par quartile
        'mean_ir_q1': float(np.mean(ir_q1_vals)) if len(ir_q1_vals) > 0 else np.nan,
        'mean_ir_q2': float(np.mean(ir_q2_vals)) if len(ir_q2_vals) > 0 else np.nan,
        'mean_ir_q3': float(np.mean(ir_q3_vals)) if len(ir_q3_vals) > 0 else np.nan,
        'mean_ir_q4': float(np.mean(ir_q4_vals)) if len(ir_q4_vals) > 0 else np.nan,
        # Top 5 intensités
        'ir_top1': float(sorted_ir_desc[0]) if len(sorted_ir_desc) > 0 else np.nan,
        'ir_top2': float(sorted_ir_desc[1]) if len(sorted_ir_desc) > 1 else np.nan,
        'ir_top3': float(sorted_ir_desc[2]) if len(sorted_ir_desc) > 2 else np.nan,
        'ir_top4': float(sorted_ir_desc[3]) if len(sorted_ir_desc) > 3 else np.nan,
        'ir_top5': float(sorted_ir_desc[4]) if len(sorted_ir_desc) > 4 else np.nan,
        # Fréquence du max IR
        'freq_of_max_ir': freq_of_max_ir,
        # Fréquence pondérée par IR
        'freq_ir_weighted': freq_ir_weighted,
        # IR par plage de fréquence
        'sum_ir_low': sum_ir_low,
        'sum_ir_medium': sum_ir_medium,
        'sum_ir_high': sum_ir_high,
        'mean_ir_low': float(np.mean(ir_low)) if len(ir_low) > 0 else np.nan,
        'mean_ir_medium': float(np.mean(ir_medium)) if len(ir_medium) > 0 else np.nan,
        'mean_ir_high': float(np.mean(ir_high)) if len(ir_high) > 0 else np.nan,
        # Fraction IR par plage
        'frac_ir_low': sum_ir_low / ir_sum if ir_sum > 0 else np.nan,
        'frac_ir_medium': sum_ir_medium / ir_sum if ir_sum > 0 else np.nan,
        'frac_ir_high': sum_ir_high / ir_sum if ir_sum > 0 else np.nan,
    }


def compute_frequency_descriptors(freq_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcule tous les descripteurs de fréquences à partir des données extraites.

    Args:
        freq_data (Dict): Données extraites par extract_all_frequency_data()
                          Doit contenir: 'frequencies', 'ir_intensities', 'zpe_kj'

    Returns:
        Dict contenant tous les descripteurs:
            - Statistiques de base: freq_max, freq_min, freq_mean, freq_std, freq_median, freq_spread
            - Comptages: n_modes_total, n_modes_low, n_modes_medium_low, n_modes_medium, n_modes_high
            - Énergie: zpe_kj
            - Indices: flexibility_index, stiffness_index
            - IR: ir_max, ir_sum, freq_of_max_ir

    Example:
        >>> freq_data = extract_all_frequency_data("path/to/adf.rkf")
        >>> descriptors = compute_frequency_descriptors(freq_data)
        >>> print(f"Flexibility index: {descriptors['flexibility_index']:.3f}")
    """
    frequencies = freq_data.get('frequencies', np.array([]))
    ir_intensities = freq_data.get('ir_intensities', None)

    # Initialiser le dictionnaire des descripteurs
    descriptors = {}

    # 1. Statistiques de base
    descriptors.update(compute_frequency_statistics(frequencies))

    # 2. Comptages par plage
    mode_counts = compute_mode_counts(frequencies)
    descriptors.update(mode_counts)

    # 3. ZPE (utiliser la valeur extraite si disponible, sinon calculer)
    zpe = freq_data.get('zpe_kj')
    if zpe is None or np.isnan(zpe):
        zpe = compute_zpe(frequencies)
    descriptors['zpe_kj'] = zpe

    # 4. Indices de flexibilité
    descriptors.update(compute_flexibility_indices(mode_counts))

    # 5. Descripteurs IR
    descriptors.update(compute_ir_descriptors(frequencies, ir_intensities))

    return descriptors


def get_descriptor_names() -> list:
    """
    Retourne la liste des noms de tous les descripteurs calculés.

    Returns:
        list: Liste des noms de descripteurs
    """
    return [
        # Statistiques de base
        'freq_max', 'freq_min', 'freq_mean', 'freq_std', 'freq_median', 'freq_spread',
        # Statistiques avancées
        'freq_q25', 'freq_q75', 'freq_iqr', 'freq_skewness', 'freq_kurtosis', 'freq_cv',
        'freq_harmonic_mean', 'freq_geometric_mean',
        # Premières fréquences individuelles
        'freq_1', 'freq_2', 'freq_3', 'freq_4', 'freq_5',
        # Sommes et moyennes par plage spectrale
        'sum_freq_low', 'sum_freq_medium', 'sum_freq_high',
        'mean_freq_low', 'mean_freq_medium', 'mean_freq_high',
        # Moyennes par quartile de la distribution
        'mean_freq_q1', 'mean_freq_q2', 'mean_freq_q3', 'mean_freq_q4',
        # Comptages
        'n_modes_total', 'n_modes_low', 'n_modes_medium_low', 'n_modes_medium', 'n_modes_high',
        # Énergie
        'zpe_kj',
        # Indices et ratios
        'flexibility_index', 'stiffness_index',
        'ratio_low_high', 'ratio_low_medium', 'ratio_medium_high',
        'frac_bending', 'frac_stretching',
        # IR - Statistiques de base
        'ir_max', 'ir_min', 'ir_mean', 'ir_std', 'ir_median', 'ir_sum', 'ir_spread',
        # IR - Percentiles
        'ir_q25', 'ir_q75', 'ir_iqr',
        # IR - Moyennes par quartile
        'mean_ir_q1', 'mean_ir_q2', 'mean_ir_q3', 'mean_ir_q4',
        # IR - Top 5 intensités
        'ir_top1', 'ir_top2', 'ir_top3', 'ir_top4', 'ir_top5',
        # IR - Fréquence associée
        'freq_of_max_ir', 'freq_ir_weighted',
        # IR - Par plage de fréquence
        'sum_ir_low', 'sum_ir_medium', 'sum_ir_high',
        'mean_ir_low', 'mean_ir_medium', 'mean_ir_high',
        'frac_ir_low', 'frac_ir_medium', 'frac_ir_high',
    ]


def describe_descriptor(name: str) -> str:
    """
    Retourne une description textuelle d'un descripteur.

    Args:
        name (str): Nom du descripteur

    Returns:
        str: Description du descripteur
    """
    descriptions = {
        'freq_max': "Fréquence maximale (cm⁻¹)",
        'freq_min': "Fréquence minimale positive (cm⁻¹)",
        'freq_mean': "Fréquence moyenne (cm⁻¹)",
        'freq_std': "Écart-type des fréquences (cm⁻¹)",
        'freq_median': "Fréquence médiane (cm⁻¹)",
        'freq_spread': "Étendue des fréquences (cm⁻¹)",
        'n_modes_total': "Nombre total de modes vibrationnels",
        'n_modes_low': "Modes basse fréquence (0-500 cm⁻¹) - flexibilité",
        'n_modes_medium_low': "Modes moyens-bas (500-1000 cm⁻¹) - bending",
        'n_modes_medium': "Modes moyens (1000-2000 cm⁻¹) - stretching",
        'n_modes_high': "Modes haute fréquence (>2000 cm⁻¹) - X-H stretches",
        'zpe_kj': "Énergie de point zéro (kJ/mol)",
        'flexibility_index': "Indice de flexibilité (n_low/n_total)",
        'stiffness_index': "Indice de rigidité (n_high/n_total)",
        'ir_max': "Intensité IR maximale (km/mol)",
        'ir_sum': "Somme des intensités IR (km/mol)",
        'freq_of_max_ir': "Fréquence du pic IR maximum (cm⁻¹)",
    }

    return descriptions.get(name, f"Descripteur inconnu: {name}")
