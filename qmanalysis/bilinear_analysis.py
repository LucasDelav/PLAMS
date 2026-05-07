"""
Analyse des corrélations bilinéaires entre descripteurs de fréquences et Edef.

Ce module teste toutes les combinaisons de paires de descripteurs
pour des régressions bilinéaires de la forme: y = a*x1 + b*x2 + c
"""

from typing import Dict, List, Tuple, Any
from itertools import combinations

import numpy as np


def bilinear_regression(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Effectue une régression bilinéaire: y = a*x1 + b*x2 + c

    Args:
        x1: Premier descripteur
        x2: Second descripteur
        y: Variable cible

    Returns:
        Dict avec: a, b, c (coefficients), r2, adj_r2, p_value, n
    """
    from scipy.stats import f as f_dist

    # Filtrer les NaN
    mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
    x1_clean = x1[mask]
    x2_clean = x2[mask]
    y_clean = y[mask]

    n = len(y_clean)

    if n < 4:  # Besoin d'au moins 4 points pour 3 paramètres
        return {
            'a': np.nan, 'b': np.nan, 'c': np.nan,
            'r2': np.nan, 'adj_r2': np.nan, 'p_value': np.nan, 'n': n
        }

    # Matrice de design: [x1, x2, 1]
    X = np.column_stack([x1_clean, x2_clean, np.ones(n)])

    try:
        # Résoudre par moindres carrés
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y_clean, rcond=None)
        a, b, c = coeffs

        # Prédictions
        y_pred = a * x1_clean + b * x2_clean + c

        # R²
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)

        if ss_tot == 0:
            r2 = np.nan
            adj_r2 = np.nan
        else:
            r2 = 1 - ss_res / ss_tot
            # R² ajusté
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 3)  # 3 paramètres

        # Test F pour p-value
        k = 2  # Nombre de prédicteurs (hors constante)
        if ss_res > 0 and n > 3:
            ms_reg = (ss_tot - ss_res) / k
            ms_res = ss_res / (n - k - 1)
            if ms_res > 0:
                f_stat = ms_reg / ms_res
                p_value = 1 - f_dist.cdf(f_stat, k, n - k - 1)
            else:
                p_value = np.nan
        else:
            p_value = np.nan

        return {
            'a': float(a), 'b': float(b), 'c': float(c),
            'r2': float(r2), 'adj_r2': float(adj_r2),
            'p_value': float(p_value), 'n': int(n)
        }

    except np.linalg.LinAlgError:
        return {
            'a': np.nan, 'b': np.nan, 'c': np.nan,
            'r2': np.nan, 'adj_r2': np.nan, 'p_value': np.nan, 'n': n
        }


def compute_all_bilinear_correlations(
    descriptors: Dict[str, np.ndarray],
    y: np.ndarray,
    target_name: str = "target"
) -> List[Dict[str, Any]]:
    """
    Teste toutes les combinaisons de paires de descripteurs.

    Args:
        descriptors: Dict {nom_descripteur: valeurs}
        y: Variable cible
        target_name: Nom de la cible pour l'affichage

    Returns:
        Liste de dicts triés par R² décroissant
    """
    desc_names = list(descriptors.keys())
    results = []

    for name1, name2 in combinations(desc_names, 2):
        x1 = descriptors[name1]
        x2 = descriptors[name2]

        reg_result = bilinear_regression(x1, x2, y)

        if not np.isnan(reg_result['r2']):
            results.append({
                'desc1': name1,
                'desc2': name2,
                'target': target_name,
                **reg_result
            })

    # Trier par R² décroissant
    results.sort(key=lambda x: x['r2'], reverse=True)

    return results


def run_bilinear_analysis(
    descriptors_by_mol: Dict[str, Dict[str, float]],
    edef_values: Dict[str, Dict[str, float]],
    molecule_names: List[str]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Exécute l'analyse bilinéaire complète.

    Args:
        descriptors_by_mol: {mol_name: {descriptor: value}}
        edef_values: {mol_name: {'Edef_red_kJ': float, 'Edef_ox_kJ': float}}
        molecule_names: Liste des noms de molécules

    Returns:
        Tuple (results_reduction, results_oxidation) avec listes triées par R²
    """
    # Convertir en arrays
    desc_names = set()
    for mol_desc in descriptors_by_mol.values():
        desc_names.update(mol_desc.keys())
    desc_names = sorted(desc_names)

    # Filtrer les descripteurs qui ne sont pas tous NaN
    valid_desc_names = []
    descriptors = {}

    for name in desc_names:
        values = []
        for mol in molecule_names:
            val = descriptors_by_mol.get(mol, {}).get(name, np.nan)
            values.append(val)
        arr = np.array(values)

        # Garder si au moins 4 valeurs non-NaN
        if np.sum(~np.isnan(arr)) >= 4:
            descriptors[name] = arr
            valid_desc_names.append(name)

    # Cibles
    y_red = np.array([edef_values.get(mol, {}).get('Edef_red_kJ', np.nan)
                      for mol in molecule_names])
    y_ox = np.array([edef_values.get(mol, {}).get('Edef_ox_kJ', np.nan)
                     for mol in molecule_names])

    # Analyse pour réduction
    results_red = compute_all_bilinear_correlations(descriptors, y_red, "Edef_red_kJ")

    # Analyse pour oxydation
    results_ox = compute_all_bilinear_correlations(descriptors, y_ox, "Edef_ox_kJ")

    return results_red, results_ox


def format_bilinear_results(results: List[Dict], top_n: int = 20) -> str:
    """
    Formate les résultats pour affichage.

    Args:
        results: Liste des résultats de régression
        top_n: Nombre de résultats à afficher

    Returns:
        String formaté
    """
    if not results:
        return "Aucun résultat disponible"

    lines = []
    lines.append(f"{'Rang':<5} {'Desc1':<25} {'Desc2':<25} {'R²':<8} {'R²adj':<8} {'p-value':<12} {'n':<4}")
    lines.append("-" * 95)

    for i, res in enumerate(results[:top_n], 1):
        lines.append(
            f"{i:<5} {res['desc1']:<25} {res['desc2']:<25} "
            f"{res['r2']:.4f}  {res['adj_r2']:.4f}  {res['p_value']:.2e}  {res['n']:<4}"
        )

    return "\n".join(lines)
