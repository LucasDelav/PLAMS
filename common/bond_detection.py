"""
Détection de liaisons covalentes et analyse de rupture de liaison.

Module partagé entre le pipeline redox (qmredox/analysis.py) et
les scripts d'analyse post-calcul (analysis/analyze_bond_breaking.py).
"""

import numpy as np


# Rayons covalents (Å) - Cordero et al. 2008
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'Al': 1.21,
}

BOND_TOLERANCE = 0.40  # Å ajoutés à la somme des rayons covalents
BREAKING_THRESHOLD = 1.0  # Å - rupture de liaison
ELONGATION_THRESHOLD = 0.5  # Å - changement significatif

BOHR_TO_ANG = 0.529177249

Z_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl',
    13: 'Al', 35: 'Br', 53: 'I',
}


def detect_bonds(symbols, coords):
    """Détecte les liaisons covalentes par critère de distance.

    Args:
        symbols: list[str] - symboles atomiques
        coords: np.ndarray shape (n, 3) - coordonnées en Å

    Returns:
        list of (i, j) avec i < j
    """
    n = len(symbols)
    bonds = []
    for i in range(n):
        ri = COVALENT_RADII.get(symbols[i], 0.77)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(symbols[j], 0.77)
            d = np.linalg.norm(coords[i] - coords[j])
            if d < ri + rj + BOND_TOLERANCE:
                bonds.append((i, j))
    return bonds


def bond_type_label(sym1, sym2):
    """Retourne un label canonique C-Br, C-O, etc. (élément lourd en second)."""
    pair = tuple(sorted([sym1, sym2], key=lambda s: COVALENT_RADII.get(s, 0)))
    return f'{pair[0]}-{pair[1]}'


def compare_bonds(symbols, coords_ref, coords_charged, threshold=BREAKING_THRESHOLD):
    """Compare les distances de liaison entre deux géométries.

    Détecte les liaisons dans la géométrie de référence et calcule
    l'élongation dans la géométrie chargée.

    Args:
        symbols: list[str] - symboles atomiques (identiques pour les deux géométries)
        coords_ref: np.ndarray (n, 3) - géométrie de référence (neutre) en Å
        coords_charged: np.ndarray (n, 3) - géométrie chargée (réduit/oxydé) en Å
        threshold: float - seuil de rupture en Å

    Returns:
        list of dict: liaisons rompues (élongation > threshold), chaque dict contient :
            atom1_idx, atom1_symbol, atom2_idx, atom2_symbol, bond_type,
            d_ref, d_charged, elongation
    """
    bonds = detect_bonds(symbols, coords_ref)
    broken = []

    for i, j in bonds:
        d_ref = np.linalg.norm(coords_ref[i] - coords_ref[j])
        d_charged = np.linalg.norm(coords_charged[i] - coords_charged[j])
        elongation = d_charged - d_ref

        if elongation > threshold:
            broken.append({
                'atom1_idx': i + 1,  # 1-indexed
                'atom1_symbol': symbols[i],
                'atom2_idx': j + 1,
                'atom2_symbol': symbols[j],
                'bond_type': bond_type_label(symbols[i], symbols[j]),
                'd_ref': round(d_ref, 3),
                'd_charged': round(d_charged, 3),
                'elongation': round(elongation, 3),
            })

    return broken
