"""
Analyse des combinaisons bilinéaires pour les relations structure-propriété.

Ce module recherche les paires de descripteurs (x1, x2) qui, combinées linéairement,
prédisent bien une propriété cible y :

    y = a·x1 + b·x2 + c

L'objectif est de trouver des relations avec R² ≥ seuil (par défaut 0.8).

Usage:
    from visualization.bilinear_analysis import find_bilinear_correlations

    results = find_bilinear_correlations(df, r2_threshold=0.8)
    export_bilinear_results(results, "bilinear_results.txt")
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd


def fit_bilinear(
    df: pd.DataFrame,
    x1_col: str,
    x2_col: str,
    y_col: str
) -> Optional[Dict]:
    """
    Ajuste un modèle bilinéaire y = a·x1 + b·x2 + c.

    Args:
        df: DataFrame avec les données
        x1_col: Première variable explicative
        x2_col: Deuxième variable explicative
        y_col: Variable cible

    Returns:
        Dictionnaire avec les résultats ou None si pas assez de données:
        - r2: Coefficient de détermination
        - coeffs: (a, b, c) coefficients du modèle
        - n_points: Nombre de points utilisés
    """
    # Vérifier les colonnes
    if x1_col not in df.columns or x2_col not in df.columns or y_col not in df.columns:
        return None

    # Données valides (sans NaN)
    valid = df[[x1_col, x2_col, y_col]].dropna()
    n_points = len(valid)

    # Minimum 4 points pour 3 paramètres
    if n_points < 4:
        return None

    # Construire la matrice de design [x1, x2, 1]
    X = np.column_stack([
        valid[x1_col].values,
        valid[x2_col].values,
        np.ones(n_points)
    ])
    y = valid[y_col].values

    try:
        # Résolution par moindres carrés
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Prédictions
        y_pred = X @ coeffs

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return None

        r2 = 1 - ss_res / ss_tot

        # R² ajusté (pénalise le nombre de paramètres)
        r2_adj = 1 - (1 - r2) * (n_points - 1) / (n_points - 3)

        return {
            'r2': r2,
            'r2_adj': r2_adj,
            'coeffs': tuple(coeffs),
            'n_points': n_points
        }

    except Exception:
        return None


def get_neutral_only_descriptors(df: pd.DataFrame) -> List[str]:
    """
    Retourne uniquement les colonnes descripteurs du conformère neutre.

    Exclut:
    - Toute colonne contenant '_red' ou '_ox' (états chargés)
    - Toute colonne contenant 'delta_' (variations)
    - Les colonnes cibles (potentiels, énergies redox)

    Returns:
        Liste des noms de colonnes neutres uniquement
    """
    # Colonnes à exclure (cibles redox)
    redox_targets = {
        'E_reduction_V', 'E_oxidation_V',
        'EA_V', 'IP_V', 'EA_kJ', 'IP_kJ',
        'dG_reduction_kJ', 'dG_oxidation_kJ',
        'Edef_red_kJ', 'Edef_ox_kJ'
    }

    neutral_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        # Exclure si contient _red ou _ox
        if '_red' in col or '_ox' in col:
            continue
        # Exclure si contient delta_
        if 'delta_' in col:
            continue
        # Exclure les cibles redox
        if col in redox_targets:
            continue
        # Exclure molecule_name si présent
        if col == 'molecule_name':
            continue
        neutral_cols.append(col)

    return neutral_cols


def get_redox_targets(df: pd.DataFrame) -> List[str]:
    """
    Retourne les colonnes cibles à prédire (propriétés redox).

    Inclut:
    - Potentiels redox (E_reduction_V, E_oxidation_V)
    - Énergies (EA, IP, ΔG, Edef)
    - Variations de charges Bader (delta_*_red, delta_*_ox)
    - Variations d'énergies IQA (delta_*_iqa_red, delta_*_iqa_ox)

    Returns:
        Liste des noms de colonnes cibles
    """
    targets = []

    # Colonnes explicites
    explicit_targets = [
        'E_reduction_V', 'E_oxidation_V',
        'EA_V', 'IP_V', 'EA_kJ', 'IP_kJ',
        'dG_reduction_kJ', 'dG_oxidation_kJ',
        'Edef_red_kJ', 'Edef_ox_kJ'
    ]

    for col in df.columns:
        # Ajouter les cibles explicites
        if col in explicit_targets:
            targets.append(col)
        # Ajouter les deltas (variations lors de la réduction/oxydation)
        elif 'delta_' in col and ('_red' in col or '_ox' in col):
            targets.append(col)

    return targets


def find_neutral_only_bilinear(
    df: pd.DataFrame,
    r2_threshold: float = 0.8,
    min_points: int = 10
) -> List[Dict]:
    """
    Recherche les combinaisons bilinéaires utilisant UNIQUEMENT les descripteurs neutres.

    Objectif: Prédire les propriétés redox à partir d'un seul calcul (conformère neutre).

    Modèle: y_redox = a·x1_neutre + b·x2_neutre + c

    Args:
        df: DataFrame avec les données
        r2_threshold: Seuil R² minimum (défaut: 0.8)
        min_points: Nombre minimum de points (défaut: 10)

    Returns:
        Liste de dictionnaires avec les résultats triés par R² décroissant
    """
    # Obtenir les descripteurs neutres uniquement
    neutral_descriptors = get_neutral_only_descriptors(df)

    # Obtenir les cibles redox
    redox_targets = get_redox_targets(df)

    # Filtrer les colonnes présentes
    neutral_descriptors = [c for c in neutral_descriptors if c in df.columns]
    redox_targets = [c for c in redox_targets if c in df.columns]

    print("=" * 80)
    print("ANALYSE BILINÉAIRE - PRÉDICTION DEPUIS CONFORMÈRE NEUTRE UNIQUEMENT")
    print("=" * 80)
    print()
    print(f"Descripteurs neutres ({len(neutral_descriptors)}):")
    for col in neutral_descriptors:
        print(f"  - {col}")
    print()
    print(f"Cibles redox à prédire ({len(redox_targets)}):")
    for col in redox_targets[:10]:
        print(f"  - {col}")
    if len(redox_targets) > 10:
        print(f"  ... et {len(redox_targets) - 10} autres")
    print()

    n_combinations = len(redox_targets) * len(neutral_descriptors) * (len(neutral_descriptors) - 1) // 2
    print(f"Combinaisons à tester: {n_combinations}")
    print(f"Seuil R² >= {r2_threshold}, minimum {min_points} points")
    print()

    results = []
    n_tested = 0
    n_found = 0

    # Triple boucle: cible redox, descripteur1 neutre, descripteur2 neutre
    for y_col in redox_targets:
        for i, x1_col in enumerate(neutral_descriptors):
            for x2_col in neutral_descriptors[i+1:]:
                n_tested += 1

                # Ajuster le modèle bilinéaire
                result = fit_bilinear(df, x1_col, x2_col, y_col)

                if result is not None and result['r2'] >= r2_threshold and result['n_points'] >= min_points:
                    n_found += 1
                    results.append({
                        'target': y_col,
                        'x1': x1_col,
                        'x2': x2_col,
                        'r2': result['r2'],
                        'r2_adj': result['r2_adj'],
                        'a': result['coeffs'][0],
                        'b': result['coeffs'][1],
                        'c': result['coeffs'][2],
                        'n_points': result['n_points']
                    })

    print(f"Résultats: {n_found}/{n_tested} combinaisons avec R² >= {r2_threshold}")

    # Trier par R² décroissant
    results.sort(key=lambda x: x['r2'], reverse=True)

    return results


def find_bilinear_correlations(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    descriptor_cols: Optional[List[str]] = None,
    r2_threshold: float = 0.8,
    min_points: int = 10,
    exclude_self: bool = True,
    exclude_trivial: bool = True
) -> List[Dict]:
    """
    Recherche toutes les combinaisons bilinéaires avec R² >= seuil.

    Args:
        df: DataFrame avec les données
        target_cols: Colonnes cibles à prédire (défaut: potentiels redox + énergies)
        descriptor_cols: Colonnes descripteurs (défaut: toutes sauf targets)
        r2_threshold: Seuil R² minimum (défaut: 0.8)
        min_points: Nombre minimum de points pour éviter le surapprentissage (défaut: 10)
        exclude_self: Exclure les paires où x1 ou x2 est la cible
        exclude_trivial: Exclure les combinaisons triviales (même famille)

    Returns:
        Liste de dictionnaires avec les résultats triés par R² décroissant
    """
    # Colonnes cibles par défaut
    if target_cols is None:
        target_cols = [
            'E_reduction_V', 'E_oxidation_V',
            'dG_reduction_kJ', 'dG_oxidation_kJ',
            'EA_kJ', 'IP_kJ', 'EA_V', 'IP_V'
        ]

    # Filtrer les cibles présentes
    target_cols = [c for c in target_cols if c in df.columns]

    # Colonnes descripteurs par défaut
    if descriptor_cols is None:
        # Toutes les colonnes numériques sauf molecule_name et targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        descriptor_cols = [c for c in numeric_cols if c not in target_cols]

    # Filtrer les descripteurs présents
    descriptor_cols = [c for c in descriptor_cols if c in df.columns]

    print(f"Analyse bilinéaire:")
    print(f"  - {len(target_cols)} cibles: {', '.join(target_cols[:5])}...")
    print(f"  - {len(descriptor_cols)} descripteurs")
    print(f"  - Seuil R² >= {r2_threshold}")
    print(f"  - Minimum {min_points} points (anti-surapprentissage)")
    print(f"  - Combinaisons à tester: {len(target_cols) * len(descriptor_cols) * (len(descriptor_cols)-1) // 2}")
    print()

    results = []
    n_tested = 0
    n_found = 0

    # Triple boucle: cible, descripteur1, descripteur2
    for y_col in target_cols:
        for i, x1_col in enumerate(descriptor_cols):
            # Exclure si x1 est la cible
            if exclude_self and x1_col == y_col:
                continue

            for x2_col in descriptor_cols[i+1:]:  # Éviter doublons (x1,x2) et (x2,x1)
                # Exclure si x2 est la cible
                if exclude_self and x2_col == y_col:
                    continue

                # Exclure combinaisons triviales (même base de descripteur)
                if exclude_trivial:
                    # Exemple: delta_O7_red + delta_O8_red pour prédire delta_q_O_total_red
                    x1_base = x1_col.replace('_red', '').replace('_ox', '').replace('delta_', '')
                    x2_base = x2_col.replace('_red', '').replace('_ox', '').replace('delta_', '')
                    y_base = y_col.replace('_red', '').replace('_ox', '').replace('delta_', '')

                    if x1_base == y_base or x2_base == y_base:
                        continue

                n_tested += 1

                # Ajuster le modèle bilinéaire
                result = fit_bilinear(df, x1_col, x2_col, y_col)

                if result is not None and result['r2'] >= r2_threshold and result['n_points'] >= min_points:
                    n_found += 1
                    results.append({
                        'target': y_col,
                        'x1': x1_col,
                        'x2': x2_col,
                        'r2': result['r2'],
                        'r2_adj': result['r2_adj'],
                        'a': result['coeffs'][0],
                        'b': result['coeffs'][1],
                        'c': result['coeffs'][2],
                        'n_points': result['n_points']
                    })

    print(f"  - {n_tested} combinaisons testées")
    print(f"  - {n_found} avec R² >= {r2_threshold}")

    # Trier par R² décroissant
    results.sort(key=lambda x: x['r2'], reverse=True)

    return results


def export_bilinear_results(
    results: List[Dict],
    output_file: Path,
    top_n: Optional[int] = None,
    title: str = "ANALYSE DES COMBINAISONS BILINÉAIRES"
) -> None:
    """
    Exporte les résultats de l'analyse bilinéaire dans un fichier texte.

    Args:
        results: Liste des résultats de find_bilinear_correlations
        output_file: Fichier de sortie
        top_n: Limiter aux N meilleurs résultats (défaut: tous)
        title: Titre du rapport
    """
    output_file = Path(output_file)

    if top_n is not None:
        results = results[:top_n]

    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 100 + "\n\n")
        f.write("Modèle: y = a·x1 + b·x2 + c\n\n")
        f.write(f"Nombre de combinaisons trouvées: {len(results)}\n\n")

        if not results:
            f.write("Aucune combinaison avec R² suffisant.\n")
            return

        # Grouper par cible
        by_target = {}
        for r in results:
            target = r['target']
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(r)

        for target, target_results in by_target.items():
            f.write("-" * 100 + "\n")
            f.write(f"CIBLE: {target}\n")
            f.write("-" * 100 + "\n\n")

            for i, r in enumerate(target_results, 1):
                f.write(f"  [{i}] R² = {r['r2']:.4f} (R²_adj = {r['r2_adj']:.4f})\n")
                f.write(f"      {target} = {r['a']:+.4f}·{r['x1']} {r['b']:+.4f}·{r['x2']} {r['c']:+.4f}\n")
                f.write(f"      (n = {r['n_points']} points)\n\n")

        # Top 10 global
        f.write("\n" + "=" * 100 + "\n")
        f.write("TOP 10 MEILLEURES COMBINAISONS (toutes cibles confondues)\n")
        f.write("=" * 100 + "\n\n")

        for i, r in enumerate(results[:10], 1):
            f.write(f"  {i:2d}. R² = {r['r2']:.4f}  |  {r['target']}\n")
            f.write(f"      = {r['a']:+.4f}·{r['x1']}\n")
            f.write(f"        {r['b']:+.4f}·{r['x2']}\n")
            f.write(f"        {r['c']:+.4f}\n\n")

    print(f"✓ Résultats exportés: {output_file}")


def plot_best_bilinear(
    df: pd.DataFrame,
    results: List[Dict],
    output_dir: Path,
    top_n: int = 6
) -> None:
    """
    Génère des graphiques 3D pour les meilleures combinaisons bilinéaires.

    Args:
        df: DataFrame avec les données
        results: Liste des résultats
        output_dir: Répertoire de sortie
        top_n: Nombre de graphiques à générer
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sélectionner les top_n résultats
    top_results = results[:top_n]

    if not top_results:
        print("Pas de résultats à visualiser")
        return

    # Créer une grille de subplots
    n_cols = min(3, len(top_results))
    n_rows = (len(top_results) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for idx, r in enumerate(top_results):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        # Données
        valid = df[[r['x1'], r['x2'], r['target'], 'molecule_name']].dropna()

        x1 = valid[r['x1']].values
        x2 = valid[r['x2']].values
        y = valid[r['target']].values

        # Points
        ax.scatter(x1, x2, y, c='steelblue', s=50, alpha=0.8)

        # Plan de régression
        x1_grid = np.linspace(x1.min(), x1.max(), 10)
        x2_grid = np.linspace(x2.min(), x2.max(), 10)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        Y_pred = r['a'] * X1 + r['b'] * X2 + r['c']

        ax.plot_surface(X1, X2, Y_pred, alpha=0.3, color='red')

        # Labels
        ax.set_xlabel(r['x1'][:15], fontsize=8)
        ax.set_ylabel(r['x2'][:15], fontsize=8)
        ax.set_zlabel(r['target'][:15], fontsize=8)
        ax.set_title(f"R² = {r['r2']:.3f}", fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)

    plt.suptitle('Meilleures Combinaisons Bilinéaires', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "bilinear_3d_plots.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphiques 3D sauvegardés: {output_file.name}")


if __name__ == "__main__":
    # Test avec les données existantes
    import sys

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "batch_viz_test/molecules_summary.csv"

    print(f"Chargement de {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"  {len(df)} molécules, {len(df.columns)} colonnes\n")

    # Mode par défaut: analyse neutre uniquement (prédiction depuis conformère neutre)
    mode = sys.argv[2] if len(sys.argv) > 2 else "neutral"

    if mode == "neutral":
        # NOUVEAU: Prédiction depuis conformère neutre uniquement
        results = find_neutral_only_bilinear(df, r2_threshold=0.8, min_points=10)

        # Export
        output_file = Path(csv_file).parent / "bilinear_neutral_only.txt"
        export_bilinear_results(
            results, output_file,
            title="PRÉDICTION REDOX DEPUIS CONFORMÈRE NEUTRE UNIQUEMENT"
        )

        # Graphiques 3D
        if results:
            plot_best_bilinear(df, results, Path(csv_file).parent, top_n=6)

    elif mode == "all":
        # Ancien mode: tous les descripteurs
        results = find_bilinear_correlations(df, r2_threshold=0.8, min_points=10)

        # Export
        output_file = Path(csv_file).parent / "bilinear_analysis.txt"
        export_bilinear_results(results, output_file)

        # Graphiques 3D
        if results:
            plot_best_bilinear(df, results, Path(csv_file).parent, top_n=6)

    else:
        print(f"Mode inconnu: {mode}")
        print("Modes disponibles: neutral, all")
