"""
Module de visualisation batch pour comparaisons inter-molécules.

Ce module génère des graphiques comparant plusieurs molécules entre elles,
où chaque point représente une molécule différente (pas un conformère).

Graphiques générés:
- Tous les graphiques de corrélation (charges Bader + IQA vs potentiels redox)
- Corrélations orbitales frontières (HOMO vs E_ox, LUMO vs E_red)
- Grille complète assemblée en format rectangulaire
- Heatmap de corrélation complète

Usage:
    from visualization.batch_correlations import generate_batch_plots
    import pandas as pd

    df = pd.DataFrame({...})  # 1 ligne par molécule
    generate_batch_plots(df, output_dir=Path("batch_viz"))
"""

from pathlib import Path
from typing import Optional, List, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def plot_correlation_heatmap_batch(
    df: pd.DataFrame,
    output_dir: Path,
    molecule_name: Optional[str] = None
) -> None:
    """
    Heatmap de corrélation R² entre toutes les propriétés.

    Args:
        df: DataFrame avec toutes les propriétés numériques
        output_dir: Répertoire de sortie
        molecule_name: Ignoré en mode batch

    Exports:
        - batch_correlation_heatmap.png
    """
    output_dir = Path(output_dir)

    # Sélectionner colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        warnings.warn("Pas assez de colonnes numériques pour la heatmap")
        return

    # Calculer corrélations R² (carré des corrélations)
    corr_matrix = df[numeric_cols].corr() ** 2

    # Taille adaptée au nombre de colonnes
    n_cols = len(numeric_cols)
    fig_size = max(16, n_cols * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    # Essayer seaborn si disponible, sinon matplotlib pur
    try:
        import seaborn as sns
        sns.heatmap(
            corr_matrix,
            annot=False,  # Pas d'annotations - juste les couleurs
            cmap='Reds',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.3,
            cbar_kws={"shrink": 0.6, "label": "R²"},
            ax=ax,
            xticklabels=True,
            yticklabels=True
        )
        # Ajuster la taille des labels
        ax.tick_params(axis='x', labelsize=7, rotation=90)
        ax.tick_params(axis='y', labelsize=7, rotation=0)
        print("  (utilisé seaborn pour rendu amélioré)")

    except ImportError:
        # Fallback matplotlib pur
        im = ax.imshow(corr_matrix, cmap='Reds', vmin=0, vmax=1, aspect='equal')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('R²', rotation=270, labelpad=15)

        # Labels
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90, ha='center', fontsize=7)
        ax.set_yticklabels(numeric_cols, fontsize=7)

        print("  (utilisé matplotlib pur - seaborn non disponible)")

    ax.set_title('Matrice de Corrélation R² (Comparaison Inter-Molécules)',
                fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / "batch_correlation_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Heatmap sauvegardée: batch_correlation_heatmap.png")


def plot_property_vs_potential(
    df: pd.DataFrame,
    property_col: str,
    potential_col: str,
    output_file: Path,
    title: Optional[str] = None
) -> Tuple[float, float]:
    """
    Génère un graphique de corrélation générique property vs potential.

    Args:
        df: DataFrame avec les données
        property_col: Nom de la colonne propriété (X)
        potential_col: Nom de la colonne potentiel (Y)
        output_file: Fichier de sortie
        title: Titre optionnel

    Returns:
        Tuple (R², slope) de la régression linéaire
    """
    # Vérifier que les colonnes existent
    if property_col not in df.columns or potential_col not in df.columns:
        warnings.warn(f"Colonnes manquantes: {property_col} ou {potential_col}")
        return (0.0, 0.0)

    # Supprimer les NaN
    df_clean = df[[property_col, potential_col, 'molecule_name']].dropna()

    if len(df_clean) < 2:
        warnings.warn(f"Pas assez de données pour {property_col} vs {potential_col}")
        return (0.0, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(df_clean[property_col], df_clean[potential_col], s=100, alpha=0.7, color='steelblue')

    # Labels des molécules
    for _, row in df_clean.iterrows():
        ax.annotate(
            row['molecule_name'],
            (row[property_col], row[potential_col]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=9, alpha=0.8
        )

    # Régression linéaire
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(df_clean[property_col], df_clean[potential_col])

    x_fit = np.linspace(df_clean[property_col].min(), df_clean[property_col].max(), 100)
    y_fit = slope * x_fit + intercept
    y_pred_on_data = slope * df_clean[property_col].values + intercept
    mae = np.mean(np.abs(df_clean[potential_col].values - y_pred_on_data))
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'MAE = {mae:.1f} kJ/mol, R² = {r_value**2:.3f}')

    # Labels
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'{property_col} vs {potential_col}', fontsize=11, fontweight='bold')

    ax.set_xlabel(property_col, fontsize=10)
    ax.set_ylabel(potential_col, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return (r_value**2, slope)


def generate_frontier_orbital_correlations(
    df: pd.DataFrame,
    output_dir: Path
) -> List[Tuple[str, str, float, float, Path]]:
    """
    Génère les graphiques de corrélation des orbitales frontières vs potentiels redox.
    
    Graphiques générés:
    - HOMO vs E_oxidation_V
    - LUMO vs E_reduction_V
    
    Args:
        df: DataFrame avec les données (doit contenir HOMO_eV, LUMO_eV, E_oxidation_V, E_reduction_V)
        output_dir: Répertoire de sortie (single_graphs/)
        
    Returns:
        Liste de tuples (property, potential, R², slope, filepath)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print("\n" + "-"*80)
    print(f"GÉNÉRATION DES CORRÉLATIONS ORBITALES FRONTIÈRES")
    print("-"*80)
    
    # 1. HOMO vs E_oxidation
    if 'HOMO_eV' in df.columns and 'E_oxidation_V' in df.columns:
        output_file = output_dir / "HOMO_vs_oxidation.png"
        title = "HOMO Energy vs Oxidation Potential"
        
        r2, slope = plot_property_vs_potential(
            df, 
            'HOMO_eV', 
            'E_oxidation_V', 
            output_file, 
            title
        )
        
        results.append(('HOMO_eV', 'E_oxidation_V', r2, slope, output_file))
        print(f"✓ HOMO vs E_ox (R² = {r2:.3f})")
    else:
        print("✗ HOMO vs E_ox : colonnes manquantes")
    
    # 2. LUMO vs E_reduction
    if 'LUMO_eV' in df.columns and 'E_reduction_V' in df.columns:
        output_file = output_dir / "LUMO_vs_reduction.png"
        title = "LUMO Energy vs Reduction Potential"
        
        r2, slope = plot_property_vs_potential(
            df, 
            'LUMO_eV', 
            'E_reduction_V', 
            output_file, 
            title
        )
        
        results.append(('LUMO_eV', 'E_reduction_V', r2, slope, output_file))
        print(f"✓ LUMO vs E_red (R² = {r2:.3f})")
    else:
        print("✗ LUMO vs E_red : colonnes manquantes")
    
    print(f"\n✓ {len(results)} corrélations orbitales frontières générées")
    
    return results


def generate_all_correlations(
    df: pd.DataFrame,
    output_dir: Path,
    save_individual: bool = True
) -> List[Tuple[str, str, float, float, Path]]:
    """
    Génère tous les graphiques de corrélation (charges Bader + IQA) vs potentiels redox.

    Args:
        df: DataFrame avec toutes les données
        output_dir: Répertoire de sortie
        save_individual: Si True, sauvegarde aussi les graphiques individuels dans single_graphs/

    Returns:
        Liste de tuples (property, potential, R², slope, filepath)

    Example:
        Les graphiques générés incluent :
        - Toutes les charges Bader (bader_charge_X) vs E_reduction_V et E_oxidation_V
        - Toutes les énergies IQA (*_iqa) vs E_reduction_V et E_oxidation_V
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_individual:
        single_dir = output_dir / "single_graphs"
        single_dir.mkdir(parents=True, exist_ok=True)
    else:
        single_dir = output_dir

    # Identifier les propriétés d'intérêt (format harmonisé: C1_bader, C1_iqa, etc.)
    bader_cols = [col for col in df.columns if col.endswith('_bader')]
    iqa_cols = [col for col in df.columns if col.endswith('_iqa')]

    # Δ-descripteurs de transfert de charge (delta_C1_red, delta_O7_ox, etc.)
    delta_cols = [col for col in df.columns if col.startswith('delta_')]

    # Potentiels redox
    potentials = ['E_reduction_V', 'E_oxidation_V']

    results = []

    all_props = bader_cols + iqa_cols + delta_cols

    print("\n" + "-"*80)
    print(f"GÉNÉRATION DE TOUS LES GRAPHIQUES DE CORRÉLATION")
    print("-"*80)
    print(f"Propriétés Bader:      {len(bader_cols)}")
    print(f"Propriétés IQA:        {len(iqa_cols)}")
    print(f"Δ-descripteurs:        {len(delta_cols)}")
    print(f"Potentiels:            {len(potentials)}")
    print(f"Total graphiques:      {len(all_props) * len(potentials)}")
    print()

    # Générer tous les graphiques
    total = len(all_props) * len(potentials)
    count = 0

    for property_col in all_props:
        for potential_col in potentials:
            count += 1

            # Nom de fichier (format harmonisé selon le type de propriété)
            if property_col.endswith('_bader'):
                property_name = property_col.replace('_bader', '_Bader')
            elif property_col.endswith('_iqa'):
                property_name = property_col.replace('_iqa', '_IQA')
            elif property_col.startswith('delta_'):
                property_name = property_col  # delta_C1_red, delta_O7_ox, etc.
            else:
                property_name = property_col  # hardness_eV, etc.

            potential_name = potential_col.replace('E_', '').replace('_V', '')
            filename = f"{property_name}_vs_{potential_name}.png"

            output_file = single_dir / filename

            # Titre explicite selon le type de propriété
            if property_col.endswith('_bader'):
                atom_label = property_col.replace('_bader', '')
                prop_label = f"Bader Charge ({atom_label})"
            elif property_col.endswith('_iqa'):
                atom_label = property_col.replace('_iqa', '')
                prop_label = f"IQA Energy ({atom_label})"
            elif property_col.startswith('delta_'):
                # delta_C1_red → ΔBader C1 (Red), delta_O7_ox → ΔBader O7 (Ox)
                parts = property_col.replace('delta_', '').rsplit('_', 1)
                atom_label = parts[0] if len(parts) > 0 else property_col
                process = parts[1] if len(parts) > 1 else ''
                process_label = "Red" if process == "red" else "Ox" if process == "ox" else process
                prop_label = f"ΔBader {atom_label} ({process_label})"
            else:
                prop_label = property_col

            if potential_col == 'E_reduction_V':
                pot_label = "Reduction Potential (V)"
            elif potential_col == 'E_oxidation_V':
                pot_label = "Oxidation Potential (V)"
            else:
                pot_label = potential_col

            title = f"{prop_label} vs {pot_label}"

            # Générer le graphique
            r2, slope = plot_property_vs_potential(df, property_col, potential_col, output_file, title)

            results.append((property_col, potential_col, r2, slope, output_file))

            # Progress
            if count % 10 == 0 or count == total:
                print(f"  Progression: {count}/{total} graphiques générés")

    print(f"\n✓ {len(results)} graphiques de corrélation générés")

    if save_individual:
        print(f"  → Graphiques individuels: {single_dir}/")

    return results


def create_correlation_grid(
    df: pd.DataFrame,
    output_dir: Path,
    grid_file: str = "all_correlations_grid.png"
) -> None:
    """
    Crée une grille rectangulaire avec tous les graphiques de corrélation.

    Layout : 3 colonnes pour réduction + espace + 3 colonnes pour oxydation = 6 colonnes effectives
    Les propriétés sont distribuées en lignes (3 propriétés par ligne)

    Args:
        df: DataFrame avec toutes les données
        output_dir: Répertoire de sortie
        grid_file: Nom du fichier de grille

    Note:
        Génère une grande image avec tous les graphiques assemblés en format rectangulaire.
    """
    output_dir = Path(output_dir)

    # Identifier les propriétés (format harmonisé: C1_bader, C1_iqa, etc.)
    bader_cols = [col for col in df.columns if col.endswith('_bader')]
    iqa_cols = [col for col in df.columns if col.endswith('_iqa')]

    # Δ-descripteurs de transfert de charge
    delta_cols = [col for col in df.columns if col.startswith('delta_')]

    all_properties = bader_cols + iqa_cols + delta_cols
    n_props = len(all_properties)

    # Configuration de la grille : 3 propriétés par ligne
    props_per_row = 4
    n_rows = int(np.ceil(n_props / props_per_row))

    # Colonnes : 3 pour réduction + 1 espace + 3 pour oxydation = 7 colonnes dans gridspec
    n_cols = props_per_row * 2 + 1  # 3 red + 1 space + 3 ox = 7

    # Taille de chaque subplot
    subplot_width = 3.5
    subplot_height = 2.5

    fig_width = (props_per_row * 2) * subplot_width + 1  # +1 pour l'espace
    fig_height = n_rows * subplot_height

    print("\n" + "-"*80)
    print(f"CRÉATION DE LA GRILLE COMPLÈTE (FORMAT RECTANGULAIRE)")
    print("-"*80)
    print(f"Dimensions: {n_rows} lignes x 8 colonnes effectives ({n_props} propriétés)")
    print(f"Layout: 4 colonnes réduction | espace | 4 colonnes oxydation")
    print(f"Taille figure: {fig_width:.1f} x {fig_height:.1f} inches")
    print()

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25,
                          width_ratios=[1, 1, 1, 1, 0.1, 1, 1, 1, 1])  # Colonne 3 = espace étroit

    from scipy.stats import linregress

    potentials = ['E_reduction_V', 'E_oxidation_V']

    for idx, property_col in enumerate(all_properties):
        # Position dans la grille
        row = idx // props_per_row
        col_offset = idx % props_per_row

        for pot_idx, potential_col in enumerate(potentials):
            # Colonne dans gridspec :
            # - Réduction : colonnes 0, 1, 2
            # - Oxydation : colonnes 4, 5, 6 (colonne 3 = espace)
            if pot_idx == 0:  # Réduction
                col = col_offset
            else:  # Oxydation
                col = col_offset + 5  # +4 car colonne 3 est l'espace

            ax = fig.add_subplot(gs[row, col])

            # Supprimer les NaN
            df_clean = df[[property_col, potential_col, 'molecule_name']].dropna()

            if len(df_clean) < 2:
                ax.text(0.5, 0.5, 'Données\ninsuffisantes', ha='center', va='center',
                       transform=ax.transAxes, fontsize=8)
                ax.set_xlabel(property_col, fontsize=6)
                ax.set_ylabel(potential_col, fontsize=6)
                ax.tick_params(labelsize=5)
                continue

            # Scatter
            ax.scatter(df_clean[property_col], df_clean[potential_col], s=40, alpha=0.7, color='steelblue')

            # Labels (petits pour la grille)
            for _, row_data in df_clean.iterrows():
                ax.annotate(
                    row_data['molecule_name'],
                    (row_data[property_col], row_data[potential_col]),
                    xytext=(2, 2), textcoords='offset points',
                    fontsize=5, alpha=0.7
                )

            # Régression
            slope, intercept, r_value, _, _ = linregress(df_clean[property_col], df_clean[potential_col])
            x_fit = np.linspace(df_clean[property_col].min(), df_clean[property_col].max(), 50)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r--', linewidth=1.2, alpha=0.8)

            # Titre compact (format harmonisé selon type de propriété)
            if property_col.endswith('_bader'):
                atom_label = property_col.replace('_bader', '')
                title = f"Bader {atom_label}"
            elif property_col.endswith('_iqa'):
                atom_label = property_col.replace('_iqa', '')
                title = f"IQA {atom_label}"
            elif property_col.startswith('delta_'):
                # delta_C1_red → Δ C1 (Red)
                parts = property_col.replace('delta_', '').rsplit('_', 1)
                atom_label = parts[0] if len(parts) > 0 else property_col
                process = parts[1] if len(parts) > 1 else ''
                proc_label = "R" if process == "red" else "O" if process == "ox" else process
                title = f"Δ {atom_label} ({proc_label})"
            else:
                title = property_col[:12]  # Tronquer si trop long

            pot_label = "Red" if potential_col == 'E_reduction_V' else "Ox"
            y_pred_on_data = slope * df_clean[property_col].values + intercept
            mae = np.mean(np.abs(df_clean[potential_col].values - y_pred_on_data))
            ax.set_title(f"{title} (MAE={mae:.2f}, R²={r_value**2:.2f})", fontsize=7, fontweight='bold')

            # Labels avec tailles plus petites
            ax.set_xlabel(property_col if len(property_col) < 18 else property_col[:15]+'...', fontsize=6)
            ax.set_ylabel(pot_label, fontsize=6)
            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.3, linewidth=0.5)

    # Ajouter des titres de colonnes
    fig.text(0.19, 0.98, 'RÉDUCTION', ha='center', fontsize=14, fontweight='bold', color='darkred')
    fig.text(0.75, 0.98, 'OXYDATION', ha='center', fontsize=14, fontweight='bold', color='darkblue')

    plt.suptitle('Corrélations Complètes: Propriétés QTAIM/IQA vs Potentiels Redox',
                 fontsize=16, fontweight='bold', y=0.995)

    output_file = output_dir / grid_file
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Grille complète sauvegardée: {grid_file}")
    print(f"  Taille: {output_file.stat().st_size / (1024*1024):.1f} MB")


def compute_r2_matrix(
    df: pd.DataFrame,
    row_cols: List[str],
    col_cols: List[str]
) -> np.ndarray:
    """
    Calcule une matrice R² (corrélation au carré) entre deux ensembles de colonnes.

    Args:
        df: DataFrame avec les données
        row_cols: Liste des colonnes pour les lignes de la matrice
        col_cols: Liste des colonnes pour les colonnes de la matrice

    Returns:
        Matrice R² de dimensions (len(row_cols), len(col_cols))
    """
    from scipy.stats import pearsonr

    n_rows, n_cols = len(row_cols), len(col_cols)
    r2_matrix = np.full((n_rows, n_cols), np.nan)

    for i, row_col in enumerate(row_cols):
        for j, col_col in enumerate(col_cols):
            if row_col not in df.columns or col_col not in df.columns:
                continue
            # Diagonal: correlation with itself is always 1.0
            if row_col == col_col:
                r2_matrix[i, j] = 1.0
                continue
            valid = df[[row_col, col_col]].dropna()
            if len(valid) >= 2:
                r, _ = pearsonr(valid[row_col], valid[col_col])
                r2_matrix[i, j] = float(r) ** 2

    return r2_matrix


def plot_delta_bader_vs_iqa_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    process: str
) -> None:
    """
    Génère une heatmap 8×8 de R² entre ΔBader et ΔIQA pour un processus redox.

    Args:
        df: DataFrame avec les données
        output_dir: Répertoire de sortie
        process: "red" (réduction) ou "ox" (oxydation)

    Exports:
        - delta_bader_vs_iqa_{process}_heatmap.png
    """
    output_dir = Path(output_dir)

    # Labels des 8 atomes du squelette catéchol
    atom_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'O7', 'O8']

    # Colonnes ΔBader (lignes de la heatmap)
    bader_cols = [f'delta_{a}_{process}' for a in atom_labels]

    # Colonnes ΔIQA (colonnes de la heatmap)
    iqa_cols = [f'delta_{a}_iqa_{process}' for a in atom_labels]

    # Vérifier que les colonnes existent
    bader_present = [col for col in bader_cols if col in df.columns]
    iqa_present = [col for col in iqa_cols if col in df.columns]

    if len(bader_present) < 2 or len(iqa_present) < 2:
        warnings.warn(f"Pas assez de colonnes pour la heatmap ΔBader vs ΔIQA ({process})")
        return

    # Calculer la matrice R²
    r2_matrix = compute_r2_matrix(df, bader_cols, iqa_cols)

    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    try:
        import seaborn as sns
        sns.heatmap(
            r2_matrix,
            annot=True,
            fmt='.2f',
            cmap='Reds',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=[f'Δ{a}_IQA' for a in atom_labels],
            yticklabels=[f'Δ{a}_Bader' for a in atom_labels],
            cbar_kws={"shrink": 0.8, "label": "R²"},
            ax=ax
        )
    except ImportError:
        # Fallback matplotlib pur
        im = ax.imshow(r2_matrix, cmap='Reds', vmin=0, vmax=1, aspect='equal')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('R²', rotation=270, labelpad=15)

        # Labels
        ax.set_xticks(np.arange(len(atom_labels)))
        ax.set_yticks(np.arange(len(atom_labels)))
        ax.set_xticklabels([f'Δ{a}_IQA' for a in atom_labels], rotation=45, ha='right')
        ax.set_yticklabels([f'Δ{a}_Bader' for a in atom_labels])

        # Annotations
        for i in range(len(atom_labels)):
            for j in range(len(atom_labels)):
                if not np.isnan(r2_matrix[i, j]):
                    text_color = 'white' if r2_matrix[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{r2_matrix[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=9)

    # Titre
    process_name = "Réduction" if process == "red" else "Oxydation"
    ax.set_title(f'Corrélations R² : ΔBader vs ΔIQA ({process_name})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('ΔIQA (Hartree)', fontsize=11)
    ax.set_ylabel('ΔBader (e)', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / f"delta_bader_vs_iqa_{process}_heatmap.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Heatmap ΔBader vs ΔIQA ({process_name}) sauvegardée: {output_file.name}")


def plot_redox_decomposition_heatmap(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Génère une heatmap 8×8 de R² pour les termes de décomposition énergétique redox + orbitales frontières.

    Args:
        df: DataFrame avec les données
        output_dir: Répertoire de sortie

    Exports:
        - redox_decomposition_heatmap.png
    """
    output_dir = Path(output_dir)

    # Colonnes de décomposition énergétique + orbitales frontières
    decomp_cols = [
        'HOMO_eV', 'LUMO_eV',
        'dG_reduction_kJ', 'EA_kJ', 'Edef_red_kJ',
        'dG_oxidation_kJ', 'IP_kJ', 'Edef_ox_kJ'
    ]

    # Labels plus lisibles
    decomp_labels = [
        'HOMO', 'LUMO',
        'ΔG_red', 'EA', 'E_def(red)',
        'ΔG_ox', 'IP', 'E_def(ox)'
    ]

    # Vérifier que les colonnes existent
    present_cols = [col for col in decomp_cols if col in df.columns]
    if len(present_cols) < 2:
        warnings.warn("Pas assez de colonnes pour la heatmap de décomposition redox")
        return

    # Calculer la matrice R²
    r2_matrix = compute_r2_matrix(df, decomp_cols, decomp_cols)

    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    try:
        import seaborn as sns
        sns.heatmap(
            r2_matrix,
            annot=True,
            fmt='.2f',
            cmap='Reds',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=decomp_labels,
            yticklabels=decomp_labels,
            cbar_kws={"shrink": 0.8, "label": "R²"},
            ax=ax
        )
    except ImportError:
        # Fallback matplotlib pur
        im = ax.imshow(r2_matrix, cmap='Reds', vmin=0, vmax=1, aspect='equal')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('R²', rotation=270, labelpad=15)

        # Labels
        ax.set_xticks(np.arange(len(decomp_labels)))
        ax.set_yticks(np.arange(len(decomp_labels)))
        ax.set_xticklabels(decomp_labels, rotation=45, ha='right')
        ax.set_yticklabels(decomp_labels)

        # Annotations
        for i in range(len(decomp_labels)):
            for j in range(len(decomp_labels)):
                if not np.isnan(r2_matrix[i, j]):
                    text_color = 'white' if r2_matrix[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{r2_matrix[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=9)

    # Titre
    ax.set_title('Corrélations R² : Décomposition Énergétique Redox',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    output_file = output_dir / "redox_decomposition_heatmap.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Heatmap décomposition redox sauvegardée: {output_file.name}")


def plot_cross_correlations(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Génère une grille de corrélations croisées structure-propriété.

    Corrélations analysées :
    - LUMO vs ΔBader_O7_red : Où va l'électron lors de la réduction ?
    - HOMO vs ΔBader_O8_ox : D'où part l'électron lors de l'oxydation ?
    - EA_kJ vs delta_iqa_O_total_red : Affinité électronique vs stabilisation IQA des O
    - IP_kJ vs gap_eV : Relation ionisation/gap
    - q_O_total vs E_reduction_V : Charge O vs potentiel de réduction
    - q_O_asymmetry vs E_oxidation_V : Asymétrie vs potentiel d'oxydation

    Args:
        df: DataFrame avec les données
        output_dir: Répertoire de sortie

    Exports:
        - cross_correlations_grid.png
    """
    from scipy.stats import linregress

    output_dir = Path(output_dir)

    # Définir les paires de corrélations croisées
    cross_pairs = [
        # (x_col, y_col, x_label, y_label, title)
        ('LUMO_eV', 'delta_O7_red', 'LUMO (eV)', 'ΔBader O7 (red)',
         'LUMO vs ΔBader O7 (réduction)'),
        ('HOMO_eV', 'delta_O8_ox', 'HOMO (eV)', 'ΔBader O8 (ox)',
         'HOMO vs ΔBader O8 (oxydation)'),
        ('EA_kJ', 'delta_iqa_O_total_red', 'EA (kJ/mol)', 'ΔIQA O total (red)',
         'EA vs ΔIQA O total (réduction)'),
        ('IP_kJ', 'gap_eV', 'IP (kJ/mol)', 'Gap HOMO-LUMO (eV)',
         'IP vs Gap'),
        ('q_O_total', 'E_reduction_V', 'q(O7+O8) (e)', 'E_red (V)',
         'Charge O totale vs E_réduction'),
        ('q_O_asymmetry', 'E_oxidation_V', '|q(O7)-q(O8)| (e)', 'E_ox (V)',
         'Asymétrie charge O vs E_oxydation'),
        ('delta_q_O_total_red', 'dG_reduction_kJ', 'Δq_O total (red)', 'ΔG_red (kJ/mol)',
         'Δ charge O vs ΔG réduction'),
        ('delta_q_O_total_ox', 'dG_oxidation_kJ', 'Δq_O total (ox)', 'ΔG_ox (kJ/mol)',
         'Δ charge O vs ΔG oxydation'),
    ]

    # Filtrer les paires disponibles
    available_pairs = []
    for x_col, y_col, x_label, y_label, title in cross_pairs:
        if x_col in df.columns and y_col in df.columns:
            valid = df[[x_col, y_col]].dropna()
            if len(valid) >= 3:
                available_pairs.append((x_col, y_col, x_label, y_label, title))

    if not available_pairs:
        warnings.warn("Pas assez de données pour les corrélations croisées")
        return

    # Créer la grille
    n_pairs = len(available_pairs)
    n_cols = min(4, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (x_col, y_col, x_label, y_label, title) in enumerate(available_pairs):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Données valides
        valid = df[['molecule_name', x_col, y_col]].dropna()

        # Scatter
        ax.scatter(valid[x_col], valid[y_col], s=80, alpha=0.7, color='steelblue')

        # Labels des molécules
        for _, r in valid.iterrows():
            ax.annotate(
                r['molecule_name'], (r[x_col], r[y_col]),
                xytext=(3, 3), textcoords='offset points',
                fontsize=7, alpha=0.7
            )

        # Régression
        slope, intercept, r_value, _, _ = linregress(valid[x_col], valid[y_col])
        x_fit = np.linspace(valid[x_col].min(), valid[x_col].max(), 50)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r--', linewidth=1.5, alpha=0.8)

        # Titre avec MAE et R²
        y_pred_on_data = slope * valid[x_col].values + intercept
        mae = np.mean(np.abs(valid[y_col].values - y_pred_on_data))
        ax.set_title(f'{title}\nMAE = {mae:.1f}, R² = {r_value**2:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Masquer les axes vides
    for idx in range(len(available_pairs), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Corrélations Croisées Structure-Propriété',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = output_dir / "cross_correlations_grid.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Corrélations croisées sauvegardées: {output_file.name}")
    print(f"  {len(available_pairs)} corrélations analysées")


def generate_batch_plots(
    df: pd.DataFrame,
    output_dir: Path,
    interactive: bool = False
) -> None:
    """
    Génère tous les plots de comparaison inter-molécules.

    Args:
        df: DataFrame avec 1 ligne par molécule (colonnes : molecule_name, propriétés)
        output_dir: Répertoire de sortie pour les graphiques
        interactive: Ignoré (pas de plotly en mode batch)

    Exports:
        - all_correlations_grid.png (grille complète rectangulaire)
        - single_graphs/ (tous les graphiques individuels + orbitales frontières)
        - batch_correlation_heatmap.png (matrice de corrélation)
        - molecules_summary.csv (données consolidées)

    Example:
        >>> from visualization.batch_correlations import generate_batch_plots
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'molecule_name': ['DHI', 'O-DHI'],
        ...     'LUMO_eV': [-0.017, -0.024],
        ...     'E_reduction_V': [-1.208, -0.913],
        ...     # ... autres colonnes
        ... })
        >>> generate_batch_plots(df, Path("output"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print(f"GÉNÉRATION DE VISUALISATIONS BATCH ({len(df)} molécules)")
    print("="*80 + "\n")

    print(f"Molécules: {', '.join(df['molecule_name'].tolist())}\n")

    print("Génération des plots...\n")

    # 1. Heatmap de corrélation
    try:
        plot_correlation_heatmap_batch(df, output_dir)
    except Exception as e:
        print(f"✗ Erreur Heatmap: {e}")

    # 2. TOUS LES GRAPHIQUES DE CORRÉLATION (charges Bader + IQA vs potentiels)
    try:
        generate_all_correlations(df, output_dir, save_individual=True)
    except Exception as e:
        print(f"✗ Erreur corrélations complètes: {e}")

    # 3. CORRÉLATIONS ORBITALES FRONTIÈRES (HOMO vs E_ox, LUMO vs E_red)
    try:
        single_dir = output_dir / "single_graphs"
        single_dir.mkdir(parents=True, exist_ok=True)
        generate_frontier_orbital_correlations(df, single_dir)
    except Exception as e:
        print(f"✗ Erreur corrélations orbitales frontières: {e}")

    # 4. GRILLE COMPLÈTE RECTANGULAIRE (tous les graphiques assemblés)
    try:
        create_correlation_grid(df, output_dir)
    except Exception as e:
        print(f"✗ Erreur grille complète: {e}")

    # 5. HEATMAP ΔBader vs ΔIQA (RÉDUCTION)
    try:
        plot_delta_bader_vs_iqa_heatmap(df, output_dir, "red")
    except Exception as e:
        print(f"⚠ Erreur heatmap ΔBader vs ΔIQA réduction: {e}")

    # 6. HEATMAP ΔBader vs ΔIQA (OXYDATION)
    try:
        plot_delta_bader_vs_iqa_heatmap(df, output_dir, "ox")
    except Exception as e:
        print(f"⚠ Erreur heatmap ΔBader vs ΔIQA oxydation: {e}")

    # 7. HEATMAP DÉCOMPOSITION REDOX
    try:
        plot_redox_decomposition_heatmap(df, output_dir)
    except Exception as e:
        print(f"⚠ Erreur heatmap décomposition redox: {e}")

    # 8. CORRÉLATIONS CROISÉES STRUCTURE-PROPRIÉTÉ
    try:
        plot_cross_correlations(df, output_dir)
    except Exception as e:
        print(f"⚠ Erreur corrélations croisées: {e}")

    # Exporter le DataFrame consolidé
    try:
        csv_path = output_dir / "molecules_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Données consolidées: molecules_summary.csv")
    except Exception as e:
        print(f"\n✗ Erreur export CSV: {e}")

    print("\n" + "="*80)
    print("✓ VISUALISATIONS BATCH GÉNÉRÉES AVEC SUCCÈS")
    print("="*80)

