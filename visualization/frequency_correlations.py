"""
Visualisation des corrélations fréquence vs propriétés thermodynamiques.

Ce module génère des graphiques pour visualiser les corrélations entre
les descripteurs de fréquences vibrationnelles et les cibles thermodynamiques
(Edef, ΔΔU, TΔS).
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def plot_top_correlations(
    df_corr,
    df_desc,
    output_dir: Path,
    top_n: int = 10
) -> None:
    """
    Génère les graphiques des top N corrélations pour réduction et oxydation.

    Supporte les corrélations Edef classiques (depuis compute_correlations)
    et les corrélations focalisées ΔΔU/TΔS (depuis compute_focused_correlations).

    Args:
        df_corr: DataFrame des corrélations (depuis compute_correlations)
        df_desc: DataFrame des descripteurs par molécule
        output_dir (Path): Répertoire de sortie
        top_n (int): Nombre de top corrélations à afficher

    Exports:
        - top_correlations_reduction.png
        - top_correlations_oxidation.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PANDAS:
        print("  ⚠ pandas requis pour plot_top_correlations")
        return

    # Convertir en DataFrame si nécessaire
    if isinstance(df_corr, list):
        df_corr = pd.DataFrame(df_corr)
    if isinstance(df_desc, dict):
        df_desc = pd.DataFrame(df_desc)

    # Top corrélations pour RÉDUCTION
    _plot_top_for_target(
        df_corr, df_desc,
        target_col='Edef_red_kJ',
        r2_col='r2_reduction',
        output_file=output_dir / "top_correlations_reduction.png",
        title="Top Corrélations: Descripteurs de Fréquence vs Edef (Réduction)",
        top_n=top_n
    )

    # Top corrélations pour OXYDATION
    _plot_top_for_target(
        df_corr, df_desc,
        target_col='Edef_ox_kJ',
        r2_col='r2_oxidation',
        output_file=output_dir / "top_correlations_oxidation.png",
        title="Top Corrélations: Descripteurs de Fréquence vs Edef (Oxydation)",
        top_n=top_n
    )


def plot_top_focused_correlations(
    df_focused,
    df_desc,
    df_thermo,
    output_dir: Path,
    top_n: int = 10
) -> None:
    """
    Génère les grilles de top corrélations focalisées pour ΔΔU et TΔS.

    Args:
        df_focused: DataFrame des corrélations focalisées (depuis compute_focused_correlations)
        df_desc: DataFrame des descripteurs par molécule
        df_thermo: DataFrame des cibles thermodynamiques par molécule
        output_dir (Path): Répertoire de sortie
        top_n (int): Nombre de top corrélations à afficher

    Exports:
        - top_correlations_ddU_reduction.png
        - top_correlations_ddU_oxidation.png
        - top_correlations_TdS_reduction.png
        - top_correlations_TdS_oxidation.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PANDAS:
        print("  ⚠ pandas requis pour plot_top_focused_correlations")
        return

    if isinstance(df_focused, list):
        df_focused = pd.DataFrame(df_focused)
    if isinstance(df_desc, dict):
        df_desc = pd.DataFrame(df_desc)
    if isinstance(df_thermo, dict):
        df_thermo = pd.DataFrame(df_thermo)

    if df_focused.empty:
        print("  ⚠ Aucune corrélation focalisée à tracer")
        return

    # Fusionner descripteurs et cibles pour avoir toutes les colonnes
    if 'molecule' in df_desc.columns and 'molecule' in df_thermo.columns:
        df_merged = pd.merge(df_desc, df_thermo, on='molecule', how='outer',
                           suffixes=('', '_thermo'))
    else:
        df_merged = df_desc

    targets = [
        ('ddU_red_kJ', 'ΔΔU (Réduction)', 'top_correlations_ddU_reduction.png'),
        ('ddU_ox_kJ', 'ΔΔU (Oxydation)', 'top_correlations_ddU_oxidation.png'),
        ('TdS_red_kJ', 'TΔS (Réduction)', 'top_correlations_TdS_reduction.png'),
        ('TdS_ox_kJ', 'TΔS (Oxydation)', 'top_correlations_TdS_oxidation.png'),
    ]

    for target_col, target_label, filename in targets:
        target_rows = df_focused[df_focused['target'] == target_col]
        if target_rows.empty:
            continue

        _plot_top_focused_grid(
            target_rows.head(top_n),
            df_merged,
            target_col=target_col,
            output_file=output_dir / filename,
            title=f"Top Corrélations: Descripteurs vs {target_label}",
        )


def _plot_top_for_target(
    df_corr,
    df_desc,
    target_col: str,
    r2_col: str,
    output_file: Path,
    title: str,
    top_n: int
) -> None:
    """
    Génère une grille de graphiques pour les top corrélations avec une cible.

    Args:
        df_corr: DataFrame des corrélations
        df_desc: DataFrame des descripteurs
        target_col: Colonne cible (Edef_red_kJ ou Edef_ox_kJ)
        r2_col: Colonne R² correspondante
        output_file: Fichier de sortie
        title: Titre principal
        top_n: Nombre de corrélations à afficher
    """
    from scipy.stats import linregress

    # Trier par R² et prendre les top N
    df_sorted = df_corr.sort_values(r2_col, ascending=False).head(top_n)

    # Nombre de lignes et colonnes pour la grille
    n_cols = min(3, len(df_sorted))
    n_rows = (len(df_sorted) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        ax_row, ax_col = idx // n_cols, idx % n_cols
        ax = axes[ax_row, ax_col]

        desc_name = row['descriptor']
        r2 = row[r2_col]

        # Extraire les données
        if desc_name not in df_desc.columns or target_col not in df_desc.columns:
            ax.text(0.5, 0.5, 'Données\nmanquantes', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        x = df_desc[desc_name].values
        y = df_desc[target_col].values
        molecules = df_desc['molecule'].values if 'molecule' in df_desc.columns else None

        # Filtrer les NaN
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 2:
            ax.text(0.5, 0.5, 'Données\ninsuffisantes', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Scatter plot
        ax.scatter(x_valid, y_valid, s=80, alpha=0.7, color='steelblue')

        # Labels des molécules
        if molecules is not None:
            mol_valid = molecules[mask]
            for i, mol in enumerate(mol_valid):
                ax.annotate(
                    mol, (x_valid[i], y_valid[i]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=7, alpha=0.7
                )

        # Régression linéaire
        slope, intercept, r_value, _, _ = linregress(x_valid, y_valid)
        x_fit = np.linspace(x_valid.min(), x_valid.max(), 50)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r--', linewidth=1.5, alpha=0.8)

        # Titre avec MAE et R²
        y_pred_on_data = slope * x_valid + intercept
        mae = np.mean(np.abs(y_valid - y_pred_on_data))
        ax.set_title(f'{desc_name}\nMAE = {mae:.1f} kJ/mol, R² = {r2:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel(desc_name, fontsize=9)
        ax.set_ylabel(target_col, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Masquer les axes vides
    for idx in range(len(df_sorted), n_rows * n_cols):
        ax_row, ax_col = idx // n_cols, idx % n_cols
        axes[ax_row, ax_col].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ {output_file.name} sauvegardé")


def _plot_top_focused_grid(
    df_rows,
    df_data,
    target_col: str,
    output_file: Path,
    title: str
) -> None:
    """
    Génère une grille de scatter plots pour les corrélations focalisées.

    Args:
        df_rows: DataFrame subset avec les corrélations à tracer (contient 'descriptor', 'r2', etc.)
        df_data: DataFrame avec descripteurs et cibles par molécule
        target_col: Colonne cible dans df_data
        output_file: Fichier de sortie
        title: Titre principal
    """
    from scipy.stats import linregress

    n_plots = len(df_rows)
    if n_plots == 0:
        return

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (_, row) in enumerate(df_rows.iterrows()):
        ax_row, ax_col = idx // n_cols, idx % n_cols
        ax = axes[ax_row, ax_col]

        desc_name = row['descriptor']
        r2 = row.get('r2', np.nan)
        pearson_p = row.get('pearson_p', np.nan)
        spearman_rho = row.get('spearman_rho', np.nan)

        if desc_name not in df_data.columns or target_col not in df_data.columns:
            ax.text(0.5, 0.5, 'Données\nmanquantes', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        x = df_data[desc_name].values.astype(float)
        y = df_data[target_col].values.astype(float)
        molecules = df_data['molecule'].values if 'molecule' in df_data.columns else None

        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 2:
            ax.text(0.5, 0.5, 'Données\ninsuffisantes', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        ax.scatter(x_valid, y_valid, s=80, alpha=0.7, color='steelblue')

        if molecules is not None:
            mol_valid = molecules[mask]
            for i, mol in enumerate(mol_valid):
                ax.annotate(mol, (x_valid[i], y_valid[i]),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=7, alpha=0.7)

        slope, intercept, _, _, _ = linregress(x_valid, y_valid)
        x_fit = np.linspace(x_valid.min(), x_valid.max(), 50)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r--', linewidth=1.5, alpha=0.8)

        # Titre avec MAE, R² et Spearman
        y_pred_on_data = slope * x_valid + intercept
        mae = np.mean(np.abs(y_valid - y_pred_on_data))
        subtitle = f'MAE={mae:.1f}, R²={r2:.3f}'
        if not np.isnan(spearman_rho):
            subtitle += f', ρ={spearman_rho:.3f}'
        ax.set_title(f'{desc_name}\n{subtitle}', fontsize=10, fontweight='bold')
        ax.set_xlabel(desc_name, fontsize=9)
        ax.set_ylabel(target_col, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Masquer les axes vides
    for idx in range(n_plots, n_rows * n_cols):
        ax_row, ax_col = idx // n_cols, idx % n_cols
        axes[ax_row, ax_col].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ {output_file.name} sauvegardé")


def plot_focused_correlation(
    x: np.ndarray,
    y: np.ndarray,
    stats: Dict[str, float],
    desc_name: str,
    target_label: str,
    output_file: Path,
    molecules: Optional[np.ndarray] = None,
    color: str = 'steelblue'
) -> None:
    """
    Scatter plot individuel avec annotation statistique complète.

    Affiche toutes les statistiques de corrélation dans une boîte de texte :
    Pearson r, Spearman rho, Kendall tau, R², R² ajusté, LOOCV R²,
    Bootstrap 95% CI, Shapiro-Wilk.

    Args:
        x: Valeurs du descripteur (sans NaN)
        y: Valeurs cibles (sans NaN)
        stats: Dict des statistiques (depuis _compute_single_correlation_stats)
        desc_name: Nom du descripteur
        target_label: Label de la cible pour le titre
        output_file: Fichier de sortie PNG
        molecules: Noms des molécules pour annotations (optionnel)
        color: Couleur de la droite de régression
    """
    from scipy.stats import linregress

    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter plot
    ax.scatter(x, y, s=100, alpha=0.7, color='steelblue', zorder=3)

    # Labels des molécules
    if molecules is not None:
        for i, mol in enumerate(molecules):
            ax.annotate(mol, (x[i], y[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)

    # Régression linéaire
    slope = stats.get('slope', 0)
    intercept = stats.get('intercept', 0)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, '--', color=color, linewidth=2, zorder=2)

    # Boîte de texte avec toutes les stats
    r2_boot_lo = stats.get('r2_bootstrap_lo', np.nan)
    r2_boot_hi = stats.get('r2_bootstrap_hi', np.nan)
    shapiro_p = stats.get('shapiro_p', np.nan)

    y_pred_on_data = slope * x + intercept
    mae = np.mean(np.abs(y - y_pred_on_data))
    lines = [
        f"MAE = {mae:.1f} kJ/mol",
        f"Pearson r = {stats.get('pearson_r', np.nan):.3f}, p = {stats.get('pearson_p', np.nan):.2e}",
        f"Spearman ρ = {stats.get('spearman_rho', np.nan):.3f}, p = {stats.get('spearman_p', np.nan):.2e}",
        f"Kendall τ = {stats.get('kendall_tau', np.nan):.3f}, p = {stats.get('kendall_p', np.nan):.2e}",
        f"R² = {stats.get('r2', np.nan):.3f}, adj-R² = {stats.get('r2_adj', np.nan):.3f}",
        f"LOOCV R² = {stats.get('r2_loocv', np.nan):.3f}",
    ]

    if not np.isnan(r2_boot_lo) and not np.isnan(r2_boot_hi):
        lines.append(f"Bootstrap 95% CI: [{r2_boot_lo:.2f}, {r2_boot_hi:.2f}]")

    if not np.isnan(shapiro_p):
        normality = "normal" if shapiro_p > 0.05 else "non-normal"
        lines.append(f"Shapiro-Wilk p = {shapiro_p:.3f} ({normality})")

    lines.append(f"n = {stats.get('n', len(x))}")

    textstr = '\n'.join(lines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace', bbox=props)

    ax.set_title(f'{desc_name} vs {target_label}', fontsize=13, fontweight='bold')
    ax.set_xlabel(desc_name, fontsize=11)
    ax.set_ylabel(f'{target_label} (kJ/mol)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_individual_correlations(
    df_desc,
    output_dir: Path,
    descriptors: Optional[List[str]] = None,
    df_thermo=None
) -> List[Path]:
    """
    Génère des graphiques individuels pour chaque corrélation.

    Trace les corrélations pour les 4 cibles thermodynamiques (ΔΔU et TΔS)
    ainsi que les cibles Edef classiques si disponibles.

    Args:
        df_desc: DataFrame des descripteurs par molécule
        output_dir (Path): Répertoire de sortie (single_graphs/ sera créé)
        descriptors (List[str], optional): Liste des descripteurs à tracer.
                                          Si None, utilise tous les descripteurs numériques.
        df_thermo: DataFrame des cibles thermodynamiques (optionnel).
                   Si fourni, trace ΔΔU/TΔS au lieu de Edef.

    Returns:
        List[Path]: Liste des fichiers générés

    Exports:
        - single_graphs/<descriptor>_vs_<target>.png
    """
    from scipy.stats import linregress

    output_dir = Path(output_dir) / "single_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PANDAS:
        print("  ⚠ pandas requis pour plot_individual_correlations")
        return []

    # Convertir en DataFrame si nécessaire
    if isinstance(df_desc, dict):
        df_desc = pd.DataFrame(df_desc)

    # Fusionner avec données thermo si disponibles
    if df_thermo is not None:
        if isinstance(df_thermo, dict):
            df_thermo = pd.DataFrame(df_thermo)
        if 'molecule' in df_desc.columns and 'molecule' in df_thermo.columns:
            df_merged = pd.merge(df_desc, df_thermo, on='molecule', how='outer',
                               suffixes=('', '_thermo'))
        else:
            df_merged = df_desc
    else:
        df_merged = df_desc

    # Déterminer les descripteurs à tracer
    if descriptors is None:
        exclude = ['molecule', 'Edef_red_kJ', 'Edef_ox_kJ',
                   'ddU_red_kJ', 'ddU_ox_kJ', 'TdS_red_kJ', 'TdS_ox_kJ']
        descriptors = [col for col in df_merged.columns
                      if col not in exclude and df_merged[col].dtype in ['float64', 'int64']]

    generated_files = []

    # Définir les cibles
    targets = [
        ('ddU_red_kJ', 'ΔΔU (Réduction)', 'darkgreen'),
        ('ddU_ox_kJ', 'ΔΔU (Oxydation)', 'darkblue'),
        ('TdS_red_kJ', 'TΔS (Réduction)', 'darkorange'),
        ('TdS_ox_kJ', 'TΔS (Oxydation)', 'darkred'),
    ]

    # Fallback sur Edef si pas de données thermo
    available_targets = [(col, label, color) for col, label, color in targets
                        if col in df_merged.columns]
    if not available_targets:
        available_targets = [
            ('Edef_red_kJ', 'Edef (Réduction)', 'darkred'),
            ('Edef_ox_kJ', 'Edef (Oxydation)', 'darkblue'),
        ]

    for desc_name in descriptors:
        if desc_name not in df_merged.columns:
            continue

        for target_col, target_label, color in available_targets:
            if target_col not in df_merged.columns:
                continue

            x = df_merged[desc_name].values.astype(float)
            y = df_merged[target_col].values.astype(float)
            molecules = df_merged['molecule'].values if 'molecule' in df_merged.columns else None

            # Filtrer les NaN
            mask = ~np.isnan(x) & ~np.isnan(y)
            x_valid = x[mask]
            y_valid = y[mask]

            if len(x_valid) < 2:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))

            # Scatter plot
            ax.scatter(x_valid, y_valid, s=100, alpha=0.7, color='steelblue')

            # Labels des molécules
            if molecules is not None:
                mol_valid = molecules[mask]
                for i, mol in enumerate(mol_valid):
                    ax.annotate(
                        mol, (x_valid[i], y_valid[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8
                    )

            # Régression linéaire
            slope, intercept, r_value, p_value, _ = linregress(x_valid, y_valid)
            x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
            y_fit = slope * x_fit + intercept
            y_pred_on_data = slope * x_valid + intercept
            mae = np.mean(np.abs(y_valid - y_pred_on_data))
            ax.plot(x_fit, y_fit, '--', color=color, linewidth=2,
                   label=f'MAE = {mae:.1f} kJ/mol, R² = {r_value**2:.3f}')

            ax.set_title(f'{desc_name} vs {target_label}', fontsize=12, fontweight='bold')
            ax.set_xlabel(desc_name, fontsize=11)
            ax.set_ylabel(f'{target_col} (kJ/mol)', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Noms de fichier basés sur la cible
            target_suffix = target_col.replace('_kJ', '')
            output_file = output_dir / f"{desc_name}_vs_{target_suffix}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            generated_files.append(output_file)

    print(f"✓ {len(generated_files)} graphiques individuels générés dans single_graphs/")

    return generated_files


def plot_focused_individual_correlations(
    df_focused,
    df_desc,
    df_thermo,
    output_dir: Path
) -> List[Path]:
    """
    Génère des scatter plots individuels avec stats complètes pour les corrélations focalisées.

    Utilise plot_focused_correlation() pour chaque paire significative.

    Args:
        df_focused: DataFrame des corrélations focalisées
        df_desc: DataFrame des descripteurs par molécule
        df_thermo: DataFrame des cibles thermodynamiques
        output_dir: Répertoire de sortie

    Returns:
        List[Path]: Fichiers générés
    """
    output_dir = Path(output_dir) / "single_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PANDAS:
        print("  ⚠ pandas requis pour plot_focused_individual_correlations")
        return []

    if isinstance(df_focused, list):
        df_focused = pd.DataFrame(df_focused)
    if isinstance(df_desc, dict):
        df_desc = pd.DataFrame(df_desc)
    if isinstance(df_thermo, dict):
        df_thermo = pd.DataFrame(df_thermo)

    if df_focused.empty:
        return []

    # Fusionner descripteurs et cibles
    if 'molecule' in df_desc.columns and 'molecule' in df_thermo.columns:
        df_merged = pd.merge(df_desc, df_thermo, on='molecule', how='outer',
                           suffixes=('', '_thermo'))
    else:
        df_merged = df_desc

    target_labels = {
        'ddU_red_kJ': 'ΔΔU (Réduction)',
        'ddU_ox_kJ': 'ΔΔU (Oxydation)',
        'TdS_red_kJ': 'TΔS (Réduction)',
        'TdS_ox_kJ': 'TΔS (Oxydation)',
    }

    target_colors = {
        'ddU_red_kJ': 'darkgreen',
        'ddU_ox_kJ': 'darkblue',
        'TdS_red_kJ': 'darkorange',
        'TdS_ox_kJ': 'darkred',
    }

    generated_files = []
    molecules = df_merged['molecule'].values if 'molecule' in df_merged.columns else None

    for _, row in df_focused.iterrows():
        desc_name = row['descriptor']
        target_col = row['target']

        if desc_name not in df_merged.columns or target_col not in df_merged.columns:
            continue

        x = df_merged[desc_name].values.astype(float)
        y = df_merged[target_col].values.astype(float)

        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 3:
            continue

        mol_valid = molecules[mask] if molecules is not None else None
        target_label = target_labels.get(target_col, target_col)
        color = target_colors.get(target_col, 'darkgray')

        # Construire les stats depuis la row
        stats = {col: row[col] for col in row.index if col not in ['descriptor', 'target']}

        target_suffix = target_col.replace('_kJ', '')
        out_file = output_dir / f"{desc_name}_vs_{target_suffix}.png"

        plot_focused_correlation(
            x_valid, y_valid, stats,
            desc_name, target_label, out_file,
            molecules=mol_valid, color=color
        )

        generated_files.append(out_file)

    print(f"✓ {len(generated_files)} graphiques focalisés générés dans single_graphs/")
    return generated_files


def generate_all_frequency_plots(
    df_corr,
    df_desc,
    output_dir: Path,
    top_n: int = 10,
    df_focused=None,
    df_thermo=None
) -> None:
    """
    Génère tous les graphiques de corrélation fréquence.

    Args:
        df_corr: DataFrame des corrélations Edef
        df_desc: DataFrame des descripteurs par molécule
        output_dir (Path): Répertoire de sortie
        top_n (int): Nombre de top corrélations pour les graphiques principaux
        df_focused: DataFrame des corrélations focalisées ΔΔU/TΔS (optionnel)
        df_thermo: DataFrame des cibles thermodynamiques (optionnel)

    Exports:
        - top_correlations_reduction.png / _oxidation.png (Edef)
        - top_correlations_ddU_reduction.png / _oxidation.png (si df_focused)
        - top_correlations_TdS_reduction.png / _oxidation.png (si df_focused)
        - single_graphs/*.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 80 + "\n")

    # 1. Top corrélations Edef
    print("Génération des top corrélations Edef...")
    plot_top_correlations(df_corr, df_desc, output_dir, top_n)

    # 2. Top corrélations focalisées ΔΔU/TΔS
    if df_focused is not None and df_thermo is not None:
        print("\nGénération des top corrélations focalisées ΔΔU/TΔS...")
        plot_top_focused_correlations(df_focused, df_desc, df_thermo, output_dir, top_n)

        # 3. Graphiques individuels focalisés (avec stats complètes)
        print("\nGénération des graphiques individuels focalisés...")
        plot_focused_individual_correlations(df_focused, df_desc, df_thermo, output_dir)
    else:
        # 3. Graphiques individuels classiques
        print("\nGénération des graphiques individuels...")
        plot_individual_correlations(df_desc, output_dir, df_thermo=df_thermo)

    print("\n" + "=" * 80)
    print("✓ VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS")
    print("=" * 80)
