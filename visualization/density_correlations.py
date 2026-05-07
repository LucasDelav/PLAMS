"""
Visualisation des correlations densite electronique QTAIM vs EA/IP.

Ce module genere des graphiques pour visualiser les correlations entre
les descripteurs de densite et les cibles EA/IP verticaux, ainsi que
la comparaison avec le theoreme de Koopmans.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def plot_top_density_correlations(
    df_corr,
    df_desc,
    output_dir: Path,
    top_n: int = 10,
    target_col: str = 'EA_kJ',
    title_suffix: str = 'EA',
) -> None:
    """
    Grille scatter des meilleures correlations descripteurs-densite vs une cible.

    Args:
        df_corr: DataFrame des correlations (avec 'descriptor', 'r2', etc.)
        df_desc: DataFrame des descripteurs par molecule
        output_dir: Repertoire de sortie
        top_n: Nombre de top correlations a afficher
        target_col: Colonne cible dans df_desc
        title_suffix: Suffixe pour le titre et le nom de fichier
    """
    from scipy.stats import linregress

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PANDAS:
        print("  ⚠ pandas requis pour plot_top_density_correlations")
        return

    if isinstance(df_corr, list):
        df_corr = pd.DataFrame(df_corr)
    if isinstance(df_desc, dict):
        df_desc = pd.DataFrame(df_desc)

    if df_corr.empty:
        print(f"  ⚠ Aucune correlation {title_suffix} a tracer")
        return

    # Trier par R² et prendre les top N
    df_sorted = df_corr.sort_values('r2', ascending=False).head(top_n)

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
        r2 = row.get('r2', np.nan)
        spearman_rho = row.get('spearman_rho', np.nan)

        if desc_name not in df_desc.columns or target_col not in df_desc.columns:
            ax.text(0.5, 0.5, 'Donnees\nmanquantes', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        x = df_desc[desc_name].values.astype(float)
        y = df_desc[target_col].values.astype(float)
        molecules = df_desc['molecule'].values if 'molecule' in df_desc.columns else None

        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 2:
            ax.text(0.5, 0.5, 'Donnees\ninsuffisantes', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        ax.scatter(x_valid, y_valid, s=80, alpha=0.7, color='steelblue',
                   edgecolors='white', linewidths=0.5, zorder=3)

        if molecules is not None:
            mol_valid = molecules[mask]
            for i, mol in enumerate(mol_valid):
                ax.annotate(mol, (x_valid[i], y_valid[i]),
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=7, alpha=0.7)

        slope, intercept, _, _, _ = linregress(x_valid, y_valid)
        x_fit = np.linspace(x_valid.min(), x_valid.max(), 50)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.8)

        y_pred = slope * x_valid + intercept
        mae = np.mean(np.abs(y_valid - y_pred))

        subtitle = f'MAE={mae:.1f}, R²={r2:.3f}'
        if not np.isnan(spearman_rho):
            subtitle += f', ρ={spearman_rho:.3f}'
        ax.set_title(f'{desc_name}\n{subtitle}', fontsize=9, fontweight='bold')
        ax.set_xlabel(desc_name, fontsize=8)
        ax.set_ylabel(f'{target_col} (kJ/mol)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, zorder=0)

    # Masquer les axes vides
    for idx in range(len(df_sorted), n_rows * n_cols):
        ax_row, ax_col = idx // n_cols, idx % n_cols
        axes[ax_row, ax_col].set_visible(False)

    plt.suptitle(
        f"Top Correlations: Descripteurs Densite vs {title_suffix}",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    output_file = output_dir / f"top_correlations_{title_suffix}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ {output_file.name} sauvegarde")


def plot_koopmans_comparison(df_desc, output_dir: Path) -> None:
    """
    Scatter HOMO vs IP et -LUMO vs EA pour reference Koopmans.

    Args:
        df_desc: DataFrame avec EA_kJ, IP_kJ, HOMO_eV, LUMO_eV
        output_dir: Repertoire de sortie
    """
    from scipy.stats import linregress, pearsonr

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PANDAS:
        print("  ⚠ pandas requis pour plot_koopmans_comparison")
        return

    if isinstance(df_desc, dict):
        df_desc = pd.DataFrame(df_desc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    molecules = df_desc['molecule'].values if 'molecule' in df_desc.columns else None

    # --- Panel 1: -HOMO vs IP ---
    if 'HOMO_eV' in df_desc.columns and 'IP_kJ' in df_desc.columns:
        homo = -df_desc['HOMO_eV'].values.astype(float)  # -HOMO en eV
        ip = df_desc['IP_kJ'].values.astype(float)

        mask = ~np.isnan(homo) & ~np.isnan(ip)
        if np.sum(mask) >= 3:
            x, y = homo[mask], ip[mask]
            ax1.scatter(x, y, s=90, alpha=0.7, color='#2980b9',
                        edgecolors='white', linewidths=0.5, zorder=3)

            if molecules is not None:
                for i, mol in enumerate(molecules[mask]):
                    ax1.annotate(mol, (x[i], y[i]),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=7, alpha=0.8)

            slope, intercept, _, _, _ = linregress(x, y)
            r, _ = pearsonr(x, y)
            mae = np.mean(np.abs(y - (slope * x + intercept)))
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax1.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=2,
                     label=f'MAE $= {mae:.1f}$ kJ/mol, $R^2 = {r**2:.3f}$\n$y = {slope:.1f}x {intercept:+.1f}$')
            ax1.set_title('-HOMO vs IP (Koopmans)', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10, loc='best')

        ax1.set_xlabel('-HOMO (eV)', fontsize=11)
        ax1.set_ylabel('IP (kJ/mol)', fontsize=11)
        ax1.grid(True, alpha=0.3, zorder=0)

    # --- Panel 2: -LUMO vs EA ---
    if 'LUMO_eV' in df_desc.columns and 'EA_kJ' in df_desc.columns:
        lumo = -df_desc['LUMO_eV'].values.astype(float)  # -LUMO en eV
        ea = df_desc['EA_kJ'].values.astype(float)

        mask = ~np.isnan(lumo) & ~np.isnan(ea)
        if np.sum(mask) >= 3:
            x, y = lumo[mask], ea[mask]
            ax2.scatter(x, y, s=90, alpha=0.7, color='#c0392b',
                        edgecolors='white', linewidths=0.5, zorder=3)

            if molecules is not None:
                for i, mol in enumerate(molecules[mask]):
                    ax2.annotate(mol, (x[i], y[i]),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=7, alpha=0.8)

            slope, intercept, _, _, _ = linregress(x, y)
            r, _ = pearsonr(x, y)
            mae = np.mean(np.abs(y - (slope * x + intercept)))
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax2.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=2,
                     label=f'MAE $= {mae:.1f}$ kJ/mol, $R^2 = {r**2:.3f}$\n$y = {slope:.1f}x {intercept:+.1f}$')
            ax2.set_title('-LUMO vs EA (Koopmans)', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10, loc='best')

        ax2.set_xlabel('-LUMO (eV)', fontsize=11)
        ax2.set_ylabel('EA (kJ/mol)', fontsize=11)
        ax2.grid(True, alpha=0.3, zorder=0)

    plt.suptitle("Comparaison Koopmans: Orbitales Frontieres vs EA/IP DFT",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = output_dir / "koopmans_vs_dft.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ {output_file.name} sauvegarde")


def plot_density_individual(
    df_corr,
    df_desc,
    output_dir: Path,
    target_col: str = 'EA_kJ',
    target_label: str = 'EA',
    color: str = 'steelblue',
) -> List[Path]:
    """
    Plots individuels detailles avec stats completes.

    Args:
        df_corr: DataFrame des correlations pour cette cible
        df_desc: DataFrame des descripteurs par molecule
        output_dir: Repertoire de sortie
        target_col: Colonne cible
        target_label: Label pour le titre
        color: Couleur de la regression

    Returns:
        Liste des fichiers generes
    """
    from scipy.stats import linregress

    single_dir = Path(output_dir) / "single_graphs"
    single_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PANDAS:
        return []

    if isinstance(df_corr, list):
        df_corr = pd.DataFrame(df_corr)
    if isinstance(df_desc, dict):
        df_desc = pd.DataFrame(df_desc)

    if df_corr.empty:
        return []

    generated = []
    molecules = df_desc['molecule'].values if 'molecule' in df_desc.columns else None

    for _, row in df_corr.iterrows():
        desc_name = row['descriptor']

        if desc_name not in df_desc.columns or target_col not in df_desc.columns:
            continue

        x = df_desc[desc_name].values.astype(float)
        y = df_desc[target_col].values.astype(float)

        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 3:
            continue

        mol_valid = molecules[mask] if molecules is not None else None

        # Construire les stats depuis la row
        stats = {col: row[col] for col in row.index if col not in ['descriptor', 'target']}

        fig, ax = plt.subplots(figsize=(7, 5.5))

        ax.scatter(x_valid, y_valid, s=90, alpha=0.7, color='steelblue',
                   edgecolors='white', linewidths=0.5, zorder=3)

        if mol_valid is not None:
            for i, mol in enumerate(mol_valid):
                ax.annotate(mol, (x_valid[i], y_valid[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.8)

        slope = stats.get('slope', 0)
        intercept = stats.get('intercept', 0)
        x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'k--', linewidth=2, zorder=2)

        # Boite de texte avec stats
        y_pred = slope * x_valid + intercept
        mae = np.mean(np.abs(y_valid - y_pred))

        r2_boot_lo = stats.get('r2_bootstrap_lo', np.nan)
        r2_boot_hi = stats.get('r2_bootstrap_hi', np.nan)
        shapiro_p = stats.get('shapiro_p', np.nan)

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
        lines.append(f"n = {stats.get('n', len(x_valid))}")

        textstr = '\n'.join(lines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace', bbox=props)

        ax.set_title(f'{desc_name} vs {target_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel(desc_name, fontsize=11)
        ax.set_ylabel(f'{target_col} (kJ/mol)', fontsize=11)
        ax.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        out_file = single_dir / f"{desc_name}_vs_{target_label}_kJ.png"
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close()

        generated.append(out_file)

    print(f"✓ {len(generated)} graphiques individuels {target_label} generes")
    return generated


def generate_all_density_plots(
    df_corr_ea,
    df_corr_ip,
    df_desc,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """
    Orchestre toute la generation de plots densite.

    Args:
        df_corr_ea: DataFrame correlations EA
        df_corr_ip: DataFrame correlations IP
        df_desc: DataFrame descripteurs avec EA_kJ, IP_kJ, HOMO_eV, LUMO_eV
        output_dir: Repertoire de sortie
        top_n: Nombre de top correlations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATION DES VISUALISATIONS DENSITE")
    print("=" * 80 + "\n")

    # 1. Top correlations EA
    print("Generation des top correlations EA...")
    plot_top_density_correlations(
        df_corr_ea, df_desc, output_dir, top_n,
        target_col='EA_kJ', title_suffix='EA'
    )

    # 2. Top correlations IP
    print("Generation des top correlations IP...")
    plot_top_density_correlations(
        df_corr_ip, df_desc, output_dir, top_n,
        target_col='IP_kJ', title_suffix='IP'
    )

    # 3. Comparaison Koopmans
    print("\nGeneration du graphique Koopmans...")
    plot_koopmans_comparison(df_desc, output_dir)

    # 4. Graphiques individuels
    print("\nGeneration des graphiques individuels EA...")
    plot_density_individual(
        df_corr_ea, df_desc, output_dir,
        target_col='EA_kJ', target_label='EA', color='darkorange'
    )

    print("Generation des graphiques individuels IP...")
    plot_density_individual(
        df_corr_ip, df_desc, output_dir,
        target_col='IP_kJ', target_label='IP', color='darkgreen'
    )

    print("\n" + "=" * 80)
    print("✓ VISUALISATIONS DENSITE GENEREES AVEC SUCCES")
    print("=" * 80)
