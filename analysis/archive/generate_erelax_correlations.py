#!/usr/bin/env python3
"""
Génération de graphiques de corrélation entre descripteurs moléculaires et E_relax.

Ce script génère des graphiques montrant les corrélations entre les descripteurs
de l'état neutre (HOMO/LUMO, charges de Bader, énergies IQA, EA/IP) et l'énergie
de relaxation E_relax pour la réduction et l'oxydation.

Usage:
    # Depuis un répertoire de workdirs (collecte les données)
    ./generate_erelax_correlations.py --batch-dir 1594174/dir_workdir \
                                       --output-dir rapport_erelax_correlations

    # Depuis un CSV déjà généré par visualize_batch.py
    ./generate_erelax_correlations.py --csv molecules_summary.csv \
                                       --output-dir rapport_erelax_correlations
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr


# =============================================================================
# Configuration
# =============================================================================

# Descripteurs d'intérêt, groupés par catégorie
DESCRIPTOR_GROUPS = {
    'Orbitales frontières': {
        'HOMO_eV': 'HOMO (eV)',
        'LUMO_eV': 'LUMO (eV)',
        'gap_eV': 'Gap HOMO-LUMO (eV)',
    },
    'Énergies verticales': {
        'EA_kJ': 'EA (kJ/mol)',
        'IP_kJ': 'IP (kJ/mol)',
    },
    'Charges de Bader (neutre)': {
        'C1_bader': 'q(C1)',
        'C2_bader': 'q(C2)',
        'C3_bader': 'q(C3)',
        'C4_bader': 'q(C4)',
        'C5_bader': 'q(C5)',
        'C6_bader': 'q(C6)',
        'O7_bader': 'q(O7)',
        'O8_bader': 'q(O8)',
    },
    'Énergies IQA (neutre)': {
        'C1_iqa': 'IQA(C1)',
        'C2_iqa': 'IQA(C2)',
        'C3_iqa': 'IQA(C3)',
        'C4_iqa': 'IQA(C4)',
        'C5_iqa': 'IQA(C5)',
        'C6_iqa': 'IQA(C6)',
        'O7_iqa': 'IQA(O7)',
        'O8_iqa': 'IQA(O8)',
    },
    'Descripteurs dérivés': {
        'q_O_total': 'q(O7+O8)',
        'q_O_asymmetry': '|q(O7)-q(O8)|',
    },
    'Statistiques fréquentielles': {
        'freq_mean': 'freq moy. (cm⁻¹)',
        'freq_std': 'freq σ (cm⁻¹)',
        'freq_median': 'freq médiane',
        'freq_min': 'freq min',
        'freq_max': 'freq max',
        'freq_spread': 'freq spread',
        'zpe_kj': 'ZPE (kJ/mol)',
    },
    'Distribution des modes': {
        'n_modes_total': 'N modes total',
        'n_modes_low': 'N modes 0-500',
        'n_modes_medium_low': 'N modes 500-1000',
        'n_modes_medium': 'N modes 1000-2000',
        'n_modes_high': 'N modes >2000',
        'flexibility_index': 'indice flexibilité',
        'stiffness_index': 'indice rigidité',
    },
    'Intensités IR': {
        'ir_max': 'IR max (km/mol)',
        'ir_sum': 'IR somme',
        'ir_mean': 'IR moy.',
        'freq_of_max_ir': 'freq(IR max)',
        'freq_ir_weighted': 'freq pondérée IR',
    },
    'Sommes de fréquences par plage': {
        'sum_freq_low': 'Σfreq 0-500',
        'sum_freq_medium_low': 'Σfreq 500-1000',
        'sum_freq_medium': 'Σfreq 1000-2000',
        'sum_freq_high': 'Σfreq >2000',
        'mean_freq_low': 'moy. freq 0-500',
        'mean_freq_medium': 'moy. freq 1000-2000',
    },
}

# Cibles
TARGETS = {
    'Edef_red_kJ': '$E_{relax}$ réduction (kJ/mol)',
    'Edef_ox_kJ': '$E_{relax}$ oxydation (kJ/mol)',
}


# =============================================================================
# Fonctions de corrélation
# =============================================================================

def compute_all_correlations(df, descriptors, targets):
    """
    Calcule R² pour chaque paire (descripteur, cible).

    Returns:
        Liste de dicts avec les résultats triés par R² décroissant.
    """
    results = []
    for desc_col, desc_label in descriptors.items():
        for target_col, target_label in targets.items():
            if desc_col not in df.columns or target_col not in df.columns:
                continue
            valid = df[[desc_col, target_col]].dropna()
            if len(valid) < 3:
                continue
            r, p = pearsonr(valid[desc_col], valid[target_col])
            slope, intercept, _, _, _ = linregress(valid[desc_col], valid[target_col])
            y_pred = slope * valid[desc_col].values + intercept
            mae = np.mean(np.abs(valid[target_col].values - y_pred))
            results.append({
                'descriptor': desc_col,
                'descriptor_label': desc_label,
                'target': target_col,
                'target_label': target_label,
                'R2': r ** 2,
                'r': r,
                'p_value': p,
                'slope': slope,
                'intercept': intercept,
                'n': len(valid),
                'MAE': mae,
            })
    return sorted(results, key=lambda x: x['R2'], reverse=True)


# =============================================================================
# Fonctions utilitaires per-conformère
# =============================================================================

def get_molecule_color_map(molecule_names):
    """Retourne un mapping molécule -> couleur distincte."""
    unique = sorted(set(molecule_names))
    n = len(unique)
    cmap = plt.colormaps.get_cmap('tab20').resampled(max(n, 20))
    return {mol: cmap(i) for i, mol in enumerate(unique)}


def _suffix(per_conformer):
    """Suffixe pour les noms de fichiers en mode per-conformer."""
    return '_per_conformer' if per_conformer else ''


# =============================================================================
# Fonctions de visualisation
# =============================================================================

def plot_scatter_correlation(df, desc_col, desc_label, target_col, target_label,
                             output_file, color='steelblue', per_conformer=False):
    """Scatter plot avec régression linéaire et labels des molécules ou conformères."""
    if per_conformer:
        needed = ['molecule_name', 'conformer', desc_col, target_col]
    else:
        needed = ['molecule_name', desc_col, target_col]
    valid = df[[c for c in needed if c in df.columns]].dropna(subset=[desc_col, target_col])
    if len(valid) < 3:
        return None

    fig, ax = plt.subplots(figsize=(8, 6) if per_conformer else (7, 5.5))

    if per_conformer:
        color_map = get_molecule_color_map(valid['molecule_name'])
        for mol_name, group in valid.groupby('molecule_name'):
            ax.scatter(group[desc_col], group[target_col], s=70, alpha=0.7,
                       color=color_map[mol_name], edgecolors='white', linewidths=0.5,
                       zorder=3, label=mol_name)
        for _, row in valid.iterrows():
            ax.annotate(
                str(int(row['conformer'])),
                (row[desc_col], row[target_col]),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7, alpha=0.8
            )
    else:
        ax.scatter(valid[desc_col], valid[target_col], s=90, alpha=0.7,
                   color=color, edgecolors='white', linewidths=0.5, zorder=3)
        for _, row in valid.iterrows():
            ax.annotate(
                row['molecule_name'],
                (row[desc_col], row[target_col]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8
            )

    slope, intercept, r_value, _, _ = linregress(valid[desc_col], valid[target_col])
    x_fit = np.linspace(valid[desc_col].min(), valid[desc_col].max(), 100)
    y_fit = slope * x_fit + intercept
    y_pred_on_data = slope * valid[desc_col].values + intercept
    mae = np.mean(np.abs(valid[target_col].values - y_pred_on_data))
    ax.plot(x_fit, y_fit, 'r--', linewidth=2,
            label=f'MAE $= {mae:.1f}$ kJ/mol, $R^2 = {r_value**2:.3f}$\n$y = {slope:.2f}x {intercept:+.2f}$')

    ax.set_xlabel(desc_label, fontsize=11)
    ax.set_ylabel(target_label, fontsize=11)
    ax.set_title(f'{desc_label}  vs  {target_label}', fontsize=12, fontweight='bold')
    if per_conformer:
        ax.legend(fontsize=7, loc='best', ncol=2, framealpha=0.8)
    else:
        ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    return r_value ** 2


def plot_heatmap_erelax(df, descriptors, targets, output_file, title=None):
    """
    Heatmap R² entre descripteurs (lignes) et E_relax cibles (colonnes).
    """
    desc_cols = [c for c in descriptors if c in df.columns]
    target_cols = [c for c in targets if c in df.columns]

    if not desc_cols or not target_cols:
        return

    r2_matrix = np.full((len(desc_cols), len(target_cols)), np.nan)

    for i, dc in enumerate(desc_cols):
        for j, tc in enumerate(target_cols):
            valid = df[[dc, tc]].dropna()
            if len(valid) >= 3:
                r, _ = pearsonr(valid[dc], valid[tc])
                r2_matrix[i, j] = r ** 2

    desc_labels = [descriptors[c] for c in desc_cols]
    target_labels = [targets[c] for c in target_cols]

    fig_height = max(6, len(desc_cols) * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(6, fig_height))

    try:
        import seaborn as sns
        sns.heatmap(
            r2_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            vmin=0, vmax=1,
            linewidths=0.5,
            xticklabels=target_labels,
            yticklabels=desc_labels,
            cbar_kws={"shrink": 0.7, "label": "$R^2$"},
            ax=ax
        )
    except ImportError:
        im = ax.imshow(r2_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('$R^2$', rotation=270, labelpad=15)
        ax.set_xticks(np.arange(len(target_labels)))
        ax.set_yticks(np.arange(len(desc_labels)))
        ax.set_xticklabels(target_labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(desc_labels, fontsize=9)
        for i in range(r2_matrix.shape[0]):
            for j in range(r2_matrix.shape[1]):
                if not np.isnan(r2_matrix[i, j]):
                    color = 'white' if r2_matrix[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{r2_matrix[i, j]:.2f}',
                            ha='center', va='center', color=color, fontsize=9)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    else:
        ax.set_title('Corrélations $R^2$ : Descripteurs vs $E_{relax}$',
                      fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap sauvegardee: {output_file.name}")


def plot_top_correlations_grid(df, corr_results, output_file, top_n=8,
                                process_label="", per_conformer=False):
    """
    Grille des top N meilleures corrélations pour un processus (red/ox).
    """
    if len(corr_results) == 0:
        return

    top = corr_results[:top_n]
    n = len(top)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    color_map = get_molecule_color_map(df['molecule_name']) if per_conformer else None

    for idx, res in enumerate(top):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        desc_col = res['descriptor']
        target_col = res['target']
        if per_conformer:
            needed_cols = ['molecule_name', 'conformer', desc_col, target_col]
        else:
            needed_cols = ['molecule_name', desc_col, target_col]
        valid = df[[c for c in needed_cols if c in df.columns]].dropna(subset=[desc_col, target_col])

        y_pred = res['slope'] * valid[desc_col].values + res['intercept']
        y_real = valid[target_col].values

        if per_conformer:
            mol_arr = valid['molecule_name'].values
            for mol_name in sorted(set(mol_arr)):
                mask = mol_arr == mol_name
                ax.scatter(y_pred[mask], y_real[mask],
                           s=50, alpha=0.7, color=color_map[mol_name],
                           edgecolors='white', linewidths=0.3, zorder=3)
            conf_arr = valid['conformer'].values
            for i in range(len(y_pred)):
                ax.annotate(str(int(conf_arr[i])), (y_pred[i], y_real[i]),
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=5, alpha=0.7)
        else:
            ax.scatter(y_pred, y_real, s=70, alpha=0.7,
                       color='steelblue', edgecolors='white', linewidths=0.5, zorder=3)
            for mol_name, xp, yr in zip(valid['molecule_name'], y_pred, y_real):
                ax.annotate(mol_name, (xp, yr),
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=6, alpha=0.7)

        all_vals = np.concatenate([y_real, y_pred])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        margin = (vmax - vmin) * 0.1
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'r--', linewidth=1.5, alpha=0.8)

        mae = np.mean(np.abs(y_real - y_pred))
        ax.set_title(f"{res['descriptor_label']}\nMAE $= {mae:.1f}$ kJ/mol, $R^2 = {res['R2']:.3f}$",
                      fontsize=9, fontweight='bold')
        ax.set_xlabel('$E_{relax}$ prédit (kJ/mol)', fontsize=8)
        ax.set_ylabel('$E_{relax}$ réel (kJ/mol)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, zorder=0)

    # Masquer axes vides
    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    suptitle = f'Top {n} corrélations avec $E_{{relax}}$ ({process_label})'
    if per_conformer:
        suptitle += ' [per-conformère]'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    # Légende partagée en mode per-conformer
    if per_conformer and color_map:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color_map[m], markersize=8, label=m)
                   for m in sorted(color_map.keys())]
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(6, len(color_map)), fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Grille top {n} ({process_label}): {output_file.name}")


def plot_grouped_heatmaps(df, output_dir):
    """
    Heatmaps R² par groupe de descripteurs.
    """
    for group_name, desc_dict in DESCRIPTOR_GROUPS.items():
        present = {k: v for k, v in desc_dict.items() if k in df.columns}
        if len(present) < 2:
            continue
        safe_name = group_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        safe_name = safe_name.replace("é", "e").replace("è", "e")
        output_file = output_dir / f"heatmap_{safe_name}.png"
        plot_heatmap_erelax(df, present, TARGETS, output_file,
                            title=f'$R^2$ : {group_name} vs $E_{{relax}}$')


def export_correlation_table(corr_results, output_file):
    """Exporte les corrélations en CSV et en texte lisible."""
    if not corr_results:
        return

    # CSV
    df_corr = pd.DataFrame(corr_results)
    csv_file = output_file.with_suffix('.csv')
    df_corr.to_csv(csv_file, index=False)

    # Texte formaté
    txt_file = output_file.with_suffix('.txt')
    with open(txt_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("CORRÉLATIONS DESCRIPTEURS vs E_relax\n")
        f.write("=" * 100 + "\n\n")

        for target_col, target_label in TARGETS.items():
            f.write(f"\n{'─' * 80}\n")
            f.write(f"  Cible : {target_label}\n")
            f.write(f"{'─' * 80}\n")
            f.write(f"{'Descripteur':<30} {'R²':>8} {'r':>8} {'p-value':>12} {'n':>5}\n")
            f.write(f"{'─' * 80}\n")

            target_results = [r for r in corr_results if r['target'] == target_col]
            target_results.sort(key=lambda x: x['R2'], reverse=True)

            for res in target_results:
                f.write(f"{res['descriptor_label']:<30} {res['R2']:>8.4f} "
                        f"{res['r']:>8.4f} {res['p_value']:>12.2e} {res['n']:>5d}\n")

    print(f"  Table exportee: {csv_file.name} + {txt_file.name}")


# =============================================================================
# Chargement des données
# =============================================================================

def load_from_csv(csv_path):
    """Charge les données depuis un CSV pré-existant."""
    df = pd.read_csv(csv_path)
    print(f"Donnees chargees depuis {csv_path}: {len(df)} molecules")
    return df


def _load_multi_molecule_module():
    """Charge dynamiquement le module common/multi_molecule_data."""
    import importlib.util
    import os

    project_root = str(Path(__file__).resolve().parent.parent)
    spec = importlib.util.spec_from_file_location(
        "multi_molecule_data",
        os.path.join(project_root, "common", "multi_molecule_data.py")
    )
    if spec is None or spec.loader is None:
        print("Impossible de charger common/multi_molecule_data.py")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_from_workdirs(batch_dir):
    """Charge les données depuis les workdirs en utilisant multi_molecule_data."""
    mod = _load_multi_molecule_module()

    workdirs = sorted([d for d in Path(batch_dir).glob("*_workdir") if d.is_dir()])
    if not workdirs:
        print(f"Aucun workdir trouve dans {batch_dir}")
        sys.exit(1)

    print(f"{len(workdirs)} workdirs trouves")
    df = mod.collect_all_molecules(workdirs)
    return df


def load_from_workdirs_per_conformer(batch_dir, freq_csv=None):
    """Charge les données per-conformère depuis les workdirs."""
    mod = _load_multi_molecule_module()

    workdirs = sorted([d for d in Path(batch_dir).glob("*_workdir") if d.is_dir()])
    if not workdirs:
        print(f"Aucun workdir trouve dans {batch_dir}")
        sys.exit(1)

    print(f"{len(workdirs)} workdirs trouves (mode per-conformere)")
    df = mod.collect_all_conformers(workdirs, freq_csv=freq_csv)
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correlations descripteurs moleculaires vs E_relax"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str, help="CSV deja genere (molecules_summary.csv)")
    group.add_argument('--batch-dir', type=str, help="Repertoire contenant les workdirs")

    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'erelax_correlations'),
                        help="Repertoire de sortie")
    parser.add_argument('--top-n', type=int, default=8,
                        help="Nombre de top correlations a afficher")
    parser.add_argument('--per-conformer', action='store_true', default=False,
                        help="Afficher tous les conformeres individuellement")
    parser.add_argument('--freq-csv', type=str, default=None,
                        help="CSV des descripteurs frequentiels per-conformer")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sfx = _suffix(args.per_conformer)

    print("=" * 80)
    print("CORRELATIONS DESCRIPTEURS MOLECULAIRES vs E_relax")
    if args.per_conformer:
        print("  Mode: PER-CONFORMERE (1 point par conformere)")
    print("=" * 80)

    # 1. Charger les données
    if args.per_conformer:
        if args.csv:
            df = load_from_csv(args.csv)
        elif args.batch_dir:
            df = load_from_workdirs_per_conformer(args.batch_dir, freq_csv=args.freq_csv)
        else:
            print("ERREUR: --csv ou --batch-dir requis")
            sys.exit(1)
    else:
        if args.csv:
            df = load_from_csv(args.csv)
        else:
            df = load_from_workdirs(args.batch_dir)

    # Vérifier que E_relax est disponible
    available_targets = {k: v for k, v in TARGETS.items() if k in df.columns}
    if not available_targets:
        print("ERREUR: colonnes Edef_red_kJ / Edef_ox_kJ absentes du CSV")
        sys.exit(1)

    print(f"\nCibles disponibles: {list(available_targets.keys())}")
    if args.per_conformer:
        n_mols = df['molecule_name'].nunique()
        print(f"Conformeres: {len(df)} ({n_mols} molecules)")
    else:
        print(f"Molecules: {len(df)}")

    # Aplatir tous les descripteurs
    all_descriptors = {}
    for group_descs in DESCRIPTOR_GROUPS.values():
        all_descriptors.update(group_descs)

    present_descriptors = {k: v for k, v in all_descriptors.items() if k in df.columns}
    print(f"Descripteurs disponibles: {len(present_descriptors)}")

    # 2. Calculer toutes les corrélations
    print("\n" + "-" * 80)
    print("CALCUL DES CORRELATIONS")
    print("-" * 80)

    all_corr = compute_all_correlations(df, present_descriptors, available_targets)
    print(f"\n{len(all_corr)} correlations calculees")

    # 3. Exporter la table de corrélations
    export_correlation_table(all_corr, output_dir / f"erelax_correlations{sfx}")

    # 4. Heatmap globale (tous descripteurs vs E_relax)
    print("\n" + "-" * 80)
    print("GENERATION DES HEATMAPS")
    print("-" * 80)

    plot_heatmap_erelax(df, present_descriptors, available_targets,
                        output_dir / f"heatmap_all_vs_erelax{sfx}.png",
                        title='$R^2$ : Tous descripteurs vs $E_{relax}$')

    # 5. Heatmaps par groupe
    plot_grouped_heatmaps(df, output_dir)

    # 6. Grilles des top corrélations par processus
    print("\n" + "-" * 80)
    print("GENERATION DES TOP CORRELATIONS")
    print("-" * 80)

    for target_col, target_label in available_targets.items():
        process = "reduction" if "red" in target_col else "oxydation"
        target_corr = [r for r in all_corr if r['target'] == target_col]
        safe_process = process.replace("é", "e")
        plot_top_correlations_grid(
            df, target_corr,
            output_dir / f"top_correlations_{safe_process}{sfx}.png",
            top_n=args.top_n,
            process_label=process,
            per_conformer=args.per_conformer
        )

    # 7. Graphiques individuels des meilleures corrélations
    print("\n" + "-" * 80)
    print("GENERATION DES GRAPHIQUES INDIVIDUELS")
    print("-" * 80)

    single_dir = output_dir / "single_graphs"
    single_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for res in all_corr:
        process = "red" if "red" in res['target'] else "ox"
        safe_desc = res['descriptor'].replace('_', '-')
        filename = f"{safe_desc}_vs_Erelax_{process}{sfx}.png"
        color = '#c0392b' if process == 'red' else '#2980b9'
        plot_scatter_correlation(
            df, res['descriptor'], res['descriptor_label'],
            res['target'], res['target_label'],
            single_dir / filename, color=color,
            per_conformer=args.per_conformer
        )
        count += 1

    print(f"  {count} graphiques individuels generes dans {single_dir}/")

    # 8. Résumé des meilleures corrélations
    print("\n" + "=" * 80)
    print("RESUME DES MEILLEURES CORRELATIONS")
    print("=" * 80)

    for target_col, target_label in available_targets.items():
        print(f"\n  {target_label}:")
        target_corr = [r for r in all_corr if r['target'] == target_col][:5]
        for i, res in enumerate(target_corr, 1):
            print(f"    {i}. {res['descriptor_label']:<25} R² = {res['R2']:.4f}  "
                  f"(r = {res['r']:+.4f}, p = {res['p_value']:.2e})")

    print(f"\nResultats sauvegardes dans: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
