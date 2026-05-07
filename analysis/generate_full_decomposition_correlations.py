#!/usr/bin/env python3
"""
Génération des corrélations pour les 4 termes de la décomposition de ΔG.

Produit :
  - Figure 1 : Barres empilées montrant le poids relatif de chaque terme
  - Figure 2 : EA vs LUMO, IP vs HOMO (théorème de Koopmans)
  - Figure 3 : Heatmap + top corrélations pour ΔΔU
  - Figure 4 : Heatmap + top corrélations pour TΔS
  - Figure 5 : Heatmap de synthèse combinée (8 cibles × descripteurs clés)

Usage:
    # Avec thermodynamic_terms.csv déjà généré (étape 1)
    ./generate_full_decomposition_correlations.py \
        --csv rapport_erelax_correlations/molecules_summary_with_freq.csv \
        --tds-csv rapport_full_decomposition/thermodynamic_terms.csv \
        --output-dir rapport_full_decomposition

    # Avec parsing fallback depuis les analysis.txt
    ./generate_full_decomposition_correlations.py \
        --csv rapport_erelax_correlations/molecules_summary_with_freq.csv \
        --batch-dir 1594174/dir_workdir \
        --output-dir rapport_full_decomposition
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from analysis import DEA_EXCLUDED


# =============================================================================
# Configuration : descripteurs (réutilisés de generate_erelax_correlations.py)
# =============================================================================

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
# Fonctions utilitaires
# =============================================================================

def compute_all_correlations(df, descriptors, targets):
    """
    Calcule R² pour chaque paire (descripteur, cible).
    Returns: Liste de dicts triés par R² décroissant.
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


def parse_tds_from_analysis_txt(batch_dir):
    """
    Parse TΔS depuis les fichiers _redox_analysis.txt (fallback sans amspython).
    Returns: DataFrame avec colonnes molecule_name, TdS_red_kJ, TdS_ox_kJ
    """
    batch_dir = Path(batch_dir)
    workdirs = sorted([d for d in batch_dir.glob("*_workdir") if d.is_dir()])

    results = []
    for workdir in workdirs:
        mol_name = workdir.name.replace('_workdir', '')
        analysis_file = workdir / "redox" / "results" / f"{mol_name}_redox_analysis.txt"

        if not analysis_file.exists():
            continue

        weights = []
        tds_red_values = []
        tds_ox_values = []
        current_weight = None
        current_section = None

        with open(analysis_file, 'r') as f:
            for line in f:
                if 'poids =' in line:
                    try:
                        current_weight = float(line.split('poids = ')[1].split(')')[0])
                    except (ValueError, IndexError):
                        current_weight = None
                elif 'RÉDUCTION:' in line or 'REDUCTION:' in line:
                    current_section = 'red'
                elif 'OXYDATION:' in line or 'OXIDATION:' in line:
                    current_section = 'ox'
                elif 'TΔS' in line and '=' in line and 'kJ/mol' in line:
                    try:
                        val = float(line.split('=')[1].strip().split()[0])
                        if current_section == 'red' and current_weight is not None:
                            tds_red_values.append(val)
                            if len(weights) < len(tds_red_values):
                                weights.append(current_weight)
                        elif current_section == 'ox':
                            tds_ox_values.append(val)
                    except (ValueError, IndexError):
                        pass

        if not tds_red_values or not tds_ox_values or not weights:
            continue

        n = min(len(tds_red_values), len(tds_ox_values), len(weights))
        total_w = sum(weights[:n])
        if total_w == 0:
            continue

        avg_red = sum(v * w for v, w in zip(tds_red_values[:n], weights[:n])) / total_w
        avg_ox = sum(v * w for v, w in zip(tds_ox_values[:n], weights[:n])) / total_w

        results.append({
            'molecule_name': mol_name,
            'TdS_red_kJ': avg_red,
            'TdS_ox_kJ': avg_ox,
        })

    return pd.DataFrame(results) if results else None


# =============================================================================
# Figure 1 : Poids relatif des termes (barres empilées)
# =============================================================================

def plot_decomposition_magnitude_bars(df, output_dir):
    """
    Diagramme en barres empilées montrant |EA/IP|, |E_relax|, |Rés| pour chaque molécule.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    molecules = df['molecule_name'].values

    # --- Oxydation ---
    ax = axes[0]
    ip_abs = df['IP_kJ'].abs().values
    erelax_ox_abs = df['Edef_ox_kJ'].abs().values
    res_ox_abs = df['Res_ox_kJ'].abs().values

    x = np.arange(len(molecules))
    width = 0.65

    ax.bar(x, ip_abs, width, label='|IP|', color='#3498db', alpha=0.85)
    ax.bar(x, erelax_ox_abs, width, bottom=ip_abs, label='|$E_{relax}$|', color='#e74c3c', alpha=0.85)
    ax.bar(x, res_ox_abs, width, bottom=ip_abs + erelax_ox_abs,
           label='|Rés.| ($\\Delta\\Delta U - T\\Delta S$)', color='#2ecc71', alpha=0.85)

    ax.set_ylabel('|Contribution| (kJ/mol)', fontsize=11)
    ax.set_title('Oxydation : décomposition de $|\\Delta G_{ox}|$', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # --- Réduction ---
    ax = axes[1]
    ea_abs = df['EA_kJ'].abs().values
    erelax_red_abs = df['Edef_red_kJ'].abs().values
    res_red_abs = df['Res_red_kJ'].abs().values

    ax.bar(x, ea_abs, width, label='|EA|', color='#3498db', alpha=0.85)
    ax.bar(x, erelax_red_abs, width, bottom=ea_abs, label='|$E_{relax}$|', color='#e74c3c', alpha=0.85)
    ax.bar(x, res_red_abs, width, bottom=ea_abs + erelax_red_abs,
           label='|Rés.| ($\\Delta\\Delta U - T\\Delta S$)', color='#2ecc71', alpha=0.85)

    ax.set_ylabel('|Contribution| (kJ/mol)', fontsize=11)
    ax.set_title('Réduction : décomposition de $|\\Delta G_{red}|$', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = output_dir / "decomposition_magnitude_bars.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure 1 sauvegardée: {out.name}")

    # Tableau récapitulatif
    print("\n  Tableau récapitulatif des poids relatifs :")
    print(f"  {'Terme':<20} {'Moy. %|ΔG|':>12} {'Min %':>8} {'Max %':>8}")
    print(f"  {'-'*50}")

    for process, terms in [
        ('Oxydation', [
            ('IP', ip_abs, df['dG_oxidation_kJ'].abs().values),
            ('E_relax', erelax_ox_abs, df['dG_oxidation_kJ'].abs().values),
            ('Rés.', res_ox_abs, df['dG_oxidation_kJ'].abs().values),
        ]),
        ('Réduction', [
            ('EA', ea_abs, df['dG_reduction_kJ'].abs().values),
            ('E_relax', erelax_red_abs, df['dG_reduction_kJ'].abs().values),
            ('Rés.', res_red_abs, df['dG_reduction_kJ'].abs().values),
        ]),
    ]:
        print(f"\n  {process}:")
        for name, vals, dg_vals in terms:
            pct = vals / dg_vals * 100
            print(f"  {name:<20} {pct.mean():>10.1f}% {pct.min():>7.1f}% {pct.max():>7.1f}%")


# =============================================================================
# Figure 2 : EA vs LUMO, IP vs HOMO (Koopmans)
# =============================================================================

def plot_koopmans(df, output_dir, per_conformer=False):
    """Scatter plots EA vs -LUMO et IP vs -HOMO avec fit linéaire (kJ/mol)."""
    EH_TO_KJ = 2625.499638  # Hartree -> kJ/mol
    sfx = _suffix(per_conformer)

    df = df.copy()
    df['EA_kJ_abs'] = df['EA_kJ'].abs()

    # Convert HOMO/LUMO from Hartree to kJ/mol (Koopmans: -epsilon)
    # (Column names say _eV but AMS Properties/HOMO returns Hartree)
    df['neg_HOMO_kJ'] = -df['HOMO_eV'] * EH_TO_KJ
    df['neg_LUMO_kJ'] = -df['LUMO_eV'] * EH_TO_KJ

    color_map = get_molecule_color_map(df['molecule_name']) if per_conformer else None

    # --- Helper function for scatter + fit ---
    def _scatter_fit(ax, valid, x_col, y_col, color, label_prefix=""):
        if per_conformer:
            for mol_name, group in valid.groupby('molecule_name'):
                ax.scatter(group[x_col], group[y_col], s=60, alpha=0.7,
                           color=color_map[mol_name], edgecolors='white',
                           linewidths=0.5, zorder=3, label=mol_name)
            for _, row in valid.iterrows():
                ax.annotate(str(int(row['conformer'])), (row[x_col], row[y_col]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=6, alpha=0.8)
        else:
            ax.scatter(valid[x_col], valid[y_col], s=90, alpha=0.7,
                       color=color, edgecolors='white', linewidths=0.5, zorder=3)
            for _, row in valid.iterrows():
                ax.annotate(row['molecule_name'], (row[x_col], row[y_col]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.8)
        slope, intercept, r_value, p_value, _ = linregress(valid[x_col], valid[y_col])
        x_fit = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        y_fit = slope * x_fit + intercept
        # MAE = erreur de prédiction Koopmans (vs droite y=x)
        mae = np.mean(np.abs(valid[y_col].values - valid[x_col].values))
        ax.plot(x_fit, y_fit, 'k--', linewidth=2,
                label=f'{label_prefix}MAE $= {mae:.1f}$ kJ/mol, $R^2 = {r_value**2:.3f}$\n$y = {slope:.1f}x {intercept:+.1f}$')
        return r_value**2

    # --- Helper to select valid columns ---
    def _valid_cols(cols):
        base = ['molecule_name']
        if per_conformer:
            base.append('conformer')
        return base + [c for c in cols if c not in base]

    # --- Combined figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pairs = [
        ('neg_LUMO_kJ', 'EA_kJ_abs', '$-\\varepsilon_{LUMO}$ (kJ/mol)', '|EA| (kJ/mol)',
         'Affinité Électronique vs $-\\varepsilon_{LUMO}$ (Koopmans)', '#c0392b'),
        ('neg_HOMO_kJ', 'IP_kJ', '$-\\varepsilon_{HOMO}$ (kJ/mol)', 'IP (kJ/mol)',
         'Potentiel d\'Ionisation vs $-\\varepsilon_{HOMO}$ (Koopmans)', '#2980b9'),
    ]
    for ax, (x_col, y_col, x_label, y_label, title, color) in zip(axes, pairs):
        valid = df[_valid_cols([x_col, y_col])].dropna(subset=[x_col, y_col])
        if len(valid) < 3:
            continue
        _scatter_fit(ax, valid, x_col, y_col, color)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if per_conformer:
            ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
        else:
            ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    out = output_dir / f"koopmans_EA_LUMO_IP_HOMO{sfx}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure 2 sauvegardée: {out.name}")

    # --- Individual EA vs LUMO ---
    fig, ax = plt.subplots(figsize=(8, 6) if per_conformer else (7, 5.5))
    valid = df[_valid_cols(['neg_LUMO_kJ', 'EA_kJ_abs'])].dropna(subset=['neg_LUMO_kJ', 'EA_kJ_abs'])
    _scatter_fit(ax, valid, 'neg_LUMO_kJ', 'EA_kJ_abs', '#c0392b')
    ax.set_xlabel('$-\\varepsilon_{LUMO}$ (kJ/mol)', fontsize=11)
    ax.set_ylabel('|EA| (kJ/mol)', fontsize=11)
    ax.set_title('Affinité Électronique vs $-\\varepsilon_{LUMO}$', fontsize=12, fontweight='bold')
    if per_conformer:
        ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
    else:
        ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_dir / f"EA_vs_LUMO{sfx}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Individuel: EA_vs_LUMO{sfx}.png")

    # --- Individual IP vs HOMO (all molecules) ---
    fig, ax = plt.subplots(figsize=(8, 6) if per_conformer else (7, 5.5))
    valid_all = df[_valid_cols(['neg_HOMO_kJ', 'IP_kJ'])].dropna(subset=['neg_HOMO_kJ', 'IP_kJ'])
    _scatter_fit(ax, valid_all, 'neg_HOMO_kJ', 'IP_kJ', '#2980b9')
    ax.set_xlabel('$-\\varepsilon_{HOMO}$ (kJ/mol)', fontsize=11)
    ax.set_ylabel('IP (kJ/mol)', fontsize=11)
    ax.set_title('Potentiel d\'Ionisation vs $-\\varepsilon_{HOMO}$ (toutes molécules)', fontsize=12, fontweight='bold')
    if per_conformer:
        ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
    else:
        ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_dir / f"IP_vs_HOMO{sfx}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Individuel: IP_vs_HOMO{sfx}.png")

    # --- EA vs LUMO WITHOUT Br-HMC outlier ---
    outliers = []
    valid_ea = df[_valid_cols(['neg_LUMO_kJ', 'EA_kJ_abs'])].dropna(subset=['neg_LUMO_kJ', 'EA_kJ_abs'])
    valid_ea_filtered = valid_ea[~valid_ea['molecule_name'].isin(outliers)].copy()
    if len(valid_ea_filtered) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6) if per_conformer else (7, 5.5))
        # Plot outliers in grey first
        ea_outliers = valid_ea[valid_ea['molecule_name'].isin(outliers)]
        if len(ea_outliers) > 0:
            ax.scatter(ea_outliers['neg_LUMO_kJ'], ea_outliers['EA_kJ_abs'],
                       s=90, alpha=0.4, color='grey', edgecolors='white',
                       linewidths=0.5, zorder=2, marker='x')
            for _, row in ea_outliers.iterrows():
                label_txt = row['molecule_name']
                if per_conformer:
                    label_txt += f"_{int(row['conformer'])}"
                label_txt += ' (exclu)'
                ax.annotate(label_txt,
                            (row['neg_LUMO_kJ'], row['EA_kJ_abs']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.5, fontstyle='italic')
        # Plot and fit without outliers
        _scatter_fit(ax, valid_ea_filtered, 'neg_LUMO_kJ', 'EA_kJ_abs', '#c0392b')
        ax.set_xlabel('$-\\varepsilon_{LUMO}$ (kJ/mol)', fontsize=11)
        ax.set_ylabel('|EA| (kJ/mol)', fontsize=11)
        ax.set_title('Affinité Électronique vs $-\\varepsilon_{LUMO}$',
                      fontsize=12, fontweight='bold')
        if per_conformer:
            ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
        else:
            ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout()
        plt.savefig(output_dir / f"EA_vs_LUMO_sans_outliers{sfx}.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Individuel: EA_vs_LUMO_sans_outliers{sfx}.png")

    # --- IP vs HOMO WITHOUT Br-HMC outlier ---
    valid_filtered = valid_all[~valid_all['molecule_name'].isin(outliers)].copy()
    if len(valid_filtered) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6) if per_conformer else (7, 5.5))
        # Plot outliers in grey first
        valid_outliers = valid_all[valid_all['molecule_name'].isin(outliers)]
        if len(valid_outliers) > 0:
            ax.scatter(valid_outliers['neg_HOMO_kJ'], valid_outliers['IP_kJ'],
                       s=90, alpha=0.4, color='grey', edgecolors='white',
                       linewidths=0.5, zorder=2, marker='x')
            for _, row in valid_outliers.iterrows():
                label_txt = row['molecule_name']
                if per_conformer:
                    label_txt += f"_{int(row['conformer'])}"
                label_txt += ' (exclu)'
                ax.annotate(label_txt,
                            (row['neg_HOMO_kJ'], row['IP_kJ']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.5, fontstyle='italic')
        # Plot and fit without outliers
        _scatter_fit(ax, valid_filtered, 'neg_HOMO_kJ', 'IP_kJ', '#2980b9')
        ax.set_xlabel('$-\\varepsilon_{HOMO}$ (kJ/mol)', fontsize=11)
        ax.set_ylabel('IP (kJ/mol)', fontsize=11)
        ax.set_title('Potentiel d\'Ionisation vs $-\\varepsilon_{HOMO}$',
                      fontsize=12, fontweight='bold')
        if per_conformer:
            ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
        else:
            ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout()
        plt.savefig(output_dir / f"IP_vs_HOMO_sans_outliers{sfx}.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Individuel: IP_vs_HOMO_sans_outliers{sfx}.png")

        # Print R² for both
        r_all, _ = pearsonr(valid_all['neg_HOMO_kJ'], valid_all['IP_kJ'])
        r_filt, _ = pearsonr(valid_filtered['neg_HOMO_kJ'], valid_filtered['IP_kJ'])
        print(f"    IP vs HOMO: R²={r_all**2:.3f} (tout), R²={r_filt**2:.3f}")

        r_ea_all, _ = pearsonr(valid_ea['neg_LUMO_kJ'], valid_ea['EA_kJ_abs'])
        r_ea_filt, _ = pearsonr(valid_ea_filtered['neg_LUMO_kJ'], valid_ea_filtered['EA_kJ_abs'])
        print(f"    EA vs LUMO: R²={r_ea_all**2:.3f} (tout), R²={r_ea_filt**2:.3f}")


# =============================================================================
# Figure 3 & 4 : Heatmap + top corrélations pour ΔΔU et TΔS
# =============================================================================

def plot_heatmap(df, descriptors, targets, output_file, title=None):
    """
    Heatmap R² entre descripteurs (lignes) et cibles (colonnes).
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
    fig, ax = plt.subplots(figsize=(max(6, len(target_cols) * 2.5), fig_height))

    try:
        import seaborn as sns
        sns.heatmap(
            r2_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, linewidths=0.5,
            xticklabels=target_labels, yticklabels=desc_labels,
            cbar_kws={"shrink": 0.7, "label": "$R^2$"}, ax=ax
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

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap sauvegardée: {output_file.name}")


def plot_top_correlations_grid(df, corr_results, output_file, top_n=8,
                                target_label_short="", per_conformer=False):
    """
    Grille des top N meilleures corrélations (prédit vs réel).
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
            needed = ['molecule_name', 'conformer', desc_col, target_col]
        else:
            needed = ['molecule_name', desc_col, target_col]
        valid = df[[c for c in needed if c in df.columns]].dropna(subset=[desc_col, target_col])

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
        ax.set_xlabel(f'{target_label_short} prédit (kJ/mol)', fontsize=8)
        ax.set_ylabel(f'{target_label_short} réel (kJ/mol)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, zorder=0)

    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    suptitle = f'Top {n} corrélations avec {target_label_short}'
    if per_conformer:
        suptitle += ' [per-conformère]'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

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
    print(f"  Grille top {n} ({target_label_short}): {output_file.name}")


# =============================================================================
# Figure 5 : Heatmap de synthèse combinée
# =============================================================================

def plot_combined_synthesis_heatmap(df, output_dir):
    """
    Heatmap combinée avec les 8 cibles en colonnes et les descripteurs clés en lignes.
    """
    # Sélection des descripteurs clés
    key_descriptors = {
        'HOMO_eV': 'HOMO (eV)',
        'LUMO_eV': 'LUMO (eV)',
        'gap_eV': 'Gap (eV)',
        'EA_kJ': 'EA (kJ/mol)',
        'IP_kJ': 'IP (kJ/mol)',
        'O7_bader': 'q(O7)',
        'O8_bader': 'q(O8)',
        'C6_bader': 'q(C6)',
        'q_O_total': 'q(O7+O8)',
        'O7_iqa': 'IQA(O7)',
        'O8_iqa': 'IQA(O8)',
        'C6_iqa': 'IQA(C6)',
        'freq_mean': 'freq moy.',
        'freq_max': 'freq max',
        'freq_std': 'freq σ',
        'flexibility_index': 'indice flex.',
        'stiffness_index': 'indice rigid.',
        'zpe_kj': 'ZPE (kJ/mol)',
        'n_modes_low': 'N modes 0-500',
        'ir_max': 'IR max',
    }

    # Les 8 cibles
    all_targets = {
        'EA_kJ': 'EA',
        'IP_kJ': 'IP',
        'Edef_red_kJ': '$E_{relax}^{red}$',
        'Edef_ox_kJ': '$E_{relax}^{ox}$',
        'ddU_red_kJ': '$\\Delta\\Delta U^{red}$',
        'ddU_ox_kJ': '$\\Delta\\Delta U^{ox}$',
        'TdS_red_kJ': '$T\\Delta S^{red}$',
        'TdS_ox_kJ': '$T\\Delta S^{ox}$',
    }

    # Filtrer descripteurs et cibles présents
    desc_cols = [c for c in key_descriptors if c in df.columns]
    target_cols = [c for c in all_targets if c in df.columns]

    # Exclure les auto-corrélations (EA vs EA_kJ, IP vs IP_kJ)
    r2_matrix = np.full((len(desc_cols), len(target_cols)), np.nan)

    for i, dc in enumerate(desc_cols):
        for j, tc in enumerate(target_cols):
            # Skip auto-correlation
            if dc == tc:
                r2_matrix[i, j] = np.nan
                continue
            valid = df[[dc, tc]].dropna()
            if len(valid) >= 3:
                r, _ = pearsonr(valid[dc], valid[tc])
                r2_matrix[i, j] = r ** 2

    desc_labels = [key_descriptors[c] for c in desc_cols]
    target_labels = [all_targets[c] for c in target_cols]

    fig_height = max(8, len(desc_cols) * 0.5 + 2)
    fig_width = max(10, len(target_cols) * 1.5 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    try:
        import seaborn as sns
        sns.heatmap(
            r2_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, linewidths=0.5,
            xticklabels=target_labels, yticklabels=desc_labels,
            cbar_kws={"shrink": 0.7, "label": "$R^2$"}, ax=ax,
            mask=np.isnan(r2_matrix)
        )
    except ImportError:
        masked = np.ma.masked_invalid(r2_matrix)
        im = ax.imshow(masked, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
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
                            ha='center', va='center', color=color, fontsize=8)

    ax.set_title('Synthèse : $R^2$ des corrélations descripteurs vs termes de $\\Delta G$',
                  fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    out = output_dir / "combined_decomposition_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure 5 (synthèse) sauvegardée: {out.name}")


# =============================================================================
# Tableau des valeurs ΔΔU et TΔS
# =============================================================================

def export_decomposition_table(df, output_dir):
    """Exporte un tableau récapitulatif des valeurs ΔΔU et TΔS."""
    cols = ['molecule_name', 'Res_red_kJ', 'Res_ox_kJ']
    if 'TdS_red_kJ' in df.columns:
        cols.extend(['TdS_red_kJ', 'TdS_ox_kJ'])
    if 'ddU_red_kJ' in df.columns:
        cols.extend(['ddU_red_kJ', 'ddU_ox_kJ'])

    available = [c for c in cols if c in df.columns]
    table_df = df[available].copy()
    table_df.to_csv(output_dir / "decomposition_ddU_TdS.csv", index=False)

    txt_file = output_dir / "decomposition_ddU_TdS.txt"
    with open(txt_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("DÉCOMPOSITION : Résiduel, TΔS et ΔΔU par molécule\n")
        f.write("=" * 90 + "\n\n")

        header = f"{'Molécule':<12}"
        header += f"{'Rés_red':>10} {'Rés_ox':>10}"
        if 'TdS_red_kJ' in df.columns:
            header += f" {'TΔS_red':>10} {'TΔS_ox':>10}"
        if 'ddU_red_kJ' in df.columns:
            header += f" {'ΔΔU_red':>10} {'ΔΔU_ox':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for _, row in df.iterrows():
            line = f"{row['molecule_name']:<12}"
            line += f"{row['Res_red_kJ']:>10.2f} {row['Res_ox_kJ']:>10.2f}"
            if 'TdS_red_kJ' in df.columns:
                line += f" {row['TdS_red_kJ']:>10.2f} {row['TdS_ox_kJ']:>10.2f}"
            if 'ddU_red_kJ' in df.columns:
                line += f" {row['ddU_red_kJ']:>10.2f} {row['ddU_ox_kJ']:>10.2f}"
            f.write(line + "\n")

    print(f"  Tableau exporté: {txt_file.name}")


# =============================================================================
# Main
# =============================================================================

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


def main():
    parser = argparse.ArgumentParser(
        description="Corrélations pour les 4 termes de la décomposition de ΔG"
    )
    parser.add_argument('--csv', type=str, default=None,
                        help="CSV principal (molecules_summary_with_freq.csv)")
    parser.add_argument('--tds-csv', type=str, default=None,
                        help="CSV des termes TΔS (thermodynamic_terms.csv)")
    parser.add_argument('--batch-dir', type=str, default=None,
                        help="Répertoire des workdirs (fallback pour TΔS)")
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'full_decomposition'),
                        help="Répertoire de sortie")
    parser.add_argument('--top-n', type=int, default=8,
                        help="Nombre de top corrélations")
    parser.add_argument('--per-conformer', action='store_true', default=False,
                        help="Afficher tous les conformeres individuellement")
    parser.add_argument('--freq-csv', type=str, default=None,
                        help="CSV des descripteurs frequentiels per-conformer")

    args = parser.parse_args()

    if not args.csv and not args.batch_dir:
        parser.error("--csv ou --batch-dir requis")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sfx = _suffix(args.per_conformer)

    print("=" * 80)
    print("CORRÉLATIONS POUR LES 4 TERMES DE LA DÉCOMPOSITION DE ΔG")
    if args.per_conformer:
        print("  Mode: PER-CONFORMÈRE (1 point par conformère)")
    print("=" * 80)

    # 1. Charger les données principales
    if args.per_conformer and args.batch_dir and not args.csv:
        mod = _load_multi_molecule_module()
        workdirs = sorted([d for d in Path(args.batch_dir).glob("*_workdir") if d.is_dir()])
        if not workdirs:
            print(f"Aucun workdir trouvé dans {args.batch_dir}")
            sys.exit(1)
        print(f"{len(workdirs)} workdirs trouvés (mode per-conformère)")
        df = mod.collect_all_conformers(workdirs, freq_csv=args.freq_csv)
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        print("ERREUR: --csv requis (ou --batch-dir avec --per-conformer)")
        sys.exit(1)

    # Exclure les molécules DEA (rupture C-X lors de la réduction)
    if 'molecule_name' in df.columns:
        n_before = len(df)
        df = df[~df['molecule_name'].isin(DEA_EXCLUDED)]
        n_excl = n_before - len(df)
        if n_excl > 0:
            print(f"  Exclusion DEA: {n_excl} entrées retirées ({', '.join(sorted(DEA_EXCLUDED))})")

    if args.per_conformer:
        n_mols = df['molecule_name'].nunique()
        print(f"\nDonnées chargées: {len(df)} conformères ({n_mols} molécules), {len(df.columns)} colonnes")
    else:
        print(f"\nDonnées chargées: {len(df)} molécules, {len(df.columns)} colonnes")

    # 2. Calculer les résiduels
    # Convention analysis.py: EA = Ee(neutre) - Ee(anion_sp) > 0 (positif)
    # ΔG_red = -EA + Edef + ΔΔU + TΔS  =>  Rés = ΔG_red + EA - Edef
    df['Res_red_kJ'] = df['dG_reduction_kJ'] + df['EA_kJ'] - df['Edef_red_kJ']
    # IP = Ee(cation_sp) - Ee(neutre) > 0 (positif)
    # ΔG_ox = IP + Edef + ΔΔU + TΔS  =>  Rés = ΔG_ox - IP - Edef
    df['Res_ox_kJ'] = df['dG_oxidation_kJ'] - df['IP_kJ'] - df['Edef_ox_kJ']

    print(f"\nRésiduels calculés:")
    print(f"  Rés_red: [{df['Res_red_kJ'].min():.1f}, {df['Res_red_kJ'].max():.1f}] kJ/mol")
    print(f"  Rés_ox:  [{df['Res_ox_kJ'].min():.1f}, {df['Res_ox_kJ'].max():.1f}] kJ/mol")

    # 3. Charger TΔS
    tds_loaded = False

    if args.tds_csv and Path(args.tds_csv).exists():
        tds_df = pd.read_csv(args.tds_csv)
        # Ne merger que les colonnes TΔS (et ddU si présentes) pour éviter les conflits
        # avec les colonnes déjà calculées (Res_red_kJ, etc.)
        tds_cols = ['molecule_name']
        for col in ['TdS_red_kJ', 'TdS_ox_kJ', 'ddU_red_kJ', 'ddU_ox_kJ']:
            if col in tds_df.columns and col not in df.columns:
                tds_cols.append(col)
        if len(tds_cols) > 1:
            df = df.merge(tds_df[tds_cols], on='molecule_name', how='left')
        tds_loaded = True
        print(f"\nTΔS chargé depuis {args.tds_csv}: {tds_df.shape[0]} molécules")

    if not tds_loaded and args.batch_dir:
        print(f"\nTΔS: parsing fallback depuis analysis.txt dans {args.batch_dir}...")
        tds_df = parse_tds_from_analysis_txt(args.batch_dir)
        if tds_df is not None and len(tds_df) > 0:
            tds_cols = ['molecule_name']
            for col in ['TdS_red_kJ', 'TdS_ox_kJ', 'ddU_red_kJ', 'ddU_ox_kJ']:
                if col in tds_df.columns and col not in df.columns:
                    tds_cols.append(col)
            if len(tds_cols) > 1:
                df = df.merge(tds_df[tds_cols], on='molecule_name', how='left')
            tds_loaded = True
            print(f"  ✓ TΔS parsé pour {len(tds_df)} molécules")

    if not tds_loaded:
        # Try default path
        default_tds = Path(args.output_dir) / "thermodynamic_terms.csv"
        if default_tds.exists():
            tds_df = pd.read_csv(default_tds)
            tds_cols = ['molecule_name']
            for col in ['TdS_red_kJ', 'TdS_ox_kJ', 'ddU_red_kJ', 'ddU_ox_kJ']:
                if col in tds_df.columns and col not in df.columns:
                    tds_cols.append(col)
            if len(tds_cols) > 1:
                df = df.merge(tds_df[tds_cols], on='molecule_name', how='left')
            tds_loaded = True
            print(f"\nTΔS chargé depuis {default_tds}")

    # 4. Calculer ΔΔU si TΔS est disponible et ΔΔU pas encore dans le df
    # Formule correcte : ΔG = -EA + Erelax + ΔΔU - TΔS
    # Donc Rés = ΔΔU - TΔS => ΔΔU = Rés + TΔS
    if tds_loaded and 'TdS_red_kJ' in df.columns:
        if 'ddU_red_kJ' not in df.columns:
            df['ddU_red_kJ'] = df['Res_red_kJ'] + df['TdS_red_kJ']
            df['ddU_ox_kJ'] = df['Res_ox_kJ'] + df['TdS_ox_kJ']
            print(f"\nΔΔU calculé (Rés + TΔS, car ΔG = -EA + Erelax + ΔΔU - TΔS):")
        else:
            print(f"\nΔΔU chargé directement depuis le CSV:")
        print(f"  ΔΔU_red: [{df['ddU_red_kJ'].min():.1f}, {df['ddU_red_kJ'].max():.1f}] kJ/mol")
        print(f"  ΔΔU_ox:  [{df['ddU_ox_kJ'].min():.1f}, {df['ddU_ox_kJ'].max():.1f}] kJ/mol")
        print(f"  TΔS_red: [{df['TdS_red_kJ'].min():.1f}, {df['TdS_red_kJ'].max():.1f}] kJ/mol")
        print(f"  TΔS_ox:  [{df['TdS_ox_kJ'].min():.1f}, {df['TdS_ox_kJ'].max():.1f}] kJ/mol")
    else:
        print("\n⚠ TΔS non disponible → ΔΔU ne peut pas être calculé séparément")
        print("  Les figures 3-5 utiliseront le résiduel (ΔΔU + TΔS) comme cible")

    # Aplatir tous les descripteurs
    all_descriptors = {}
    for group_descs in DESCRIPTOR_GROUPS.values():
        all_descriptors.update(group_descs)
    present_descriptors = {k: v for k, v in all_descriptors.items() if k in df.columns}
    print(f"\nDescripteurs disponibles: {len(present_descriptors)}")

    # =========================================================================
    # FIGURE 1 : Poids relatif des termes (skip en mode per-conformer)
    # =========================================================================
    if not args.per_conformer:
        print("\n" + "-" * 80)
        print("FIGURE 1 : POIDS RELATIF DES TERMES")
        print("-" * 80)
        plot_decomposition_magnitude_bars(df, output_dir)
    else:
        print("\n  (Figure 1 : barres empilées ignorée en mode per-conformère)")

    # =========================================================================
    # FIGURE 2 : EA vs LUMO, IP vs HOMO (Koopmans)
    # =========================================================================
    print("\n" + "-" * 80)
    print("FIGURE 2 : KOOPMANS (EA vs LUMO, IP vs HOMO)")
    print("-" * 80)
    plot_koopmans(df, output_dir, per_conformer=args.per_conformer)

    # =========================================================================
    # FIGURES 3 & 4 : Corrélations ΔΔU et TΔS
    # =========================================================================
    if tds_loaded and 'ddU_red_kJ' in df.columns:
        # --- Figure 3 : ΔΔU ---
        print("\n" + "-" * 80)
        print("FIGURE 3 : CORRÉLATIONS ΔΔU")
        print("-" * 80)

        ddU_targets = {
            'ddU_red_kJ': '$\\Delta\\Delta U$ réduction (kJ/mol)',
            'ddU_ox_kJ': '$\\Delta\\Delta U$ oxydation (kJ/mol)',
        }
        plot_heatmap(df, present_descriptors, ddU_targets,
                     output_dir / f"heatmap_ddU{sfx}.png",
                     title='$R^2$ : Descripteurs vs $\\Delta\\Delta U$')

        ddU_corr = compute_all_correlations(df, present_descriptors, ddU_targets)
        for target_col, _ in ddU_targets.items():
            process = "réduction" if "red" in target_col else "oxydation"
            safe = "red" if "red" in target_col else "ox"
            target_corr = [r for r in ddU_corr if r['target'] == target_col]
            plot_top_correlations_grid(
                df, target_corr,
                output_dir / f"top_correlations_ddU_{safe}{sfx}.png",
                top_n=args.top_n,
                target_label_short=f"$\\Delta\\Delta U$ ({process})",
                per_conformer=args.per_conformer
            )

        # --- Figure 4 : TΔS ---
        print("\n" + "-" * 80)
        print("FIGURE 4 : CORRÉLATIONS TΔS")
        print("-" * 80)

        tds_targets = {
            'TdS_red_kJ': '$T\\Delta S$ réduction (kJ/mol)',
            'TdS_ox_kJ': '$T\\Delta S$ oxydation (kJ/mol)',
        }
        plot_heatmap(df, present_descriptors, tds_targets,
                     output_dir / f"heatmap_TdS{sfx}.png",
                     title='$R^2$ : Descripteurs vs $T\\Delta S$')

        tds_corr = compute_all_correlations(df, present_descriptors, tds_targets)
        for target_col, _ in tds_targets.items():
            process = "réduction" if "red" in target_col else "oxydation"
            safe = "red" if "red" in target_col else "ox"
            target_corr = [r for r in tds_corr if r['target'] == target_col]
            plot_top_correlations_grid(
                df, target_corr,
                output_dir / f"top_correlations_TdS_{safe}{sfx}.png",
                top_n=args.top_n,
                target_label_short=f"$T\\Delta S$ ({process})",
                per_conformer=args.per_conformer
            )

        # --- Seconde passe : sans Br-HMC (outlier connu) ---
        outliers = []
        df_filtered = df[~df['molecule_name'].isin(outliers)].copy()
        n_excluded = len(df) - len(df_filtered)
        if n_excluded > 0 and len(df_filtered) >= 3:
            print(f"\n" + "-" * 80)
            print(f"FIGURES 3b & 4b : CORRÉLATIONS ΔΔU / TΔS SANS {', '.join(outliers)} ({n_excluded} points exclus)")
            print("-" * 80)

            # ΔΔU sans outlier
            ddU_corr_filt = compute_all_correlations(df_filtered, present_descriptors, ddU_targets)
            for target_col, _ in ddU_targets.items():
                process = "réduction" if "red" in target_col else "oxydation"
                safe = "red" if "red" in target_col else "ox"
                target_corr = [r for r in ddU_corr_filt if r['target'] == target_col]
                plot_top_correlations_grid(
                    df_filtered, target_corr,
                    output_dir / f"top_correlations_ddU_{safe}_sans_outliers{sfx}.png",
                    top_n=args.top_n,
                    target_label_short=f"$\\Delta\\Delta U$ ({process}, sans Br-HMC)",
                    per_conformer=args.per_conformer
                )

            # TΔS sans outlier
            tds_corr_filt = compute_all_correlations(df_filtered, present_descriptors, tds_targets)
            for target_col, _ in tds_targets.items():
                process = "réduction" if "red" in target_col else "oxydation"
                safe = "red" if "red" in target_col else "ox"
                target_corr = [r for r in tds_corr_filt if r['target'] == target_col]
                plot_top_correlations_grid(
                    df_filtered, target_corr,
                    output_dir / f"top_correlations_TdS_{safe}_sans_outliers{sfx}.png",
                    top_n=args.top_n,
                    target_label_short=f"$T\\Delta S$ ({process}, sans Br-HMC)",
                    per_conformer=args.per_conformer
                )

            # Afficher les R² comparatifs
            print(f"\n  Comparaison R² (avec/sans Br-HMC) :")
            for target_col in ['ddU_red_kJ', 'ddU_ox_kJ', 'TdS_red_kJ', 'TdS_ox_kJ']:
                all_corr = ddU_corr if 'ddU' in target_col else tds_corr
                filt_corr = ddU_corr_filt if 'ddU' in target_col else tds_corr_filt
                best_all = [r for r in all_corr if r['target'] == target_col]
                best_filt = [r for r in filt_corr if r['target'] == target_col]
                if best_all and best_filt:
                    print(f"    {target_col}: top desc = {best_all[0]['descriptor']}")
                    print(f"      Toutes: R²={best_all[0]['R2']:.3f} (n={best_all[0]['n']})")
                    print(f"      Filtrées: R²={best_filt[0]['R2']:.3f} (n={best_filt[0]['n']})")
    else:
        # Fallback : utiliser le résiduel comme cible
        print("\n" + "-" * 80)
        print("FIGURES 3-4 : CORRÉLATIONS RÉSIDUEL (ΔΔU + TΔS)")
        print("-" * 80)

        res_targets = {
            'Res_red_kJ': 'Résiduel réduction (kJ/mol)',
            'Res_ox_kJ': 'Résiduel oxydation (kJ/mol)',
        }
        plot_heatmap(df, present_descriptors, res_targets,
                     output_dir / f"heatmap_residuel{sfx}.png",
                     title='$R^2$ : Descripteurs vs Résiduel ($\\Delta\\Delta U + T\\Delta S$)')

        res_corr = compute_all_correlations(df, present_descriptors, res_targets)
        for target_col, _ in res_targets.items():
            process = "réduction" if "red" in target_col else "oxydation"
            safe = "red" if "red" in target_col else "ox"
            target_corr = [r for r in res_corr if r['target'] == target_col]
            plot_top_correlations_grid(
                df, target_corr,
                output_dir / f"top_correlations_residuel_{safe}{sfx}.png",
                top_n=args.top_n,
                target_label_short=f"Résiduel ({process})",
                per_conformer=args.per_conformer
            )

    # =========================================================================
    # FIGURE 5 : Heatmap de synthèse combinée
    # =========================================================================
    print("\n" + "-" * 80)
    print("FIGURE 5 : HEATMAP DE SYNTHÈSE")
    print("-" * 80)
    plot_combined_synthesis_heatmap(df, output_dir)

    # =========================================================================
    # Tableau récapitulatif
    # =========================================================================
    if not args.per_conformer:
        print("\n" + "-" * 80)
        print("TABLEAU RÉCAPITULATIF")
        print("-" * 80)
        export_decomposition_table(df, output_dir)

    # =========================================================================
    # Sauvegarder le CSV enrichi
    # =========================================================================
    enriched_csv = output_dir / f"molecules_summary_full_decomposition{sfx}.csv"
    df.to_csv(enriched_csv, index=False)
    print(f"\nCSV enrichi sauvegardé: {enriched_csv}")

    # =========================================================================
    # Copier les figures dans results/figures/ pour le LaTeX
    # =========================================================================
    if not args.per_conformer:
        rapport_fig_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
        if rapport_fig_dir.exists():
            import shutil
            figures_to_copy = [
                "decomposition_magnitude_bars.png",
                "EA_vs_LUMO.png",
                "EA_vs_LUMO_sans_outliers.png",
                "IP_vs_HOMO.png",
                "IP_vs_HOMO_sans_outliers.png",
                "combined_decomposition_heatmap.png",
            ]
            if tds_loaded:
                figures_to_copy.extend([
                    "heatmap_ddU.png",
                    "heatmap_TdS.png",
                    "top_correlations_ddU_red.png",
                    "top_correlations_ddU_ox.png",
                    "top_correlations_TdS_red.png",
                    "top_correlations_TdS_ox.png",
                    "top_correlations_ddU_red_sans_outliers.png",
                    "top_correlations_ddU_ox_sans_outliers.png",
                    "top_correlations_TdS_red_sans_outliers.png",
                    "top_correlations_TdS_ox_sans_outliers.png",
                ])
            else:
                figures_to_copy.extend([
                    "heatmap_residuel.png",
                    "top_correlations_residuel_red.png",
                    "top_correlations_residuel_ox.png",
                ])

            for fname in figures_to_copy:
                src = output_dir / fname
                if src.exists():
                    shutil.copy2(src, rapport_fig_dir / fname)
            print(f"\nFigures copiées dans {rapport_fig_dir}/")

    # =========================================================================
    # Résumé
    # =========================================================================
    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"\nRésultats sauvegardés dans: {output_dir}/")
    print(f"CSV enrichi: {enriched_csv}")
    if args.per_conformer:
        print(f"Mode: per-conformère ({len(df)} points)")
    if tds_loaded:
        print("TΔS et ΔΔU : calculés séparément")
    else:
        print("TΔS et ΔΔU : fusionnés comme résiduel")
    print("=" * 80)


if __name__ == "__main__":
    main()
