#!/usr/bin/env python3
"""
Parity plots IP/EA : DFT vs Koopmans vs Politzer (ALIE).

Génère :
  - IP_parity_per_molecule.png   : IP_DFT vs IP_Koopmans & IP_Politzer (moyennes Boltzmann)
  - EA_parity_per_molecule.png   : EA_DFT vs EA_Koopmans (moyennes Boltzmann)
  - IP_parity_per_conformer.png  : IP per-conformère, color map par molécule
  - metrics_comparison.png       : Barres groupées R², MAE, RMSE pour chaque méthode

Usage:
  python3 generate_ip_ea_comparison.py \\
      --conf-csv politzer_results/politzer_per_conformer.csv \\
      --mol-csv politzer_results/politzer_per_molecule.csv \\
      --output-dir politzer_results
"""

import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from analysis import DEA_EXCLUDED

# Aucun outlier exclu (données corrigées)
OUTLIERS = []


def read_csv(path):
    """Lit un CSV et retourne une liste de dicts avec conversion numérique."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def get_molecule_color_map(molecule_names):
    """Génère une couleur unique par molécule."""
    unique = sorted(set(molecule_names))
    n = max(len(unique), 20)
    try:
        cmap = plt.colormaps.get_cmap('tab20').resampled(n)
    except AttributeError:
        # Anciennes versions de matplotlib (< 3.7)
        cmap = plt.cm.get_cmap('tab20', n)
    return {mol: cmap(i) for i, mol in enumerate(unique)}


def _is_valid(val):
    """Vérifie qu'une valeur est numérique et non-NaN."""
    try:
        return not np.isnan(float(val))
    except (ValueError, TypeError):
        return False


# ============================================================================
# PLOT 1 & 2 : PARITY PLOTS PER-MOLECULE (IP et EA)
# ============================================================================

def _plot_calibrated_parity(ax, x_dft, y_raw, labels, color, marker, method_name, raw_label):
    """Plot calibré : fit y_raw → x_dft, puis scatter des valeurs corrigées."""
    if len(x_dft) < 2:
        return {}

    # Calibration: fit DFT = a * raw + b
    slope_cal, intercept_cal, r_val, p_val, _ = linregress(y_raw, x_dft)
    y_cal = slope_cal * y_raw + intercept_cal
    r2 = r_val ** 2
    mae = np.mean(np.abs(y_cal - x_dft))
    rmse = np.sqrt(np.mean((y_cal - x_dft) ** 2))

    all_vals = np.concatenate([x_dft, y_cal])
    vmin, vmax = np.min(all_vals) - 10, np.max(all_vals) + 10
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, alpha=0.5, label='y = x')

    ax.scatter(x_dft, y_cal, s=80, color=color, marker=marker, edgecolors='white',
               linewidths=0.5, zorder=3)

    for i, mol in enumerate(labels):
        ax.annotate(mol, (x_dft[i], y_cal[i]), xytext=(5, 5),
                   textcoords='offset points', fontsize=6, alpha=0.8, color=color)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')

    ax.legend([ax.get_lines()[0]],
              [f'y = x\n$R^2$={r2:.3f}, MAE={mae:.1f} kJ/mol\n'
               f'$y = {slope_cal:.2f} \\times {raw_label} {intercept_cal:+.0f}$'],
              fontsize=9, loc='upper left', framealpha=0.9)

    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'n': len(x_dft),
            'cal_slope': slope_cal, 'cal_intercept': intercept_cal}


def plot_parity_ip_per_molecule(mol_rows, output_dir):
    """
    Deux panneaux IP calibrés : (a) Koopmans, (b) Politzer.
    Les deux méthodes sont calibrées via régression linéaire → parity sur y=x.
    Couleur unique bleue (#2980b9) pour IP.
    """
    COLOR_IP = '#2980b9'
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    x_k, y_k, labels_k = [], [], []
    x_p, y_p, labels_p = [], [], []

    for row in mol_rows:
        mol = row['molecule_name']
        ip_dft = row.get('IP_kJ')
        ip_koop = row.get('IP_Koopmans_kJ')
        ip_pol = row.get('IP_Politzer_kJ')

        if _is_valid(ip_dft) and _is_valid(ip_koop):
            x_k.append(float(ip_dft))
            y_k.append(float(ip_koop))
            labels_k.append(mol)

        if _is_valid(ip_dft) and _is_valid(ip_pol):
            x_p.append(float(ip_dft))
            y_p.append(float(ip_pol))
            labels_p.append(mol)

    x_k, y_k = np.array(x_k), np.array(y_k)
    x_p, y_p = np.array(x_p), np.array(y_p)

    stats = {}

    # (a) Koopmans calibré
    stats['Koopmans_IP'] = _plot_calibrated_parity(
        axes[0], x_k, y_k, labels_k, COLOR_IP, 'o',
        'Koopmans', '$-\\epsilon_{HOMO}$')
    axes[0].set_xlabel('IP$_{DFT}$ (kJ/mol)', fontsize=12)
    axes[0].set_ylabel('IP$_{Koopmans,cal}$ (kJ/mol)', fontsize=12)
    axes[0].set_title('(a) Koopmans calibré', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, zorder=0)

    # (b) Politzer calibré
    stats['Politzer_IP'] = _plot_calibrated_parity(
        axes[1], x_p, y_p, labels_p, COLOR_IP, '^',
        'Politzer', '$\\bar{I}_{S,min}$')
    axes[1].set_xlabel('IP$_{DFT}$ (kJ/mol)', fontsize=12)
    axes[1].set_ylabel('IP$_{Politzer,cal}$ (kJ/mol)', fontsize=12)
    axes[1].set_title('(b) Politzer calibré', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    out = Path(output_dir) / "IP_parity_per_molecule.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {out.name}")

    return stats


def plot_parity_ea_per_molecule(mol_rows, output_dir):
    """
    Deux panneaux EA calibrés : (a) Koopmans, (b) Politzer.
    Couleur unique rouge (#c0392b) pour EA.
    """
    COLOR_EA = '#c0392b'
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    x_k, y_k, labels_k = [], [], []
    x_v, y_v, labels_v = [], [], []

    for row in mol_rows:
        mol = row['molecule_name']
        ea_dft = row.get('EA_kJ')
        ea_koop = row.get('EA_Koopmans_kJ')
        v_s_min_kj = row.get('V_S_min_kJ')

        if _is_valid(ea_dft) and _is_valid(ea_koop):
            x_k.append(float(ea_dft))
            y_k.append(float(ea_koop))
            labels_k.append(mol)

        if _is_valid(ea_dft) and _is_valid(v_s_min_kj):
            x_v.append(float(ea_dft))
            y_v.append(-float(v_s_min_kj))
            labels_v.append(mol)

    x_k, y_k = np.array(x_k), np.array(y_k)
    x_v, y_v = np.array(x_v), np.array(y_v)

    stats = {}

    # (a) Koopmans calibré
    stats['Koopmans_EA'] = _plot_calibrated_parity(
        axes[0], x_k, y_k, labels_k, COLOR_EA, 'o',
        'Koopmans', '$-\\epsilon_{LUMO}$')
    axes[0].set_xlabel('EA$_{DFT}$ (kJ/mol)', fontsize=12)
    axes[0].set_ylabel('EA$_{Koopmans,cal}$ (kJ/mol)', fontsize=12)
    axes[0].set_title('(a) Koopmans calibré', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, zorder=0)

    # (b) Politzer calibré
    stats['Politzer_EA'] = _plot_calibrated_parity(
        axes[1], x_v, y_v, labels_v, COLOR_EA, 's',
        'Politzer', '$-V_{S,min}$')
    axes[1].set_xlabel('EA$_{DFT}$ (kJ/mol)', fontsize=12)
    axes[1].set_ylabel('EA$_{Politzer,cal}$ (kJ/mol)', fontsize=12)
    axes[1].set_title('(b) Politzer calibré', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    out = Path(output_dir) / "EA_parity_per_molecule.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {out.name}")
    return stats


# ============================================================================
# PLOT 3 : PER-CONFORMER IP SCATTER
# ============================================================================

def plot_ip_per_conformer(conf_rows, output_dir):
    """
    Scatter per-conformère IP : DFT vs Koopmans & Politzer, color map par molécule.
    Outliers grisés.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    mol_names = [r['molecule_name'] for r in conf_rows]
    color_map = get_molecule_color_map(mol_names)

    for ax_idx, (y_col, y_label, title_suffix, marker, default_color) in enumerate([
        ('IP_Koopmans_kJ', 'IP$_{Koopmans}$ (kJ/mol)', 'Koopmans', 'o', '#2980b9'),
        ('IP_Politzer_kJ', 'IP$_{Politzer}$ (kJ/mol)', 'Politzer (ALIE)', '^', '#c0392b'),
    ]):
        ax = axes[ax_idx]

        # Séparer outliers et données normales
        normal_x, normal_y, normal_cols, normal_labels = [], [], [], []
        outlier_x, outlier_y, outlier_labels = [], [], []

        for row in conf_rows:
            mol = row['molecule_name']
            conf = int(row['conformer'])
            ip_dft = row.get('IP_kJ')
            ip_pred = row.get(y_col)

            if not (_is_valid(ip_dft) and _is_valid(ip_pred)):
                continue

            is_outlier = any(mol == m and conf == c for m, c in OUTLIERS)
            if is_outlier:
                outlier_x.append(float(ip_dft))
                outlier_y.append(float(ip_pred))
                outlier_labels.append(f"{mol}\nconf_{conf}")
            else:
                normal_x.append(float(ip_dft))
                normal_y.append(float(ip_pred))
                normal_cols.append(color_map[mol])
                normal_labels.append(f"{conf}")

        # Diagonale
        all_vals = normal_x + normal_y + outlier_x + outlier_y
        if not all_vals:
            continue
        vmin = min(all_vals) - 20
        vmax = max(all_vals) + 20
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, alpha=0.5)

        # Outliers
        if outlier_x:
            ax.scatter(outlier_x, outlier_y, s=80, alpha=0.4, color='grey',
                       marker='x', linewidths=2, zorder=2)
            for i, lbl in enumerate(outlier_labels):
                ax.annotate(lbl, (outlier_x[i], outlier_y[i]),
                           xytext=(8, -8), textcoords='offset points',
                           fontsize=6, alpha=0.6, color='grey', style='italic')

        # Données normales avec couleur par molécule
        if normal_x:
            ax.scatter(normal_x, normal_y, s=60, alpha=0.7, c=normal_cols,
                       marker=marker, edgecolors='white', linewidths=0.5, zorder=3)

            for i, lbl in enumerate(normal_labels):
                ax.annotate(lbl, (normal_x[i], normal_y[i]),
                           xytext=(4, 4), textcoords='offset points',
                           fontsize=5, alpha=0.7)

            # Régression (hors outliers)
            nx, ny = np.array(normal_x), np.array(normal_y)
            slope, intercept, r_val, p_val, _ = linregress(nx, ny)
            r2 = r_val ** 2
            x_fit = np.linspace(vmin, vmax, 100)
            ax.plot(x_fit, slope * x_fit + intercept, '-', color=default_color,
                    linewidth=2, alpha=0.7,
                    label=f'$R^2$={r2:.3f}, n={len(nx)}, p={p_val:.1e}')

        ax.set_xlabel('IP$_{DFT}$ (kJ/mol)', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'IP per-conformère : {title_suffix}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    out = Path(output_dir) / "IP_parity_per_conformer.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {out.name}")


# ============================================================================
# PLOT 4 : TABLEAU DE MÉTRIQUES
# ============================================================================

def plot_metrics_comparison(all_stats, output_dir):
    """
    Barres groupées R², MAE, RMSE pour chaque méthode.
    """
    if not all_stats:
        return

    methods = list(all_stats.keys())
    metrics = ['R2', 'MAE', 'RMSE']
    metric_labels = ['$R^2$', 'MAE (kJ/mol)', 'RMSE (kJ/mol)']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(methods))
    width = 0.6

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [all_stats[m].get(metric, 0) for m in methods]
        bars = axes[i].bar(x, vals, width, color=color, alpha=0.8, edgecolor='white')

        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.3f}' if metric == 'R2' else f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9)

        axes[i].set_ylabel(label, fontsize=11)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        axes[i].grid(True, alpha=0.3, axis='y')

        if metric == 'R2':
            axes[i].set_ylim(0, 1.1)

    fig.suptitle('Comparaison des méthodes de prédiction IP/EA', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path(output_dir) / "metrics_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {out.name}")


# ============================================================================
# FONCTION PUBLIQUE
# ============================================================================

def generate_all_plots(conf_csv: str, mol_csv: str, output_dir: str):
    """
    Génère tous les parity plots depuis les CSV produits par politzer_analysis.py.

    Args:
        conf_csv: Chemin vers politzer_per_conformer.csv
        mol_csv: Chemin vers politzer_per_molecule.csv
        output_dir: Répertoire de sortie pour les PNG
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mol_rows = read_csv(mol_csv)
    conf_rows = read_csv(conf_csv)

    # Exclure les molécules DEA (rupture C-X en réduction)
    mol_rows = [r for r in mol_rows if r.get('molecule_name') not in DEA_EXCLUDED]
    conf_rows = [r for r in conf_rows if r.get('molecule_name') not in DEA_EXCLUDED]

    print(f"  {len(mol_rows)} molécules, {len(conf_rows)} conformères (DEA exclues)")

    # Plot 1 : IP parity per-molecule
    ip_stats = plot_parity_ip_per_molecule(mol_rows, output_dir)

    # Plot 2 : EA parity per-molecule
    ea_stats = plot_parity_ea_per_molecule(mol_rows, output_dir)

    # Plot 3 : IP per-conformer
    plot_ip_per_conformer(conf_rows, output_dir)

    # Plot 4 : Metrics comparison
    all_stats = {**ip_stats, **ea_stats}
    plot_metrics_comparison(all_stats, output_dir)

    print(f"\n  Métriques:")
    for method, stats in all_stats.items():
        print(f"    {method}: R²={stats['R2']:.4f}, MAE={stats['MAE']:.1f}, RMSE={stats['RMSE']:.1f} kJ/mol (n={stats['n']})")


def main():
    parser = argparse.ArgumentParser(
        description="Parity plots IP/EA : DFT vs Koopmans vs Politzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--conf-csv', type=str, required=True,
                        help="CSV per-conformère (politzer_per_conformer.csv)")
    parser.add_argument('--mol-csv', type=str, required=True,
                        help="CSV per-molécule (politzer_per_molecule.csv)")
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'politzer'),
                        help="Répertoire de sortie")

    args = parser.parse_args()
    generate_all_plots(args.conf_csv, args.mol_csv, args.output_dir)


if __name__ == '__main__':
    main()
