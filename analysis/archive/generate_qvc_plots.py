#!/usr/bin/env python3
"""
Génère des graphiques comparant LVC et QVC pour le rapport.

Lit les données depuis lvc_report_extended/all_methods_comparison.txt
et produit des scatter plots QVC vs réel + comparaison LVC/QVC.

Usage:
    python3 generate_qvc_plots.py [--output-dir rapport_figures]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def parse_comparison_table(filepath):
    """Parse all_methods_comparison.txt et retourne les données par processus."""
    data = {'reduction': [], 'oxidation': []}
    current_process = None

    with open(filepath, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if 'RÉDUCTION' in line_stripped:
                current_process = 'reduction'
                continue
            elif 'OXYDATION' in line_stripped:
                current_process = 'oxidation'
                continue
            elif 'STATISTIQUES GLOBALES' in line_stripped:
                break

            if current_process is None:
                continue
            if line_stripped.startswith('---') or line_stripped.startswith('==='):
                continue
            if line_stripped.startswith('Molécule') or '(kJ/mol)' in line_stripped:
                continue
            if not line_stripped:
                continue

            parts = line_stripped.split()
            if len(parts) < 8:
                continue

            mol_name = parts[0]
            try:
                real_val = float(parts[1])
                lvc_val = float(parts[2])
                dusch_val = float(parts[3])
                qvc_val = float(parts[4])
            except (ValueError, IndexError):
                continue

            if np.isnan(real_val) or np.isnan(lvc_val) or np.isnan(qvc_val):
                continue

            data[current_process].append({
                'molecule': mol_name,
                'real': real_val,
                'lvc': lvc_val,
                'qvc': qvc_val,
            })

    return data


def plot_qvc_vs_real(data, process, output_path):
    """Scatter plot QVC vs réel (même style que les plots LVC existants)."""
    molecules = [d['molecule'] for d in data]
    real_vals = np.array([d['real'] for d in data])
    qvc_vals = np.array([d['qvc'] for d in data])

    slope, intercept, r_value, p_value, std_err = stats.linregress(real_vals, qvc_vals)
    r2 = r_value**2
    mae = np.mean(np.abs(qvc_vals - real_vals))

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(real_vals, qvc_vals, color='#e74c3c', s=60, zorder=5, edgecolors='white', linewidth=0.5)

    # Ligne y=x
    all_vals = np.concatenate([real_vals, qvc_vals])
    margin = (all_vals.max() - all_vals.min()) * 0.1
    lim_min = all_vals.min() - margin
    lim_max = all_vals.max() + margin
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='gray', linewidth=1, label='y = x')

    # Fit linéaire
    x_fit = np.linspace(lim_min, lim_max, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, '-', color='#e74c3c', linewidth=1.5,
            label=f'Fit: $R^2$ = {r2:.2f}, MAE = {mae:.1f} kJ/mol')

    # Annotations pour molécules remarquables
    for d in data:
        err = abs(d['qvc'] - d['real'])
        if err > 10:
            ax.annotate(d['molecule'], (d['real'], d['qvc']),
                        fontsize=7, ha='left', va='bottom',
                        xytext=(4, 4), textcoords='offset points')

    process_label = 'Oxydation' if process == 'oxidation' else 'Réduction'
    ax.set_xlabel(r'$E_{\mathrm{relax}}$ réel (kJ/mol)', fontsize=11)
    ax.set_ylabel(r'$E_{\mathrm{relax}}^{\mathrm{QVC}}$ (kJ/mol)', fontsize=11)
    ax.set_title(f'QVC vs Réel - {process_label}', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_path}")


def plot_lvc_qvc_comparison(data, process, output_path):
    """Scatter plot comparant LVC et QVC côte à côte."""
    molecules = [d['molecule'] for d in data]
    real_vals = np.array([d['real'] for d in data])
    lvc_vals = np.array([d['lvc'] for d in data])
    qvc_vals = np.array([d['qvc'] for d in data])

    # Stats LVC
    sl, il, rl, pl, _ = stats.linregress(real_vals, lvc_vals)
    r2_lvc = rl**2
    mae_lvc = np.mean(np.abs(lvc_vals - real_vals))

    # Stats QVC
    sq, iq, rq, pq, _ = stats.linregress(real_vals, qvc_vals)
    r2_qvc = rq**2
    mae_qvc = np.mean(np.abs(qvc_vals - real_vals))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    all_vals = np.concatenate([real_vals, lvc_vals, qvc_vals])
    margin = (all_vals.max() - all_vals.min()) * 0.1
    lim_min = all_vals.min() - margin
    lim_max = all_vals.max() + margin

    for ax, method_vals, r2, mae, color, label in [
        (axes[0], lvc_vals, r2_lvc, mae_lvc, '#3498db', 'LVC'),
        (axes[1], qvc_vals, r2_qvc, mae_qvc, '#e74c3c', 'QVC'),
    ]:
        ax.scatter(real_vals, method_vals, color=color, s=60, zorder=5,
                   edgecolors='white', linewidth=0.5)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='gray', linewidth=1)

        sl_m, il_m, _, _, _ = stats.linregress(real_vals, method_vals)
        x_fit = np.linspace(lim_min, lim_max, 100)
        y_fit = sl_m * x_fit + il_m
        ax.plot(x_fit, y_fit, '-', color=color, linewidth=1.5,
                label=f'$R^2$ = {r2:.3f}\nMAE = {mae:.2f} kJ/mol')

        # Annotations
        for d in data:
            val = d['lvc'] if label == 'LVC' else d['qvc']
            err = abs(val - d['real'])
            if err > 10:
                ax.annotate(d['molecule'], (d['real'], val),
                            fontsize=7, ha='left', va='bottom',
                            xytext=(4, 4), textcoords='offset points')

        process_label = 'Oxydation' if process == 'oxidation' else 'Réduction'
        ax.set_xlabel(r'$E_{\mathrm{relax}}$ réel (kJ/mol)', fontsize=11)
        ax.set_ylabel(rf'$E_{{\mathrm{{relax}}}}^{{\mathrm{{{label}}}}}$ (kJ/mol)', fontsize=11)
        ax.set_title(f'{label} - {process_label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_path}")


def plot_correction_factors(data_red, data_ox, output_path):
    """Graphique des facteurs de correction QVC/LVC par molécule."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Facteurs de correction = QVC/LVC
    molecules_red = [d['molecule'] for d in data_red]
    factors_red = [d['qvc'] / d['lvc'] if d['lvc'] != 0 else 1.0 for d in data_red]

    molecules_ox = [d['molecule'] for d in data_ox]
    factors_ox = [d['qvc'] / d['lvc'] if d['lvc'] != 0 else 1.0 for d in data_ox]

    # Combiner les molécules
    all_mols = sorted(set(molecules_red + molecules_ox))
    x = np.arange(len(all_mols))
    width = 0.35

    red_factors = []
    ox_factors = []
    red_map = {d['molecule']: d['qvc'] / d['lvc'] if d['lvc'] != 0 else np.nan for d in data_red}
    ox_map = {d['molecule']: d['qvc'] / d['lvc'] if d['lvc'] != 0 else np.nan for d in data_ox}

    for mol in all_mols:
        red_factors.append(red_map.get(mol, np.nan))
        ox_factors.append(ox_map.get(mol, np.nan))

    bars_red = ax.bar(x - width/2, red_factors, width, color='#3498db', alpha=0.8, label='Réduction')
    bars_ox = ax.bar(x + width/2, ox_factors, width, color='#e74c3c', alpha=0.8, label='Oxydation')

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(all_mols, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Facteur de correction QVC/LVC', fontsize=11)
    ax.set_title('Facteurs de correction quadratique par molécule', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Limites y
    all_f = [f for f in red_factors + ox_factors if not np.isnan(f)]
    if all_f:
        y_min = min(all_f) - 0.005
        y_max = max(all_f) + 0.005
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Graphiques QVC pour le rapport")
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument('--input', type=str,
                        default=str(_root / 'results' / 'lvc_edef' / 'all_methods_comparison.txt'))
    parser.add_argument('--output-dir', type=str, default=str(_root / 'results' / 'figures'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATION DES GRAPHIQUES QVC")
    print("=" * 60)

    # Lire les données
    data = parse_comparison_table(args.input)

    n_red = len(data['reduction'])
    n_ox = len(data['oxidation'])
    print(f"\nDonnees: {n_red} molecules (reduction), {n_ox} molecules (oxydation)")

    # 1. Comparaison LVC vs QVC côte à côte - Oxydation
    print("\n[1] Comparaison LVC vs QVC - Oxydation")
    plot_lvc_qvc_comparison(data['oxidation'], 'oxidation',
                            output_dir / 'lvc_vs_qvc_oxidation.png')

    # 2. Comparaison LVC vs QVC côte à côte - Réduction
    print("[2] Comparaison LVC vs QVC - Réduction")
    plot_lvc_qvc_comparison(data['reduction'], 'reduction',
                            output_dir / 'lvc_vs_qvc_reduction.png')

    # 3. Facteurs de correction QVC/LVC
    print("[3] Facteurs de correction QVC/LVC")
    plot_correction_factors(data['reduction'], data['oxidation'],
                            output_dir / 'qvc_correction_factors.png')

    print(f"\n=> {3} figures generees dans {output_dir}/")


if __name__ == '__main__':
    main()
