#!/usr/bin/env python3
"""
Graphiques per-conformère Politzer (ALIE) dans le style des figures Koopmans.

Produit :
  - IP_vs_ALIE_per_conformer.png   : IP_DFT vs Ī_S,min per-conformère
  - EA_vs_VS_per_conformer.png     : |EA_DFT| vs -V_S,min per-conformère
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from analysis import DEA_EXCLUDED

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONF_CSV = _PROJECT_ROOT / "results" / "politzer" / "politzer_per_conformer.csv"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "politzer"


def read_csv(path):
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


def get_molecule_color_map(rows):
    unique = sorted(set(r['molecule_name'] for r in rows))
    cmap = plt.cm.get_cmap('tab20', max(len(unique), 20))
    return {mol: cmap(i) for i, mol in enumerate(unique)}


def is_valid(val):
    try:
        return not np.isnan(float(val))
    except (ValueError, TypeError):
        return False


def plot_politzer_per_conformer(rows, x_col, y_col, x_label, y_label,
                                 title, output_file):
    """Scatter plot per-conformère dans le style des figures Koopmans 3c-3d."""
    color_map = get_molecule_color_map(rows)

    # Filtrer les lignes valides
    valid_rows = [r for r in rows if is_valid(r.get(x_col)) and is_valid(r.get(y_col))]
    if not valid_rows:
        print(f"  Aucune donnée valide pour {title}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter par molécule
    by_mol = {}
    for r in valid_rows:
        mol = r['molecule_name']
        if mol not in by_mol:
            by_mol[mol] = []
        by_mol[mol].append(r)

    for mol_name in sorted(by_mol.keys()):
        mol_rows = by_mol[mol_name]
        xs = [float(r[x_col]) for r in mol_rows]
        ys = [float(r[y_col]) for r in mol_rows]
        ax.scatter(xs, ys, s=60, alpha=0.7, color=color_map[mol_name],
                   edgecolors='white', linewidths=0.5, zorder=3, label=mol_name)

    # Annotations : numéros de conformère
    for r in valid_rows:
        ax.annotate(str(int(r['conformer'])),
                     (float(r[x_col]), float(r[y_col])),
                     xytext=(4, 4), textcoords='offset points',
                     fontsize=6, alpha=0.8)

    # Régression linéaire
    xs_all = np.array([float(r[x_col]) for r in valid_rows])
    ys_all = np.array([float(r[y_col]) for r in valid_rows])
    slope, intercept, r_value, p_value, _ = linregress(xs_all, ys_all)
    r2 = r_value ** 2
    n = len(valid_rows)

    x_fit = np.linspace(xs_all.min(), xs_all.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k--', linewidth=2,
            label=f'$R^2 = {r2:.3f}$, $n = {n}$\n'
                  f'$y = {slope:.1f}x {intercept:+.1f}$\n'
                  f'$p = {p_value:.1e}$')

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {output_file.name}  (R² = {r2:.4f}, n = {n}, p = {p_value:.2e})")


def main():
    print("=" * 70)
    print("POLITZER PER-CONFORMÈRE")
    print("=" * 70)

    rows = read_csv(CONF_CSV)

    # Exclure les molécules DEA (rupture C-X en réduction)
    rows = [r for r in rows if r.get('molecule_name') not in DEA_EXCLUDED]

    molecules = sorted(set(r['molecule_name'] for r in rows))
    print(f"\n{len(rows)} conformères, {len(molecules)} molécules (DEA exclues)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # IP vs Ī_S,min
    plot_politzer_per_conformer(
        rows,
        x_col='I_S_min_eV', y_col='IP_kJ',
        x_label='$\\bar{I}_{S,\\min}$ (eV)',
        y_label='IP (kJ/mol)',
        title='IP vs $\\bar{I}_{S,\\min}$ per-conformère (Politzer)',
        output_file=OUTPUT_DIR / "IP_vs_ALIE_per_conformer.png",
    )

    # |EA| vs -V_S,min
    # Préparer les colonnes dérivées
    for r in rows:
        ea = r.get('EA_kJ')
        vs = r.get('V_S_min_eV')
        r['EA_kJ_abs'] = abs(float(ea)) if is_valid(ea) else float('nan')
        r['neg_V_S_min_eV'] = -float(vs) if is_valid(vs) else float('nan')

    plot_politzer_per_conformer(
        rows,
        x_col='neg_V_S_min_eV', y_col='EA_kJ_abs',
        x_label='$-V_{S,\\min}$ (eV)',
        y_label='|EA| (kJ/mol)',
        title='|EA| vs $-V_{S,\\min}$ per-conformère (Politzer)',
        output_file=OUTPUT_DIR / "EA_vs_VS_per_conformer.png",
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
