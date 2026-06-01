#!/usr/bin/env python3
"""
Génère les graphiques per-conformère Koopmans sans les outliers
Br-HMC conf_1 et F-HMC conf_2.

Produit :
  - EA_vs_LUMO_per_conformer_sans_outliers.png
  - IP_vs_HOMO_per_conformer_sans_outliers.png
"""

import sys
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from analysis import DEA_EXCLUDED

# Importer collect_all_conformers sans passer par common/__init__
_project_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "multi_molecule_data",
    _project_root / "common" / "multi_molecule_data.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
collect_all_conformers = _mod.collect_all_conformers

BATCH_DIR = _project_root / "data" / "1594174" / "dir_workdir"
OUTPUT_DIR = _project_root / "results" / "figures"
EH_TO_EV = 27.211386245988

OUTLIERS = []


def get_molecule_color_map(molecule_names):
    unique = sorted(set(molecule_names))
    cmap = plt.cm.get_cmap('tab20', max(len(unique), 20))
    return {mol: cmap(i) for i, mol in enumerate(unique)}


def plot_koopmans_filtered(df_all, df_filtered, x_col, y_col, x_label, y_label,
                            title, output_file, color):
    """Scatter plot per-conformère avec outliers grisés et fit sur données filtrées."""
    color_map = get_molecule_color_map(df_all['molecule_name'])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot outliers in grey with X marker
    outlier_mask = pd.Series([False] * len(df_all), index=df_all.index)
    for mol, conf in OUTLIERS:
        outlier_mask |= ((df_all['molecule_name'] == mol) & (df_all['conformer'] == conf))

    df_outliers = df_all[outlier_mask].dropna(subset=[x_col, y_col])
    if len(df_outliers) > 0:
        ax.scatter(df_outliers[x_col], df_outliers[y_col],
                   s=80, alpha=0.4, color='grey', marker='x',
                   linewidths=2, zorder=2)
        for _, row in df_outliers.iterrows():
            label = f"{row['molecule_name']}\nconf_{int(row['conformer'])}"
            ax.annotate(label, (row[x_col], row[y_col]),
                       xytext=(8, -8), textcoords='offset points',
                       fontsize=7, alpha=0.6, color='grey', style='italic')

    # Plot filtered conformers with molecule colors
    valid = df_filtered[['molecule_name', 'conformer', x_col, y_col]].dropna(subset=[x_col, y_col])
    for mol_name in sorted(valid['molecule_name'].unique()):
        mask = valid['molecule_name'] == mol_name
        ax.scatter(valid.loc[mask, x_col], valid.loc[mask, y_col],
                   s=60, alpha=0.7, color=color_map[mol_name],
                   edgecolors='white', linewidths=0.5, zorder=3, label=mol_name)

    # Annotate conformer numbers
    for _, row in valid.iterrows():
        ax.annotate(str(int(row['conformer'])), (row[x_col], row[y_col]),
                   xytext=(4, 4), textcoords='offset points',
                   fontsize=6, alpha=0.8)

    # Fit on filtered data
    slope, intercept, r_value, p_value, _ = linregress(valid[x_col], valid[y_col])
    x_fit = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
    y_fit = slope * x_fit + intercept
    r2 = r_value ** 2
    n = len(valid)
    ax.plot(x_fit, y_fit, 'k--', linewidth=2,
            label=f'$R^2 = {r2:.3f}$, $n = {n}$\n$y = {slope:.1f}x {intercept:+.1f}$\n$p = {p_value:.1e}$')

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {output_file.name}  (R² = {r2:.4f}, n = {n}, p = {p_value:.2e})")
    return r2, n


def main():
    print("=" * 70)
    print("KOOPMANS PER-CONFORMÈRE SANS OUTLIERS")
    print(f"  Outliers exclus: {OUTLIERS}")
    print("=" * 70)

    # Charger données
    workdirs = sorted([d for d in BATCH_DIR.glob("*_workdir") if d.is_dir()])
    df = collect_all_conformers(workdirs)
    print(f"\n{len(df)} conformères, {df['molecule_name'].nunique()} molécules")

    # Exclure les molécules DEA (rupture C-X en réduction)
    df = df[~df['molecule_name'].isin(DEA_EXCLUDED)].copy()
    print(f"Après exclusion DEA ({', '.join(sorted(DEA_EXCLUDED))}): "
          f"{len(df)} conformères, {df['molecule_name'].nunique()} molécules")

    # Convertir HOMO/LUMO en vrais eV
    df['HOMO_real_eV'] = df['HOMO_eV'] * EH_TO_EV
    df['LUMO_real_eV'] = df['LUMO_eV'] * EH_TO_EV
    df['EA_kJ_abs'] = df['EA_kJ'].abs()

    # Filtrer les outliers
    mask = pd.Series([True] * len(df), index=df.index)
    for mol, conf in OUTLIERS:
        mask &= ~((df['molecule_name'] == mol) & (df['conformer'] == conf))
    df_filtered = df[mask].copy()
    print(f"Après filtrage: {len(df_filtered)} conformères")

    # Aussi montrer les valeurs des outliers
    for mol, conf in OUTLIERS:
        row = df[(df['molecule_name'] == mol) & (df['conformer'] == conf)]
        if not row.empty:
            r = row.iloc[0]
            print(f"  Exclu: {mol} conf_{conf}:  HOMO={r.get('HOMO_real_eV', float('nan')):.3f} eV, "
                  f"LUMO={r.get('LUMO_real_eV', float('nan')):.3f} eV, "
                  f"EA={r.get('EA_kJ', float('nan')):.1f} kJ/mol, IP={r.get('IP_kJ', float('nan')):.1f} kJ/mol")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Corrélations AVANT filtrage (pour comparaison)
    print("\nAvant filtrage:")
    for x_col, y_col, label in [('LUMO_real_eV', 'EA_kJ_abs', 'EA vs LUMO'),
                                  ('HOMO_real_eV', 'IP_kJ', 'IP vs HOMO')]:
        valid = df[[x_col, y_col]].dropna()
        if len(valid) >= 3:
            _, _, r, _, _ = linregress(valid[x_col], valid[y_col])
            print(f"  {label}: R² = {r**2:.4f} (n = {len(valid)})")

    print("\nAprès filtrage:")

    # (c) EA vs LUMO sans outliers
    r2_ea, n_ea = plot_koopmans_filtered(
        df, df_filtered,
        'LUMO_real_eV', 'EA_kJ_abs', 'LUMO (eV)', '|EA| (kJ/mol)',
        'EA vs LUMO per-conformère',
        OUTPUT_DIR / "EA_vs_LUMO_per_conformer_sans_outliers.png",
        '#c0392b'
    )

    # (d) IP vs HOMO sans outliers
    r2_ip, n_ip = plot_koopmans_filtered(
        df, df_filtered,
        'HOMO_real_eV', 'IP_kJ', 'HOMO (eV)', 'IP (kJ/mol)',
        'IP vs HOMO per-conformère',
        OUTPUT_DIR / "IP_vs_HOMO_per_conformer_sans_outliers.png",
        '#2980b9'
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
