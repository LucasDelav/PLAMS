#!/usr/bin/env amspython
"""
Histogrammes ΔΔU, -TΔS et ΔΔU - TΔS pour les 19 catéchols (input.txt).

Produit 3 histogrammes avec ajustement gaussien (loi normale),
séparés réduction / oxydation, per-conformère pondéré par Boltzmann.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

_project_root = Path(__file__).resolve().parent.parent
BATCH_DIR = _project_root / "data" / "Acetonitrile"
OUTPUT_DIR = _project_root / "results" / "figures"

# 19 catéchols de input.txt (lignes 9-27)
CATECHOLS_19 = [
    'HMC', 'F-HMC', 'Cl-HMC', 'Br-HMC', 'I-HMC',
    'DA', 'F-DA', 'Cl-DA', 'Br-DA', 'I-DA',
    'L-DA', 'CA', 'Al-CA', 'HO-CA', 'H2C-CA',
    'DHI', 'O-DHI', 'S-DHI', 'P-DHI',
]


def load_all_conformers():
    """Charge les données redox per-conformère pour les 19 catéchols."""
    rows = []
    for mol in CATECHOLS_19:
        csv_path = BATCH_DIR / f"{mol}_workdir" / "redox" / "results" / f"{mol}_redox_conformers.csv"
        if not csv_path.exists():
            print(f"  ! {mol}: CSV non trouvé")
            continue
        df = pd.read_csv(csv_path)
        df['molecule'] = mol
        rows.append(df)
        print(f"  {mol}: {len(df)} conformères")

    return pd.concat(rows, ignore_index=True)


def plot_histogram(values, weights, xlabel, title, filename, color, unit='kJ/mol'):
    """Histogramme pondéré avec ajustement gaussien."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogramme pondéré
    n_bins = 25
    counts, bin_edges, patches = ax.hist(
        values, bins=n_bins, weights=weights,
        density=True, alpha=0.6, color=color, edgecolor='white', linewidth=0.8,
        label='Données pondérées'
    )

    # Ajustement gaussien (moyenne et std pondérées)
    w = np.array(weights)
    v = np.array(values)
    mu = np.average(v, weights=w)
    var = np.average((v - mu)**2, weights=w)
    sigma = np.sqrt(var)

    x_fit = np.linspace(v.min() - 0.1 * abs(v.max() - v.min()),
                         v.max() + 0.1 * abs(v.max() - v.min()), 300)
    y_fit = norm.pdf(x_fit, mu, sigma)
    ax.plot(x_fit, y_fit, 'k-', linewidth=2, label=f'$\\mathcal{{N}}(\\mu={mu:.2f},\\,\\sigma={sigma:.2f})$')

    # Ligne verticale à μ
    ax.axvline(mu, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'$\\mu = {mu:.2f}$ {unit}')
    ax.axvline(0, color='grey', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Densité de probabilité', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2, zorder=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → {filename.name}  (μ={mu:.2f}, σ={sigma:.2f} {unit}, n={len(v)})")


def main():
    print("=" * 60)
    print("HISTOGRAMMES ΔΔU, -TΔS, ΔΔU-TΔS — 19 catéchols")
    print("=" * 60)

    df = load_all_conformers()
    print(f"\nTotal: {len(df)} conformères, {df['molecule'].nunique()} molécules")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # On fait réduction ET oxydation séparément
    for redox, suffix, col_ddu, col_tds in [
        ('Réduction', 'red', 'ddU_red_kJ', 'TdS_red_kJ'),
        ('Oxydation', 'ox', 'ddU_ox_kJ', 'TdS_ox_kJ'),
    ]:
        print(f"\n--- {redox} ---")

        valid = df.dropna(subset=[col_ddu, col_tds, 'weight']).copy()
        ddu = valid[col_ddu].values
        tds = valid[col_tds].values
        neg_tds = -tds  # -TΔS
        diff = ddu - tds  # ΔΔU - TΔS
        weights = valid['weight'].values

        # Normaliser les poids par molécule pour que chaque molécule pèse pareil
        # (sinon les molécules avec plus de conformères dominent)
        mol_weights = []
        for _, row in valid.iterrows():
            mol = row['molecule']
            mol_total_w = valid.loc[valid['molecule'] == mol, 'weight'].sum()
            mol_weights.append(row['weight'] / mol_total_w / valid['molecule'].nunique())
        mol_weights = np.array(mol_weights)
        mol_weights /= mol_weights.sum()  # Normaliser à 1

        # 1. ΔΔU
        plot_histogram(
            ddu, mol_weights,
            f'ΔΔU ({redox}) (kJ/mol)',
            f'Distribution de ΔΔU — {redox} (19 catéchols)',
            OUTPUT_DIR / f"hist_ddU_{suffix}.png",
            color='#3498db'
        )

        # 2. -TΔS
        plot_histogram(
            neg_tds, mol_weights,
            f'−TΔS ({redox}) (kJ/mol)',
            f'Distribution de −TΔS — {redox} (19 catéchols)',
            OUTPUT_DIR / f"hist_neg_TdS_{suffix}.png",
            color='#e74c3c'
        )

        # 3. ΔΔU - TΔS
        plot_histogram(
            diff, mol_weights,
            f'ΔΔU − TΔS ({redox}) (kJ/mol)',
            f'Distribution de ΔΔU − TΔS — {redox} (19 catéchols)',
            OUTPUT_DIR / f"hist_ddU_minus_TdS_{suffix}.png",
            color='#9b59b6'
        )

    # ===================================================================
    # Figure combinée : les 3 histogrammes côte à côte (oxydation)
    # ===================================================================
    for redox, suffix, col_ddu, col_tds in [
        ('Réduction', 'red', 'ddU_red_kJ', 'TdS_red_kJ'),
        ('Oxydation', 'ox', 'ddU_ox_kJ', 'TdS_ox_kJ'),
    ]:
        valid = df.dropna(subset=[col_ddu, col_tds, 'weight']).copy()
        ddu = valid[col_ddu].values
        tds = valid[col_tds].values
        neg_tds = -tds
        diff = ddu - tds

        mol_weights = []
        for _, row in valid.iterrows():
            mol = row['molecule']
            mol_total_w = valid.loc[valid['molecule'] == mol, 'weight'].sum()
            mol_weights.append(row['weight'] / mol_total_w / valid['molecule'].nunique())
        mol_weights = np.array(mol_weights)
        mol_weights /= mol_weights.sum()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, vals, color, label in [
            (axes[0], ddu, '#3498db', 'ΔΔU'),
            (axes[1], neg_tds, '#e74c3c', '−TΔS'),
            (axes[2], diff, '#9b59b6', 'ΔΔU − TΔS'),
        ]:
            ax.hist(vals, bins=25, weights=mol_weights, density=True,
                    alpha=0.6, color=color, edgecolor='white', linewidth=0.8)

            mu = np.average(vals, weights=mol_weights)
            var = np.average((vals - mu)**2, weights=mol_weights)
            sigma = np.sqrt(var)

            x_fit = np.linspace(vals.min() - 0.15 * abs(vals.ptp()),
                                vals.max() + 0.15 * abs(vals.ptp()), 300)
            ax.plot(x_fit, norm.pdf(x_fit, mu, sigma), 'k-', linewidth=2)
            ax.axvline(mu, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(0, color='grey', linestyle=':', linewidth=1, alpha=0.5)

            ax.set_xlabel(f'{label} (kJ/mol)', fontsize=11)
            ax.set_ylabel('Densité', fontsize=10)
            ax.set_title(f'{label}\n$\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$ kJ/mol',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2, zorder=0)

        plt.suptitle(f'{redox} — 19 catéchols', fontsize=14, fontweight='bold')
        plt.tight_layout()
        outfile = OUTPUT_DIR / f"hist_combined_{suffix}.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  → Combiné: {outfile}")

    print("\nDone.")


if __name__ == '__main__':
    main()
