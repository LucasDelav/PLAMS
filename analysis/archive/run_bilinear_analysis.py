#!/usr/bin/env python3
"""
Script pour lancer l'analyse bilinéaire des corrélations E_relax.

Lit les données depuis molecules_summary_with_freq.csv et
erelax_correlations.csv, puis teste toutes les paires de descripteurs.

Usage:
    python3 run_bilinear_analysis.py [--top-n 8] [--output-dir rapport_erelax_correlations]
"""

import sys
import os
import argparse
import csv
from pathlib import Path

import numpy as np

# Ajouter la racine du projet au path pour les imports
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

# Import direct pour éviter la dépendance à scm.plams via qmanalysis.__init__
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bilinear_analysis",
    os.path.join(_project_root, "qmanalysis", "bilinear_analysis.py")
)
_bilinear_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bilinear_mod)
compute_all_bilinear_correlations = _bilinear_mod.compute_all_bilinear_correlations
format_bilinear_results = _bilinear_mod.format_bilinear_results


def load_data(csv_path):
    """Charge les données depuis le CSV summary."""
    molecules = []
    data = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            mol_name = row['molecule_name']
            molecules.append(mol_name)
            for key in headers:
                if key == 'molecule_name':
                    continue
                if key not in data:
                    data[key] = []
                val = row[key]
                if val == '' or val is None:
                    data[key].append(np.nan)
                else:
                    try:
                        data[key].append(float(val))
                    except ValueError:
                        data[key].append(np.nan)

    # Convertir en arrays numpy
    for key in data:
        data[key] = np.array(data[key])

    return molecules, data


def get_descriptor_label(col_name):
    """Mapping des noms de colonnes CSV vers des labels lisibles."""
    label_map = {
        'HOMO_eV': 'HOMO (eV)',
        'LUMO_eV': 'LUMO (eV)',
        'gap_eV': 'Gap HOMO-LUMO (eV)',
        'EA_kJ': 'EA (kJ/mol)',
        'IP_kJ': 'IP (kJ/mol)',
        'C1_bader': 'q(C1)', 'C2_bader': 'q(C2)', 'C3_bader': 'q(C3)',
        'C4_bader': 'q(C4)', 'C5_bader': 'q(C5)', 'C6_bader': 'q(C6)',
        'O7_bader': 'q(O7)', 'O8_bader': 'q(O8)',
        'C1_iqa': 'IQA(C1)', 'C2_iqa': 'IQA(C2)', 'C3_iqa': 'IQA(C3)',
        'C4_iqa': 'IQA(C4)', 'C5_iqa': 'IQA(C5)', 'C6_iqa': 'IQA(C6)',
        'O7_iqa': 'IQA(O7)', 'O8_iqa': 'IQA(O8)',
        'q_O_total': 'q(O7+O8)',
        'q_O_asymmetry': '|q(O7)-q(O8)|',
        'freq_max': 'freq max', 'freq_min': 'freq min',
        'freq_mean': 'freq moy.', 'freq_std': 'freq σ',
        'freq_median': 'freq médiane', 'freq_spread': 'freq spread',
        'freq_cv': 'freq CV',
        'freq_ir_weighted': 'freq pondérée IR',
        'freq_of_max_ir': 'freq(IR max)',
        'flexibility_index': 'indice flexibilité',
        'stiffness_index': 'indice rigidité',
        'n_modes_low': 'N modes 0-500',
        'n_modes_medium_low': 'N modes 500-1000',
        'n_modes_medium': 'N modes 1000-2000',
        'n_modes_high': 'N modes >2000',
        'n_modes_total': 'N modes total',
        'zpe_kj': 'ZPE (kJ/mol)',
        'ir_max': 'IR max (km/mol)',
        'ir_sum': 'IR somme',
        'ir_mean': 'IR moy.',
        'mean_freq_low': 'moy. freq 0-500',
        'mean_freq_medium': 'moy. freq 1000-2000',
    }
    return label_map.get(col_name, col_name)


# Colonnes à utiliser comme descripteurs (même ensemble que l'analyse monolinéaire)
DESCRIPTOR_COLUMNS = [
    'HOMO_eV', 'LUMO_eV', 'gap_eV', 'EA_kJ', 'IP_kJ',
    'C1_bader', 'C2_bader', 'C3_bader', 'C4_bader', 'C5_bader', 'C6_bader',
    'O7_bader', 'O8_bader',
    'C1_iqa', 'C2_iqa', 'C3_iqa', 'C4_iqa', 'C5_iqa', 'C6_iqa',
    'O7_iqa', 'O8_iqa',
    'q_O_total', 'q_O_asymmetry',
    'freq_max', 'freq_min', 'freq_mean', 'freq_std', 'freq_median',
    'freq_spread', 'freq_ir_weighted', 'freq_of_max_ir',
    'flexibility_index', 'stiffness_index',
    'n_modes_low', 'n_modes_medium_low', 'n_modes_medium', 'n_modes_high',
    'n_modes_total',
    'zpe_kj',
    'ir_max', 'ir_sum', 'ir_mean',
    'mean_freq_low', 'mean_freq_medium',
]


def plot_top_bilinear_grid(molecules, descriptors, y, results, output_file,
                           top_n=8, process_label=""):
    """
    Grille des top N bilinéaires : prédit vs réel (même style que monolinéaire).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(results) == 0:
        return

    top = results[:top_n]
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

    for idx, res in enumerate(top):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        x1 = descriptors[res['desc1']]
        x2 = descriptors[res['desc2']]
        a, b, c = res['a'], res['b'], res['c']

        # Masque NaN
        mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
        y_real = y[mask]
        y_pred = a * x1[mask] + b * x2[mask] + c
        mol_names = [molecules[i] for i in range(len(molecules)) if mask[i]]

        ax.scatter(y_pred, y_real, s=70, alpha=0.7,
                   color='steelblue', edgecolors='white', linewidths=0.5, zorder=3)

        for name, xp, yr in zip(mol_names, y_pred, y_real):
            ax.annotate(name, (xp, yr), xytext=(3, 3), textcoords='offset points',
                        fontsize=6, alpha=0.7)

        # Diagonale y = x
        all_vals = np.concatenate([y_real, y_pred])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        margin = (vmax - vmin) * 0.1
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'r--', linewidth=1.5, alpha=0.8)
        ax.set_xlim(vmin - margin, vmax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)

        # Titre avec les deux descripteurs
        title = f"{res['desc1']}\n+ {res['desc2']}"
        mae = np.mean(np.abs(y_real - y_pred))
        ax.set_title(f"{title}\nMAE $= {mae:.1f}$ kJ/mol, $R^2 = {res['r2']:.3f}$",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('$E_{relax}$ prédit (kJ/mol)', fontsize=8)
        ax.set_ylabel('$E_{relax}$ réel (kJ/mol)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, zorder=0)

    # Masquer axes vides
    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Top {n} corrélations bilinéaires avec $E_{{relax}}$ ({process_label})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyse bilinéaire des corrélations E_relax")
    parser.add_argument('--top-n', type=int, default=8, help='Nombre de top corrélations (défaut: 8)')
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument('--output-dir', type=str,
                        default=str(_root / 'results' / 'erelax_correlations'),
                        help='Répertoire de sortie')
    parser.add_argument('--csv', type=str,
                        default=str(_root / 'results' / 'erelax_correlations' / 'molecules_summary_with_freq.csv'),
                        help='Fichier CSV source')
    args = parser.parse_args()

    print("=" * 80)
    print("  ANALYSE BILINÉAIRE : E_relax = a*X1 + b*X2 + c")
    print("=" * 80)

    # Charger les données
    molecules, data = load_data(args.csv)
    print(f"\nMolécules chargées: {len(molecules)}")
    print(f"  {', '.join(molecules)}")

    # Préparer les descripteurs
    descriptors = {}
    for col in DESCRIPTOR_COLUMNS:
        if col in data:
            arr = data[col]
            n_valid = np.sum(~np.isnan(arr))
            if n_valid >= 4:
                label = get_descriptor_label(col)
                descriptors[label] = arr

    print(f"\nDescripteurs valides: {len(descriptors)}")
    n_pairs = len(descriptors) * (len(descriptors) - 1) // 2
    print(f"Paires à tester: {n_pairs}")

    # Cibles
    y_red = data['Edef_red_kJ']
    y_ox = data['Edef_ox_kJ']

    # Analyse réduction
    print("\n" + "-" * 80)
    print("  RÉDUCTION : E_relax_red")
    print("-" * 80)
    results_red = compute_all_bilinear_correlations(descriptors, y_red, "Edef_red_kJ")
    print(f"\nCorrélations calculées: {len(results_red)}")
    print(f"\nTop {args.top_n} corrélations bilinéaires (réduction):\n")
    print(format_bilinear_results(results_red, args.top_n))

    # Détails du top
    print(f"\n\nDétails des {args.top_n} meilleures (réduction):")
    print("-" * 80)
    for i, res in enumerate(results_red[:args.top_n], 1):
        print(f"\n  #{i}: {res['desc1']} + {res['desc2']}")
        print(f"      R² = {res['r2']:.4f}, R²adj = {res['adj_r2']:.4f}, p = {res['p_value']:.2e}")
        print(f"      E_relax = {res['a']:.4f} * X1 + {res['b']:.4f} * X2 + {res['c']:.4f}")

    # Analyse oxydation
    print("\n" + "-" * 80)
    print("  OXYDATION : E_relax_ox")
    print("-" * 80)
    results_ox = compute_all_bilinear_correlations(descriptors, y_ox, "Edef_ox_kJ")
    print(f"\nCorrélations calculées: {len(results_ox)}")
    print(f"\nTop {args.top_n} corrélations bilinéaires (oxydation):\n")
    print(format_bilinear_results(results_ox, args.top_n))

    print(f"\n\nDétails des {args.top_n} meilleures (oxydation):")
    print("-" * 80)
    for i, res in enumerate(results_ox[:args.top_n], 1):
        print(f"\n  #{i}: {res['desc1']} + {res['desc2']}")
        print(f"      R² = {res['r2']:.4f}, R²adj = {res['adj_r2']:.4f}, p = {res['p_value']:.2e}")
        print(f"      E_relax = {res['a']:.4f} * X1 + {res['b']:.4f} * X2 + {res['c']:.4f}")

    # Figures
    os.makedirs(args.output_dir, exist_ok=True)

    plot_top_bilinear_grid(
        molecules, descriptors, y_red, results_red,
        os.path.join(args.output_dir, "top_bilinear_reduction.png"),
        top_n=args.top_n, process_label="réduction"
    )
    plot_top_bilinear_grid(
        molecules, descriptors, y_ox, results_ox,
        os.path.join(args.output_dir, "top_bilinear_oxydation.png"),
        top_n=args.top_n, process_label="oxydation"
    )

    # Export CSV

    for name, results in [("reduction", results_red), ("oxydation", results_ox)]:
        csv_path = os.path.join(args.output_dir, f"bilinear_correlations_{name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'rank', 'desc1', 'desc2', 'target', 'r2', 'adj_r2', 'p_value', 'a', 'b', 'c', 'n'
            ])
            writer.writeheader()
            for i, res in enumerate(results, 1):
                writer.writerow({'rank': i, **res})
        print(f"\nExporté: {csv_path} ({len(results)} lignes)")

    # Export résumé texte
    txt_path = os.path.join(args.output_dir, "bilinear_correlations_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("CORRÉLATIONS BILINÉAIRES : E_relax = a*X1 + b*X2 + c\n")
        f.write("=" * 100 + "\n")
        f.write(f"\nNombre de molécules: {len(molecules)}\n")
        f.write(f"Nombre de descripteurs: {len(descriptors)}\n")
        f.write(f"Nombre de paires testées: {n_pairs}\n")

        for name, results in [("Réduction", results_red), ("Oxydation", results_ox)]:
            f.write(f"\n\n{'─' * 100}\n")
            f.write(f"  Cible : E_relax {name.lower()}\n")
            f.write(f"{'─' * 100}\n\n")
            f.write(format_bilinear_results(results, args.top_n))
            f.write("\n\nDétails:\n")
            for i, res in enumerate(results[:args.top_n], 1):
                f.write(f"\n  #{i}: {res['desc1']} + {res['desc2']}\n")
                f.write(f"      R² = {res['r2']:.4f}, R²adj = {res['adj_r2']:.4f}, "
                        f"p = {res['p_value']:.2e}, n = {res['n']}\n")
                f.write(f"      E_relax = {res['a']:.6f} * X1 + {res['b']:.6f} * X2 "
                        f"+ {res['c']:.6f}\n")

    print(f"Exporté: {txt_path}")
    print("\n" + "=" * 80)
    print("  ANALYSE TERMINÉE")
    print("=" * 80)


if __name__ == '__main__':
    main()
