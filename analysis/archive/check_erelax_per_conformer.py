#!/usr/bin/env python3
"""
Vérification des données E_relax per-conformère et génération de graphiques
sans les outliers Br-HMC conf_1 et F-HMC conf_2.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

# Importer directement multi_molecule_data (bypass common/__init__ qui a besoin de scm.plams)
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "multi_molecule_data",
    _project_root / "common" / "multi_molecule_data.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
collect_all_conformers = _mod.collect_all_conformers

BATCH_DIR = _project_root / "data" / "1594174" / "dir_workdir"
OUTPUT_DIR = _project_root / "results" / "erelax_correlations"

# Descripteurs à tester
DESCRIPTOR_GROUPS = {
    'Orbitales frontières': {
        'HOMO_eV': 'HOMO (eV)',
        'LUMO_eV': 'LUMO (eV)',
        'gap_eV': 'Gap (eV)',
    },
    'Énergies verticales': {
        'EA_kJ': 'EA (kJ/mol)',
        'IP_kJ': 'IP (kJ/mol)',
    },
    'Charges Bader': {
        'C1_bader': 'q(C1)', 'C2_bader': 'q(C2)', 'C3_bader': 'q(C3)',
        'C4_bader': 'q(C4)', 'C5_bader': 'q(C5)', 'C6_bader': 'q(C6)',
        'O7_bader': 'q(O7)', 'O8_bader': 'q(O8)',
    },
    'IQA': {
        'C1_iqa': 'IQA(C1)', 'C2_iqa': 'IQA(C2)', 'C3_iqa': 'IQA(C3)',
        'C4_iqa': 'IQA(C4)', 'C5_iqa': 'IQA(C5)', 'C6_iqa': 'IQA(C6)',
        'O7_iqa': 'IQA(O7)', 'O8_iqa': 'IQA(O8)',
    },
}


def load_per_conformer_data():
    """Charge les données per-conformère depuis les workdirs (avec tous les descripteurs)."""
    workdirs = sorted([d for d in BATCH_DIR.glob("*_workdir") if d.is_dir()])
    if not workdirs:
        print(f"Aucun workdir trouvé dans {BATCH_DIR}")
        sys.exit(1)
    return collect_all_conformers(workdirs)


def compute_correlations(df, descriptors, target_col, target_label):
    """Calcule les corrélations pour un target donné."""
    results = []
    for desc_col, desc_label in descriptors.items():
        if desc_col not in df.columns or target_col not in df.columns:
            continue
        valid = df[[desc_col, target_col]].dropna()
        if len(valid) < 5:
            continue
        r, p = pearsonr(valid[desc_col], valid[target_col])
        slope, intercept, _, _, _ = linregress(valid[desc_col], valid[target_col])
        results.append({
            'descriptor': desc_col,
            'descriptor_label': desc_label,
            'target': target_col,
            'target_label': target_label,
            'R2': r**2, 'r': r, 'p_value': p,
            'slope': slope, 'intercept': intercept,
            'n': len(valid),
        })
    return sorted(results, key=lambda x: x['R2'], reverse=True)


def get_molecule_color_map(molecule_names):
    unique = sorted(set(molecule_names))
    cmap = plt.colormaps.get_cmap('tab20').resampled(max(len(unique), 20))
    return {mol: cmap(i) for i, mol in enumerate(unique)}


def plot_top_correlations(df, corr_results, output_file, target_label_short, top_n=8):
    """Grille des top N corrélations per-conformère."""
    if not corr_results:
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

    color_map = get_molecule_color_map(df['molecule_name'])

    for idx, res in enumerate(top):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        desc_col = res['descriptor']
        target_col = res['target']
        valid = df[['molecule_name', 'conformer', desc_col, target_col]].dropna(
            subset=[desc_col, target_col])

        y_pred = res['slope'] * valid[desc_col].values + res['intercept']
        y_real = valid[target_col].values

        mol_arr = valid['molecule_name'].values
        for mol_name in sorted(set(mol_arr)):
            mask = mol_arr == mol_name
            ax.scatter(y_pred[mask], y_real[mask], s=50, alpha=0.7,
                      color=color_map[mol_name], edgecolors='white',
                      linewidths=0.3, zorder=3)

        conf_arr = valid['conformer'].values
        for i in range(len(y_pred)):
            ax.annotate(str(int(conf_arr[i])), (y_pred[i], y_real[i]),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=5, alpha=0.7)

        all_vals = np.concatenate([y_real, y_pred])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        margin = (vmax - vmin) * 0.1
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'r--', linewidth=1.5, alpha=0.8)

        mae = np.mean(np.abs(y_real - y_pred))
        ax.set_title(f"{res['descriptor_label']}\nMAE $= {mae:.1f}$ kJ/mol, $R^2 = {res['R2']:.3f}$ (n={res['n']})",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel(f'{target_label_short} prédit (kJ/mol)', fontsize=8)
        ax.set_ylabel(f'{target_label_short} réel (kJ/mol)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, zorder=0)

    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Top {n} corrélations avec {target_label_short} [per-conformère]',
                fontsize=14, fontweight='bold', y=1.02)

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
    print(f"  Figure sauvegardée: {output_file}")


def main():
    print("=" * 80)
    print("VÉRIFICATION E_relax PER-CONFORMÈRE")
    print("=" * 80)

    # 1. Charger les données
    df = load_per_conformer_data()
    print(f"\n{len(df)} conformères chargés ({df['molecule_name'].nunique()} molécules)")

    # 2. Afficher toutes les valeurs Edef
    print("\n" + "=" * 80)
    print("VALEURS Edef PAR CONFORMÈRE")
    print("=" * 80)
    for mol in sorted(df['molecule_name'].unique()):
        sub = df[df['molecule_name'] == mol].sort_values('conformer')
        print(f"\n  {mol}:")
        for _, row in sub.iterrows():
            flag_red = " *** POSITIF" if row.get('Edef_red_kJ', -1) > 0 else ""
            flag_ox = " *** POSITIF" if row.get('Edef_ox_kJ', -1) > 0 else ""
            rmsd_red = row.get('RMSD_red_A', float('nan'))
            rmsd_ox = row.get('RMSD_ox_A', float('nan'))
            print(f"    conf_{int(row['conformer']):d}  w={row.get('weight', 0):.3f}  "
                  f"Edef_red={row.get('Edef_red_kJ', float('nan')):8.1f}{flag_red}  "
                  f"Edef_ox={row.get('Edef_ox_kJ', float('nan')):8.1f}{flag_ox}  "
                  f"RMSD_red={rmsd_red:.3f}  RMSD_ox={rmsd_ox:.3f}")

    # 3. Aucun outlier exclu (données corrigées)
    outliers = []

    # Aussi lister tous les conformères avec Edef_red positif
    if 'Edef_red_kJ' in df.columns:
        pos_edef = df[df['Edef_red_kJ'] > 0]
        if not pos_edef.empty:
            print("\n" + "=" * 80)
            print("CONFORMÈRES AVEC Edef_red POSITIF (anomalies)")
            print("=" * 80)
            for _, row in pos_edef.iterrows():
                print(f"  {row['molecule_name']} conf_{int(row['conformer'])}:  "
                      f"Edef_red = {row['Edef_red_kJ']:+.1f} kJ/mol  "
                      f"RMSD_red = {row.get('RMSD_red_A', float('nan')):.3f} Å")

    # 4. Tous les descripteurs
    all_descriptors = {}
    for group in DESCRIPTOR_GROUPS.values():
        all_descriptors.update(group)

    targets = {
        'Edef_red_kJ': '$E_{relax}^{red}$ (kJ/mol)',
        'Edef_ox_kJ': '$E_{relax}^{ox}$ (kJ/mol)',
    }

    # 5. Corrélations AVEC tous les conformères
    print("\n" + "=" * 80)
    print("CORRÉLATIONS AVEC TOUS LES CONFORMÈRES")
    print("=" * 80)

    for target_col, target_label in targets.items():
        corr_all = compute_correlations(df, all_descriptors, target_col, target_label)
        short = "E_relax_red" if "red" in target_col else "E_relax_ox"
        print(f"\n  Top 8 pour {short} (n={len(df)}):")
        for i, c in enumerate(corr_all[:8]):
            print(f"    {i+1}. {c['descriptor_label']:20s}  R²={c['R2']:.4f}  "
                  f"p={c['p_value']:.2e}  n={c['n']}")

        plot_top_correlations(
            df, corr_all,
            OUTPUT_DIR / f"top_correlations_{short}_all_conformers.png",
            short, top_n=8)

    # 6. Corrélations SANS les outliers spécifiés
    mask = pd.Series([True] * len(df), index=df.index)
    for mol, conf in outliers:
        mask &= ~((df['molecule_name'] == mol) & (df['conformer'] == conf))
    df_filtered = df[mask].copy()

    print(f"\n" + "=" * 80)
    print(f"CORRÉLATIONS SANS OUTLIERS: {outliers}")
    print(f"  {len(df)} -> {len(df_filtered)} conformères")
    print("=" * 80)

    for target_col, target_label in targets.items():
        corr_filt = compute_correlations(df_filtered, all_descriptors, target_col, target_label)
        short = "E_relax_red" if "red" in target_col else "E_relax_ox"
        print(f"\n  Top 8 pour {short} (n={len(df_filtered)}):")
        for i, c in enumerate(corr_filt[:8]):
            print(f"    {i+1}. {c['descriptor_label']:20s}  R²={c['R2']:.4f}  "
                  f"p={c['p_value']:.2e}  n={c['n']}")

        plot_top_correlations(
            df_filtered, corr_filt,
            OUTPUT_DIR / f"top_correlations_{short}_sans_outliers.png",
            short, top_n=8)

    # 7. Comparaison R² avant/après
    print("\n" + "=" * 80)
    print("COMPARAISON R² AVANT/APRÈS SUPPRESSION DES OUTLIERS")
    print("=" * 80)

    for target_col, target_label in targets.items():
        short = "E_relax_red" if "red" in target_col else "E_relax_ox"
        corr_all = compute_correlations(df, all_descriptors, target_col, target_label)
        corr_filt = compute_correlations(df_filtered, all_descriptors, target_col, target_label)

        # Créer lookup
        r2_all = {c['descriptor']: c['R2'] for c in corr_all}
        r2_filt = {c['descriptor']: c['R2'] for c in corr_filt}

        print(f"\n  {short}:")
        print(f"  {'Descripteur':20s}  {'R² (tous)':>10s}  {'R² (filtré)':>12s}  {'Delta':>8s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*8}")

        # Top 10 par R² filtré
        for c in corr_filt[:10]:
            desc = c['descriptor']
            r2_a = r2_all.get(desc, float('nan'))
            r2_f = c['R2']
            delta = r2_f - r2_a
            print(f"  {c['descriptor_label']:20s}  {r2_a:10.4f}  {r2_f:12.4f}  {delta:+8.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
