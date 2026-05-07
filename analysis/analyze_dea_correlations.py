#!/usr/bin/env amspython
"""
Analyse des correlations pour predire le residu DEA (E_break).

Pour les 8 molecules halogenees (F/Cl/Br/I x DA/HMC), cherche des correlations
entre descripteurs moleculaires et le residu E_relax - E_harm^LVC.

Descripteurs testes :
  - BDE(C-X) tabulee
  - Polarisabilite atomique de X
  - Affinite electronique de X
  - Electronegativite de X (Pauling)
  - Longueur de liaison C-X (neutre, depuis RKF)
  - Energie LUMO du neutre
  - Energie de sigma*(C-X) (identifiee par analyse SFO)
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats

from scm.plams import KFReader

# -- Constantes ---------------------------------------------------------------

BOHR_TO_ANG = 0.529177249
EH_TO_EV = 27.211386245988

Z_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 13: 'Al',
    16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
}
SYMBOL_TO_Z = {v: k for k, v in Z_TO_SYMBOL.items()}

# -- Donnees tabulees ----------------------------------------------------------

# BDE aryl-X typiques (kJ/mol) -- PhX -> Ph. + X.
# Sources : CRC Handbook, Blanksby & Ellison (2003)
BDE_ARYL = {
    'F':  526.0,   # C(sp2)-F
    'Cl': 399.0,   # C(sp2)-Cl
    'Br': 336.0,   # C(sp2)-Br
    'I':  272.0,   # C(sp2)-I
}

# Polarisabilite atomique (ang^3) -- CRC / Schwerdtfeger (2019)
POLARIZABILITY = {
    'F':  0.557,
    'Cl': 2.18,
    'Br': 3.05,
    'I':  4.70,
}

# Affinite electronique de X (eV) -- NIST
ELECTRON_AFFINITY = {
    'F':  3.401,
    'Cl': 3.613,
    'Br': 3.364,
    'I':  3.059,
}

# Electronegativite de Pauling
ELECTRONEGATIVITY = {
    'F':  3.98,
    'Cl': 3.16,
    'Br': 2.96,
    'I':  2.66,
}

# -- Donnees du rapport (residus) ----------------------------------------------

MOLECULES_HAL = {
    'F-DA':   {'halogen': 'F',  'family': 'DA',  'E_relax': -32.9,  'E_lvc': -24.8, 'breaks': False},
    'Cl-DA':  {'halogen': 'Cl', 'family': 'DA',  'E_relax': -30.7,  'E_lvc': -24.7, 'breaks': False},
    'F-HMC':  {'halogen': 'F',  'family': 'HMC', 'E_relax': -45.8,  'E_lvc': -26.2, 'breaks': False},
    'Cl-HMC': {'halogen': 'Cl', 'family': 'HMC', 'E_relax': -40.1,  'E_lvc': -24.8, 'breaks': False},
}

for _mol, _data in MOLECULES_HAL.items():
    _data['residual'] = _data['E_relax'] - _data['E_lvc']


# -- Extraction depuis RKF ----------------------------------------------------

def find_neutral_rkf(batch_dir, mol_name):
    """Trouve le RKF neutre du conformere 1."""
    plams_dir = Path(batch_dir) / f'{mol_name}_workdir' / 'redox' / 'calculations' / 'plams_workdir'
    rkf = plams_dir / f'{mol_name}_conf_1_neutral' / 'adf.rkf'
    if rkf.is_file():
        return rkf
    return None


def extract_orbital_and_geometry_data(rkf_path, halogen_symbol):
    """Extrait donnees orbitales et geometriques depuis un RKF neutre."""
    kf = KFReader(str(rkf_path))

    # -- Geometrie --
    n_atoms = kf.read('Molecule', 'nAtoms')
    atom_numbers = list(kf.read('Molecule', 'AtomicNumbers'))
    coords_bohr = np.array(kf.read('Molecule', 'Coords'), dtype=float).reshape(n_atoms, 3)
    coords_ang = coords_bohr * BOHR_TO_ANG

    # Trouver l'halogene
    hal_z = SYMBOL_TO_Z[halogen_symbol]
    hal_indices = [i for i in range(n_atoms) if atom_numbers[i] == hal_z]
    if not hal_indices:
        raise ValueError(f"No {halogen_symbol} atom found")
    hal_idx = hal_indices[0]

    # Trouver le C le plus proche
    c_indices = [i for i in range(n_atoms) if atom_numbers[i] == 6]
    distances = {c: np.linalg.norm(coords_ang[c] - coords_ang[hal_idx]) for c in c_indices}
    c_partner = min(distances, key=lambda k: distances[k])
    cx_bond_length = distances[c_partner]

    # -- HOMO / LUMO --
    homo_eh = kf.read('Properties', 'HOMO')
    lumo_eh = kf.read('Properties', 'LUMO')

    # -- Identification de sigma*(C-X) via SFO popul --
    n_sfo = kf.read('SFOs', 'number')
    sfo_fragments = list(kf.read('SFOs', 'fragment'))  # 1-indexed

    # SFOs de l'halogene et du C partenaire (1-indexed pour SFO popul)
    hal_sfos = set(i + 1 for i in range(n_sfo) if sfo_fragments[i] == hal_idx + 1)
    c_sfos = set(i + 1 for i in range(n_sfo) if sfo_fragments[i] == c_partner + 1)

    # Donnees orbitales
    n_mo = kf.read('A', 'nmo_A')
    occupations = np.array(kf.read('A', 'froc_A'), dtype=float)
    n_occ = int(np.sum(occupations > 0.5))
    orbital_energies = np.array(kf.read('A', 'eps_A'), dtype=float)

    # SFO popul : format packed
    n_contrib = list(kf.read('SFO popul', 'number of contributions'))
    sfo_pop_idx = list(kf.read('SFO popul', 'sfo_index'))
    sfo_pop_val = list(kf.read('SFO popul', 'sfo_pop'))

    n_virtual_check = min(30, n_mo - n_occ)
    sigma_star_candidates = []
    offset = 0
    for mo_idx in range(n_mo):
        nc = n_contrib[mo_idx]
        if n_occ <= mo_idx < n_occ + n_virtual_check:
            idx_slice = sfo_pop_idx[offset:offset + nc]
            val_slice = sfo_pop_val[offset:offset + nc]

            hal_w = sum(abs(p) for s, p in zip(idx_slice, val_slice) if s in hal_sfos)
            c_w = sum(abs(p) for s, p in zip(idx_slice, val_slice) if s in c_sfos)
            tot_w = sum(abs(p) for p in val_slice)

            sigma_star_candidates.append({
                'mo_idx': mo_idx,
                'energy_ev': orbital_energies[mo_idx] * EH_TO_EV,
                'hal_weight': hal_w,
                'c_weight': c_w,
                'hal_fraction': hal_w / tot_w if tot_w > 0 else 0,
            })
        offset += nc

    sigma_star_candidates.sort(key=lambda x: x['hal_weight'], reverse=True)
    best = sigma_star_candidates[0]

    print(f"    Top 5 sigma*(C-X) candidates (by halogen weight):")
    for cand in sigma_star_candidates[:5]:
        print(f"      MO {cand['mo_idx']+1}: E={cand['energy_ev']:+.3f} eV, "
              f"X_frac={cand['hal_fraction']:.3f}, "
              f"X_weight={cand['hal_weight']:.3f}, C_weight={cand['c_weight']:.3f}")

    # Indice de sigma* par rapport a la LUMO (LUMO = 1, LUMO+1 = 2, etc.)
    sigma_star_lumo_index = best['mo_idx'] - n_occ + 1

    return {
        'lumo_ev': lumo_eh * EH_TO_EV,
        'homo_ev': homo_eh * EH_TO_EV,
        'cx_bond_length': cx_bond_length,
        'sigma_star_ev': best['energy_ev'],
        'sigma_star_hal_weight': best['hal_weight'],
        'sigma_star_mo': best['mo_idx'] + 1,
        'sigma_star_lumo_index': sigma_star_lumo_index,
        'hal_atom_idx': hal_idx + 1,
        'c_partner_idx': c_partner + 1,
    }


# -- Correlations --------------------------------------------------------------

def compute_correlations(data_list, descriptors, target_key):
    """Calcule les correlations entre descripteurs et variable cible."""
    results = []
    y = np.array([d[target_key] for d in data_list])
    n = len(y)

    for desc_key, desc_label in descriptors:
        x = np.array([d[desc_key] for d in data_list])
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            continue

        res = stats.linregress(x, y)
        r2 = res.rvalue ** 2
        y_pred = res.slope * x + res.intercept
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        results.append({
            'descriptor': desc_label,
            'desc_key': desc_key,
            'R2': r2,
            'r': res.rvalue,
            'p_value': res.pvalue,
            'slope': res.slope,
            'intercept': res.intercept,
            'MAE': mae,
            'RMSE': rmse,
            'n': n,
        })

    results.sort(key=lambda x: x['R2'], reverse=True)
    return results


# -- Plots ---------------------------------------------------------------------

def make_correlation_plots(data_list, descriptors, target_key, target_label,
                           output_dir, filename_prefix):
    """Cree les scatter plots pour les meilleures correlations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    y = np.array([d[target_key] for d in data_list])
    names = [d['molecule'] for d in data_list]
    breaks = [d['breaks'] for d in data_list]

    n_desc = len(descriptors)
    n_cols = min(3, n_desc)
    n_rows = (n_desc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (desc_key, desc_label) in enumerate(descriptors):
        ax = axes[idx // n_cols, idx % n_cols]
        x = np.array([d[desc_key] for d in data_list])

        colors = ['#d62728' if b else '#1f77b4' for b in breaks]
        ax.scatter(x, y, c=colors, s=80, zorder=3, edgecolors='black', linewidth=0.5)

        res = stats.linregress(x, y)
        x_line = np.linspace(x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x), 100)
        ax.plot(x_line, res.slope * x_line + res.intercept, 'k--', alpha=0.5, linewidth=1)

        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), fontsize=7,
                        textcoords='offset points', xytext=(4, 4))

        ax.set_xlabel(desc_label, fontsize=10)
        ax.set_ylabel(target_label, fontsize=10)
        ax.set_title(f'R2 = {res.rvalue**2:.3f}, p = {res.pvalue:.3f}', fontsize=10)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
        ax.grid(True, alpha=0.3)

    for idx in range(len(descriptors), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    legend_elements = [Patch(facecolor='#d62728', edgecolor='black', label='DEA (C-X rompu)'),
                       Patch(facecolor='#1f77b4', edgecolor='black', label='C-X intact')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(f'Correlations : {target_label}', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_dir / f'{filename_prefix}.png'}")


# -- Main ----------------------------------------------------------------------

def main():
    _root = Path(__file__).resolve().parent.parent
    batch_dir = str(_root / 'data' / '1594174' / 'dir_workdir')
    output_dir = _root / 'results' / 'dea_correlations'
    output_dir.mkdir(exist_ok=True)

    print('=' * 70)
    print('ANALYSE DES CORRELATIONS DEA')
    print('=' * 70)
    print()

    # -- Extraction des donnees RKF --
    data_list = []

    for mol_name, mol_data in sorted(MOLECULES_HAL.items()):
        print(f'  {mol_name}...')
        hal = mol_data['halogen']

        rkf_path = find_neutral_rkf(batch_dir, mol_name)
        if rkf_path is None:
            print(f'    ERREUR: RKF non trouve')
            continue

        try:
            rkf_data = extract_orbital_and_geometry_data(rkf_path, hal)
        except Exception as e:
            print(f'    ERREUR: {e}')
            continue

        entry = {
            'molecule': mol_name,
            'halogen': hal,
            'family': mol_data['family'],
            'breaks': mol_data['breaks'],
            'E_relax': mol_data['E_relax'],
            'E_lvc': mol_data['E_lvc'],
            'residual': mol_data['residual'],
            'bde': BDE_ARYL[hal],
            'polarizability': POLARIZABILITY[hal],
            'electron_affinity': ELECTRON_AFFINITY[hal],
            'electronegativity': ELECTRONEGATIVITY[hal],
            'cx_bond_length': rkf_data['cx_bond_length'],
            'lumo_ev': rkf_data['lumo_ev'],
            'homo_ev': rkf_data['homo_ev'],
            'sigma_star_ev': rkf_data['sigma_star_ev'],
            'sigma_star_lumo_index': rkf_data['sigma_star_lumo_index'],
            'homo_lumo_gap': rkf_data['lumo_ev'] - rkf_data['homo_ev'],
        }
        data_list.append(entry)

        print(f'    C-{hal} = {rkf_data["cx_bond_length"]:.3f} A '
              f'(atoms {rkf_data["c_partner_idx"]}-{rkf_data["hal_atom_idx"]})')
        lumo_idx_label = {1: 'LUMO', 2: 'LUMO+1', 3: 'LUMO+2'}.get(
            rkf_data['sigma_star_lumo_index'],
            f'LUMO+{rkf_data["sigma_star_lumo_index"]-1}')
        print(f'    LUMO = {rkf_data["lumo_ev"]:.3f} eV, '
              f'sigma*(C-{hal}) = {rkf_data["sigma_star_ev"]:.3f} eV '
              f'({lumo_idx_label}, MO {rkf_data["sigma_star_mo"]}, '
              f'X_weight = {rkf_data["sigma_star_hal_weight"]:.3f})')
        print()

    if len(data_list) < 4:
        print("Pas assez de donnees!")
        return

    # -- Liste des descripteurs --
    descriptors = [
        ('bde', 'BDE C-X (kJ/mol)'),
        ('polarizability', 'Polarisabilite X (A^3)'),
        ('electron_affinity', 'EA(X) (eV)'),
        ('electronegativity', 'Electronegativite X (Pauling)'),
        ('cx_bond_length', 'Longueur C-X (A)'),
        ('lumo_ev', 'LUMO neutre (eV)'),
        ('sigma_star_ev', 'sigma*(C-X) (eV)'),
        ('homo_lumo_gap', 'Gap HOMO-LUMO (eV)'),
    ]

    # -- Correlations sur les 8 molecules halogenees --
    print('=' * 70)
    print('CORRELATIONS : RESIDU vs DESCRIPTEURS (8 molecules halogenees)')
    print('=' * 70)
    print()

    corr_results = compute_correlations(data_list, descriptors, 'residual')

    print(f'{"Descripteur":<35} {"R2":>6} {"r":>7} {"p":>8} '
          f'{"Pente":>10} {"MAE":>8} {"n":>3}')
    print('-' * 85)
    for r in corr_results:
        print(f'{r["descriptor"]:<35} {r["R2"]:>6.3f} {r["r"]:>+7.3f} {r["p_value"]:>8.4f} '
              f'{r["slope"]:>10.3f} {r["MAE"]:>8.1f} {r["n"]:>3}')
    print()

    # -- Correlations sur les 4 DEA uniquement --
    dea_data = [d for d in data_list if d['breaks']]
    if len(dea_data) >= 3:
        print('=' * 70)
        print('CORRELATIONS : RESIDU vs DESCRIPTEURS (4 molecules DEA seulement)')
        print('=' * 70)
        print()

        corr_dea = compute_correlations(dea_data, descriptors, 'residual')

        print(f'{"Descripteur":<35} {"R2":>6} {"r":>7} {"p":>8} '
              f'{"Pente":>10} {"MAE":>8} {"n":>3}')
        print('-' * 85)
        for r in corr_dea:
            print(f'{r["descriptor"]:<35} {r["R2"]:>6.3f} {r["r"]:>+7.3f} {r["p_value"]:>8.4f} '
                  f'{r["slope"]:>10.3f} {r["MAE"]:>8.1f} {r["n"]:>3}')
        print()

    # -- Correlations par famille --
    for family in ['DA', 'HMC']:
        fam_data = [d for d in data_list if d['family'] == family]
        if len(fam_data) >= 3:
            print(f'--- Famille {family} (n={len(fam_data)}) ---')
            corr_fam = compute_correlations(fam_data, descriptors, 'residual')
            for r in corr_fam[:3]:
                print(f'  {r["descriptor"]:<35} R2={r["R2"]:.3f} '
                      f'r={r["r"]:+.3f} p={r["p_value"]:.3f}')
            print()

    # -- Tableau recapitulatif --
    print('=' * 70)
    print('TABLEAU RECAPITULATIF')
    print('=' * 70)
    print()
    print(f'{"Molecule":<10} {"X":>3} {"BDE":>6} {"a(X)":>6} {"C-X(A)":>7} '
          f'{"LUMO":>7} {"s*(CX)":>8} {"idx":>4} {"E_relax":>8} {"Residu":>8} {"DEA?":>5}')
    print('-' * 85)
    for d in sorted(data_list, key=lambda x: x['residual']):
        idx = d['sigma_star_lumo_index']
        idx_label = {1: 'L', 2: 'L+1', 3: 'L+2'}.get(idx, f'L+{idx-1}')
        print(f'{d["molecule"]:<10} {d["halogen"]:>3} {d["bde"]:>6.0f} '
              f'{d["polarizability"]:>6.2f} {d["cx_bond_length"]:>7.3f} '
              f'{d["lumo_ev"]:>7.3f} {d["sigma_star_ev"]:>8.3f} '
              f'{idx_label:>4} '
              f'{d["E_relax"]:>8.1f} {d["residual"]:>8.1f} '
              f'{"OUI" if d["breaks"] else "non":>5}')
    print()

    # -- Plots --
    top_desc = [(r['desc_key'], r['descriptor']) for r in corr_results[:6]]
    make_correlation_plots(data_list, top_desc, 'residual',
                           'Residu = E_relax - E_harm (kJ/mol)',
                           output_dir, 'dea_correlations_all')

    if len(dea_data) >= 3:
        make_correlation_plots(dea_data, top_desc, 'residual',
                               'Residu DEA (kJ/mol)',
                               output_dir, 'dea_correlations_dea_only')

    make_correlation_plots(data_list, top_desc, 'E_relax',
                           'E_relax (kJ/mol)',
                           output_dir, 'erelax_correlations_all')

    # -- Export CSV --
    csv_path = output_dir / 'dea_descriptors.csv'
    fields = ['molecule', 'halogen', 'family', 'breaks', 'E_relax', 'E_lvc',
              'residual', 'bde', 'polarizability', 'electron_affinity',
              'electronegativity', 'cx_bond_length', 'lumo_ev', 'homo_ev',
              'sigma_star_ev', 'homo_lumo_gap']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for d in sorted(data_list, key=lambda x: x['molecule']):
            writer.writerow(d)
    print(f'  -> {csv_path}')

    corr_csv_path = output_dir / 'dea_correlations.csv'
    corr_fields = ['descriptor', 'R2', 'r', 'p_value', 'slope', 'intercept', 'MAE', 'RMSE', 'n']
    with open(corr_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=corr_fields, extrasaction='ignore')
        writer.writeheader()
        for r in corr_results:
            writer.writerow(r)
    print(f'  -> {corr_csv_path}')

    print()
    print('Termine.')


if __name__ == '__main__':
    main()
