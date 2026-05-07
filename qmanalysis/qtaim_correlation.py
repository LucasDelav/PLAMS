"""
Analyse de correlation entre descripteurs de densite QTAIM et EA/IP verticaux.

Ce module charge les cibles EA/IP depuis les resultats redox, les donnees
orbitales HOMO/LUMO, et orchestre l'analyse complete de correlation.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .common import (
    print_header, print_section, print_success, print_warning, print_error,
    EH_TO_EV, EH_TO_KJ,
)
from .correlation_analysis import (
    compute_boltzmann_averaged_descriptors,
    _read_conformer_weights_from_summary,
    _compute_single_correlation_stats,
)
from .qtaim_extraction import find_qtaim_neutral_rkf_files, extract_density_properties
from .qtaim_descriptors import compute_density_descriptors, get_density_descriptor_names


def load_ea_ip_targets(workdir, molecule_name: str) -> Dict[int, Dict[str, float]]:
    """
    Charge EA_kJ et IP_kJ depuis les resultats redox.

    Lit depuis: <workdir>/redox/results/<mol>_redox_conformers.csv

    Args:
        workdir (str or Path): Repertoire de travail principal
        molecule_name (str): Nom de la molecule

    Returns:
        Dict[int, Dict]: {conf_num: {'EA_kJ': float, 'IP_kJ': float, 'weight': float}}
    """
    workdir = Path(workdir)
    targets = {}

    csv_path = workdir / "redox" / "results" / f"{molecule_name}_redox_conformers.csv"

    if not csv_path.exists():
        print_warning(f"Fichier redox conformers non trouve: {csv_path}")
        return targets

    try:
        if HAS_PANDAS:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                conf_num = int(row['conformer'])
                targets[conf_num] = {
                    'EA_kJ': float(row.get('EA_kJ', np.nan)),
                    'IP_kJ': float(row.get('IP_kJ', np.nan)),
                    'Edef_red_kJ': float(row.get('Edef_red_kJ', np.nan)),
                    'Edef_ox_kJ': float(row.get('Edef_ox_kJ', np.nan)),
                    'weight': float(row.get('weight', 0.0)),
                }
        else:
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
                col_idx = {name: i for i, name in enumerate(header)}

                for line in f:
                    parts = line.strip().split(',')
                    conf_num = int(parts[col_idx['conformer']])
                    targets[conf_num] = {}
                    for key in ['EA_kJ', 'IP_kJ', 'Edef_red_kJ', 'Edef_ox_kJ', 'weight']:
                        idx = col_idx.get(key, -1)
                        if idx >= 0 and idx < len(parts) and parts[idx]:
                            targets[conf_num][key] = float(parts[idx])
                        else:
                            targets[conf_num][key] = np.nan

        print_success(f"Cibles EA/IP chargees: {len(targets)} conformeres")

    except Exception as e:
        print_error(f"Erreur lecture {csv_path}: {e}")

    return targets


def load_orbital_data(workdir, molecule_name: str) -> Dict[int, Dict[str, float]]:
    """
    Charge HOMO/LUMO depuis les resultats redox.

    Lit depuis: <workdir>/redox/results/<mol>_orbital_data.csv

    Pour comparaison Koopmans:
    - EA_koopmans = -LUMO (en Hartree) * EH_TO_KJ
    - IP_koopmans = -HOMO (en Hartree) * EH_TO_KJ

    Note: Les valeurs HOMO/LUMO dans le CSV sont en Hartree.

    Args:
        workdir (str or Path): Repertoire de travail principal
        molecule_name (str): Nom de la molecule

    Returns:
        Dict[int, Dict]: {conf_num: {'HOMO_Ha': float, 'LUMO_Ha': float,
                                      'gap_Ha': float, 'HOMO_eV': float,
                                      'LUMO_eV': float, 'gap_eV': float}}
    """
    workdir = Path(workdir)
    orbital = {}

    csv_path = workdir / "redox" / "results" / f"{molecule_name}_orbital_data.csv"

    if not csv_path.exists():
        print_warning(f"Fichier orbital_data non trouve: {csv_path}")
        return orbital

    try:
        if HAS_PANDAS:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                conf_num = int(row['conformer'])
                # Les colonnes sont HOMO_eV, LUMO_eV, gap_eV mais les valeurs
                # sont en Hartree (nommage historique)
                homo_ha = float(row.get('HOMO_eV', np.nan))
                lumo_ha = float(row.get('LUMO_eV', np.nan))
                gap_ha = float(row.get('gap_eV', np.nan))

                orbital[conf_num] = {
                    'HOMO_Ha': homo_ha,
                    'LUMO_Ha': lumo_ha,
                    'gap_Ha': gap_ha,
                    'HOMO_eV': homo_ha * EH_TO_EV,
                    'LUMO_eV': lumo_ha * EH_TO_EV,
                    'gap_eV': gap_ha * EH_TO_EV,
                }
        else:
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
                col_idx = {name: i for i, name in enumerate(header)}

                for line in f:
                    parts = line.strip().split(',')
                    conf_num = int(parts[col_idx['conformer']])
                    homo_ha = float(parts[col_idx.get('HOMO_eV', -1)]) if 'HOMO_eV' in col_idx else np.nan
                    lumo_ha = float(parts[col_idx.get('LUMO_eV', -1)]) if 'LUMO_eV' in col_idx else np.nan
                    gap_ha = float(parts[col_idx.get('gap_eV', -1)]) if 'gap_eV' in col_idx else np.nan

                    orbital[conf_num] = {
                        'HOMO_Ha': homo_ha,
                        'LUMO_Ha': lumo_ha,
                        'gap_Ha': gap_ha,
                        'HOMO_eV': homo_ha * EH_TO_EV,
                        'LUMO_eV': lumo_ha * EH_TO_EV,
                        'gap_eV': gap_ha * EH_TO_EV,
                    }

        print_success(f"Donnees orbitales chargees: {len(orbital)} conformeres")

    except Exception as e:
        print_error(f"Erreur lecture {csv_path}: {e}")

    return orbital


def _compute_density_correlations(
    descriptors: Dict[str, Dict[str, float]],
    target_values: Dict[str, Dict[str, float]],
    molecule_names: List[str],
    target_name: str,
) -> Any:
    """
    Calcule les correlations entre tous les descripteurs de densite et une cible.

    Args:
        descriptors: {mol_name: {descriptor: value}}
        target_values: {mol_name: {target_name: value}}
        molecule_names: Liste des molecules
        target_name: Nom de la cible ('EA_kJ' ou 'IP_kJ')

    Returns:
        DataFrame ou list de dicts avec les correlations completes
    """
    descriptor_names = get_density_descriptor_names()

    # Construire les vecteurs
    X = {}
    Y = []

    for mol_name in molecule_names:
        mol_desc = descriptors.get(mol_name, {})
        mol_target = target_values.get(mol_name, {})

        for name in descriptor_names:
            if name not in X:
                X[name] = []
            X[name].append(mol_desc.get(name, np.nan))

        Y.append(mol_target.get(target_name, np.nan))

    y_arr = np.array(Y)

    # Calculer les correlations
    correlations = []

    for desc_name in descriptor_names:
        if desc_name not in X:
            continue

        x_arr = np.array(X[desc_name])
        mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)

        if np.sum(mask) < 3:
            continue

        x_valid = x_arr[mask]
        y_valid = y_arr[mask]

        # Verifier qu'il y a de la variance
        if np.std(x_valid) == 0 or np.std(y_valid) == 0:
            continue

        stats = _compute_single_correlation_stats(x_valid, y_valid)
        stats['descriptor'] = desc_name
        stats['target'] = target_name
        correlations.append(stats)

    if HAS_PANDAS:
        df = pd.DataFrame(correlations)
        if not df.empty:
            df = df.sort_values('r2', ascending=False)
        return df
    else:
        correlations.sort(
            key=lambda x: x.get('r2', 0) if not np.isnan(x.get('r2', 0)) else 0,
            reverse=True
        )
        return correlations


def run_density_analysis(
    workdirs: List[Path],
    output_dir: Path
) -> Tuple[Any, Any]:
    """
    Pipeline complet d'analyse de correlation densite QTAIM vs EA/IP.

    1. Pour chaque molecule: extraction QTAIM, descripteurs, cibles
    2. Moyennes de Boltzmann
    3. Correlations globales
    4. Comparaison Koopmans
    5. Export CSVs

    Args:
        workdirs: Liste des repertoires de travail des molecules
        output_dir: Repertoire de sortie pour les resultats

    Returns:
        Tuple (df_correlations, df_descriptors)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("ANALYSE DENSITE ELECTRONIQUE QTAIM vs EA/IP")

    all_descriptors = {}   # {mol_name: {descriptor: value}}
    all_targets = {}       # {mol_name: {'EA_kJ': float, 'IP_kJ': float}}
    all_orbital = {}       # {mol_name: {'HOMO_eV': float, 'LUMO_eV': float, ...}}
    molecule_names = []

    for workdir in workdirs:
        workdir = Path(workdir)
        mol_name = workdir.name.replace('_workdir', '')
        molecule_names.append(mol_name)

        print_section(f"Traitement de {mol_name}")

        # 1. Trouver les fichiers RKF QTAIM neutres
        rkf_files = find_qtaim_neutral_rkf_files(workdir, mol_name)

        if not rkf_files:
            print_warning(f"Aucun fichier RKF QTAIM neutre pour {mol_name}")
            all_descriptors[mol_name] = {}
            all_targets[mol_name] = {'EA_kJ': np.nan, 'IP_kJ': np.nan}
            all_orbital[mol_name] = {}
            continue

        print(f"  {len(rkf_files)} fichiers RKF QTAIM neutres trouves")

        # 2. Extraire les proprietes de densite pour chaque conformere
        descriptors_by_conf = {}
        for conf_num, rkf_path in sorted(rkf_files.items()):
            density_data = extract_density_properties(rkf_path)
            if density_data['success']:
                desc = compute_density_descriptors(density_data)
                if desc:
                    descriptors_by_conf[conf_num] = desc

        print(f"  {len(descriptors_by_conf)} conformeres analyses avec succes")

        # 3. Charger les poids de Boltzmann
        weights = _read_conformer_weights_from_summary(workdir)
        if weights is None:
            # Fallback depuis les donnees redox
            ea_ip = load_ea_ip_targets(workdir, mol_name)
            weights = {conf: data.get('weight', 0.0) for conf, data in ea_ip.items()}
        else:
            ea_ip = load_ea_ip_targets(workdir, mol_name)
            print_success("Poids de Boltzmann lus depuis conformers_summary.txt")

        # 4. Moyenner les descripteurs
        avg_desc = compute_boltzmann_averaged_descriptors(descriptors_by_conf, weights)
        all_descriptors[mol_name] = avg_desc

        # 5. Moyenner les cibles EA/IP
        if ea_ip:
            total_w = sum(weights.get(c, 0) for c in ea_ip)
            if total_w > 0:
                avg_ea = sum(weights.get(c, 0) * d.get('EA_kJ', 0)
                             for c, d in ea_ip.items()) / total_w
                avg_ip = sum(weights.get(c, 0) * d.get('IP_kJ', 0)
                             for c, d in ea_ip.items()) / total_w
                all_targets[mol_name] = {'EA_kJ': avg_ea, 'IP_kJ': avg_ip}
            else:
                all_targets[mol_name] = {'EA_kJ': np.nan, 'IP_kJ': np.nan}
        else:
            all_targets[mol_name] = {'EA_kJ': np.nan, 'IP_kJ': np.nan}

        # 6. Charger et moyenner les donnees orbitales
        orbital = load_orbital_data(workdir, mol_name)
        if orbital:
            total_w = sum(weights.get(c, 0) for c in orbital)
            if total_w > 0:
                avg_orb = {}
                for key in ['HOMO_Ha', 'LUMO_Ha', 'gap_Ha', 'HOMO_eV', 'LUMO_eV', 'gap_eV']:
                    avg_orb[key] = sum(
                        weights.get(c, 0) * d.get(key, 0)
                        for c, d in orbital.items()
                    ) / total_w
                all_orbital[mol_name] = avg_orb
            else:
                all_orbital[mol_name] = {}
        else:
            all_orbital[mol_name] = {}

        # Afficher un apercu
        if avg_desc:
            rho_mean = avg_desc.get('rho_heavy_mean', np.nan)
            if not np.isnan(rho_mean):
                print(f"  rho(heavy) moyen: {rho_mean:.4f}")
        print(f"  EA: {all_targets[mol_name]['EA_kJ']:.2f} kJ/mol")
        print(f"  IP: {all_targets[mol_name]['IP_kJ']:.2f} kJ/mol")
        if all_orbital.get(mol_name):
            print(f"  HOMO: {all_orbital[mol_name].get('HOMO_eV', np.nan):.3f} eV")
            print(f"  LUMO: {all_orbital[mol_name].get('LUMO_eV', np.nan):.3f} eV")

    # ========================================================================
    # Correlations globales
    # ========================================================================
    print_section("Calcul des correlations densite vs EA/IP")

    df_corr_ea = _compute_density_correlations(
        all_descriptors, all_targets, molecule_names, 'EA_kJ'
    )
    df_corr_ip = _compute_density_correlations(
        all_descriptors, all_targets, molecule_names, 'IP_kJ'
    )

    # ========================================================================
    # Construire le DataFrame des descripteurs
    # ========================================================================
    descriptor_names = get_density_descriptor_names()
    desc_data = {'molecule': molecule_names}
    for name in descriptor_names:
        desc_data[name] = [all_descriptors.get(m, {}).get(name, np.nan) for m in molecule_names]

    # Ajouter les cibles
    desc_data['EA_kJ'] = [all_targets.get(m, {}).get('EA_kJ', np.nan) for m in molecule_names]
    desc_data['IP_kJ'] = [all_targets.get(m, {}).get('IP_kJ', np.nan) for m in molecule_names]

    # Ajouter les donnees orbitales
    for key in ['HOMO_eV', 'LUMO_eV', 'gap_eV']:
        desc_data[key] = [all_orbital.get(m, {}).get(key, np.nan) for m in molecule_names]

    if HAS_PANDAS:
        df_desc = pd.DataFrame(desc_data)
    else:
        df_desc = desc_data

    # ========================================================================
    # Comparaison Koopmans
    # ========================================================================
    print_section("Comparaison avec le theoreme de Koopmans")

    koopmans_records = []
    for mol_name in molecule_names:
        orb = all_orbital.get(mol_name, {})
        tgt = all_targets.get(mol_name, {})

        record = {
            'molecule': mol_name,
            'EA_DFT_kJ': tgt.get('EA_kJ', np.nan),
            'IP_DFT_kJ': tgt.get('IP_kJ', np.nan),
        }

        # Koopmans: EA ≈ -LUMO, IP ≈ -HOMO (en Hartree, converti en kJ/mol)
        lumo_ha = orb.get('LUMO_Ha', np.nan)
        homo_ha = orb.get('HOMO_Ha', np.nan)

        if not np.isnan(lumo_ha):
            record['EA_koopmans_kJ'] = -lumo_ha * EH_TO_KJ
        else:
            record['EA_koopmans_kJ'] = np.nan

        if not np.isnan(homo_ha):
            record['IP_koopmans_kJ'] = -homo_ha * EH_TO_KJ
        else:
            record['IP_koopmans_kJ'] = np.nan

        record['HOMO_eV'] = orb.get('HOMO_eV', np.nan)
        record['LUMO_eV'] = orb.get('LUMO_eV', np.nan)
        record['gap_eV'] = orb.get('gap_eV', np.nan)

        koopmans_records.append(record)

    # ========================================================================
    # Export
    # ========================================================================
    print_section("Export des resultats")

    if HAS_PANDAS:
        # Export correlations EA
        if isinstance(df_corr_ea, pd.DataFrame) and not df_corr_ea.empty:
            ea_csv = output_dir / "density_correlations_EA.csv"
            df_corr_ea.to_csv(ea_csv, index=False)
            print_success(f"Correlations EA exportees: {ea_csv}")

        # Export correlations IP
        if isinstance(df_corr_ip, pd.DataFrame) and not df_corr_ip.empty:
            ip_csv = output_dir / "density_correlations_IP.csv"
            df_corr_ip.to_csv(ip_csv, index=False)
            print_success(f"Correlations IP exportees: {ip_csv}")

        # Export descripteurs
        desc_csv = output_dir / "density_descriptors.csv"
        df_desc.to_csv(desc_csv, index=False)
        print_success(f"Descripteurs exportes: {desc_csv}")

        # Export cibles EA/IP
        targets_records = []
        for mol_name in molecule_names:
            rec = {'molecule': mol_name}
            rec.update(all_targets.get(mol_name, {}))
            rec.update({f'orb_{k}': v for k, v in all_orbital.get(mol_name, {}).items()})
            targets_records.append(rec)
        df_targets = pd.DataFrame(targets_records)
        targets_csv = output_dir / "ea_ip_targets.csv"
        df_targets.to_csv(targets_csv, index=False)
        print_success(f"Cibles EA/IP exportees: {targets_csv}")

        # Export comparaison Koopmans
        df_koopmans = pd.DataFrame(koopmans_records)
        koopmans_csv = output_dir / "koopmans_comparison.csv"
        df_koopmans.to_csv(koopmans_csv, index=False)
        print_success(f"Comparaison Koopmans exportee: {koopmans_csv}")

        # Afficher les meilleures correlations
        print("\nMeilleures correlations avec EA_kJ:")
        print("-" * 80)
        if isinstance(df_corr_ea, pd.DataFrame) and not df_corr_ea.empty:
            for _, row in df_corr_ea.head(10).iterrows():
                print(f"  {row['descriptor']:35s}  R²={row['r2']:.3f}  "
                      f"p={row['pearson_p']:.2e}  rho={row['spearman_rho']:.3f}")

        print("\nMeilleures correlations avec IP_kJ:")
        print("-" * 80)
        if isinstance(df_corr_ip, pd.DataFrame) and not df_corr_ip.empty:
            for _, row in df_corr_ip.head(10).iterrows():
                print(f"  {row['descriptor']:35s}  R²={row['r2']:.3f}  "
                      f"p={row['pearson_p']:.2e}  rho={row['spearman_rho']:.3f}")

        # Afficher la comparaison Koopmans
        print("\nComparaison Koopmans:")
        print("-" * 80)
        from scipy.stats import pearsonr
        ea_dft = df_koopmans['EA_DFT_kJ'].values
        ea_koop = df_koopmans['EA_koopmans_kJ'].values
        ip_dft = df_koopmans['IP_DFT_kJ'].values
        ip_koop = df_koopmans['IP_koopmans_kJ'].values

        mask_ea = ~np.isnan(ea_dft) & ~np.isnan(ea_koop)
        mask_ip = ~np.isnan(ip_dft) & ~np.isnan(ip_koop)

        if np.sum(mask_ea) >= 3:
            r_ea, _ = pearsonr(ea_dft[mask_ea], ea_koop[mask_ea])
            print(f"  EA_koopmans vs EA_DFT:  R² = {r_ea**2:.3f}  (n={np.sum(mask_ea)})")
        if np.sum(mask_ip) >= 3:
            r_ip, _ = pearsonr(ip_dft[mask_ip], ip_koop[mask_ip])
            print(f"  IP_koopmans vs IP_DFT:  R² = {r_ip**2:.3f}  (n={np.sum(mask_ip)})")

    else:
        # Export texte sans pandas
        corr_txt = output_dir / "density_correlations.txt"
        with open(corr_txt, 'w') as f:
            f.write("# Correlations densite vs EA/IP\n")
            f.write("# descriptor,target,r2,pearson_p,spearman_rho,n\n")
            for corr_list in [df_corr_ea, df_corr_ip]:
                if isinstance(corr_list, list):
                    for row in corr_list:
                        f.write(f"{row['descriptor']},{row['target']},"
                                f"{row['r2']:.6f},{row['pearson_p']:.6e},"
                                f"{row['spearman_rho']:.6f},{row['n']}\n")
        print_success(f"Correlations exportees: {corr_txt}")

    return df_corr_ea, df_desc
