"""
Analyse des résultats redox : extraction d'énergies, calculs de potentiels
avec schéma composite TZ2P+DZP, moyennes de Boltzmann et export des résultats.
"""

import os
import re

import numpy as np
from scm.plams import KFReader

from .common import R, F, hartree_to_kj, kj_to_volt, hartree_to_ev, kabsch_rmsd
from common.bond_detection import compare_bonds, BOHR_TO_ANG, Z_TO_SYMBOL


def extract_energies_from_rkf(rkf_path, method="direct"):
    """
    Extrait les énergies thermodynamiques depuis un fichier adf.rkf.

    Args:
        rkf_path (str): Chemin vers le fichier adf.rkf
        method (str): Méthode de calcul de U ("direct" ou "h_minus_rt")

    Returns:
        dict: Dictionnaire contenant les énergies en Hartree
              {'Ee', 'H', 'G', 'TS', 'U', 'T', 'P', 'HOMO', 'LUMO'}
              Pour les SinglePoints sans thermo, H/G/TS/U seront None.
    """
    try:
        kf_reader = KFReader(rkf_path)

        energies = {}

        # Énergie électronique (toujours disponible)
        energies['Ee'] = kf_reader.read("Energy", "SCF Bond Energy")

        # Thermodynamique (peut ne pas être disponible pour les SinglePoints)
        try:
            energies['H'] = kf_reader.read("Thermodynamics", "Enthalpy")
            energies['G'] = kf_reader.read("Thermodynamics", "Gibbs free Energy")
            energies['TS'] = kf_reader.read("Thermodynamics", "TS")
            energies['T'] = kf_reader.read("Thermodynamics", "Temperature")
            energies['P'] = kf_reader.read("Thermodynamics", "Pressure")

            if method == "direct":
                try:
                    energies['U'] = kf_reader.read("Thermodynamics", "Internal Energy total")
                except KeyError:
                    T = energies['T']
                    R_ha = R / 4.184 / 627.509
                    energies['U'] = energies['H'] - R_ha * T
            elif method == "h_minus_rt":
                T = energies['T']
                R_ha = R / 4.184 / 627.509
                energies['U'] = energies['H'] - R_ha * T
        except (KeyError, Exception):
            # SP sans thermo
            energies['H'] = None
            energies['G'] = None
            energies['TS'] = None
            energies['U'] = None
            energies['T'] = None
            energies['P'] = None

        # Orbitales frontières
        try:
            energies['HOMO'] = hartree_to_ev(kf_reader.read("Properties", "HOMO"))
            energies['LUMO'] = hartree_to_ev(kf_reader.read("Properties", "LUMO"))
        except KeyError:
            energies['HOMO'] = None
            energies['LUMO'] = None

        return energies

    except Exception as e:
        print(f"Erreur lors de l'extraction des énergies depuis {rkf_path}: {str(e)}")
        return None


def extract_all_energies(jobs_data, u_method="direct"):
    """
    Extrait les énergies de tous les jobs organisés par conformère.

    Gère le cas où 'neutral_rkf' est un chemin (str) et non un job PLAMS,
    et où 'neutral_tz2p', 'reduced_tz2p', 'oxidized_tz2p' sont des jobs SP.

    Args:
        jobs_data (dict): {conf_num: {'neutral_rkf': str, 'neutral_tz2p': job, ...}}
        u_method (str): Méthode de calcul de U

    Returns:
        dict: {conf_num: {'neutral': energies, 'neutral_tz2p': energies, ...}}
    """
    all_energies = {}

    for conf_num, jobs in jobs_data.items():
        all_energies[conf_num] = {}

        for job_type, job_or_path in jobs.items():
            if job_or_path is None:
                all_energies[conf_num][job_type] = None
                continue

            try:
                if job_type == 'neutral_rkf':
                    # C'est un chemin vers adf.rkf (str)
                    rkf_path = str(job_or_path)
                    energies = extract_energies_from_rkf(rkf_path, method=u_method)
                    all_energies[conf_num]['neutral'] = energies
                else:
                    # C'est un job PLAMS
                    rkf_path = job_or_path.results.rkfpath(file='adf')
                    energies = extract_energies_from_rkf(rkf_path, method=u_method)
                    all_energies[conf_num][job_type] = energies

            except Exception as e:
                print(f"Erreur extraction conf {conf_num} {job_type}: {str(e)}")
                key = 'neutral' if job_type == 'neutral_rkf' else job_type
                all_energies[conf_num][key] = None

    return all_energies


def _is_section_header(line: str) -> bool:
    """Détecte l'en-tête de section CONFORMÈRES FINAUX."""
    return "CONFORMÈRES FINAUX" in line


def _is_separator(line: str) -> bool:
    """Détecte les lignes de séparation."""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("===") or stripped.startswith("---"):
        return True
    if "#" in stripped and "ΔE" in stripped:
        return True
    return False


def _is_end_of_section(line: str) -> bool:
    """Détecte la fin de la section des conformères."""
    stripped = line.strip()
    return stripped.startswith("Fichiers XYZ") or stripped.startswith("Note:")


def _parse_conformer_line(line: str):
    """Extrait le numéro de conformère et son poids de Boltzmann."""
    parts = line.strip().split()
    if len(parts) >= 3:
        try:
            conf_idx = int(parts[0])
            weight = float(parts[2])
            return (conf_idx, weight)
        except (ValueError, IndexError):
            return None
    return None


def read_conformer_weights_from_summary(workdir):
    """
    Lit les poids de Boltzmann depuis conformers_summary.txt.

    Args:
        workdir (Path or str): Répertoire de travail principal

    Returns:
        dict: {conf_num: weight} ou None si fichier absent
    """
    from pathlib import Path

    workdir = Path(workdir)
    summary_file = workdir / "conformers" / "results" / "conformers_summary.txt"

    if not summary_file.exists():
        return None

    weights = {}
    in_conformers_section = False

    with open(summary_file, 'r') as f:
        for line in f:
            if _is_section_header(line):
                in_conformers_section = True
                continue

            if in_conformers_section:
                if _is_end_of_section(line):
                    break
                if _is_separator(line):
                    continue
                result = _parse_conformer_line(line)
                if result:
                    conf_idx, weight = result
                    weights[conf_idx] = weight

    if weights:
        print(f"✓ Poids de Boltzmann lus depuis {summary_file}")
        return weights
    else:
        return None


def calculate_boltzmann_weights(conformers_energies, temperature=298.15):
    """
    Calcule les poids de Boltzmann depuis les énergies de Gibbs des neutres.

    NOTE: Fallback si conformers_summary.txt absent.
    """
    from common.utils import boltzmann_weights
    from .common import hartree_to_kj

    gibbs_energies = {}
    for conf_num, jobs_energies in conformers_energies.items():
        if jobs_energies.get('neutral') and jobs_energies['neutral'].get('G') is not None:
            gibbs_energies[conf_num] = jobs_energies['neutral']['G']

    if not gibbs_energies:
        print("Aucune énergie de Gibbs disponible pour les poids de Boltzmann")
        return {}

    min_G = min(gibbs_energies.values())
    conf_nums = sorted(gibbs_energies.keys())
    relative_energies_kj = [hartree_to_kj(gibbs_energies[conf] - min_G)
                           for conf in conf_nums]

    weights_array, Z = boltzmann_weights(relative_energies_kj, temperature)

    weights = {conf_num: float(weight)
              for conf_num, weight in zip(conf_nums, weights_array)}

    return weights


def calculate_redox_parameters(conformers_energies):
    """
    Calcule les paramètres redox avec le schéma composite TZ2P+DZP.

    Énergie composite : G_composite = Ee(TZ2P) + [G(DZP) - Ee(DZP)]
    ΔG = G_composite(ion) - G_composite(neutral)

    EA/IP en TZ2P (bases cohérentes pour tous les SP).
    Edef en TZ2P (ion opt - ion sp, déformation géométrique pure).
    ΔΔU et TΔS en DZP (corrections thermiques).

    Args:
        conformers_energies (dict): {conf_num: {'neutral': energies,
            'neutral_tz2p': energies, 'reduced_sp': energies,
            'reduced_opt': energies, 'reduced_tz2p': energies,
            'oxidized_sp': energies, 'oxidized_opt': energies,
            'oxidized_tz2p': energies}}

    Returns:
        dict: {conf_num: {'reduction': params_dict, 'oxidation': params_dict}}
    """
    redox_params = {}

    for conf_num, e in conformers_energies.items():
        # Vérifier les données requises
        required = ['neutral', 'neutral_tz2p', 'reduced_sp', 'reduced_opt',
                    'reduced_tz2p', 'oxidized_sp', 'oxidized_opt', 'oxidized_tz2p']

        if not all(e.get(k) for k in required):
            missing = [k for k in required if not e.get(k)]
            print(f"Avertissement : Conformère {conf_num} incomplet (manque: {missing}), ignoré")
            continue

        neutral = e['neutral']
        neutral_tz2p = e['neutral_tz2p']
        red_sp = e['reduced_sp']
        red_opt = e['reduced_opt']
        red_tz2p = e['reduced_tz2p']
        ox_sp = e['oxidized_sp']
        ox_opt = e['oxidized_opt']
        ox_tz2p = e['oxidized_tz2p']

        # Vérifier que thermo est dispo pour neutral et les opts
        if any(v is None for v in [neutral.get('G'), neutral.get('U'), neutral.get('TS'),
                                    red_opt.get('G'), red_opt.get('U'), red_opt.get('TS'),
                                    ox_opt.get('G'), ox_opt.get('U'), ox_opt.get('TS')]):
            print(f"Avertissement : Données thermo incomplètes pour conformère {conf_num}, ignoré")
            continue

        # ============================================================
        # RÉDUCTION
        # ============================================================
        reduction = {}

        # ΔG composite TZ2P
        # G_composite = Ee(TZ2P) + [G(DZP) - Ee(DZP)]
        G_composite_neutral = neutral_tz2p['Ee'] + (neutral['G'] - neutral['Ee'])
        G_composite_red = red_tz2p['Ee'] + (red_opt['G'] - red_opt['Ee'])
        reduction['delta_G_ha'] = G_composite_red - G_composite_neutral
        reduction['delta_G_kj'] = hartree_to_kj(reduction['delta_G_ha'])

        # EA (TZ2P cohérent)
        reduction['EA_ha'] = neutral_tz2p['Ee'] - red_sp['Ee']
        reduction['EA_kj'] = hartree_to_kj(reduction['EA_ha'])

        # Edef (TZ2P cohérent : ion géom opt - ion géom neutre)
        reduction['Edef_ha'] = red_tz2p['Ee'] - red_sp['Ee']
        reduction['Edef_kj'] = hartree_to_kj(reduction['Edef_ha'])

        # ΔΔU (corrections thermiques DZP) : [U-Ee](ion) - [U-Ee](neutral)
        reduction['delta_delta_U_ha'] = (red_opt['U'] - red_opt['Ee']) - (neutral['U'] - neutral['Ee'])
        reduction['delta_delta_U_kj'] = hartree_to_kj(reduction['delta_delta_U_ha'])

        # TΔS (corrections DZP)
        reduction['T_delta_S_ha'] = red_opt['TS'] - neutral['TS']
        reduction['T_delta_S_kj'] = hartree_to_kj(reduction['T_delta_S_ha'])

        # Résidu (capture la différence de base TZ2P vs DZP)
        # ΔG = -EA + Edef + ΔΔU + TΔS + résidu
        sum_contributions = (-reduction['EA_ha'] + reduction['Edef_ha'] +
                            reduction['delta_delta_U_ha'] + reduction['T_delta_S_ha'])
        reduction['residual_ha'] = reduction['delta_G_ha'] - sum_contributions
        reduction['residual_kj'] = hartree_to_kj(reduction['residual_ha'])

        # ============================================================
        # OXYDATION
        # ============================================================
        oxidation = {}

        # ΔG composite TZ2P
        G_composite_ox = ox_tz2p['Ee'] + (ox_opt['G'] - ox_opt['Ee'])
        oxidation['delta_G_ha'] = G_composite_ox - G_composite_neutral
        oxidation['delta_G_kj'] = hartree_to_kj(oxidation['delta_G_ha'])

        # IP (TZ2P cohérent)
        oxidation['IP_ha'] = ox_sp['Ee'] - neutral_tz2p['Ee']
        oxidation['IP_kj'] = hartree_to_kj(oxidation['IP_ha'])

        # Edef (TZ2P cohérent : ion géom opt - ion géom neutre)
        oxidation['Edef_ha'] = ox_tz2p['Ee'] - ox_sp['Ee']
        oxidation['Edef_kj'] = hartree_to_kj(oxidation['Edef_ha'])

        # ΔΔU (corrections thermiques DZP) : [U-Ee](ion) - [U-Ee](neutral)
        oxidation['delta_delta_U_ha'] = (ox_opt['U'] - ox_opt['Ee']) - (neutral['U'] - neutral['Ee'])
        oxidation['delta_delta_U_kj'] = hartree_to_kj(oxidation['delta_delta_U_ha'])

        # TΔS (corrections DZP)
        oxidation['T_delta_S_ha'] = ox_opt['TS'] - neutral['TS']
        oxidation['T_delta_S_kj'] = hartree_to_kj(oxidation['T_delta_S_ha'])

        # Résidu
        sum_contributions = (oxidation['IP_ha'] + oxidation['Edef_ha'] +
                            oxidation['delta_delta_U_ha'] + oxidation['T_delta_S_ha'])
        oxidation['residual_ha'] = oxidation['delta_G_ha'] - sum_contributions
        oxidation['residual_kj'] = hartree_to_kj(oxidation['residual_ha'])

        redox_params[conf_num] = {
            'reduction': reduction,
            'oxidation': oxidation
        }

    return redox_params


def calculate_boltzmann_averaged_potentials(redox_params, boltzmann_weights):
    """
    Calcule les potentiels redox moyennés par les poids de Boltzmann.

    Args:
        redox_params (dict): {conf_num: {'reduction': params, 'oxidation': params}}
        boltzmann_weights (dict): {conf_num: weight}

    Returns:
        dict: {'reduction': {...}, 'oxidation': {...}} avec potentiels moyennés
    """
    avg_reduction = {
        'delta_G_kj': 0.0,
        'EA_kj': 0.0,
        'Edef_kj': 0.0,
        'delta_delta_U_kj': 0.0,
        'T_delta_S_kj': 0.0,
        'residual_kj': 0.0
    }

    avg_oxidation = {
        'delta_G_kj': 0.0,
        'IP_kj': 0.0,
        'Edef_kj': 0.0,
        'delta_delta_U_kj': 0.0,
        'T_delta_S_kj': 0.0,
        'residual_kj': 0.0
    }

    total_weight = 0.0
    for conf_num, weight in boltzmann_weights.items():
        if conf_num not in redox_params:
            continue

        total_weight += weight
        reduction_data = redox_params[conf_num]['reduction']
        oxidation_data = redox_params[conf_num]['oxidation']

        for key in avg_reduction.keys():
            avg_reduction[key] += weight * reduction_data[key]

        for key in avg_oxidation.keys():
            avg_oxidation[key] += weight * oxidation_data[key]

    if total_weight > 0 and total_weight != 1.0:
        for key in avg_reduction:
            avg_reduction[key] /= total_weight
        for key in avg_oxidation:
            avg_oxidation[key] /= total_weight

    reduction_potentials = {
        'E_deltaG': kj_to_volt(avg_reduction['delta_G_kj']),
        'E_EA': kj_to_volt(avg_reduction['EA_kj']),
        'E_Edef': kj_to_volt(avg_reduction['Edef_kj']),
        'E_delta_delta_U': kj_to_volt(avg_reduction['delta_delta_U_kj']),
        'E_T_delta_S': kj_to_volt(avg_reduction['T_delta_S_kj']),
        'residual_V': kj_to_volt(avg_reduction['residual_kj'])
    }

    oxidation_potentials = {
        'E_deltaG': kj_to_volt(avg_oxidation['delta_G_kj']),
        'E_IP': kj_to_volt(avg_oxidation['IP_kj']),
        'E_Edef': kj_to_volt(avg_oxidation['Edef_kj']),
        'E_delta_delta_U': kj_to_volt(avg_oxidation['delta_delta_U_kj']),
        'E_T_delta_S': kj_to_volt(avg_oxidation['T_delta_S_kj']),
        'residual_V': kj_to_volt(avg_oxidation['residual_kj'])
    }

    return {
        'reduction': reduction_potentials,
        'oxidation': oxidation_potentials,
        'reduction_kj': avg_reduction,
        'oxidation_kj': avg_oxidation
    }


def compare_conformers_rmsd(jobs_data):
    """
    Compare les structures optimisées avec la structure neutre via RMSD Kabsch.

    Args:
        jobs_data (dict): {mol_name: {conf_num: {'neutral_rkf': str, 'reduced_opt': job, ...}}}

    Returns:
        tuple: (rmsd_results, warnings)
    """
    rmsd_results = {}
    warnings = []
    threshold_rmsd = 0.3

    for mol_name, conformers in jobs_data.items():
        rmsd_results[mol_name] = {}

        for conf_num, jobs in conformers.items():
            if not all(jobs.get(k) for k in ['reduced_opt', 'oxidized_opt']):
                continue

            try:
                # Neutre : extraire depuis le job reduced_sp (même géométrie)
                # ou depuis reduced_opt initial mol
                mol_neutral = jobs['reduced_sp'].results.get_input_molecule() if jobs.get('reduced_sp') else None
                if mol_neutral is None:
                    continue

                mol_reduced = jobs['reduced_opt'].results.get_main_molecule()
                mol_oxidized = jobs['oxidized_opt'].results.get_main_molecule()

                rmsd_red = kabsch_rmsd(mol_neutral, mol_reduced)
                rmsd_ox = kabsch_rmsd(mol_neutral, mol_oxidized)

                rmsd_results[mol_name][conf_num] = {
                    'reduced': rmsd_red,
                    'oxidized': rmsd_ox
                }

                if rmsd_red > threshold_rmsd:
                    warnings.append(
                        f"{mol_name} conf {conf_num}: RMSD réduit = {rmsd_red:.3f} Å "
                        f"(changement conformationnel possible)"
                    )

                if rmsd_ox > threshold_rmsd:
                    warnings.append(
                        f"{mol_name} conf {conf_num}: RMSD oxydé = {rmsd_ox:.3f} Å "
                        f"(changement conformationnel possible)"
                    )

            except Exception as e:
                print(f"Erreur RMSD pour {mol_name} conf {conf_num}: {str(e)}")
                continue

    return rmsd_results, warnings


def detect_bond_breaking(jobs_data, threshold=1.0):
    """
    Détecte les ruptures de liaison entre géométries neutre et chargée.

    Args:
        jobs_data: {mol_name: {conf_num: {'neutral_rkf': str, 'reduced_opt': job, ...}}}
        threshold: élongation en Å au-delà de laquelle une liaison est rompue

    Returns:
        tuple: (bond_data, warnings)
    """
    bond_data = {}
    warnings = []

    for mol_name, conformers in jobs_data.items():
        bond_data[mol_name] = {}

        for conf_num, jobs in conformers.items():
            if not jobs.get('neutral_rkf'):
                continue

            result = {'reduced': [], 'oxidized': []}

            try:
                # Extraire géométrie neutre depuis le RKF
                neutral_rkf = str(jobs['neutral_rkf'])
                kf_n = KFReader(neutral_rkf)
                n_atoms = kf_n.read('Molecule', 'nAtoms')
                atomic_numbers = kf_n.read('Molecule', 'AtomicNumbers')
                symbols = [Z_TO_SYMBOL.get(z, f'Z{z}') for z in atomic_numbers]
                coords_n = np.array(kf_n.read('Molecule', 'Coords'), dtype=float).reshape(n_atoms, 3) * BOHR_TO_ANG

                # Comparer avec réduit
                if jobs.get('reduced_opt'):
                    try:
                        red_rkf = jobs['reduced_opt'].results.rkfpath(file='adf')
                        kf_r = KFReader(str(red_rkf))
                        coords_r = np.array(kf_r.read('Molecule', 'Coords'), dtype=float).reshape(n_atoms, 3) * BOHR_TO_ANG
                        broken_red = compare_bonds(symbols, coords_n, coords_r, threshold)
                        result['reduced'] = broken_red
                        for b in broken_red:
                            warnings.append(
                                f"{mol_name} conf {conf_num} (réduit): liaison {b['bond_type']} rompue "
                                f"(Δd = +{b['elongation']:.3f} Å)"
                            )
                    except Exception as e:
                        print(f"  ⚠ Erreur bond-breaking réduit {mol_name} conf {conf_num}: {e}")

                # Comparer avec oxydé
                if jobs.get('oxidized_opt'):
                    try:
                        ox_rkf = jobs['oxidized_opt'].results.rkfpath(file='adf')
                        kf_o = KFReader(str(ox_rkf))
                        coords_o = np.array(kf_o.read('Molecule', 'Coords'), dtype=float).reshape(n_atoms, 3) * BOHR_TO_ANG
                        broken_ox = compare_bonds(symbols, coords_n, coords_o, threshold)
                        result['oxidized'] = broken_ox
                        for b in broken_ox:
                            warnings.append(
                                f"{mol_name} conf {conf_num} (oxydé): liaison {b['bond_type']} rompue "
                                f"(Δd = +{b['elongation']:.3f} Å)"
                            )
                    except Exception as e:
                        print(f"  ⚠ Erreur bond-breaking oxydé {mol_name} conf {conf_num}: {e}")

            except Exception as e:
                print(f"  ⚠ Erreur bond-breaking {mol_name} conf {conf_num}: {e}")
                continue

            bond_data[mol_name][conf_num] = result

    return bond_data, warnings


def extract_orbital_energies(jobs_data, boltzmann_weights):
    """
    Extrait les énergies HOMO/LUMO et calcule les moyennes pondérées de Boltzmann.

    Args:
        jobs_data (dict): {mol_name: {conf_num: {'neutral_rkf': str, ...}}}
        boltzmann_weights (dict): {mol_name: {conf_num: weight}}

    Returns:
        tuple: (orbital_results, orbital_by_conformer)
    """
    orbital_results = {}
    orbital_by_conformer = {}

    for mol_name, conformers in jobs_data.items():
        homo_sum = 0.0
        lumo_sum = 0.0
        total_weight = 0.0

        weights = boltzmann_weights.get(mol_name, {})
        orbital_by_conformer[mol_name] = {}

        for conf_num, jobs in conformers.items():
            weight = weights.get(conf_num, 0.0)

            if weight == 0.0 or not jobs.get('neutral_rkf'):
                continue

            try:
                rkf_path = str(jobs['neutral_rkf'])
                energies = extract_energies_from_rkf(rkf_path)

                if energies and energies['HOMO'] is not None and energies['LUMO'] is not None:
                    homo = energies['HOMO']
                    lumo = energies['LUMO']
                    gap = lumo - homo

                    orbital_by_conformer[mol_name][conf_num] = {
                        'HOMO': homo,
                        'LUMO': lumo,
                        'gap': gap
                    }

                    homo_sum += weight * homo
                    lumo_sum += weight * lumo
                    total_weight += weight

            except Exception as e:
                print(f"Erreur extraction orbitales pour {mol_name} conf {conf_num}: {str(e)}")
                continue

        if total_weight > 0:
            homo_avg = homo_sum / total_weight
            lumo_avg = lumo_sum / total_weight
        else:
            homo_avg = homo_sum
            lumo_avg = lumo_sum

        orbital_results[mol_name] = {
            'HOMO_avg': homo_avg,
            'LUMO_avg': lumo_avg,
            'gap_avg': lumo_avg - homo_avg
        }

    return orbital_results, orbital_by_conformer


def export_results_to_file(workdir, mol_name, results_data):
    """
    Exporte tous les résultats dans un fichier texte détaillé.

    Args:
        workdir (str): Répertoire de travail principal
        mol_name (str): Nom de la molécule
        results_data (dict): Dictionnaire contenant tous les résultats
    """
    results_dir = os.path.join(workdir, "redox", "results")
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"{mol_name}_redox_analysis.txt")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ANALYSE DES POTENTIELS REDOX (composite TZ2P+DZP) : {mol_name}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Méthode de calcul de U : {results_data['u_method']}\n")
        f.write(f"Température : {results_data.get('temperature', 298.15)} K\n")
        f.write(f"Schéma : G_composite = Ee(TZ2P) + [G(DZP) - Ee(DZP)]\n\n")

        # Poids de Boltzmann
        f.write("-"*80 + "\n")
        f.write("POIDS DE BOLTZMANN DES CONFORMÈRES\n")
        f.write("-"*80 + "\n")
        if 'weights_source' in results_data:
            f.write(f"Source : {results_data['weights_source']}\n")
        for conf_num, weight in sorted(results_data['boltzmann_weights'].items()):
            f.write(f"  Conformère {conf_num:2d}: {weight:6.4f} ({weight*100:5.2f}%)\n")
        f.write("\n")

        # Potentiels de réduction
        f.write("-"*80 + "\n")
        f.write("POTENTIELS DE RÉDUCTION (M + e⁻ → M⁻)\n")
        f.write("-"*80 + "\n")
        reduction_data = results_data['potentials']['reduction']
        f.write(f"  E(ΔG)        = {reduction_data['E_deltaG']:8.4f} V  (composite TZ2P)\n")
        f.write(f"  E(EA)        = {reduction_data['E_EA']:8.4f} V  (TZ2P)\n")
        f.write(f"  E(Edef)      = {reduction_data['E_Edef']:8.4f} V  (TZ2P)\n")
        f.write(f"  E(ΔΔU)       = {reduction_data['E_delta_delta_U']:8.4f} V  (DZP)\n")
        f.write(f"  E(TΔS)       = {reduction_data['E_T_delta_S']:8.4f} V  (DZP)\n")
        f.write(f"  Résidu base  = {reduction_data['residual_V']:8.4f} V\n\n")

        # Potentiels d'oxydation
        f.write("-"*80 + "\n")
        f.write("POTENTIELS D'OXYDATION (M → M⁺ + e⁻)\n")
        f.write("-"*80 + "\n")
        oxidation_data = results_data['potentials']['oxidation']
        f.write(f"  E(ΔG)        = {oxidation_data['E_deltaG']:8.4f} V  (composite TZ2P)\n")
        f.write(f"  E(IP)        = {oxidation_data['E_IP']:8.4f} V  (TZ2P)\n")
        f.write(f"  E(Edef)      = {oxidation_data['E_Edef']:8.4f} V  (TZ2P)\n")
        f.write(f"  E(ΔΔU)       = {oxidation_data['E_delta_delta_U']:8.4f} V  (DZP)\n")
        f.write(f"  E(TΔS)       = {oxidation_data['E_T_delta_S']:8.4f} V  (DZP)\n")
        f.write(f"  Résidu base  = {oxidation_data['residual_V']:8.4f} V\n\n")

        # Orbitales
        if 'orbital_data' in results_data:
            orb = results_data['orbital_data']
            f.write("-"*80 + "\n")
            f.write("ORBITALES MOLÉCULAIRES FRONTIÈRES (moyennes pondérées)\n")
            f.write("-"*80 + "\n")
            f.write(f"  HOMO = {orb['HOMO_avg']:8.4f} eV\n")
            f.write(f"  LUMO = {orb['LUMO_avg']:8.4f} eV\n")
            f.write(f"  Gap  = {orb['gap_avg']:8.4f} eV\n\n")

        # RMSD
        if 'rmsd_data' in results_data and results_data['rmsd_data']:
            f.write("-"*80 + "\n")
            f.write("ANALYSE RMSD (Changements structuraux)\n")
            f.write("-"*80 + "\n")
            for conf_num in sorted(results_data['rmsd_data'].keys()):
                rmsd = results_data['rmsd_data'][conf_num]
                f.write(f"  Conformère {conf_num:2d}:\n")
                f.write(f"    RMSD(réduit)  = {rmsd['reduced']:6.3f} Å\n")
                f.write(f"    RMSD(oxydé)   = {rmsd['oxidized']:6.3f} Å\n")
            f.write("\n")

        # Ruptures de liaison
        if 'bond_breaking' in results_data and results_data['bond_breaking']:
            has_any = any(
                bb.get('reduced') or bb.get('oxidized')
                for bb in results_data['bond_breaking'].values()
            )
            if has_any:
                f.write("-"*80 + "\n")
                f.write("RUPTURES DE LIAISON DÉTECTÉES\n")
                f.write("-"*80 + "\n")
                for conf_num in sorted(results_data['bond_breaking'].keys()):
                    bb = results_data['bond_breaking'][conf_num]
                    for state, label in [('reduced', 'réduit'), ('oxidized', 'oxydé')]:
                        broken = bb.get(state, [])
                        if broken:
                            for b in broken:
                                f.write(f"  Conformère {conf_num:2d} ({label}): "
                                        f"{b['bond_type']} Δd = +{b['elongation']:.3f} Å\n")
                f.write("\n⚠ ATTENTION: Des ruptures de liaison ont été détectées.\n\n")

        # Avertissements
        if 'warnings' in results_data and results_data['warnings']:
            f.write("-"*80 + "\n")
            f.write("AVERTISSEMENTS\n")
            f.write("-"*80 + "\n")
            for warning in results_data['warnings']:
                f.write(f"  ⚠ {warning}\n")
            f.write("\n")

        # Détails par conformère
        f.write("="*80 + "\n")
        f.write("DÉTAILS PAR CONFORMÈRE\n")
        f.write("="*80 + "\n\n")

        for conf_num in sorted(results_data['redox_params'].keys()):
            params = results_data['redox_params'][conf_num]
            weight = results_data['boltzmann_weights'].get(conf_num, 0.0)

            f.write(f"--- Conformère {conf_num} (poids = {weight:.4f}) ---\n\n")

            red = params['reduction']
            f.write("  RÉDUCTION:\n")
            f.write(f"    ΔG (comp)  = {red['delta_G_kj']:10.4f} kJ/mol\n")
            f.write(f"    EA (TZ2P)  = {red['EA_kj']:10.4f} kJ/mol\n")
            f.write(f"    Edef       = {red['Edef_kj']:10.4f} kJ/mol\n")
            f.write(f"    ΔΔU (DZP)  = {red['delta_delta_U_kj']:10.4f} kJ/mol\n")
            f.write(f"    TΔS (DZP)  = {red['T_delta_S_kj']:10.4f} kJ/mol\n")
            f.write(f"    Résidu     = {red['residual_kj']:10.4f} kJ/mol\n\n")

            ox = params['oxidation']
            f.write("  OXYDATION:\n")
            f.write(f"    ΔG (comp)  = {ox['delta_G_kj']:10.4f} kJ/mol\n")
            f.write(f"    IP (TZ2P)  = {ox['IP_kj']:10.4f} kJ/mol\n")
            f.write(f"    Edef       = {ox['Edef_kj']:10.4f} kJ/mol\n")
            f.write(f"    ΔΔU (DZP)  = {ox['delta_delta_U_kj']:10.4f} kJ/mol\n")
            f.write(f"    TΔS (DZP)  = {ox['T_delta_S_kj']:10.4f} kJ/mol\n")
            f.write(f"    Résidu     = {ox['residual_kj']:10.4f} kJ/mol\n\n")

    print(f"\n✓ Résultats exportés dans : {output_file}")

    # Export CSV
    try:
        summary_csv = os.path.join(results_dir, f"{mol_name}_redox_summary.csv")
        export_redox_summary_csv(results_data, summary_csv, mol_name)

        conformers_csv = os.path.join(results_dir, f"{mol_name}_redox_conformers.csv")
        export_redox_conformers_csv(results_data, conformers_csv)

        if 'orbital_data_by_conformer' in results_data:
            orbital_csv = os.path.join(results_dir, f"{mol_name}_orbital_data.csv")
            export_orbital_data_csv(results_data, orbital_csv)

    except Exception as e:
        print(f"⚠ Échec export CSV: {e}")


def export_redox_summary_csv(results_data, output_file, mol_name):
    """Exporte un résumé CSV des potentiels redox moyens."""
    from pathlib import Path

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reduction_data = results_data['potentials']['reduction']
    oxidation_data = results_data['potentials']['oxidation']
    orbital_data = results_data.get('orbital_data', {})

    with open(output_file, 'w') as f:
        f.write("molecule,T_K,u_method,E_reduction_V,E_oxidation_V,EA_V,IP_V,")
        f.write("residual_red_V,residual_ox_V,HOMO_eV,LUMO_eV,gap_eV\n")

        f.write(f"{mol_name},")
        f.write(f"{results_data.get('temperature', 298.15)},")
        f.write(f"{results_data.get('u_method', 'direct')},")
        f.write(f"{reduction_data['E_deltaG']:.6f},")
        f.write(f"{oxidation_data['E_deltaG']:.6f},")
        f.write(f"{reduction_data['E_EA']:.6f},")
        f.write(f"{oxidation_data['E_IP']:.6f},")
        f.write(f"{reduction_data['residual_V']:.6f},")
        f.write(f"{oxidation_data['residual_V']:.6f},")

        if orbital_data:
            f.write(f"{orbital_data['HOMO_avg']:.6f},")
            f.write(f"{orbital_data['LUMO_avg']:.6f},")
            f.write(f"{orbital_data['gap_avg']:.6f}\n")
        else:
            f.write(",,\n")

    print(f"✓ CSV résumé exporté: {output_file}")


def export_redox_conformers_csv(results_data, output_file):
    """Exporte les données redox par conformère en CSV."""
    from pathlib import Path

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("conformer,weight,dG_reduction_kJ,EA_kJ,Edef_red_kJ,ddU_red_kJ,TdS_red_kJ,residual_red_kJ,")
        f.write("dG_oxidation_kJ,IP_kJ,Edef_ox_kJ,ddU_ox_kJ,TdS_ox_kJ,residual_ox_kJ,RMSD_red_A,RMSD_ox_A,")
        f.write("bond_breaking_red,broken_bonds_red,bond_breaking_ox,broken_bonds_ox\n")

        for conf_num in sorted(results_data['redox_params'].keys()):
            params = results_data['redox_params'][conf_num]
            weight = results_data['boltzmann_weights'].get(conf_num, 0.0)

            red = params['reduction']
            ox = params['oxidation']

            rmsd_red = ""
            rmsd_ox = ""
            if 'rmsd_data' in results_data and conf_num in results_data['rmsd_data']:
                rmsd_red = f"{results_data['rmsd_data'][conf_num]['reduced']:.6f}"
                rmsd_ox = f"{results_data['rmsd_data'][conf_num]['oxidized']:.6f}"

            bb_red = "False"
            bb_red_bonds = "none"
            bb_ox = "False"
            bb_ox_bonds = "none"
            if 'bond_breaking' in results_data and conf_num in results_data.get('bond_breaking', {}):
                bb = results_data['bond_breaking'][conf_num]
                if bb.get('reduced'):
                    bb_red = "True"
                    bb_red_bonds = ";".join(f"{b['bond_type']}(+{b['elongation']:.3f})" for b in bb['reduced'])
                if bb.get('oxidized'):
                    bb_ox = "True"
                    bb_ox_bonds = ";".join(f"{b['bond_type']}(+{b['elongation']:.3f})" for b in bb['oxidized'])

            f.write(f"{conf_num},{weight:.8f},")
            f.write(f"{red['delta_G_kj']:.6f},")
            f.write(f"{red['EA_kj']:.6f},")
            f.write(f"{red['Edef_kj']:.6f},")
            f.write(f"{red['delta_delta_U_kj']:.6f},")
            f.write(f"{red['T_delta_S_kj']:.6f},")
            f.write(f"{red['residual_kj']:.6f},")
            f.write(f"{ox['delta_G_kj']:.6f},")
            f.write(f"{ox['IP_kj']:.6f},")
            f.write(f"{ox['Edef_kj']:.6f},")
            f.write(f"{ox['delta_delta_U_kj']:.6f},")
            f.write(f"{ox['T_delta_S_kj']:.6f},")
            f.write(f"{ox['residual_kj']:.6f},")
            f.write(f"{rmsd_red},{rmsd_ox},")
            f.write(f"{bb_red},{bb_red_bonds},{bb_ox},{bb_ox_bonds}\n")

    print(f"✓ CSV conformères exporté: {output_file}")


def export_orbital_data_csv(results_data, output_file):
    """Exporte les données orbitales par conformère en CSV."""
    from pathlib import Path

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if 'orbital_data_by_conformer' not in results_data:
        return

    with open(output_file, 'w') as f:
        f.write("conformer,weight,HOMO_eV,LUMO_eV,gap_eV\n")

        orbital_by_conf = results_data['orbital_data_by_conformer']

        for conf_num in sorted(orbital_by_conf.keys()):
            weight = results_data['boltzmann_weights'].get(conf_num, 0.0)
            orb = orbital_by_conf[conf_num]

            if orb['HOMO'] is not None and orb['LUMO'] is not None:
                gap = orb['LUMO'] - orb['HOMO']
                f.write(f"{conf_num},{weight:.8f},{orb['HOMO']:.6f},{orb['LUMO']:.6f},{gap:.6f}\n")
            else:
                f.write(f"{conf_num},{weight:.8f},,,\n")

    print(f"✓ CSV orbitales exporté: {output_file}")


def create_rkf_symlinks(workdir, mol_name, jobs_data):
    """
    Crée des symlinks pour les RKF clés dans results/rkf/.

    Args:
        workdir (str): Répertoire de travail principal
        mol_name (str): Nom de la molécule
        jobs_data (dict): {conf_num: {'neutral_rkf': str, 'neutral_tz2p': job, ...}}
    """
    from pathlib import Path

    rkf_dir = Path(workdir) / "redox" / "results" / "rkf"
    rkf_dir.mkdir(parents=True, exist_ok=True)

    for conf_num, jobs in sorted(jobs_data.items()):
        prefix = f"conf_{conf_num}"

        # neutral DZP (symlink vers conformères)
        if jobs.get('neutral_rkf'):
            src = Path(str(jobs['neutral_rkf']))
            dst = rkf_dir / f"{prefix}_neutral_DZP.rkf"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

        # Les jobs PLAMS
        job_map = {
            'neutral_tz2p': f"{prefix}_neutral_TZ2P.rkf",
            'reduced_sp': f"{prefix}_reduced_sp_DZP.rkf",
            'reduced_opt': f"{prefix}_reduced_opt_DZP.rkf",
            'reduced_tz2p': f"{prefix}_reduced_TZ2P.rkf",
            'oxidized_sp': f"{prefix}_oxidized_sp_DZP.rkf",
            'oxidized_opt': f"{prefix}_oxidized_opt_DZP.rkf",
            'oxidized_tz2p': f"{prefix}_oxidized_TZ2P.rkf",
        }

        for job_type, filename in job_map.items():
            if jobs.get(job_type):
                try:
                    src = Path(jobs[job_type].results.rkfpath(file='adf'))
                    dst = rkf_dir / filename
                    if src.exists() and not dst.exists():
                        dst.symlink_to(src)
                except Exception:
                    pass

    print(f"✓ Symlinks RKF créés dans: {rkf_dir}")


def analyze_molecule_redox(mol_name, jobs_data, workdir, temperature=298.15,
                          u_method="direct"):
    """
    Fonction principale d'analyse redox pour une molécule avec ses conformères.

    Args:
        mol_name (str): Nom de la molécule
        jobs_data (dict): {conf_num: {'neutral_rkf': str, 'neutral_tz2p': job, ...}}
        workdir (str): Répertoire de travail
        temperature (float): Température en K
        u_method (str): Méthode de calcul de U

    Returns:
        dict: Résultats complets de l'analyse
    """
    print(f"\n{'='*80}")
    print(f"ANALYSE REDOX (composite TZ2P+DZP) : {mol_name}")
    print(f"{'='*80}\n")

    # 1. Extraction des énergies
    print("Extraction des énergies depuis les fichiers adf.rkf...")
    conformers_energies = extract_all_energies(jobs_data, u_method=u_method)

    # 2. Poids de Boltzmann
    print("Calcul des poids de Boltzmann...")
    boltzmann_weights = read_conformer_weights_from_summary(workdir)

    if boltzmann_weights:
        weights_source = "Pipeline conformères (conformers_summary.txt)"
        print("  → Utilisation des poids du pipeline conformères")
    else:
        weights_source = "Recalculé depuis énergies redox"
        print("  → Fichier conformères non trouvé, recalcul depuis énergies redox")
        boltzmann_weights = calculate_boltzmann_weights(conformers_energies, temperature)

    print("\nPoids de Boltzmann:")
    for conf_num, weight in sorted(boltzmann_weights.items()):
        print(f"  Conformère {conf_num}: {weight:.4f} ({weight*100:.2f}%)")

    # 3. Paramètres redox par conformère
    print("\nCalcul des paramètres redox (schéma composite TZ2P+DZP)...")
    redox_params = calculate_redox_parameters(conformers_energies)

    # 4. Moyennes de Boltzmann
    print("Calcul des potentiels moyennés...")
    potentials = calculate_boltzmann_averaged_potentials(redox_params, boltzmann_weights)

    # 5. RMSD
    print("Analyse des changements structuraux (RMSD)...")
    rmsd_data, rmsd_warnings = compare_conformers_rmsd({mol_name: jobs_data})

    # 5b. Bond breaking
    print("Détection de ruptures de liaison...")
    bond_breaking_data, bb_warnings = detect_bond_breaking({mol_name: jobs_data})
    bond_breaking = bond_breaking_data.get(mol_name, {})

    # 6. Orbitales
    print("Extraction des énergies HOMO/LUMO...")
    orbital_results, orbital_by_conformer = extract_orbital_energies(
        {mol_name: jobs_data},
        {mol_name: boltzmann_weights}
    )
    orbital_data = orbital_results[mol_name]
    orbital_data_by_conformer = orbital_by_conformer.get(mol_name, {})

    # 7. Affichage
    print(f"\n{'='*80}")
    print("RÉSULTATS")
    print(f"{'='*80}\n")

    print("POTENTIELS DE RÉDUCTION (M + e⁻ → M⁻):")
    red = potentials['reduction']
    print(f"  E(ΔG)   = {red['E_deltaG']:8.4f} V  (composite TZ2P)")
    print(f"  E(EA)   = {red['E_EA']:8.4f} V  (TZ2P)")
    print(f"  E(Edef) = {red['E_Edef']:8.4f} V  (TZ2P)")
    print(f"  E(ΔΔU)  = {red['E_delta_delta_U']:8.4f} V  (DZP)")
    print(f"  E(TΔS)  = {red['E_T_delta_S']:8.4f} V  (DZP)")
    print(f"  Résidu  = {red['residual_V']:8.4f} V")

    print("\nPOTENTIELS D'OXYDATION (M → M⁺ + e⁻):")
    ox = potentials['oxidation']
    print(f"  E(ΔG)   = {ox['E_deltaG']:8.4f} V  (composite TZ2P)")
    print(f"  E(IP)   = {ox['E_IP']:8.4f} V  (TZ2P)")
    print(f"  E(Edef) = {ox['E_Edef']:8.4f} V  (TZ2P)")
    print(f"  E(ΔΔU)  = {ox['E_delta_delta_U']:8.4f} V  (DZP)")
    print(f"  E(TΔS)  = {ox['E_T_delta_S']:8.4f} V  (DZP)")
    print(f"  Résidu  = {ox['residual_V']:8.4f} V")

    print("\nORBITALES FRONTIÈRES:")
    print(f"  HOMO = {orbital_data['HOMO_avg']:8.4f} eV")
    print(f"  LUMO = {orbital_data['LUMO_avg']:8.4f} eV")
    print(f"  Gap  = {orbital_data['gap_avg']:8.4f} eV")

    all_warnings = rmsd_warnings + bb_warnings
    if all_warnings:
        print("\n⚠ AVERTISSEMENTS:")
        for warning in all_warnings:
            print(f"  {warning}")

    # 8. Export
    results_package = {
        'potentials': potentials,
        'redox_params': redox_params,
        'boltzmann_weights': boltzmann_weights,
        'weights_source': weights_source,
        'rmsd_data': rmsd_data.get(mol_name, {}),
        'bond_breaking': bond_breaking,
        'warnings': all_warnings,
        'orbital_data': orbital_data,
        'orbital_data_by_conformer': orbital_data_by_conformer,
        'energies': conformers_energies,
        'u_method': u_method,
        'temperature': temperature
    }

    export_results_to_file(workdir, mol_name, results_package)

    # Créer les symlinks RKF
    create_rkf_symlinks(workdir, mol_name, jobs_data)

    return results_package


def calculate_predict_dg_parameters(conformers_energies):
    """
    Calcule les paramètres redox en mode predict-dg (SP sur géom neutre : NormalModes
    sur le neutre, gradient seul sur les ions).

    DG = G(ion_sp) - G(neutral_sp), EA/IP verticaux depuis Ee.

    Args:
        conformers_energies (dict): {conf_num: {'neutral_sp': energies, 'reduced_sp': energies, 'oxidized_sp': energies}}

    Returns:
        dict: {conf_num: {'reduction': {...}, 'oxidation': {...}}}
    """
    redox_params = {}

    for conf_num, e in conformers_energies.items():
        neutral = e.get('neutral_sp')
        red_sp = e.get('reduced_sp')
        ox_sp = e.get('oxidized_sp')

        if not all([neutral, red_sp, ox_sp]):
            missing = [k for k in ['neutral_sp', 'reduced_sp', 'oxidized_sp'] if not e.get(k)]
            print(f"⚠ Conformère {conf_num} incomplet (manque: {missing}), ignoré")
            continue

        # Vérifier disponibilité thermo (SP+NormalModes devrait fournir G)
        if any(v is None for v in [neutral.get('G'), red_sp.get('G'), ox_sp.get('G')]):
            print(f"⚠ Données thermo incomplètes pour conformère {conf_num}")
            # Fallback : utiliser Ee seul si pas de thermo
            has_thermo = False
        else:
            has_thermo = True

        # Réduction
        reduction = {}
        reduction['EA_ha'] = neutral['Ee'] - red_sp['Ee']
        reduction['EA_kj'] = hartree_to_kj(reduction['EA_ha'])
        reduction['EA_ev'] = hartree_to_ev(reduction['EA_ha'])

        if has_thermo:
            reduction['delta_G_ha'] = red_sp['G'] - neutral['G']
            reduction['delta_G_kj'] = hartree_to_kj(reduction['delta_G_ha'])
        else:
            reduction['delta_G_ha'] = red_sp['Ee'] - neutral['Ee']
            reduction['delta_G_kj'] = hartree_to_kj(reduction['delta_G_ha'])

        # Oxydation
        oxidation = {}
        oxidation['IP_ha'] = ox_sp['Ee'] - neutral['Ee']
        oxidation['IP_kj'] = hartree_to_kj(oxidation['IP_ha'])
        oxidation['IP_ev'] = hartree_to_ev(oxidation['IP_ha'])

        if has_thermo:
            oxidation['delta_G_ha'] = ox_sp['G'] - neutral['G']
            oxidation['delta_G_kj'] = hartree_to_kj(oxidation['delta_G_ha'])
        else:
            oxidation['delta_G_ha'] = ox_sp['Ee'] - neutral['Ee']
            oxidation['delta_G_kj'] = hartree_to_kj(oxidation['delta_G_ha'])

        redox_params[conf_num] = {
            'reduction': reduction,
            'oxidation': oxidation,
            'has_thermo': has_thermo
        }

    return redox_params


def calculate_boltzmann_averaged_predict_dg(redox_params, boltzmann_weights):
    """
    Moyenne de Boltzmann des paramètres predict-dg.

    Args:
        redox_params (dict): Sortie de calculate_predict_dg_parameters()
        boltzmann_weights (dict): {conf_num: weight}

    Returns:
        dict: Potentiels moyennés
    """
    avg_red = {'delta_G_kj': 0.0, 'EA_kj': 0.0}
    avg_ox = {'delta_G_kj': 0.0, 'IP_kj': 0.0}

    total_weight = 0.0
    for conf_num, weight in boltzmann_weights.items():
        if conf_num not in redox_params:
            continue
        total_weight += weight
        red = redox_params[conf_num]['reduction']
        ox = redox_params[conf_num]['oxidation']
        avg_red['delta_G_kj'] += weight * red['delta_G_kj']
        avg_red['EA_kj'] += weight * red['EA_kj']
        avg_ox['delta_G_kj'] += weight * ox['delta_G_kj']
        avg_ox['IP_kj'] += weight * ox['IP_kj']

    if total_weight > 0 and total_weight != 1.0:
        for d in [avg_red, avg_ox]:
            for key in d:
                d[key] /= total_weight

    return {
        'reduction': {
            'E_deltaG': kj_to_volt(avg_red['delta_G_kj']),
            'E_EA': kj_to_volt(avg_red['EA_kj']),
            'delta_G_kj': avg_red['delta_G_kj'],
            'EA_kj': avg_red['EA_kj']
        },
        'oxidation': {
            'E_deltaG': kj_to_volt(avg_ox['delta_G_kj']),
            'E_IP': kj_to_volt(avg_ox['IP_kj']),
            'delta_G_kj': avg_ox['delta_G_kj'],
            'IP_kj': avg_ox['IP_kj']
        }
    }


def export_predict_dg_results(workdir, mol_name, results_data):
    """
    Exporte les résultats predict-dg vers fichier texte et CSV.

    Args:
        workdir (str): Répertoire de travail
        mol_name (str): Nom de la molécule
        results_data (dict): Résultats complets
    """
    results_dir = os.path.join(workdir, "redox", "results")
    os.makedirs(results_dir, exist_ok=True)

    potentials = results_data['potentials']
    red = potentials['reduction']
    ox = potentials['oxidation']

    # Fichier texte principal
    txt_file = os.path.join(results_dir, f"{mol_name}_predict_dg_analysis.txt")
    with open(txt_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"ANALYSE PREDICT-DG (criblage rapide) : {mol_name}\n")
        f.write(f"{'='*80}\n\n")

        f.write("Mode : SP+NormalModes sur géométrie neutre\n")
        f.write("ConceptualDFT : Activé (Full)\n")
        f.write("QTAIM : Activé (Full)\n")
        f.write(f"Température : {results_data['temperature']:.2f} K\n")
        f.write(f"Source des poids : {results_data['weights_source']}\n\n")

        f.write(f"{'='*80}\n")
        f.write("POTENTIELS REDOX (approximation SP+NormalModes)\n")
        f.write(f"{'='*80}\n\n")

        f.write("RÉDUCTION (M + e⁻ → M⁻) :\n")
        f.write(f"  ΔG       = {red['delta_G_kj']:10.2f} kJ/mol  ({red['E_deltaG']:8.4f} V)\n")
        f.write(f"  EA(vert) = {red['EA_kj']:10.2f} kJ/mol  ({red['E_EA']:8.4f} V)\n\n")

        f.write("OXYDATION (M → M⁺ + e⁻) :\n")
        f.write(f"  ΔG       = {ox['delta_G_kj']:10.2f} kJ/mol  ({ox['E_deltaG']:8.4f} V)\n")
        f.write(f"  IP(vert) = {ox['IP_kj']:10.2f} kJ/mol  ({ox['E_IP']:8.4f} V)\n\n")

        # Orbitales
        orbital_data = results_data.get('orbital_data', {})
        if orbital_data:
            f.write(f"{'='*80}\n")
            f.write("ORBITALES FRONTIÈRES\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"  HOMO = {orbital_data.get('HOMO_avg', 0):8.4f} eV\n")
            f.write(f"  LUMO = {orbital_data.get('LUMO_avg', 0):8.4f} eV\n")
            f.write(f"  Gap  = {orbital_data.get('gap_avg', 0):8.4f} eV\n\n")

        # Détail par conformère
        redox_params = results_data['redox_params']
        boltzmann_weights = results_data['boltzmann_weights']

        f.write(f"{'='*80}\n")
        f.write("DÉTAIL PAR CONFORMÈRE\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"{'Conf':>5}  {'Poids':>7}  {'ΔG_red':>10}  {'EA':>10}  {'ΔG_ox':>10}  {'IP':>10}\n")
        f.write(f"{'':>5}  {'':>7}  {'kJ/mol':>10}  {'kJ/mol':>10}  {'kJ/mol':>10}  {'kJ/mol':>10}\n")
        f.write(f"{'-'*65}\n")

        for conf_num in sorted(redox_params.keys()):
            w = boltzmann_weights.get(conf_num, 0)
            rp = redox_params[conf_num]
            f.write(f"{conf_num:>5}  {w:7.4f}  {rp['reduction']['delta_G_kj']:10.2f}  "
                    f"{rp['reduction']['EA_kj']:10.2f}  {rp['oxidation']['delta_G_kj']:10.2f}  "
                    f"{rp['oxidation']['IP_kj']:10.2f}\n")

        f.write(f"\nNote : Les données ConceptualDFT et QTAIM sont dans les fichiers RKF.\n")
        f.write(f"Consulter les RKF avec AMSview ou KFReader pour les extraire.\n")

    print(f"✓ Résultats exportés : {txt_file}")

    # CSV résumé
    csv_file = os.path.join(results_dir, f"{mol_name}_predict_dg_summary.csv")
    with open(csv_file, 'w') as f:
        f.write("molecule,process,delta_G_kJ_mol,delta_G_V,EA_or_IP_kJ_mol,EA_or_IP_V,"
                "HOMO_eV,LUMO_eV,gap_eV\n")

        homo = orbital_data.get('HOMO_avg', '') if orbital_data else ''
        lumo = orbital_data.get('LUMO_avg', '') if orbital_data else ''
        gap = orbital_data.get('gap_avg', '') if orbital_data else ''

        f.write(f"{mol_name},reduction,{red['delta_G_kj']:.4f},{red['E_deltaG']:.4f},"
                f"{red['EA_kj']:.4f},{red['E_EA']:.4f},"
                f"{homo},{lumo},{gap}\n")
        f.write(f"{mol_name},oxidation,{ox['delta_G_kj']:.4f},{ox['E_deltaG']:.4f},"
                f"{ox['IP_kj']:.4f},{ox['E_IP']:.4f},"
                f"{homo},{lumo},{gap}\n")

    print(f"✓ CSV exporté : {csv_file}")

    # CSV par conformère
    conf_csv_file = os.path.join(results_dir, f"{mol_name}_predict_dg_conformers.csv")
    with open(conf_csv_file, 'w') as f:
        f.write("molecule,conformer,weight,delta_G_red_kJ,EA_kJ,delta_G_ox_kJ,IP_kJ\n")
        for conf_num in sorted(redox_params.keys()):
            w = boltzmann_weights.get(conf_num, 0)
            rp = redox_params[conf_num]
            f.write(f"{mol_name},{conf_num},{w:.6f},"
                    f"{rp['reduction']['delta_G_kj']:.4f},{rp['reduction']['EA_kj']:.4f},"
                    f"{rp['oxidation']['delta_G_kj']:.4f},{rp['oxidation']['IP_kj']:.4f}\n")

    print(f"✓ CSV conformères exporté : {conf_csv_file}")


def analyze_molecule_predict_dg(mol_name, jobs_data, workdir, temperature=298.15):
    """
    Fonction principale d'analyse en mode predict-dg.

    Args:
        mol_name (str): Nom de la molécule
        jobs_data (dict): {conf_num: {'neutral_rkf': str, 'neutral_sp': job, 'reduced_sp': job, 'oxidized_sp': job}}
        workdir (str): Répertoire de travail
        temperature (float): Température en K

    Returns:
        dict: Résultats complets de l'analyse
    """
    print(f"\n{'='*80}")
    print(f"ANALYSE PREDICT-DG (criblage rapide) : {mol_name}")
    print(f"{'='*80}\n")

    # 1. Extraction des énergies
    print("Extraction des énergies depuis les fichiers adf.rkf...")
    conformers_energies = extract_all_energies(jobs_data, u_method="direct")

    # 2. Poids de Boltzmann
    print("Lecture des poids de Boltzmann...")
    boltzmann_weights = read_conformer_weights_from_summary(workdir)

    if boltzmann_weights:
        weights_source = "Pipeline conformères (conformers_summary.txt)"
        print("  → Utilisation des poids du pipeline conformères")
    else:
        weights_source = "Recalculé depuis énergies"
        print("  → Fichier conformères non trouvé, recalcul depuis énergies")
        boltzmann_weights = calculate_boltzmann_weights(conformers_energies, temperature)

    print("\nPoids de Boltzmann:")
    for conf_num, weight in sorted(boltzmann_weights.items()):
        print(f"  Conformère {conf_num}: {weight:.4f} ({weight*100:.2f}%)")

    # 3. Paramètres redox predict-dg
    print("\nCalcul des paramètres redox (SP+NormalModes)...")
    redox_params = calculate_predict_dg_parameters(conformers_energies)

    # 4. Moyennes de Boltzmann
    print("Calcul des potentiels moyennés...")
    potentials = calculate_boltzmann_averaged_predict_dg(redox_params, boltzmann_weights)

    # 5. Orbitales
    print("Extraction des énergies HOMO/LUMO...")
    orbital_results, orbital_by_conformer = extract_orbital_energies(
        {mol_name: jobs_data},
        {mol_name: boltzmann_weights}
    )
    orbital_data = orbital_results.get(mol_name, {})

    # 6. Affichage
    print(f"\n{'='*80}")
    print("RÉSULTATS")
    print(f"{'='*80}\n")

    red = potentials['reduction']
    ox = potentials['oxidation']

    print("RÉDUCTION (M + e⁻ → M⁻) :")
    print(f"  ΔG       = {red['delta_G_kj']:10.2f} kJ/mol  ({red['E_deltaG']:8.4f} V)")
    print(f"  EA(vert) = {red['EA_kj']:10.2f} kJ/mol  ({red['E_EA']:8.4f} V)")

    print("\nOXYDATION (M → M⁺ + e⁻) :")
    print(f"  ΔG       = {ox['delta_G_kj']:10.2f} kJ/mol  ({ox['E_deltaG']:8.4f} V)")
    print(f"  IP(vert) = {ox['IP_kj']:10.2f} kJ/mol  ({ox['E_IP']:8.4f} V)")

    if orbital_data:
        print(f"\nORBITALES FRONTIÈRES :")
        print(f"  HOMO = {orbital_data.get('HOMO_avg', 0):8.4f} eV")
        print(f"  LUMO = {orbital_data.get('LUMO_avg', 0):8.4f} eV")
        print(f"  Gap  = {orbital_data.get('gap_avg', 0):8.4f} eV")

    # 7. Export
    results_package = {
        'potentials': potentials,
        'redox_params': redox_params,
        'boltzmann_weights': boltzmann_weights,
        'weights_source': weights_source,
        'orbital_data': orbital_data,
        'energies': conformers_energies,
        'temperature': temperature
    }

    export_predict_dg_results(workdir, mol_name, results_package)

    # Créer les symlinks RKF
    create_rkf_symlinks(workdir, mol_name, jobs_data)

    return results_package
