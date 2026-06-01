# -*- coding: utf-8 -*-
"""
Pipeline de génération et optimisation de conformères.

Pipeline en 4 étapes :
1. Generate - Génération avec RDKit + UFF
2. Optimize - Optimisation GFN1-xTB (sans fenêtre d'énergie)
3. Boltzmann - Sélection par poids statistique (seuil 0.10)
4. DFT Opt + Freq - Optimisation DFT + fréquences + solvant (obligatoire)
"""

import os
import json

from scm.conformers import ConformersJob
from scm.plams import AMSJob
from scm.plams.core.settings import Settings

from common.utils import (
    DefaultJobManager,
    boltzmann_weights,
    print_error,
    print_success,
    print_warning,
)
from common.workspace import setup_workspace as common_setup_workspace

from .common import UNIT, TEMPERATURE, configure_functional


def setup_workspace(name):
    """
    Prépare les sous-dossiers pour les calculs de conformères.

    Args:
        name (str): Nom de la molécule

    Returns:
        tuple: (workdir, calc_dir, results_dir) en tant que Path objects
    """
    workdir, calc_dir, results_dir = common_setup_workspace(
        name=name,
        subfolder="conformers"
    )
    return workdir, calc_dir, results_dir


def _safe_relative_energies(job, unit=UNIT):
    """
    Retourne une liste d'énergies relatives (kJ/mol).

    Args:
        job: Job PLAMS avec résultats de conformères
        unit (str): Unité d'énergie (défaut: kJ/mol)

    Returns:
        list: Énergies relatives

    Raises:
        RuntimeError: Si les énergies ne peuvent pas être obtenues
    """
    try:
        energies = job.results.get_relative_energies(unit)
    except Exception:
        energies = None

    if energies is None:
        jpath = getattr(job, "path", None)
        raise RuntimeError(
            f"Impossible d'obtenir les énergies relatives pour le job '{job.name}'. "
            f"Le job a probablement échoué. Dossier du job: {jpath}"
        )

    return energies


def print_results(job, temperature=TEMPERATURE, unit=UNIT):
    """
    Affiche ΔE et poids Boltzmann pour un ConformersJob.

    Args:
        job: Job PLAMS avec résultats de conformères
        temperature (float): Température pour Boltzmann (K)
        unit (str): Unité d'énergie

    Returns:
        tuple: (energies, weights)
    """
    energies = _safe_relative_energies(job, unit)
    weights, Z = boltzmann_weights(energies, temperature)

    print(f"\nRésultats (T = {temperature:.2f} K | UNIT = {unit})")
    print(f"{'#':>3s} {'ΔE [' + unit + ']':>14s} {'w':>10s}")
    for i, (e, w) in enumerate(zip(energies, weights), 1):
        print(f"{i:3d} {e:14.3f} {w:10.6f}")
    print(f"Z = {Z:.6e}")

    return energies, weights


def generate_conformers(molecule, calc_dir):
    """
    Étape 1 : Génération des conformères (RDKit + UFF).

    Args:
        molecule: Molécule PLAMS
        calc_dir: Répertoire pour les calculs

    Returns:
        ConformersJob: Job PLAMS exécuté
    """
    print("\n[Étape 1] Génération des conformères (RDKit + UFF)")

    settings = Settings()
    settings.input.ams.Task = "Generate"
    settings.input.ams.Generator.Method = "RDKit"
    settings.input.ams.Generator.RDKit.InitialNConformers = 1000
    settings.input.ForceField.Type = "UFF"

    job_dir = os.path.join(calc_dir, "01_generate")
    os.makedirs(job_dir, exist_ok=True)

    job = ConformersJob(molecule=molecule, settings=settings, name="generate_conformers")

    with DefaultJobManager(job_dir):
        _ = job.run()

    if not job.results:
        raise RuntimeError("Échec de la génération des conformères.")

    return job


def optimize_conformers(previous_job, calc_dir):
    """
    Étape 2 : Optimisation des conformères avec GFN1-xTB (sans fenêtre d'énergie).

    Args:
        previous_job: Job précédent (generation)
        calc_dir: Répertoire pour les calculs

    Returns:
        ConformersJob: Job PLAMS exécuté
    """
    print("\n[Étape 2] Optimisation des conformères (GFN1-xTB)")

    settings = Settings()
    settings.input.ams.Task = "Optimize"
    settings.input.ams.InputConformersSet = os.path.abspath(previous_job.results.rkfpath())
    settings.input.DFTB.Model = "GFN1-xTB"

    job_dir = os.path.join(calc_dir, "02_optimize")
    os.makedirs(job_dir, exist_ok=True)

    job = ConformersJob(settings=settings, name="optimize_conformers")

    with DefaultJobManager(job_dir):
        _ = job.run()

    if not job.results:
        raise RuntimeError("Échec de l'optimisation des conformères.")

    return job


def filter_boltzmann_weights(previous_job, calc_dir, weight_threshold=0.10):
    """
    Filtrer les conformères selon w_i >= seuil (UNIT interne kJ/mol).

    Les poids sont calculés sur les énergies GFN1-xTB de l'étape 2.

    Args:
        previous_job: Job précédent (optimize)
        calc_dir: Répertoire pour les calculs
        weight_threshold (float): Seuil de poids minimal (défaut: 0.10)

    Returns:
        tuple: (selected_indices, selected_conformers)
    """
    print(f"\n[Étape 3] Filtrage Boltzmann (w >= {weight_threshold:.2f})")

    energies_kj = _safe_relative_energies(previous_job, UNIT)
    weights, _ = boltzmann_weights(energies_kj, TEMPERATURE)
    conformers = previous_job.results.get_conformers()

    selected_indices = [i for i, w in enumerate(weights) if w >= weight_threshold]
    selected_conformers = [conformers[i] for i in selected_indices]

    summary = os.path.join(calc_dir, "boltzmann_filter_summary.txt")
    with open(summary, "w") as f:
        f.write(f"# Filtrage Boltzmann (T={TEMPERATURE} K | UNIT={UNIT})\n")
        f.write(f"{'#':>3s} {'idx':>5s} {'ΔE [kJ/mol]':>14s} {'w':>10s}\n")
        for j, i in enumerate(selected_indices, 1):
            f.write(f"{j:3d} {i+1:5d} {energies_kj[i]:14.3f} {weights[i]:10.6f}\n")
        f.write(f"\nNombre retenu: {len(selected_indices)} / {len(energies_kj)}\n")

    print(f"  {len(selected_indices)} conformères retenus sur {len(energies_kj)}")

    return selected_indices, selected_conformers


def dft_optimize_with_freq(conformers, molecule_name, calc_dir,
                           functional="PBE0", basis="DZP",
                           solvent="Acetonitrile", max_attempts=3):
    """
    Étape 4 : Optimisation DFT + fréquences + solvant (obligatoire).

    Pour chaque conformère retenu par Boltzmann :
    1. GeometryOptimization functional/basis + NormalModes + COSMO(solvant)
    2. check_and_fix_frequencies() si fréquences imaginaires

    Args:
        conformers (list): Conformères retenus par Boltzmann
        molecule_name (str): Nom de la molécule
        calc_dir: Répertoire pour les calculs
        functional (str): Fonctionnelle DFT (défaut: PBE0)
        basis (str): Base atomique (défaut: DZP)
        solvent (str): Solvant COSMO (défaut: Acetonitrile)
        max_attempts (int): Tentatives de correction de fréquences (défaut: 3)

    Returns:
        tuple: (valid_conformers, valid_indices, rkf_paths)
               - valid_conformers (list): Conformères validés (vrais minima)
               - valid_indices (list): Indices dans la liste d'entrée
               - rkf_paths (dict): {conf_num: chemin_vers_adf.rkf}
    """
    from common.frequency import run_frequency_calculation, check_and_fix_frequencies

    print(f"\n[Étape 4] Optimisation DFT + fréquences ({functional}/{basis}, solvant={solvent})")
    print(f"  Seuil: 0.0 cm⁻¹ (aucune fréquence négative tolérée)")
    print(f"  Correction automatique: activée (max {max_attempts} tentatives)")

    dft_dir = os.path.join(calc_dir, "03_dft_opt")
    os.makedirs(dft_dir, exist_ok=True)

    valid_conformers = []
    valid_indices = []
    rkf_paths = {}

    for i, conf in enumerate(conformers):
        conf_num = i + 1
        print(f"\n  → Conformère {conf_num}/{len(conformers)}:")

        job_name = f"conf_{conf_num}_dft_opt"
        conf_dir = os.path.join(dft_dir, f"conf_{conf_num}")
        os.makedirs(conf_dir, exist_ok=True)

        # Lancer l'optimisation DFT + fréquences avec solvant
        with DefaultJobManager(conf_dir):
            initial_job = run_frequency_calculation(
                conf,
                functional,
                basis,
                job_name,
                charge=0,
                spin_polarization=0,
                quality="Good",
                optimize=True,
                allow_transition_state=True,
                solvent=solvent
            )

        if initial_job is None:
            print_error(f"    Échec du job DFT (crash)")
            continue

        # Vérifier et corriger les fréquences si nécessaire
        with DefaultJobManager(conf_dir):
            final_job, success = check_and_fix_frequencies(
                initial_job,
                functional,
                basis,
                job_name,
                charge=0,
                spin_polarization=0,
                quality="Good",
                max_attempts=max_attempts,
                perturbation_scale=0.2,
                threshold=0.0,
                solvent=solvent
            )

        if success:
            valid_conformers.append(final_job.results.get_main_molecule())
            valid_indices.append(i)

            # Récupérer le chemin vers adf.rkf
            try:
                rkf_path = final_job.results.rkfpath(file='adf')
                rkf_paths[conf_num] = str(rkf_path)
            except Exception:
                print_warning(f"    Impossible de récupérer le chemin adf.rkf")

            print_success(f"    Conformère {conf_num} validé")
        else:
            print_warning(f"    Conformère {conf_num} rejeté (correction impossible)")

    # Résumé
    print(f"\n  Résultat final: {len(valid_conformers)}/{len(conformers)} conformères validés")

    # Sauvegarder le résumé
    summary_file = os.path.join(dft_dir, "dft_opt_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"# Optimisation DFT + fréquences pour {molecule_name}\n")
        f.write(f"# Méthode: {functional}/{basis} + COSMO({solvent})\n")
        f.write(f"# Seuil: 0.0 cm⁻¹ (aucune fréquence négative tolérée)\n\n")
        f.write(f"{'Conformère':>12}  {'Status':>10}  {'RKF':>60}\n")
        f.write(f"{'-'*90}\n")
        for i in range(len(conformers)):
            conf_num = i + 1
            if i in valid_indices:
                rkf = rkf_paths.get(conf_num, "N/A")
                f.write(f"{conf_num:>12}  {'VALIDE':>10}  {rkf:>60}\n")
            else:
                f.write(f"{conf_num:>12}  {'REJETÉ':>10}  {'N/A':>60}\n")
        f.write(f"\nTotal valide: {len(valid_conformers)}/{len(conformers)}\n")

    print_success(f"  Résumé sauvegardé: {summary_file}")

    return valid_conformers, valid_indices, rkf_paths


def optimize_molecules_xtb(molecules, calc_dir):
    """
    Optimise une liste de Molecule avec GFN1-xTB (jobs individuels AMSJob).

    Utilisee en Phase 2 du cluster-continuum pour optimiser les systemes
    solute+solvant sans passer par ConformersJob/RDKit.

    Args:
        molecules: list[Molecule] - systemes augmentes
        calc_dir: repertoire de calcul

    Returns:
        list of (optimized_molecule, energy_hartree) tuples, triees par energie
    """
    xtb_dir = os.path.join(calc_dir, "02_optimize_cc")
    os.makedirs(xtb_dir, exist_ok=True)

    print(f"\n[Phase 2 - Étape 2] Optimisation xTB des systèmes augmentés")
    print(f"  {len(molecules)} systèmes à optimiser")

    results = []

    for i, mol in enumerate(molecules):
        conf_num = i + 1
        print(f"  → Système {conf_num}/{len(molecules)}...", end=" ")

        s = Settings()
        s.input.ams.Task = "GeometryOptimization"
        s.input.DFTB.Model = "GFN1-xTB"

        job_name = f"cc_optimize_{conf_num}"

        with DefaultJobManager(xtb_dir):
            job = AMSJob(molecule=mol, settings=s, name=job_name)
            job.run()

        if not job.ok():
            print_warning(f"Échec de l'optimisation xTB pour le système {conf_num}")
            continue

        try:
            opt_mol = job.results.get_main_molecule()
            energy = job.results.get_energy(unit='hartree')
            results.append((opt_mol, energy))
            print_success(f"E = {energy:.6f} Hartree")
        except Exception as e:
            print_warning(f"Impossible de lire les résultats pour le système {conf_num}: {e}")

    # Trier par energie croissante
    results.sort(key=lambda x: x[1])

    print(f"  {len(results)}/{len(molecules)} systèmes optimisés avec succès")
    return results


def compute_dft_boltzmann_weights(rkf_paths, temperature=TEMPERATURE):
    """
    Recalcule les poids de Boltzmann à partir des énergies de Gibbs DFT.

    Après l'étape 4 (DFT opt+freq), les adf.rkf contiennent G (Gibbs).
    Ces poids sont plus fiables que ceux de l'étape GFN1-xTB.

    Args:
        rkf_paths (dict): {conf_num: chemin_vers_adf.rkf}
        temperature (float): Température en K (défaut: 298.15)

    Returns:
        tuple: (weights_dict, energies_dict)
               - weights_dict (dict): {conf_num: poids_boltzmann}
               - energies_dict (dict): {conf_num: ΔG_relatif_kJ}
    """
    from scm.plams import KFReader

    EH_TO_KJ = 2625.499638

    # Extraire les énergies de Gibbs
    gibbs = {}
    for conf_num, rkf_path in rkf_paths.items():
        try:
            kf = KFReader(rkf_path)
            G = kf.read("Thermodynamics", "Gibbs free Energy")
            gibbs[conf_num] = G
        except Exception as e:
            print_warning(f"  Impossible de lire G pour conf {conf_num}: {e}")

    if not gibbs:
        print_error("  Aucune énergie de Gibbs disponible pour recalculer Boltzmann")
        return {}, {}

    # Énergies relatives en kJ/mol
    G_min = min(gibbs.values())
    relative_kj = {cn: (G - G_min) * EH_TO_KJ for cn, G in gibbs.items()}

    # Poids de Boltzmann
    conf_nums = sorted(relative_kj.keys())
    energies_list = [relative_kj[cn] for cn in conf_nums]
    weights_array, Z = boltzmann_weights(energies_list, temperature)

    weights_dict = {cn: float(w) for cn, w in zip(conf_nums, weights_array)}

    print(f"\n  Poids de Boltzmann recalculés (DFT {temperature:.1f} K):")
    for cn in conf_nums:
        print(f"    Conformère {cn}: ΔG = {relative_kj[cn]:8.3f} kJ/mol  w = {weights_dict[cn]:.6f}")

    return weights_dict, relative_kj
