#!/usr/bin/env amspython

import os
import argparse
import glob
import math
import numpy as np
import re
import shutil
from scm.plams import *

# Constantes physiques
F = 96485.33212  # Constante de Faraday en C/mol

# Définition des fonctionnelles par catégorie
LDA_FUNCTIONALS = ["VWN", "XALPHA", "Xonly", "Stoll"]
GGA_FUNCTIONALS = ["PBE", "RPBE", "revPBE", "PBEsol", "BLYP", "BP86", "PW91", "mPW", "OLYP", "OPBE", 
                  "KT1", "KT2", "BEE", "BJLDA", "BJPBE", "BJGGA", "S12G", "LB94", "mPBE", "B3LYPgauss"]
METAGGA_FUNCTIONALS = ["M06L", "TPSS", "revTPSS", "MVS", "SCAN", "revSCAN", "r2SCAN", "tb-mBJ"]
HYBRID_FUNCTIONALS = ["B3LYP", "B3LYP*", "B1LYP", "O3LYP", "X3LYP", "BHandH", "BHandHLYP", 
                     "B1PW91", "MPW1PW", "MPW1K", "PBE0", "OPBE0", "TPSSh", "M06", "M06-2X", "S12H"]
METAHYBRID_FUNCTIONALS = ["M08-HX", "M08-SO", "M11", "TPSSH", "PW6B95", "MPW1B95", "MPWB1K", 
                         "PWB6K", "M06-HF", "BMK"]

ALL_FUNCTIONALS = ['HF'] + LDA_FUNCTIONALS + GGA_FUNCTIONALS + METAGGA_FUNCTIONALS + HYBRID_FUNCTIONALS + METAHYBRID_FUNCTIONALS

def setup_workspace(input_dir):
    """
    Configure l'environnement de travail pour les calculs redox.
    
    Cette fonction:
    1. Détermine le chemin parent approprié
    2. Initialise PLAMS avec un dossier de travail dédié
    3. Renvoie le chemin du dossier de travail
    
    Args:
        input_dir (str): Chemin du répertoire contenant les fichiers d'entrée
        
    Returns:
        str: Chemin du répertoire de travail PLAMS
    """
    # Extraction du chemin parent à partir de input_dir
    parent_dir = os.path.dirname(os.path.abspath(input_dir))
    if parent_dir.endswith('/results'):
        parent_dir = os.path.dirname(parent_dir)
        
    # Préserver le format original du chemin
    workdir = os.path.join(parent_dir, "redox")
    
    # Initialisation de PLAMS
    init(folder=workdir)
    
    # Création du dossier pour les résultats
    results_dir = os.path.join(workdir, 'redox_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    return workdir

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script pour calculs redox de conformères"
    )
    parser.add_argument("--input_dir", help="Répertoire des .xyz ou du workdir si mode analysis-only")
    parser.add_argument("--solvent", default="Acetonitrile", help="Solvant")
    parser.add_argument("--functional", "-f", default="PBE0", help="Fonctionnelle")
    parser.add_argument("--basis", "-b", default="DZP", help="Basis set")
    parser.add_argument("--analysis-only", action="store_true", help="Ne fait que l'extraction d'énergies et l'analyse redox sur un workdir existant")
    parser.add_argument("--workdir", help="Chemin du dossier PLAMS (redox) si --analysis-only")
    return parser.parse_args()

def configure_functional(s, functional):
    """
    Configure la fonctionnelle appropriée dans les paramètres Settings
    
    Args:
        s (Settings): Objet Settings à modifier
        functional (str): Nom de la fonctionnelle à configurer
    """
    if functional == "HF":
        s.input.adf.XC.HF = "Yes"
        return
    
    if functional in LDA_FUNCTIONALS:
        s.input.adf.XC.LDA = functional
        return

    if functional in GGA_FUNCTIONALS:
        s.input.adf.XC.GGA = functional
        return

    if functional in METAGGA_FUNCTIONALS:
        s.input.adf.XC.MetaGGA = functional
        s.input.adf.NumericalQuality = "Good"
        return

    if functional in HYBRID_FUNCTIONALS:
        s.input.adf.XC.Hybrid = functional
        return

    if functional in METAHYBRID_FUNCTIONALS:
        s.input.adf.XC.MetaHybrid = functional
        s.input.adf.NumericalQuality = "Good"
        return

    print(f"Attention : Fonctionnelle {functional} non reconnue, utilisation de PBE0 par défaut")
    s.input.adf.XC.Hybrid = "PBE0"

def setup_adf_settings(task='GeometryOptimization', charge=0, spin_polarization=0, 
                       solvent="Acetonitrile", functional="PBE0", basis="DZP"):
    """
    Configure ADF settings object based on task and molecular state
    
    Args:
        task (str): Tâche AMS à exécuter
        charge (int): Charge moléculaire
        spin_polarization (float): Polarisation de spin
        solvent (str): Nom du solvant pour COSMO
        functional (str): Fonctionnelle à utiliser
        basis (str): Base à utiliser
        
    Returns:
        Settings: Objet Settings configuré pour le calcul
    """
    s = Settings()
    s.input.ams.Task = task
    
    # Active le calcul des modes normaux
    s.input.ams.Properties.NormalModes = "Yes"
    
    # Configuration du moteur ADF avec les paramètres spécifiés
    s.input.adf.Basis.Type = basis
    s.input.adf.Basis.Core = "None"
    
    # Configuration de la fonctionnelle en utilisant la fonction dédiée
    configure_functional(s, functional)
    
    s.input.adf.Relativity.Level = "None"
    
    # Paramètres pour les molécules chargées ou avec des électrons non appariés
    if charge != 0:
        s.input.ams.System.Charge = charge
    
    if spin_polarization > 0:
        s.input.adf.SpinPolarization = spin_polarization
        s.input.adf.Unrestricted = "Yes"
    
    # Configuration de la solvatation COSMO
    s.input.adf.Solvation.Solv = f"name={solvent}"

    return s


def optimize_neutral(mol, name, solvent="Acetonitrile", functional="PBE0", basis="DZP", max_attempts=3):
    """
    Étape 1: Optimisation géométrique de la molécule neutre
    
    Args:
        mol (Molecule): Molécule à optimiser
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        max_attempts (int): Nombre maximal de tentatives de correction
        
    Returns:
        AMSJob: Job d'optimisation terminé ou None en cas d'échec
    """
    print(f"\nÉtape 1: Optimisation géométrique de {name} (neutre)")
    settings = setup_adf_settings(task="GeometryOptimization", charge=0, 
                                 solvent=solvent, functional=functional, basis=basis)
    
    # S'assurer que le calcul inclut les modes normaux
    settings.input.ams.Properties.NormalModes = "Yes"
    
    job = AMSJob(settings=settings, name=f"{name}_neutre_opt", molecule=mol)
    job.run()
    
    if job.check():
        print(f"  Optimisation neutre réussie pour {name}")
        
        print(f"  Vérification des fréquences imaginaires...")
        job, fixed = check_and_fix_frequencies(job, f"{name}_neutre", settings, charge=0, max_attempts=max_attempts)
        if fixed:
            print(f"  Structure sans fréquence imaginaire obtenue pour {name} (neutre)")
        else:
            print(f"  AVERTISSEMENT: Des fréquences imaginaires persistent pour {name} (neutre)")
        return job
    else:
        print(f"  ERREUR: Optimisation neutre échouée pour {name}")
        return None


def sp_reduced(job_neutral, name, solvent="Acetonitrile", functional="PBE0", basis="DZP", max_attempts=3):
    """
    Étape 2: Calcul en simple point de la molécule réduite (charge -1)

    Args:
        job_neutral (AMSJob): Job d'optimisation de la molécule neutre
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        max_attempts (int): Nombre maximal de tentatives de correction

    Returns:
        AMSJob: Job de single point terminé ou None en cas d'échec
    """
    # Récupérer la molécule optimisée de l'étape 1
    mol_opt = job_neutral.results.get_main_molecule()

    print(f"\nÉtape 2: Calcul en simple point de {name} (réduit, charge -1)")
    settings = setup_adf_settings(task="SinglePoint", charge=-1, spin_polarization=1.0,
                                 solvent=solvent, functional=functional, basis=basis)

    job = AMSJob(settings=settings, name=f"{name}_reduit_sp", molecule=mol_opt)
    job.run()

    if job.check():
        print(f"  Calcul simple point réussi pour {name} (réduit)")
        return job
    else:
        print(f"  ERREUR: Calcul simple point échoué pour {name} (réduit)")
        return None


def optimize_reduced(job_sp, name, solvent="Acetonitrile", functional="PBE0", basis="DZP", max_attempts=3):
    """
    Étape 3: Optimisation géométrique de la molécule réduite (charge -1)

    Args:
        job_sp (AMSJob): Job de single point de la molécule réduite
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        max_attempts (int): Nombre maximal de tentatives de correction

    Returns:
        AMSJob: Job d'optimisation terminé ou None en cas d'échec
    """
    # Récupérer la molécule réduite du calcul single point
    mol_opt = job_sp.results.get_main_molecule()

    print(f"\nÉtape 3: Optimisation géométrique de {name} (réduit, charge -1)")
    settings = setup_adf_settings(task="GeometryOptimization", charge=-1, spin_polarization=1.0,
                                 solvent=solvent, functional=functional, basis=basis)

    job = AMSJob(settings=settings, name=f"{name}_reduit_opt", molecule=mol_opt)
    job.run()

    if job.check():
        print(f"  Optimisation réduite réussie pour {name}")

        print(f"  Vérification des fréquences imaginaires...")
        job, fixed = check_and_fix_frequencies(job, f"{name}_reduit", settings,
                                              charge=-1, spin_polarization=1.0, max_attempts=max_attempts)
        if fixed:
            print(f"  Structure sans fréquence imaginaire obtenue pour {name} (réduit)")
        else:
            print(f"  AVERTISSEMENT: Des fréquences imaginaires persistent pour {name} (réduit)")
        return job
    else:
        print(f"  ERREUR: Optimisation réduite échouée pour {name}")
        # return None


def sp_oxidized(job_neutral, name, solvent="Acetonitrile", functional="PBE0", basis="DZP", max_attempts=3):
    """
    Étape 4: Calcul en simple point de la molécule oxidée (charge +1)

    Args:
        job_neutral (AMSJob): Job d'optimisation de la molécule neutre
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        max_attempts (int): Nombre maximal de tentatives de correction

    Returns:
        AMSJob: Job de single point terminé ou None en cas d'échec
    """
    # Récupérer la molécule optimisée de l'étape 1
    mol_opt = job_neutral.results.get_main_molecule()

    print(f"\nÉtape 4: Calcul en simple point de {name} (oxidé, charge +1)")
    settings = setup_adf_settings(task="SinglePoint", charge=1, spin_polarization=1.0,
                                 solvent=solvent, functional=functional, basis=basis)

    job = AMSJob(settings=settings, name=f"{name}_oxidé_sp", molecule=mol_opt)
    job.run()

    if job.check():
        print(f"  Calcul simple point réussi pour {name} (oxidé)")
        return job
    else:
        print(f"  ERREUR: Calcul simple point échoué pour {name} (oxidé)")
        return None


def optimize_oxidized(job_sp, name, solvent="Acetonitrile", functional="PBE0", basis="DZP", max_attempts=3):
    """
    Étape 5: Optimisation géométrique de la molécule réduite (charge -1)
    
    Args:
        job_sp (AMSJob): Job de single point de la molécule réduite
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        max_attempts (int): Nombre maximal de tentatives de correction
        
    Returns:
        AMSJob: Job d'optimisation terminé ou None en cas d'échec
    """
    # Récupérer la molécule réduite du calcul single point
    mol_opt = job_sp.results.get_main_molecule()
    
    print(f"\nÉtape 5: Optimisation géométrique de {name} (oxidé, charge +1)")
    settings = setup_adf_settings(task="GeometryOptimization", charge=1, spin_polarization=1.0, 
                                 solvent=solvent, functional=functional, basis=basis)
    
    job = AMSJob(settings=settings, name=f"{name}_oxidé_opt", molecule=mol_opt)
    job.run()
    
    if job.check():
        print(f"  Optimisation oxidée réussie pour {name}")
        
        print(f"  Vérification des fréquences imaginaires...")
        job, fixed = check_and_fix_frequencies(job, f"{name}_oxidé", settings, 
                                              charge=+1, spin_polarization=1.0, max_attempts=max_attempts)
        if fixed:
            print(f"  Structure sans fréquence imaginaire obtenue pour {name} (oxidé)")
        else:
            print(f"  AVERTISSEMENT: Des fréquences imaginaires persistent pour {name} (oxidé)")
        return job
    else:
        print(f"  ERREUR: Optimisation oxidée échouée pour {name}")
        return None


def check_and_fix_frequencies(job_initial, name, settings, charge=0, spin_polarization=0, max_attempts=3):
    """
    Vérifie et corrige les fréquences imaginaires d'un calcul.

    Args:
        job_initial (AMSJob): Job initial dont on veut vérifier/corriger les fréquences
        name (str): Nom de base pour les jobs de correction
        settings (Settings): Paramètres de calcul à utiliser pour les corrections
        charge (int): Charge de la molécule
        spin_polarization (float): Polarisation de spin pour les systèmes à couche ouverte
        max_attempts (int): Nombre maximal de tentatives de correction

    Returns:
        tuple: (job_final, fixed_successfully)
            - job_final: Meilleur job obtenu (avec ou sans fréquences imaginaires)
            - fixed_successfully: Booléen indiquant si toutes les fréquences imaginaires ont été corrigées
    """
    # S'assurer que le calcul initial a réussi
    if not job_initial.check():
        print(f"  ERREUR: Le job initial {job_initial.name} a échoué, impossible de vérifier les fréquences")
        return job_initial, False

    # Créer un dossier pour les corrections
    # correction_dir = os.path.join(os.path.dirname(job_initial.path), f"{name}_freq_corrections")
    # os.makedirs(correction_dir, exist_ok=True)

    # Vérifier les fréquences imaginaires
    output_file = os.path.join(job_initial.path, f"{job_initial.name}.out")
    imaginary_modes = extract_imaginary_modes(output_file)

    # Si pas de fréquence imaginaire, retourner le job initial
    if not imaginary_modes:
        print(f"  Pas de fréquence imaginaire pour {name}")
        return job_initial, True

    # Si des fréquences imaginaires sont détectées, tenter de les corriger
    freqs = [f"{mode['mode']}:{mode['frequency']:.2f} cm-1" for mode in imaginary_modes]
    print(f"  Fréquences imaginaires détectées: {', '.join(freqs)}")
    print(f"  Tentative de correction des fréquences imaginaires...")

    # Obtenir la molécule optimisée depuis le job initial
    current_mol = job_initial.results.get_main_molecule()

    # S'assurer que les settings incluent le calcul des modes normaux
    settings_copy = settings.copy()
    settings_copy.input.ams.Properties.NormalModes = "Yes"

    # Mettre à jour les paramètres de charge et spin si nécessaire
    if charge != 0:
        settings_copy.input.ams.System.Charge = charge
    if spin_polarization > 0:
        settings_copy.input.adf.SpinPolarization = spin_polarization
        settings_copy.input.adf.Unrestricted = "Yes"

    # Perturbation initiale
    perturbation_scale = 0.5

    # Tentatives de correction
    best_job = job_initial
    best_imaginary_count = len(imaginary_modes)

    for attempt in range(1, max_attempts + 1):
        # Trouver le mode avec la fréquence la plus négative
        worst_mode = min(imaginary_modes, key=lambda x: x["frequency"])

        print(f"  Tentative {attempt}/{max_attempts} - Perturbation du mode {worst_mode['mode']} "
              f"(fréquence: {worst_mode['frequency']:.2f} cm-1) avec un facteur d'échelle de {perturbation_scale}")

        # Créer deux versions perturbées (positive et négative)
        pos_perturbed_mol = perturb_molecule(current_mol, worst_mode, perturbation_scale)
        neg_perturbed_mol = perturb_molecule(current_mol, worst_mode, -perturbation_scale)

        # Optimiser la géométrie avec perturbation positive
        job_pos = AMSJob(settings=settings_copy,
                        name=f"{name}_corr_{attempt}_pos",
                        molecule=pos_perturbed_mol)

        # Définir le répertoire de travail pour ce job
        # job_pos.settings.runscript.pre = f"mkdir -p {correction_dir}/{job_pos.name}\ncd {correction_dir}/{job_pos.name}"

        # Exécuter le calcul
        job_pos.run()
        pos_success = job_pos.check()

        # Analyser les résultats de la perturbation positive
        imaginary_modes_pos = []
        if pos_success:
            output_file_pos = os.path.join(job_pos.path, f"{job_pos.name}.out")
            imaginary_modes_pos = extract_imaginary_modes(output_file_pos)

            if not imaginary_modes_pos:
                print(f"  Correction réussie avec perturbation positive (tentative {attempt})")
                return job_pos, True

        # Optimiser la géométrie avec perturbation négative
        job_neg = AMSJob(settings=settings_copy,
                        name=f"{name}_corr_{attempt}_neg",
                        molecule=neg_perturbed_mol)

        # Définir le répertoire de travail pour ce job
        # job_neg.settings.runscript.pre = f"mkdir -p {correction_dir}/{job_neg.name}\ncd {correction_dir}/{job_neg.name}"

        # Exécuter le calcul
        job_neg.run()
        neg_success = job_neg.check()

        # Analyser les résultats de la perturbation négative
        imaginary_modes_neg = []
        if neg_success:
            output_file_neg = os.path.join(job_neg.path, f"{job_neg.name}.out")
            imaginary_modes_neg = extract_imaginary_modes(output_file_neg)

            if not imaginary_modes_neg:
                print(f"  Correction réussie avec perturbation négative (tentative {attempt})")
                return job_neg, True

        # Mettre à jour le meilleur job si nécessaire
        if pos_success and len(imaginary_modes_pos) < best_imaginary_count:
            best_job = job_pos
            best_imaginary_count = len(imaginary_modes_pos)
            print(f"  Nouveau meilleur résultat: perturbation positive avec {best_imaginary_count} fréquences imaginaires")

        if neg_success and len(imaginary_modes_neg) < best_imaginary_count:
            best_job = job_neg
            best_imaginary_count = len(imaginary_modes_neg)
            print(f"  Nouveau meilleur résultat: perturbation négative avec {best_imaginary_count} fréquences imaginaires")

        # Si les deux approches ont échoué, augmenter le facteur de perturbation
        if not pos_success and not neg_success:
            print(f"  Les deux perturbations ont échoué, augmentation du facteur d'échelle")
            perturbation_scale += 0.25
            continue

        # Choisir la meilleure géométrie pour la prochaine itération
        if pos_success and neg_success:
            if len(imaginary_modes_pos) <= len(imaginary_modes_neg):
                current_mol = job_pos.results.get_main_molecule()
                imaginary_modes = imaginary_modes_pos
                print(f"  Continuation avec résultat de perturbation positive")
            else:
                current_mol = job_neg.results.get_main_molecule()
                imaginary_modes = imaginary_modes_neg
                print(f"  Continuation avec résultat de perturbation négative")
        elif pos_success:
            current_mol = job_pos.results.get_main_molecule()
            imaginary_modes = imaginary_modes_pos
            print(f"  Continuation avec résultat de perturbation positive (seul calcul réussi)")
        else:  # neg_success
            current_mol = job_neg.results.get_main_molecule()
            imaginary_modes = imaginary_modes_neg
            print(f"  Continuation avec résultat de perturbation négative (seul calcul réussi)")

        # Augmenter légèrement le facteur de perturbation pour la prochaine itération
        perturbation_scale += 0.1

    # Si nous arrivons ici, c'est que nous n'avons pas réussi à corriger toutes les fréquences imaginaires
    print(f"  AVERTISSEMENT: Fréquences imaginaires persistantes après {max_attempts} tentatives")

    # Retourner le meilleur job obtenu
    return best_job, (best_imaginary_count == 0)


def extract_imaginary_modes(output_file):
    """
    Extrait les modes normaux à fréquence imaginaire du fichier de sortie.

    Args:
        output_file (str): Chemin vers le fichier de sortie du calcul

    Returns:
        list: Liste des modes imaginaires, chacun contenant fréquence et déplacements atomiques
    """
    imaginary_modes = []

    with open(output_file, 'r') as f:
        content = f.read()

    # Utiliser la regex fournie pour trouver tous les modes à fréquence négative
    pattern = r' Mode:\s+(\d+)\s+Frequency \(cm-1\):\s+([-]\d+\.\d+)\s+Intensity \(km\/mol\):\s+\d+\.\d+\s*\n Index\s+Atom\s+\-+ Displacements \(x\/y\/z\) \-+\n((?:\s+\d+\s+\w+\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s*\n)+)'

    # Trouver tous les modes à fréquence négative
    for match in re.finditer(pattern, content):
        mode_num = int(match.group(1))
        frequency = float(match.group(2))  # Déjà négatif par construction du pattern
        displacements_text = match.group(3)

        # Traiter les déplacements atomiques
        mode_displacements = []
        for line in displacements_text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 5:  # Format attendu: index, symbole, x, y, z
                atom_index = int(parts[0])
                atom_symbol = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])

                mode_displacements.append({
                    'index': atom_index,
                    'symbol': atom_symbol,
                    'displacements': (x, y, z)
                })

        # Ajouter ce mode imaginaire à notre liste
        imaginary_modes.append({
            'mode': mode_num,
            'frequency': frequency,
            'displacements': mode_displacements
        })

    return imaginary_modes


def perturb_molecule(molecule, mode_info, scale_factor=0.5):
    """
    Crée une nouvelle géométrie moléculaire en perturbant la structure initiale le long d'un mode normal.

    Args:
        molecule (Molecule): L'objet Molecule à perturber
        mode_info (dict): Informations sur le mode normal (fréquence et déplacements)
        scale_factor (float): Facteur d'échelle pour le déplacement (positif)

    Returns:
        Molecule: Nouvelle molécule avec géométrie perturbée
    """
    # Créer une copie profonde de la molécule
    perturbed_mol = molecule.copy()

    # Appliquer les déplacements
    for atom_info in mode_info['displacements']:
        atom_index = atom_info['index'] - 1  # PLAMS utilise des indices commençant à 0
        dx, dy, dz = atom_info['displacements']

        # Coordonnées actuelles de l'atome
        current_coords = perturbed_mol.atoms[atom_index].coords

        # Nouvelles coordonnées (perturbées dans la direction du mode)
        # Pour les modes imaginaires, on perturbe dans le sens positif du vecteur
        # car on cherche à "sortir" du point selle
        new_coords = (
            current_coords[0] + scale_factor * dx,
            current_coords[1] + scale_factor * dy,
            current_coords[2] + scale_factor * dz
        )

        # Mettre à jour les coordonnées
        perturbed_mol.atoms[atom_index].coords = new_coords

    return perturbed_mol


def kabsch_rmsd(mol1, mol2):
    """
    Calcule le RMSD entre deux structures moléculaires après alignement optimal (algorithme de Kabsch).

    Args:
        mol1, mol2 (Molecule): Objets Molecule de PLAMS à comparer

    Returns:
        float: RMSD après alignement optimal en Å
    """
    import numpy as np

    # Vérifie que les molécules ont le même nombre d'atomes
    if len(mol1) != len(mol2):
        print(f"Erreur: les molécules ont un nombre différent d'atomes ({len(mol1)} vs {len(mol2)})")
        return float('inf')

    # Extraire les coordonnées
    coords1 = np.array([atom.coords for atom in mol1])
    coords2 = np.array([atom.coords for atom in mol2])

    # Centrer les structures
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Calculer la matrice de covariance
    covariance = np.dot(coords1_centered.T, coords2_centered)

    # Décomposition en valeurs singulières
    try:
        U, S, Vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        print("Erreur lors de la décomposition SVD")
        return float('inf')

    # Vérifier si une réflexion est nécessaire
    d = np.linalg.det(np.dot(Vt.T, U.T))
    if d < 0:
        U[:, -1] = -U[:, -1]

    # Matrice de rotation optimale
    rotation = np.dot(Vt.T, U.T)

    # Appliquer la rotation à coords1_centered
    coords1_rotated = np.dot(coords1_centered, rotation)

    # Calculer le RMSD
    squared_diff = np.sum((coords1_rotated - coords2_centered) ** 2, axis=1)
    rmsd = np.sqrt(np.mean(squared_diff))

    return rmsd

def compare_conformers_rmsd(job_results):
    """
    Compare chaque conformère réduit avec tous les conformères neutres
    pour identifier les changements structurels potentiels.

    Args:
        job_results (dict): Dictionnaire contenant les résultats des jobs

    Returns:
        tuple: Un tuple contenant (rmsd_results, rmsd_export_data)
               rmsd_results: Dictionnaire avec les valeurs RMSD
               rmsd_export_data: Données formatées pour l'export
    """
    rmsd_results = {}
    reduced_rmsd_summary = []  # Pour stocker les résultats des conformères réduits
    oxidized_rmsd_summary = []  # Pour stocker les résultats des conformères oxidés
    rmsd_warnings = []  # Pour stocker les avertissements

    # Liste des noms de tous les conformères
    conformer_names = list(job_results.keys())

    print("\n" + "="*80)
    print("ANALYSE STRUCTURELLE PAR RMSD (ALGORITHME DE KABSCH)")
    print("="*80)

    # Pour chaque conformère réduit
    for reduced_name in conformer_names:
        if 'réduit_opt' not in job_results[reduced_name]:
            continue

        reduced_job = job_results[reduced_name]['réduit_opt']
        if not reduced_job or not reduced_job.ok():
            continue

        reduced_mol = reduced_job.results.get_main_molecule()
        rmsd_results[reduced_name] = {}

        # Comparer avec tous les neutres
        min_reduced_rmsd_value = float('inf')
        min_reduced_rmsd_name = None

        for neutral_name in conformer_names:
            if 'neutre_opt' not in job_results[neutral_name]:
                continue

            neutral_job = job_results[neutral_name]['neutre_opt']
            if not neutral_job or not neutral_job.ok():
                continue

            neutral_mol = neutral_job.results.get_main_molecule()

            # Calculer le RMSD
            try:
                rmsd = kabsch_rmsd(reduced_mol, neutral_mol)
                rmsd_results[reduced_name][neutral_name] = rmsd

                # Garder trace du RMSD minimum
                if rmsd < min_reduced_rmsd_value:
                    min_reduced_rmsd_value = rmsd
                    min_reduced_rmsd_name = neutral_name

            except Exception as e:
                print(f"Erreur lors du calcul RMSD entre {reduced_name} et {neutral_name}: {str(e)}")
                rmsd_results[reduced_name][neutral_name] = float('inf')

        # Ajouter le meilleur résultat au résumé
        if min_reduced_rmsd_name:
            reduced_rmsd_summary.append((reduced_name, min_reduced_rmsd_name, min_reduced_rmsd_value))

            # Déterminer si c'est un changement de conformère
            if reduced_name != min_reduced_rmsd_name:
                warning = f"ATTENTION: Le conformère réduit {reduced_name} correspond mieux au conformère neutre {min_reduced_rmsd_name} (RMSD: {min_reduced_rmsd_value:.4f} Å)"
                rmsd_warnings.append(warning)
                print(warning)
            else:
                # Vérifier si le RMSD est élevé même pour le même conformère
                if min_reduced_rmsd_value > 0.5:  # Seuil arbitraire, à ajuster selon vos molécules
                    warning = f"ATTENTION: Le conformère réduit {reduced_name} présente un RMSD élevé ({min_reduced_rmsd_value:.4f} Å) par rapport à son conformère neutre"
                    rmsd_warnings.append(warning)
                    print(warning)
                else:
                    msg = f"Le conformère réduit {reduced_name} correspond bien à son conformère neutre (RMSD: {min_reduced_rmsd_value:.4f} Å)"
                    rmsd_warnings.append(msg)
                    print(msg)

    # Pour chaque conformère oxidé
    for oxidized_name in conformer_names:
        if 'oxidé_opt' not in job_results[oxidized_name]:
            continue

        oxidized_job = job_results[oxidized_name]['oxidé_opt']
        if not oxidized_job or not oxidized_job.ok():
            continue

        oxidized_mol = oxidized_job.results.get_main_molecule()
        if oxidized_name not in rmsd_results:
            rmsd_results[oxidized_name] = {}

        # Comparer avec tous les neutres
        min_oxidized_rmsd_value = float('inf')
        min_oxidized_rmsd_name = None

        for neutral_name in conformer_names:
            if 'neutre_opt' not in job_results[neutral_name]:
                continue

            neutral_job = job_results[neutral_name]['neutre_opt']
            if not neutral_job or not neutral_job.ok():
                continue

            neutral_mol = neutral_job.results.get_main_molecule()

            # Calculer le RMSD
            try:
                rmsd = kabsch_rmsd(oxidized_mol, neutral_mol)
                rmsd_results[oxidized_name][neutral_name] = rmsd

                # Garder trace du RMSD minimum
                if rmsd < min_oxidized_rmsd_value:
                    min_oxidized_rmsd_value = rmsd
                    min_oxidized_rmsd_name = neutral_name

            except Exception as e:
                print(f"Erreur lors du calcul RMSD entre {oxidized_name} et {neutral_name}: {str(e)}")
                rmsd_results[oxidized_name][neutral_name] = float('inf')

        # Ajouter le meilleur résultat au résumé
        if min_oxidized_rmsd_name:
            oxidized_rmsd_summary.append((oxidized_name, min_oxidized_rmsd_name, min_oxidized_rmsd_value))

            # Déterminer si c'est un changement de conformère
            if oxidized_name != min_oxidized_rmsd_name:
                warning = f"ATTENTION: Le conformère oxidé {oxidized_name} correspond mieux au conformère neutre {min_oxidized_rmsd_name} (RMSD: {min_oxidized_rmsd_value:.4f} Å)"
                rmsd_warnings.append(warning)
                print(warning)
            else:
                # Vérifier si le RMSD est élevé même pour le même conformère
                if min_oxidized_rmsd_value > 0.5:  # Seuil arbitraire, à ajuster selon vos molécules
                    warning = f"ATTENTION: Le conformère oxidé {oxidized_name} présente un RMSD élevé ({min_oxidized_rmsd_value:.4f} Å) par rapport à son conformère neutre"
                    rmsd_warnings.append(warning)
                    print(warning)
                else:
                    msg = f"Le conformère oxidé {oxidized_name} correspond bien à son conformère neutre (RMSD: {min_oxidized_rmsd_value:.4f} Å)"
                    rmsd_warnings.append(msg)
                    print(msg)

    # Afficher uniquement le RMSD minimum pour chaque conformère
    print("\n" + "="*80)
    print("RÉSUMÉ DE L'ANALYSE RMSD")
    print("="*80)
    print(f"{'Conformère réduit':<25} {'Conformère neutre le plus proche':<30} {'RMSD (Å)':<10}")
    print("-"*70)

    for reduced_name, min_reduced_rmsd_name, min_reduced_rmsd_value in reduced_rmsd_summary:
        print(f"{reduced_name:<25} {min_reduced_rmsd_name:<30} {min_reduced_rmsd_value:.4f}")

    print("-"*70)
    print(f"{'Conformère oxidé':<25} {'Conformère neutre le plus proche':<30} {'RMSD (Å)':<10}")
    print("-"*70)

    for oxidized_name, min_oxidized_rmsd_name, min_oxidized_rmsd_value in oxidized_rmsd_summary:
        print(f"{oxidized_name:<25} {min_oxidized_rmsd_name:<30} {min_oxidized_rmsd_value:.4f}")

    # Créer le dictionnaire d'export
    rmsd_export_data = {
        'summary_reduced': reduced_rmsd_summary,
        'summary_oxidized': oxidized_rmsd_summary,
        'warnings': rmsd_warnings
    }

    return rmsd_results, rmsd_export_data

def load_existing_calculations(workdir):
    """
    Charge les calculs existants à partir d'un répertoire de travail PLAMS en mode analysis-only,
    en appliquant une logique de sélection intelligente pour choisir les meilleurs calculs disponibles.

    Règles de priorité:
    1. Pour les calculs avec corrections: choisir le numéro de correction le plus élevé
    2. Pour les options pos/neg: priorité à neg puis pos
    3. Si aucune correction n'est disponible, utiliser la version _opt standard

    Args:
        workdir (str): Chemin vers le répertoire de travail PLAMS (contenant le dossier 'redox')

    Returns:
        dict: Dictionnaire contenant les résultats des calculs pour chaque molécule et conformère
    """
    print(f"Chargement des calculs existants depuis {workdir}...")

    # S'assurer que le chemin existe
    if not os.path.exists(workdir) or not os.path.isdir(workdir):
        raise ValueError(f"Le répertoire {workdir} n'existe pas ou n'est pas un dossier")

    # Définir le chemin du répertoire redox
    if os.path.basename(workdir) == 'redox':
        redox_dir = workdir
    else:
        redox_dir = os.path.join(workdir, 'redox')

    if not os.path.exists(redox_dir):
        raise ValueError(f"Le répertoire redox {redox_dir} n'existe pas")

    # Collecter tous les dossiers de calcul disponibles
    calc_dirs = [d for d in os.listdir(redox_dir)
               if os.path.isdir(os.path.join(redox_dir, d)) and
               "_conf_" in d and
               os.path.exists(os.path.join(redox_dir, d, 'ams.rkf'))]

    # Organiser les dossiers par molécule, conformère et type de calcul
    organized_calcs = {}  # Structure: {mol_name: {conf_num: {job_type: [dirs]}}}

    for calc_dir in calc_dirs:
        try:
            # Analyser le nom du dossier pour extraire les composants
            parts = calc_dir.split('_')

            # Trouver l'index du "conf"
            if "conf" not in parts:
                continue
            conf_idx = parts.index("conf")

            # Extraire les informations
            mol_name_parts = parts[:conf_idx-1]  # Tous les éléments avant "conf"
            mol_name = "_".join(mol_name_parts)
            conf_num = parts[conf_idx+1]  # Numéro après "conf"

            # Déterminer le type de calcul (neutre, reduit, oxidé)
            job_type = None
            for job in ["neutre", "reduit", "oxidé"]:
                if job in parts:
                    job_type = job
                    break

            if not job_type:
                continue  # Ignorer si pas de type reconnu

            # Initialiser la structure si nécessaire
            if mol_name not in organized_calcs:
                organized_calcs[mol_name] = {}
            if conf_num not in organized_calcs[mol_name]:
                organized_calcs[mol_name][conf_num] = {}
            if job_type not in organized_calcs[mol_name][conf_num]:
                organized_calcs[mol_name][conf_num][job_type] = []

            # Ajouter ce calcul à la liste
            organized_calcs[mol_name][conf_num][job_type].append(calc_dir)

        except Exception as e:
            print(f"Erreur lors de l'analyse du dossier {calc_dir}: {str(e)}")
            continue

    # Initialiser le dictionnaire des résultats
    job_results = {}

    # Sélectionner les meilleurs calculs et les charger
    for mol_name, conformers in organized_calcs.items():
        for conf_num, job_types in conformers.items():
            conformer_name = f"{mol_name}_conf_{conf_num}"
            job_results[conformer_name] = {}

            print(f"\nTraitement du conformère: {conformer_name}")

            for job_type, calc_dirs in job_types.items():
                # Sélectionner le meilleur calcul pour ce type
                best_calc = select_best_calculation(calc_dirs)

                if best_calc:
                    # Normaliser le type de job pour la sortie finale
                    # Corriger "reduit" en "réduit" avec accent pour la compatibilité
                    normalized_job_type = job_type
                    if job_type == "reduit":
                        normalized_job_type = "réduit"

                    output_job_type = f"{normalized_job_type}_opt"

                    try:
                        # Charger le calcul
                        ams_rkf_path = os.path.join(redox_dir, best_calc, 'ams.rkf')
                        loaded_job = AMSJob.load_external(ams_rkf_path)

                        # Vérifier que le job est valide
                        if loaded_job and loaded_job.ok():
                            job_results[conformer_name][output_job_type] = loaded_job
                            print(f"  Chargé: {output_job_type} depuis {best_calc}")
                        else:
                            print(f"  Erreur: Le job {output_job_type} ({best_calc}) n'est pas valide")
                            job_results[conformer_name][output_job_type] = None
                    except Exception as e:
                        print(f"  Erreur lors du chargement de {best_calc}: {str(e)}")
                        job_results[conformer_name][output_job_type] = None
                else:
                    normalized_job_type = job_type
                    if job_type == "reduit":
                        normalized_job_type = "réduit"

                    output_job_type = f"{normalized_job_type}_opt"
                    print(f"  Aucun calcul valide trouvé pour {output_job_type}")
                    job_results[conformer_name][output_job_type] = None

    # Afficher un résumé
    print("\nRésumé des calculs chargés:")
    total_loaded = 0
    for conformer, jobs in job_results.items():
        loaded_jobs = sum(1 for job in jobs.values() if job is not None)
        total_loaded += loaded_jobs
        print(f"{conformer}: {loaded_jobs}/{len(jobs)} calculs chargés")

    print(f"\nTotal: {total_loaded} calculs chargés")

    return job_results

def select_best_calculation(calc_dirs):
    """
    Sélectionne le meilleur calcul parmi une liste de dossiers selon les priorités:
    1. Correction avec numéro le plus élevé
    2. Préférence pour 'neg' sur 'pos'
    3. Calcul standard si aucun calcul corrigé n'est disponible

    Args:
        calc_dirs (list): Liste des noms de dossiers de calculs

    Returns:
        str: Nom du meilleur dossier, ou None si aucun dossier valide
    """
    if not calc_dirs:
        return None

    # Séparer les calculs standards et les calculs corrigés
    standard = []
    corrected_info = []  # Liste de tuples (dir_name, corr_num, is_neg, is_pos)

    for dir_name in calc_dirs:
        parts = dir_name.split('_')

        # Vérifier s'il s'agit d'un calcul standard (se termine par _opt)
        if "corr" not in parts and dir_name.endswith("_opt"):
            standard.append(dir_name)
            continue

        # Traiter les calculs corrigés
        try:
            # Trouver l'index de "corr"
            if "corr" in parts:
                corr_idx = parts.index("corr")
                corr_num = int(parts[corr_idx + 1])

                # Déterminer si pos ou neg
                is_neg = "neg" in parts
                is_pos = "pos" in parts

                corrected_info.append((dir_name, corr_num, is_neg, is_pos))
        except (ValueError, IndexError):
            # Si nous ne pouvons pas extraire correctement l'info, on l'ignore
            continue

    # Trier selon les priorités: numéro de correction décroissant, puis neg > pos
    if corrected_info:
        sorted_info = sorted(corrected_info,
                             key=lambda x: (-x[1], x[2], -x[3]))  # -x[1] pour trier par corr_num décroissant
        return sorted_info[0][0]  # Retourner le nom du meilleur dossier

    # Si aucun calcul corrigé n'est disponible ou valide, utiliser le calcul standard
    if standard:
        return standard[0]

    # Si rien n'est disponible, retourner le premier de la liste originale
    return calc_dirs[0] if calc_dirs else None

def export_molecules(jobs_data, prefix="redox_results"):
    """
    Exporte les fichiers de sortie .out dans un sous-dossier spécifique du dossier de travail PLAMS

    Args:
        jobs_data (dict): Dictionnaire avec les résultats des jobs par nom de molécule
        prefix (str, optional): Nom du sous-dossier pour l'exportation. Par défaut "redox_results"
    """
    try:
        # Utiliser le dossier de travail actuel de PLAMS
        current_workdir = config.default_jobmanager.workdir
        output_dir = os.path.join(current_workdir, prefix)

        # Créer le dossier de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nExportation des résultats de calcul dans {output_dir}/")

        for name, jobs in jobs_data.items():
            # Ignorer les molécules dont tous les calculs n'ont pas réussi
            if not all(jobs.values()):
                continue

            # Traiter chaque type de job (neutral, reduced_sp, reduced_opt, oxidized_sp, oxidized_opt)
            for job_type, job in jobs.items():
                # Construire le chemin du fichier source
                out_file = os.path.join(job.path, f"{job.name}.out")

                try:
                    # Définir le chemin cible
                    target_file = os.path.join(output_dir, f"{name}_{job_type}.out")

                    # Copier le fichier avec ses métadonnées
                    import shutil
                    shutil.copy2(out_file, target_file)
                    print(f"  Succès: fichier {name}_{job_type}.out exporté")
                except FileNotFoundError:
                    print(f"  Avertissement: fichier de sortie pour {name}_{job_type} introuvable ({out_file})")
                except PermissionError:
                    print(f"  Erreur: permissions insuffisantes pour copier {out_file}")
                except Exception as e:
                    print(f"  Erreur lors de la copie de {out_file}: {str(e)}")

        print(f"Exportation terminée.")
    except Exception as e:
        print(f"Erreur globale lors de l'exportation des fichiers: {str(e)}")


def extract_engine_energies(workdir_parent):
    """
    Lit les fichiers .out dans le dossier redox*/redox_results/,
    extrait diverses valeurs d'énergie (Engine Energy, Internal Energy U, -T*S, Gibbs free energy)
    ainsi que les valeurs HOMO et LUMO et les affiche.

    Args:
        workdir_parent (str): Chemin du dossier parent où se trouvent les dossiers redox*

    Returns:
        dict: Dictionnaire contenant les énergies extraites par fichier
    """
    import os
    import re

    print("\n" + "="*80)
    print("ÉNERGIES EXTRAITES DES FICHIERS DE SORTIE")
    print("="*80)

    # Dictionnaire pour stocker les résultats
    energy_results = {}
    debug_info = {}  # Dictionnaire pour stocker les informations de débogage

    # Définition des patterns regex pour chaque valeur d'énergie
    patterns = {
        'Ee': r"Energy from Engine:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)",
        'U': r"Internal Energy U:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)",
        'TS': r"  -T\*S:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)",
        'G': r"Gibbs free energy:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)",
        # Nouveaux patterns pour HOMO et LUMO pouvant avoir une ou deux valeurs
        'HOMO': r"HOMO \(eV\):\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?",
        'LUMO': r"LUMO \(eV\):\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?"
    }

    try:
        # Vérifier que le dossier parent existe
        if not os.path.exists(workdir_parent):
            print(f"ERREUR: Le dossier parent '{workdir_parent}' n'existe pas")
            return energy_results

        # Trouver tous les dossiers redox*
        try:
            redox_dirs = [d for d in os.listdir(workdir_parent)
                          if d.startswith('redox') and os.path.isdir(os.path.join(workdir_parent, d))]
        except PermissionError:
            print(f"ERREUR: Permission refusée pour accéder à '{workdir_parent}'")
            return energy_results

        if not redox_dirs:
            print(f"ERREUR: Aucun dossier redox* trouvé dans {workdir_parent}")
            return energy_results

        # Utiliser le dossier redox le plus récent (numériquement le plus élevé)
        latest_redox = sorted(redox_dirs)[-1]
        results_dir = os.path.join(workdir_parent, latest_redox, 'redox_results')
        print(f"Utilisation du dossier le plus récent: {latest_redox}")

        if not os.path.exists(results_dir):
            print(f"ERREUR: Le dossier {results_dir} n'existe pas")
            return energy_results

        # Trouver tous les fichiers .out
        try:
            out_files = [f for f in os.listdir(results_dir) if f.endswith('.out')]
            print(f"Fichiers .out trouvés: {out_files}")
        except PermissionError:
            print(f"ERREUR: Permission refusée pour accéder à '{results_dir}'")
            return energy_results

        if not out_files:
            print(f"ERREUR: Aucun fichier .out trouvé dans {results_dir}")
            return energy_results

        # En-tête du tableau
        print(f"{'Fichier':<30} {'Ee':<15} {'U':<15} {'-T*S':<15} {'Gibbs Energy':<15} {'HOMO (eV)':<15} {'LUMO (eV)':<15}")
        print("-"*120)

        # Analyser chaque fichier .out
        for out_file in sorted(out_files):
            file_path = os.path.join(results_dir, out_file)
            energy_values = {key: None for key in patterns.keys()}

            # Informations de débogage pour ce fichier
            debug_info[out_file] = {
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'found_patterns': {key: False for key in patterns},
                'raw_matches': {}
            }

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()

                    # Rechercher toutes les valeurs avec regex
                    for key, pattern in patterns.items():
                        if key in ['HOMO', 'LUMO']:
                            # Traitement spécial pour HOMO et LUMO qui peuvent avoir deux valeurs
                            match = re.search(pattern, content)
                            if match:
                                debug_info[out_file]['found_patterns'][key] = True
                                debug_info[out_file]['raw_matches'][key] = match.group(0)

                                try:
                                    value1 = float(match.group(1))
                                    value2 = None
                                    if match.group(2):
                                        value2 = float(match.group(2))

                                    # Choisir la valeur en fonction de HOMO ou LUMO
                                    if key == 'HOMO':
                                        # Pour HOMO, prendre la valeur la plus haute (moins négative)
                                        if value2 is not None:
                                            energy_values[key] = max(value1, value2)
                                        else:
                                            energy_values[key] = value1
                                    else:  # LUMO
                                        # Pour LUMO, prendre la valeur la plus basse
                                        if value2 is not None:
                                            energy_values[key] = min(value1, value2)
                                        else:
                                            energy_values[key] = value1

                                except ValueError as e:
                                    debug_info[out_file]['raw_matches'][f"{key}_error"] = str(e)
                        else:
                            # Traitement standard pour les autres valeurs
                            match = re.search(pattern, content)
                            if match:
                                debug_info[out_file]['found_patterns'][key] = True
                                debug_info[out_file]['raw_matches'][key] = match.group(0)
                                try:
                                    energy_values[key] = float(match.group(1))
                                except ValueError as e:
                                    debug_info[out_file]['raw_matches'][f"{key}_error"] = str(e)

            except Exception as e:
                print(f"ERREUR lors de la lecture du fichier {file_path}: {str(e)}")
                debug_info[out_file]['error'] = str(e)
                continue

            # Stocker les résultats dans le dictionnaire
            energy_results[out_file] = energy_values

            # Formater les valeurs pour l'affichage
            formatted_values = {}
            for key, value in energy_values.items():
                if value is None:
                    formatted_values[key] = "Non trouvée"
                elif key in ['HOMO', 'LUMO']:  # Valeurs en eV
                    formatted_values[key] = f"{value:.4f}"
                else:  # Valeurs en kJ/mol
                    formatted_values[key] = f"{value:.2f}"

            # Afficher les résultats
            print(f"{out_file:<30} "
                  f"{formatted_values['Ee']:<15} "
                  f"{formatted_values['U']:<15} "
                  f"{formatted_values['TS']:<15} "
                  f"{formatted_values['G']:<15} "
                  f"{formatted_values['HOMO']:<15} "
                  f"{formatted_values['LUMO']:<15}")

            # Vérifier si toutes les valeurs sont None et générer un avertissement
            if all(v is None for v in energy_values.values()):
                print(f"  AVERTISSEMENT: Aucune valeur d'énergie trouvée dans {out_file}!")

        print("="*120)
        print("Les énergies Ee, U, TS et G sont exprimées en kJ/mol, HOMO et LUMO en eV")

        # Statistiques de réussite
        total_files = len(out_files)
        successful_files = sum(1 for file in out_files if any(energy_results.get(file, {}).get(key) is not None for key in patterns))
        print(f"Extraction réussie pour {successful_files}/{total_files} fichiers ({successful_files/total_files*100:.1f}%)")

    except Exception as e:
        print(f"ERREUR GLOBALE lors de l'extraction des énergies: {str(e)}")
        import traceback
        traceback.print_exc()

    return energy_results


def analyze_redox_energies(energy_data, workdir, temperature=298.15, rmsd_export_data=None):
    """
    Analyse les valeurs d'énergie extraites et calcule les paramètres redox.
    """
    # Constante énergétique
    F = 96.48533212  # Constante de Faraday en kC/mol
    R = 8.314462618  # Constante des gaz parfaits en J/(mol·K)

    print("\n" + "="*80)
    print("ANALYSE ÉNERGÉTIQUE DU PROCESSUS REDOX")
    print("="*80)

    try:
        # Initialiser les dictionnaires pour stocker les résultats
        conformers_data = {}
        redox_parameters = {}
        avg_params = {}
        potentials = {
            'reduction': {},
            'oxidation': {}
        }
        boltzmann_weights = {}

        # Modifier le pattern regex pour correspondre à vos noms de fichiers réels
        pattern = r'(.+?)_(neutre|réduit|oxidé)_(opt|sp)\.out'

        # Extraire les informations de chaque fichier
        for filename, energies in energy_data.items():
            match = re.match(pattern, filename)
            if not match:
                print(f"Format de nom de fichier non reconnu: {filename}")
                continue

            conformer = match.group(1)
            calc_type = match.group(2)  # 'neutre', 'réduit', ou 'oxidé'
            job_type = match.group(3)   # 'opt' ou 'sp'

            # Convertir en clé standardisée
            if calc_type == 'neutre':
                standardized_calc_type = 'neutral'
            elif calc_type == 'réduit':
                standardized_calc_type = 'reduced' if job_type == 'sp' else 'reduced_opt'
            elif calc_type == 'oxidé':
                standardized_calc_type = 'oxidized' if job_type == 'sp' else 'oxidized_opt'
            else:
                continue

            # Initialiser le dictionnaire pour ce conformère si nécessaire
            if conformer not in conformers_data:
                conformers_data[conformer] = {}

            # Ajouter les données d'énergie pour ce calcul
            conformers_data[conformer][standardized_calc_type] = energies

        # Vérifier quels conformères ont toutes les données nécessaires
        required_calc_types = ['neutral', 'reduced', 'reduced_opt', 'oxidized', 'oxidized_opt']
        complete_conformers = []

        for conformer, calcs in conformers_data.items():
            all_energies_available = True
            # Vérifier que tous les types de calcul sont disponibles
            for calc_type in required_calc_types:
                if calc_type not in calcs:
                    print(f"Conformère {conformer} manque le calcul {calc_type}")
                    all_energies_available = False
                    break
                # Vérifier que toutes les énergies sont disponibles pour ce calcul
                for energy_type in ['Ee', 'U', 'TS', 'G']:
                    if calcs[calc_type][energy_type] is None:
                        print(f"Conformère {conformer}, calcul {calc_type} manque l'énergie {energy_type}")
                        all_energies_available = False
                        break

            if all_energies_available:
                complete_conformers.append(conformer)

        # Cette partie était indentée incorrectement et faisait partie de la boucle
        if not complete_conformers:
            print("ERREUR: Aucun conformère n'a tous les calculs nécessaires avec toutes les énergies")
            return {}, {}, {}, {}

        print(f"Conformères complets à analyser: {', '.join(complete_conformers)}")

        # Calculer Delta U pour chaque cas
        for conformer in complete_conformers:
            for calc_type in required_calc_types:
                energies = conformers_data[conformer][calc_type]
                if energies['Ee'] is not None and energies['U'] is not None:
                    energies['delta_U'] = energies['U'] - energies['Ee']
                else:
                    energies['delta_U'] = 0.0

        # Calculer les paramètres redox pour chaque conformère
        for conformer in complete_conformers:
            neutral = conformers_data[conformer]['neutral']
            red_sp = conformers_data[conformer]['reduced']
            red_opt = conformers_data[conformer]['reduced_opt']
            ox_sp = conformers_data[conformer]['oxidized']
            ox_opt = conformers_data[conformer]['oxidized_opt']

            # Calcul pour le processus de réduction
            red_params = {}

            # Calcul des paramètres énergétiques de base
            red_params['delta_G'] = red_opt['G'] - neutral['G']
            red_params['EA'] = red_sp['Ee'] - neutral['Ee']
            red_params['Edef'] = red_opt['Ee'] - red_sp['Ee']
            red_params['delta_delta_U'] = red_opt['delta_U'] - neutral['delta_U']
            red_params['T_delta_S'] = red_opt['TS'] - neutral['TS']

            # Calcul pour le processus d'oxydation
            ox_params = {}

            # Paramètres énergétiques pour l'oxydation
            ox_params['delta_G'] = neutral['G'] - ox_opt['G']
            ox_params['EI'] = neutral['Ee'] - ox_sp['Ee']
            ox_params['Edef'] = ox_sp['Ee'] - ox_opt['Ee']
            ox_params['delta_delta_U'] = neutral['delta_U'] - ox_opt['delta_U']
            ox_params['T_delta_S'] = neutral['TS'] - ox_opt['TS']

            # Stocker les paramètres calculés
            redox_parameters[conformer] = {
                'reduction': red_params,
                'oxidation': ox_params
            }

        # Calculer les poids de Boltzmann en fonction des énergies de Gibbs des conformères neutres
        try:
            # Trouver l'énergie de Gibbs minimum parmi tous les conformères
            min_G = min(conformers_data[conf]['neutral']['G'] for conf in complete_conformers
                       if conformers_data[conf]['neutral']['G'] is not None)

            # Calculer les facteurs de Boltzmann
            denominator = 0.0
            for conformer in complete_conformers:
                G = conformers_data[conformer]['neutral']['G']
                if G is not None:
                    rel_G = G - min_G
                    # Conversion kJ/mol en J/mol pour correspondre à R
                    rel_G_joules = rel_G * 1000
                    boltzmann_factor = math.exp(-rel_G_joules / (R * temperature))
                    boltzmann_weights[conformer] = boltzmann_factor
                    denominator += boltzmann_factor
                else:
                    boltzmann_weights[conformer] = 0.0

            # Normaliser les poids
            if denominator > 0:
                for conformer in complete_conformers:
                    boltzmann_weights[conformer] /= denominator
            else:
                # Si problème, attribuer poids égaux
                for conformer in complete_conformers:
                    boltzmann_weights[conformer] = 1.0 / len(complete_conformers)

        except Exception as e:
            print(f"ERREUR lors du calcul des poids de Boltzmann: {str(e)}")
            # Attribuer poids égaux en cas d'erreur
            for conformer in complete_conformers:
                boltzmann_weights[conformer] = 1.0 / len(complete_conformers)

        # Calculer les moyennes pondérées pour la réduction
        avg_params_red = {
            'delta_G': 0.0,
            'EA': 0.0,
            'Edef': 0.0,
            'delta_delta_U': 0.0,
            'T_delta_S': 0.0
        }

        # Calculer les moyennes pondérées pour l'oxydation
        avg_params_ox = {
            'delta_G': 0.0,
            'EI': 0.0,
            'Edef': 0.0,
            'delta_delta_U': 0.0,
            'T_delta_S': 0.0
        }

        # Calculer les moyennes pondérées en utilisant les poids de Boltzmann
        for conformer in complete_conformers:
            weight = boltzmann_weights[conformer]

            # Paramètres de réduction
            for param in avg_params_red:
                avg_params_red[param] += weight * redox_parameters[conformer]['reduction'][param]

            # Paramètres d'oxydation
            for param in avg_params_ox:
                avg_params_ox[param] += weight * redox_parameters[conformer]['oxidation'][param]

        # Stocker les résultats combinés
        avg_params = {
            'reduction': avg_params_red,
            'oxidation': avg_params_ox
        }

        # Calculer les potentiels (V vs référence)
        potentials = {
            'reduction': {param: -value/F for param, value in avg_params_red.items()},
            'oxidation': {param: value/F for param, value in avg_params_ox.items()}  # Inverser le signe pour l'oxydation
        }

        # Récupérer les données HOMO/LUMO
        neutral_orbitals = {}
        for conformer in complete_conformers:
            # Pour chaque conformère, chercher les données du calcul neutre
            if conformer in conformers_data and 'neutral' in conformers_data[conformer]:
                neutral_calc = conformers_data[conformer]['neutral']
                if 'HOMO' in neutral_calc and 'LUMO' in neutral_calc and neutral_calc['HOMO'] is not None and neutral_calc['LUMO'] is not None:
                    neutral_orbitals[conformer] = {
                        'HOMO': neutral_calc['HOMO'],
                        'LUMO': neutral_calc['LUMO'],
                        'Gap': neutral_calc['LUMO'] - neutral_calc['HOMO']
                    }

        # Calculer moyennes pondérées des orbitales
        weighted_homo = 0.0
        weighted_lumo = 0.0
        total_weight = 0.0  # Pour s'assurer que les poids sont normalisés

        for conformer in complete_conformers:
            if conformer in neutral_orbitals and conformer in boltzmann_weights:
                weight = boltzmann_weights[conformer]
                weighted_homo += neutral_orbitals[conformer]['HOMO'] * weight
                weighted_lumo += neutral_orbitals[conformer]['LUMO'] * weight
                total_weight += weight

        # S'assurer que nous utilisons des moyennes valides
        if total_weight > 0:
            weighted_gap = weighted_lumo - weighted_homo
        else:
            # Si aucun poids valide n'a été trouvé
            weighted_homo = 0.0
            weighted_lumo = 0.0
            weighted_gap = 0.0
            print("AVERTISSEMENT: Aucune donnée d'orbitales valide n'a été trouvée pour calculer les moyennes.")

        # Afficher les résultats - Réduction
        print("\n=== POTENTIELS DE RÉDUCTION ===")
        print(f"E(∆G)   = {potentials['reduction']['delta_G']:.3f} V")
        print(f"E(EA)   = {potentials['reduction']['EA']:.3f} V")
        print(f"E(Edef) = {potentials['reduction']['Edef']:.3f} V")
        print(f"E(∆∆U)  = {potentials['reduction']['delta_delta_U']:.3f} V")
        print(f"E(T∆S)  = {potentials['reduction']['T_delta_S']:.3f} V")

        # Vérification de la cohérence thermodynamique - Réduction
        sum_contributions = (potentials['reduction']['EA'] +
                            potentials['reduction']['Edef'] +
                            potentials['reduction']['delta_delta_U'] +
                            potentials['reduction']['T_delta_S'])

        print("\nSomme des contributions (réduction):")
        print(f"E(EA) + E(Edef) + E(∆∆U) + E(T∆S) = {sum_contributions:.3f} V")
        print(f"E(∆G) = {potentials['reduction']['delta_G']:.3f} V")
        print(f"Écart = {potentials['reduction']['delta_G'] - sum_contributions:.3f} V")

        # Afficher les résultats - Oxydation
        print("\n=== POTENTIELS D'OXYDATION ===")
        print(f"E(∆G)   = {potentials['oxidation']['delta_G']:.3f} V")
        print(f"E(EI)   = {potentials['oxidation']['EI']:.3f} V")
        print(f"E(Edef) = {potentials['oxidation']['Edef']:.3f} V")
        print(f"E(∆∆U)  = {potentials['oxidation']['delta_delta_U']:.3f} V")
        print(f"E(T∆S)  = {potentials['oxidation']['T_delta_S']:.3f} V")

        # Vérification de la cohérence thermodynamique - Oxydation
        sum_contributions_ox = (potentials['oxidation']['EI'] +
                              potentials['oxidation']['Edef'] +
                              potentials['oxidation']['delta_delta_U'] +
                              potentials['oxidation']['T_delta_S'])

        print("\nSomme des contributions (oxydation):")
        print(f"E(EI) + E(Edef) + E(∆∆U) + E(T∆S) = {sum_contributions_ox:.3f} V")
        print(f"E(∆G) = {potentials['oxidation']['delta_G']:.3f} V")
        print(f"Écart = {potentials['oxidation']['delta_G'] - sum_contributions_ox:.3f} V")

        # Afficher les données des orbitales
        print("\n=== ORBITALES MOLÉCULAIRES MOYENNES PONDÉRÉES ===")
        print(f"HOMO moyenne: {weighted_homo:.4f} eV")
        print(f"LUMO moyenne: {weighted_lumo:.4f} eV")
        print(f"Gap HOMO-LUMO moyen: {weighted_gap:.4f} eV")

        # Enregistrer les résultats dans un fichier
        try:
            # Déterminer le chemin du dossier results
            from scm.plams import config
            # workdir = config.default_jobmanager.workdir
            results_dir = os.path.join(workdir, 'redox_results')

            # Créer le dossier s'il n'existe pas
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Nom du fichier d'output
            output_file = os.path.join(results_dir, f"redox_potentials.txt")

            with open(output_file, 'w') as f:
                # Écrire l'en-tête
                f.write("POTENTIELS REDOX CALCULÉS\n")
                f.write("=" * 50 + "\n\n")

                # Informations générales
                f.write(f"Température: {temperature} K\n")
                f.write(f"Nombre de conformères: {len(complete_conformers)}\n\n")

                # === SECTION RÉDUCTION ===
                f.write("POTENTIELS DE RÉDUCTION (V vs. référence):\n")
                f.write("-" * 50 + "\n")
                f.write(f"E(∆G) = {potentials['reduction']['delta_G']:.4f} V\n")
                f.write("\nContributions:\n")
                f.write(f"E(EA) = {potentials['reduction']['EA']:.4f} V\n")
                f.write(f"E(Edef) = {potentials['reduction']['Edef']:.4f} V\n")
                f.write(f"E(∆∆U) = {potentials['reduction']['delta_delta_U']:.4f} V\n")
                f.write(f"E(T∆S) = {potentials['reduction']['T_delta_S']:.4f} V\n")
                f.write(f"Somme: {sum_contributions:.4f} V\n")
                f.write(f"Écart: {potentials['reduction']['delta_G'] - sum_contributions:.4f} V\n\n")

                # === SECTION OXYDATION ===
                f.write("POTENTIELS D'OXYDATION (V vs. référence):\n")
                f.write("-" * 50 + "\n")
                f.write(f"E(∆G) = {potentials['oxidation']['delta_G']:.4f} V\n")
                f.write("\nContributions:\n")
                f.write(f"E(EI) = {potentials['oxidation']['EI']:.4f} V\n")
                f.write(f"E(Edef) = {potentials['oxidation']['Edef']:.4f} V\n")
                f.write(f"E(∆∆U) = {potentials['oxidation']['delta_delta_U']:.4f} V\n")
                f.write(f"E(T∆S) = {potentials['oxidation']['T_delta_S']:.4f} V\n")
                f.write(f"Somme: {sum_contributions_ox:.4f} V\n")
                f.write(f"Écart: {potentials['oxidation']['delta_G'] - sum_contributions_ox:.4f} V\n\n")

                # === DÉTAILS PAR CONFORMÈRE ===
                f.write("DÉTAILS PAR CONFORMÈRE:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Conformère':<15} {'Poids':<10} {'E_red(∆G) [V]':<15} {'E_ox(∆G) [V]':<15}\n")
                f.write("-" * 60 + "\n")

                for conformer in sorted(complete_conformers):
                    weight = boltzmann_weights[conformer]
                    e_red = -redox_parameters[conformer]['reduction']['delta_G']/F
                    e_ox = redox_parameters[conformer]['oxidation']['delta_G']/F
                    f.write(f"{conformer:<16} {weight:.4f} {e_red:13.4f} {e_ox:15.4f}\n")

                # === ORBITALES MOLÉCULAIRES ===
                f.write("\n\n" + "="*50 + "\n")
                f.write("ÉNERGIES DES ORBITALES MOLÉCULAIRES\n")
                f.write("="*50 + "\n\n")

                # Moyennes pondérées
                f.write("Moyennes pondérées:\n")
                f.write(f"HOMO: {weighted_homo:.4f} eV\n")
                f.write(f"LUMO: {weighted_lumo:.4f} eV\n")
                f.write(f"Gap: {weighted_gap:.4f} eV\n\n")

                # Par conformère
                f.write(f"{'Conformère':<15} {'HOMO (eV)':<15} {'LUMO (eV)':<15} {'Gap (eV)':<15}\n")
                f.write("-" * 60 + "\n")

                for conformer in sorted(complete_conformers):
                    if conformer in neutral_orbitals:
                        homo = neutral_orbitals[conformer]['HOMO']
                        lumo = neutral_orbitals[conformer]['LUMO']
                        gap = neutral_orbitals[conformer]['Gap']
                        f.write(f"{conformer:<15} {homo:.4f} {lumo:10.4f} {gap:8.4f}\n")
                    else:
                        f.write(f"{conformer:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}\n")

                # === SECTION RMSD ===
                if rmsd_export_data:
                    f.write("\n\n" + "="*50 + "\n")
                    f.write("ANALYSE STRUCTURELLE PAR RMSD\n")
                    f.write("="*50 + "\n\n")
    
                    # Afficher les résultats des conformères réduits
                    if 'summary_reduced' in rmsd_export_data and rmsd_export_data['summary_reduced']:
                        f.write("CONFORMÈRES RÉDUITS\n")
                        f.write(f"{'Conformère réduit':<25} {'Conformère neutre':<25} {'RMSD (Å)':<10}\n")
                        f.write("-"*65 + "\n")
        
                        for reduced_name, neutral_name, rmsd_value in rmsd_export_data['summary_reduced']:
                            f.write(f"{reduced_name:<25} {neutral_name:<25} {rmsd_value:.4f}\n")
        
                        f.write("\n")
    
                    # Afficher les résultats des conformères oxidés
                    if 'summary_oxidized' in rmsd_export_data and rmsd_export_data['summary_oxidized']:
                        f.write("CONFORMÈRES OXIDÉS\n")
                        f.write(f"{'Conformère oxidé':<25} {'Conformère neutre':<25} {'RMSD (Å)':<10}\n")
                        f.write("-"*65 + "\n")
        
                        for oxidized_name, neutral_name, rmsd_value in rmsd_export_data['summary_oxidized']:
                            f.write(f"{oxidized_name:<25} {neutral_name:<25} {rmsd_value:.4f}\n")
        
                        f.write("\n")
    
                    # Afficher les avertissements
                    if 'warnings' in rmsd_export_data and rmsd_export_data['warnings']:
                        f.write("\nAVERTISSEMENTS DE CHANGEMENT CONFORMATIONNEL:\n")
                        f.write("-"*50 + "\n")
                        for warning in rmsd_export_data['warnings']:
                            f.write(f"- {warning}\n")

            print(f"\nRésultats enregistrés dans {output_file}")

        except Exception as e:
            print(f"\nErreur lors de l'enregistrement des résultats: {str(e)}")
            import traceback
            traceback.print_exc()

        return avg_params, potentials, redox_parameters, boltzmann_weights

    except Exception as e:
        print(f"ERREUR CRITIQUE: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, {}, {}, {}

def main():
    """Fonction principale du programme."""
    # Parse command line arguments
    args = parse_arguments()

    # Si on est en mode analysis-only, on saute toute la partie calculs
    if args.analysis_only:
        if not args.workdir:
            print("ERREUR: en mode --analysis-only, il faut fournir --workdir")
            return

        print(">>> Mode ANALYSIS-ONLY: extraction des énergies et calculs redox")
        # on part du parent du workdir pour extract_engine_energies
        job_results = load_existing_calculations(args.workdir)
        rmsd_results, rmsd_export_data = compare_conformers_rmsd(job_results)
        energy_data = extract_engine_energies(args.workdir)
        workdir = os.path.join(args.workdir, "redox")
        analyze_redox_energies(energy_data, workdir=workdir, temperature=298.15, rmsd_export_data=rmsd_export_data)
        return

    # Initialize PLAMS with workspace setup
    workdir = setup_workspace(args.input_dir)

    # Affiche les paramètres de calcul
    print("\n" + "="*50)
    print(f"PARAMÈTRES DE CALCUL:")
    print(f"Fonctionnelle: {args.functional}")
    print(f"Base: {args.basis}")
    print(f"Solvant: {args.solvent}")
    print("="*50 + "\n")

    # Find all XYZ files in input directory
    xyz_files = glob.glob(os.path.join(args.input_dir, "*.xyz"))
    if not xyz_files:
        print(f"Aucun fichier XYZ trouvé dans {args.input_dir}")
        finish()
        return

    print(f"Trouvé {len(xyz_files)} fichiers XYZ à traiter")

    # Store all job results
    job_results = {}

    # Process each XYZ file
    for xyz_file in xyz_files:
        basename = os.path.basename(xyz_file)
        name = os.path.splitext(basename)[0]  # Remove extension

        print(f"\nTraitement de {basename}")

        # Read molecule from XYZ
        mol = Molecule(xyz_file)
        job_results[name] = {'neutre_opt': None, 'réduit_sp': None, 'réduit_opt': None, 'oxidé_sp': None, 'oxidé_opt': None}

        try:
            # Step 1: Optimize neutral
            job_neutral = optimize_neutral(mol, name, args.solvent, args.functional, args.basis)
            job_results[name]['neutre_opt'] = job_neutral

            # Step 2: Single point reduced
            job_reduced_sp = sp_reduced(job_neutral, name, args.solvent, args.functional, args.basis)
            job_results[name]['réduit_sp'] = job_reduced_sp

            # Step 3: Optimize reduced
            job_reduced_opt = optimize_reduced(job_neutral, name, args.solvent, args.functional, args.basis)
            job_results[name]['réduit_opt'] = job_reduced_opt

            # Step 4: Single point oxidized
            job_oxidized_sp = sp_oxidized(job_neutral, name, args.solvent, args.functional, args.basis)
            job_results[name]['oxidé_sp'] = job_oxidized_sp

            # Step 5: Optimize oxidized
            job_oxidized_opt = optimize_oxidized(job_neutral, name, args.solvent, args.functional, args.basis)
            job_results[name]['oxidé_opt'] = job_oxidized_opt

        except Exception as e:
            print(f"ERREUR lors du traitement de {basename}: {str(e)}")
            import traceback
            traceback.print_exc()  # Affiche la trace complète pour faciliter le débogage
            continue

    # Analyse RMSD
    rmsd_results, rmsd_export_data = compare_conformers_rmsd(job_results)

    # Export output files
    export_molecules(job_results)

    # Extraire et afficher les énergies
    parent_dir = os.path.dirname(config.default_jobmanager.workdir)
    energy_data = extract_engine_energies(parent_dir)

    # Analyser les énergies redox
    analyze_redox_energies(energy_data, workdir=workdir, temperature=298.15, rmsd_export_data=rmsd_export_data)
    # Finalize PLAMS
    finish()


if __name__ == "__main__":
    main()
