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
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Script pour calculs redox de conformères"
    )
    parser.add_argument("input_dir", help="Dossier contenant les fichiers .xyz des conformères")
    parser.add_argument("--solvent", default="Acetonitrile", help="Solvant pour les calculs")
    
    # Ajout des paramètres pour le niveau de théorie
    parser.add_argument("--functional", "-f", default="PBE0", 
                        help="Fonctionnelle à utiliser pour les calculs (défaut: PBE0)")
    
    # Ajout des paramètres pour le basis set
    parser.add_argument("--basis", "-b", default="DZP", 
                        help="Basis set à utiliser pour les calculs (défaut: DZP)")
    
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

def optimize_neutral(mol, name, solvent="Acetonitrile", functional="PBE0", basis="DZP"):
    """
    Étape 1: Optimisation géométrique de la molécule neutre
    
    Args:
        mol (Molecule): Molécule à optimiser
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        
    Returns:
        AMSJob: Job d'optimisation terminé ou None en cas d'échec
    """
    print(f"\nÉtape 1: Optimisation géométrique de {name} (neutre)")
    settings = setup_adf_settings(task="GeometryOptimization", charge=0, 
                                 solvent=solvent, functional=functional, basis=basis)
    
    job = AMSJob(settings=settings, name=f"{name}_neutre_opt", molecule=mol)
    job.run()
    
    if job.check():
        print(f"  Optimisation neutre réussie pour {name}")
        return job
    else:
        print(f"  ERREUR: Optimisation neutre échouée pour {name}")
        return None

def sp_reduced(job_neutral, name, solvent="Acetonitrile", functional="PBE0", basis="DZP"):
    """
    Étape 2: Calcul en simple point de la molécule réduite (charge -1)
    
    Args:
        job_neutral (AMSJob): Job d'optimisation de la molécule neutre
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        
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

def optimize_reduced(job_sp, name, solvent="Acetonitrile", functional="PBE0", basis="DZP"):
    """
    Étape 3: Optimisation géométrique de la molécule réduite (charge -1)
    
    Args:
        job_sp (AMSJob): Job de single point de la molécule réduite
        name (str): Nom de base pour le job
        solvent (str): Solvant à utiliser
        functional (str): Fonctionnelle DFT
        basis (str): Base à utiliser
        
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
        return job
    else:
        print(f"  ERREUR: Optimisation réduite échouée pour {name}")
        return None

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
    rmsd_summary = []  # Pour stocker les résultats à exporter
    rmsd_warnings = []  # Pour stocker les avertissements

    # Liste des noms de tous les conformères
    conformer_names = list(job_results.keys())

    print("\n" + "="*80)
    print("ANALYSE STRUCTURELLE PAR RMSD (ALGORITHME DE KABSCH)")
    print("="*80)

    # Pour chaque conformère réduit
    for reduced_name in conformer_names:
        reduced_job = job_results[reduced_name]['opt']
        if not reduced_job or not reduced_job.ok():
            continue

        reduced_mol = reduced_job.results.get_main_molecule()
        rmsd_results[reduced_name] = {}

        # Comparer avec tous les neutres
        min_rmsd_value = float('inf')
        min_rmsd_name = None

        for neutral_name in conformer_names:
            neutral_job = job_results[neutral_name]['neutral']
            if not neutral_job or not neutral_job.ok():
                continue

            neutral_mol = neutral_job.results.get_main_molecule()

            # Calculer le RMSD
            try:
                rmsd = kabsch_rmsd(reduced_mol, neutral_mol)
                rmsd_results[reduced_name][neutral_name] = rmsd

                # Garder trace du RMSD minimum
                if rmsd < min_rmsd_value:
                    min_rmsd_value = rmsd
                    min_rmsd_name = neutral_name

            except Exception as e:
                print(f"Erreur lors du calcul RMSD entre {reduced_name} et {neutral_name}: {str(e)}")
                rmsd_results[reduced_name][neutral_name] = float('inf')

        # Ajouter le meilleur résultat au résumé
        if min_rmsd_name:
            rmsd_summary.append((reduced_name, min_rmsd_name, min_rmsd_value))

            # Déterminer si c'est un changement de conformère
            if reduced_name != min_rmsd_name:
                warning = f"ATTENTION: Le conformère réduit {reduced_name} correspond mieux au conformère neutre {min_rmsd_name} (RMSD: {min_rmsd_value:.4f} Å)"
                rmsd_warnings.append(warning)
                print(warning)
            else:
                # Vérifier si le RMSD est élevé même pour le même conformère
                if min_rmsd_value > 0.3:  # Seuil arbitraire, à ajuster selon vos molécules
                    warning = f"ATTENTION: Le conformère réduit {reduced_name} présente un RMSD élevé ({min_rmsd_value:.4f} Å) par rapport à son conformère neutre"
                    rmsd_warnings.append(warning)
                    print(warning)
                else:
                    msg = f"Le conformère réduit {reduced_name} correspond bien à son conformère neutre (RMSD: {min_rmsd_value:.4f} Å)"
                    rmsd_warnings.append(msg)
                    print(msg)

    # Afficher uniquement le RMSD minimum pour chaque conformère
    print("\n" + "="*80)
    print("RÉSUMÉ DE L'ANALYSE RMSD")
    print("="*80)
    print(f"{'Conformère réduit':<25} {'Conformère neutre le plus proche':<30} {'RMSD (Å)':<10}")
    print("-"*70)

    for reduced_name, neutral_name, rmsd_value in rmsd_summary:
        print(f"{reduced_name:<25} {neutral_name:<30} {rmsd_value:.4f}")

    # Créer le dictionnaire d'export
    rmsd_export_data = {
        'summary': rmsd_summary,
        'warnings': rmsd_warnings
    }

    return rmsd_results, rmsd_export_data

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

            # Traiter chaque type de job (neutral, sp, opt)
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
    et les affiche.

    Args:
        workdir_parent (str): Chemin du dossier parent où se trouvent les dossiers redox*

    Returns:
        dict: Dictionnaire contenant les énergies extraites par fichier
    """
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
        'HOMO': r"HOMO \(eV\):\s+([-\d.]+)",
        'LUMO': r"LUMO \(eV\):\s+([-\d.]+)"
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
        print(f"{'Fichier':<30} {'Engine Energy':<15} {'Internal Energy U':<15} {'-T*S':<15} {'Gibbs Energy':<15} {'HOMO (eV)':<15} {'LUMO (eV)':<15}")
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

def analyze_redox_energies(energy_data, temperature=298.15, rmsd_export_data=None):
    """
    Analyse les valeurs d'énergie extraites et calcule les paramètres redox.

    Args:
        energy_data (dict): Dictionnaire contenant les énergies extraites par fichier
        temperature (float): Température en K (par défaut: 298.15K)
        rmsd_export_data (dict, optional): Données RMSD à inclure dans le fichier d'export
    """
    try:
        import math
        import re
        import os
        from scm.plams import config

        # Constantes
        R = 8.314462618e-3  # Constante des gaz parfaits en kJ/(mol·K)
        RT = R * temperature  # kJ/mol à 298.15K ≈ 2.48 kJ/mol
        F = 96.485  # Constante de Faraday en kJ/(mol·V)

        print("\n" + "="*80)
        print("ANALYSE ÉNERGÉTIQUE DU PROCESSUS REDOX")
        print("="*80)

        # Vérifier si energy_data est vide
        if not energy_data:
            print("ERREUR: Aucune donnée d'énergie fournie (energy_data est vide)")
            return {}, {}, {}, {}

        # Initialiser les structures de données résultantes
        avg_params = {}
        potentials = {}
        redox_parameters = {}
        boltzmann_weights = {}

        # Organiser les données par conformères et type de calcul
        try:
            conformers_data = {}
            pattern = r'(.+?)_(neutral|sp|opt)\.out'

            for filename, energies in energy_data.items():
                try:
                    match = re.search(pattern, filename)
                    if not match:
                        print(f"AVERTISSEMENT: Le fichier {filename} ne correspond pas au format attendu")
                        continue

                    conformer = match.group(1)
                    calc_type = match.group(2)

                    if conformer not in conformers_data:
                        conformers_data[conformer] = {}

                    conformers_data[conformer][calc_type] = energies
                except Exception as e:
                    print(f"ERREUR lors du traitement du fichier {filename}: {str(e)}")

            if not conformers_data:
                print("ERREUR: Aucun fichier n'a pu être correctement analysé")
                return {}, {}, {}, {}
        except Exception as e:
            print(f"ERREUR lors de l'organisation des données: {str(e)}")
            return {}, {}, {}, {}

        # Vérifier quels conformères ont tous les calculs nécessaires
        try:
            complete_conformers = []
            for conformer, calcs in conformers_data.items():
                if all(calc_type in calcs for calc_type in ['neutral', 'sp', 'opt']):
                    try:
                        all_energies_available = True
                        for calc_type in ['neutral', 'sp', 'opt']:
                            if (calcs[calc_type]['Ee'] is None or calcs[calc_type]['U'] is None or
                                calcs[calc_type]['TS'] is None or calcs[calc_type]['G'] is None):
                                all_energies_available = False
                                missing = []
                                if calcs[calc_type]['Ee'] is None: missing.append('Ee')
                                if calcs[calc_type]['U'] is None: missing.append('U')
                                if calcs[calc_type]['TS'] is None: missing.append('TS')
                                if calcs[calc_type]['G'] is None: missing.append('G')
                                print(f"AVERTISSEMENT: Énergies manquantes pour {conformer}_{calc_type}: {', '.join(missing)}")
                                break

                        if all_energies_available:
                            complete_conformers.append(conformer)
                    except KeyError as e:
                        print(f"ERREUR: Clé manquante pour {conformer}: {str(e)}")
                else:
                    missing = [t for t in ['neutral', 'sp', 'opt'] if t not in calcs]
                    print(f"AVERTISSEMENT: Types de calcul manquants pour {conformer}: {', '.join(missing)}")

            if not complete_conformers:
                print("ERREUR: Aucun conformère n'a tous les calculs nécessaires avec toutes les énergies")
                return {}, {}, {}, {}

            print(f"Nombre de conformères complets: {len(complete_conformers)}")
            print(f"Conformères à analyser: {', '.join(complete_conformers)}")
        except Exception as e:
            print(f"ERREUR lors de la vérification des conformères: {str(e)}")
            return {}, {}, {}, {}

        # Calculer Delta U pour chaque cas
        try:
            for conformer in complete_conformers:
                for calc_type in ['neutral', 'sp', 'opt']:
                    try:
                        energies = conformers_data[conformer][calc_type]
                        if energies['Ee'] is not None and energies['U'] is not None:
                            energies['delta_U'] = energies['U'] - energies['Ee']
                        else:
                            energies['delta_U'] = 0.0
                            print(f"AVERTISSEMENT: Impossible de calculer delta_U pour {conformer}_{calc_type}")
                    except Exception as e:
                        print(f"ERREUR lors du calcul de Delta U pour {conformer}_{calc_type}: {str(e)}")
                        energies['delta_U'] = 0.0
        except Exception as e:
            print(f"ERREUR lors du calcul des Delta U: {str(e)}")

        # Calculer les paramètres redox pour chaque conformère
        try:
            for conformer in complete_conformers:
                try:
                    neutral_data = conformers_data[conformer]['neutral']
                    sp_data = conformers_data[conformer]['sp']
                    opt_data = conformers_data[conformer]['opt']

                    try:
                        delta_G = opt_data['G'] - neutral_data['G']
                    except (TypeError, KeyError):
                        delta_G = 0.0
                        print(f"ERREUR: Impossible de calculer delta_G pour {conformer}")

                    try:
                        EA = sp_data['Ee'] - neutral_data['Ee']
                    except (TypeError, KeyError):
                        EA = 0.0
                        print(f"ERREUR: Impossible de calculer EA pour {conformer}")

                    try:
                        Edef = opt_data['Ee'] - sp_data['Ee']
                    except (TypeError, KeyError):
                        Edef = 0.0
                        print(f"ERREUR: Impossible de calculer Edef pour {conformer}")

                    try:
                        delta_delta_U = opt_data['delta_U'] - neutral_data['delta_U']
                    except (TypeError, KeyError):
                        delta_delta_U = 0.0
                        print(f"ERREUR: Impossible de calculer delta_delta_U pour {conformer}")

                    try:
                        T_delta_S = opt_data['TS'] - neutral_data['TS']
                    except (TypeError, KeyError):
                        T_delta_S = 0.0
                        print(f"ERREUR: Impossible de calculer T_delta_S pour {conformer}")

                    redox_parameters[conformer] = {
                        'delta_G': delta_G,
                        'EA': EA,
                        'Edef': Edef,
                        'delta_delta_U': delta_delta_U,
                        'T_delta_S': T_delta_S
                    }
                except Exception as e:
                    print(f"ERREUR lors du calcul des paramètres redox pour {conformer}: {str(e)}")

            if not redox_parameters:
                print("ERREUR: Aucun paramètre redox n'a pu être calculé")
                return {}, {}, {}, {}
        except Exception as e:
            print(f"ERREUR lors du calcul des paramètres redox: {str(e)}")
            return {}, {}, {}, {}

        # Calculer les poids de Boltzmann en fonction des énergies de Gibbs des conformères neutres
        try:
            try:
                min_G = min([conformers_data[conf]['neutral']['G']
                             for conf in complete_conformers
                             if conformers_data[conf]['neutral']['G'] is not None])
            except ValueError:
                print("AVERTISSEMENT: Aucune valeur G valide trouvée pour le calcul de Boltzmann")
                # Attribuer un poids égal à tous les conformères
                for conformer in complete_conformers:
                    boltzmann_weights[conformer] = 1.0 / len(complete_conformers)
                return avg_params, potentials, redox_parameters, boltzmann_weights

            denominator = 0.0
            for conformer in complete_conformers:
                try:
                    G = conformers_data[conformer]['neutral']['G']
                    if G is not None:
                        rel_G = G - min_G
                        try:
                            boltzmann_factor = math.exp(-rel_G/(R*temperature))
                            boltzmann_weights[conformer] = boltzmann_factor
                            denominator += boltzmann_factor
                        except OverflowError:
                            print(f"AVERTISSEMENT: Overflow pour {conformer}, rel_G = {rel_G}")
                            boltzmann_weights[conformer] = 0.0
                    else:
                        print(f"AVERTISSEMENT: G est None pour {conformer}, attribution d'un poids nul")
                        boltzmann_weights[conformer] = 0.0
                except Exception as e:
                    print(f"ERREUR lors du calcul du facteur de Boltzmann pour {conformer}: {str(e)}")
                    boltzmann_weights[conformer] = 0.0

            # Vérifier si le dénominateur est non nul
            if denominator <= 0:
                print("AVERTISSEMENT: La somme des facteurs de Boltzmann est nulle ou négative")
                # Attribuer un poids égal à tous les conformères
                for conformer in complete_conformers:
                    boltzmann_weights[conformer] = 1.0 / len(complete_conformers)
            else:
                # Normaliser les poids
                for conformer in complete_conformers:
                    boltzmann_weights[conformer] /= denominator
        except Exception as e:
            print(f"ERREUR lors du calcul des poids de Boltzmann: {str(e)}")
            # Attribuer un poids égal à tous les conformères
            for conformer in complete_conformers:
                boltzmann_weights[conformer] = 1.0 / len(complete_conformers)

        # Calculer les moyennes pondérées
        try:
            avg_params = {
                'delta_G': 0.0,
                'EA': 0.0,
                'Edef': 0.0,
                'delta_delta_U': 0.0,
                'T_delta_S': 0.0
            }

            for conformer in complete_conformers:
                weight = boltzmann_weights[conformer]
                for param in avg_params:
                    try:
                        avg_params[param] += weight * redox_parameters[conformer][param]
                    except Exception as e:
                        print(f"ERREUR lors du calcul de la moyenne pour {param} de {conformer}: {str(e)}")

            # Vérifier l'égalité thermodynamique
            try:
                equality_check = avg_params['EA'] + avg_params['Edef'] + avg_params['delta_delta_U'] + avg_params['T_delta_S']
                equality_diff = avg_params['delta_G'] - equality_check
            except Exception as e:
                print(f"ERREUR lors de la vérification thermodynamique: {str(e)}")
                equality_check = 0.0
                equality_diff = 0.0
        except Exception as e:
            print(f"ERREUR lors du calcul des moyennes: {str(e)}")
            return {}, {}, {}, {}

        # Calculer les potentiels de réduction (V)
        try:
            potentials = {}
            for param, value in avg_params.items():
                try:
                    key = f"E_{param}" if param != 'T_delta_S' else "E_T_delta_S"
                    potentials[key] = -value/F
                except Exception as e:
                    print(f"ERREUR lors du calcul du potentiel pour {param}: {str(e)}")
                    potentials[f"E_{param}"] = 0.0
        except Exception as e:
            print(f"ERREUR lors du calcul des potentiels: {str(e)}")
            return avg_params, {}, redox_parameters, boltzmann_weights

        neutral_orbitals = {}  # Dictionnaire pour stocker les valeurs HOMO/LUMO des neutres optimisés

        for filename, energies in energy_data.items():
            match = re.search(pattern, filename)
            if match and match.group(2) == 'neutral':
                conformer = match.group(1)
                if 'HOMO' in energies and 'LUMO' in energies:
                    neutral_orbitals[conformer] = {
                        'HOMO': energies['HOMO'],
                        'LUMO': energies['LUMO']
                    }

        # Calculer les moyennes pondérées des HOMO et LUMO
        weighted_homo = 0.0
        weighted_lumo = 0.0
        homo_lumo_available = False

        for conformer in complete_conformers:
            if conformer in neutral_orbitals:
                weight = boltzmann_weights.get(conformer, 0.0)
                if 'HOMO' in neutral_orbitals[conformer] and neutral_orbitals[conformer]['HOMO'] is not None:
                    weighted_homo += neutral_orbitals[conformer]['HOMO'] * weight
                    homo_lumo_available = True
                if 'LUMO' in neutral_orbitals[conformer] and neutral_orbitals[conformer]['LUMO'] is not None:
                    weighted_lumo += neutral_orbitals[conformer]['LUMO'] * weight
                    homo_lumo_available = True

        # Calculer le gap HOMO-LUMO moyen pondéré
        weighted_gap = weighted_lumo - weighted_homo if homo_lumo_available else None

        try:
            # Vérifier l'égalité thermodynamique
            print(f"\nVérification de l'égalité thermodynamique:")
            print(f"∆G = {avg_params['delta_G']:.2f} kJ/mol")
            print(f"EA + Edef + ∆∆U - T∆S = {equality_check:.2f} kJ/mol")
            print(f"Différence = {equality_diff:.2f} kJ/mol")

            # Afficher les potentiels de réduction
            print("\nPotentiels de réduction:")
            print(f"E(∆G)   = {potentials.get('E_delta_G', 0):.3f} V")
            print(f"E(EA)   = {potentials.get('E_EA', 0):.3f} V")
            print(f"E(Edef) = {potentials.get('E_Edef', 0):.3f} V")
            print(f"E(∆∆U)  = {potentials.get('E_delta_delta_U', 0):.3f} V")
            print(f"E(T∆S)  = {potentials.get('E_T_delta_S', 0):.3f} V")

            sum_potentials = (potentials.get('E_EA', 0) + potentials.get('E_Edef', 0) +
                              potentials.get('E_delta_delta_U', 0) + potentials.get('E_T_delta_S', 0))

            print("\nSomme des contributions:")
            print(f"E(EA) + E(Edef) + E(∆∆U) - E(T∆S) = {sum_potentials:.3f} V")
            print(f"E(∆G) = {potentials.get('E_delta_G', 0):.3f} V")

            # Afficher les moyennes pondérées des HOMO et LUMO
            if homo_lumo_available:
                print("\nValeurs moyennes pondérées des orbitales:")
                print(f"HOMO moyenne: {weighted_homo:.4f} eV")
                print(f"LUMO moyenne: {weighted_lumo:.4f} eV")
                print(f"Gap HOMO-LUMO moyen: {weighted_gap:.4f} eV")
            else:
                print("\nImpossible de calculer les moyennes pondérées des HOMO/LUMO: données insuffisantes")

        except Exception as e:
            print(f"ERREUR lors de l'affichage des moyennes et potentiels: {str(e)}")

        print("="*80)

        # Enregistrer les résultats dans un fichier texte
        try:
            # Déterminer le chemin du dossier results
            try:
                workdir = config.default_jobmanager.workdir
                results_dir = os.path.join(workdir, 'redox_results')

                # Créer le dossier s'il n'existe pas
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
            except Exception as e:
                print(f"ERREUR lors de la création du dossier de résultats: {str(e)}")
                results_dir = "."  # Utiliser le répertoire courant en cas d'erreur

            # Nom du fichier d'output
            output_file = os.path.join(results_dir, f"redox_potentials.txt")

            with open(output_file, 'w') as f:
                f.write("POTENTIELS DE RÉDUCTION\n")
                f.write("=" * 50 + "\n\n")

                # Informations générales
                f.write(f"Température: {temperature} K\n")
                f.write(f"RT: {RT:.4f} kJ/mol\n")
                f.write(f"Nombre de conformères: {len(complete_conformers)}\n\n")

                # Paramètres de calcul
                try:
                    # S'il y a une variable globale ou un paramètre de configuration, l'ajouter ici
                    f.write(f"Commentaires de calcul: Analyse effectuée avec succès\n\n")
                except Exception:
                    pass

                # Potentiels moyens
                f.write("POTENTIELS MOYENS (V vs. référence):\n")
                f.write("-" * 50 + "\n")
                f.write(f"E(∆G) = {potentials.get('E_delta_G', 0):.4f} V\n")
                f.write("\nContributions:\n")
                f.write(f"E(EA) = {potentials.get('E_EA', 0):.4f} V\n")
                f.write(f"E(Edef) = {potentials.get('E_Edef', 0):.4f} V\n")
                f.write(f"E(∆∆U) = {potentials.get('E_delta_delta_U', 0):.4f} V\n")
                f.write(f"E(T∆S) = {potentials.get('E_T_delta_S', 0):.4f} V\n")

                # Calcul de la somme
                try:
                    required_keys = ['E_EA', 'E_Edef', 'E_delta_delta_U', 'E_T_delta_S']
                    if all(k in potentials for k in required_keys):
                        sum_potentials = (potentials['E_EA'] + potentials['E_Edef'] +
                                         potentials['E_delta_delta_U'] + potentials['E_T_delta_S'])
                        f.write(f"Somme: {sum_potentials:.4f} V\n\n")
                    else:
                        missing_keys = [k for k in required_keys if k not in potentials]
                        f.write(f"Somme: N/A (données incomplètes - manquants: {missing_keys})\n\n")
                except Exception as e:
                    f.write(f"Somme: N/A (erreur: {str(e)})\n\n")

                # Données par conformère
                f.write("DÉTAILS PAR CONFORMÈRE:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Conformère':<15} {'Poids':<10} {'E(∆G) [V]':<12}\n")
                f.write("-" * 40 + "\n")

                for conformer in sorted(complete_conformers):
                    try:
                        weight = boltzmann_weights.get(conformer, 0)
                        if conformer in redox_parameters and 'delta_G' in redox_parameters[conformer]:
                            e_value = -redox_parameters[conformer]['delta_G']/F
                        else:
                            e_value = 0
                        f.write(f"{conformer:<15} {weight:.4f} {e_value:12.4f}\n")
                    except Exception as e:
                        f.write(f"{conformer:<15} ERROR: {str(e)}\n")

                # Ajouter les moyennes pondérées des HOMO/LUMO
                f.write("\n\n" + "="*50 + "\n")
                f.write("ÉNERGIES DES ORBITALES MOLÉCULAIRES MOYENNES PONDÉRÉES\n")
                f.write("="*50 + "\n\n")

                if homo_lumo_available:
                    f.write(f"HOMO moyenne pondérée: {weighted_homo:.4f} eV\n")
                    f.write(f"LUMO moyenne pondérée: {weighted_lumo:.4f} eV\n")
                    f.write(f"Gap HOMO-LUMO moyen pondéré: {weighted_gap:.4f} eV\n")
                else:
                    f.write("Impossible de calculer les moyennes pondérées des orbitales: données insuffisantes\n")

                # Ajouter les résultats RMSD si disponibles
                if rmsd_export_data and 'summary' in rmsd_export_data and 'warnings' in rmsd_export_data:
                    f.write("\n\n" + "="*50 + "\n")
                    f.write("ANALYSE STRUCTURELLE PAR RMSD (ALGORITHME DE KABSCH)\n")
                    f.write("="*50 + "\n\n")

                    f.write(f"{'Conformère réduit':<25} {'Conformère neutre le plus proche':<30} {'RMSD (Å)':<10}\n")
                    f.write("-"*70 + "\n")

                    for reduced_name, neutral_name, rmsd_value in rmsd_export_data['summary']:
                        f.write(f"{reduced_name:<25} {neutral_name:<30} {rmsd_value:.4f}\n")

                    # Ajouter les avertissements de changement de conformère
                    f.write("\n\n" + "="*50 + "\n")
                    f.write("IDENTIFICATION DES POSSIBLE CHANGEMENTS DE CONFORMÈRE\n")
                    f.write("="*50 + "\n\n")

                    for warning in rmsd_export_data['warnings']:
                        f.write(warning + "\n\n")

                # Ajouter une section pour les niveaux HOMO/LUMO
                f.write("\n\n" + "="*50 + "\n")
                f.write("NIVEAUX ÉNERGÉTIQUES DES ORBITALES MOLÉCULAIRES PAR CONFORMÈRE\n")
                f.write("="*50 + "\n\n")

                # En-têtes du tableau
                f.write(f"{'Conformère':<15} {'HOMO (eV)':<15} {'LUMO (eV)':<15} {'Gap (eV)':<15}\n")
                f.write("-" * 60 + "\n")

                # Écrire les valeurs pour chaque conformère
                for conformer in sorted(complete_conformers):
                    if conformer in neutral_orbitals:
                        homo = neutral_orbitals[conformer].get('HOMO')
                        lumo = neutral_orbitals[conformer].get('LUMO')

                        homo_str = f"{homo:.4f}" if homo is not None else "N/A"
                        lumo_str = f"{lumo:.4f}" if lumo is not None else "N/A"

                        # Calculer le gap HOMO-LUMO
                        if homo is not None and lumo is not None:
                            gap = lumo - homo
                            gap_str = f"{gap:.4f}"
                        else:
                            gap_str = "N/A"

                        f.write(f"{conformer:<15} {homo_str:<15} {lumo_str:<15} {gap_str:<15}\n")
                    else:
                        f.write(f"{conformer:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}\n")

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
        job_results[name] = {'neutral': None, 'sp': None, 'opt': None}

        try:
            # Step 1: Optimize neutral
            job_neutral = optimize_neutral(mol, name, args.solvent, args.functional, args.basis)
            if not job_neutral:
                continue
            job_results[name]['neutral'] = job_neutral

            # Step 2: Single point reduced
            job_sp = sp_reduced(job_neutral, name, args.solvent, args.functional, args.basis)
            if not job_sp:
                continue
            job_results[name]['sp'] = job_sp

            # Step 3: Optimize reduced
            job_opt = optimize_reduced(job_sp, name, args.solvent, args.functional, args.basis)
            if not job_opt:
                continue
            job_results[name]['opt'] = job_opt
        except Exception as e:
            print(f"ERREUR lors du traitement de {basename}: {str(e)}")
            continue

    # Analyse RMSD
    rmsd_results, rmsd_export_data = compare_conformers_rmsd(job_results)

    # Export output files
    export_molecules(job_results)

    # Extraire et afficher les énergies
    parent_dir = os.path.dirname(config.default_jobmanager.workdir)
    energy_data = extract_engine_energies(parent_dir)

    # Analyser les énergies redox
    analyze_redox_energies(energy_data, rmsd_export_data=rmsd_export_data)

    # Finalize PLAMS
    finish()

if __name__ == "__main__":
    main()
