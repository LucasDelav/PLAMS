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
    debug_info = {}  # Nouveau dictionnaire pour stocker les informations de débogage

    try:
        #print(f"Recherche des dossiers redox* dans: {workdir_parent}")

        # Vérifier que le dossier parent existe
        if not os.path.exists(workdir_parent):
            print(f"ERREUR: Le dossier parent '{workdir_parent}' n'existe pas")
            return energy_results

        # Trouver tous les dossiers redox*
        try:
            redox_dirs = [d for d in os.listdir(workdir_parent)
                          if d.startswith('redox') and os.path.isdir(os.path.join(workdir_parent, d))]
            #print(f"Dossiers redox* trouvés: {redox_dirs}")
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
        # print(f"Chemin complet du dossier de résultats: {results_dir}")

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

        # print(f"Dossier analysé: {results_dir}")
        print(f"{'Fichier':<30} {'Engine Energy':<15} {'Internal Energy U':<15} {'-T*S':<15} {'Gibbs Energy':<15}")
        print("-"*95)

        # Analyser chaque fichier .out
        for out_file in sorted(out_files):
            file_path = os.path.join(results_dir, out_file)
            Ee = None
            U = None
            TS = None
            G = None

            # Informations de débogage pour ce fichier
            debug_info[out_file] = {
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'found_patterns': {
                    'Energy from Engine': False,
                    'Internal Energy U': False,
                    '-T*S': False,
                    'Gibbs free energy': False
                },
                'raw_lines': {}
            }

            try:
                with open(file_path, 'r') as file:
                    line_count = 0
                    for line in file:
                        line_count += 1

                        # Recherche "Energy from Engine:"
                        if "Energy from Engine:" in line:
                            debug_info[out_file]['found_patterns']['Energy from Engine'] = True
                            debug_info[out_file]['raw_lines']['Energy from Engine'] = line.strip()
                            parts = line.split()
                            if len(parts) >= 5:  # Au moins 5 parties après le split
                                try:
                                    # Prendre la dernière valeur (en kJ/mol)
                                    Ee = float(parts[-1])
                                except ValueError as e:
                                    debug_info[out_file]['raw_lines']['Energy from Engine_error'] = str(e)

                        # Recherche "Internal Energy U:"
                        elif "Internal Energy U:" in line:
                            debug_info[out_file]['found_patterns']['Internal Energy U'] = True
                            debug_info[out_file]['raw_lines']['Internal Energy U'] = line.strip()
                            parts = line.split()
                            if len(parts) >= 5:  # Assez de parties après le split
                                try:
                                    U = float(parts[-1])  # Dernière valeur (kJ/mol)
                                except ValueError as e:
                                    debug_info[out_file]['raw_lines']['Internal Energy U_error'] = str(e)

                        # Recherche "-T*S:"
                        elif "-T*S:" in line:
                            debug_info[out_file]['found_patterns']['-T*S'] = True
                            debug_info[out_file]['raw_lines']['-T*S'] = line.strip()
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    TS = float(parts[-1])  # Dernière valeur (kJ/mol)
                                except ValueError as e:
                                    debug_info[out_file]['raw_lines']['-T*S_error'] = str(e)

                        # Recherche "Gibbs free energy:"
                        elif "Gibbs free energy:" in line:
                            debug_info[out_file]['found_patterns']['Gibbs free energy'] = True
                            debug_info[out_file]['raw_lines']['Gibbs free energy'] = line.strip()
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    G = float(parts[-1])  # Dernière valeur (kJ/mol)
                                except ValueError as e:
                                    debug_info[out_file]['raw_lines']['Gibbs free energy_error'] = str(e)

                    # Enregistrer le nombre de lignes lues
                    debug_info[out_file]['total_lines'] = line_count

            except Exception as e:
                print(f"ERREUR lors de la lecture du fichier {file_path}: {str(e)}")
                debug_info[out_file]['error'] = str(e)
                continue

            # Stocker les résultats dans le dictionnaire
            energy_results[out_file] = {
                'Ee': Ee,
                'U': U,
                'TS': TS,
                'G': G
            }

            # Afficher les résultats
            Ee_str = f"{Ee:.2f}" if Ee is not None else "Non trouvée"
            U_str = f"{U:.2f}" if U is not None else "Non trouvée"
            TS_str = f"{TS:.2f}" if TS is not None else "Non trouvée"
            G_str = f"{G:.2f}" if G is not None else "Non trouvée"

            print(f"{out_file:<30} {Ee_str:<15} {U_str:<15} {TS_str:<15} {G_str:<15}")

            # Vérifier si toutes les valeurs sont None et générer un avertissement
            if all(v is None for v in [Ee, U, TS, G]):
                print(f"  AVERTISSEMENT: Aucune valeur d'énergie trouvée dans {out_file}!")

        print("="*95)
        print("Toutes les énergies sont exprimées en kJ/mol")

    except Exception as e:
        print(f"ERREUR GLOBALE lors de l'extraction des énergies: {str(e)}")
        import traceback
        traceback.print_exc()

    return energy_results

def analyze_redox_energies(energy_data, temperature=298.15):
    """
    Analyse les valeurs d'énergie extraites et calcule les paramètres redox.

    Args:
        energy_data (dict): Dictionnaire contenant les énergies extraites par fichier
        temperature (float): Température en K (par défaut: 298.15K)

    Returns:
        tuple: (avg_params, potentials, redox_parameters, boltzmann_weights)
               - avg_params: Paramètres redox moyens
               - potentials: Potentiels de réduction
               - redox_parameters: Paramètres redox par conformère
               - boltzmann_weights: Poids de Boltzmann par conformère
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

        # # Afficher les résultats par conformère
        # try:
        #     print("\nParamètres redox par conformère:")
        #     print(f"{'Conformère':<15} {'Poids':<10} {'∆G (kJ/mol)':<12} {'EA (kJ/mol)':<12} {'Edef (kJ/mol)':<14} {'∆∆U (kJ/mol)':<14} {'-T∆S (kJ/mol)':<14}")
        #     print("-"*95)
        #     for conformer in sorted(complete_conformers):
        #         params = redox_parameters[conformer]
        #         weight = boltzmann_weights[conformer]
        #         print(f"{conformer:<15} {weight:.4f} {params['delta_G']:12.2f} {params['EA']:12.2f} {params['Edef']:14.2f} {params['delta_delta_U']:14.2f} {-params['T_delta_S']:14.2f}")
        # except Exception as e:
        #     print(f"ERREUR lors de l'affichage des paramètres par conformère: {str(e)}")

        # Afficher les moyennes pondérées
        try:
            print("\nMoyennes pondérées des conformers (kJ/mol):")
            print(f"∆G   = {avg_params['delta_G']:.2f}")
            print(f"EA   = {avg_params['EA']:.2f}")
            print(f"Edef = {avg_params['Edef']:.2f}")
            print(f"∆∆U  = {avg_params['delta_delta_U']:.2f}")
            print(f"-T∆S = {-avg_params['T_delta_S']:.2f}")

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

            # Nom du fichier avec un horodatage pour éviter les écrasements
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
                f.write(f"E(∆G)   = {potentials.get('E_delta_G', 0):.4f} V\n")
                f.write("\nContributions:\n")
                f.write(f"E(EA)   = {potentials.get('E_EA', 0):.4f} V\n")
                f.write(f"E(Edef) = {potentials.get('E_Edef', 0):.4f} V\n")
                f.write(f"E(∆∆U)  = {potentials.get('E_delta_delta_U', 0):.4f} V\n")
                f.write(f"E(T∆S)  = {potentials.get('E_T_delta_S', 0):.4f} V\n")

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

                # f.write("\n\nÉNERGIES (kJ/mol):\n")
                # f.write("-" * 50 + "\n")
                # for param, value in avg_params.items():
                #     f.write(f"{param} = {value:.4f}\n")
                # f.write(f"RT = {RT:.4f}\n")

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
            print(f"Erreur lors du traitement de {basename}: {str(e)}")

    # Export output files
    export_molecules(job_results)
    
    # Extraire et afficher les énergies
    parent_dir = os.path.dirname(config.default_jobmanager.workdir)
    energy_data = extract_engine_energies(parent_dir)
    
    # Analyser les énergies redox
    analyze_redox_energies(energy_data)

    # Finalize PLAMS
    finish()

if __name__ == "__main__":
    main()
