#!/usr/bin/env amspython

import os
import argparse
import glob
import math
import numpy as np
import re
from scm.plams import *

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

def configure_functional(s, functional):
    """
    Configure la fonctionnelle appropriée dans les paramètres Settings
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

    print(f"Attention : Fonctionnelle {functional} non reconnue, utilisation de PBE0 par defaut")
    s.input.adf.XC.Hybrid = "PBE0"

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

def init_workdir(input_dir):
    """
    Initialize PLAMS with a work directory in the same location as conformers_gen.py
    """
    # Extraction du chemin parent à partir de input_dir
    parent_dir = os.path.dirname(os.path.abspath(input_dir))
    if parent_dir.endswith('/results'):
        parent_dir = os.path.dirname(parent_dir)
        
    # Préserver le format original du chemin (avec le point)
    workdir = os.path.join(parent_dir, "redox")
    
    init(folder=workdir)
    return workdir

def setup_adf_settings(task='GeometryOptimization', charge=0, spin_polarization=0, 
                       solvent="Acetonitrile", functional="PBE0", basis="DZP"):
    """
    Configure ADF settings object based on task and molecular state
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
    
    # Configuration de la solvatation COSMO (modèle modifié)
    s.input.adf.Solvation.Solv = f"name={solvent}"

    return s

def optimize_neutral(mol, name, solvent="Acetonitrile", functional="PBE0", basis="DZP"):
    """
    Étape 1: Optimisation géométrique de la molécule neutre
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
    Étape 3: Optimisation géométrique de la molécule réduite
    """
    # Récupérer la molécule du calcul simple point
    mol = job_sp.results.get_main_molecule()
    
    print(f"\nÉtape 3: Optimisation géométrique de {name} (réduit, charge -1)")
    settings = setup_adf_settings(task="GeometryOptimization", charge=-1, spin_polarization=1.0, 
                                solvent=solvent, functional=functional, basis=basis)
    
    job = AMSJob(settings=settings, name=f"{name}_reduit_opt", molecule=mol)
    job.run()
    
    if job.check():
        print(f"  Optimisation réussie pour {name} (réduit)")
        return job
    else:
        print(f"  ERREUR: Optimisation échouée pour {name} (réduit)")
        return None

def export_molecules(jobs_data, prefix="redox_results"):
    """
    Exporter les fichiers de sortie .out dans le dossier de travail actuel de PLAMS
    """
    # Utiliser le dossier de travail actuel de PLAMS
    current_workdir = config.default_jobmanager.workdir
    output_dir = os.path.join(current_workdir, prefix)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nExportation des résultats de calcul dans {output_dir}/")
    
    for name, jobs in jobs_data.items():
        if not all(jobs.values()):
            continue

        # Exporter les fichiers de résultats pour chaque calcul
        for job_type, job in jobs.items():
            # Chercher le fichier .out avec le bon nom dans le répertoire du job
            job_name = job.name
            out_file = os.path.join(job.path, f"{job_name}.out")
            
            if os.path.exists(out_file):
                # Nom du fichier de sortie cible
                target_file = os.path.join(output_dir, f"{name}_{job_type}.out")
                # Copier le fichier
                import shutil
                shutil.copy2(out_file, target_file)
                print(f"  Succès: fichier {name}_{job_type}.out exporté")
            else:
                print(f"  Avertissement: fichier de sortie pour {name}_{job_type} introuvable ({out_file})")
    
    print(f"Exportation terminée.")

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
    
    # Trouver tous les dossiers redox*
    redox_dirs = [d for d in os.listdir(workdir_parent) if d.startswith('redox') and os.path.isdir(os.path.join(workdir_parent, d))]
    
    if not redox_dirs:
        print(f"Aucun dossier redox* trouvé dans {workdir_parent}")
        return energy_results
    
    # Utiliser le dossier redox le plus récent (numériquement le plus élevé)
    latest_redox = sorted(redox_dirs)[-1]
    results_dir = os.path.join(workdir_parent, latest_redox, 'redox_results')
    
    if not os.path.exists(results_dir):
        print(f"Le dossier {results_dir} n'existe pas")
        return energy_results
    
    # Trouver tous les fichiers .out
    out_files = [f for f in os.listdir(results_dir) if f.endswith('.out')]
    
    if not out_files:
        print(f"Aucun fichier .out trouvé dans {results_dir}")
        return energy_results
    
    print(f"Dossier analysé: {results_dir}")
    print(f"{'Fichier':<30} {'Engine Energy':<15} {'Internal Energy U':<15} {'-T*S':<15} {'Gibbs Energy':<15}")
    print("-"*95)
    
    # Analyser chaque fichier .out
    for out_file in sorted(out_files):
        file_path = os.path.join(results_dir, out_file)
        Ee = None
        U = None
        TS = None
        G = None
        
        with open(file_path, 'r') as file:
            for line in file:
                if "Energy from Engine:" in line:
                    # Format attendu: "Energy from Engine: xxx xxx xxx -5499.61"
                    parts = line.split()
                    if len(parts) >= 5:  # Au moins 5 parties après le split
                        try:
                            # Prendre la dernière valeur (en kJ/mol)
                            Ee = float(parts[-1])
                        except ValueError:
                            pass
                elif "Internal Energy U:" in line:
                    # Extraire l'énergie interne U
                    parts = line.split()
                    if len(parts) >= 5:  # Assez de parties après le split
                        try:
                            U = float(parts[-1])  # Dernière valeur (kJ/mol)
                        except ValueError:
                            pass
                elif "-T*S:" in line:
                    # Extraire l'entropie (-T*S)
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            TS = float(parts[-1])  # Dernière valeur (kJ/mol)
                        except ValueError:
                            pass
                elif "Gibbs free energy:" in line:
                    # Extraire l'énergie libre de Gibbs
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            G = float(parts[-1])  # Dernière valeur (kJ/mol)
                        except ValueError:
                            pass
        
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
    
    print("="*95)
    print("Toutes les énergies sont exprimées en kJ/mol")
    
    return energy_results


def analyze_redox_energies(energy_data, temperature=298.15):
    """
    Analyse les valeurs d'énergie extraites et calcule les paramètres redox.
    
    Args:
        energy_data (dict): Dictionnaire contenant les énergies extraites par fichier
        temperature (float): Température en K (par défaut: 298.15K)
    """
    import math
    import re
    
    # Constantes
    R = 8.314462618e-3  # Constante des gaz parfaits en kJ/(mol·K)
    RT = R * temperature  # kJ/mol à 298.15K ≈ 2.48 kJ/mol
    F = 96.485  # Constante de Faraday en kJ/(mol·V)
    
    print("\n" + "="*80)
    print("ANALYSE ÉNERGÉTIQUE DU PROCESSUS REDOX")
    print("="*80)
    
    # Organiser les données par conformères et type de calcul
    conformers_data = {}
    pattern = r'(.+?)_(neutral|sp|opt)\.out'
    
    for filename, energies in energy_data.items():
        match = re.search(pattern, filename)
        if not match:
            continue
            
        conformer = match.group(1)
        calc_type = match.group(2)
        
        if conformer not in conformers_data:
            conformers_data[conformer] = {}
        
        conformers_data[conformer][calc_type] = energies
    
    # Vérifier quels conformères ont tous les calculs nécessaires
    complete_conformers = []
    for conformer, calcs in conformers_data.items():
        if all(calc_type in calcs for calc_type in ['neutral', 'sp', 'opt']):
            if all(calcs[calc_type]['Ee'] is not None and 
                  calcs[calc_type]['U'] is not None and 
                  calcs[calc_type]['TS'] is not None and 
                  calcs[calc_type]['G'] is not None 
                  for calc_type in ['neutral', 'sp', 'opt']):
                complete_conformers.append(conformer)
    
    if not complete_conformers:
        print("Aucun conformère n'a tous les calculs nécessaires avec toutes les énergies")
        return
    
    # Calculer Delta U pour chaque cas
    for conformer in complete_conformers:
        for calc_type in ['neutral', 'sp', 'opt']:
            energies = conformers_data[conformer][calc_type]
            energies['delta_U'] = energies['U'] - energies['Ee']
    
    # Calculer les paramètres redox pour chaque conformère
    redox_parameters = {}
    for conformer in complete_conformers:
        neutral_data = conformers_data[conformer]['neutral']
        sp_data = conformers_data[conformer]['sp']
        opt_data = conformers_data[conformer]['opt']
        
        delta_G = opt_data['G'] - neutral_data['G']
        EA = sp_data['Ee'] - neutral_data['Ee']
        Edef = opt_data['Ee'] - sp_data['Ee']
        delta_delta_U = opt_data['delta_U'] - neutral_data['delta_U']
        T_delta_S = opt_data['TS'] - neutral_data['TS']
        
        redox_parameters[conformer] = {
            'delta_G': delta_G,
            'EA': EA,
            'Edef': Edef,
            'delta_delta_U': delta_delta_U,
            'T_delta_S': T_delta_S
        }
    
    # Calculer les poids de Boltzmann en fonction des énergies de Gibbs des conformères neutres
    min_G = min([conformers_data[conf]['neutral']['G'] for conf in complete_conformers])
    
    boltzmann_weights = {}
    denominator = 0.0
    for conformer in complete_conformers:
        rel_G = conformers_data[conformer]['neutral']['G'] - min_G
        boltzmann_factor = math.exp(-rel_G/(R*temperature))
        boltzmann_weights[conformer] = boltzmann_factor
        denominator += boltzmann_factor
    
    # Normaliser les poids
    for conformer in complete_conformers:
        boltzmann_weights[conformer] /= denominator
    
    # Calculer les moyennes pondérées
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
            avg_params[param] += weight * redox_parameters[conformer][param]
    
    # Vérifier l'égalité thermodynamique
    equality_check = avg_params['EA'] + avg_params['Edef'] + avg_params['delta_delta_U'] + avg_params['T_delta_S']
    equality_diff = avg_params['delta_G'] - equality_check
    
    # Calculer les potentiels de réduction (V)
    potentials = {
        'E_delta_G': -avg_params['delta_G']/F,
        'E_EA': -avg_params['EA']/F,
        'E_Edef': -avg_params['Edef']/F,
        'E_delta_delta_U': -avg_params['delta_delta_U']/F,
        'E_T_delta_S': -avg_params['T_delta_S']/F,
    }
    
    # Afficher les résultats par conformère
    print("\nParamètres redox par conformère:")
    print(f"{'Conformère':<15} {'Poids':<10} {'∆G (kJ/mol)':<12} {'EA (kJ/mol)':<12} {'Edef (kJ/mol)':<14} {'∆∆U (kJ/mol)':<14} {'-T∆S (kJ/mol)':<14}")
    print("-"*95)
    for conformer in sorted(complete_conformers):
        params = redox_parameters[conformer]
        weight = boltzmann_weights[conformer]
        print(f"{conformer:<15} {weight:.4f} {params['delta_G']:12.2f} {params['EA']:12.2f} {params['Edef']:14.2f} {params['delta_delta_U']:14.2f} {-params['T_delta_S']:14.2f}")
    
    # Afficher les moyennes pondérées
    print("\nMoyennes pondérées (kJ/mol):")
    print(f"∆G = {avg_params['delta_G']:.2f}")
    print(f"EA = {avg_params['EA']:.2f}")
    print(f"Edef = {avg_params['Edef']:.2f}")
    print(f"∆∆U = {avg_params['delta_delta_U']:.2f}")
    print(f"-T∆S = {-avg_params['T_delta_S']:.2f}")
    
    # Vérifier l'égalité thermodynamique
    print(f"\nVérification de l'égalité thermodynamique:")
    print(f"∆G = {avg_params['delta_G']:.2f} kJ/mol")
    print(f"EA + Edef + ∆∆U - T∆S = {equality_check:.2f} kJ/mol")
    print(f"Différence = {equality_diff:.2f} kJ/mol")
    
    # Afficher les potentiels de réduction
    print("\nPotentiels de réduction (V vs. référence):")
    print(f"E(∆G) = {potentials['E_delta_G']:.3f} V")
    print(f"E(EA) = {potentials['E_EA']:.3f} V")
    print(f"E(Edef) = {potentials['E_Edef']:.3f} V")
    print(f"E(∆∆U) = {potentials['E_delta_delta_U']:.3f} V")
    print(f"E(T∆S) = {potentials['E_T_delta_S']:.3f} V")
    
    print("\nSomme des contributions:")
    print(f"E(EA) + E(Edef) + E(∆∆U) - E(T∆S) = {potentials['E_EA'] + potentials['E_Edef'] + potentials['E_delta_delta_U'] + potentials['E_T_delta_S']:.3f} V")
    print(f"E(∆G) = {potentials['E_delta_G']:.3f} V")
    
    print("="*80)

    # Enregistrer les résultats dans un fichier texte dans ../results/
    try:
        # Déterminer le chemin du dossier results
        workdir = config.default_jobmanager.workdir
        results_dir = os.path.join(workdir, 'redox_results')
        
        # Créer le dossier s'il n'existe pas
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Nom du fichier avec un horodatage pour éviter les écrasements
        output_file = os.path.join(results_dir, f"redox_potentials.txt")
        
        with open(output_file, 'w') as f:
            f.write("POTENTIELS DE RÉDUCTION\n")
            f.write("=" * 50 + "\n\n")
            
            # Informations générales
            f.write(f"Température: {temperature} K\n")
            f.write(f"RT: {RT:.4f} kJ/mol\n")
            f.write(f"Nombre de conformères: {len(complete_conformers)}\n\n")
            
            # Potentiels moyens
            f.write("POTENTIELS MOYENS (V vs. référence):\n")
            f.write("-" * 50 + "\n")
            f.write(f"E(∆G) = {potentials['E_delta_G']:.4f} V\n")
            f.write("\nContributions:\n")
            f.write(f"E(EA) = {potentials['E_EA']:.4f} V\n")
            f.write(f"E(Edef) = {potentials['E_Edef']:.4f} V\n")
            f.write(f"E(∆∆U) = {potentials['E_delta_delta_U']:.4f} V\n")
            f.write(f"E(T∆S) = {potentials['E_T_delta_S']:.4f} V\n")
            f.write(f"Somme: {potentials['E_EA'] + potentials['E_Edef'] + potentials['E_delta_delta_U'] + potentials['E_T_delta_S']:.4f} V\n\n")
            
            # Données par conformère
            f.write("DÉTAILS PAR CONFORMÈRE:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Conformère':<15} {'Poids':<10} {'E(∆G) [V]':<12}\n")
            f.write("-" * 40 + "\n")
            
            for conformer in sorted(complete_conformers):
                weight = boltzmann_weights[conformer]
                e_value = -redox_parameters[conformer]['delta_G']/F
                f.write(f"{conformer:<15} {weight:.4f} {e_value:12.4f}\n")
                
            f.write("\n\nÉNERGIES (kJ/mol):\n")
            f.write("-" * 50 + "\n")
            f.write(f"∆G = {avg_params['delta_G']:.4f}\n")
            f.write(f"EA = {avg_params['EA']:.4f}\n")
            f.write(f"Edef = {avg_params['Edef']:.4f}\n")
            f.write(f"∆∆U = {avg_params['delta_delta_U']:.4f}\n")
            f.write(f"RT = {RT:.4f}\n")
            f.write(f"T∆S = {avg_params['T_delta_S']:.4f}\n")
            
        print(f"\nRésultats enregistrés dans {output_file}")
    except Exception as e:
        print(f"\nErreur lors de l'enregistrement des résultats: {e}")
    
    return avg_params, potentials, redox_parameters, boltzmann_weights

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize PLAMS
    workdir = init_workdir(args.input_dir)
    
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
