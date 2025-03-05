#!/usr/bin/env amspython

import os
import argparse
import numpy as np
import shutil
import sys
from scm.plams import *
from scm.conformers import ConformersJob
from scm.plams.core.settings import Settings

# CONSTANTES GLOBALES
TEMPERATURE = 298  # Temperature en Kelvin
R = 8.314  # Constante des gaz parfaits en J/(mol*K)
UNIT = "kJ/mol"  # Unite d'energie par defaut pour le script

# Liste des fonctionnelles disponibles dans ADF
LDA_FUNCTIONALS = ["VWN", "XALPHA", "Xonly", "Stoll"]
GGA_FUNCTIONALS = ["PBE", "RPBE", "revPBE", "PBEsol", "BLYP", "BP86", "PW91", "mPW", "OLYP", "OPBE", 
                  "KT1", "KT2", "BEE", "BJLDA", "BJPBE", "BJGGA", "S12G", "LB94", "mPBE", "B3LYPgauss"]
METAGGA_FUNCTIONALS = ["M06L", "TPSS", "revTPSS", "MVS", "SCAN", "revSCAN", "r2SCAN", "tb-mBJ"]
HYBRID_FUNCTIONALS = ["B3LYP", "B3LYP*", "B1LYP", "O3LYP", "X3LYP", "BHandH", "BHandHLYP", 
                     "B1PW91", "MPW1PW", "MPW1K", "PBE0", "OPBE0", "TPSSh", "M06", "M06-2X", "S12H"]
METAHYBRID_FUNCTIONALS = ["M08-HX", "M08-SO", "M11", "TPSSH", "PW6B95", "MPW1B95", "MPWB1K", 
                         "PWB6K", "M06-HF", "BMK"]

# Liste complete pour l'argument --functional
ALL_FUNCTIONALS = ['HF'] + LDA_FUNCTIONALS + GGA_FUNCTIONALS + METAGGA_FUNCTIONALS + HYBRID_FUNCTIONALS + METAHYBRID_FUNCTIONALS

# ------------------------------------------------------------
# Organisation des dossiers de travail
# ------------------------------------------------------------
def organize_workdir(workdir, molecule_name):
    """
    Organise les dossiers du working directory pour plus de proprete.
    """
    # Creer un dossier pour les calculs intermediaires
    calc_dir = os.path.join(workdir, "calculations")
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
    
    # Creer un dossier pour les resultats finaux
    results_dir = os.path.join(workdir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Creer un dossier pour les calculs de frequence
    freq_dir = os.path.join(calc_dir, "freq_calculations")
    if not os.path.exists(freq_dir):
        os.makedirs(freq_dir)
    
    return calc_dir, results_dir, freq_dir

# Fonction pour obtenir le dossier de travail actuel (même si c'est un .002, .003, etc.)
def get_actual_workdir_path(base_name):
    """Retourne le chemin réel du dossier de travail PLAMS actuel"""
    if hasattr(config, 'default_jobmanager') and config.default_jobmanager:
        return os.path.abspath(config.default_jobmanager.workdir)
    else:
        return None

# ------------------------------------------------------------
# Gestion des arguments en ligne de commande
# ------------------------------------------------------------
def parse_arguments():
    """
    Definit et analyse les arguments en ligne de commande.
    """
    parser = argparse.ArgumentParser(
        description="Script d'analyse des conformers pour une molecule donnee.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Arguments obligatoires
    parser.add_argument("smiles", help="SMILES de la molecule", type=str)
    parser.add_argument(
        "name", help="Nom de la molecule", nargs="?", default="Molecule", type=str
    )
    
    # Options pour l'etape 2 (Optimisation)
    parser.add_argument("--e-window-opt", type=float, default=20,
                        help="Fenetre energetique (kJ/mol) pour l'etape d'optimisation")
    
    # Options pour l'etape 3 (Scoring)
    parser.add_argument("--functional", choices=ALL_FUNCTIONALS, default="PBE0",
                        help="Fonctionnelle pour le scoring")
    parser.add_argument("--basis", choices=["SZ", "DZ", "DZP", "TZP", "TZ2P", "QZ4P"], default="TZP",
                        help="Base utilisee pour le scoring")
    parser.add_argument("--e-window-score", type=float, default=8,
                        help="Fenetre energetique (kJ/mol) pour l'etape de scoring")
    parser.add_argument("--dispersion", choices=["None", "GRIMME3", "GRIMME4", "UFF", "GRIMME3 BJDAMP"], default="GRIMME3",
                        help="Correction de dispersion pour le scoring")
    
    # Options pour l'etape 4 (Filtrage)
    parser.add_argument("--e-window-filter", type=float, default=4,
                        help="Fenetre energetique (kJ/mol) pour l'etape de filtrage")
    parser.add_argument("--rmsd-threshold", type=float, default=1.0,
                        help="Seuil RMSD (A) pour considerer deux conformeres comme identiques")
    
    # Options pour la derniere etape (frequences)
    parser.add_argument("--freq-functional", choices=ALL_FUNCTIONALS, default=None,
                        help="Fonctionnelle pour le calcul des frequences (par defaut: meme que scoring)")
    parser.add_argument("--freq-basis", choices=["SZ", "DZ", "DZP", "TZP", "TZ2P"], default="DZ",
                        help="Base utilisee pour le calcul des frequences")
    
    return parser.parse_args()

# ------------------------------------------------------------
# Initialisation de PLAMS avec un dossier personnalise
# ------------------------------------------------------------
def init_with_custom_folder(name):
    """
    Initialise PLAMS avec un nom de dossier personnalise base sur le nom de la molecule.
    """
    # Creer un nom de dossier valide en remplacant les caracteres problematiques
    folder_name = f"{name}_workdir"
    folder_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in folder_name)
    
    init(folder=folder_name)
    
    # Obtenir le chemin réel du dossier de travail après initialisation
    actual_workdir = get_actual_workdir_path(folder_name)
    
    if actual_workdir is None:
        # Fallback si le dossier n'a pas pu être déterminé
        actual_workdir = folder_name
    
    return actual_workdir

# ------------------------------------------------------------
# Calcul des poids de Boltzmann
# ------------------------------------------------------------
def boltzmann_weights(energies, temperature):
    """
    Calcule les poids de Boltzmann pour un ensemble d'energies.

    Args:
        energies (list): Liste des energies des conformers (en kJ/mol).
        temperature (float): Temperature (en Kelvin).

    Returns:
        tuple: Poids normalises et fonction de partition.
    """
    beta = 1 / (R / 1000 * temperature)  # R divise par 1000 pour convertir en kJ/mol
    exponentials = np.exp(-beta * np.array(energies))
    partition_function = sum(exponentials)
    weights = exponentials / partition_function
    return weights, partition_function

# ------------------------------------------------------------
# Impression des resultats
# ------------------------------------------------------------
def print_results(job: ConformersJob, temperature=298, unit=UNIT):
    """
    Affiche les resultats en termes d'energies relatives et de poids de Boltzmann.

    Args:
        job (ConformersJob): Tache contenant les resultats a traiter.
    """
    energies = job.results.get_relative_energies(unit)
    weights, Z = boltzmann_weights(energies, temperature)

    print(f"\nResultats (Temperature = {temperature} K) :")
    print(f'{"#":>4s} {"Delta E [{}]".format(unit):>16s} {"Poids Boltzmann":>16s}')

    for i, (energy, weight) in enumerate(zip(energies, weights)):
        print(f"{i+1:4d} {energy:16.2f} {weight:16.8f}")

    print("\nResume :")
    print(f"Nombre total de conformers : {len(energies)}")
    print(f"Fonction de partition (Z) = {Z:.8f}")

    return energies, weights

# ------------------------------------------------------------
# Configuration d'une fonctionnelle
# ------------------------------------------------------------
def configure_functional(s, functional):
    """
    Configure les parametres de la fonctionnelle dans les settings.

    Args:
        s (Settings): Objet Settings a configurer
        functional (str): Nom de la fonctionnelle a utiliser
    """
    # Hartree-Fock pur
    if functional == "HF":
        s.input.adf.XC.HF = "Yes"
        return
    
    # LDA
    if functional in LDA_FUNCTIONALS:
        s.input.adf.XC.LDA = functional
        return
        
    # GGA
    if functional in GGA_FUNCTIONALS:
        s.input.adf.XC.GGA = functional
        return
        
    # Meta-GGA
    if functional in METAGGA_FUNCTIONALS:
        s.input.adf.XC.MetaGGA = functional
        s.input.adf.NumericalQuality = "Good"  # Important pour meta-GGA
        return
        
    # Hybride
    if functional in HYBRID_FUNCTIONALS:
        s.input.adf.XC.Hybrid = functional
        return
        
    # Meta-Hybride
    if functional in METAHYBRID_FUNCTIONALS:
        s.input.adf.XC.MetaHybrid = functional
        s.input.adf.NumericalQuality = "Good"  # Important pour meta-hybride
        return
        
    # Par defaut, si la fonctionnelle n'est pas reconnue
    print(f"Attention : Fonctionnelle {functional} non reconnue, utilisation de PBE0 par defaut")
    s.input.adf.XC.Hybrid = "PBE0"

# ------------------------------------------------------------
# Etape 1 : Generation des conformers
# ------------------------------------------------------------
def generate_conformers(molecule, calc_dir):
    print("\n[Etape 1] Generation des conformers avec RDKit...")
    s = Settings()
    s.input.ams.Task = "Generate"
    s.input.ams.Generator.Method = "RDKit"
    s.input.ams.Generator.RDKit.InitialNConformers = 1000
    s.input.ForceField.Type = "UFF"
    
    # CORRECTION: Création d'un sous-dossier pour ce calcul
    job_dir = os.path.join(calc_dir, "generate_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    # Configuration du job
    job = ConformersJob(molecule=molecule, settings=s, name="generate_conformers")
    
    # CORRECTION: Exécuter le job dans le sous-dossier spécifique
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError("La generation des conformers a echoue.")
    return job

# ------------------------------------------------------------
# Etape 2 : Optimisation des conformers
# ------------------------------------------------------------
def optimize_conformers(previous_job, calc_dir, e_window=20):
    print(f"\n[Etape 2] Optimisation des conformers geometriques avec DFTB3 (fenetre d'energie = {e_window} kJ/mol)...")
    s = Settings()
    s.input.ams.Task = "Optimize"
    # CORRECTION: Utiliser le chemin absolu pour le fichier RKF
    rkf_path = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputConformersSet = rkf_path
    s.input.ams.InputMaxEnergy = e_window
    s.input.dftb.Model = "DFTB3"
    s.input.dftb.ResourcesDir = "DFTB.org/3ob-3-1"
    
    # CORRECTION: Création d'un sous-dossier pour ce calcul
    job_dir = os.path.join(calc_dir, "optimize_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    job = ConformersJob(settings=s, name="optimize_conformers")
    
    # CORRECTION: Exécuter le job dans le sous-dossier spécifique
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError("L'optimisation des conformers a echoue.")
    return job

# ------------------------------------------------------------
# Etape 3 : Scoring des conformers
# ------------------------------------------------------------
def score_conformers(previous_job, calc_dir, functional="PBE0", basis="TZP", e_window=8, dispersion="GRIMME3"):
    print(f"\n[Etape 3] Calcul des energies des conformers avec {functional}/{basis} (fenetre d'energie = {e_window} kJ/mol)...")
    s = Settings()
    s.input.ams.Task = "Score"
    # CORRECTION: Utiliser le chemin absolu pour le fichier RKF
    rkf_path = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputConformersSet = rkf_path
    s.input.ams.InputMaxEnergy = e_window
    
    # Configuration de la dispersion
    if dispersion != "None":
        if dispersion == "GRIMME3 BJDAMP":
            s.input.adf.XC.DISPERSION = "GRIMME3"
            s.input.adf.XC.DISPERSION += " BJDAMP"
        else:
            s.input.adf.XC.DISPERSION = dispersion
    
    # Configuration de la fonctionnelle
    configure_functional(s, functional)
    
    # Configuration de la base
    s.input.adf.BASIS.Type = basis
    
    # Pour les meta-GGA et meta-hybrides, meilleure qualite numerique requise
    if functional in METAGGA_FUNCTIONALS or functional in METAHYBRID_FUNCTIONALS:
        s.input.adf.NumericalQuality = "Good"
    
    # CORRECTION: Création d'un sous-dossier pour ce calcul
    job_dir = os.path.join(calc_dir, "score_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    job = ConformersJob(settings=s, name="score_conformers")
    
    # CORRECTION: Exécuter le job dans le sous-dossier spécifique
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError(f"Le scoring des conformers avec {functional}/{basis} a echoue.")
    return job

# ------------------------------------------------------------
# Etape 4 : Filtrage final
# ------------------------------------------------------------
def filter_conformers(previous_job, calc_dir, e_window=4, rmsd_threshold=1.0):
    print(f"\n[Etape 4] Filtrage des conformers (fenetre d'energie = {e_window} kJ/mol, RMSD = {rmsd_threshold} A)...")
    s = Settings()
    s.input.ams.Task = "Filter"
    # CORRECTION: Utiliser le chemin absolu pour le fichier RKF
    rkf_path = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputConformersSet = rkf_path
    s.input.ams.InputMaxEnergy = e_window
    s.input.ams.Equivalence.CREST.RMSDThreshold = rmsd_threshold
    
    # CORRECTION: Création d'un sous-dossier pour ce calcul
    job_dir = os.path.join(calc_dir, "filter_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    job = ConformersJob(settings=s, name="filter_conformers")
    
    # CORRECTION: Exécuter le job dans le sous-dossier spécifique
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError("Le filtrage des conformers a echoue.")
    return job

# ------------------------------------------------------------
# Etape 5 : Optimisation de geometrie et calcul de frequences
# ------------------------------------------------------------
def verify_frequencies(previous_job, molecule_name, results_dir, freq_dir, freq_functional="PBE0", freq_basis="DZ"):
    print(f"\n[Etape 5] Optimisation de géométrie et vérification des frequences avec {freq_functional}/{freq_basis}...")
    
    # Recuperer les conformeres filtres
    conformers = previous_job.results.get_conformers()
    energies = previous_job.results.get_relative_energies(UNIT)
    print(f"Verification de {len(conformers)} conformeres...")
    
    # Liste pour stocker les conformeres valides et leurs informations
    valid_conformers = []
    valid_indices = []
    
    for i, conf in enumerate(conformers):
        print(f"\nOptimisation de géométrie du conformer {i+1}/{len(conformers)}...")
        
        # Configurer le calcul de frequence
        s = Settings()
        s.input.ams.task = 'GeometryOptimization'
        s.input.ams.properties.NormalModes = 'Yes'
        
        # Configuration de la fonctionnelle et de la base
        configure_functional(s, freq_functional)
        s.input.adf.basis.type = freq_basis
        
        # Pour les meta-GGA et meta-hybrides, meilleure qualite numerique requise
        if freq_functional in METAGGA_FUNCTIONALS or freq_functional in METAHYBRID_FUNCTIONALS:
            s.input.adf.NumericalQuality = "Good"
        
        # CORRECTION: Création d'un sous-dossier pour ce calcul
        job_dir = os.path.join(freq_dir, f"freq_check_conf_{i+1}")
        os.makedirs(job_dir, exist_ok=True)
        
        # CORRECTION: Utiliser le nom de base simple
        job_name = f"freq_check_conf_{i+1}"
        job = AMSJob(molecule=conf, settings=s, name=job_name)
        
        # CORRECTION: Exécuter le job dans le sous-dossier spécifique
        with DefaultJobManager(job_dir):
            result = job.run()
        
        # Verifier si le calcul a reussi
        if not job.ok():
            print(f"  [X] Le calcul a echoue pour le conformere {i+1}")
            continue
            
        # Verifier les frequences
        try:
            frequencies = job.results.get_frequencies()
            has_imaginary = any(freq < 0 for freq in frequencies)
            
            if has_imaginary:
                print(f"  [X] Conformere {i+1} : frequences imaginaires detectees")
                # Afficher les frequences negatives
                neg_freqs = [f for f in frequencies if f < 0]
                print(f"     Frequences imaginaires : {neg_freqs} cm-1")
            else:
                print(f"  [V] Conformere {i+1} : toutes les frequences sont positives")
                valid_conformers.append(conf)
                valid_indices.append(i)
                
                # OPTIMISATION: Sauvegarder directement le conformère valide dans results_dir
                xyz_filename = os.path.join(results_dir, f"{molecule_name}_conf_{len(valid_conformers)}.xyz")
                conf.write(xyz_filename)
                print(f"     Conformere valide sauvegarde : {xyz_filename}")
        except Exception as e:
            print(f"  [X] Erreur lors de l'analyse des frequences du conformere {i+1}: {str(e)}")
    
    # Créer un fichier récapitulatif des conformères valides avec leurs énergies relatives
    if valid_conformers:
        summary_file = os.path.join(results_dir, "conformers_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"# Conformeres valides pour {molecule_name}\n")
            f.write(f"# Calculs: Scoring: {freq_functional}/{freq_basis}\n")
            f.write(f"# Temperature: {TEMPERATURE} K\n\n")
            f.write(f"{'#':>3s} {'Conf. original':>15s} {'Delta E [kJ/mol]':>16s} {'Poids Boltzmann':>16s}\n")
            
            # Calculer les poids de Boltzmann uniquement pour les conformères valides
            valid_energies = [energies[i] for i in valid_indices]
            weights, Z = boltzmann_weights(valid_energies, TEMPERATURE)
            
            for j, (idx, energy, weight) in enumerate(zip(valid_indices, valid_energies, weights)):
                f.write(f"{j+1:3d} {idx+1:15d} {energy:16.4f} {weight:16.8f}\n")
                
            f.write(f"\nNombre total de conformeres valides: {len(valid_conformers)}\n")
            f.write(f"Fonction de partition (Z): {Z:.8f}\n")
        
        print(f"\nRecapitulatif des conformeres valides ecrit dans: {summary_file}")
    
    # Resume final
    print(f"\nResultat de l'analyse des frequences :")
    print(f"  Conformeres analyses : {len(conformers)}")
    if len(conformers) > 0:
        print(f"  Conformeres valides  : {len(valid_conformers)} ({len(valid_conformers)/len(conformers)*100:.1f}%)")
        print(f"  Conformeres rejetes  : {len(conformers)-len(valid_conformers)} ({(len(conformers)-len(valid_conformers))/len(conformers)*100:.1f}%)")
    else:
        print("  Aucun conformere a analyser!")
    
    return valid_conformers

# Classe pour gérer l'exécution des jobs dans un dossier spécifique
class DefaultJobManager:
    def __init__(self, job_dir):
        self.job_dir = job_dir
        self.original_jobmanager = None
        self.original_workdir = None
    
    def __enter__(self):
        # Sauvegarder le job manager actuel et son workdir
        if hasattr(config, 'default_jobmanager'):
            self.original_jobmanager = config.default_jobmanager
            if hasattr(config.default_jobmanager, 'workdir'):
                self.original_workdir = config.default_jobmanager.workdir
        
        # Créer un nouveau job manager pour le dossier spécifique
        os.makedirs(self.job_dir, exist_ok=True)
        
        # Configurer manuellement le dossier de travail
        if hasattr(config, 'default_jobmanager') and config.default_jobmanager:
            config.default_jobmanager.workdir = self.job_dir
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurer le workdir original
        if self.original_workdir and hasattr(config, 'default_jobmanager') and config.default_jobmanager:
            config.default_jobmanager.workdir = self.original_workdir

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
def main():
    # Analyse des arguments
    args = parse_arguments()
    
    # Si freq_functional est None, utiliser la meme fonctionnelle que pour le scoring
    if args.freq_functional is None:
        args.freq_functional = args.functional
    
    # Initialisation avec un dossier de travail personnalise
    workdir = init_with_custom_folder(args.name)
    print(f"\nDossier de travail cree: {workdir}")
    
    # Organisation des dossiers de travail
    calc_dir, results_dir, freq_dir = organize_workdir(workdir, args.name)
    print(f"Organisation du dossier de travail :")
    print(f"- Calculs intermediaires : {calc_dir}")
    print(f"- Resultats finaux : {results_dir}")
    print(f"- Calculs de frequences : {freq_dir}")

    print(f"Creation de la molecule '{args.name}' a partir du SMILES : {args.smiles}")
    try:
        mol = from_smiles(args.smiles)
        mol.properties.name = args.name
    except Exception as e:
        print(f"[X] Erreur lors de la creation de la molecule a partir du SMILES: {str(e)}")
        print("   Verifiez que le SMILES est valide et que RDKit est correctement installe.")
        finish()
        return 1

    # Afficher les parametres utilises
    print("\nParametres de calcul:")
    print(f"  Etape 2 : Optimisation      - E window: {args.e_window_opt} kJ/mol")
    print(f"  Etape 3 : Scoring           - Methode: {args.functional}/{args.basis}")
    print(f"  Etape 3 : Scoring           - E window: {args.e_window_score} kJ/mol")
    print(f"  Etape 3 : Scoring           - Dispersion: {args.dispersion}")
    print(f"  Etape 4 : Filtrage          - E window: {args.e_window_filter} kJ/mol")
    print(f"  Etape 4 : Filtrage          - RMSD: {args.rmsd_threshold} A")
    print(f"  Etape 5 : Frequences        - Methode: {args.freq_functional}/{args.freq_basis}")

    try:
        # Etape 1 : Generation
        generate_job = generate_conformers(mol, calc_dir)
        print_results(generate_job, temperature=TEMPERATURE, unit=UNIT)

        # Etape 2 : Optimisation
        optimize_job = optimize_conformers(generate_job, calc_dir, args.e_window_opt)
        print_results(optimize_job, temperature=TEMPERATURE, unit=UNIT)

        # Etape 3 : Scoring
        score_job = score_conformers(optimize_job, calc_dir, args.functional, args.basis, args.e_window_score, args.dispersion)
        print_results(score_job, temperature=TEMPERATURE, unit=UNIT)

        # Etape 4 : Filtrage
        filter_job = filter_conformers(score_job, calc_dir, args.e_window_filter, args.rmsd_threshold)
        energies, weights = print_results(filter_job, temperature=TEMPERATURE, unit=UNIT)
        
        # Sauvegarde du récapitulatif des conformères filtrés
        summary_file = os.path.join(calc_dir, "filter_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"# Conformeres apres filtrage pour {args.name}\n")
            f.write(f"# Criteres de filtrage: E window = {args.e_window_filter} kJ/mol, RMSD = {args.rmsd_threshold} A\n")
            f.write(f"# Temperature: {TEMPERATURE} K\n\n")
            f.write(f"{'#':>3s} {'Delta E [kJ/mol]':>16s} {'Poids Boltzmann':>16s}\n")
            
            for i, (energy, weight) in enumerate(zip(energies, weights)):
                f.write(f"{i+1:3d} {energy:16.4f} {weight:16.8f}\n")
                
            f.write(f"\nNombre total de conformeres: {len(energies)}\n")
        
        # Etape 5 : Verification des frequences
        valid_conformers = verify_frequencies(
            filter_job, args.name, results_dir, freq_dir, args.freq_functional, args.freq_basis
        )
        
        # Résumé final
        print(f"\nRecapitulatif final:")
        print(f"  Nombre total de conformeres identifies: {len(filter_job.results.get_conformers())}")
        print(f"  Nombre de conformeres valides (sans frequences imaginaires): {len(valid_conformers)}")
        print(f"\nLes conformeres valides sont disponibles dans: {results_dir}/")
        print(f"Un fichier de resume 'conformers_summary.txt' a ete cree dans ce dossier.")

    except RuntimeError as e:
        print(f"[X] Erreur : {e}")
        finish()
        return 1
    except Exception as e:
        import traceback
        print(f"[X] Erreur inattendue: {str(e)}")
        print(traceback.format_exc())
        finish()
        return 1
    finally:
        finish()
    
    return 0

if __name__ == "__main__":
    main()
