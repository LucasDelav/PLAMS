#!/usr/bin/env amspython

import os
import argparse
import numpy as np
import shutil
import sys
import matplotlib.pyplot as plt
from scm.plams import *
from scm.conformers import ConformersJob
from scm.plams.core.settings import Settings

TEMPERATURE = 298
R = 8.314
UNIT = "kJ/mol"

LDA_FUNCTIONALS = ["VWN", "XALPHA", "Xonly", "Stoll"]
GGA_FUNCTIONALS = ["PBE", "RPBE", "revPBE", "PBEsol", "BLYP", "BP86", "PW91", "mPW", "OLYP", "OPBE", 
                  "KT1", "KT2", "BEE", "BJLDA", "BJPBE", "BJGGA", "S12G", "LB94", "mPBE", "B3LYPgauss"]
METAGGA_FUNCTIONALS = ["M06L", "TPSS", "revTPSS", "MVS", "SCAN", "revSCAN", "r2SCAN", "tb-mBJ"]
HYBRID_FUNCTIONALS = ["B3LYP", "B3LYP*", "B1LYP", "O3LYP", "X3LYP", "BHandH", "BHandHLYP", 
                     "B1PW91", "MPW1PW", "MPW1K", "PBE0", "OPBE0", "TPSSh", "M06", "M06-2X", "S12H"]
METAHYBRID_FUNCTIONALS = ["M08-HX", "M08-SO", "M11", "TPSSH", "PW6B95", "MPW1B95", "MPWB1K", 
                         "PWB6K", "M06-HF", "BMK"]

ALL_FUNCTIONALS = ['HF'] + LDA_FUNCTIONALS + GGA_FUNCTIONALS + METAGGA_FUNCTIONALS + HYBRID_FUNCTIONALS + METAHYBRID_FUNCTIONALS

def organize_workdir(workdir, molecule_name):
    calc_dir = os.path.join(workdir, "calculations")
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
    
    results_dir = os.path.join(workdir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    return calc_dir, results_dir

def get_actual_workdir_path(base_name):
    if hasattr(config, 'default_jobmanager') and config.default_jobmanager:
        return os.path.abspath(config.default_jobmanager.workdir)
    else:
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script d'analyse des conformers pour une molecule donnee.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("smiles", help="SMILES de la molecule", type=str)
    parser.add_argument(
        "name", help="Nom de la molecule", nargs="?", default="Molecule", type=str
    )
    
    parser.add_argument("--e-window-opt", type=float, default=10,
                        help="Fenetre energetique (kJ/mol) pour l'etape d'optimisation")
    
    parser.add_argument("--functional", choices=ALL_FUNCTIONALS, default="PBE0",
                        help="Fonctionnelle pour le scoring")
    parser.add_argument("--basis", choices=["SZ", "DZ", "DZP", "TZP", "TZ2P", "QZ4P"], default="TZP",
                        help="Base utilisee pour le scoring")
    parser.add_argument("--e-window-score", type=float, default=5,
                        help="Fenetre energetique (kJ/mol) pour l'etape de scoring")
    parser.add_argument("--dispersion", choices=["None", "GRIMME3", "GRIMME4", "UFF", "GRIMME3 BJDAMP"], default="GRIMME3",
                        help="Correction de dispersion pour le scoring")
    
    parser.add_argument("--e-window-filter", type=float, default=2,
                        help="Fenetre energetique (kJ/mol) pour l'etape de filtrage")
    parser.add_argument("--rmsd-threshold", type=float, default=1.0,
                        help="Seuil RMSD (A) pour considerer deux conformeres comme identiques")
    
    parser.add_argument("--freq-functional", choices=ALL_FUNCTIONALS, default=None,
                        help="Fonctionnelle pour le calcul des frequences (par defaut: meme que scoring)")
    parser.add_argument("--freq-basis", choices=["SZ", "DZ", "DZP", "TZP", "TZ2P"], default="DZ",
                        help="Base utilisee pour le calcul des frequences")
    
    return parser.parse_args()

def init_with_custom_folder(name):
    folder_name = f"{name}_workdir"
    folder_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in folder_name)
    
    init(folder=folder_name)
    
    actual_workdir = get_actual_workdir_path(folder_name)
    
    if actual_workdir is None:
        actual_workdir = folder_name
    
    return actual_workdir

def boltzmann_weights(energies, temperature):
    beta = 1 / (R / 1000 * temperature)
    exponentials = np.exp(-beta * np.array(energies))
    partition_function = sum(exponentials)
    weights = exponentials / partition_function
    return weights, partition_function

def print_results(job, temperature=298, unit=UNIT):
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

def configure_functional(s, functional):
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

def generate_conformers(molecule, calc_dir):
    print("\n[Etape 1] Generation des conformers avec RDKit...")
    s = Settings()
    s.input.ams.Task = "Generate"
    s.input.ams.Generator.Method = "RDKit"
    s.input.ams.Generator.RDKit.InitialNConformers = 1000
    s.input.ForceField.Type = "UFF"
    
    job_dir = os.path.join(calc_dir, "generate_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    job = ConformersJob(molecule=molecule, settings=s, name="generate_conformers")
    
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError("La generation des conformers a echoue.")
    return job

def optimize_conformers(previous_job, calc_dir, e_window=10):
    print(f"\n[Etape 2] Optimisation des conformers geometriques avec DFTB3 (fenetre d'energie = {e_window} kJ/mol)...")
    s = Settings()
    s.input.ams.Task = "Optimize"
    rkf_path = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputConformersSet = rkf_path
    s.input.ams.InputMaxEnergy = e_window
    s.input.dftb.Model = "DFTB3"
    s.input.dftb.ResourcesDir = "DFTB.org/3ob-3-1"
    
    job_dir = os.path.join(calc_dir, "optimize_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    job = ConformersJob(settings=s, name="optimize_conformers")
    
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError("L'optimisation des conformers a echoue.")
    return job

def score_conformers(previous_job, calc_dir, functional="PBE0", basis="TZP", e_window=5, dispersion="GRIMME3"):
    print(f"\n[Etape 3] Calcul des energies des conformers avec {functional}/{basis} (fenetre d'energie = {e_window} kJ/mol)...")
    s = Settings()
    s.input.ams.Task = "Score"
    rkf_path = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputConformersSet = rkf_path
    s.input.ams.InputMaxEnergy = e_window
    
    if dispersion != "None":
        if dispersion == "GRIMME3 BJDAMP":
            s.input.adf.XC.DISPERSION = "GRIMME3"
            s.input.adf.XC.DISPERSION += " BJDAMP"
        else:
            s.input.adf.XC.DISPERSION = dispersion
    
    configure_functional(s, functional)
    
    s.input.adf.BASIS.Type = basis
    
    if functional in METAGGA_FUNCTIONALS or functional in METAHYBRID_FUNCTIONALS:
        s.input.adf.NumericalQuality = "Good"
    
    job_dir = os.path.join(calc_dir, "score_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    job = ConformersJob(settings=s, name="score_conformers")
    
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError(f"Le scoring des conformers avec {functional}/{basis} a echoue.")
    return job

def filter_conformers(previous_job, calc_dir, e_window=2, rmsd_threshold=1.0):
    print(f"\n[Etape 4] Filtrage des conformers (fenetre d'energie = {e_window} kJ/mol, RMSD = {rmsd_threshold} A)...")
    s = Settings()
    s.input.ams.Task = "Filter"
    rkf_path = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputConformersSet = rkf_path
    s.input.ams.InputMaxEnergy = e_window
    s.input.ams.Equivalence.CREST.RMSDThreshold = rmsd_threshold
    
    job_dir = os.path.join(calc_dir, "filter_conformers")
    os.makedirs(job_dir, exist_ok=True)
    
    job = ConformersJob(settings=s, name="filter_conformers")
    
    with DefaultJobManager(job_dir):
        result = job.run()
    
    if not job.results:
        raise RuntimeError("Le filtrage des conformers a echoue.")
    return job

def verify_frequencies(previous_job, molecule_name, results_dir, calc_dir, freq_functional="PBE0", freq_basis="DZ"):
    print(f"\n[Etape 5] Optimisation de géométrie et vérification des fréquences avec {freq_functional}/{freq_basis}...")
    
    # Récupérer les conformères de l'étape précédente
    input_conformers = previous_job.results.get_conformers()
    num_conformers = len(input_conformers)
    
    print(f"  Analyse de {num_conformers} conformères...")
    
    # Créer le dossier pour les calculs de fréquence
    freq_dir = os.path.join(calc_dir, "frequency_calculations")
    os.makedirs(freq_dir, exist_ok=True)
    
    valid_conformers = []
    valid_indices = []
    
    # Traiter chaque conformère individuellement
    for i, conf in enumerate(input_conformers):
        print(f"  Traitement du conformère {i+1}/{num_conformers}...")
        
        # Configurer le calcul d'optimisation + fréquences
        s = Settings()
        s.input.ams.Task = "GeometryOptimization"
        s.input.ams.Properties.NormalModes = "Yes"
        
        # Configuration de la fonctionnelle et de la base
        configure_functional(s, freq_functional)
        s.input.adf.basis.Type = freq_basis
        
        # Qualité de calcul pour certaines fonctionnelles
        if freq_functional in METAGGA_FUNCTIONALS or freq_functional in METAHYBRID_FUNCTIONALS:
            s.input.adf.NumericalQuality = "Good"
        
        # Créer et exécuter le job directement dans le dossier conf_N
        job_name = f"conf_{i+1}"
        mol_job = AMSJob(molecule=conf, settings=s, name=job_name)
        
        try:
            # Utiliser DefaultJobManager avec le dossier conf_N comme répertoire de travail
            with DefaultJobManager(job_dir=freq_dir):
                mol_result = mol_job.run()
            
            # Vérifier si le calcul a réussi
            if mol_result.ok():
                # Lire le fichier de sortie pour extraire les fréquences
                output_file = os.path.join(freq_dir, job_name, f"{job_name}.out")
                
                if os.path.exists(output_file):
                    # Extraire les fréquences en utilisant un pattern actualisé
                    import re
                    with open(output_file, 'r') as f:
                        content = f.read()
                    
                    # Rechercher la section de fréquences des modes normaux
                    freq_section = re.search(r'-+\s*Normal Mode Frequencies\s*-+\s*.*?(?=\s*Zero-point|$)', 
                                            content, re.DOTALL)
                    
                    if freq_section:
                        # Extraire les lignes contenant les fréquences (en ignorant l'en-tête)
                        freq_lines = freq_section.group(0).strip().split('\n')
                        # Ignorer les lignes d'en-tête (généralement 3-4 lignes)
                        data_lines = [line for line in freq_lines if re.match(r'\s*\d+\s+[-+]?\d+\.\d+', line)]
                        
                        if data_lines:
                            # Extraire les fréquences de chaque ligne
                            frequencies = []
                            for line in data_lines:
                                # Matcher la deuxième colonne qui contient la fréquence
                                match = re.search(r'\s*\d+\s+([-+]?\d+\.\d+)', line)
                                if match:
                                    frequencies.append(float(match.group(1)))
                            
                            # Vérifier s'il y a des fréquences imaginaires
                            has_imaginary = any(freq < 0 for freq in frequencies)
                            
                            if has_imaginary:
                                neg_freqs = [f for f in frequencies if f < 0]
                                print(f"  [X] Conformère {i+1} : fréquences imaginaires détectées")
                                print(f"     Fréquences imaginaires : {neg_freqs} cm-1")
                            else:
                                print(f"  [V] Conformère {i+1} : toutes les fréquences sont positives")
                                # Récupérer la géométrie optimisée
                                opt_mol = mol_result.get_main_molecule()
                                png_file = os.path.join(results_dir, f"{job_name}.png")
                                plot_molecule(opt_mol)
                                plt.tight_layout()
                                plt.savefig(png_file)
                                
                                # Enregistrer le conformère valide
                                valid_conformers.append(opt_mol)
                                valid_indices.append(i)
                                
                                # Sauvegarder la géométrie au format XYZ
                                xyz_file = os.path.join(results_dir, f"{molecule_name}_conf_{i+1}.xyz")
                                opt_mol.write(xyz_file)
                        else:
                            print(f"  [X] Aucune donnée de fréquence extraite pour le conformère {i+1}")
                    else:
                        print(f"  [X] Section des fréquences non trouvée pour le conformère {i+1}")
                else:
                    print(f"  [X] Fichier de sortie introuvable pour le conformère {i+1}")
            else:
                print(f"  [X] Le calcul du conformère {i+1} a échoué")
        except Exception as e:
            print(f"  [X] Erreur lors de l'analyse du conformère {i+1}: {str(e)}")
    
    # Créer un fichier de résumé
    summary_file = os.path.join(results_dir, "conformers_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"# Conformères valides pour {molecule_name}\n")
        f.write(f"# Méthode de calcul des fréquences: {freq_functional}/{freq_basis}\n\n")
        
        if valid_conformers:
            f.write(f"{'#':>3s} {'Index original':>16s}\n")
            
            for idx, i in enumerate(valid_indices):
                f.write(f"{idx+1:3d} {i+1:16d}\n")
            
            f.write(f"\nNombre de conformères valides: {len(valid_conformers)}\n")
        else:
            f.write("Aucun conformère valide trouvé (tous ont des fréquences imaginaires)\n")
    
    return valid_conformers

class DefaultJobManager:
    def __init__(self, job_dir):
        self.job_dir = job_dir
        self.original_jobmanager = None
        self.original_workdir = None
    
    def __enter__(self):
        if hasattr(config, 'default_jobmanager'):
            self.original_jobmanager = config.default_jobmanager
            if hasattr(config.default_jobmanager, 'workdir'):
                self.original_workdir = config.default_jobmanager.workdir
        
        os.makedirs(self.job_dir, exist_ok=True)
        
        if hasattr(config, 'default_jobmanager') and config.default_jobmanager:
            config.default_jobmanager.workdir = self.job_dir
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_workdir and hasattr(config, 'default_jobmanager') and config.default_jobmanager:
            config.default_jobmanager.workdir = self.original_workdir

def main():
    args = parse_arguments()
    
    if args.freq_functional is None:
        args.freq_functional = args.functional
    
    workdir = init_with_custom_folder(args.name)
    print(f"\nDossier de travail cree: {workdir}")
    
    calc_dir, results_dir = organize_workdir(workdir, args.name)
    print(f"Organisation du dossier de travail :")
    print(f"- Calculs intermediaires : {calc_dir}")
    print(f"- Resultats finaux : {results_dir}")

    print(f"Creation de la molecule '{args.name}' a partir du SMILES : {args.smiles}")
    try:
        mol = from_smiles(args.smiles)
        mol.properties.name = args.name
    except Exception as e:
        print(f"[X] Erreur lors de la creation de la molecule a partir du SMILES: {str(e)}")
        print("   Verifiez que le SMILES est valide et que RDKit est correctement installe.")
        finish()
        return 1

    print("\nParametres de calcul:")
    print(f"  Etape 2 : Optimisation      - E window: {args.e_window_opt} kJ/mol")
    print(f"  Etape 3 : Scoring           - Methode: {args.functional}/{args.basis}")
    print(f"  Etape 3 : Scoring           - E window: {args.e_window_score} kJ/mol")
    print(f"  Etape 3 : Scoring           - Dispersion: {args.dispersion}")
    print(f"  Etape 4 : Filtrage          - E window: {args.e_window_filter} kJ/mol")
    print(f"  Etape 4 : Filtrage          - RMSD: {args.rmsd_threshold} A")
    print(f"  Etape 5 : Frequences        - Methode: {args.freq_functional}/{args.freq_basis}")

    try:
        generate_job = generate_conformers(mol, calc_dir)
        print_results(generate_job, temperature=TEMPERATURE, unit=UNIT)

        optimize_job = optimize_conformers(generate_job, calc_dir, args.e_window_opt)
        print_results(optimize_job, temperature=TEMPERATURE, unit=UNIT)

        score_job = score_conformers(optimize_job, calc_dir, args.functional, args.basis, args.e_window_score, args.dispersion)
        print_results(score_job, temperature=TEMPERATURE, unit=UNIT)

        filter_job = filter_conformers(score_job, calc_dir, args.e_window_filter, args.rmsd_threshold)
        energies, weights = print_results(filter_job, temperature=TEMPERATURE, unit=UNIT)
        
        summary_file = os.path.join(calc_dir, "filter_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"# Conformeres apres filtrage pour {args.name}\n")
            f.write(f"# Criteres de filtrage: E window = {args.e_window_filter} kJ/mol, RMSD = {args.rmsd_threshold} A\n")
            f.write(f"# Temperature: {TEMPERATURE} K\n\n")
            f.write(f"{'#':>3s} {'Delta E [kJ/mol]':>16s} {'Poids Boltzmann':>16s}\n")
            
            for i, (energy, weight) in enumerate(zip(energies, weights)):
                f.write(f"{i+1:3d} {energy:16.4f} {weight:16.8f}\n")
                
            f.write(f"\nNombre total de conformeres: {len(energies)}\n")
        
        valid_conformers = verify_frequencies(
            filter_job, args.name, results_dir, calc_dir, args.freq_functional, args.freq_basis
        )
        
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
