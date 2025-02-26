#!/usr/bin/env amspython

import os
import argparse
import glob
import time
from datetime import timedelta
from scm.plams import *

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Script pour calculs redox de conformères"
    )
    parser.add_argument("input_dir", help="Dossier contenant les fichiers .xyz des conformères")
    parser.add_argument("--prefix", default="conf", help="Préfixe pour les fichiers de sortie")
    parser.add_argument("--solvent", default="Acetonitrile", help="Solvant pour les calculs")
    parser.add_argument("--skip-confirm", action="store_true", help="Ne pas demander confirmation avant de démarrer")
    return parser.parse_args()

def init_workdir(prefix):
    """
    Initialize PLAMS with custom work directory
    """
    workdir = f"{prefix}_redox_workdir"
    workdir = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in workdir)
    init(folder=workdir)
    return workdir

def estimate_calculation_time(n_conformers):
    """
    Estime le temps de calcul en fonction du nombre de conformères
    
    Ces estimations sont basées sur les temps réels observés pour la molécule L-DOPA.
    """
    # Temps réels observés en minutes pour chaque étape
    time_neutral_opt = 32    # ~32 minutes pour optimisation neutre
    time_reduced_sp = 43     # ~43 minutes pour calcul simple point réduit
    time_reduced_opt = 50    # ~50 minutes pour optimisation réduite
    
    # Temps total pour un conformère
    total_per_conformer = time_neutral_opt + time_reduced_sp + time_reduced_opt
    
    # Temps total estimé en minutes
    total_time_minutes = n_conformers * total_per_conformer
    
    # Conversion en heures et minutes pour l'affichage
    total_time_hours = total_time_minutes / 60
    
    return {
        'per_conformer_minutes': total_per_conformer,
        'total_minutes': total_time_minutes,
        'total_hours': total_time_hours,
        'neutral_opt': time_neutral_opt,
        'reduced_sp': time_reduced_sp,
        'reduced_opt': time_reduced_opt
    }

def display_time_estimate(n_conformers):
    """
    Affiche une estimation du temps de calcul basée sur des données réelles
    """
    estimate = estimate_calculation_time(n_conformers)
    
    print("\n" + "="*80)
    print("ESTIMATION DU TEMPS DE CALCUL (basée sur des données réelles pour L-DOPA)")
    print("="*80)
    print(f"Nombre de conformères: {n_conformers}")
    print(f"Temps estimé par conformère: {estimate['per_conformer_minutes']:.1f} minutes")
    print(f"  - Optimisation neutre: {estimate['neutral_opt']} minutes (intervalle typique: 21-40 min)")
    print(f"  - Simple point réduit: {estimate['reduced_sp']} minutes (intervalle typique: 27-52 min)")
    print(f"  - Optimisation réduite: {estimate['reduced_opt']} minutes (intervalle typique: 30-60 min)")
    
    # Calcul des bornes inférieures et supérieures
    lower_bound = n_conformers * 78  # 21+27+30 (bornes inférieures observées)
    upper_bound = n_conformers * 152  # 40+52+60 (bornes supérieures observées)
    
    print(f"Temps total estimé: {estimate['total_minutes']:.1f} minutes " +
          f"(environ {estimate['total_hours']:.1f} heures)")
    print(f"Intervalle probable: {lower_bound/60:.1f} à {upper_bound/60:.1f} heures")
    
    print("=" * 80)
    print("Note: Ces estimations sont basées sur les temps observés lors des exécutions précédentes.")
    print("Les variations peuvent être dues à la complexité des conformères individuels et à la charge")
    print("du système. Pour 11 conformères, le calcul a pris environ 22 heures.")
    print("=" * 80)
    
    return estimate

def setup_adf_settings(task='GeometryOptimization', charge=0, spin_polarization=0, solvent="Acetonitrile"):
    """
    Configure ADF settings object based on task and molecular state
    """
    s = Settings()
    s.input.ams.Task = task
    
    # Active le calcul des modes normaux
    s.input.ams.Properties.NormalModes = "Yes"
    
    # Configuration du moteur ADF
    s.input.adf.Basis.Type = "TZP"
    s.input.adf.Basis.Core = "None"
    s.input.adf.XC.Hybrid = "PBE0"
    s.input.adf.Relativity.Level = "None"
    
    # Paramètres pour les molécules chargées ou avec des électrons non appariés
    if charge != 0:
        s.input.ams.System.Charge = charge
    
    if spin_polarization > 0:
        s.input.adf.SpinPolarization = spin_polarization
        s.input.adf.Unrestricted = "Yes"
    
    # Configuration de la solvatation
    s.input.adf.Solvation.Surf = "Delley"
    s.input.adf.Solvation.Solv = f"name={solvent} cav0=0.0 cav1=0.0067639"
    s.input.adf.Solvation.Charged = "method=CONJ"
    s.input.adf.Solvation["C-Mat"] = "POT"
    s.input.adf.Solvation.SCF = "VAR ALL"
    s.input.adf.Solvation.CSMRSP = ""
    
    return s

def optimize_neutral(mol, name, solvent="Acetonitrile"):
    """
    Étape 1: Optimisation géométrique de la molécule neutre
    """
    print(f"\nÉtape 1: Optimisation géométrique de {name} (neutre)")
    settings = setup_adf_settings(task="GeometryOptimization", charge=0, solvent=solvent)
    
    job = AMSJob(settings=settings, name=f"{name}_neutre_opt", molecule=mol)
    job.run()
    
    if job.check():
        print(f"  Optimisation neutre réussie pour {name}")
        return job
    else:
        print(f"  ERREUR: Optimisation neutre échouée pour {name}")
        return None

def sp_reduced(job_neutral, name, solvent="Acetonitrile"):
    """
    Étape 2: Calcul en simple point de la molécule réduite (charge -1)
    """
    # Récupérer la molécule optimisée de l'étape 1
    mol_opt = job_neutral.results.get_main_molecule()
    
    print(f"\nÉtape 2: Calcul en simple point de {name} (réduit, charge -1)")
    settings = setup_adf_settings(task="SinglePoint", charge=-1, spin_polarization=1.0, solvent=solvent)
    
    job = AMSJob(settings=settings, name=f"{name}_reduit_sp", molecule=mol_opt)
    job.run()
    
    if job.check():
        print(f"  Calcul simple point réussi pour {name} (réduit)")
        return job
    else:
        print(f"  ERREUR: Calcul simple point échoué pour {name} (réduit)")
        return None

def optimize_reduced(job_sp, name, solvent="Acetonitrile"):
    """
    Étape 3: Optimisation géométrique de la molécule réduite
    """
    # Récupérer la molécule du calcul simple point
    mol = job_sp.results.get_main_molecule()
    
    print(f"\nÉtape 3: Optimisation géométrique de {name} (réduit, charge -1)")
    settings = setup_adf_settings(task="GeometryOptimization", charge=-1, spin_polarization=1.0, solvent=solvent)
    
    job = AMSJob(settings=settings, name=f"{name}_reduit_opt", molecule=mol)
    job.run()
    
    if job.check():
        print(f"  Optimisation réussie pour {name} (réduit)")
        return job
    else:
        print(f"  ERREUR: Optimisation échouée pour {name} (réduit)")
        return None

def collect_results(jobs_data):
    """
    Collecter et résumer les résultats des calculs
    """
    print("\n" + "="*80)
    print("RÉSUMÉ DES CALCULS REDOX")
    print("="*80)
    
    print(f"{'Conformère':<20} {'E(neutre)':<15} {'E(réduit SP)':<15} {'E(réduit Opt)':<15} {'?E(red-neu)':<15}")
    print("-"*80)
    
    # Stocker les énergies pour calculer les énergies relatives
    energies_neutral = []
    energies_reduced = []
    deltas = []
    
    for name, jobs in jobs_data.items():
        if not all(jobs.values()):
            print(f"{name:<20} CALCUL INCOMPLET")
            continue
        
        # Énergies en hartree (unité par défaut d'ADF)
        e_neutral = jobs['neutral'].results.get_energy()
        e_red_sp = jobs['sp'].results.get_energy() 
        e_red_opt = jobs['opt'].results.get_energy()
        
        # Conversion en kJ/mol (1 hartree = 2625.5 kJ/mol)
        conversion = 2625.5
        delta_e = (e_red_opt - e_neutral) * conversion
        
        energies_neutral.append((name, e_neutral))
        energies_reduced.append((name, e_red_opt))
        deltas.append((name, delta_e))
        
        print(f"{name:<20} {e_neutral:<15.6f} {e_red_sp:<15.6f} {e_red_opt:<15.6f} {delta_e:<15.2f}")
    
    # Trouver le conformère le plus stable pour chaque état
    if energies_neutral:
        best_neutral = min(energies_neutral, key=lambda x: x[1])
        best_reduced = min(energies_reduced, key=lambda x: x[1])
        best_delta = min(deltas, key=lambda x: x[1])
        
        print("="*80)
        print(f"Conformère neutre le plus stable: {best_neutral[0]} (E = {best_neutral[1]:.6f} hartree)")
        print(f"Conformère réduit le plus stable: {best_reduced[0]} (E = {best_reduced[1]:.6f} hartree)")
        print(f"Conformère avec l'énergie de réduction la plus favorable: {best_delta[0]} (?E = {best_delta[1]:.2f} kJ/mol)")
    
    print("="*80)
    print("Énergies en hartree, ?E en kJ/mol")
    print("Note: ?E = E(réduit optimisé) - E(neutre optimisé)")

def export_molecules(jobs_data, output_dir="redox_structures"):
    """
    Exporter les structures optimisées en XYZ
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\nExportation des structures optimisées dans {output_dir}/")
    
    for name, jobs in jobs_data.items():
        if not all(jobs.values()):
            continue
            
        # Exporter la molécule neutre optimisée
        mol_neutral = jobs['neutral'].results.get_main_molecule()
        neutral_path = os.path.join(output_dir, f"{name}_neutre_opt.xyz")
        mol_neutral.write(neutral_path)
        
        # Exporter la molécule réduite optimisée
        mol_reduced = jobs['opt'].results.get_main_molecule()
        reduced_path = os.path.join(output_dir, f"{name}_reduit_opt.xyz")
        mol_reduced.write(reduced_path)
        
        print(f"  {name}: structures exportées")
    
    print(f"Exportation terminée.")

def format_time(seconds):
    """Format time in seconds to hours, minutes, seconds"""
    return str(timedelta(seconds=round(seconds)))

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize PLAMS
    workdir = init_workdir(args.prefix)
    print(f"Dossier de travail créé: {workdir}")
    
    # Find all XYZ files in input directory
    xyz_files = glob.glob(os.path.join(args.input_dir, "*.xyz"))
    if not xyz_files:
        print(f"Aucun fichier XYZ trouvé dans {args.input_dir}")
        finish()
        return
    
    n_conformers = len(xyz_files)
    print(f"Trouvé {n_conformers} fichiers XYZ à traiter")
    
    # Display time estimate
    time_estimate = display_time_estimate(n_conformers)
    
    # Ask for confirmation before starting calculations
    if not args.skip_confirm:
        answer = input("\nDémarrer les calculs? (o/n): ").lower()
        if answer != 'o':
            print("Calculs annulés.")
            finish()
            return
    
    # Store all job results
    job_results = {}
    
    # Start time measurement
    start_time = time.time()
    
    # Process each XYZ file
    for i, xyz_file in enumerate(xyz_files):
        basename = os.path.basename(xyz_file)
        name = os.path.splitext(basename)[0]  # Remove extension
        
        # Calculate progress and estimated time remaining
        progress = i / n_conformers
        elapsed_time = time.time() - start_time
        if i > 0:  # avoid division by zero
            estimated_total = elapsed_time / i * n_conformers
            remaining = estimated_total - elapsed_time
        else:
            remaining = time_estimate['total_minutes'] * 60  # convert minutes to seconds
        
        print(f"\n{'='*80}")
        print(f"TRAITEMENT DE {basename} ({i+1}/{n_conformers}, {progress*100:.1f}%)")
        print(f"Temps écoulé: {format_time(elapsed_time)}, Temps restant estimé: {format_time(remaining)}")
        print(f"{'='*80}")
        
        # Read molecule from XYZ
        mol = Molecule(xyz_file)
        job_results[name] = {'neutral': None, 'sp': None, 'opt': None}
        
        # Record start time for this conformer
        conf_start_time = time.time()
        
        try:
            # Step 1: Optimize neutral
            job_neutral = optimize_neutral(mol, name, args.solvent)
            if not job_neutral:
                continue
            job_results[name]['neutral'] = job_neutral
            
            # Temps écoulé pour l'étape 1
            step1_time = time.time() - conf_start_time
            print(f"  Temps pour l'étape 1: {format_time(step1_time)}")
            
            # Step 2: Single point reduced
            step2_start = time.time()
            job_sp = sp_reduced(job_neutral, name, args.solvent)
            if not job_sp:
                continue
            job_results[name]['sp'] = job_sp
            
            # Temps écoulé pour l'étape 2
            step2_time = time.time() - step2_start
            print(f"  Temps pour l'étape 2: {format_time(step2_time)}")
            
            # Step 3: Optimize reduced
            step3_start = time.time()
            job_opt = optimize_reduced(job_sp, name, args.solvent)
            if not job_opt:
                continue
            job_results[name]['opt'] = job_opt
            
            # Temps écoulé pour l'étape 3
            step3_time = time.time() - step3_start
            print(f"  Temps pour l'étape 3: {format_time(step3_time)}")
            
            # Calculate actual time for this conformer
            conf_time = time.time() - conf_start_time
            print(f"\nTemps réel pour ce conformère: {format_time(conf_time)}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {basename}: {str(e)}")
    
    # Total elapsed time
    total_elapsed = time.time() - start_time
    print(f"\nTemps total de calcul: {format_time(total_elapsed)}")
    print(f"Temps moyen par conformère: {format_time(total_elapsed/n_conformers)}")
    
    # Collect and print results
    collect_results(job_results)
    
    # Export optimized structures
    export_molecules(job_results)
    
    # Finalize PLAMS
    finish()

if __name__ == "__main__":
    main()
