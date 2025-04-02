#!/usr/bin/env amspython

import os
import argparse
import glob
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

def collect_results(jobs_data):
    """
    Collecter et résumer les résultats des calculs, avec énergies en kJ/mol
    """
    print("\n" + "="*80)
    print("RÉSUMÉ DES CALCULS REDOX")
    print("="*80)
    
    print(f"{'Conformère':<15} {'E(neutre)':<15} {'E(réduit SP)':<15} {'E(réduit Opt)':<15} {'ΔE(red-neu)':<15}")
    print("-"*80)
    
    for name, jobs in jobs_data.items():
        if not all(jobs.values()):
            print(f"{name:<15} CALCUL INCOMPLET")
            continue
        
        # Énergies en hartree (unité par défaut d'ADF)
        e_neutral = jobs['neutral'].results.get_energy()
        e_red_sp = jobs['sp'].results.get_energy() 
        e_red_opt = jobs['opt'].results.get_energy()
        
        # Conversion directe en kJ/mol (1 hartree = 2625.5 kJ/mol)
        conversion = 2625.5
        e_neutral_kj = e_neutral * conversion
        e_red_sp_kj = e_red_sp * conversion
        e_red_opt_kj = e_red_opt * conversion
        delta_e = e_red_opt_kj - e_neutral_kj
        
        print(f"{name:<15} {e_neutral_kj:<15.2f} {e_red_sp_kj:<15.2f} {e_red_opt_kj:<15.2f} {delta_e:<15.2f}")
    
    print("="*80)
    print("Toutes les énergies sont exprimées en kJ/mol")
    print("Note: ΔE = E(réduit optimisé) - E(neutre optimisé)")

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
    
    # Collect and print results
    collect_results(job_results)
    
    # Export output files dans le dossier de travail actuel de PLAMS
    export_molecules(job_results)
    
    # Finalize PLAMS
    finish()

if __name__ == "__main__":
    main()
