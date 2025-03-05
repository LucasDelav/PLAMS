#!/usr/bin/env amspython

import os
import argparse
import glob
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
    
    # Ajout des paramètres pour le niveau de théorie
    parser.add_argument("--functional", "-f", default="PBE0", 
                        help="Fonctionnelle à utiliser pour les calculs (défaut: PBE0)")
    
    # Ajout des paramètres pour le basis set
    parser.add_argument("--basis", "-b", default="TZP", 
                        help="Basis set à utiliser pour les calculs (défaut: TZP)")
    
    return parser.parse_args()

def init_workdir(prefix):
    """
    Initialize PLAMS with custom work directory
    """
    workdir = f"{prefix}_redox_workdir"
    workdir = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in workdir)
    init(folder=workdir)
    return workdir

def setup_adf_settings(task='GeometryOptimization', charge=0, spin_polarization=0, 
                       solvent="Acetonitrile", functional="PBE0", basis="TZP"):
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
    
    # Configuration de la fonctionnelle
    if functional in ["PBE0", "B3LYP", "BLYP", "BP86", "PBE", "revPBE", "OLYP", "OPBE"]:
        if functional in ["PBE0", "B3LYP"]:
            s.input.adf.XC.Hybrid = functional
        else:
            s.input.adf.XC.GGA = functional
    else:
        # Cas non standard, on définit directement la fonctionnelle
        s.input.adf.XC.Functional = functional
        
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

def optimize_neutral(mol, name, solvent="Acetonitrile", functional="PBE0", basis="TZP"):
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

def sp_reduced(job_neutral, name, solvent="Acetonitrile", functional="PBE0", basis="TZP"):
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

def optimize_reduced(job_sp, name, solvent="Acetonitrile", functional="PBE0", basis="TZP"):
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
    Collecter et résumer les résultats des calculs
    """
    print("\n" + "="*80)
    print("RÉSUMÉ DES CALCULS REDOX")
    print("="*80)
    
    print(f"{'Conformère':<15} {'E(neutre)':<15} {'E(réduit SP)':<15} {'E(réduit Opt)':<15} {'?E(red-neu)':<15}")
    print("-"*80)
    
    for name, jobs in jobs_data.items():
        if not all(jobs.values()):
            print(f"{name:<15} CALCUL INCOMPLET")
            continue
        
        # Énergies en hartree (unité par défaut d'ADF)
        e_neutral = jobs['neutral'].results.get_energy()
        e_red_sp = jobs['sp'].results.get_energy() 
        e_red_opt = jobs['opt'].results.get_energy()
        
        # Conversion en kJ/mol (1 hartree = 2625.5 kJ/mol)
        conversion = 2625.5
        delta_e = (e_red_opt - e_neutral) * conversion
        
        print(f"{name:<15} {e_neutral:<15.6f} {e_red_sp:<15.6f} {e_red_opt:<15.6f} {delta_e:<15.2f}")
    
    print("="*80)
    print("Énergies en hartree, ?E en kJ/mol")
    print("Note: ?E = E(réduit optimisé) - E(neutre optimisé)")

def export_molecules(jobs_data, prefix="redox_structures"):
    """
    Exporter les structures optimisées en XYZ dans le dossier de travail actuel de PLAMS
    """
    # Utiliser le dossier de travail actuel de PLAMS
    current_workdir = config.default_jobmanager.workdir
    output_dir = os.path.join(current_workdir, prefix)
    
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

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize PLAMS
    workdir = init_workdir(args.prefix)
    print(f"Dossier de travail créé: {workdir}")
    
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
    
    # Export optimized structures dans le dossier de travail actuel de PLAMS
    export_molecules(job_results)
    
    # Finalize PLAMS
    finish()

if __name__ == "__main__":
    main()
