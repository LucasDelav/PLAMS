#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
from scm.plams import Molecule, Settings, init, finish, AMSJob, JobRunner, config

def parse_arguments():
    """
    Fonction qui gère les arguments donnés par l'utilisateur.
    
    Returns:
        argparse.Namespace: Les arguments parsés.
    """
    parser = argparse.ArgumentParser(description='Programme QTAIM avec PLAMS')
    parser.add_argument('--dir', type=str, required=True, 
                        help='Répertoire de la molécule (ex: EtOH_workdir/redox/)')
    
    return parser.parse_args()

def extract_coordinates(base_path):
    """
    Extrait les coordonnées à partir des fichiers .xyz pour tous les conformères
    dans l'état spécifié.
    
    Args:
        base_path (str): Chemin de base vers le répertoire des conformères.
    Returns:
        list: Liste de tuples (nom_conformère, molécule_PLAMS)
    """
    molecules = []
    
    pattern = os.path.join(base_path, 'redox', "*_conf_*_neutre*/output.xyz")
    xyz_files = glob.glob(pattern)
    
    if not xyz_files:
        print(f"Aucun fichier .xyz trouvé dans {pattern}")
        return molecules
    
    for xyz_path in xyz_files:
        # Extraire le nom du conformère à partir du chemin
        conf_dir = os.path.dirname(xyz_path)
        conf_name = os.path.basename(conf_dir)
        
        try:
            # Charger la molécule avec PLAMS
            molecule = Molecule(xyz_path)
            molecules.append((conf_name, molecule))
            print(f"Coordonnées extraites pour {conf_name}")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {xyz_path}: {e}")
    
    return molecules

def setup_qtaim_settings(molecule, charge=0, spinpol=0):
    """
    Configure les paramètres pour le calcul QTAIM.
    
    Args:
        molecule (Molecule): La molécule PLAMS.
        charge (int): La charge de la molécule.
        spinpol (int): La polarisation de spin.
    
    Returns:
        Settings: Les paramètres PLAMS pour le calcul QTAIM.
    """
    settings = Settings()
    
    # Paramètres ADF
    settings.input.ams.Task = 'SinglePoint'
    settings.input.adf.basis.type = 'TZP'
    settings.input.adf.basis.core = 'None'
    settings.input.adf.xc.hybrid = 'PBE0'
    settings.input.adf.relativity.level = 'None'
    settings.input.adf.solvation.solv = 'name=Acetonitrile'
    settings.input.adf.iqa.enabled = 'Yes'
    settings.input.adf.qtaim.enabled = 'Yes'
    settings.input.adf.qtaim.analysislevel = 'Full'
    settings.input.adf.qtaim.source = 'Yes'
    
    # Charge et spin
    settings.input.ams.system.charge = charge
    
    if spinpol > 0:
        settings.input.adf.unrestricted = 'Yes'
        settings.input.adf.SpinPolarization = spinpol
    
    return settings

def run_qtaim_calculation(conf_name, molecule, settings, output_dir, state):
    """
    Initialise et lance les calculs QTAIM.
    
    Args:
        conf_name (str): Nom du conformère.
        molecule (Molecule): La molécule PLAMS.
        settings (Settings): Paramètres du calcul.
        output_dir (str): Répertoire de sortie.
        state (str): État de la molécule ('neutre', 'oxidized', 'reduced').
    
    Returns:
        AMSJob: Le job AMS exécuté.
    """
    # Créer un nom pour le job qui inclut l'état et le nom du conformère
    job_name = f"{conf_name}_{state}_qtaim"
    
    # Définir le répertoire de travail spécifique pour ce job
    config.default_jobmanager.workdir = os.path.join(output_dir, state)
    
    # Créer et exécuter le job
    job = AMSJob(molecule=molecule, settings=settings, name=job_name)
    job.run(jobrunner=JobRunner(parallel=True, maxjobs=config.default_jobrunner.maxjobs))
    
    # Log le résultat
    if job.ok():
        print(f"Le calcul QTAIM pour {conf_name} ({state}) a réussi.")
    else:
        print(f"Le calcul QTAIM pour {conf_name} ({state}) a échoué.")
    
    return job

def process_conformers(conformers, charge, spinpol, output_dir, state):
    """
    Traite un ensemble de conformères avec des paramètres donnés.
    
    Args:
        conformers (list): Liste de tuples (nom_conformère, molécule_PLAMS).
        charge (int): Charge à appliquer.
        spinpol (int): Polarisation de spin à appliquer.
        output_dir (str): Répertoire de sortie.
        state (str): État de la molécule ('neutre', 'oxidized', 'reduced').
    """
    print(f"\n--- Traitement des conformères {state} ---")
    
    for conf_name, molecule in conformers:
        settings = setup_qtaim_settings(molecule, charge=charge, spinpol=spinpol)
        real_conf_name = conf_name.split('_neutre')
        run_qtaim_calculation(real_conf_name[0], molecule, settings, output_dir, state)

def main():
    """
    Fonction principale qui orchestre le workflow complet.
    """
    # Analyser les arguments
    args = parse_arguments()
    base_path = args.dir
    output_dir = os.path.join(os.path.dirname(base_path), 'qtaims')

    # Initialiser PLAMS avec le répertoire qtaims comme base
    init(folder=output_dir)
 
    # Créer les sous-répertoires pour chaque état
    for state in ['neutre', 'oxidized', 'reduced']:
        os.makedirs(os.path.join(output_dir, state), exist_ok=True)
   
    # Extraire les coordonnées des conformères neutres (utilisées pour tous les états)
    neutral_conformers = extract_coordinates(base_path)
    if not neutral_conformers:
        print("Aucun conformère trouvé. Arrêt du programme.")
        finish()
        return
    
    # 1. Traiter les conformères neutres
    process_conformers(neutral_conformers, charge=0, spinpol=0, output_dir=output_dir, state='neutre')
    
    # 2. Traiter les conformères oxydés (en utilisant les structures neutres avec charge +1)
    process_conformers(neutral_conformers, charge=1, spinpol=1, output_dir=output_dir, state='oxidized')
    
    # 3. Traiter les conformères réduits (en utilisant les structures neutres avec charge -1)
    process_conformers(neutral_conformers, charge=-1, spinpol=1, output_dir=output_dir, state='reduced')
    
    # Finaliser PLAMS
    finish()

if __name__ == "__main__":
    main()

