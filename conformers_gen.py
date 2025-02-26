#!/usr/bin/env amspython

import os
import argparse
import numpy as np
from scm.plams import *
from scm.conformers import ConformersJob
from scm.plams.core.settings import Settings

# CONSTANTES GLOBALES
TEMPERATURE = 298  # Température en Kelvin
R = 8.314  # Constante des gaz parfaits en J/(mol*K)
UNIT = "kJ/mol"  # Unité d'énergie par défaut pour le script

# ------------------------------------------------------------
# Gestion des arguments en ligne de commande
# ------------------------------------------------------------
def parse_arguments():
    """
    Définit et analyse les arguments en ligne de commande (SMILES + nom de la molécule).
    """
    parser = argparse.ArgumentParser(
        description="Script d'analyse des conformers pour une molécule donnée."
    )
    parser.add_argument("smiles", help="SMILES de la molécule", type=str)
    parser.add_argument(
        "name", help="Nom de la molécule", nargs="?", default="Molecule", type=str
    )
    return parser.parse_args()

# ------------------------------------------------------------
# Initialisation de PLAMS avec un dossier personnalisé
# ------------------------------------------------------------
def init_with_custom_folder(name):
    """
    Initialise PLAMS avec un nom de dossier personnalisé basé sur le nom de la molécule.
    """
    # Créer un nom de dossier valide en remplaçant les caractères problématiques
    folder_name = f"{name}_workdir"
    folder_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in folder_name)
    
    init(folder=folder_name)
    return folder_name

# ------------------------------------------------------------
# Exportation des conformères en fichiers SDF individuels
# ------------------------------------------------------------
def export_conformers_to_sdf(job, output_dir=None):
    """
    Exporte chaque conformère en fichier SDF/XYZ individuel
    
    Args:
        job (ConformersJob): Job contenant les conformères
        output_dir (str, optional): Dossier où sauvegarder les fichiers
        
    Returns:
        str: Chemin du dossier d'exportation
    """
    if output_dir is None:
        output_dir = "exported_conformers"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Obtenir les conformères
    conformers = job.results.get_conformers()
    base_name = job.name
    
    print(f"\nExportation de {len(conformers)} conformères dans {output_dir}/")
    
    for i, conf in enumerate(conformers):
        # Essayer d'abord le format XYZ qui est généralement bien supporté
        filename = f"{base_name}_conf_{i+1}.xyz"
        filepath = os.path.join(output_dir, filename)
        
        try:
            conf.write(filepath)
            print(f"  Conformère {i+1} sauvegardé : {filename}")
        except Exception as e:
            print(f"  Erreur lors de la sauvegarde du conformère {i+1}: {str(e)}")
    
    return output_dir

# ------------------------------------------------------------
# Calcul des poids de Boltzmann
# ------------------------------------------------------------
def boltzmann_weights(energies, temperature):
    """
    Calcule les poids de Boltzmann pour un ensemble d'énergies.

    Args:
        energies (list): Liste des énergies des conformers (en kJ/mol).
        temperature (float): Température (en Kelvin).

    Returns:
        tuple: Poids normalisés et fonction de partition.
    """
    beta = 1 / (R / 1000 * temperature)  # R divisé par 1000 pour convertir en kJ/mol
    exponentials = np.exp(-beta * np.array(energies))
    partition_function = sum(exponentials)
    weights = exponentials / partition_function
    return weights, partition_function

# ------------------------------------------------------------
# Impression des résultats
# ------------------------------------------------------------
def print_results(job: ConformersJob, temperature=298, unit=UNIT):
    """
    Affiche les résultats en termes d'énergies relatives et de poids de Boltzmann.

    Args:
        job (ConformersJob): Tâche contenant les résultats à traiter.
    """
    energies = job.results.get_relative_energies(unit)
    weights, Z = boltzmann_weights(energies, temperature)

    print(f"\nRésultats (Température = {temperature} K) :")
    print(f'{"#":>4s} {"?E [{}]".format(unit):>16s} {"Poids Boltzmann":>16s}')

    for i, (energy, weight) in enumerate(zip(energies, weights)):
        print(f"{i+1:4d} {energy:16.2f} {weight:16.8f}")

    print("\nRésumé :")
    print(f"Nombre total de conformers : {len(energies)}")
    print(f"Fonction de partition (Z) = {Z:.8f}")

    return energies, weights

# ------------------------------------------------------------
# Étape 1 : Génération des conformers
# ------------------------------------------------------------
def generate_conformers(molecule):
    print("\n[Étape 1] Génération des conformers avec RDKit...")
    s = Settings()
    s.input.ams.Task = "Generate"
    s.input.ams.Generator.Method = "RDKit"
    s.input.ams.Generator.RDKit.InitialNConformers = 1000
    s.input.ForceField.Type = "UFF"
    generate_job = ConformersJob(name="generate_conformers", molecule=molecule, settings=s)
    generate_job.run()
    if not generate_job.results:
        raise RuntimeError("La génération des conformers a échoué.")
    return generate_job

# ------------------------------------------------------------
# Étape 2 : Optimisation des conformers
# ------------------------------------------------------------
def optimize_conformers(previous_job):
    print("\n[Étape 2] Optimisation des conformers géométriques avec DFTB3...")
    s = Settings()
    s.input.ams.Task = "Optimize"
    s.input.ams.InputConformersSet = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputMaxEnergy = 20  # Fenêtre énergétique de 20 kJ/mol
    s.input.dftb.Model = "DFTB3"
    s.input.dftb.ResourcesDir = "DFTB.org/3ob-3-1"
    optimize_job = ConformersJob(name="optimize_conformers", settings=s)
    optimize_job.run()
    if not optimize_job.results:
        raise RuntimeError("L'optimisation des conformers a échoué.")
    return optimize_job

# ------------------------------------------------------------
# Étape 3 : Scoring des conformers
# ------------------------------------------------------------
def score_conformers(previous_job):
    print("\n[Étape 3] Calcul des énergies des conformers avec ADF (PBE0)...")
    s = Settings()
    s.input.ams.Task = "Score"
    s.input.ams.InputConformersSet = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputMaxEnergy = 8  # Fenêtre énergétique de 8 kJ/mol
    s.input.adf.XC.DISPERSION = "GRIMME3"
    s.input.adf.XC.Hybrid = "PBE0"
    s.input.adf.BASIS.Type = "TZP"
    score_job = ConformersJob(name="score_conformers", settings=s)
    score_job.run()
    if not score_job.results:
        raise RuntimeError("Le scoring des conformers a échoué.")
    return score_job

# ------------------------------------------------------------
# Étape 4 : Filtrage final
# ------------------------------------------------------------
def filter_conformers(previous_job):
    print("\n[Étape 4] Filtrage des conformers en utilisant RMSDThreshold...")
    s = Settings()
    s.input.ams.Task = "Filter"
    s.input.ams.InputConformersSet = os.path.abspath(previous_job.results.rkfpath())
    s.input.ams.InputMaxEnergy = 4  # Fenêtre énergétique de 4 kJ/mol
    s.input.ams.Equivalence.CREST.RMSDThreshold = 0.5
    filter_job = ConformersJob(name="filter_conformers", settings=s)
    filter_job.run()
    if not filter_job.results:
        raise RuntimeError("Le filtrage des conformers a échoué.")
    return filter_job

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
def main():
    # Analyse des arguments
    args = parse_arguments()
    
    # Initialisation avec un dossier de travail personnalisé
    workdir = init_with_custom_folder(args.name)
    print(f"\nDossier de travail créé: {workdir}")

    print(f"Création de la molécule '{args.name}' à partir du SMILES : {args.smiles}")
    mol = from_smiles(args.smiles)
    mol.properties.name = args.name

    try:
        # Étape 1 : Génération
        generate_job = generate_conformers(mol)
        print_results(generate_job, temperature=TEMPERATURE, unit=UNIT)

        # Étape 2 : Optimisation
        optimize_job = optimize_conformers(generate_job)
        print_results(optimize_job, temperature=TEMPERATURE, unit=UNIT)

        # Étape 3 : Scoring
        score_job = score_conformers(optimize_job)
        print_results(score_job, temperature=TEMPERATURE, unit=UNIT)

        # Étape 4 : Filtrage
        filter_job = filter_conformers(score_job)
        print_results(filter_job, temperature=TEMPERATURE, unit=UNIT)
        
        # Exportation des conformères finaux en SDF
        export_dir = export_conformers_to_sdf(filter_job, output_dir=f"{args.name}_conformers")
        print(f"\nLes conformères ont été exportés dans le dossier: {export_dir}")

    except RuntimeError as e:
        print(f"Erreur : {e}")
    finally:
        finish()

if __name__ == "__main__":
    main()
