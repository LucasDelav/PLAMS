#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scm.plams import Molecule, Settings, init, finish, AMSJob, JobRunner, config, KFFile
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px

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

def run_qtaim_calculation(conf_name, molecule, settings, state):
    """
    Initialise et lance les calculs QTAIM.
    
    Args:
        conf_name (str): Nom du conformère.
        molecule (Molecule): La molécule PLAMS.
        settings (Settings): Paramètres du calcul.
        state (str): État de la molécule ('neutre', 'oxidized', 'reduced').
    
    Returns:
        AMSJob: Le job AMS exécuté.
    """
    # Créer un nom unique pour le job qui inclut l'état et le nom du conformère
    job_name = f"{conf_name}_{state}_qtaim"
    
    # IMPORTANT: Ne pas redéfinir config.default_jobmanager.workdir
    # Laisser PLAMS gérer automatiquement la numérotation des répertoires
    
    # Créer et exécuter le job
    job = AMSJob(molecule=molecule, settings=settings, name=job_name)
    job.run(jobrunner=JobRunner(parallel=True, maxjobs=config.default_jobrunner.maxjobs))
    
    # Log le résultat
    if job.ok():
        print(f"Le calcul QTAIM pour {conf_name} ({state}) a réussi.")
    else:
        print(f"Le calcul QTAIM pour {conf_name} ({state}) a échoué.")
    
    return job

def extract_bader_charges(job, conf_name, state):
    """
    Extrait les charges de Bader à partir des résultats du job.

    Args:
        job (AMSJob): Le job AMS exécuté.
        conf_name (str): Nom du conformère.
        state (str): État de la molécule ('neutre', 'oxidized', 'reduced').

    Returns:
        dict: Dictionnaire des charges atomiques avec indices des atomes comme clés.
    """
    charges = {}

    try:
        # Chemin vers le fichier de résultat ADF
        adf_file = job.results.rkfpath('adf')
        kf = KFFile(adf_file)  # Important: créer l'objet KFFile

        # Maintenant vous pouvez utiliser kf.read() pour accéder aux données
        try:
            # Essayer de lire les charges directement
            atomic_charges = kf.read('Properties', 'Bader atomic charges')

            # Si on arrive ici, c'est que la lecture a réussi
            for i, charge in enumerate(atomic_charges, 1):
                charges[i] = charge

            print(f"Charges de Bader extraites pour {conf_name} ({state})")
        except Exception as e1:
            print(f"Echec de l'extraction: {e1}")
            return charges

    except Exception as e:
        print(f"Erreur lors de l'accès au fichier KF pour {conf_name} ({state}): {e}")

    return charges

def extract_iqa_energies(job, conf_name, state):
    """
    Extrait les énergies IQA additives à partir des résultats du job AMS.

    Args:
        job (AMSJob): Le job AMS exécuté.
        conf_name (str): Nom du conformère.
        state (str): État de la molécule ('neutre', 'oxidized', 'reduced').

    Returns:
        dict: Dictionnaire contenant les énergies IQA additives par atome et le total.
    """
    iqa_data = {}

    try:
        # Obtenir le chemin du fichier de sortie
        output_file = job.results.job.path + "/" + job.name + ".out"

        if not os.path.exists(output_file):
            print(f"Fichier de sortie non trouvé: {output_file}")
            return iqa_data

        # Lire le fichier ligne par ligne pour éviter de charger tout en mémoire
        atomic_energies = {}
        total_energy = None
        in_additive_section = False

        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Détecter le début de la section IQA
                if "C - Additive IQA Energies" in line:
                    in_additive_section = True
                    continue

                # Si on est dans la section et qu'on trouve une ligne d'atome
                if in_additive_section and line.startswith("Atom "):
                    # Découper la ligne : "Atom C1     * EaddIQA   =     -38.113457"
                    parts = line.split()
                    if len(parts) >= 5 and parts[2] == "*" and parts[3] == "EaddIQA" and parts[4] == "=":
                        atom_label = parts[1]  # Ex: "C1", "O3", "H4"
                        energy_str = parts[5]  # Ex: "-38.113457"

                        try:
                            energy = float(energy_str)

                            # Extraire le symbole et l'indice de l'atome avec split
                            atom_symbol = ""
                            atom_number_str = ""

                            # Séparer lettres et chiffres manuellement
                            for char in atom_label:
                                if char.isalpha():
                                    atom_symbol += char
                                elif char.isdigit():
                                    atom_number_str += char

                            if atom_symbol and atom_number_str:
                                atom_index = int(atom_number_str)

                                atomic_energies[atom_index] = {
                                    'symbol': atom_symbol,
                                    'label': atom_label,
                                    'EaddIQA': energy
                                }

                        except ValueError:
                            continue

                # Détecter la ligne Total et sortir
                elif in_additive_section and line.startswith("Total"):
                    # Découper : "Total                   =    -154.929122"
                    parts = line.split("=")
                    if len(parts) == 2:
                        total_str = parts[1].strip()
                        try:
                            total_energy = float(total_str)
                            # Fin de la section - sortir de la boucle
                            break
                        except ValueError:
                            continue

        # Assembler les résultats
        if atomic_energies and total_energy is not None:
            iqa_data = {
                'atomic_energies': atomic_energies,
                'total_energy': total_energy
            }
            print(f"Énergies IQA additives extraites pour {conf_name} ({state}): {len(atomic_energies)} atomes")
        else:
            print(f"Section 'Additive IQA Energies' non trouvée ou incomplète pour {conf_name} ({state})")

    except Exception as e:
        print(f"Erreur lors de l'extraction IQA pour {conf_name} ({state}): {e}")

    return iqa_data

def calculate_iqa_differences(neutral_iqa, oxidized_iqa, reduced_iqa):
    """
    Calcule les différences d'énergies IQA additives.

    Returns:
        dict: Dictionnaire avec les différences par atome et les totaux.
    """
    iqa_differences = {}

    neutral_atomic = neutral_iqa.get('atomic_energies', {})
    oxidized_atomic = oxidized_iqa.get('atomic_energies', {})
    reduced_atomic = reduced_iqa.get('atomic_energies', {})

    # Différences atomiques
    atomic_deltas = {}
    for atom_idx in neutral_atomic:
        if atom_idx in oxidized_atomic and atom_idx in reduced_atomic:
            neutral_energy = neutral_atomic[atom_idx]['EaddIQA']
            oxidized_energy = oxidized_atomic[atom_idx]['EaddIQA']
            reduced_energy = reduced_atomic[atom_idx]['EaddIQA']

            atomic_deltas[atom_idx] = {
                'label': neutral_atomic[atom_idx]['label'],
                'symbol': neutral_atomic[atom_idx]['symbol'],
                'delta_ox': oxidized_energy - neutral_energy,
                'delta_red': reduced_energy - neutral_energy
            }

    # Différences totales
    neutral_total = neutral_iqa.get('total_energy', 0)
    oxidized_total = oxidized_iqa.get('total_energy', 0)
    reduced_total = reduced_iqa.get('total_energy', 0)

    iqa_differences = {
        'atomic_deltas': atomic_deltas,
        'total_delta_ox': oxidized_total - neutral_total,
        'total_delta_red': reduced_total - neutral_total,
        'neutral_total': neutral_total,
        'oxidized_total': oxidized_total,
        'reduced_total': reduced_total
    }

    return iqa_differences

def create_charge_difference_plots(df, conf_name, output_dir):
    """
    Crée des graphiques pour visualiser les différences de charges.
    
    Args:
        df (DataFrame): DataFrame contenant les données de charges.
        conf_name (str): Nom du conformère.
        output_dir (str): Répertoire de sortie pour les graphiques.
    """
    # Trier par valeur absolue des différences de charges (oxydé)
    df_sorted_ox = df.sort_values(by='Δ(q) Ox-Neu', key=abs, ascending=False)
    
    # Limitons aux 10 premiers atomes avec les plus grandes différences
    df_top_ox = df_sorted_ox.head(10)
    
    # Créer le graphique pour oxydation
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [f"{row['Atom Symbol']}{row['Atom Index']}" for _, row in df_top_ox.iterrows()],
        df_top_ox['Δ(q) Ox-Neu']
    )
    
    # Colorer selon le signe
    for i, bar in enumerate(bars):
        if bar.get_height() < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f'Top 10 Atomic Charge Differences (Oxidized - Neutral) for {conf_name}')
    plt.xlabel('Atom')
    plt.ylabel('Δ(q) [e]')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(os.path.join(output_dir, f'{conf_name}_ox_charge_diff.png'), dpi=300)
    plt.close()  # Fermer la figure pour libérer la mémoire
    
    # Faire de même pour la réduction
    df_sorted_red = df.sort_values(by='Δ(q) Red-Neu', key=abs, ascending=False)
    df_top_red = df_sorted_red.head(10)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [f"{row['Atom Symbol']}{row['Atom Index']}" for _, row in df_top_red.iterrows()],
        df_top_red['Δ(q) Red-Neu']
    )
    
    # Colorer selon le signe
    for i, bar in enumerate(bars):
        if bar.get_height() < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f'Top 10 Atomic Charge Differences (Reduced - Neutral) for {conf_name}')
    plt.xlabel('Atom')
    plt.ylabel('Δ(q) [e]')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(os.path.join(output_dir, f'{conf_name}_red_charge_diff.png'), dpi=300)
    plt.close()  # Fermer la figure pour libérer la mémoire
    
    print(f"Graphiques des différences de charges créés pour {conf_name}")

def calculate_charge_differences(neutral_jobs, oxidized_jobs, reduced_jobs, plams_workdir):
    """
    Calcule les différences de charges Δ(q) entre neutre vs. oxydé et neutre vs. réduit
    pour chaque atome et génère des visualisations et des rapports.
    """
    # Créer un répertoire pour les rapports de charges dans le répertoire de travail PLAMS
    charge_dir = os.path.join(plams_workdir, 'charge_analysis')
    os.makedirs(charge_dir, exist_ok=True)

    print("\n--- Calcul des différences de charges QTAIM ---")

    # Pour chaque conformère
    for conf_name in neutral_jobs:
        if conf_name not in oxidized_jobs or conf_name not in reduced_jobs:
            print(f"Les calculs pour tous les états de {conf_name} ne sont pas disponibles.")
            continue

        neutral_job = neutral_jobs[conf_name]
        oxidized_job = oxidized_jobs[conf_name]
        reduced_job = reduced_jobs[conf_name]

        # Extraire les charges QTAIM
        neutral_charges = extract_bader_charges(neutral_job, conf_name, 'neutre')
        oxidized_charges = extract_bader_charges(oxidized_job, conf_name, 'oxidized')
        reduced_charges = extract_bader_charges(reduced_job, conf_name, 'reduced')

        if not neutral_charges or not oxidized_charges or not reduced_charges:
            print(f"Données de charges manquantes pour {conf_name}. Passage au suivant.")
            continue

        # Calculer les différences de charges
        delta_q_ox = {}
        delta_q_red = {}

        # Molécule pour référence des noms d'atomes
        molecule = neutral_job.molecule

        # Créer des tableaux pour le rapport
        data = []

        for idx in neutral_charges:
            # Calculer Δ(q) = q(état) - q(neutre)
            delta_ox = oxidized_charges[idx] - neutral_charges[idx]
            delta_red = reduced_charges[idx] - neutral_charges[idx]

            delta_q_ox[idx] = delta_ox
            delta_q_red[idx] = delta_red

            # Ajouter les données au tableau
            atom_symbol = molecule[idx].symbol
            data.append({
                'Atom Index': idx,
                'Atom Symbol': atom_symbol,
                'Neutral Charge': neutral_charges[idx],
                'Oxidized Charge': oxidized_charges[idx],
                'Reduced Charge': reduced_charges[idx],
                'Δ(q) Ox-Neu': delta_ox,
                'Δ(q) Red-Neu': delta_red
            })

        # Créer un DataFrame et sauvegarder en CSV
        df = pd.DataFrame(data)

        # Créer des graphiques pour visualisation
        create_charge_difference_plots(df, conf_name, charge_dir)

        # NOUVELLE SECTION - Extraire les énergies IQA
        print(f"Extraction des énergies IQA additives pour {conf_name}...")
        neutral_iqa = extract_iqa_energies(neutral_job, conf_name, 'neutre')
        oxidized_iqa = extract_iqa_energies(oxidized_job, conf_name, 'oxidized')
        reduced_iqa = extract_iqa_energies(reduced_job, conf_name, 'reduced')

        # Calculer les différences IQA si toutes les données sont disponibles
        iqa_differences = None
        if neutral_iqa and oxidized_iqa and reduced_iqa:
            iqa_differences = calculate_iqa_differences(neutral_iqa, oxidized_iqa, reduced_iqa)
            print(f"Différences IQA calculées pour {conf_name}")

        # Produire et sauvegarder les statistiques AVEC les données IQA
        analyze_charge_statistics(df, conf_name, charge_dir, iqa_differences)
        
        # plot_charge_differences_on_molecule_enhanced(
        #     molecule=neutral_job.molecule,
        #     neutral_charges=neutral_charges,
        #     oxidized_charges=oxidized_charges,
        #     reduced_charges=reduced_charges,
        #     conf_name=conf_name,
        #     output_dir=charge_dir)

def analyze_charge_statistics(df, conf_name, output_dir, iqa_differences=None):
    """
    Analyse statistique des différences de charges avec données IQA optionnelles.

    Args:
        df (DataFrame): DataFrame contenant les données de charges.
        conf_name (str): Nom du conformère.
        output_dir (str): Répertoire de sortie pour les statistiques.
        iqa_differences (dict): Données IQA optionnelles.
    """
    # Créer un dictionnaire pour stocker les statistiques
    stats = {}

    # ---- STATISTIQUES POUR L'OXYDATION ----
    # Maximum de différence de charge (plus grande valeur positive)
    max_ox_idx = df['Δ(q) Ox-Neu'].idxmax()
    max_ox_value = df.loc[max_ox_idx, 'Δ(q) Ox-Neu']
    max_ox_atom = f"{df.loc[max_ox_idx, 'Atom Symbol']}{df.loc[max_ox_idx, 'Atom Index']}"
    stats['max_ox_value'] = max_ox_value
    stats['max_ox_atom'] = max_ox_atom

    # Valeur absolue maximale (changement le plus important, quel que soit le signe)
    abs_max_ox_idx = df['Δ(q) Ox-Neu'].abs().idxmax()
    abs_max_ox_value = df.loc[abs_max_ox_idx, 'Δ(q) Ox-Neu']
    abs_max_ox_atom = f"{df.loc[abs_max_ox_idx, 'Atom Symbol']}{df.loc[abs_max_ox_idx, 'Atom Index']}"
    stats['abs_max_ox_value'] = abs_max_ox_value
    stats['abs_max_ox_atom'] = abs_max_ox_atom

    # Moyenne des valeurs positives (atomes qui perdent des électrons)
    positive_ox = df[df['Δ(q) Ox-Neu'] > 0]['Δ(q) Ox-Neu']
    stats['mean_positive_ox'] = positive_ox.mean() if not positive_ox.empty else 0
    stats['count_positive_ox'] = len(positive_ox)

    # Moyenne des valeurs négatives (atomes qui gagnent des électrons)
    negative_ox = df[df['Δ(q) Ox-Neu'] < 0]['Δ(q) Ox-Neu']
    stats['mean_negative_ox'] = negative_ox.mean() if not negative_ox.empty else 0
    stats['count_negative_ox'] = len(negative_ox)

    # Écart-type (mesure de la dispersion des différences de charge)
    stats['std_ox'] = df['Δ(q) Ox-Neu'].std()

    # Somme des différences de charge (devrait être proche de +1.0 pour l'oxydation)
    stats['sum_ox'] = df['Δ(q) Ox-Neu'].sum()

    # ---- STATISTIQUES POUR LA RÉDUCTION ----
    # Minimum de différence de charge (plus grande valeur négative)
    min_red_idx = df['Δ(q) Red-Neu'].idxmin()
    min_red_value = df.loc[min_red_idx, 'Δ(q) Red-Neu']
    min_red_atom = f"{df.loc[min_red_idx, 'Atom Symbol']}{df.loc[min_red_idx, 'Atom Index']}"
    stats['min_red_value'] = min_red_value
    stats['min_red_atom'] = min_red_atom

    # Valeur absolue maximale (changement le plus important, quel que soit le signe)
    abs_max_red_idx = df['Δ(q) Red-Neu'].abs().idxmax()
    abs_max_red_value = df.loc[abs_max_red_idx, 'Δ(q) Red-Neu']
    abs_max_red_atom = f"{df.loc[abs_max_red_idx, 'Atom Symbol']}{df.loc[abs_max_red_idx, 'Atom Index']}"
    stats['abs_max_red_value'] = abs_max_red_value
    stats['abs_max_red_atom'] = abs_max_red_atom

    # Moyenne des valeurs négatives (atomes qui gagnent des électrons)
    negative_red = df[df['Δ(q) Red-Neu'] < 0]['Δ(q) Red-Neu']
    stats['mean_negative_red'] = negative_red.mean() if not negative_red.empty else 0
    stats['count_negative_red'] = len(negative_red)

    # Moyenne des valeurs positives (atomes qui perdent des électrons)
    positive_red = df[df['Δ(q) Red-Neu'] > 0]['Δ(q) Red-Neu']
    stats['mean_positive_red'] = positive_red.mean() if not positive_red.empty else 0
    stats['count_positive_red'] = len(positive_red)

    # Écart-type (mesure de la dispersion des différences de charge)
    stats['std_red'] = df['Δ(q) Red-Neu'].std()

    # Somme des différences de charge (devrait être proche de -1.0 pour la réduction)
    stats['sum_red'] = df['Δ(q) Red-Neu'].sum()

    # Écrire les statistiques dans un fichier
    stats_file = os.path.join(output_dir, f'{conf_name}_charge_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Statistiques des différences de charges pour {conf_name}\n")
        f.write("="*50 + "\n\n")

        f.write("STATISTIQUES D'OXYDATION\n")
        f.write("-"*30 + "\n")
        f.write(f"Maximum positif: {max_ox_value:.4f} sur l'atome {max_ox_atom}\n")
        f.write(f"Changement maximal (abs): {abs_max_ox_value:.4f} sur l'atome {abs_max_ox_atom}\n")
        f.write(f"Moyenne des charges positives: {stats['mean_positive_ox']:.4f} (sur {stats['count_positive_ox']} atomes)\n")
        f.write(f"Moyenne des charges négatives: {stats['mean_negative_ox']:.4f} (sur {stats['count_negative_ox']} atomes)\n")
        f.write(f"Écart-type: {stats['std_ox']:.4f}\n")
        f.write(f"Somme totale: {stats['sum_ox']:.4f} (attendu: +1 e)\n")

        f.write("\nSTATISTIQUES DE RÉDUCTION\n")
        f.write("-"*30 + "\n")
        f.write(f"Maximum négatif: {min_red_value:.4f} sur l'atome {min_red_atom}\n")
        f.write(f"Changement maximal (abs): {abs_max_red_value:.4f} sur l'atome {abs_max_red_atom}\n")
        f.write(f"Moyenne des charges négatives: {stats['mean_negative_red']:.4f} (sur {stats['count_negative_red']} atomes)\n")
        f.write(f"Moyenne des charges positives: {stats['mean_positive_red']:.4f} (sur {stats['count_positive_red']} atomes)\n")
        f.write(f"Écart-type: {stats['std_red']:.4f}\n")
        f.write(f"Somme totale: {stats['sum_red']:.4f} (attendu: -1 e)\n")

        # NOUVELLE SECTION IQA
        if iqa_differences:
            f.write("\n" + "="*60 + "\n")
            f.write("STATISTIQUES IQA ADDITIVES\n")
            f.write("="*60 + "\n\n")

            # Énergies totales
            f.write("ÉNERGIES TOTALES ADDITIVES\n")
            f.write("-"*30 + "\n")
            f.write(f"État neutre:    {iqa_differences['neutral_total']:12.6f} hartree\n")
            f.write(f"État oxydé:     {iqa_differences['oxidized_total']:12.6f} hartree\n")
            f.write(f"État réduit:    {iqa_differences['reduced_total']:12.6f} hartree\n\n")

            f.write(f"Δ(oxydation):   {iqa_differences['total_delta_ox']:12.6f} hartree\n")
            f.write(f"Δ(réduction):   {iqa_differences['total_delta_red']:12.6f} hartree\n\n")

            # Contributions atomiques
            f.write("CONTRIBUTIONS ATOMIQUES IQA\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Atome':>6} {'Δ(IQA) Ox-Neu':>15} {'Δ(IQA) Red-Neu':>15}\n")
            f.write("-"*50 + "\n")

            # Trier par valeur absolue de changement d'oxydation
            atomic_deltas = iqa_differences['atomic_deltas']

            for atom_idx, data in atomic_deltas.items():
                f.write(f"{data['label']:>6} {data['delta_ox']:15.6f} {data['delta_red']:15.6f}\n")

            # Statistiques par type d'atome
            f.write(f"\nSTATISTIQUES PAR TYPE D'ATOME\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Type':>4} {'Nb':>3} {'Δ_Ox_Moy':>12} {'Δ_Red_Moy':>12} {'Δ_Ox_Max':>12} {'Δ_Red_Max':>12}\n")
            f.write("-"*60 + "\n")

            # Calculer statistiques par type d'atome
            atom_types = {}
            for atom_idx, data in atomic_deltas.items():
                symbol = data['symbol']
                if symbol not in atom_types:
                    atom_types[symbol] = {'ox': [], 'red': []}
                atom_types[symbol]['ox'].append(data['delta_ox'])
                atom_types[symbol]['red'].append(data['delta_red'])

            for symbol in sorted(atom_types.keys()):
                ox_values = atom_types[symbol]['ox']
                red_values = atom_types[symbol]['red']

                count = len(ox_values)
                ox_mean = sum(ox_values) / count
                red_mean = sum(red_values) / count
                ox_max = max(ox_values, key=abs)
                red_max = max(red_values, key=abs)

                f.write(f"{symbol:>4} {count:>3} {ox_mean:12.6f} {red_mean:12.6f} {ox_max:12.6f} {red_max:12.6f}\n")

    print(f"Statistiques complètes (charges + IQA) sauvegardées: {stats_file}")

    # Mettre à jour le DataFrame avec les données d'impact
    df.to_csv(os.path.join(output_dir, f'{conf_name}_charge_differences.csv'), index=False)

    return stats

def plot_conformer_with_charges(molecule, charges, conf_name, state, output_dir,
                               charge_type="bader", view_angles=None):
    """
    Visualise un conformère en 3D avec les charges atomiques affichées sur chaque atome (version statique).
    """
    if view_angles is None:
        view_angles = (20, 45)

    # Créer la figure 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extraire les coordonnées et informations atomiques
    coords = []
    symbols = []
    atom_charges = []

    for i, atom in enumerate(molecule, 1):
        coords.append([atom.coords[0], atom.coords[1], atom.coords[2]])
        symbols.append(atom.symbol)
        atom_charges.append(charges.get(i, 0.0))

    coords = np.array(coords)
    atom_charges = np.array(atom_charges)

    # === AMÉLIORATION 1: Calcul des limites des axes basées sur la molécule + marge ===
    margin = 0.2  # Marge de 0.2 Angstrom
    x_min, x_max = np.min(coords[:, 0]) - margin, np.max(coords[:, 0]) + margin
    y_min, y_max = np.min(coords[:, 1]) - margin, np.max(coords[:, 1]) + margin
    z_min, z_max = np.min(coords[:, 2]) - margin, np.max(coords[:, 2]) + margin

    # Couleurs par élément
    element_colors = {
        'H': '#FFFFFF',  'C': '#909090',  'N': '#3050F8',  'O': '#FF0D0D',
        'F': '#90E050',  'P': '#FF8000',  'S': '#FFFF30',  'Cl': '#1FF01F',
        'Br': '#A62929', 'I': '#940094',  'B': '#FFB5B5',  'Si': '#F0C8A0'
    }

    element_sizes = {
        'H': 50,   'C': 150,  'N': 140,  'O': 130,
        'F': 120,  'P': 180,  'S': 170,  'Cl': 175,
        'Br': 185, 'I': 198,  'B': 85,   'Si': 210
    }

    # === AMÉLIORATION 2: Normalisation des charges basée sur les valeurs réelles ===
    if len(atom_charges) > 0 and np.any(atom_charges != 0):
        charge_min = np.min(atom_charges)
        charge_max = np.max(atom_charges)

        # Si toutes les charges sont identiques
        if charge_min == charge_max:
            if charge_min == 0:
                charge_min, charge_max = -0.1, 0.1
            else:
                charge_range = abs(charge_min) * 0.1
                charge_min = charge_min - charge_range
                charge_max = charge_min + charge_range
    else:
        charge_min, charge_max = -0.1, 0.1

    # Normalisation pour la carte de couleurs
    cmap = cm.RdBu_r  # Rouge=positif, Bleu=négatif
    norm = Normalize(vmin=charge_min, vmax=charge_max)

    # Dessiner les liaisons d'abord (arrière-plan)
    draw_bonds(ax, coords, symbols)

    # Dessiner les atomes avec couleurs basées sur les charges
    for i, (coord, symbol, charge) in enumerate(zip(coords, symbols, atom_charges)):
        color = cmap(norm(charge))
        size = element_sizes.get(symbol, 100)

        # Dessiner l'atome
        ax.scatter(coord[0], coord[1], coord[2],
                  c=[color], s=size, alpha=0.8, edgecolors='black', linewidth=1,
                  zorder=10)  # zorder élevé pour les atomes

        # === AMÉLIORATION 3: Améliorer l'affichage du texte ===
        # Ajouter le texte avec fond et bordure pour la lisibilité
        ax.text(coord[0], coord[1], coord[2],
               f'{symbol}{i+1}\n{charge:.3f}',
               fontsize=8, fontweight='bold', color='black',
               ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
               zorder=1000)  # zorder très élevé pour forcer le texte au premier plan

    # Définir les limites des axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Labels des axes
    ax.set_xlabel('X (Å)', fontweight='bold')
    ax.set_ylabel('Y (Å)', fontweight='bold')
    ax.set_zlabel('Z (Å)', fontweight='bold')

    # Titre
    ax.set_title(f'{conf_name} - {state} State\n{charge_type.capitalize()} Charges',
                fontsize=16, fontweight='bold', pad=20)

    # Vue en 3D
    ax.view_init(elev=view_angles[0], azim=view_angles[1])

    # Barre de couleur pour les charges
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30, pad=0.1)
    cbar.set_label(f'{charge_type.capitalize()} Charge (e)', fontsize=12)

    # Personnaliser les ticks de la barre de couleur pour afficher les vraies valeurs
    if charge_max != charge_min:
        tick_values = np.linspace(charge_min, charge_max, 5)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.2f}' for val in tick_values])

    # Grille et esthétique
    ax.grid(True, alpha=0.3)

    # Sauvegarder avec haute résolution
    filename = f'{conf_name}_{state.lower()}_{charge_type}_charges_3D.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Visualisation 3D statique sauvegardée: {filepath}")

def draw_bonds(ax, coords, symbols):
    """
    Dessine les liaisons chimiques approximatives basées sur les rayons covalents.

    Args:
        ax: Axes matplotlib 3D.
        coords (np.array): Coordonnées atomiques.
        symbols (list): Symboles atomiques.
    """
    # Rayons covalents approximatifs (en Angstrom)
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39,
        'B': 0.84, 'Si': 1.11
    }

    n_atoms = len(coords)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            r1 = covalent_radii.get(symbols[i], 1.0)
            r2 = covalent_radii.get(symbols[j], 1.0)
            expected_bond_length = r1 + r2

            # Liaison si la distance est dans une plage raisonnable
            if distance <= expected_bond_length * 1.3:
                ax.plot([coords[i][0], coords[j][0]],
                       [coords[i][1], coords[j][1]],
                       [coords[i][2], coords[j][2]],
                       'gray', linewidth=2, alpha=0.7, zorder=1)  # zorder bas pour les liaisons

def plot_conformer_with_charges_interactive(molecule, charges, conf_name, state, output_dir,
                                          charge_type="bader"):
    """
    Crée une visualisation 3D interactive avec Plotly que vous pouvez tourner dans le navigateur.
    """
    # Extraire les coordonnées et informations atomiques
    coords = []
    symbols = []
    atom_charges = []

    for i, atom in enumerate(molecule, 1):
        coords.append([atom.coords[0], atom.coords[1], atom.coords[2]])
        symbols.append(atom.symbol)
        atom_charges.append(charges.get(i, 0.0))

    coords = np.array(coords)
    atom_charges = np.array(atom_charges)

    # Échelle des charges réaliste
    charge_min = np.min(atom_charges)
    charge_max = np.max(atom_charges)

    if charge_min == charge_max:
        if charge_min == 0:
            charge_min, charge_max = -0.1, 0.1
        else:
            charge_range = abs(charge_min) * 0.1
            charge_min = charge_min - charge_range
            charge_max = charge_min + charge_range

    # Couleurs et tailles par élément
    element_colors = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
        'F': '#90E050', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F',
        'Br': '#A62929', 'I': '#940094', 'B': '#FFB5B5', 'Si': '#F0C8A0'
    }

    element_sizes = {
        'H': 5, 'C': 15, 'N': 14, 'O': 13, 'F': 12, 'P': 18,
        'S': 17, 'Cl': 17, 'Br': 18, 'I': 20, 'B': 8, 'Si': 21
    }

    # Préparer les données pour Plotly
    atom_colors = [element_colors.get(symbol, '#909090') for symbol in symbols]
    atom_sizes = [element_sizes.get(symbol, 10) for symbol in symbols]

    # Créer les traces pour les atomes
    fig = go.Figure()

    # Ajouter les atomes avec la colorbar corrigée
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(
            size=atom_sizes,
            color=atom_charges,
            colorscale='RdBu_r',
            cmin=charge_min,
            cmax=charge_max,
            colorbar=dict(
                title=f"{charge_type.capitalize()} Charge (e)",
                # Supprimé titleside qui n'existe pas
                thickness=20,
                len=0.8
            ),
            line=dict(width=2, color='black')
        ),
        text=[f"{symbol}{i+1}<br>{charge:.3f}"
              for i, (symbol, charge) in enumerate(zip(symbols, atom_charges))],
        textposition="middle center",
        textfont=dict(size=10, color="black"),
        name="Atoms",
        hovertemplate=(
            "Atome: %{text}<br>" +
            "X: %{x:.3f} Å<br>" +
            "Y: %{y:.3f} Å<br>" +
            "Z: %{z:.3f} Å<br>" +
            "<extra></extra>"
        )
    ))

    # Ajouter les liaisons
    bond_traces = create_bonds_plotly(coords, symbols)
    for bond_trace in bond_traces:
        fig.add_trace(bond_trace)

    # Configuration de la mise en page avec les limites améliorées
    margin = 0.2
    x_min, x_max = np.min(coords[:, 0]) - margin, np.max(coords[:, 0]) + margin
    y_min, y_max = np.min(coords[:, 1]) - margin, np.max(coords[:, 1]) + margin
    z_min, z_max = np.min(coords[:, 2]) - margin, np.max(coords[:, 2]) + margin

    fig.update_layout(
        title=f'{conf_name} - {state} State - {charge_type.capitalize()} Charges',
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max]),
            aspectmode='cube'
        ),
        width=900,
        height=700
    )

    # Sauvegarder en HTML interactif
    filename = f'{conf_name}_{state.lower()}_{charge_type}_charges_interactive.html'
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath)

    print(f"Visualisation 3D interactive sauvegardée: {filepath}")
    print("Ouvrez le fichier HTML dans votre navigateur pour interagir avec la molécule !")

    return fig

def create_bonds_plotly(coords, symbols):
    """
    Crée les traces de liaisons pour Plotly.
    """
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39,
        'B': 0.84, 'Si': 1.11
    }

    bond_traces = []
    n_atoms = len(coords)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            r1 = covalent_radii.get(symbols[i], 1.0)
            r2 = covalent_radii.get(symbols[j], 1.0)
            expected_bond_length = r1 + r2

            if distance <= expected_bond_length * 1.3:
                bond_trace = go.Scatter3d(
                    x=[coords[i][0], coords[j][0], None],
                    y=[coords[i][1], coords[j][1], None],
                    z=[coords[i][2], coords[j][2], None],
                    mode='lines',
                    line=dict(color='gray', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                )
                bond_traces.append(bond_trace)

    return bond_traces

def plot_charge_differences_on_molecule_enhanced(molecule, neutral_charges, oxidized_charges,
                                               reduced_charges, conf_name, output_dir):
    """
    Crée toutes les visualisations (statiques ET interactives) pour un conformère.
    """
    # Calculer les différences
    delta_ox = {}
    delta_red = {}

    for idx in neutral_charges:
        if idx in oxidized_charges:
            delta_ox[idx] = oxidized_charges[idx] - neutral_charges[idx]
        if idx in reduced_charges:
            delta_red[idx] = reduced_charges[idx] - neutral_charges[idx]

    # Créer toutes les visualisations
    visualizations = [
        (neutral_charges, 'Neutral'),
        (oxidized_charges, 'Oxidized'),
        (reduced_charges, 'Reduced'),
        (delta_ox, 'Delta_Ox_vs_Neutral'),
        (delta_red, 'Delta_Red_vs_Neutral')
    ]

    for charges, state_label in visualizations:
        if charges:
            # Version statique ET interactive
            plot_conformer_with_charges_both_formats(
                molecule=molecule,
                charges=charges,
                conf_name=conf_name,
                state=state_label,
                output_dir=output_dir,
                charge_type="bader"
            )

def plot_conformer_with_charges_both_formats(molecule, charges, conf_name, state, output_dir,
                                           charge_type="bader", view_angles=None):
    """
    Crée à la fois une version statique (PNG) et interactive (HTML).
    """
    # Version statique (matplotlib)
    plot_conformer_with_charges(molecule, charges, conf_name, state, output_dir,
                              charge_type, view_angles)

    # Version interactive (plotly)
    plot_conformer_with_charges_interactive(molecule, charges, conf_name, state, output_dir,
                                          charge_type)

def main():
    """
    Fonction principale qui orchestre le workflow complet.
    """
    # Analyser les arguments
    args = parse_arguments()
    base_path = args.dir
    
    # Déterminer le répertoire de base pour les calculs QTAIM
    qtaim_base_dir = os.path.join(os.path.dirname(base_path), 'qtaims')
    
    # Initialiser PLAMS avec numérotation automatique
    # PLAMS créera automatiquement qtaims.001, qtaims.002, etc.
    init(folder=qtaim_base_dir)
    
    # Obtenir le répertoire de travail réel créé par PLAMS (avec numérotation)
    plams_workdir = config.default_jobmanager.workdir
    print(f"Répertoire de travail PLAMS: {plams_workdir}")
    
    # Extraire les coordonnées des conformères neutres (utilisées pour tous les états)
    neutral_conformers = extract_coordinates(base_path)
    if not neutral_conformers:
        print("Aucun conformère trouvé. Arrêt du programme.")
        finish()
        return
    
    # Dictionnaires pour stocker les jobs par conformère
    neutral_jobs = {}
    oxidized_jobs = {}
    reduced_jobs = {}
    
    # 1. Traiter les conformères neutres
    print("\n--- Traitement des conformères neutres ---")
    for conf_name, molecule in neutral_conformers:
        settings = setup_qtaim_settings(molecule, charge=0, spinpol=0)
        real_conf_name = conf_name.split('_neutre')[0]
        job = run_qtaim_calculation(real_conf_name, molecule, settings, 'neutre')
        if job.ok():
            neutral_jobs[real_conf_name] = job
    
    # 2. Traiter les conformères oxydés
    print("\n--- Traitement des conformères oxydés ---")
    for conf_name, molecule in neutral_conformers:
        settings = setup_qtaim_settings(molecule, charge=1, spinpol=1)
        real_conf_name = conf_name.split('_neutre')[0]
        job = run_qtaim_calculation(real_conf_name, molecule, settings, 'oxidized')
        if job.ok():
            oxidized_jobs[real_conf_name] = job
    
    # 3. Traiter les conformères réduits
    print("\n--- Traitement des conformères réduits ---")
    for conf_name, molecule in neutral_conformers:
        settings = setup_qtaim_settings(molecule, charge=-1, spinpol=1)
        real_conf_name = conf_name.split('_neutre')[0]
        job = run_qtaim_calculation(real_conf_name, molecule, settings, 'reduced')
        if job.ok():
            reduced_jobs[real_conf_name] = job
    
    # 4. Calculer et analyser les différences de charges
    calculate_charge_differences(neutral_jobs, oxidized_jobs, reduced_jobs, plams_workdir)
    
    # Finaliser PLAMS
    finish()

if __name__ == "__main__":
    main()

