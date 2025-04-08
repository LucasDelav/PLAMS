#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# $AMSBIN/amspython analysis.py MeOH_workdir/redox/ EtOH_workdir/redox/ EtOMe_workdir/redox/


"""
Analyse des potentiels redox à partir des fichiers redox_potentials.txt
Génère des graphiques de corrélation entre le potentiel global et ses contributions

Usage:
    python analysis.py chemin1/redox.XXX/ chemin2/redox.YYY/ ...
    
Exemple:
    python analysis.py EtOH_workdir/redox.011/ MeOH_workdir/redox.042/
"""

import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from pathlib import Path

# Configuration de matplotlib pour des graphiques de qualité publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5

def parse_redox_file(filepath, parent_dir_name):
    """
    Extrait les valeurs de potentiel et leurs contributions d'un fichier redox_potentials.txt
    
    Args:
        filepath: Chemin vers le fichier redox_potentials.txt
        parent_dir_name: Nom du dossier parent (pour l'identification)
        
    Returns:
        Un dictionnaire contenant les valeurs des potentiels et leurs contributions
    """
    data = {'molecule': parent_dir_name}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Extraction du potentiel global
            global_match = re.search(r'E\(∆G\) = ([-\d.]+) V', content)
            if global_match:
                data['global_potential'] = float(global_match.group(1))
            else:
                print(f"Avertissement: Potentiel global non trouvé dans {filepath}")
            
            # Extraction des contributions
            ea_match = re.search(r'E\(EA\) = ([-\d.]+) V', content)
            if ea_match:
                data['EA'] = float(ea_match.group(1))
            else:
                print(f"Avertissement: Contribution EA non trouvée dans {filepath}")
                
            edef_match = re.search(r'E\(Edef\) = ([-\d.]+) V', content)
            if edef_match:
                data['Edef'] = float(edef_match.group(1))
            else:
                print(f"Avertissement: Contribution Edef non trouvée dans {filepath}")
                
            delta_u_match = re.search(r'E\(∆∆U\) = ([-\d.]+) V', content)
            if delta_u_match:
                data['delta_delta_U'] = float(delta_u_match.group(1))
            else:
                print(f"Avertissement: Contribution ∆∆U non trouvée dans {filepath}")
                
            t_delta_s_match = re.search(r'E\(T∆S\) = ([-\d.]+) V', content)
            if t_delta_s_match:
                data['T_delta_S'] = float(t_delta_s_match.group(1))
            else:
                print(f"Avertissement: Contribution T∆S non trouvée dans {filepath}")
            
            return data
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {filepath}: {e}")
        return None

def collect_data(paths):
    """
    Collecte les données des fichiers redox_potentials.txt dans les chemins spécifiés
    
    Args:
        paths: Liste des chemins vers les dossiers parents
        
    Returns:
        Une liste de dictionnaires contenant les données des fichiers
    """
    data_list = []
    
    for path in paths:
        # Construire le chemin complet vers le fichier redox_potentials.txt
        path = Path(path)
        redox_file = path / "redox_results" / "redox_potentials.txt"
        
        if not redox_file.exists():
            print(f"Attention: Fichier non trouvé - {redox_file}")
            continue
        
        # Extraire le nom du dossier parent pour l'identification
        # Prend le premier segment du chemin (ex: "EtOH_workdir" dans "EtOH_workdir/redox.011/")
        parent_name = path.parts[0] if path.parts else "Unknown"
        
        # Ajouter un identifiant de redox (ex: redox.011)
        for part in path.parts:
            if part.startswith("redox."):
                parent_name = f"{parent_name}_{part}"
                break
        
        # Analyser le fichier
        data = parse_redox_file(redox_file, parent_name)
        
        if data:
            data_list.append(data)
            print(f"Données extraites pour {parent_name} : E(ΔG) = {data['global_potential']} V")
    
    return data_list

def create_correlation_plots(data_list):
    """
    Crée des graphiques de corrélation entre le potentiel global et ses contributions
    
    Args:
        data_list: Liste de dictionnaires contenant les données des potentiels
    """
    if not data_list:
        print("Aucune donnée disponible pour créer les graphiques.")
        return
    
    # La correspondance entre les clés et les noms d'affichage
    contributions = {
        'EA': 'E(EA) (V)',
        'Edef': 'E(Edef) (V)',
        'delta_delta_U': 'E(ΔΔU) (V)',
        'T_delta_S': 'E(TΔS) (V)'
    }
    
    # Création d'un répertoire numéroté pour les graphiques
    output_dir = create_output_directory()
    
    # Création des graphiques pour chaque contribution
    for contrib_key, contrib_label in contributions.items():
        # Extraction des données pour cette contribution
        x_values = [data[contrib_key] for data in data_list if contrib_key in data]
        y_values = [data['global_potential'] for data in data_list if contrib_key in data]
        labels = [data['molecule'] for data in data_list if contrib_key in data]
        
        if not x_values or not y_values:
            print(f"Données insuffisantes pour la contribution {contrib_key}")
            continue
            
        # Calcul de la régression linéaire
        slope, intercept = np.polyfit(x_values, y_values, 1)
        regression_line = [slope * x + intercept for x in x_values]
        
        # Calcul du coefficient de détermination R²
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy**2
        
        # Création du graphique
        plt.figure(figsize=(10, 8))
        
        # Tracé des points avec les noms des molécules
        plt.scatter(x_values, y_values, s=80, alpha=0.7, c='blue', edgecolors='black')
        
        # Ajout des étiquettes pour les points
        for i, label in enumerate(labels):
            plt.annotate(label, (x_values[i], y_values[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        # Tracé de la droite de régression
        plt.plot(x_values, regression_line, 'r-', linewidth=2,
                label=f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_squared:.4f}')
        
        plt.xlabel(contrib_label, fontsize=14)
        plt.ylabel('E(redox) (V)', fontsize=14)
        plt.title(f'Corrélation entre {contrib_label} et le potentiel global', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Enregistrement du graphique
        output_filename = output_dir / f"correlation_{contrib_key}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Graphique enregistré: {output_filename}")
        
        plt.close()
    
    print(f"Tous les graphiques ont été générés avec succès dans le dossier '{output_dir}'.")

def create_output_directory():
    """
    Crée un répertoire pour les graphiques avec numérotation automatique
    si le dossier existe déjà
    
    Returns:
        Path: Chemin vers le répertoire créé
    """
    base_dir_name = "redox_analysis_plots"
    output_dir = Path(base_dir_name)
    
    # Si le répertoire de base existe déjà, crée un nouveau avec numéro incrémenté
    if output_dir.exists():
        counter = 1
        while True:
            # Essayer avec des numéros à 3 chiffres (001, 002, etc.)
            new_dir_name = f"{base_dir_name}.{counter:03d}"
            output_dir = Path(new_dir_name)
            
            if not output_dir.exists():
                break
                
            counter += 1
    
    output_dir.mkdir(exist_ok=True)
    print(f"Création du dossier de sortie: {output_dir}")
    return output_dir

def main():
    """
    Fonction principale - Traite les arguments de la ligne de commande
    """
    # Vérification des arguments
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} chemin1/redox.XXX/ chemin2/redox.YYY/ ...")
        print(f"Exemple: {sys.argv[0]} EtOH_workdir/redox.011/ MeOH_workdir/redox.042/")
        sys.exit(1)
    
    # Collecte des chemins à partir des arguments
    paths = sys.argv[1:]
    print(f"Analyse des potentiels redox pour {len(paths)} chemins...")
    
    # Collecte des données
    data_list = collect_data(paths)
    
    if not data_list:
        print("Aucune donnée n'a pu être extraite. Vérifiez les chemins fournis.")
        sys.exit(1)
    
    print(f"Données extraites pour {len(data_list)} molécules.")
    
    # Création des graphiques de corrélation
    create_correlation_plots(data_list)

if __name__ == "__main__":
    main()
