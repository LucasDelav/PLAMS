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
            global_match = re.search(r'E\(∆G\)\s+ = ([-\d.]+) V', content)
            if global_match:
                data['global_potential'] = float(global_match.group(1))
            else:
                print(f"Avertissement: Potentiel global non trouvé dans {filepath}")
            
            # Extraction des contributions
            ea_match = re.search(r'E\(EA\)\s+ = ([-\d.]+) V', content)
            if ea_match:
                data['EA'] = float(ea_match.group(1))
            else:
                print(f"Avertissement: Contribution EA non trouvée dans {filepath}")
                
            edef_match = re.search(r'E\(Edef\) = ([-\d.]+) V', content)
            if edef_match:
                data['Edef'] = float(edef_match.group(1))
            else:
                print(f"Avertissement: Contribution Edef non trouvée dans {filepath}")
                
            delta_u_match = re.search(r'E\(∆∆U\)\s+ = ([-\d.]+) V', content)
            if delta_u_match:
                data['delta_delta_U'] = float(delta_u_match.group(1))
            else:
                print(f"Avertissement: Contribution ∆∆U non trouvée dans {filepath}")
                
            t_delta_s_match = re.search(r'E\(T∆S\)\s+ = ([-\d.]+) V', content)
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

def add_linear_regression(ax, x_values, y_values, color='red'):
    """
    Ajoute une régression linéaire (y = ax + b) au graphique
    
    Args:
        ax: Axes matplotlib
        x_values: Valeurs x (données indépendantes)
        y_values: Valeurs y (données dépendantes)
        color: Couleur de la ligne de régression
        
    Returns:
        tuple: (équation sous forme de chaîne, valeur R²)
    """
    # Régression linéaire (y = mx + b)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    
    # Points x pour tracer une ligne lisse
    x_seq = np.linspace(min(x_values), max(x_values), 100)
    y_seq = slope * x_seq + intercept
    
    # Tracé de la droite de régression
    ax.plot(x_seq, y_seq, color=color, linestyle='-', linewidth=2)
    
    # Calcul du R²
    y_pred = slope * x_values + intercept
    ss_res = np.sum((y_values - y_pred)**2)
    ss_tot = np.sum((y_values - np.mean(y_values))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Formulation de l'équation
    equation = f"y = {slope:.4f}x + {intercept:.4f}"
    
    return equation, r_squared

# Variable globale pour suivre si l'avertissement polynomial a été affiché
_polynomial_warning_shown = False

def add_polynomial_regression(ax, x_values, y_values, degree=2, color='green'):
    """
    Ajoute une régression polynomiale au graphique
    """
    global _polynomial_warning_shown
    
    n = len(x_values)  # Nombre de points
    p = degree + 1     # Nombre de paramètres (degré + terme constant)
    
    # Vérifier le risque d'overfitting - n'afficher l'avertissement qu'une seule fois
    if n <= p and not _polynomial_warning_shown:
        print(f"Avertissement: Trop peu de points ({n}) pour une régression de degré {degree} ({p} paramètres).")
        print("Le R² sera artificiellement élevé ou égal à 1.")
        _polynomial_warning_shown = True
    
    # Calcul des coefficients
    coeffs = np.polyfit(x_values, y_values, degree)
    
    # Création d'un polynôme avec ces coefficients
    poly = np.poly1d(coeffs)
    
    # Points x pour tracer une courbe lisse
    x_seq = np.linspace(min(x_values), max(x_values), 100)
    y_seq = poly(x_seq)
    
    # Tracé de la courbe
    ax.plot(x_seq, y_seq, color=color, linestyle='--', linewidth=2)
    
    # Calcul du R² standard
    y_pred = poly(x_values)
    ss_res = np.sum((np.array(y_values) - y_pred)**2)
    ss_tot = np.sum((np.array(y_values) - np.mean(y_values))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Création de l'équation pour l'étiquette
    equation = "y = "
    for i, coef in enumerate(coeffs):
        power = degree - i
        if power == 0:
            equation += f"{coef:.4f}"
        elif power == 1:
            equation += f"{coef:.4f}x + "
        else:
            equation += f"{coef:.4f}x^{power} + "
    
    return equation, r_squared

def add_exponential_regression(ax, x_values, y_values, color='purple'):
    """
    Ajoute une régression exponentielle (y = a*exp(b*x))
    """
    # Ajustement pour les valeurs négatives ou nulles
    valid_indices = [i for i, y in enumerate(y_values) if y > 0]
    if len(valid_indices) < len(y_values):
        print("Avertissement: Certains points ignorés pour la régression exponentielle (valeurs ≤ 0)")
    
    x_valid = [x_values[i] for i in valid_indices]
    y_valid = [y_values[i] for i in valid_indices]
    
    if len(x_valid) < 2:
        return None, None
    
    # Transformation logarithmique pour ajustement linéaire
    log_y = np.log(y_valid)
    params, cov = np.polyfit(x_valid, log_y, 1, cov=True)
    
    # Extraction des paramètres
    a = np.exp(params[1])
    b = params[0]
    
    # Points x pour tracer une courbe lisse
    x_seq = np.linspace(min(x_values), max(x_values), 100)
    y_seq = a * np.exp(b * x_seq)
    
    # Tracé de la courbe
    ax.plot(x_seq, y_seq, color=color, linestyle='-.', linewidth=2)
    
    # Calcul du R²
    y_pred = a * np.exp(b * np.array(x_valid))
    r_squared = 1 - np.sum((np.array(y_valid) - y_pred)**2) / np.sum((np.array(y_valid) - np.mean(y_valid))**2)
    
    return f"y = {a:.4f}·e^({b:.4f}x)", r_squared

def add_logarithmic_regression(ax, x_values, y_values, color='orange'):
    """
    Ajoute une régression logarithmique (y = a + b*ln(x))
    """
    # Ajustement pour les valeurs négatives ou nulles
    valid_indices = [i for i, x in enumerate(x_values) if x > 0]
    if len(valid_indices) < len(x_values):
        print("Avertissement: Certains points ignorés pour la régression logarithmique (valeurs x ≤ 0)")
    
    x_valid = [x_values[i] for i in valid_indices]
    y_valid = [y_values[i] for i in valid_indices]
    
    if len(x_valid) < 2:
        return None, None
    
    # Transformation logarithmique
    log_x = np.log(x_valid)
    params = np.polyfit(log_x, y_valid, 1)
    
    # Extraction des paramètres
    a = params[1]
    b = params[0]
    
    # Points pour courbe lisse
    x_seq = np.linspace(min(x_valid), max(x_valid), 100)
    y_seq = a + b * np.log(x_seq)
    
    # Tracé de la courbe
    ax.plot(x_seq, y_seq, color=color, linestyle=':', linewidth=2)
    
    # Calcul du R²
    y_pred = a + b * np.log(np.array(x_valid))
    r_squared = 1 - np.sum((np.array(y_valid) - y_pred)**2) / np.sum((np.array(y_valid) - np.mean(y_valid))**2)
    
    return f"y = {a:.4f} + {b:.4f}·ln(x)", r_squared

def create_correlation_plots(data_list, regressions=['linear', 'polynomial', 'exponential', 'logarithmic']):
    """
    Crée des graphiques de corrélation entre le potentiel global et ses contributions
    avec différents types de régressions
    
    Args:
        data_list: Liste de dictionnaires contenant les données des potentiels
        regressions: Liste des types de régressions à inclure 
                    (options: 'linear', 'polynomial', 'exponential', 'logarithmic')
    """
    # Réinitialiser l'avertissement polynomial au début
    global _polynomial_warning_shown
    _polynomial_warning_shown = False
    
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
    
    # Styles de couleurs et de lignes pour les différentes régressions
    colors = {
        'linear': 'red',
        'polynomial': 'green',
        'exponential': 'purple',
        'logarithmic': 'orange'
    }
    
    linestyles = {
        'linear': '-',
        'polynomial': '--',
        'exponential': '-.',
        'logarithmic': ':'
    }
    
    # Création des graphiques pour chaque contribution
    for contrib_key, contrib_label in contributions.items():
        # Extraction des données pour cette contribution
        x_values = [data[contrib_key] for data in data_list if contrib_key in data]
        y_values = [data['global_potential'] for data in data_list if contrib_key in data]
        labels = [data['molecule'] for data in data_list if contrib_key in data]
        
        if not x_values or not y_values:
            print(f"Données insuffisantes pour la contribution {contrib_key}")
            continue
        
        # Conversion en numpy arrays pour les calculs
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Tracé des points avec les noms des molécules
        ax.scatter(x_values, y_values, s=80, alpha=0.7, c='blue', edgecolors='black')
        
        # Ajout des étiquettes pour les points
        for i, label in enumerate(labels):
            ax.annotate(label, (x_values[i], y_values[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, alpha=0.8)
        
        # Liste pour stocker les lignes et labels de la légende
        legend_handles = []
        legend_labels = []
        
        # 1. Régression linéaire
        if 'linear' in regressions:
            equation, r_squared = add_linear_regression(
                ax, x_values, y_values, 
                color=colors['linear']
            )
            
            # Ajouter une ligne à la légende
            from matplotlib.lines import Line2D
            legend_handles.append(Line2D([0], [0], color=colors['linear'], 
                                        linestyle=linestyles['linear'], linewidth=2))
            legend_labels.append(f'Linéaire: {equation}, R² = {r_squared:.4f}')
        
        # 2. Régression polynomiale
        if 'polynomial' in regressions:
            equation, r_squared = add_polynomial_regression(
                ax, x_values, y_values, 
                degree=2, 
                color=colors['polynomial']
            )
            
            # Ajouter une ligne à la légende
            from matplotlib.lines import Line2D
            legend_handles.append(Line2D([0], [0], color=colors['polynomial'], 
                                        linestyle=linestyles['polynomial'], linewidth=2))
            legend_labels.append(f'Quadratique: {equation}, R² = {r_squared:.4f}')
        
        # 3. Régression exponentielle
        if 'exponential' in regressions:
            result = add_exponential_regression(
                ax, x_values, y_values, 
                color=colors['exponential']
            )
            
            if result is not None and None not in result:
                equation, r_squared = result
                # Ajouter une ligne à la légende
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([0], [0], color=colors['exponential'], 
                                            linestyle=linestyles['exponential'], linewidth=2))
                legend_labels.append(f'Exponentielle: {equation}, R² = {r_squared:.4f}')
            else:
                print(f"Avertissement: Régression exponentielle ignorée pour {contrib_key}")
        
        # 4. Régression logarithmique
        if 'logarithmic' in regressions:
            result = add_logarithmic_regression(
                ax, x_values, y_values, 
                color=colors['logarithmic']
            )
            
            if result is not None and None not in result:
                equation, r_squared = result
                # Ajouter une ligne à la légende
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([0], [0], color=colors['logarithmic'], 
                                            linestyle=linestyles['logarithmic'], linewidth=2))
                legend_labels.append(f'Logarithmique: {equation}, R² = {r_squared:.4f}')
            else:
                print(f"Avertissement: Régression logarithmique ignorée pour {contrib_key}")
        
        # Configuration des axes et légendes
        ax.set_xlabel(contrib_label, fontsize=14)
        ax.set_ylabel('E(redox) (V)', fontsize=14)
        ax.set_title(f'Corrélation entre {contrib_label} et le potentiel global', fontsize=16)
        
        # Utiliser les handles personnalisés pour la légende
        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=10)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()
        
        # Enregistrement du graphique
        output_filename = output_dir / f"correlation_{contrib_key}.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Graphique enregistré: {output_filename}")
        
        plt.close(fig)
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse des potentiels redox")
    parser.add_argument('paths', nargs='+', help="Chemins vers les dossiers de redox à analyser")
    parser.add_argument('--regressions', nargs='+', 
                        choices=['linear', 'polynomial', 'exponential', 'logarithmic'],
                        default=['linear', 'polynomial', 'exponential', 'logarithmic'], 
                        help="Types de régressions à inclure dans les graphiques")
    
    args = parser.parse_args()
    
    print(f"Analyse des potentiels redox pour {len(args.paths)} chemins...")
    print(f"Régressions sélectionnées: {', '.join(args.regressions)}")
    
    # Collecte des données
    data_list = collect_data(args.paths)
    
    if not data_list:
        print("Aucune donnée n'a pu être extraite. Vérifiez les chemins fournis.")
        sys.exit(1)
    
    print(f"Données extraites pour {len(data_list)} molécules.")
    
    # Création des graphiques de corrélation
    create_correlation_plots(data_list, regressions=args.regressions)

if __name__ == "__main__":
    main()
