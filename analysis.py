#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import glob
from matplotlib.ticker import FormatStrFormatter

def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        argparse.Namespace: Les arguments analysés.
    """
    parser = argparse.ArgumentParser(description='Analyse des corrélations dans les potentiels redox.')
    parser.add_argument('workdirs', nargs='+', help='Chemins vers les dossiers de travail contenant les résultats redox.')
    parser.add_argument('--regression', choices=['linear', 'polynomial', 'exponential', 'logarithmic', 'all'],
                        default='all', help='Type de régression à appliquer. "all" pour toutes les régressions.')
    args = parser.parse_args()
    
    # Convertir en liste de régressions si l'utilisateur spécifie 'all'
    if args.regression == 'all':
        args.regression = ['linear', 'polynomial', 'exponential', 'logarithmic']
    else:
        args.regression = [args.regression]
        
    return args

def parse_redox_file(filepath):
    """
    Extrait les valeurs d'énergie d'un fichier redox_potentials.txt.
    
    Args:
        filepath (str): Chemin vers le fichier redox_potentials.txt.
        
    Returns:
        dict: Dictionnaire contenant les valeurs extraites.
    """
    if not os.path.exists(filepath):
        print(f"Erreur: Le fichier {filepath} n'existe pas.")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Dictionnaire pour stocker les valeurs extraites
    data = {}
    
    # Extraction des potentiels de réduction
    red_pattern = r"POTENTIELS DE RÉDUCTION.*?\n.*?\nE\(∆G\) = ([-\d.]+) V.*?\n+.*?\nE\(EA\) = ([-\d.]+) V\nE\(Edef\) = ([-\d.]+) V\nE\(∆∆U\) = ([-\d.]+) V\nE\(T∆S\) = ([-\d.]+) V"
    red_match = re.search(red_pattern, content, re.DOTALL)
    
    if red_match:
        data['E_red_deltaG'] = float(red_match.group(1))
        data['E_red_EA'] = float(red_match.group(2))
        data['E_red_Edef'] = float(red_match.group(3))
        data['E_red_deltaU'] = float(red_match.group(4))
        data['E_red_TdeltaS'] = float(red_match.group(5))
    
    # Extraction des potentiels d'oxydation
    ox_pattern = r"POTENTIELS D'OXYDATION.*?\n.*?\nE\(∆G\) = ([-\d.]+) V.*?\n+.*?\nE\(EI\) = ([-\d.]+) V\nE\(Edef\) = ([-\d.]+) V\nE\(∆∆U\) = ([-\d.]+) V\nE\(T∆S\) = ([-\d.]+) V"
    ox_match = re.search(ox_pattern, content, re.DOTALL)
    
    if ox_match:
        data['E_ox_deltaG'] = float(ox_match.group(1))
        data['E_ox_EI'] = float(ox_match.group(2))
        data['E_ox_Edef'] = float(ox_match.group(3))
        data['E_ox_deltaU'] = float(ox_match.group(4))
        data['E_ox_TdeltaS'] = float(ox_match.group(5))
    
    # Extraction des valeurs d'orbitales moléculaires
    mo_pattern = r"Moyennes pondérées:\nHOMO: ([-\d.]+) eV\nLUMO: ([-\d.]+) eV"
    mo_match = re.search(mo_pattern, content, re.DOTALL)
    
    if mo_match:
        data['HOMO'] = float(mo_match.group(1))
        data['LUMO'] = float(mo_match.group(2))
    
    # Vérification des données extraites
    expected_keys = ['E_red_deltaG', 'E_red_EA', 'E_red_Edef', 'E_red_deltaU', 'E_red_TdeltaS',
                     'E_ox_deltaG', 'E_ox_EI', 'E_ox_Edef', 'E_ox_deltaU', 'E_ox_TdeltaS',
                     'HOMO', 'LUMO']
    
    missing_keys = [key for key in expected_keys if key not in data]
    if missing_keys:
        print(f"Avertissement: Les données suivantes n'ont pas pu être extraites: {', '.join(missing_keys)}")
    
    # Extraire le nom de la molécule depuis le chemin
    mol_name = filepath.split("_")
    data['molecule'] = mol_name[0]
    
    return data

def add_linear_regression(ax, x, y, color='red', label='Linéaire'):
    """
    Ajoute une régression linéaire au graphique.
    
    Args:
        ax (matplotlib.axes.Axes): Axes du graphique.
        x (numpy.ndarray): Données de l'axe X.
        y (numpy.ndarray): Données de l'axe Y.
        color (str): Couleur de la ligne de régression.
        label (str): Libellé de la légende.
        
    Returns:
        tuple: (R², MAE) de la régression.
    """
    # Reshape pour sklearn
    X = x.reshape(-1, 1)
    
    # Créer et entraîner le modèle
    model = LinearRegression()
    model.fit(X, y)
    
    # Prédiction sur une grille fine pour le tracé
    x_grid = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    y_pred = model.predict(x_grid)
    
    # Calcul des métriques
    y_pred_original = model.predict(X)
    r2 = r2_score(y, y_pred_original)
    mae = mean_absolute_error(y, y_pred_original)
    
    # Traçage de la ligne de régression
    ax.plot(x_grid, y_pred, color=color, linestyle='-', 
            label=f'{label}: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}\nR² = {r2:.4f}, MAE = {mae:.4f}')
    
    return (r2, mae)

def add_polynomial_regression(ax, x, y, degree=2, color='green', label='Quadratique'):
    """
    Ajoute une régression polynomiale au graphique.
    
    Args:
        ax (matplotlib.axes.Axes): Axes du graphique.
        x (numpy.ndarray): Données de l'axe X.
        y (numpy.ndarray): Données de l'axe Y.
        degree (int): Degré du polynôme.
        color (str): Couleur de la ligne de régression.
        label (str): Libellé de la légende.
        
    Returns:
        tuple: (R², MAE) de la régression.
    """
    # Reshape pour sklearn
    X = x.reshape(-1, 1)
    
    # Créer et entraîner le modèle
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    # Prédiction sur une grille fine pour le tracé
    x_grid = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    y_pred = model.predict(x_grid)
    
    # Calcul des métriques
    y_pred_original = model.predict(X)
    r2 = r2_score(y, y_pred_original)
    mae = mean_absolute_error(y, y_pred_original)

    # Extraire les coefficients pour l'affichage
    coefficients = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    
    # Formater l'équation pour l'affichage (pour un polynôme de degré 2)
    if degree == 2:
        equation = f'y = {coefficients[2]:.4f}x² + {coefficients[1]:.4f}x + {intercept:.4f}'
    else:
        equation = f'Polynôme de degré {degree}'
    
    # Traçage de la ligne de régression
    ax.plot(x_grid, y_pred, color=color, linestyle='--', 
            label=f'{label}: {equation}\nR² = {r2:.4f}, MAE = {mae:.4f}')
    
    return (r2, mae)

def add_exponential_regression(ax, x, y, color='purple', label='Exponentielle'):
    """
    Ajoute une régression exponentielle au graphique.
    
    Args:
        ax (matplotlib.axes.Axes): Axes du graphique.
        x (numpy.ndarray): Données de l'axe X.
        y (numpy.ndarray): Données de l'axe Y.
        color (str): Couleur de la ligne de régression.
        label (str): Libellé de la légende.
        
    Returns:
        tuple: (R², MAE) de la régression.
    """
    # Vérifier que toutes les valeurs y sont positives
    if np.any(y <= 0):
        print("Avertissement: Régression exponentielle impossible avec des valeurs négatives ou nulles.")
        return (np.nan, np.nan)
    
    # Transformation logarithmique
    log_y = np.log(y)
    X = x.reshape(-1, 1)
    
    # Régression linéaire sur les données transformées
    model = LinearRegression()
    model.fit(X, log_y)
    
    # Coefficients du modèle exponentiel y = a * exp(b * x)
    a = np.exp(model.intercept_)
    b = model.coef_[0]
    
    # Prédiction sur une grille fine pour le tracé
    x_grid = np.linspace(min(x), max(x), 100)
    y_pred = a * np.exp(b * x_grid)
    
    # Prédiction sur les données originales pour calcul de métriques
    y_pred_original = a * np.exp(b * x)
    r2 = r2_score(y, y_pred_original)
    mae = mean_absolute_error(y, y_pred_original)
    
    # Traçage de la ligne de régression
    ax.plot(x_grid, y_pred, color=color, linestyle='-.', 
            label=f'{label}: y = {a:.4f} exp({b:.4f}x)\nR² = {r2:.4f}, MAE = {mae:.4f}')
    
    return (r2, mae)

def add_logarithmic_regression(ax, x, y, color='orange', label='Logarithmique'):
    """
    Ajoute une régression logarithmique au graphique.
    
    Args:
        ax (matplotlib.axes.Axes): Axes du graphique.
        x (numpy.ndarray): Données de l'axe X.
        y (numpy.ndarray): Données de l'axe Y.
        color (str): Couleur de la ligne de régression.
        label (str): Libellé de la légende.
        
    Returns:
        tuple: (R², MAE) de la régression.
    """
    # Vérifier que toutes les valeurs x sont positives
    if np.any(x <= 0):
        print("Avertissement: Régression logarithmique impossible avec des valeurs x négatives ou nulles.")
        return (np.nan, np.nan)
    
    # Transformation logarithmique
    log_x = np.log(x).reshape(-1, 1)
    
    # Régression linéaire sur les données transformées
    model = LinearRegression()
    model.fit(log_x, y)
    
    # Coefficients du modèle logarithmique y = a + b * ln(x)
    a = model.intercept_
    b = model.coef_[0]
    
    # Prédiction sur une grille fine pour le tracé
    x_grid = np.linspace(min(x), max(x), 100)
    y_pred = a + b * np.log(x_grid)
    
    # Prédiction sur les données originales pour calcul de métriques
    y_pred_original = a + b * np.log(x)
    r2 = r2_score(y, y_pred_original)
    mae = mean_absolute_error(y, y_pred_original)
    
    # Traçage de la ligne de régression
    ax.plot(x_grid, y_pred, color=color, linestyle=':', 
            label=f'{label}: y = {a:.4f} + {b:.4f}ln(x)\nR² = {r2:.4f}, MAE = {mae:.4f}')
    
    return (r2, mae)

def create_correlation_plot(data_list, output_dir, regression_types):
    """
    Crée des graphiques de corrélation pour différentes paires de données.
    
    Args:
        data_list (list): Liste de dictionnaires contenant les données extraites.
        output_dir (str): Répertoire de sortie pour les graphiques.
        regression_types (list): Liste des types de régression à appliquer.
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Définir les corrélations à analyser
    correlations = [
        {'x_key': 'E_red_deltaU', 'y_key': 'E_red_deltaG', 'title': 'Corrélation ΔΔU vs ΔE (Réduction)', 
         'x_label': 'ΔΔU (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'E_ox_deltaU', 'y_key': 'E_ox_deltaG', 'title': 'Corrélation ΔΔU vs ΔE (Oxydation)',
         'x_label': 'ΔΔU (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'E_red_TdeltaS', 'y_key': 'E_red_deltaG', 'title': 'Corrélation TΔS vs ΔE (Réduction)',
         'x_label': 'TΔS (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'E_ox_TdeltaS', 'y_key': 'E_ox_deltaG', 'title': 'Corrélation TΔS vs ΔE (Oxydation)',
         'x_label': 'TΔS (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'E_red_EA', 'y_key': 'E_red_deltaG', 'title': 'Corrélation EA vs ΔE (Réduction)',
         'x_label': 'EA (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'E_ox_EI', 'y_key': 'E_ox_deltaG', 'title': 'Corrélation EI vs ΔE (Oxydation)',
         'x_label': 'EI (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'E_red_Edef', 'y_key': 'E_red_deltaG', 'title': 'Corrélation Edef vs ΔE (Réduction)',
         'x_label': 'Edef (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'E_ox_Edef', 'y_key': 'E_ox_deltaG', 'title': 'Corrélation Edef vs ΔE (Oxydation)',
         'x_label': 'Edef (V)', 'y_label': 'ΔE (V)'},
        {'x_key': 'HOMO', 'y_key': 'E_ox_EI', 'title': 'Corrélation HOMO vs EI',
         'x_label': 'HOMO (eV)', 'y_label': 'EI (V)'},
        {'x_key': 'LUMO', 'y_key': 'E_red_EA', 'title': 'Corrélation LUMO vs EA',
         'x_label': 'LUMO (eV)', 'y_label': 'EA (V)'}
    ]
    
    # Couleurs pour les différents types de régression
    regression_colors = {
        'linear': 'red',
        'polynomial': 'green',
        'exponential': 'purple',
        'logarithmic': 'orange'
    }
    
    # Labels pour les différents types de régression
    regression_labels = {
        'linear': 'Linéaire',
        'polynomial': 'Quadratique',
        'exponential': 'Exponentielle',
        'logarithmic': 'Logarithmique'
    }
    
    # Fonctions de régression
    regression_functions = {
        'linear': add_linear_regression,
        'polynomial': add_polynomial_regression,
        'exponential': add_exponential_regression,
        'logarithmic': add_logarithmic_regression
    }
    
    # Créer un graphique pour chaque corrélation
    for corr in correlations:
        # Extraire les données pour cette corrélation
        x_values = []
        y_values = []
        molecules = []
        
        for data in data_list:
            if corr['x_key'] in data and corr['y_key'] in data:
                x_values.append(data[corr['x_key']])
                y_values.append(data[corr['y_key']])
                molecules.append(data['molecule'])
        
        if len(x_values) < 3:
            print(f"Pas assez de données pour la corrélation {corr['title']}")
            continue
            
        # Conversion en arrays numpy
        x = np.array(x_values)
        y = np.array(y_values)
        
        # Création de la figure et des axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tracer les points
        sc = ax.scatter(x, y, c='black', s=50, alpha=0.7)
        
        # Ajouter les annotations avec les noms des molécules
        for i, mol in enumerate(molecules):
            ax.annotate(mol, (x[i], y[i]), fontsize=8, 
                        xytext=(5, 5), textcoords='offset points')
        
        # Appliquer les régressions sélectionnées
        for reg_type in regression_types:
            if reg_type in regression_functions:
                # Gérer les cas spéciaux
                if reg_type == 'exponential' and np.any(y <= 0):
                    print(f"Avertissement: Régression exponentielle impossible pour {corr['title']} à cause de valeurs Y négatives.")
                    continue
                if reg_type == 'logarithmic' and np.any(x <= 0):
                    print(f"Avertissement: Régression logarithmique impossible pour {corr['title']} à cause de valeurs X négatives.")
                    continue
                    
                # Appliquer la régression
                r2, mae = regression_functions[reg_type](
                    ax, x, y, 
                    color=regression_colors[reg_type], 
                    label=regression_labels[reg_type]
                )
                
                # Si la régression a échoué (NaN), ne pas l'inclure dans la légende
                if np.isnan(r2) or np.isnan(mae):
                    continue
        
        # Configurer le graphique
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams["legend.framealpha"] = 0.5
        ax.set_xlabel(corr['x_label'], fontsize=12)
        ax.set_ylabel(corr['y_label'], fontsize=12)
        ax.set_title(corr['title'], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, loc='upper right')
        
        # Ajuster les formats d'axes
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Ajuster les marges
        plt.tight_layout()
        
        # Sauvegarder le graphique
        filename = os.path.join(output_dir, f"corr_{corr['x_key']}_{corr['y_key']}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {filename}")
        
        # Fermer la figure pour libérer la mémoire
        plt.close(fig)
    
    print(f"Tous les graphiques ont été générés dans le dossier: {output_dir}")

def main():
    """Fonction principale du programme."""
    # Analyser les arguments
    args = parse_arguments()
    
    # Créer un dossier pour les résultats
    output_dir = "redox_correlations"
    
    # Collecter les données de tous les fichiers
    all_data = []
    for workdir in args.workdirs:
        redox_file = os.path.join(workdir, "redox_results", "redox_potentials.txt")
        if os.path.exists(redox_file):
            data = parse_redox_file(redox_file)
            if data:
                all_data.append(data)
        else:
            # Essayer de trouver des fichiers redox_potentials.txt dans des sous-dossiers
            redox_files = glob.glob(os.path.join(workdir, "**/redox_potentials.txt"), recursive=True)
            if redox_files:
                for file in redox_files:
                    data = parse_redox_file(file)
                    if data:
                        all_data.append(data)
            else:
                print(f"Aucun fichier redox_potentials.txt trouvé dans {workdir}")
    
    # Vérifier que nous avons des données
    if not all_data:
        print("Aucune donnée n'a été trouvée. Vérifiez les chemins fournis.")
        return
    
    print(f"Données extraites pour {len(all_data)} molécules.")
    
    # Générer les graphiques avec les régressions spécifiées
    create_correlation_plot(all_data, output_dir, args.regression)

if __name__ == "__main__":
    main()

