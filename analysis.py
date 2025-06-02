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
    parser = argparse.ArgumentParser(description='Analyse des corrélations dans les potentiels redox et données QTAIM.')
    parser.add_argument('workdirs', nargs='+', help='Chemins vers les dossiers de travail contenant les résultats redox et QTAIM.')
    parser.add_argument('--regression', choices=['linear', 'polynomial', 'exponential', 'logarithmic', 'all'],
                        default='all', help='Type de régression à appliquer. "all" pour toutes les régressions.')
    args = parser.parse_args()
    
    # Convertir en liste de régressions si l'utilisateur spécifie 'all'
    if args.regression == 'all':
        args.regression = ['linear', 'polynomial', 'exponential', 'logarithmic']
    else:
        args.regression = [args.regression]
        
    return args

def parse_qtaim_charge_file(filepath):
    """
    Extrait les statistiques de charges d'un fichier QTAIM charge_statistics.txt.
    
    Args:
        filepath (str): Chemin vers le fichier *_charge_statistics.txt.
        
    Returns:
        dict: Dictionnaire contenant les valeurs extraites.
    """
    if not os.path.exists(filepath):
        print(f"Erreur: Le fichier {filepath} n'existe pas.")
        return {}
        
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    qtaim_data = {}
    
    # Extraction des statistiques d'oxydation
    ox_pattern = r"STATISTIQUES D'OXYDATION\s*\n-+\s*\n" \
                r"Maximum positif:\s*([-\d.]+)\s+sur l'atome\s+\w+\s*\n" \
                r"Changement maximal \(abs\):\s*([-\d.]+)\s+sur l'atome\s+\w+\s*\n" \
                r"Moyenne des charges positives:\s*([-\d.]+)\s+\(sur \d+ atomes\)\s*\n" \
                r"Moyenne des charges négatives:\s*([-\d.]+)\s+\(sur \d+ atomes\)\s*\n" \
                r"Écart-type:\s*([-\d.]+)\s*\n"
    
    ox_match = re.search(ox_pattern, content, re.MULTILINE)
    if ox_match:
        qtaim_data['qtaim_ox_max'] = float(ox_match.group(1))
        qtaim_data['qtaim_ox_max_abs'] = float(ox_match.group(2))
        qtaim_data['qtaim_ox_moy_pos'] = float(ox_match.group(3))
        qtaim_data['qtaim_ox_moy_neg'] = float(ox_match.group(4))
        qtaim_data['qtaim_ox_ecart_type'] = float(ox_match.group(5))
    
    # Extraction des statistiques de réduction
    red_pattern = r"STATISTIQUES DE RÉDUCTION\s*\n-+\s*\n" \
                 r"Maximum négatif:\s*([-\d.]+)\s+sur l'atome\s+\w+\s*\n" \
                 r"Changement maximal \(abs\):\s*([-\d.]+)\s+sur l'atome\s+\w+\s*\n" \
                 r"Moyenne des charges négatives:\s*([-\d.]+)\s+\(sur \d+ atomes\)\s*\n" \
                 r"Moyenne des charges positives:\s*([-\d.]+)\s+\(sur \d+ atomes\)\s*\n" \
                 r"Écart-type:\s*([-\d.]+)\s*\n"
    
    red_match = re.search(red_pattern, content, re.MULTILINE)
    if red_match:
        qtaim_data['qtaim_red_max'] = float(red_match.group(1))
        qtaim_data['qtaim_red_max_abs'] = float(red_match.group(2))
        qtaim_data['qtaim_red_moy_neg'] = float(red_match.group(3))
        qtaim_data['qtaim_red_moy_pos'] = float(red_match.group(4))
        qtaim_data['qtaim_red_ecart_type'] = float(red_match.group(5))
    
    return qtaim_data

def aggregate_qtaim_data(qtaim_files):
    """
    Agrège les données QTAIM de plusieurs conformères en calculant des moyennes pondérées.
    
    Args:
        qtaim_files (list): Liste des chemins vers les fichiers QTAIM.
        
    Returns:
        dict: Données QTAIM agrégées.
    """
    if not qtaim_files:
        return {}
    
    all_qtaim_data = []
    for file in qtaim_files:
        data = parse_qtaim_charge_file(file)
        if data:
            all_qtaim_data.append(data)
    
    if not all_qtaim_data:
        return {}
    
    # Pour simplifier, on prend la moyenne de tous les conformères
    # (vous pouvez modifier cette logique pour utiliser des poids spécifiques)
    aggregated_data = {}
    
    # Liste des clés à agréger
    qtaim_keys = ['qtaim_ox_max', 'qtaim_ox_max_abs', 'qtaim_ox_moy_pos', 'qtaim_ox_moy_neg', 'qtaim_ox_ecart_type',
                  'qtaim_red_max', 'qtaim_red_max_abs', 'qtaim_red_moy_neg', 'qtaim_red_moy_pos', 'qtaim_red_ecart_type']
    
    for key in qtaim_keys:
        values = [data[key] for data in all_qtaim_data if key in data]
        if values:
            aggregated_data[key] = np.mean(values)
    
    return aggregated_data

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

    # Extraction des poids des conformères
    conf_pattern = r"Conform.*?Poids.*?\n-+\n([a-zA-Z\d_\s.-]+)\n\n\n"
    conf_match = re.search(conf_pattern, content, re.DOTALL)

    if conf_match:
        # Extraire les lignes de conformères et leurs poids
        conf_lines = conf_match.group(1).strip().split('\n')
        weights = []
        conf_names = []

        for line in conf_lines:
            parts = line.split()
            if len(parts) >= 4:  # Nom, poids, E_red, E_ox
                conf_names.append(parts[0])
                weights.append(float(parts[1]))

        # Trouver l'indice du conformère avec le poids le plus élevé
        if weights:
            max_weight_idx = weights.index(max(weights))
            data['max_weight_conformer'] = conf_names[max_weight_idx]
            data['max_weight'] = weights[max_weight_idx]

    # Extraction des RMSD des conformères réduits
    rmsd_red_pattern = r"CONFORMÈRES RÉDUITS\n.+\s+\n[-]+\n((?:.*?\s+.*?\s+[\d.]+\n)+)"
    rmsd_red_match = re.search(rmsd_red_pattern, content, re.DOTALL)

    if rmsd_red_match:
        rmsd_red_lines = rmsd_red_match.group(1).strip().split('\n')
        rmsd_red_values = {}

        for line in rmsd_red_lines:
            parts = line.split()
            if len(parts) >= 3:  # conf_red, conf_neutre, rmsd
                conf_red = parts[0].strip()
                conf_neutre = parts[1].strip()
                rmsd = float(parts[2])
                rmsd_red_values[conf_red] = (conf_neutre, rmsd)

        # Si nous avons un conformère de poids maximal, trouver son RMSD
        if 'max_weight_conformer' in data:
            # Trouver le conformère réduit qui correspond au conformère neutre de poids maximal
            best_rmsd = None
            for conf_red, (conf_neutre, rmsd) in rmsd_red_values.items():
                # Vérifier si ce conformère réduit correspond au conformère neutre de poids maximal
                if data['max_weight_conformer'].endswith(conf_neutre):
                    if best_rmsd is None or rmsd < best_rmsd:
                        best_rmsd = rmsd

            if best_rmsd is not None:
                data['rmsd_red'] = best_rmsd

    # Extraction des RMSD des conformères oxydés  
    rmsd_ox_pattern = r"CONFORMÈRES OXIDÉS\n.+\s+\n[-]+\n((?:.*?\s+.*?\s+[\d.]+\n)+)"
    rmsd_ox_match = re.search(rmsd_ox_pattern, content, re.DOTALL)

    if rmsd_ox_match:
        rmsd_ox_lines = rmsd_ox_match.group(1).strip().split('\n')
        rmsd_ox_values = {}
        
        for line in rmsd_ox_lines:
            parts = line.split()
            if len(parts) >= 3:  # conf_ox, conf_neutre, rmsd
                conf_ox = parts[0].strip()
                conf_neutre = parts[1].strip()
                rmsd = float(parts[2])
                rmsd_ox_values[conf_ox] = (conf_neutre, rmsd)

        # Si nous avons un conformère de poids maximal, trouver son RMSD
        if 'max_weight_conformer' in data:
            # Trouver le conformère oxydé qui correspond au conformère neutre de poids maximal
            best_rmsd = None
            for conf_ox, (conf_neutre, rmsd) in rmsd_ox_values.items():
                # Vérifier si ce conformère oxydé correspond au conformère neutre de poids maximal
                if data['max_weight_conformer'].endswith(conf_neutre):
                    if best_rmsd is None or rmsd < best_rmsd:
                        best_rmsd = rmsd

            if best_rmsd is not None:
                data['rmsd_ox'] = best_rmsd

    # Extraire le nom de la molécule depuis le chemin
    mol_name = filepath.split("_")[0]
    data['molecule'] = mol_name

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
    
    # Définir les corrélations à analyser (inclut maintenant les données QTAIM)
    correlations = [
        # Corrélations redox existantes
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
        {'x_key': 'HOMO', 'y_key': 'E_ox_deltaG', 'title': 'Corrélation HOMO vs ΔE (Oxydation)',
         'x_label': 'HOMO (eV)', 'y_label': 'ΔE (V)'},
        {'x_key': 'LUMO', 'y_key': 'E_red_deltaG', 'title': 'Corrélation LUMO vs ΔE (Réduction)',
         'x_label': 'LUMO (eV)', 'y_label': 'ΔE (V)'},
        {'x_key': 'HOMO', 'y_key': 'E_ox_EI', 'title': 'Corrélation HOMO vs EI',
         'x_label': 'HOMO (eV)', 'y_label': 'EI (V)'},
        {'x_key': 'LUMO', 'y_key': 'E_red_EA', 'title': 'Corrélation LUMO vs EA',
         'x_label': 'LUMO (eV)', 'y_label': 'EA (V)'},
        {'x_key': 'rmsd_red', 'y_key': 'E_red_Edef', 'title': 'Corrélation RMSD vs Edef (Réduction)',
         'x_label': 'RMSD (Å)', 'y_label': 'Edef (V)'},
        {'x_key': 'rmsd_ox', 'y_key': 'E_ox_Edef', 'title': 'Corrélation RMSD vs Edef (Oxydation)',
         'x_label': 'RMSD (Å)', 'y_label': 'Edef (V)'}, 

        # Nouvelles corrélations avec les données QTAIM
        {'x_key': 'qtaim_ox_max_abs', 'y_key': 'E_ox_deltaG', 'title': 'Corrélation QTAIM Max Abs (Ox) vs ΔE',
         'x_label': 'QTAIM Max Abs Charge (Ox)', 'y_label': 'ΔE (V)'},
        {'x_key': 'qtaim_red_max_abs', 'y_key': 'E_red_deltaG', 'title': 'Corrélation QTAIM Max Abs (Red) vs ΔE',
         'x_label': 'QTAIM Max Abs Charge (Red)', 'y_label': 'ΔE (V)'},
        {'x_key': 'qtaim_ox_ecart_type', 'y_key': 'E_ox_deltaG', 'title': 'Corrélation QTAIM Écart-type (Ox) vs ΔE',
         'x_label': 'QTAIM Écart-type (Ox)', 'y_label': 'ΔE (V)'},
        {'x_key': 'qtaim_red_ecart_type', 'y_key': 'E_red_deltaG', 'title': 'Corrélation QTAIM Écart-type (Red) vs ΔE',
         'x_label': 'QTAIM Écart-type (Red)', 'y_label': 'ΔE (V)'},
        {'x_key': 'qtaim_ox_moy_pos', 'y_key': 'E_ox_Edef', 'title': 'Corrélation QTAIM Moy Pos (Ox) vs Edef',
         'x_label': 'QTAIM Moyenne Positive (Ox)', 'y_label': 'Edef (V)'},
        {'x_key': 'qtaim_red_moy_neg', 'y_key': 'E_red_Edef', 'title': 'Corrélation QTAIM Moy Neg (Red) vs Edef',
         'x_label': 'QTAIM Moyenne Négative (Red)', 'y_label': 'Edef (V)'},
        {'x_key': 'qtaim_ox_max', 'y_key': 'E_ox_EI', 'title': 'Corrélation QTAIM Max (Ox) vs EI',
         'x_label': 'QTAIM Maximum Charge (Ox)', 'y_label': 'EI (V)'},
        {'x_key': 'qtaim_red_max', 'y_key': 'E_red_EA', 'title': 'Corrélation QTAIM Max (Red) vs EA',
         'x_label': 'QTAIM Maximum Charge (Red)', 'y_label': 'EA (V)'},
        
        # Corrélations croisées QTAIM
        {'x_key': 'qtaim_ox_ecart_type', 'y_key': 'qtaim_red_ecart_type', 'title': 'Corrélation QTAIM Écart-types Ox vs Red',
         'x_label': 'QTAIM Écart-type (Ox)', 'y_label': 'QTAIM Écart-type (Red)'},
        {'x_key': 'qtaim_ox_max_abs', 'y_key': 'qtaim_red_max_abs', 'title': 'Corrélation QTAIM Max Abs Ox vs Red',
         'x_label': 'QTAIM Max Abs (Ox)', 'y_label': 'QTAIM Max Abs (Red)'},
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
    
    # Configuration du style des graphiques
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["legend.framealpha"] = 0.5
    
    # Créer un graphique pour chaque corrélation
    valid_correlations = 0
    for corr in correlations:
        # Extraire les données pour cette corrélation
        x_values = []
        y_values = []
        molecules = []
        
        for data in data_list:
            if corr['x_key'] in data and corr['y_key'] in data:
                x_val = data[corr['x_key']]
                y_val = data[corr['y_key']]
                # Vérifier que les valeurs sont numériques et finies
                if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                    if np.isfinite(x_val) and np.isfinite(y_val):
                        x_values.append(x_val)
                        y_values.append(y_val)
                        molecules.append(data['molecule'])
        
        if len(x_values) < 3:
            print(f"Pas assez de données pour la corrélation {corr['title']} (seulement {len(x_values)} points)")
            continue
            
        # Conversion en arrays numpy
        x = np.array(x_values)
        y = np.array(y_values)
        
        # Création de la figure et des axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tracer les points
        ax.scatter(x, y, c='black', s=50, alpha=0.7, zorder=3)
        
        # Ajouter les annotations avec les noms des molécules
        for i, mol in enumerate(molecules):
            ax.annotate(mol, (x[i], y[i]), fontsize=8, 
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Appliquer les régressions sélectionnées
        regression_applied = False
        for reg_type in regression_types:
            if reg_type in regression_functions:
                # Gérer les cas spéciaux
                if reg_type == 'exponential' and np.any(y <= 0):
                    print(f"Avertissement: Régression exponentielle impossible pour {corr['title']} à cause de valeurs Y négatives ou nulles.")
                    continue
                if reg_type == 'logarithmic' and np.any(x <= 0):
                    print(f"Avertissement: Régression logarithmique impossible pour {corr['title']} à cause de valeurs X négatives ou nulles.")
                    continue
                    
                try:
                    # Appliquer la régression
                    r2, mae = regression_functions[reg_type](
                        ax, x, y, 
                        color=regression_colors[reg_type], 
                        label=regression_labels[reg_type]
                    )
                    
                    # Si la régression a échoué (NaN), ne pas l'inclure
                    if np.isfinite(r2) and np.isfinite(mae):
                        regression_applied = True
                        print(f"  {regression_labels[reg_type]}: R² = {r2:.4f}, MAE = {mae:.4f}")
                    
                except Exception as e:
                    print(f"Erreur lors de l'application de la régression {reg_type} pour {corr['title']}: {e}")
                    continue
        
        # Configurer le graphique
        ax.set_xlabel(corr['x_label'], fontsize=12)
        ax.set_ylabel(corr['y_label'], fontsize=12)
        ax.set_title(corr['title'], fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
        
        # Ajouter la légende seulement si des régressions ont été appliquées
        if regression_applied:
            ax.legend(fontsize=9, loc='best', fancybox=True, shadow=True)
        
        # Ajuster les formats d'axes
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        
        # Améliorer l'apparence
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Ajuster les marges
        plt.tight_layout()
        
        # Sauvegarder le graphique
        safe_filename = f"corr_{corr['x_key']}_{corr['y_key']}.png"
        filename = os.path.join(output_dir, safe_filename)
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Graphique sauvegardé: {filename}")
        
        # Fermer la figure pour libérer la mémoire
        plt.close(fig)
        valid_correlations += 1
    
    print(f"\n{valid_correlations} graphiques ont été générés dans le dossier: {output_dir}")

def main():
    """Fonction principale du programme."""
    # Analyser les arguments
    args = parse_arguments()
    
    # Créer un dossier pour les résultats
    output_dir = "correlations"
    
    # Collecter les données de tous les fichiers
    all_data = []
    for workdir in args.workdirs:
        print(f"\nTraitement du répertoire: {workdir}")
        
        # Chercher le fichier redox_potentials.txt
        redox_file = os.path.join(workdir, "redox_results", "redox_potentials.txt")
        redox_data = None
        
        if os.path.exists(redox_file):
            redox_data = parse_redox_file(redox_file)
            print(f"  Données redox extraites de: {redox_file}")
        else:
            # Essayer de trouver des fichiers redox_potentials.txt dans des sous-dossiers
            redox_files = glob.glob(os.path.join(workdir, "**/redox_potentials.txt"), recursive=True)
            if redox_files:
                redox_data = parse_redox_file(redox_files[0])  # Prendre le premier trouvé
                print(f"  Données redox extraites de: {redox_files[0]}")
            else:
                print(f"  Aucun fichier redox_potentials.txt trouvé dans {workdir}")
        
        # Chercher les fichiers QTAIM
        qtaim_files = glob.glob(os.path.join(workdir, "qtaims", "charge_analysis", "*_charge_statistics.txt"))
        qtaim_data = {}
        
        if qtaim_files:
            qtaim_data = aggregate_qtaim_data(qtaim_files)
            print(f"  Données QTAIM agrégées depuis {len(qtaim_files)} fichiers")
        else:
            print(f"  Aucun fichier QTAIM trouvé dans {workdir}qtaims/charge_analysis/")
        
        # Combiner les données si nous en avons des deux sources
        if redox_data and qtaim_data:
            combined_data = {**redox_data, **qtaim_data}
            all_data.append(combined_data)
            print(f"  Données combinées pour la molécule: {combined_data.get('molecule', 'inconnue')}")
        elif redox_data:
            all_data.append(redox_data)
            print(f"  Seulement les données redox pour: {redox_data.get('molecule', 'inconnue')}")
        elif qtaim_data:
            # Extraire le nom de la molécule du chemin si possible
            mol_name = os.path.basename(workdir)
            qtaim_data['molecule'] = mol_name
            all_data.append(qtaim_data)
            print(f"  Seulement les données QTAIM pour: {mol_name}")
    
    # Vérifier que nous avons des données
    if not all_data:
        print("\nAucune donnée n'a été trouvée. Vérifiez les chemins fournis.")
        return
    
    print(f"\nDonnées extraites pour {len(all_data)} molécules.")
    
    # Afficher un résumé des données disponibles
    print("\nRésumé des données disponibles:")
    qtaim_keys = ['qtaim_ox_max', 'qtaim_ox_max_abs', 'qtaim_ox_moy_pos', 'qtaim_ox_moy_neg', 'qtaim_ox_ecart_type',
                  'qtaim_red_max', 'qtaim_red_max_abs', 'qtaim_red_moy_neg', 'qtaim_red_moy_pos', 'qtaim_red_ecart_type']
    redox_keys = ['E_red_deltaG', 'E_ox_deltaG', 'HOMO', 'LUMO']
    
    qtaim_count = sum(1 for data in all_data if any(key in data for key in qtaim_keys))
    redox_count = sum(1 for data in all_data if any(key in data for key in redox_keys))
    
    print(f"  Molécules avec données QTAIM: {qtaim_count}")
    print(f"  Molécules avec données redox: {redox_count}")
    print(f"  Molécules avec les deux types: {len([data for data in all_data if any(key in data for key in qtaim_keys) and any(key in data for key in redox_keys)])}")
    
    # Générer les graphiques avec les régressions spécifiées
    print(f"\nGénération des graphiques avec les régressions: {', '.join(args.regression)}")
    create_correlation_plot(all_data, output_dir, args.regression)

if __name__ == "__main__":
    main()

