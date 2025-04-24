#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import traceback
import logging

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redox_analysis.log"),
        logging.StreamHandler()
    ]
)

# Variable globale pour le traçage des avertissements
_polynomial_warning_shown = False

# Configuration de matplotlib pour des graphiques de qualité publication
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
except Exception as e:
    logging.error(f"Erreur lors de la configuration de matplotlib: {e}")

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
                try:
                    data['global_potential'] = float(global_match.group(1))
                except ValueError:
                    logging.warning(f"Valeur de potentiel global non valide dans {filepath}: {global_match.group(1)}")
            else:
                logging.warning(f"Potentiel global non trouvé dans {filepath}")
            
            # Extraction des contributions
            contributions = {
                'EA': r'E\(EA\) = ([-\d.]+) V',
                'Edef': r'E\(Edef\) = ([-\d.]+) V',
                'delta_delta_U': r'E\(∆∆U\) = ([-\d.]+) V',
                'T_delta_S': r'E\(T∆S\) = ([-\d.]+) V'
            }
            
            for key, pattern in contributions.items():
                match = re.search(pattern, content)
                if match:
                    try:
                        data[key] = float(match.group(1))
                    except ValueError:
                        logging.warning(f"Valeur de {key} non valide dans {filepath}: {match.group(1)}")
                else:
                    logging.warning(f"Contribution {key} non trouvée dans {filepath}")
            
            # Extraction des valeurs HOMO et LUMO
            homo_match = re.search(r'HOMO moyenne pondérée: ([-\d.]+) eV', content)
            if homo_match:
                try:
                    data['HOMO'] = float(homo_match.group(1))
                except ValueError:
                    logging.warning(f"Valeur HOMO non valide dans {filepath}: {homo_match.group(1)}")
            else:
                logging.warning(f"Valeur HOMO non trouvée dans {filepath}")
                
            lumo_match = re.search(r'LUMO moyenne pondérée: ([-\d.]+) eV', content)
            if lumo_match:
                try:
                    data['LUMO'] = float(lumo_match.group(1))
                except ValueError:
                    logging.warning(f"Valeur LUMO non valide dans {filepath}: {lumo_match.group(1)}")
            else:
                logging.warning(f"Valeur LUMO non trouvée dans {filepath}")

            # Vérifier que les données contiennent au moins le potentiel global
            if 'global_potential' not in data:
                logging.error(f"Fichier {filepath} ne contient pas de potentiel global - données ignorées")
                return None
                
            # Vérifier si des contributions sont présentes
            contributions_found = [key for key in contributions.keys() if key in data]
            if not contributions_found:
                logging.error(f"Fichier {filepath} ne contient aucune contribution - données ignorées")
                return None
                
            return data
    except FileNotFoundError:
        logging.error(f"Fichier non trouvé: {filepath}")
    except PermissionError:
        logging.error(f"Permission refusée lors de l'accès à {filepath}")
    except UnicodeDecodeError:
        logging.error(f"Erreur d'encodage lors de la lecture de {filepath}")
    except Exception as e:
        logging.error(f"Erreur inattendue lors de la lecture de {filepath}: {e}")
        logging.debug(traceback.format_exc())
    
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
    paths_with_errors = []
    
    if not paths:
        logging.error("Aucun chemin fourni pour la collecte de données")
        return []
    
    for path in paths:
        try:
            # Construire le chemin complet vers le fichier redox_potentials.txt
            path = Path(path)
            redox_file = path / "redox_results" / "redox_potentials.txt"
            
            if not redox_file.exists():
                logging.warning(f"Fichier non trouvé: {redox_file}")
                paths_with_errors.append(str(path))
                continue
            
            # Extraire le nom du dossier parent pour l'identification
            parent_name = "Unknown"
            try:
                parent_name = path.parts[0] if path.parts else "Unknown"
                
                # Ajouter un identifiant de redox (ex: redox.011)
                for part in path.parts:
                    if part.startswith("redox."):
                        parent_name = f"{parent_name}_{part}"
                        break
            except Exception as e:
                logging.warning(f"Erreur lors de l'extraction du nom de dossier: {e}")
            
            # Analyser le fichier
            data = parse_redox_file(redox_file, parent_name)
            
            if data:
                data_list.append(data)
                logging.info(f"Données extraites pour {parent_name}: E(ΔG) = {data['global_potential']} V")
        except Exception as e:
            logging.error(f"Erreur lors du traitement du chemin {path}: {e}")
            logging.debug(traceback.format_exc())
            paths_with_errors.append(str(path))
    
    # Résumé des erreurs
    if paths_with_errors:
        logging.warning(f"Problèmes rencontrés avec {len(paths_with_errors)}/{len(paths)} chemins:")
        for path in paths_with_errors[:5]:  # Limiter à 5 pour éviter un message trop long
            logging.warning(f"  - {path}")
        if len(paths_with_errors) > 5:
            logging.warning(f"  - ... et {len(paths_with_errors) - 5} autres chemins")
    
    # Vérification des contributions disponibles
    analyze_available_contributions(data_list)
    
    return data_list

def analyze_available_contributions(data_list):
    """
    Analyse les contributions disponibles dans les données et affiche un résumé
    
    Args:
        data_list: Liste de dictionnaires contenant les données des potentiels
    """
    if not data_list:
        return
    
    expected_keys = ['global_potential', 'EA', 'Edef', 'delta_delta_U', 'T_delta_S']
    key_counts = {key: 0 for key in expected_keys}
    
    for data in data_list:
        for key in expected_keys:
            if key in data:
                key_counts[key] += 1
    
    logging.info("Résumé des contributions disponibles:")
    for key, count in key_counts.items():
        percentage = (count / len(data_list)) * 100
        logging.info(f"  - {key}: présent dans {count}/{len(data_list)} fichiers ({percentage:.1f}%)")
    
    # Avertissements pour les contributions manquantes
    for key, count in key_counts.items():
        if count == 0:
            logging.error(f"ATTENTION: La contribution '{key}' n'est présente dans aucun fichier!")
        elif count < len(data_list):
            logging.warning(f"La contribution '{key}' est absente dans {len(data_list) - count} fichiers.")

def add_linear_regression(ax, x_values, y_values, color='red'):
    """
    Ajoute une régression linéaire (y = ax + b) au graphique
    
    Args:
        ax: Axes matplotlib
        x_values: Valeurs x (données indépendantes)
        y_values: Valeurs y (données dépendantes)
        color: Couleur de la ligne de régression
        
    Returns:
        tuple: (équation sous forme de chaîne, valeur R², MAE)
    """
    try:
        # Vérifier les données d'entrée
        if len(x_values) < 2 or len(y_values) < 2:
            logging.warning("Au moins 2 points sont nécessaires pour une régression linéaire")
            return None, None, None
            
        if len(x_values) != len(y_values):
            logging.error(f"Dimensions incohérentes: x({len(x_values)}) ≠ y({len(y_values)})")
            return None, None, None
        
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
        
        if ss_tot == 0:  # Éviter la division par zéro
            logging.warning("Variance nulle dans les données y - R² indéfini")
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Calcul de MAE
        mae = np.mean(np.abs(np.array(y_values) - np.array(y_pred)))

        # Formulation de l'équation
        equation = f"y = {slope:.4f}x + {intercept:.4f}"
        
        return equation, r_squared, mae
    except Exception as e:
        logging.error(f"Erreur lors de la régression linéaire: {e}")
        logging.debug(traceback.format_exc())
        return None, None, None

def add_polynomial_regression(ax, x_values, y_values, degree=2, color='green'):
    """
    Ajoute une régression polynomiale au graphique
    
    Args:
        ax: Axes matplotlib
        x_values: Valeurs x (données indépendantes)
        y_values: Valeurs y (données dépendantes)
        degree: Degré du polynôme
        color: Couleur de la ligne de régression
        
    Returns:
        tuple: (équation sous forme de chaîne, valeur R², MAE)
    """
    global _polynomial_warning_shown
    
    try:
        # Vérifier les données d'entrée
        if len(x_values) <= degree:
            logging.warning(f"Au moins {degree+1} points sont nécessaires pour une régression de degré {degree}")
            return None, None, None
            
        if len(x_values) != len(y_values):
            logging.error(f"Dimensions incohérentes: x({len(x_values)}) ≠ y({len(y_values)})")
            return None, None, None
        
        n = len(x_values)  # Nombre de points
        p = degree + 1     # Nombre de paramètres (degré + terme constant)
        
        # Vérifier le risque d'overfitting - n'afficher l'avertissement qu'une seule fois
        if n <= p and not _polynomial_warning_shown:
            logging.warning(f"Trop peu de points ({n}) pour une régression de degré {degree} ({p} paramètres)")
            logging.warning("Le R² sera artificiellement élevé ou égal à 1.")
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
        
        if ss_tot == 0:  # Éviter la division par zéro
            logging.warning("Variance nulle dans les données y - R² indéfini")
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Calcul de MAE
        mae = np.mean(np.abs(np.array(y_values) - np.array(y_pred)))

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
        
        return equation, r_squared, mae
    except Exception as e:
        logging.error(f"Erreur lors de la régression polynomiale: {e}")
        logging.debug(traceback.format_exc())
        return None, None, None

def add_exponential_regression(ax, x_values, y_values, color='purple'):
    """
    Ajoute une régression exponentielle (y = a*exp(b*x))
    
    Args:
        ax: Axes matplotlib
        x_values: Valeurs x (données indépendantes)
        y_values: Valeurs y (données dépendantes)
        color: Couleur de la ligne de régression
        
    Returns:
        tuple: (équation sous forme de chaîne, valeur R², MAE)
    """
    try:
        # Vérifier les données d'entrée
        if len(x_values) < 2 or len(y_values) < 2:
            logging.warning("Au moins 2 points sont nécessaires pour une régression exponentielle")
            return None, None, None
            
        if len(x_values) != len(y_values):
            logging.error(f"Dimensions incohérentes: x({len(x_values)}) ≠ y({len(y_values)})")
            return None, None, None
            
        # Ajustement pour les valeurs négatives ou nulles
        valid_indices = [i for i, y in enumerate(y_values) if y > 0]
        if len(valid_indices) < len(y_values):
            logging.warning(f"Certains points ignorés pour la régression exponentielle ({len(y_values) - len(valid_indices)} valeurs y ≤ 0)")
        
        if len(valid_indices) < 2:
            logging.error("Données insuffisantes pour la régression exponentielle après filtrage")
            return None, None, None
        
        x_valid = np.array([x_values[i] for i in valid_indices])
        y_valid = np.array([y_values[i] for i in valid_indices])
        
        # Transformation logarithmique pour ajustement linéaire
        log_y = np.log(y_valid)
        
        # Vérifier les valeurs NaN
        if np.isnan(log_y).any():
            logging.error("Valeurs NaN détectées lors de la transformation logarithmique")
            return None, None, None
            
        try:
            params, cov = np.polyfit(x_valid, log_y, 1, cov=True)
        except Exception as fit_err:
            logging.error(f"Erreur lors de l'ajustement exponentiel: {fit_err}")
            return None, None, None
        
        # Extraction des paramètres
        a = np.exp(params[1])
        b = params[0]
        
        # Points x pour tracer une courbe lisse
        x_min, x_max = min(x_values), max(x_values)
        x_seq = np.linspace(x_min, x_max, 100)
        
        # Si b est positif et les valeurs x sont grandes, limiter x pour éviter l'overflow
        if b > 0 and x_max > 100:
            # Trouver la valeur x où exp(b*x) approche de la limite de float
            safe_x = np.log(sys.float_info.max) / b
            if safe_x < x_max:
                logging.warning(f"Limitant la courbe exponentielle à x={safe_x:.2f} pour éviter l'overflow")
                x_seq = np.linspace(x_min, min(x_max, safe_x), 100)
        
        try:
            y_seq = a * np.exp(b * x_seq)
            
            # Vérifier les valeurs inf/NaN
            if np.isinf(y_seq).any() or np.isnan(y_seq).any():
                logging.warning("Valeurs infinies ou NaN détectées dans la courbe exponentielle")
                # Filtrer les points problématiques
                valid_seq = ~(np.isinf(y_seq) | np.isnan(y_seq))
                x_seq = x_seq[valid_seq]
                y_seq = y_seq[valid_seq]
                
                if len(x_seq) < 2:
                    logging.error("Trop peu de points valides pour tracer la courbe exponentielle")
                    return None, None, None
        except Exception as calc_err:
            logging.error(f"Erreur lors du calcul des valeurs y pour la courbe exponentielle: {calc_err}")
            return None, None, None
        
        # Tracé de la courbe
        try:
            ax.plot(x_seq, y_seq, color=color, linestyle='-.', linewidth=2)
        except Exception as plot_err:
            logging.error(f"Erreur lors du tracé de la courbe exponentielle: {plot_err}")
        
        # Calcul du R²
        y_pred = a * np.exp(b * x_valid)
        ss_res = np.sum((y_valid - y_pred)**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
        
        if ss_tot == 0:  # Éviter la division par zéro
            logging.warning("Variance nulle dans les données y - R² indéfini")
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Calcul de MAE - utiliser uniquement les points valides
        mae = np.mean(np.abs(y_valid - y_pred))

        return f"y = {a:.4f}·e^({b:.4f}x)", r_squared, mae
    except Exception as e:
        logging.error(f"Erreur lors de la régression exponentielle: {e}")
        logging.debug(traceback.format_exc())
        return None, None, None

def add_logarithmic_regression(ax, x_values, y_values, color='orange'):
    """
    Ajoute une régression logarithmique (y = a + b*ln(x))
    
    Args:
        ax: Axes matplotlib
        x_values: Valeurs x (données indépendantes)
        y_values: Valeurs y (données dépendantes)
        color: Couleur de la ligne de régression
        
    Returns:
        tuple: (équation sous forme de chaîne, valeur R², MAE)
    """
    try:
        # Vérifier les données d'entrée
        if len(x_values) < 2 or len(y_values) < 2:
            logging.warning("Au moins 2 points sont nécessaires pour une régression logarithmique")
            return None, None, None
            
        if len(x_values) != len(y_values):
            logging.error(f"Dimensions incohérentes: x({len(x_values)}) ≠ y({len(y_values)})")
            return None, None, None
            
        # Ajustement pour les valeurs négatives ou nulles
        valid_indices = [i for i, x in enumerate(x_values) if x > 0]
        if len(valid_indices) < len(x_values):
            logging.warning(f"Certains points ignorés pour la régression logarithmique ({len(x_values) - len(valid_indices)} valeurs x ≤ 0)")
        
        if len(valid_indices) < 2:
            logging.error("Données insuffisantes pour la régression logarithmique après filtrage")
            return None, None, None
        
        x_valid = np.array([x_values[i] for i in valid_indices])
        y_valid = np.array([y_values[i] for i in valid_indices])
        
        # Transformation logarithmique
        log_x = np.log(x_valid)
        
        # Vérifier les valeurs NaN
        if np.isnan(log_x).any():
            logging.error("Valeurs NaN détectées lors de la transformation logarithmique")
            return None, None, None
            
        try:
            params = np.polyfit(log_x, y_valid, 1)
        except Exception as fit_err:
            logging.error(f"Erreur lors de l'ajustement logarithmique: {fit_err}")
            return None, None, None
        
        # Extraction des paramètres
        a = params[1]
        b = params[0]
        
        # Points pour courbe lisse
        try:
            # Utiliser une échelle logarithmique pour les points x
            x_min, x_max = min(x_valid), max(x_valid)
            x_seq = np.logspace(np.log10(x_min), np.log10(x_max), 100)
            y_seq = a + b * np.log(x_seq)
            
            # Vérifier les valeurs inf/NaN
            if np.isinf(y_seq).any() or np.isnan(y_seq).any():
                logging.warning("Valeurs infinies ou NaN détectées dans la courbe logarithmique")
                # Filtrer les points problématiques
                valid_seq = ~(np.isinf(y_seq) | np.isnan(y_seq))
                x_seq = x_seq[valid_seq]
                y_seq = y_seq[valid_seq]
                
                if len(x_seq) < 2:
                    logging.error("Trop peu de points valides pour tracer la courbe logarithmique")
                    return None, None, None
        except Exception as calc_err:
            logging.error(f"Erreur lors du calcul des valeurs y pour la courbe logarithmique: {calc_err}")
            return None, None, None
        
        # Tracé de la courbe
        try:
            ax.plot(x_seq, y_seq, color=color, linestyle=':', linewidth=2)
        except Exception as plot_err:
            logging.error(f"Erreur lors du tracé de la courbe logarithmique: {plot_err}")
        
        # Calcul du R²
        y_pred = a + b * np.log(x_valid)
        ss_res = np.sum((y_valid - y_pred)**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
        
        if ss_tot == 0:  # Éviter la division par zéro
            logging.warning("Variance nulle dans les données y - R² indéfini")
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Calcul de MAE - utiliser uniquement les points valides
        mae = np.mean(np.abs(y_valid - y_pred))

        return f"y = {a:.4f} + {b:.4f}·ln(x)", r_squared, mae
    except Exception as e:
        logging.error(f"Erreur lors de la régression logarithmique: {e}")
        logging.debug(traceback.format_exc())
        return None, None, None

def create_orbital_ea_plots(data_list, output_dir):
    """
    Crée des graphiques de corrélation entre HOMO/LUMO et EA

    Args:
        data_list: Liste de dictionnaires contenant les données
        output_dir: Répertoire de sortie pour les graphiques

    Returns:
        int: nombre de graphiques générés avec succès
    """
    successful_plots = 0

    try:
        if not data_list:
            logging.error("Aucune donnée disponible pour créer les graphiques d'orbitales.")
            return 0

        # Création des graphiques HOMO vs EA et LUMO vs EA
        for orbital_type in ['HOMO', 'LUMO']:
            try:
                # Extraction des données valides
                filtered_data = [d for d in data_list if orbital_type in d and 'EA' in d]

                if len(filtered_data) < 2:
                    logging.warning(f"Données insuffisantes pour le graphique {orbital_type} vs EA (minimum 2 points requis)")
                    continue

                x_values = [data[orbital_type] for data in filtered_data]
                y_values = [data['EA'] for data in filtered_data]
                labels = [data['molecule'] for data in filtered_data]

                # Vérifier les valeurs non finies (NaN, inf)
                invalid_indices = [i for i, (x, y) in enumerate(zip(x_values, y_values))
                                   if not (np.isfinite(x) and np.isfinite(y))]

                if invalid_indices:
                    logging.warning(f"Valeurs non finies détectées pour {orbital_type} vs EA: indices {invalid_indices}")
                    # Filtrer les valeurs invalides
                    valid_indices = [i for i in range(len(x_values)) if i not in invalid_indices]
                    x_values = [x_values[i] for i in valid_indices]
                    y_values = [y_values[i] for i in valid_indices]
                    labels = [labels[i] for i in valid_indices]

                # Conversion en numpy arrays
                x_values = np.array(x_values, dtype=float)
                y_values = np.array(y_values, dtype=float)

                # Création de la figure
                fig, ax = plt.subplots(figsize=(12, 9))

                # Tracé des points avec les noms des molécules
                ax.scatter(x_values, y_values, s=80, alpha=0.7, c='blue', edgecolors='black')

                # Ajout des étiquettes pour les points
                for i, label in enumerate(labels):
                    ax.annotate(label, (x_values[i], y_values[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, alpha=0.8)

                # Régression linéaire
                if len(x_values) >= 2:  # Au moins 2 points pour une régression
                    slope, intercept = np.polyfit(x_values, y_values, 1)

                    # Points pour la ligne de régression
                    x_reg = np.linspace(min(x_values), max(x_values), 100)
                    y_reg = slope * x_reg + intercept

                    # Tracé de la régression
                    ax.plot(x_reg, y_reg, 'r-', linewidth=2)

                    # Calcul du R²
                    y_pred = slope * x_values + intercept
                    r_squared = 1 - (sum((y_values - y_pred) ** 2) /
                                     sum((y_values - np.mean(y_values)) ** 2))

                    # Calcul de MAE
                    mae = np.mean(np.abs(np.array(y_values) - np.array(y_pred)))

                    # Ajout du texte avec l'équation et R²
                    equation_text = f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r_squared:.4f}\nMAE = {mae:.4f}"
                    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # Configuration des axes
                ax.set_xlabel(f"{orbital_type} (eV)", fontsize=14)
                ax.set_ylabel("EA (V)", fontsize=14)
                ax.set_title(f"Corrélation entre {orbital_type} et EA", fontsize=16)
                ax.grid(True, linestyle='--', alpha=0.7)

                # Ajout de marges autour des données
                x_margin = 0.1 * (max(x_values) - min(x_values))
                y_margin = 0.1 * (max(y_values) - min(y_values))

                plt.xlim(min(x_values) - x_margin, max(x_values) + x_margin)
                plt.ylim(min(y_values) - y_margin, max(y_values) + y_margin)

                # Sauvegarde du graphique
                output_path = output_dir / f"{orbital_type}_vs_EA.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                logging.info(f"Graphique {orbital_type} vs EA créé: {output_path}")
                successful_plots += 1

            except Exception as e:
                logging.error(f"Erreur lors de la création du graphique {orbital_type} vs EA: {e}")
                logging.debug(traceback.format_exc())
                plt.close('all')  # Fermer les figures en cas d'erreur

        return successful_plots

    except Exception as e:
        logging.error(f"Erreur inattendue lors de la création des graphiques d'orbitales: {e}")
        logging.debug(traceback.format_exc())
        plt.close('all')
        return successful_plots

def create_output_directory():
    """
    Crée un répertoire pour les graphiques avec numérotation automatique
    si le dossier existe déjà
    
    Returns:
        Path: Chemin vers le répertoire créé
    """
    try:
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
                    
                # Protection contre les boucles infinies
                counter += 1
                if counter > 999:
                    logging.warning("Nombre maximum de répertoires atteint. Utilisation d'un horodatage.")
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_dir_name = f"{base_dir_name}.{timestamp}"
                    output_dir = Path(new_dir_name)
                    break
        
        # Création du répertoire avec gestion d'erreurs
        try:
            output_dir.mkdir(exist_ok=True)
            logging.info(f"Création du dossier de sortie: {output_dir}")
        except PermissionError:
            logging.error(f"Permission refusée lors de la création du dossier {output_dir}")
            # Fallback au répertoire courant
            output_dir = Path(".")
            logging.warning(f"Utilisation du répertoire courant comme solution de repli")
        except Exception as dir_err:
            logging.error(f"Erreur lors de la création du dossier {output_dir}: {dir_err}")
            # Fallback au répertoire courant
            output_dir = Path(".")
            logging.warning(f"Utilisation du répertoire courant comme solution de repli")
        
        return output_dir
    except Exception as e:
        logging.error(f"Erreur inattendue lors de la création du répertoire de sortie: {e}")
        logging.debug(traceback.format_exc())
        return Path(".")  # Fallback au répertoire courant

def create_correlation_plots(data_list, regressions=['linear', 'polynomial', 'exponential', 'logarithmic']):
    """
    Crée des graphiques de corrélation entre le potentiel global et ses contributions
    avec différents types de régressions
    
    Args:
        data_list: Liste de dictionnaires contenant les données des potentiels
        regressions: Liste des types de régressions à inclure 
                    (options: 'linear', 'polynomial', 'exponential', 'logarithmic')
    """
    try:
        # Réinitialiser l'avertissement polynomial au début
        global _polynomial_warning_shown
        _polynomial_warning_shown = False
        
        if not data_list:
            logging.error("Aucune donnée disponible pour créer les graphiques.")
            return
            
        # Vérifier les types de régression demandés
        valid_regressions = ['linear', 'polynomial', 'exponential', 'logarithmic']
        invalid_regressions = [r for r in regressions if r not in valid_regressions]
        if invalid_regressions:
            logging.warning(f"Types de régression non reconnus ignorés: {', '.join(invalid_regressions)}")
            regressions = [r for r in regressions if r in valid_regressions]
            
        if not regressions:
            logging.error("Aucun type de régression valide spécifié")
            return
        
        # La correspondance entre les clés et les noms d'affichage
        contributions = {
            'EA': 'E(EA) (V)',
            'Edef': 'E(Edef) (V)',
            'delta_delta_U': 'E(ΔΔU) (V)',
            'T_delta_S': 'E(TΔS) (V)'
        }
        
        # Création d'un répertoire numéroté pour les graphiques
        try:
            output_dir = create_output_directory()
        except Exception as e:
            logging.error(f"Erreur lors de la création du répertoire de sortie: {e}")
            output_dir = Path(".")  # Utiliser le répertoire courant comme solution de repli
        
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
        
        # Compteur pour les graphiques réalisés avec succès
        successful_plots = 0
        failed_plots = 0
        
        # Création des graphiques pour chaque contribution
        for contrib_key, contrib_label in contributions.items():
            try:
                # Extraction des données pour cette contribution
                filtered_data = [d for d in data_list if contrib_key in d and 'global_potential' in d]
                if not filtered_data:
                    logging.warning(f"Aucune donnée disponible pour la contribution '{contrib_key}'")
                    continue
                    
                x_values = [data[contrib_key] for data in filtered_data]
                y_values = [data['global_potential'] for data in filtered_data]
                labels = [data['molecule'] for data in filtered_data]
                
                # Vérifier les valeurs non finies (NaN, inf)
                invalid_indices = [i for i, (x, y) in enumerate(zip(x_values, y_values)) 
                                  if not (np.isfinite(x) and np.isfinite(y))]
                                  
                if invalid_indices:
                    logging.warning(f"Valeurs non finies détectées pour {contrib_key}: indices {invalid_indices}")
                    # Filtrer les valeurs invalides
                    valid_indices = [i for i in range(len(x_values)) if i not in invalid_indices]
                    x_values = [x_values[i] for i in valid_indices]
                    y_values = [y_values[i] for i in valid_indices]
                    labels = [labels[i] for i in valid_indices]
                
                if len(x_values) < 2:
                    logging.error(f"Données insuffisantes pour la contribution {contrib_key} (minimum 2 points requis)")
                    continue
                
                # Conversion en numpy arrays pour les calculs
                try:
                    x_values = np.array(x_values, dtype=float)
                    y_values = np.array(y_values, dtype=float)
                except Exception as conv_err:
                    logging.error(f"Erreur de conversion des données pour {contrib_key}: {conv_err}")
                    continue
                
                # Création de la figure
                try:
                    fig, ax = plt.subplots(figsize=(12, 9))
                except Exception as fig_err:
                    logging.error(f"Erreur lors de la création de la figure pour {contrib_key}: {fig_err}")
                    continue
                
                # Tracé des points avec les noms des molécules
                try:
                    ax.scatter(x_values, y_values, s=80, alpha=0.7, c='blue', edgecolors='black')
                    
                    # Ajout des étiquettes pour les points
                    for i, label in enumerate(labels):
                        ax.annotate(label, (x_values[i], y_values[i]), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=10, alpha=0.8)
                except Exception as scatter_err:
                    logging.error(f"Erreur lors du tracé des points pour {contrib_key}: {scatter_err}")
                    plt.close(fig)
                    continue
                
                # Liste pour stocker les lignes et labels de la légende
                legend_handles = []
                legend_labels = []
                
                # Tentative d'appliquer chaque régression demandée
                regression_results = {}
                
                # 1. Régression linéaire
                if 'linear' in regressions:
                    try:
                        result = add_linear_regression(
                            ax, x_values, y_values, 
                            color=colors['linear']
                        )
                        
                        if result is not None and None not in result:
                            equation, r_squared, mae = result
                            # Ajouter une ligne à la légende
                            from matplotlib.lines import Line2D
                            legend_handles.append(Line2D([0], [0], color=colors['linear'], 
                                                        linestyle=linestyles['linear'], linewidth=2))
                            legend_labels.append(f'Linéaire: {equation}, R² = {r_squared:.4f}, MAE = {mae:.4f} V')
                            regression_results['linear'] = (equation, r_squared, mae)
                        else:
                            logging.warning(f"Échec de la régression linéaire pour {contrib_key}")
                    except Exception as reg_err:
                        logging.error(f"Erreur lors de la régression linéaire pour {contrib_key}: {reg_err}")
                
                # 2. Régression polynomiale
                if 'polynomial' in regressions:
                    try:
                        result = add_polynomial_regression(
                            ax, x_values, y_values, 
                            degree=2, 
                            color=colors['polynomial']
                        )
                        
                        if result is not None and None not in result:
                            equation, r_squared, mae = result
                            # Ajouter une ligne à la légende
                            from matplotlib.lines import Line2D
                            legend_handles.append(Line2D([0], [0], color=colors['polynomial'], 
                                                        linestyle=linestyles['polynomial'], linewidth=2))
                            legend_labels.append(f'Quadratique: {equation}, R² = {r_squared:.4f}, MAE = {mae:.4f} V')
                            regression_results['polynomial'] = (equation, r_squared, mae)
                        else:
                            logging.warning(f"Échec de la régression polynomiale pour {contrib_key}")
                    except Exception as reg_err:
                        logging.error(f"Erreur lors de la régression polynomiale pour {contrib_key}: {reg_err}")
                
                # 3. Régression exponentielle
                if 'exponential' in regressions:
                    try:
                        result = add_exponential_regression(
                            ax, x_values, y_values, 
                            color=colors['exponential']
                        )
                        
                        if result is not None and None not in result:
                            equation, r_squared, mae = result
                            # Ajouter une ligne à la légende
                            from matplotlib.lines import Line2D
                            legend_handles.append(Line2D([0], [0], color=colors['exponential'], 
                                                        linestyle=linestyles['exponential'], linewidth=2))
                            legend_labels.append(f'Exponentielle: {equation}, R² = {r_squared:.4f}, MAE = {mae:.4f} V')
                            regression_results['exponential'] = (equation, r_squared, mae)
                        else:
                            logging.warning(f"Échec de la régression exponentielle pour {contrib_key}")
                    except Exception as reg_err:
                        logging.error(f"Erreur lors de la régression exponentielle pour {contrib_key}: {reg_err}")
                
                # 4. Régression logarithmique
                if 'logarithmic' in regressions:
                    try:
                        result = add_logarithmic_regression(
                            ax, x_values, y_values, 
                            color=colors['logarithmic']
                        )
                        
                        if result is not None and None not in result:
                            equation, r_squared, mae = result
                            # Ajouter une ligne à la légende
                            from matplotlib.lines import Line2D
                            legend_handles.append(Line2D([0], [0], color=colors['logarithmic'], 
                                                        linestyle=linestyles['logarithmic'], linewidth=2))
                            legend_labels.append(f'Logarithmique: {equation}, R² = {r_squared:.4f}, MAE = {mae:.4f} V')
                            regression_results['logarithmic'] = (equation, r_squared, mae)
                        else:
                            logging.warning(f"Échec de la régression logarithmique pour {contrib_key}")
                    except Exception as reg_err:
                        logging.error(f"Erreur lors de la régression logarithmique pour {contrib_key}: {reg_err}")
                
                # Vérifier si au moins une régression a réussi
                if not regression_results:
                    logging.error(f"Aucune régression n'a réussi pour {contrib_key}")
                    plt.close(fig)
                    failed_plots += 1
                    continue
                
                # Configuration des axes et légendes
                try:
                    ax.set_xlabel(contrib_label, fontsize=14)
                    ax.set_ylabel('E(redox) (V)', fontsize=14)
                    ax.set_title(f'Corrélation entre {contrib_label} et le potentiel global', fontsize=16)
                    
                    # Utiliser les handles personnalisés pour la légende
                    if legend_handles:
                        ax.legend(legend_handles, legend_labels, fontsize=10)
                    
                    ax.grid(True, alpha=0.3, linestyle='--')
                    fig.tight_layout()
                except Exception as format_err:
                    logging.error(f"Erreur lors de la mise en forme du graphique pour {contrib_key}: {format_err}")
                
                # Enregistrement du graphique
                try:
                    output_filename = output_dir / f"correlation_{contrib_key}.png"
                    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
                    logging.info(f"Graphique enregistré: {output_filename}")
                    successful_plots += 1
                except Exception as save_err:
                    logging.error(f"Erreur lors de l'enregistrement du graphique pour {contrib_key}: {save_err}")
                    failed_plots += 1
                
                plt.close(fig)
                
            except Exception as contrib_err:
                logging.error(f"Erreur inattendue lors du traitement de la contribution {contrib_key}: {contrib_err}")
                logging.debug(traceback.format_exc())
                failed_plots += 1
        
        # Résumé des graphiques générés
        if successful_plots > 0:
            logging.info(f"{successful_plots} graphiques ont été générés avec succès dans le dossier '{output_dir}'.")
        if failed_plots > 0:
            logging.warning(f"{failed_plots} graphiques n'ont pas pu être générés correctement.")
        if successful_plots == 0:
            logging.error("Aucun graphique n'a pu être généré.")
        
    except Exception as e:
        logging.error(f"Erreur critique lors de la création des graphiques de corrélation: {e}")
        logging.debug(traceback.format_exc())
        
def main():
    """
    Fonction principale - Traite les arguments de la ligne de commande
    """
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Analyse des potentiels redox")
        parser.add_argument('paths', nargs='+', help="Chemins vers les dossiers de redox à analyser")
        parser.add_argument('--regressions', nargs='+', 
                            choices=['linear', 'polynomial', 'exponential', 'logarithmic'],
                            default=['linear', 'polynomial', 'exponential', 'logarithmic'], 
                            help="Types de régressions à inclure dans les graphiques")
        parser.add_argument('--verbose', '-v', action='store_true', 
                            help="Active les messages de débogage détaillés")
        
        # Tentative d'analyse des arguments
        try:
            args = parser.parse_args()
        except Exception as arg_err:
            logging.error(f"Erreur lors de l'analyse des arguments: {arg_err}")
            logging.error("Utilisation: python analysis.py chemin1/redox/ chemin2/redox/ [--regressions linear polynomial ...]")
            sys.exit(1)
        
        # Configurer le niveau de verbosité
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.info("Mode verbeux activé")
        
        # Vérifier les chemins fournis
        if not args.paths:
            logging.error("Aucun chemin spécifié.")
            sys.exit(1)
            
        # Vérifier si les chemins existent
        invalid_paths = [path for path in args.paths if not os.path.exists(path)]
        if invalid_paths:
            logging.warning(f"Chemins non trouvés: {', '.join(invalid_paths)}")
            if len(invalid_paths) == len(args.paths):
                logging.error("Aucun chemin valide spécifié.")
                sys.exit(1)
        
        logging.info(f"Analyse des potentiels redox pour {len(args.paths)} chemins...")
        logging.info(f"Régressions sélectionnées: {', '.join(args.regressions)}")
        
        # Collecte des données
        try:
            data_list = collect_data(args.paths)
        except Exception as collect_err:
            logging.error(f"Erreur critique lors de la collecte des données: {collect_err}")
            logging.debug(traceback.format_exc())
            sys.exit(1)
        
        if not data_list:
            logging.error("Aucune donnée n'a pu être extraite. Vérifiez les chemins fournis.")
            sys.exit(1)
        
        logging.info(f"Données extraites pour {len(data_list)} molécules.")
        
        # Création du répertoire de sortie
        output_dir = create_output_directory()
        
        # Création des graphiques de corrélation
        try:
            create_correlation_plots(data_list, regressions=args.regressions)
        except Exception as plot_err:
            logging.error(f"Erreur critique lors de la création des graphiques de corrélation: {plot_err}")
            logging.debug(traceback.format_exc())
            # Continuer l'exécution pour essayer de créer les autres graphiques
        
        # Création des graphiques HOMO/LUMO vs EA
        try:
            orbital_plots_count = create_orbital_ea_plots(data_list, output_dir)
            if orbital_plots_count > 0:
                logging.info(f"{orbital_plots_count} graphiques d'orbitales ont été générés avec succès.")
            else:
                logging.warning("Aucun graphique d'orbitales n'a pu être généré.")
        except Exception as orbital_err:
            logging.error(f"Erreur critique lors de la création des graphiques d'orbitales: {orbital_err}")
            logging.debug(traceback.format_exc())
            
        logging.info("Analyse terminée avec succès.")
            
    except KeyboardInterrupt:
        logging.warning("Opération interrompue par l'utilisateur.")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Erreur inattendue dans la fonction principale: {e}")
        logging.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
