#!/usr/bin/env amspython
"""
Génère un rapport comparant E_def calculé par différentes méthodes vibroniques
avec E_def réel (E_opt - E_sp) pour toutes les molécules d'un répertoire.

Méthodes implémentées:
    1. LVC (Linear Vibronic Coupling): E_def = -Σ Fᵢ²/(2kᵢ) (modes neutres, gradients SP)
    2. Displacement neutre: E_def = -½ Σ kᵢ(Nᵢ·ΔR)² (modes neutres, déplacement réel — validation LVC)
    3. Duschinsky: E_def = -½ Σ k'ⱼ(N'ⱼ·ΔR)² (modes ioniques, déplacement réel)
    4. QVC (Quadratic Vibronic Coupling): LVC × (1 + correction quadratique)

Usage:
    ./generate_lvc_edef_report.py --batch-dir <dir_workdir> --output-dir <output>
    ./generate_lvc_edef_report.py --batch-dir <dir_workdir> --output-dir <output> --rmsd-threshold 0.5

Exemple:
    ./generate_lvc_edef_report.py --batch-dir 1594174/dir_workdir --output-dir lvc_report
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from analysis import DEA_EXCLUDED
from common.utils import print_header, print_section, print_success, print_error, print_warning

from qmanalysis.lvc_analysis import (
    run_lvc_analysis_for_molecule,
    load_real_edef,
    compute_rmsd_for_molecule,
)
from qmanalysis.duschinsky_analysis import run_duschinsky_analysis_for_molecule
from qmanalysis.qvc_analysis import run_qvc_analysis_for_molecule
from qmanalysis.vpt2_analysis import run_vpt2_analysis_for_molecule
from qmanalysis.dual_hessian_analysis import run_dual_hessian_for_molecule, optimize_parameters
from qmanalysis.correlation_analysis import _read_conformer_weights_from_summary


def find_molecule_workdirs(batch_dir: Path) -> List[Tuple[Path, str]]:
    """
    Trouve tous les workdirs de molécules dans un répertoire batch.

    Returns:
        List[(workdir_path, molecule_name)]
    """
    workdirs = []

    for item in sorted(batch_dir.iterdir()):
        if item.is_dir() and item.name.endswith('_workdir'):
            mol_name = item.name.replace('_workdir', '')
            # Exclure les molécules DEA (rupture C-X lors de la réduction)
            if mol_name in DEA_EXCLUDED:
                continue
            # Vérifier que le répertoire redox existe
            if (item / 'redox' / 'calculations' / 'plams_workdir').exists():
                workdirs.append((item, mol_name))

    return workdirs


def compute_boltzmann_averaged_edef(
    lvc_results: Dict[int, Dict],
    real_edef: Dict[int, Dict],
    weights: Dict[int, float],
    process: str  # 'reduction' ou 'oxidation'
) -> Tuple[float, float]:
    """
    Calcule les E_def moyennés par Boltzmann pour LVC et réel.

    Returns:
        (avg_lvc_edef, avg_real_edef)
    """
    total_weight = 0.0
    avg_lvc = 0.0
    avg_real = 0.0

    edef_key = 'Edef_red_kJ' if process == 'reduction' else 'Edef_ox_kJ'

    for conf_num in lvc_results.keys():
        w = weights.get(conf_num, 0.0)
        if w <= 0:
            continue

        lvc_data = lvc_results[conf_num].get(process)
        real_data = real_edef.get(conf_num, {})

        if lvc_data and lvc_data.get('success'):
            lvc_val = lvc_data.get('E_def_lvc_kj', 0.0)
        else:
            continue

        real_val = real_data.get(edef_key, np.nan)
        if np.isnan(real_val):
            continue

        total_weight += w
        avg_lvc += w * lvc_val
        avg_real += w * real_val

    if total_weight > 0:
        avg_lvc /= total_weight
        avg_real /= total_weight
    else:
        avg_lvc = np.nan
        avg_real = np.nan

    return avg_lvc, avg_real


def compute_boltzmann_averaged_method(
    method_results: Dict[int, Dict],
    real_edef: Dict[int, Dict],
    weights: Dict[int, float],
    process: str,  # 'reduction' ou 'oxidation'
    edef_key_method: str  # clé pour extraire E_def de la méthode
) -> Tuple[float, float]:
    """
    Calcule les E_def moyennés par Boltzmann pour une méthode donnée et réel.

    Args:
        method_results: Résultats de la méthode {conf_num: {'reduction': {...}, 'oxidation': {...}}}
        real_edef: E_def réels {conf_num: {'Edef_red_kJ': ..., 'Edef_ox_kJ': ...}}
        weights: Poids de Boltzmann {conf_num: weight}
        process: 'reduction' ou 'oxidation'
        edef_key_method: Clé pour extraire E_def (ex: 'E_def_kJ', 'E_def_QVC_kJ')

    Returns:
        (avg_method_edef, avg_real_edef)
    """
    total_weight = 0.0
    avg_method = 0.0
    avg_real = 0.0

    edef_key_real = 'Edef_red_kJ' if process == 'reduction' else 'Edef_ox_kJ'

    for conf_num in method_results.keys():
        w = weights.get(conf_num, 0.0)
        if w <= 0:
            continue

        method_data = method_results[conf_num].get(process)
        real_data = real_edef.get(conf_num, {})

        if method_data and method_data.get('success'):
            method_val = method_data.get(edef_key_method, 0.0)
            if method_val is None:
                continue
        else:
            continue

        real_val = real_data.get(edef_key_real, np.nan)
        if np.isnan(real_val):
            continue

        total_weight += w
        avg_method += w * method_val
        avg_real += w * real_val

    if total_weight > 0:
        avg_method /= total_weight
        avg_real /= total_weight
    else:
        avg_method = np.nan
        avg_real = np.nan

    return avg_method, avg_real


def format_mode_contributions(contributions: List[Dict], top_n: int = 5) -> str:
    """Formate les contributions des modes dominants."""
    if not contributions:
        return "  (aucune contribution)"

    lines = []
    for i, c in enumerate(contributions[:top_n]):
        lines.append(f"  Mode {c['mode_num']:3d} ({c['frequency']:7.1f} cm⁻¹): "
                    f"{c['contribution_kj']:8.3f} kJ/mol  (F={c['F_i']:.4f}, k={c['k_i']:.4f})")

    return '\n'.join(lines)


def generate_comparison_report(
    all_results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Génère le rapport de comparaison E_def LVC vs E_def réel.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fichier de comparaison principal
    comparison_file = output_dir / "lvc_edef_comparison.txt"
    contributions_file = output_dir / "lvc_mode_contributions.txt"
    dominant_modes_file = output_dir / "lvc_dominant_modes.txt"
    rmsd_file = output_dir / "lvc_rmsd_analysis.txt"

    # Collecter les données pour les graphiques
    plot_data = {
        'molecules': [],
        'lvc_red': [],
        'real_red': [],
        'lvc_ox': [],
        'real_ox': [],
    }

    # Collecter les modes dominants globaux
    all_dominant_modes_red = []
    all_dominant_modes_ox = []

    with open(comparison_file, 'w') as f_comp, \
         open(contributions_file, 'w') as f_contrib, \
         open(dominant_modes_file, 'w') as f_dom:

        # En-tête du rapport de comparaison
        f_comp.write("=" * 80 + "\n")
        f_comp.write("COMPARAISON E_def LVC vs E_def RÉEL\n")
        f_comp.write("=" * 80 + "\n\n")
        f_comp.write("Méthode LVC: E_def = -Σ (Fᵢ² / 2kᵢ)\n")
        f_comp.write("Méthode réelle: E_def = E_opt - E_sp\n\n")
        f_comp.write("-" * 80 + "\n")
        f_comp.write(f"{'Molécule':<25} {'LVC_red':>10} {'Réel_red':>10} {'Δ_red':>8} "
                    f"{'LVC_ox':>10} {'Réel_ox':>10} {'Δ_ox':>8}\n")
        f_comp.write(f"{'':25} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ)':>8} "
                    f"{'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ)':>8}\n")
        f_comp.write("-" * 80 + "\n")

        # En-tête du rapport de contributions
        f_contrib.write("=" * 80 + "\n")
        f_contrib.write("CONTRIBUTIONS DES MODES NORMAUX À E_def LVC\n")
        f_contrib.write("=" * 80 + "\n\n")

        # En-tête du rapport de modes dominants
        f_dom.write("=" * 80 + "\n")
        f_dom.write("MODES DOMINANTS (top 5 par molécule)\n")
        f_dom.write("=" * 80 + "\n\n")

        for mol_name, data in sorted(all_results.items()):
            avg_lvc_red = data.get('avg_lvc_red', np.nan)
            avg_real_red = data.get('avg_real_red', np.nan)
            avg_lvc_ox = data.get('avg_lvc_ox', np.nan)
            avg_real_ox = data.get('avg_real_ox', np.nan)

            delta_red = avg_lvc_red - avg_real_red if not (np.isnan(avg_lvc_red) or np.isnan(avg_real_red)) else np.nan
            delta_ox = avg_lvc_ox - avg_real_ox if not (np.isnan(avg_lvc_ox) or np.isnan(avg_real_ox)) else np.nan

            # Ligne de comparaison
            f_comp.write(f"{mol_name:<25} {avg_lvc_red:10.2f} {avg_real_red:10.2f} {delta_red:8.2f} "
                        f"{avg_lvc_ox:10.2f} {avg_real_ox:10.2f} {delta_ox:8.2f}\n")

            # Données pour graphiques
            if not np.isnan(avg_lvc_red) and not np.isnan(avg_real_red):
                plot_data['molecules'].append(mol_name)
                plot_data['lvc_red'].append(avg_lvc_red)
                plot_data['real_red'].append(avg_real_red)
                plot_data['lvc_ox'].append(avg_lvc_ox if not np.isnan(avg_lvc_ox) else 0)
                plot_data['real_ox'].append(avg_real_ox if not np.isnan(avg_real_ox) else 0)

            # Contributions détaillées
            f_contrib.write(f"\n{'='*60}\n")
            f_contrib.write(f"{mol_name}\n")
            f_contrib.write(f"{'='*60}\n")

            for conf_num, conf_data in sorted(data.get('conformers', {}).items()):
                f_contrib.write(f"\n  Conformère {conf_num}:\n")

                # Réduction
                red_data = conf_data.get('reduction')
                if red_data and red_data.get('success'):
                    f_contrib.write(f"\n    RÉDUCTION (E_def_LVC = {red_data['E_def_lvc_kj']:.2f} kJ/mol):\n")
                    f_contrib.write(f"    Modes utilisés: {red_data['n_modes_used']}/{red_data['n_modes_total']}\n")
                    f_contrib.write("    Top 5 contributions:\n")
                    for c in red_data['top_contributions'][:5]:
                        f_contrib.write(f"      Mode {c['mode_num']:3d} ({c['frequency']:7.1f} cm⁻¹): "
                                       f"{c['contribution_kj']:8.3f} kJ/mol\n")
                        all_dominant_modes_red.append({
                            'molecule': mol_name,
                            'conf': conf_num,
                            'mode': c['mode_num'],
                            'frequency': c['frequency'],
                            'contribution': c['contribution_kj'],
                        })

                # Oxydation
                ox_data = conf_data.get('oxidation')
                if ox_data and ox_data.get('success'):
                    f_contrib.write(f"\n    OXYDATION (E_def_LVC = {ox_data['E_def_lvc_kj']:.2f} kJ/mol):\n")
                    f_contrib.write(f"    Modes utilisés: {ox_data['n_modes_used']}/{ox_data['n_modes_total']}\n")
                    f_contrib.write("    Top 5 contributions:\n")
                    for c in ox_data['top_contributions'][:5]:
                        f_contrib.write(f"      Mode {c['mode_num']:3d} ({c['frequency']:7.1f} cm⁻¹): "
                                       f"{c['contribution_kj']:8.3f} kJ/mol\n")
                        all_dominant_modes_ox.append({
                            'molecule': mol_name,
                            'conf': conf_num,
                            'mode': c['mode_num'],
                            'frequency': c['frequency'],
                            'contribution': c['contribution_kj'],
                        })

            # Modes dominants pour cette molécule
            f_dom.write(f"\n{mol_name}:\n")
            f_dom.write("-" * 40 + "\n")

            # Agréger les contributions par plage de fréquence
            freq_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000)]

            for conf_num, conf_data in sorted(data.get('conformers', {}).items()):
                f_dom.write(f"  Conf {conf_num}:\n")

                for process_name, process_key in [('RED', 'reduction'), ('OX', 'oxidation')]:
                    proc_data = conf_data.get(process_key)
                    if proc_data and proc_data.get('success'):
                        # Calculer la contribution par plage
                        range_contribs = {r: 0.0 for r in freq_ranges}
                        for c in proc_data.get('all_contributions', []):
                            freq = c['frequency']
                            for low, high in freq_ranges:
                                if low <= freq < high:
                                    range_contribs[(low, high)] += c['contribution_kj']
                                    break

                        f_dom.write(f"    {process_name}: ")
                        for (low, high), contrib in range_contribs.items():
                            f_dom.write(f"[{low}-{high}]:{contrib:.1f} ")
                        f_dom.write(f"(Total: {proc_data['E_def_lvc_kj']:.1f} kJ/mol)\n")

        # Statistiques finales
        f_comp.write("-" * 80 + "\n\n")

        # Calculer les statistiques globales
        if plot_data['lvc_red']:
            lvc_red = np.array(plot_data['lvc_red'])
            real_red = np.array(plot_data['real_red'])
            lvc_ox = np.array(plot_data['lvc_ox'])
            real_ox = np.array(plot_data['real_ox'])

            # Corrélation réduction
            mask_red = ~np.isnan(lvc_red) & ~np.isnan(real_red)
            if np.sum(mask_red) >= 3:
                from scipy.stats import pearsonr
                r_red, p_red = pearsonr(lvc_red[mask_red], real_red[mask_red])
                mae_red = np.mean(np.abs(lvc_red[mask_red] - real_red[mask_red]))
                rmse_red = np.sqrt(np.mean((lvc_red[mask_red] - real_red[mask_red])**2))

                f_comp.write("STATISTIQUES RÉDUCTION:\n")
                f_comp.write(f"  Corrélation (R): {r_red:.4f}\n")
                f_comp.write(f"  R²: {r_red**2:.4f}\n")
                f_comp.write(f"  p-value: {p_red:.2e}\n")
                f_comp.write(f"  MAE: {mae_red:.2f} kJ/mol\n")
                f_comp.write(f"  RMSE: {rmse_red:.2f} kJ/mol\n\n")

            # Corrélation oxydation
            mask_ox = ~np.isnan(lvc_ox) & ~np.isnan(real_ox) & (np.array(plot_data['lvc_ox']) != 0)
            if np.sum(mask_ox) >= 3:
                r_ox, p_ox = pearsonr(lvc_ox[mask_ox], real_ox[mask_ox])
                mae_ox = np.mean(np.abs(lvc_ox[mask_ox] - real_ox[mask_ox]))
                rmse_ox = np.sqrt(np.mean((lvc_ox[mask_ox] - real_ox[mask_ox])**2))

                f_comp.write("STATISTIQUES OXYDATION:\n")
                f_comp.write(f"  Corrélation (R): {r_ox:.4f}\n")
                f_comp.write(f"  R²: {r_ox**2:.4f}\n")
                f_comp.write(f"  p-value: {p_ox:.2e}\n")
                f_comp.write(f"  MAE: {mae_ox:.2f} kJ/mol\n")
                f_comp.write(f"  RMSE: {rmse_ox:.2f} kJ/mol\n\n")

    print_success(f"Rapport de comparaison: {comparison_file}")
    print_success(f"Contributions par mode: {contributions_file}")
    print_success(f"Modes dominants: {dominant_modes_file}")

    # Générer le rapport RMSD
    generate_rmsd_report(all_results, rmsd_file)
    print_success(f"Analyse RMSD: {rmsd_file}")

    return plot_data


def generate_rmsd_report(all_results: Dict[str, Dict], rmsd_file: Path) -> None:
    """
    Génère le rapport RMSD pour chaque conformère.

    Args:
        all_results: Résultats de toutes les molécules
        rmsd_file: Chemin du fichier de sortie
    """
    with open(rmsd_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("ANALYSE RMSD - Comparaison des géométries neutres vs réduites/oxydées\n")
        f.write("=" * 100 + "\n\n")
        f.write("RMSD < 0.3 Å : Changement géométrique minimal (même conformère)\n")
        f.write("RMSD > 0.5 Å : Changement géométrique significatif (possible changement de conformère)\n")
        f.write("RMSD > 1.0 Å : Changement conformationnel majeur\n\n")

        f.write("-" * 100 + "\n")
        f.write(f"{'Molécule':<20} {'Conf':<6} {'RMSD_red (Å)':<14} {'RMSD_ox (Å)':<14} "
               f"{'Edef_red (kJ)':<14} {'Edef_ox (kJ)':<14} {'Alerte':<20}\n")
        f.write("-" * 100 + "\n")

        # Statistiques globales
        all_rmsd_red = []
        all_rmsd_ox = []
        alerts = []

        for mol_name, data in sorted(all_results.items()):
            rmsd_data = data.get('rmsd_data', {})
            conformers = data.get('conformers', {})

            for conf_num in sorted(rmsd_data.keys()):
                rmsd_red = rmsd_data[conf_num].get('rmsd_red')
                rmsd_ox = rmsd_data[conf_num].get('rmsd_ox')

                # Récupérer les Edef réels pour ce conformère
                conf_lvc = conformers.get(conf_num, {})
                edef_red = None
                edef_ox = None

                if conf_lvc.get('reduction') and conf_lvc['reduction'].get('success'):
                    edef_red = conf_lvc['reduction'].get('E_def_lvc_kj')
                if conf_lvc.get('oxidation') and conf_lvc['oxidation'].get('success'):
                    edef_ox = conf_lvc['oxidation'].get('E_def_lvc_kj')

                # Déterminer les alertes
                alert_msg = ""
                if rmsd_red is not None:
                    all_rmsd_red.append(rmsd_red)
                    if rmsd_red > 1.0:
                        alert_msg = "⚠ MAJEUR (red)"
                        alerts.append((mol_name, conf_num, 'red', rmsd_red))
                    elif rmsd_red > 0.5:
                        alert_msg = "! Significatif (red)"
                        alerts.append((mol_name, conf_num, 'red', rmsd_red))

                if rmsd_ox is not None:
                    all_rmsd_ox.append(rmsd_ox)
                    if rmsd_ox > 1.0:
                        if alert_msg:
                            alert_msg += " + MAJEUR (ox)"
                        else:
                            alert_msg = "⚠ MAJEUR (ox)"
                        alerts.append((mol_name, conf_num, 'ox', rmsd_ox))
                    elif rmsd_ox > 0.5:
                        if alert_msg:
                            alert_msg += " + Signif (ox)"
                        else:
                            alert_msg = "! Significatif (ox)"
                        alerts.append((mol_name, conf_num, 'ox', rmsd_ox))

                # Formater les valeurs
                rmsd_red_str = f"{rmsd_red:.4f}" if rmsd_red is not None else "N/A"
                rmsd_ox_str = f"{rmsd_ox:.4f}" if rmsd_ox is not None else "N/A"
                edef_red_str = f"{edef_red:.2f}" if edef_red is not None else "N/A"
                edef_ox_str = f"{edef_ox:.2f}" if edef_ox is not None else "N/A"

                f.write(f"{mol_name:<20} {conf_num:<6} {rmsd_red_str:<14} {rmsd_ox_str:<14} "
                       f"{edef_red_str:<14} {edef_ox_str:<14} {alert_msg:<20}\n")

        f.write("-" * 100 + "\n\n")

        # Statistiques globales
        f.write("STATISTIQUES GLOBALES:\n")
        f.write("-" * 40 + "\n")

        if all_rmsd_red:
            f.write(f"\nRMSD Réduction (neutre → réduit):\n")
            f.write(f"  Nombre de conformères: {len(all_rmsd_red)}\n")
            f.write(f"  Moyenne: {np.mean(all_rmsd_red):.4f} Å\n")
            f.write(f"  Écart-type: {np.std(all_rmsd_red):.4f} Å\n")
            f.write(f"  Min: {np.min(all_rmsd_red):.4f} Å\n")
            f.write(f"  Max: {np.max(all_rmsd_red):.4f} Å\n")
            f.write(f"  Médiane: {np.median(all_rmsd_red):.4f} Å\n")

        if all_rmsd_ox:
            f.write(f"\nRMSD Oxydation (neutre → oxydé):\n")
            f.write(f"  Nombre de conformères: {len(all_rmsd_ox)}\n")
            f.write(f"  Moyenne: {np.mean(all_rmsd_ox):.4f} Å\n")
            f.write(f"  Écart-type: {np.std(all_rmsd_ox):.4f} Å\n")
            f.write(f"  Min: {np.min(all_rmsd_ox):.4f} Å\n")
            f.write(f"  Max: {np.max(all_rmsd_ox):.4f} Å\n")
            f.write(f"  Médiane: {np.median(all_rmsd_ox):.4f} Å\n")

        # Liste des cas problématiques
        if alerts:
            f.write("\n\nCAS AVEC CHANGEMENT CONFORMATIONNEL SIGNIFICATIF (RMSD > 0.5 Å):\n")
            f.write("-" * 60 + "\n")

            # Trier par RMSD décroissant
            alerts_sorted = sorted(alerts, key=lambda x: x[3], reverse=True)

            for mol, conf, proc, rmsd in alerts_sorted:
                process_name = "réduction" if proc == 'red' else "oxydation"
                f.write(f"  {mol} conf_{conf} ({process_name}): RMSD = {rmsd:.4f} Å\n")

            f.write("\n⚠ Ces cas peuvent indiquer:\n")
            f.write("  - Changement de conformère lors de l'optimisation\n")
            f.write("  - Fréquence imaginaire non corrigée\n")
            f.write("  - Réarrangement structural important\n")
            f.write("  - L'approximation LVC peut être moins fiable pour ces cas\n")


def generate_plots(plot_data: Dict, output_dir: Path) -> None:
    """
    Génère les graphiques de comparaison LVC vs réel.
    """
    if not HAS_MATPLOTLIB:
        print_warning("matplotlib non disponible, graphiques non générés")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Convertir en arrays
    lvc_red = np.array(plot_data['lvc_red'])
    real_red = np.array(plot_data['real_red'])
    lvc_ox = np.array(plot_data['lvc_ox'])
    real_ox = np.array(plot_data['real_ox'])
    molecules = plot_data['molecules']

    # 1. Scatter plot LVC vs Réel pour réduction
    fig, ax = plt.subplots(figsize=(9, 7))

    mask_red = ~np.isnan(lvc_red) & ~np.isnan(real_red)
    if np.sum(mask_red) > 0:
        ax.scatter(real_red[mask_red], lvc_red[mask_red], c='blue', alpha=0.7, s=80)

        # Annoter chaque point avec le nom de la molécule
        mol_names_red = [m for m, keep in zip(molecules, mask_red) if keep]
        for i, name in enumerate(mol_names_red):
            ax.annotate(name, (real_red[mask_red][i], lvc_red[mask_red][i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=6, ha='center', va='bottom')

        # Ligne y=x (prédiction parfaite)
        min_val = min(real_red[mask_red].min(), lvc_red[mask_red].min())
        max_val = max(real_red[mask_red].max(), lvc_red[mask_red].max())
        margin = (max_val - min_val) * 0.1
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'k--', alpha=0.5, label='y = x')

        # Régression linéaire
        if np.sum(mask_red) >= 3:
            from scipy.stats import pearsonr, linregress
            slope, intercept, r, p, se = linregress(real_red[mask_red], lvc_red[mask_red])
            x_fit = np.linspace(min_val - margin, max_val + margin, 100)
            y_fit = slope * x_fit + intercept
            mae = np.mean(np.abs(lvc_red[mask_red] - (slope * real_red[mask_red] + intercept)))
            ax.plot(x_fit, y_fit, 'r-', alpha=0.7, label=f'Fit: MAE={mae:.1f} kJ/mol, R²={r**2:.3f}')

        ax.set_xlabel('$E_{relax}$ réel (kJ/mol)', fontsize=12)
        ax.set_ylabel('$E_{relax}$ LVC (kJ/mol)', fontsize=12)
        ax.set_title('Comparaison $E_{relax}$ LVC vs Réel (Réduction)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "lvc_vs_real_reduction.png", dpi=150)
    plt.close()

    # 2. Scatter plot LVC vs Réel pour oxydation
    fig, ax = plt.subplots(figsize=(9, 7))

    mask_ox = ~np.isnan(lvc_ox) & ~np.isnan(real_ox) & (lvc_ox != 0) & (real_ox != 0)
    if np.sum(mask_ox) > 0:
        ax.scatter(real_ox[mask_ox], lvc_ox[mask_ox], c='red', alpha=0.7, s=80)

        # Annoter chaque point avec le nom de la molécule
        mol_names_ox = [m for m, keep in zip(molecules, mask_ox) if keep]
        for i, name in enumerate(mol_names_ox):
            ax.annotate(name, (real_ox[mask_ox][i], lvc_ox[mask_ox][i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=6, ha='center', va='bottom')

        min_val = min(real_ox[mask_ox].min(), lvc_ox[mask_ox].min())
        max_val = max(real_ox[mask_ox].max(), lvc_ox[mask_ox].max())
        margin = (max_val - min_val) * 0.1
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'k--', alpha=0.5, label='y = x')

        if np.sum(mask_ox) >= 3:
            from scipy.stats import linregress
            slope, intercept, r, p, se = linregress(real_ox[mask_ox], lvc_ox[mask_ox])
            x_fit = np.linspace(min_val - margin, max_val + margin, 100)
            y_fit = slope * x_fit + intercept
            mae = np.mean(np.abs(lvc_ox[mask_ox] - (slope * real_ox[mask_ox] + intercept)))
            ax.plot(x_fit, y_fit, 'r-', alpha=0.7, label=f'Fit: MAE={mae:.1f} kJ/mol, R²={r**2:.3f}')

        ax.set_xlabel('$E_{relax}$ réel (kJ/mol)', fontsize=12)
        ax.set_ylabel('$E_{relax}$ LVC (kJ/mol)', fontsize=12)
        ax.set_title('Comparaison $E_{relax}$ LVC vs Réel (Oxydation)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "lvc_vs_real_oxidation.png", dpi=150)
    plt.close()

    # 3. Bar chart comparant LVC et Réel par molécule
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    x = np.arange(len(molecules))
    width = 0.35

    # Oxydation
    ax = axes[0]
    mask_valid = (lvc_ox != 0) & (real_ox != 0)
    ax.bar(x[mask_valid] - width/2, real_ox[mask_valid], width, label='Réel', color='steelblue', alpha=0.8)
    ax.bar(x[mask_valid] + width/2, lvc_ox[mask_valid], width, label='LVC', color='coral', alpha=0.8)
    ax.set_ylabel('$E_{relax}$ (kJ/mol)')
    ax.set_title('Comparaison $E_{relax}$ LVC vs Réel - Oxydation')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Réduction
    ax = axes[1]
    ax.bar(x - width/2, real_red, width, label='Réel', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, lvc_red, width, label='LVC', color='coral', alpha=0.8)
    ax.set_ylabel('$E_{relax}$ (kJ/mol)')
    ax.set_title('Comparaison $E_{relax}$ LVC vs Réel - Réduction')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / "lvc_vs_real_comparison_bars.png", dpi=150)
    plt.close()

    print_success(f"Graphiques générés dans: {plots_dir}")


def generate_all_methods_comparison(
    all_results: Dict[str, Dict],
    output_dir: Path
) -> Dict[str, List]:
    """
    Génère le rapport comparatif de toutes les méthodes (LVC, Duschinsky, QVC).

    Returns:
        Dict avec les données pour les graphiques
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_file = output_dir / "all_methods_comparison.txt"
    duschinsky_file = output_dir / "duschinsky_edef_comparison.txt"
    qvc_file = output_dir / "qvc_edef_comparison.txt"
    huang_rhys_file = output_dir / "duschinsky_diagnostics.txt"

    # Collecter les données pour les graphiques
    plot_data = {
        'molecules': [],
        'real_red': [], 'real_ox': [],
        'lvc_red': [], 'lvc_ox': [],
        'lvc2_red': [], 'lvc2_ox': [],
        'disp_red': [], 'disp_ox': [],
        'dusch_red': [], 'dusch_ox': [],
        'morse_red': [], 'morse_ox': [],
        'qvc_red': [], 'qvc_ox': [],
        'vpt2_red': [], 'vpt2_ox': [],
        'dhi_red': [], 'dhi_ox': [],
        'mcrc_red': [], 'mcrc_ox': [],
        'ghc_red': [], 'ghc_ox': [],
        'odhd_red': [], 'odhd_ox': [],
        'essc_red': [], 'essc_ox': [],
        'rmsd_red': [], 'rmsd_ox': [],
    }

    with open(comparison_file, 'w') as f_all, \
         open(duschinsky_file, 'w') as f_dusch, \
         open(qvc_file, 'w') as f_qvc, \
         open(huang_rhys_file, 'w') as f_hr:

        # En-tête du rapport comparatif global
        f_all.write("=" * 140 + "\n")
        f_all.write("COMPARAISON DE TOUTES LES MÉTHODES VIBRONIQUES\n")
        f_all.write("=" * 140 + "\n\n")
        f_all.write("Méthodes:\n")
        f_all.write("  - LVC (Linear Vibronic Coupling): E_def = -Σ Fᵢ²/(2kᵢ) [modes neutres, gradients SP]\n")
        f_all.write("  - LVC2 (ordre 2): E_def = -Σ (g·N'ⱼ)²/(2k'ⱼ) [modes ioniques du SP, gradients SP]\n")
        f_all.write("  - Displacement neutre: E_def = -½ Σ kᵢ(Nᵢ·ΔR)² [modes neutres, déplacement réel — validation LVC]\n")
        f_all.write("  - Duschinsky: E_def = -½ Σ k'ⱼ(ΔQⱼ)² [modes ioniques, déplacement réel, harmonique]\n")
        f_all.write("  - Morse-Duschinsky: correction Morse pour modes anharmoniques [modes ioniques, gradients SP]\n")
        f_all.write("  - VPT2-cubique: correction cubique E = (1/3)F·Δq - (1/6)k·Δq² [modes ioniques, gradients SP, k=V''(Δq)]\n")
        f_all.write("  - QVC (Quadratic Vibronic Coupling): LVC × (1 + correction)\n")
        f_all.write("  - DHI (Dual-Hessian Interpolation): -½ g^T · [(1-α)H_ion + αH_neutral]^{-1} · g\n")
        f_all.write("  - MCRC (Mode Curvature Ratio): LVC2 + β·(curvature ratio correction)\n")
        f_all.write("  - GHC (Gradient-Hessian Consistency): LVC2 × (1 + γ·(A-1))\n")
        f_all.write("  - ODHD (Off-Diagonal ΔH): LVC2 + η·(inter-mode coupling from ΔH)\n")
        f_all.write("  - ESSC (Eigenvalue Spectral Shift): LVC2 × (1 + μ·S/‖H_ion‖_F)\n")
        f_all.write("  - Réel: E_def = E_opt - E_sp\n\n")

        # Tableau de réduction
        f_all.write("=" * 180 + "\n")
        f_all.write("RÉDUCTION (M + e⁻ → M⁻)\n")
        f_all.write("-" * 180 + "\n")
        f_all.write(f"{'Molécule':<20} {'Réel':>10} {'LVC':>10} {'LVC2':>10} {'Disp':>10} {'Dusch':>10} {'Morse':>10} {'VPT2':>10} {'QVC':>10} "
                   f"{'Δ_LVC':>8} {'Δ_LVC2':>8} {'Δ_Disp':>8} {'Δ_Dusch':>8} {'Δ_Morse':>8} {'Δ_VPT2':>8} {'Δ_QVC':>8} {'RMSD':>8}\n")
        f_all.write(f"{'':20} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} "
                   f"{'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(Å)':>8}\n")
        f_all.write("-" * 200 + "\n")

        # En-tête Duschinsky
        f_dusch.write("=" * 100 + "\n")
        f_dusch.write("ANALYSE DUSCHINSKY (modes ioniques) - E_def\n")
        f_dusch.write("=" * 100 + "\n\n")
        f_dusch.write("Méthode: E_def = -½ Σⱼ k'ⱼ (N'ⱼ · ΔR)²\n")
        f_dusch.write("N'ⱼ = modes normaux de l'ion optimisé, k'ⱼ = constantes de force ioniques\n")
        f_dusch.write("ΔR = coords_neutre - coords_ion (en Bohr)\n\n")

        # En-tête QVC
        f_qvc.write("=" * 100 + "\n")
        f_qvc.write("ANALYSE QVC (Quadratic Vibronic Coupling)\n")
        f_qvc.write("=" * 100 + "\n\n")
        f_qvc.write("Méthode: E_def_QVC = E_def_LVC × (1 + correction)\n")
        f_qvc.write("correction = Σᵢ (Δωᵢ/ωᵢ) × (λᵢ/λ_total)\n\n")

        # En-tête Huang-Rhys / Diagnostics
        f_hr.write("=" * 100 + "\n")
        f_hr.write("DIAGNOSTICS PAR MOLÉCULE\n")
        f_hr.write("=" * 100 + "\n\n")
        f_hr.write("Matrice de Duschinsky J: diag_norm ~ 1 = modes non mélangés, off_diag_norm ~ 0 = pas de rotation\n")
        f_hr.write("E_def par méthode: LVC (gradients), Disp (déplacement, modes neutres), Dusch (déplacement, modes ion)\n\n")

        # Données de réduction
        for mol_name, data in sorted(all_results.items()):
            real_red = data.get('avg_real_red', np.nan)
            lvc_red = data.get('avg_lvc_red', np.nan)
            lvc2_red = data.get('avg_lvc2_red', np.nan)
            disp_red = data.get('avg_disp_red', np.nan)
            dusch_red = data.get('avg_dusch_red', np.nan)
            morse_red = data.get('avg_morse_red', np.nan)
            vpt2_red = data.get('avg_vpt2_red', np.nan)
            qvc_red = data.get('avg_qvc_red', np.nan)

            # RMSD moyen pour réduction
            rmsd_data = data.get('rmsd_data', {})
            rmsd_red_vals = [v.get('rmsd_red') for v in rmsd_data.values() if v.get('rmsd_red') is not None]
            avg_rmsd_red = np.mean(rmsd_red_vals) if rmsd_red_vals else np.nan

            delta_lvc = lvc_red - real_red if not (np.isnan(lvc_red) or np.isnan(real_red)) else np.nan
            delta_lvc2 = lvc2_red - real_red if not (np.isnan(lvc2_red) or np.isnan(real_red)) else np.nan
            delta_disp = disp_red - real_red if not (np.isnan(disp_red) or np.isnan(real_red)) else np.nan
            delta_dusch = dusch_red - real_red if not (np.isnan(dusch_red) or np.isnan(real_red)) else np.nan
            delta_morse = morse_red - real_red if not (np.isnan(morse_red) or np.isnan(real_red)) else np.nan
            delta_vpt2 = vpt2_red - real_red if not (np.isnan(vpt2_red) or np.isnan(real_red)) else np.nan
            delta_qvc = qvc_red - real_red if not (np.isnan(qvc_red) or np.isnan(real_red)) else np.nan

            f_all.write(f"{mol_name:<20} {real_red:10.2f} {lvc_red:10.2f} {lvc2_red:10.2f} {disp_red:10.2f} {dusch_red:10.2f} {morse_red:10.2f} {vpt2_red:10.2f} {qvc_red:10.2f} "
                       f"{delta_lvc:8.2f} {delta_lvc2:8.2f} {delta_disp:8.2f} {delta_dusch:8.2f} {delta_morse:8.2f} {delta_vpt2:8.2f} {delta_qvc:8.2f} {avg_rmsd_red:8.4f}\n")

            # Collecter pour graphiques
            if not np.isnan(real_red):
                plot_data['molecules'].append(mol_name)
                plot_data['real_red'].append(real_red)
                plot_data['lvc_red'].append(lvc_red if not np.isnan(lvc_red) else 0)
                plot_data['lvc2_red'].append(lvc2_red if not np.isnan(lvc2_red) else 0)
                plot_data['disp_red'].append(disp_red if not np.isnan(disp_red) else 0)
                plot_data['dusch_red'].append(dusch_red if not np.isnan(dusch_red) else 0)
                plot_data['morse_red'].append(morse_red if not np.isnan(morse_red) else 0)
                plot_data['vpt2_red'].append(vpt2_red if not np.isnan(vpt2_red) else 0)
                plot_data['qvc_red'].append(qvc_red if not np.isnan(qvc_red) else 0)
                for dh_name in ['DHI', 'MCRC', 'GHC', 'ODHD', 'ESSC']:
                    dh_val = data.get(f'avg_{dh_name}_red', np.nan)
                    plot_data[f'{dh_name.lower()}_red'].append(dh_val if not np.isnan(dh_val) else 0)
                plot_data['rmsd_red'].append(avg_rmsd_red if not np.isnan(avg_rmsd_red) else 0)

                real_ox = data.get('avg_real_ox', np.nan)
                lvc_ox = data.get('avg_lvc_ox', np.nan)
                lvc2_ox = data.get('avg_lvc2_ox', np.nan)
                disp_ox = data.get('avg_disp_ox', np.nan)
                dusch_ox = data.get('avg_dusch_ox', np.nan)
                morse_ox = data.get('avg_morse_ox', np.nan)
                vpt2_ox = data.get('avg_vpt2_ox', np.nan)
                qvc_ox = data.get('avg_qvc_ox', np.nan)
                rmsd_ox_vals = [v.get('rmsd_ox') for v in rmsd_data.values() if v.get('rmsd_ox') is not None]
                avg_rmsd_ox = np.mean(rmsd_ox_vals) if rmsd_ox_vals else np.nan

                plot_data['real_ox'].append(real_ox if not np.isnan(real_ox) else 0)
                plot_data['lvc_ox'].append(lvc_ox if not np.isnan(lvc_ox) else 0)
                plot_data['lvc2_ox'].append(lvc2_ox if not np.isnan(lvc2_ox) else 0)
                plot_data['disp_ox'].append(disp_ox if not np.isnan(disp_ox) else 0)
                plot_data['dusch_ox'].append(dusch_ox if not np.isnan(dusch_ox) else 0)
                plot_data['morse_ox'].append(morse_ox if not np.isnan(morse_ox) else 0)
                plot_data['vpt2_ox'].append(vpt2_ox if not np.isnan(vpt2_ox) else 0)
                plot_data['qvc_ox'].append(qvc_ox if not np.isnan(qvc_ox) else 0)
                for dh_name in ['DHI', 'MCRC', 'GHC', 'ODHD', 'ESSC']:
                    dh_val = data.get(f'avg_{dh_name}_ox', np.nan)
                    plot_data[f'{dh_name.lower()}_ox'].append(dh_val if not np.isnan(dh_val) else 0)
                plot_data['rmsd_ox'].append(avg_rmsd_ox if not np.isnan(avg_rmsd_ox) else 0)

        # Tableau d'oxydation
        f_all.write("\n" + "=" * 200 + "\n")
        f_all.write("OXYDATION (M → M⁺ + e⁻)\n")
        f_all.write("-" * 200 + "\n")
        f_all.write(f"{'Molécule':<20} {'Réel':>10} {'LVC':>10} {'LVC2':>10} {'Disp':>10} {'Dusch':>10} {'Morse':>10} {'VPT2':>10} {'QVC':>10} "
                   f"{'Δ_LVC':>8} {'Δ_LVC2':>8} {'Δ_Disp':>8} {'Δ_Dusch':>8} {'Δ_Morse':>8} {'Δ_VPT2':>8} {'Δ_QVC':>8} {'RMSD':>8}\n")
        f_all.write(f"{'':20} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} {'(kJ/mol)':>10} "
                   f"{'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(kJ)':>8} {'(Å)':>8}\n")
        f_all.write("-" * 200 + "\n")

        for mol_name, data in sorted(all_results.items()):
            real_ox = data.get('avg_real_ox', np.nan)
            lvc_ox = data.get('avg_lvc_ox', np.nan)
            lvc2_ox = data.get('avg_lvc2_ox', np.nan)
            disp_ox = data.get('avg_disp_ox', np.nan)
            dusch_ox = data.get('avg_dusch_ox', np.nan)
            morse_ox = data.get('avg_morse_ox', np.nan)
            vpt2_ox = data.get('avg_vpt2_ox', np.nan)
            qvc_ox = data.get('avg_qvc_ox', np.nan)

            rmsd_data = data.get('rmsd_data', {})
            rmsd_ox_vals = [v.get('rmsd_ox') for v in rmsd_data.values() if v.get('rmsd_ox') is not None]
            avg_rmsd_ox = np.mean(rmsd_ox_vals) if rmsd_ox_vals else np.nan

            delta_lvc = lvc_ox - real_ox if not (np.isnan(lvc_ox) or np.isnan(real_ox)) else np.nan
            delta_lvc2 = lvc2_ox - real_ox if not (np.isnan(lvc2_ox) or np.isnan(real_ox)) else np.nan
            delta_disp = disp_ox - real_ox if not (np.isnan(disp_ox) or np.isnan(real_ox)) else np.nan
            delta_dusch = dusch_ox - real_ox if not (np.isnan(dusch_ox) or np.isnan(real_ox)) else np.nan
            delta_morse = morse_ox - real_ox if not (np.isnan(morse_ox) or np.isnan(real_ox)) else np.nan
            delta_vpt2 = vpt2_ox - real_ox if not (np.isnan(vpt2_ox) or np.isnan(real_ox)) else np.nan
            delta_qvc = qvc_ox - real_ox if not (np.isnan(qvc_ox) or np.isnan(real_ox)) else np.nan

            f_all.write(f"{mol_name:<20} {real_ox:10.2f} {lvc_ox:10.2f} {lvc2_ox:10.2f} {disp_ox:10.2f} {dusch_ox:10.2f} {morse_ox:10.2f} {vpt2_ox:10.2f} {qvc_ox:10.2f} "
                       f"{delta_lvc:8.2f} {delta_lvc2:8.2f} {delta_disp:8.2f} {delta_dusch:8.2f} {delta_morse:8.2f} {delta_vpt2:8.2f} {delta_qvc:8.2f} {avg_rmsd_ox:8.4f}\n")

        # Tableau Dual-Hessian — Réduction
        dh_names = ['DHI', 'MCRC', 'GHC', 'ODHD', 'ESSC']
        f_all.write("\n" + "=" * 140 + "\n")
        f_all.write("DUAL-HESSIAN CORRECTIONS — RÉDUCTION (M + e⁻ → M⁻)\n")
        f_all.write("-" * 140 + "\n")
        f_all.write(f"{'Molécule':<20} {'Réel':>10} {'LVC2':>10}")
        for n in dh_names:
            f_all.write(f" {n:>10}")
        for n in dh_names:
            f_all.write(f" {'Δ_'+n:>8}")
        f_all.write(f" {'RMSD':>8}\n")

        f_all.write(f"{'':20} {'(kJ/mol)':>10} {'(kJ/mol)':>10}")
        for _ in dh_names:
            f_all.write(f" {'(kJ/mol)':>10}")
        for _ in dh_names:
            f_all.write(f" {'(kJ)':>8}")
        f_all.write(f" {'(Å)':>8}\n")
        f_all.write("-" * 140 + "\n")

        for mol_name, data in sorted(all_results.items()):
            real_red = data.get('avg_real_red', np.nan)
            lvc2_red = data.get('avg_lvc2_red', np.nan)
            rmsd_data_mol = data.get('rmsd_data', {})
            rmsd_red_vals = [v.get('rmsd_red') for v in rmsd_data_mol.values() if v.get('rmsd_red') is not None]
            avg_rmsd = np.mean(rmsd_red_vals) if rmsd_red_vals else np.nan

            dh_vals = {n: data.get(f'avg_{n}_red', np.nan) for n in dh_names}
            dh_deltas = {}
            for n in dh_names:
                v = dh_vals[n]
                dh_deltas[n] = v - real_red if not (np.isnan(v) or np.isnan(real_red)) else np.nan

            f_all.write(f"{mol_name:<20} {real_red:10.2f} {lvc2_red:10.2f}")
            for n in dh_names:
                f_all.write(f" {dh_vals[n]:10.2f}")
            for n in dh_names:
                f_all.write(f" {dh_deltas[n]:8.2f}")
            f_all.write(f" {avg_rmsd:8.4f}\n")

        # Tableau Dual-Hessian — Oxydation
        f_all.write("\n" + "=" * 140 + "\n")
        f_all.write("DUAL-HESSIAN CORRECTIONS — OXYDATION (M → M⁺ + e⁻)\n")
        f_all.write("-" * 140 + "\n")
        f_all.write(f"{'Molécule':<20} {'Réel':>10} {'LVC2':>10}")
        for n in dh_names:
            f_all.write(f" {n:>10}")
        for n in dh_names:
            f_all.write(f" {'Δ_'+n:>8}")
        f_all.write(f" {'RMSD':>8}\n")

        f_all.write(f"{'':20} {'(kJ/mol)':>10} {'(kJ/mol)':>10}")
        for _ in dh_names:
            f_all.write(f" {'(kJ/mol)':>10}")
        for _ in dh_names:
            f_all.write(f" {'(kJ)':>8}")
        f_all.write(f" {'(Å)':>8}\n")
        f_all.write("-" * 140 + "\n")

        for mol_name, data in sorted(all_results.items()):
            real_ox = data.get('avg_real_ox', np.nan)
            lvc2_ox = data.get('avg_lvc2_ox', np.nan)
            rmsd_data_mol = data.get('rmsd_data', {})
            rmsd_ox_vals = [v.get('rmsd_ox') for v in rmsd_data_mol.values() if v.get('rmsd_ox') is not None]
            avg_rmsd = np.mean(rmsd_ox_vals) if rmsd_ox_vals else np.nan

            dh_vals = {n: data.get(f'avg_{n}_ox', np.nan) for n in dh_names}
            dh_deltas = {}
            for n in dh_names:
                v = dh_vals[n]
                dh_deltas[n] = v - real_ox if not (np.isnan(v) or np.isnan(real_ox)) else np.nan

            f_all.write(f"{mol_name:<20} {real_ox:10.2f} {lvc2_ox:10.2f}")
            for n in dh_names:
                f_all.write(f" {dh_vals[n]:10.2f}")
            for n in dh_names:
                f_all.write(f" {dh_deltas[n]:8.2f}")
            f_all.write(f" {avg_rmsd:8.4f}\n")

        # Statistiques globales
        f_all.write("\n" + "=" * 160 + "\n")
        f_all.write("STATISTIQUES GLOBALES\n")
        f_all.write("-" * 160 + "\n\n")

        for process, suffix in [('RÉDUCTION', '_red'), ('OXYDATION', '_ox')]:
            f_all.write(f"{process}:\n")

            real_arr = np.array(plot_data[f'real{suffix}'])
            lvc_arr = np.array(plot_data[f'lvc{suffix}'])
            lvc2_arr = np.array(plot_data[f'lvc2{suffix}'])
            disp_arr = np.array(plot_data[f'disp{suffix}'])
            dusch_arr = np.array(plot_data[f'dusch{suffix}'])
            morse_arr = np.array(plot_data[f'morse{suffix}'])
            vpt2_arr = np.array(plot_data[f'vpt2{suffix}'])
            qvc_arr = np.array(plot_data[f'qvc{suffix}'])

            methods_list = [
                ('LVC', lvc_arr), ('LVC2', lvc2_arr),
                ('Disp. neutre', disp_arr), ('Duschinsky', dusch_arr),
                ('Morse-Dusch', morse_arr), ('VPT2-cubique', vpt2_arr), ('QVC', qvc_arr),
            ]
            # Add Dual-Hessian methods
            for dh_name in dh_names:
                dh_arr = np.array(plot_data[f'{dh_name.lower()}{suffix}'])
                methods_list.append((dh_name, dh_arr))

            for method_name, method_arr in methods_list:
                mask = (method_arr != 0) & (real_arr != 0) & ~np.isnan(method_arr) & ~np.isnan(real_arr)
                if np.sum(mask) >= 3:
                    from scipy.stats import pearsonr
                    r, p = pearsonr(method_arr[mask], real_arr[mask])
                    mae = np.mean(np.abs(method_arr[mask] - real_arr[mask]))
                    rmse = np.sqrt(np.mean((method_arr[mask] - real_arr[mask])**2))
                    f_all.write(f"  {method_name:12s}: R²={r**2:.4f}, MAE={mae:.2f} kJ/mol, RMSE={rmse:.2f} kJ/mol (n={np.sum(mask)})\n")

            f_all.write("\n")

        # Détails Duschinsky par molécule
        for mol_name, data in sorted(all_results.items()):
            f_dusch.write(f"\n{'='*60}\n")
            f_dusch.write(f"{mol_name}\n")
            f_dusch.write(f"{'='*60}\n")

            for conf_num, conf_data in sorted(data.get('duschinsky_conformers', {}).items()):
                f_dusch.write(f"\n  Conformère {conf_num}:\n")

                for proc_name, proc_key in [('Réduction', 'reduction'), ('Oxydation', 'oxidation')]:
                    proc_data = conf_data.get(proc_key)
                    if proc_data and proc_data.get('success'):
                        f_dusch.write(f"\n    {proc_name} (E_def Duschinsky = {proc_data['E_def_kJ']:.2f} kJ/mol):\n")
                        f_dusch.write(f"    Modes: {proc_data.get('n_modes_used', 'N/A')}/{proc_data.get('n_modes_total', 'N/A')}\n")

                        # Displacement neutre for comparison
                        disp_data = conf_data.get(f'{proc_key}_displacement')
                        if disp_data and disp_data.get('success'):
                            f_dusch.write(f"    E_def Disp. neutre = {disp_data['E_def_kJ']:.2f} kJ/mol\n")

                        # Duschinsky matrix diagnostics
                        mat_data = conf_data.get(f'{proc_key}_duschinsky_matrix')
                        if mat_data and mat_data.get('success'):
                            f_dusch.write(f"    Matrice Duschinsky: diag_norm={mat_data['diag_norm']:.4f}, "
                                        f"off_diag_norm={mat_data['off_diag_norm']:.4f}\n")

                        if proc_data.get('top_contributions'):
                            f_dusch.write("    Top 5 modes:\n")
                            for m in proc_data['top_contributions'][:5]:
                                f_dusch.write(f"      Mode {m['mode_num']:3d} ({m['frequency']:7.1f} cm⁻¹): "
                                            f"ΔQ={m['delta_Q']:.4f} Bohr, k'={m['k_j']:.4f} Eh/Bohr², "
                                            f"E={m['contribution_kj']:.3f} kJ/mol\n")

        # Détails QVC par molécule
        for mol_name, data in sorted(all_results.items()):
            f_qvc.write(f"\n{'='*60}\n")
            f_qvc.write(f"{mol_name}\n")
            f_qvc.write(f"{'='*60}\n")

            for conf_num, conf_data in sorted(data.get('qvc_conformers', {}).items()):
                f_qvc.write(f"\n  Conformère {conf_num}:\n")

                for proc_name, proc_key in [('Réduction', 'reduction'), ('Oxydation', 'oxidation')]:
                    proc_data = conf_data.get(proc_key)
                    if proc_data and proc_data.get('success'):
                        lvc_val = proc_data.get('E_def_LVC_kJ', 0)
                        qvc_val = proc_data.get('E_def_QVC_kJ', 0)
                        corr = proc_data.get('correction_factor', 1.0)

                        f_qvc.write(f"\n    {proc_name}:\n")
                        f_qvc.write(f"      E_def_LVC = {lvc_val:.2f} kJ/mol\n")
                        f_qvc.write(f"      E_def_QVC = {qvc_val:.2f} kJ/mol\n")
                        f_qvc.write(f"      Facteur de correction = {corr:.4f}\n")

        # Diagnostics par molécule
        for mol_name, data in sorted(all_results.items()):
            f_hr.write(f"\n{'='*60}\n")
            f_hr.write(f"{mol_name}\n")
            f_hr.write(f"{'='*60}\n")

            for conf_num, conf_data in sorted(data.get('duschinsky_conformers', {}).items()):
                for proc_name, proc_key in [('RED', 'reduction'), ('OX', 'oxidation')]:
                    dusch_data = conf_data.get(proc_key)
                    disp_data = conf_data.get(f'{proc_key}_displacement')
                    mat_data = conf_data.get(f'{proc_key}_duschinsky_matrix')

                    parts = [f"  Conf {conf_num} {proc_name}:"]
                    if dusch_data and dusch_data.get('success'):
                        parts.append(f"Dusch={dusch_data['E_def_kJ']:.2f}")
                    if disp_data and disp_data.get('success'):
                        parts.append(f"Disp={disp_data['E_def_kJ']:.2f}")
                    if mat_data and mat_data.get('success'):
                        parts.append(f"J_diag={mat_data['diag_norm']:.3f}")
                        parts.append(f"J_offdiag={mat_data['off_diag_norm']:.3f}")
                    f_hr.write(" | ".join(parts) + " kJ/mol\n")

    print_success(f"Rapport comparatif: {comparison_file}")
    print_success(f"Rapport Duschinsky: {duschinsky_file}")
    print_success(f"Rapport QVC: {qvc_file}")
    print_success(f"Diagnostics Duschinsky: {huang_rhys_file}")

    return plot_data


def _scatter_method_vs_real(ax, real_arr, method_arr, mask, color, method_name,
                            mol_names=None):
    """Helper: scatter + y=x + fit on one axis."""
    if np.sum(mask) == 0:
        return
    ax.scatter(real_arr[mask], method_arr[mask], c=color, alpha=0.7, s=80)
    if mol_names is not None:
        names_masked = [m for m, keep in zip(mol_names, mask) if keep]
        for i, name in enumerate(names_masked):
            ax.annotate(name, (real_arr[mask][i], method_arr[mask][i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=6, ha='center', va='bottom')
    min_val = min(real_arr[mask].min(), method_arr[mask].min())
    max_val = max(real_arr[mask].max(), method_arr[mask].max())
    margin = (max_val - min_val) * 0.1 if max_val != min_val else 1.0
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'k--', alpha=0.5)
    if np.sum(mask) >= 3:
        from scipy.stats import linregress
        slope, intercept, r, p, se = linregress(real_arr[mask], method_arr[mask])
        x_fit = np.linspace(min_val - margin, max_val + margin, 100)
        y_fit = slope * x_fit + intercept
        mae = np.mean(np.abs(method_arr[mask] - real_arr[mask]))
        ax.plot(x_fit, y_fit, color=color, alpha=0.7,
                label=f'R²={r**2:.3f}, MAE={mae:.1f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('$E_{relax}$ réel (kJ/mol)')
    ax.set_ylabel(f'$E_{{relax}}$ {method_name} (kJ/mol)')
    ax.set_title(f'{method_name}')
    ax.grid(True, alpha=0.3)


def generate_all_methods_plots(plot_data: Dict, output_dir: Path) -> None:
    """
    Génère les graphiques comparatifs pour toutes les méthodes.
    """
    if not HAS_MATPLOTLIB:
        print_warning("matplotlib non disponible, graphiques non générés")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    molecules = plot_data['molecules']
    if not molecules:
        return

    # Convertir en arrays
    real_red = np.array(plot_data['real_red'])
    lvc_red = np.array(plot_data['lvc_red'])
    lvc2_red = np.array(plot_data['lvc2_red'])
    disp_red = np.array(plot_data['disp_red'])
    dusch_red = np.array(plot_data['dusch_red'])
    morse_red = np.array(plot_data['morse_red'])
    vpt2_red = np.array(plot_data['vpt2_red'])
    qvc_red = np.array(plot_data['qvc_red'])
    rmsd_red = np.array(plot_data['rmsd_red'])

    real_ox = np.array(plot_data['real_ox'])
    lvc_ox = np.array(plot_data['lvc_ox'])
    lvc2_ox = np.array(plot_data['lvc2_ox'])
    disp_ox = np.array(plot_data['disp_ox'])
    dusch_ox = np.array(plot_data['dusch_ox'])
    morse_ox = np.array(plot_data['morse_ox'])
    vpt2_ox = np.array(plot_data['vpt2_ox'])
    qvc_ox = np.array(plot_data['qvc_ox'])
    rmsd_ox = np.array(plot_data['rmsd_ox'])

    # Dual-Hessian arrays (used in DH-specific plots at the end)
    # Declared here so they are in scope for all plot sections

    # Helper for valid mask
    def vmask(method_arr, real_arr):
        return (method_arr != 0) & (real_arr != 0) & ~np.isnan(method_arr) & ~np.isnan(real_arr)

    # 1. All methods scatter - Réduction (LVC, LVC2, Duschinsky, Morse-Duschinsky)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Comparaison des méthodes - Réduction', fontsize=14, fontweight='bold')
    for ax, method_arr, method_name, color in [
        (axes[0], lvc_red, 'LVC', 'blue'),
        (axes[1], lvc2_red, 'LVC2 (ion SP)', 'purple'),
        (axes[2], dusch_red, 'Duschinsky (harm.)', 'green'),
        (axes[3], morse_red, 'Morse-Duschinsky', 'darkorange'),
    ]:
        _scatter_method_vs_real(ax, real_red, method_arr, vmask(method_arr, real_red), color, method_name)
    plt.tight_layout()
    plt.savefig(plots_dir / "all_methods_reduction.png", dpi=150)
    plt.close()

    # 2. All methods scatter - Oxydation
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Comparaison des méthodes - Oxydation', fontsize=14, fontweight='bold')
    for ax, method_arr, method_name, color in [
        (axes[0], lvc_ox, 'LVC', 'blue'),
        (axes[1], lvc2_ox, 'LVC2 (ion SP)', 'purple'),
        (axes[2], dusch_ox, 'Duschinsky (harm.)', 'green'),
        (axes[3], morse_ox, 'Morse-Duschinsky', 'darkorange'),
    ]:
        _scatter_method_vs_real(ax, real_ox, method_arr, vmask(method_arr, real_ox), color, method_name)
    plt.tight_layout()
    plt.savefig(plots_dir / "all_methods_oxidation.png", dpi=150)
    plt.close()

    # 3. Duschinsky vs Real standalone scatter (for rapport)
    for process_name, real_arr, dusch_arr, color, suffix in [
        ('Réduction', real_red, dusch_red, 'green', 'reduction'),
        ('Oxydation', real_ox, dusch_ox, 'green', 'oxidation'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        mask = vmask(dusch_arr, real_arr)
        _scatter_method_vs_real(ax, real_arr, dusch_arr, mask, color, f'Duschinsky ({process_name})')
        plt.tight_layout()
        plt.savefig(plots_dir / f"duschinsky_vs_real_{suffix}.png", dpi=150)
        plt.close()

    # 4. LVC vs LVC2 vs Duschinsky vs Morse side-by-side (for rapport)
    for process_name, real_arr, lvc_arr, lvc2_arr_proc, dusch_arr, morse_arr_proc, suffix in [
        ('Réduction', real_red, lvc_red, lvc2_red, dusch_red, morse_red, 'reduction'),
        ('Oxydation', real_ox, lvc_ox, lvc2_ox, dusch_ox, morse_ox, 'oxidation'),
    ]:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'LVC vs LVC2 vs Duschinsky vs Morse - {process_name}', fontsize=14, fontweight='bold')
        _scatter_method_vs_real(axes[0], real_arr, lvc_arr, vmask(lvc_arr, real_arr), 'blue', 'LVC')
        _scatter_method_vs_real(axes[1], real_arr, lvc2_arr_proc, vmask(lvc2_arr_proc, real_arr), 'purple', 'LVC2 (ion SP)')
        _scatter_method_vs_real(axes[2], real_arr, dusch_arr, vmask(dusch_arr, real_arr), 'green', 'Duschinsky (harm.)')
        _scatter_method_vs_real(axes[3], real_arr, morse_arr_proc, vmask(morse_arr_proc, real_arr), 'darkorange', 'Morse-Duschinsky')
        plt.tight_layout()
        plt.savefig(plots_dir / f"lvc_vs_duschinsky_{suffix}.png", dpi=150)
        plt.close()

    # 4b. LVC vs LVC2 standalone scatter (for rapport)
    for process_name, real_arr, lvc_arr, lvc2_arr_proc, suffix in [
        ('Réduction', real_red, lvc_red, lvc2_red, 'reduction'),
        ('Oxydation', real_ox, lvc_ox, lvc2_ox, 'oxidation'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle(f'LVC vs LVC2 - {process_name}', fontsize=14, fontweight='bold')
        _scatter_method_vs_real(axes[0], real_arr, lvc_arr, vmask(lvc_arr, real_arr), 'blue', 'LVC (modes neutres)', mol_names=molecules)
        _scatter_method_vs_real(axes[1], real_arr, lvc2_arr_proc, vmask(lvc2_arr_proc, real_arr), 'purple', 'LVC2 (modes ion SP)', mol_names=molecules)
        plt.tight_layout()
        plt.savefig(plots_dir / f"lvc_vs_lvc2_{suffix}.png", dpi=150)
        plt.close()

    # 4c. LVC vs QVC standalone scatter (for rapport)
    for process_name, real_arr, lvc_arr, qvc_arr_proc, suffix in [
        ('Réduction', real_red, lvc_red, qvc_red, 'reduction'),
        ('Oxydation', real_ox, lvc_ox, qvc_ox, 'oxidation'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle(f'LVC vs QVC - {process_name}', fontsize=14, fontweight='bold')
        _scatter_method_vs_real(axes[0], real_arr, lvc_arr, vmask(lvc_arr, real_arr), 'blue', 'LVC', mol_names=molecules)
        _scatter_method_vs_real(axes[1], real_arr, qvc_arr_proc, vmask(qvc_arr_proc, real_arr), 'teal', 'QVC', mol_names=molecules)
        plt.tight_layout()
        plt.savefig(plots_dir / f"lvc_vs_qvc_{suffix}.png", dpi=150)
        plt.close()

    # 4d. LVC vs VPT2-cubique standalone scatter (for rapport)
    for process_name, real_arr, lvc_arr, vpt2_arr_proc, suffix in [
        ('Réduction', real_red, lvc_red, vpt2_red, 'reduction'),
        ('Oxydation', real_ox, lvc_ox, vpt2_ox, 'oxidation'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle(f'LVC vs VPT2-cubique - {process_name}', fontsize=14, fontweight='bold')
        _scatter_method_vs_real(axes[0], real_arr, lvc_arr, vmask(lvc_arr, real_arr), 'blue', 'LVC', mol_names=molecules)
        _scatter_method_vs_real(axes[1], real_arr, vpt2_arr_proc, vmask(vpt2_arr_proc, real_arr), 'crimson', 'VPT2-cubique', mol_names=molecules)
        plt.tight_layout()
        plt.savefig(plots_dir / f"lvc_vs_vpt2_{suffix}.png", dpi=150)
        plt.close()

    # 5. Bar chart: Réel / LVC / LVC2 / Duschinsky / Morse par molécule (linear, clipped)
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    x = np.arange(len(molecules))
    n_bars = 5
    width = 0.8 / n_bars

    for ax_idx, (process_name, real_arr, lvc_arr, lvc2_arr_proc, dusch_arr, morse_arr_proc) in enumerate([
        ('Réduction', real_red, lvc_red, lvc2_red, dusch_red, morse_red),
        ('Oxydation', real_ox, lvc_ox, lvc2_ox, dusch_ox, morse_ox),
    ]):
        ax = axes[ax_idx]

        # Determine y-axis limits from Real, LVC, LVC2, and Morse only (exclude Duschinsky)
        all_non_dusch = np.concatenate([
            real_arr[np.isfinite(real_arr)],
            lvc_arr[np.isfinite(lvc_arr)],
            lvc2_arr_proc[np.isfinite(lvc2_arr_proc)],
            morse_arr_proc[np.isfinite(morse_arr_proc)],
        ])
        if len(all_non_dusch) > 0:
            data_min = np.min(all_non_dusch)
            data_max = np.max(all_non_dusch)
            margin = (data_max - data_min) * 0.15
            ymin = data_min - margin
            ymax = data_max + margin
        else:
            ymin, ymax = -100, 10
        # Ensure some positive headroom for annotations
        if ymax < 15:
            ymax = 15

        ax.bar(x - 2*width, real_arr, width, label='Réel', color='steelblue', alpha=0.8)
        ax.bar(x - 1*width, lvc_arr, width, label='LVC', color='coral', alpha=0.8)
        ax.bar(x + 0*width, lvc2_arr_proc, width, label='LVC2 (ion SP)', color='mediumpurple', alpha=0.8)
        ax.bar(x + 1*width, dusch_arr, width, label='Duschinsky', color='forestgreen', alpha=0.8)
        ax.bar(x + 2*width, morse_arr_proc, width, label='Morse-Dusch.', color='darkorange', alpha=0.8)

        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('$E_{relax}$ (kJ/mol)')

        # Annotate Duschinsky bars that are clipped beyond the y-axis limits
        for i, val in enumerate(dusch_arr):
            if np.isfinite(val) and val < ymin:
                ax.text(x[i] + 1*width, ymin + margin * 0.15,
                        f'{val:.0f}',
                        fontsize=5.5, ha='center', va='bottom',
                        color='forestgreen', fontweight='bold',
                        rotation=90)

        ax.set_title(f'{process_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(molecules, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7, ncol=5, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / "all_methods_comparison_bars.png", dpi=150)
    plt.close()

    # 6. Erreur LVC vs RMSD - Réduction (single panel)
    mol_arr = np.array(molecules)
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = vmask(lvc_red, real_red) & (rmsd_red > 0)
    if np.sum(mask) > 0:
        errors = np.abs(lvc_red[mask] - real_red[mask])
        ax.scatter(rmsd_red[mask], errors, c='steelblue', alpha=0.8, s=80,
                   edgecolors='navy', linewidths=0.5, zorder=5)
        for name, xv, yv in zip(mol_arr[mask], rmsd_red[mask], errors):
            ax.annotate(name, (xv, yv), fontsize=8, alpha=0.85,
                        xytext=(5, 5), textcoords='offset points')
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7,
                   linewidth=1.5, label='RMSD = 0.3 Å')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7,
                   linewidth=1.5, label='RMSD = 0.5 Å')
    ax.set_xlabel('RMSD neutre → réduit (Å)', fontsize=12)
    ax.set_ylabel('|Erreur LVC| (kJ/mol)', fontsize=12)
    ax.set_title('Erreur LVC vs RMSD en réduction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "error_vs_rmsd_reduction.png", dpi=200)
    plt.close()

    # 7. Erreur LVC vs Morse-Duschinsky vs RMSD — combined scatter with regime zones
    fig, ax = plt.subplots(figsize=(10, 6))
    mask_lvc = vmask(lvc_red, real_red) & (rmsd_red > 0)
    mask_morse = vmask(morse_red, real_red) & (rmsd_red > 0)

    if np.sum(mask_lvc) > 0:
        err_lvc = np.abs(lvc_red[mask_lvc] - real_red[mask_lvc])
        ax.scatter(rmsd_red[mask_lvc], err_lvc, c='coral', marker='o', s=80,
                   alpha=0.8, label='LVC', edgecolors='darkred', linewidths=0.5)
        for i, idx in enumerate(np.where(mask_lvc)[0]):
            ax.annotate(molecules[idx], (rmsd_red[idx], err_lvc[i]),
                       fontsize=5.5, ha='left', va='bottom', color='darkred',
                       xytext=(3, 3), textcoords='offset points')

    if np.sum(mask_morse) > 0:
        err_morse = np.abs(morse_red[mask_morse] - real_red[mask_morse])
        ax.scatter(rmsd_red[mask_morse], err_morse, c='darkorange', marker='s', s=80,
                   alpha=0.8, label='Morse-Duschinsky', edgecolors='saddlebrown', linewidths=0.5)
        for i, idx in enumerate(np.where(mask_morse)[0]):
            ax.annotate(molecules[idx], (rmsd_red[idx], err_morse[i]),
                       fontsize=5.5, ha='left', va='top', color='saddlebrown',
                       xytext=(3, -3), textcoords='offset points')

    # Regime zones
    ax.axvspan(0, 0.1, alpha=0.08, color='green', label='Morse optimal (RMSD < 0.1 Å)')
    ax.axvspan(0.1, 0.3, alpha=0.08, color='blue', label='LVC optimal (0.1–0.3 Å)')
    ax.axvspan(0.3, ax.get_xlim()[1] if ax.get_xlim()[1] > 0.5 else 1.6,
               alpha=0.08, color='red', label='Aucune méthode fiable (> 0.3 Å)')

    ax.set_xlabel('RMSD neutre → ion réduit (Å)', fontsize=11)
    ax.set_ylabel('|Erreur| (kJ/mol)', fontsize=11)
    ax.set_title('Erreur absolue vs RMSD — Réduction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_vs_rmsd_regimes.png", dpi=150)
    plt.close()

    # 8. Dual-Hessian methods scatter — Réduction (5 methods vs LVC2)
    dh_plot_names = ['DHI', 'MCRC', 'GHC', 'ODHD', 'ESSC']
    dh_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    dh_arrs_red = {n: np.array(plot_data.get(f'{n.lower()}_red', [])) for n in dh_plot_names}
    dh_arrs_ox = {n: np.array(plot_data.get(f'{n.lower()}_ox', [])) for n in dh_plot_names}

    n_dh = len(dh_plot_names)
    if len(real_red) > 0:
        fig, axes = plt.subplots(1, n_dh, figsize=(5 * n_dh, 5))
        fig.suptitle('Dual-Hessian Corrections — Réduction', fontsize=14, fontweight='bold')
        for i, (name, color) in enumerate(zip(dh_plot_names, dh_colors)):
            arr = dh_arrs_red[name]
            if len(arr) == len(real_red):
                _scatter_method_vs_real(axes[i], real_red, arr, vmask(arr, real_red),
                                        color, name, mol_names=molecules)
        plt.tight_layout()
        plt.savefig(plots_dir / "dual_hessian_reduction.png", dpi=150)
        plt.close()

    if len(real_ox) > 0:
        fig, axes = plt.subplots(1, n_dh, figsize=(5 * n_dh, 5))
        fig.suptitle('Dual-Hessian Corrections — Oxydation', fontsize=14, fontweight='bold')
        for i, (name, color) in enumerate(zip(dh_plot_names, dh_colors)):
            arr = dh_arrs_ox[name]
            if len(arr) == len(real_ox):
                _scatter_method_vs_real(axes[i], real_ox, arr, vmask(arr, real_ox),
                                        color, name, mol_names=molecules)
        plt.tight_layout()
        plt.savefig(plots_dir / "dual_hessian_oxidation.png", dpi=150)
        plt.close()

    # 9. LVC2 vs best DH method standalone (reduction)
    if len(real_red) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle('LVC2 vs DHI — Réduction', fontsize=14, fontweight='bold')
        _scatter_method_vs_real(axes[0], real_red, lvc2_red, vmask(lvc2_red, real_red),
                                'purple', 'LVC2', mol_names=molecules)
        dhi_arr = dh_arrs_red.get('DHI', np.array([]))
        if len(dhi_arr) == len(real_red):
            _scatter_method_vs_real(axes[1], real_red, dhi_arr, vmask(dhi_arr, real_red),
                                    '#e41a1c', 'DHI', mol_names=molecules)
        plt.tight_layout()
        plt.savefig(plots_dir / "lvc2_vs_dhi_reduction.png", dpi=150)
        plt.close()

    print_success(f"Graphiques comparatifs générés dans: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Génère un rapport comparant E_def par méthodes vibroniques avec E_def réel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple:
    ./generate_lvc_edef_report.py --batch-dir 1594174/dir_workdir --output-dir lvc_report
    ./generate_lvc_edef_report.py --batch-dir 1594174/dir_workdir --output-dir lvc_report --rmsd-threshold 0.5

Méthodes implémentées:
  1. LVC (Linear Vibronic Coupling): E_def = -Σ Fᵢ²/(2kᵢ)
  2. Duschinsky/Huang-Rhys: λ = Σ ωᵢDᵢ²/2 (utilise géom. ion optimisé)
  3. QVC (Quadratic Vibronic Coupling): LVC × (1 + correction quadratique)
"""
    )

    parser.add_argument('--batch-dir', type=str, required=True,
                       help="Répertoire contenant les workdirs des molécules")
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).resolve().parent.parent / 'results' / 'lvc_edef'),
                       help="Répertoire de sortie pour le rapport (défaut: results/lvc_edef)")
    parser.add_argument('--workdirs', type=str, nargs='+',
                       help="Liste spécifique de workdirs (alternative à --batch-dir)")
    parser.add_argument('--freq-threshold', type=float, default=50.0,
                       help="Seuil minimal de fréquence pour inclure un mode (défaut: 50 cm⁻¹)")
    parser.add_argument('--top-n', type=int, default=10,
                       help="Nombre de modes dominants à rapporter (défaut: 10)")
    parser.add_argument('--no-plots', action='store_true',
                       help="Ne pas générer de graphiques")
    parser.add_argument('--alpha', type=float, default=0.3,
                       help="Paramètre α pour QVC (défaut: 0.3)")
    parser.add_argument('--rmsd-threshold', type=float, default=None,
                       help="Seuil RMSD pour exclure les conformères avec changement conformationnel majeur")

    args = parser.parse_args()

    print_header("ANALYSE VIBRONIQUE - COMPARAISON E_def")
    print("Méthodes: LVC, Duschinsky/Huang-Rhys, QVC")

    # Trouver les workdirs
    if args.workdirs:
        workdirs = [(Path(w), Path(w).name.replace('_workdir', '')) for w in args.workdirs]
    else:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            print_error(f"Répertoire batch non trouvé: {batch_dir}")
            sys.exit(1)
        workdirs = find_molecule_workdirs(batch_dir)

    if not workdirs:
        print_error("Aucun workdir de molécule trouvé")
        sys.exit(1)

    print(f"Molécules trouvées: {len(workdirs)}")
    if args.rmsd_threshold:
        print(f"Seuil RMSD: {args.rmsd_threshold} Å (conformères avec RMSD > seuil seront exclus des stats)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyser chaque molécule
    all_results = {}

    for workdir, mol_name in workdirs:
        print_section(f"Analyse de {mol_name}")

        # Exécuter l'analyse LVC
        print("  → Calcul LVC...")
        lvc_results = run_lvc_analysis_for_molecule(
            workdir, mol_name,
            freq_threshold=args.freq_threshold,
            top_n_modes=args.top_n
        )

        # Exécuter l'analyse Duschinsky (modes ioniques + displacement neutre)
        print("  → Calcul Duschinsky (modes ioniques) + Displacement (modes neutres)...")
        dusch_results = run_duschinsky_analysis_for_molecule(
            workdir, mol_name,
            freq_threshold=args.freq_threshold,
            top_n_modes=args.top_n
        )

        # Exécuter l'analyse QVC
        print("  → Calcul QVC...")
        qvc_results = run_qvc_analysis_for_molecule(
            workdir, mol_name,
            alpha=args.alpha,
            freq_threshold=args.freq_threshold,
            top_n_modes=args.top_n
        )

        # Exécuter l'analyse VPT2-cubique
        print("  → Calcul VPT2-cubique...")
        vpt2_results = run_vpt2_analysis_for_molecule(
            workdir, mol_name,
            freq_threshold=args.freq_threshold,
            top_n_modes=args.top_n
        )

        # Exécuter l'analyse Dual-Hessian (5 stratégies anharmoniques)
        print("  → Calcul Dual-Hessian (DHI/MCRC/GHC/ODHD/ESSC)...")
        dh_results = run_dual_hessian_for_molecule(
            workdir, mol_name,
            freq_threshold=args.freq_threshold,
        )

        # Charger les E_def réels
        real_edef = load_real_edef(workdir, mol_name)

        # Charger les poids de Boltzmann
        weights = _read_conformer_weights_from_summary(workdir)
        if weights is None:
            weights = {conf: data.get('weight', 1.0) for conf, data in real_edef.items()}

        # Calculer les moyennes de Boltzmann pour chaque méthode
        # LVC
        avg_lvc_red, avg_real_red = compute_boltzmann_averaged_edef(
            lvc_results['conformers'], real_edef, weights, 'reduction'
        )
        avg_lvc_ox, avg_real_ox = compute_boltzmann_averaged_edef(
            lvc_results['conformers'], real_edef, weights, 'oxidation'
        )

        # Duschinsky harmonique (modes ioniques)
        avg_dusch_red, _ = compute_boltzmann_averaged_method(
            dusch_results['conformers'], real_edef, weights, 'reduction', 'E_def_kJ'
        )
        avg_dusch_ox, _ = compute_boltzmann_averaged_method(
            dusch_results['conformers'], real_edef, weights, 'oxidation', 'E_def_kJ'
        )

        # Morse-Duschinsky (correction anharmonique)
        morse_conformers = {}
        for conf_num, conf_data in dusch_results['conformers'].items():
            morse_conformers[conf_num] = {
                'reduction': conf_data.get('reduction_morse'),
                'oxidation': conf_data.get('oxidation_morse'),
            }
        avg_morse_red, _ = compute_boltzmann_averaged_method(
            morse_conformers, real_edef, weights, 'reduction', 'E_def_kJ'
        )
        avg_morse_ox, _ = compute_boltzmann_averaged_method(
            morse_conformers, real_edef, weights, 'oxidation', 'E_def_kJ'
        )

        # Displacement neutre (modes neutres — validation LVC)
        # Build a temporary dict compatible with compute_boltzmann_averaged_method
        disp_conformers = {}
        for conf_num, conf_data in dusch_results['conformers'].items():
            disp_conformers[conf_num] = {
                'reduction': conf_data.get('reduction_displacement'),
                'oxidation': conf_data.get('oxidation_displacement'),
            }
        avg_disp_red, _ = compute_boltzmann_averaged_method(
            disp_conformers, real_edef, weights, 'reduction', 'E_def_kJ'
        )
        avg_disp_ox, _ = compute_boltzmann_averaged_method(
            disp_conformers, real_edef, weights, 'oxidation', 'E_def_kJ'
        )

        # QVC
        avg_qvc_red, _ = compute_boltzmann_averaged_method(
            qvc_results['conformers'], real_edef, weights, 'reduction', 'E_def_QVC_kJ'
        )
        avg_qvc_ox, _ = compute_boltzmann_averaged_method(
            qvc_results['conformers'], real_edef, weights, 'oxidation', 'E_def_QVC_kJ'
        )

        # LVC2 (modes ioniques du SP — ordre 2)
        lvc2_conformers = {}
        for conf_num, conf_data in lvc_results['conformers'].items():
            lvc2_conformers[conf_num] = {
                'reduction': conf_data.get('reduction_lvc2'),
                'oxidation': conf_data.get('oxidation_lvc2'),
            }
        avg_lvc2_red, _ = compute_boltzmann_averaged_method(
            lvc2_conformers, real_edef, weights, 'reduction', 'E_def_lvc2_kj'
        )
        avg_lvc2_ox, _ = compute_boltzmann_averaged_method(
            lvc2_conformers, real_edef, weights, 'oxidation', 'E_def_lvc2_kj'
        )

        # VPT2-cubique
        avg_vpt2_red, _ = compute_boltzmann_averaged_method(
            vpt2_results['conformers'], real_edef, weights, 'reduction', 'E_def_kJ'
        )
        avg_vpt2_ox, _ = compute_boltzmann_averaged_method(
            vpt2_results['conformers'], real_edef, weights, 'oxidation', 'E_def_kJ'
        )

        # Dual-Hessian (5 méthodes)
        dh_method_keys = {
            'DHI': 'E_def_DHI_kj',
            'MCRC': 'E_def_MCRC_kj',
            'GHC': 'E_def_GHC_kj',
            'ODHD': 'E_def_ODHD_kj',
            'ESSC': 'E_def_ESSC_kj',
        }
        dh_avgs = {}
        for dh_name, dh_key in dh_method_keys.items():
            avg_red, _ = compute_boltzmann_averaged_method(
                dh_results['conformers'], real_edef, weights, 'reduction', dh_key
            )
            avg_ox, _ = compute_boltzmann_averaged_method(
                dh_results['conformers'], real_edef, weights, 'oxidation', dh_key
            )
            dh_avgs[f'avg_{dh_name}_red'] = avg_red
            dh_avgs[f'avg_{dh_name}_ox'] = avg_ox

        # Afficher un résumé
        dh_red_str = ", ".join(f"{n}={dh_avgs[f'avg_{n}_red']:.2f}" for n in dh_method_keys)
        dh_ox_str = ", ".join(f"{n}={dh_avgs[f'avg_{n}_ox']:.2f}" for n in dh_method_keys)
        print(f"  Réduction:")
        print(f"    Réel={avg_real_red:.2f}, LVC={avg_lvc_red:.2f}, LVC2={avg_lvc2_red:.2f}, Dusch={avg_dusch_red:.2f}, Morse={avg_morse_red:.2f}, VPT2={avg_vpt2_red:.2f} kJ/mol")
        print(f"    {dh_red_str} kJ/mol")
        print(f"  Oxydation:")
        print(f"    Réel={avg_real_ox:.2f}, LVC={avg_lvc_ox:.2f}, LVC2={avg_lvc2_ox:.2f}, Dusch={avg_dusch_ox:.2f}, Morse={avg_morse_ox:.2f}, VPT2={avg_vpt2_ox:.2f} kJ/mol")
        print(f"    {dh_ox_str} kJ/mol")

        # Calculer les RMSD pour chaque conformère
        rmsd_data = compute_rmsd_for_molecule(workdir, mol_name)

        all_results[mol_name] = {
            'conformers': lvc_results['conformers'],
            'duschinsky_conformers': dusch_results['conformers'],
            'qvc_conformers': qvc_results['conformers'],
            'vpt2_conformers': vpt2_results['conformers'],
            'dh_conformers': dh_results['conformers'],
            'avg_lvc_red': avg_lvc_red,
            'avg_real_red': avg_real_red,
            'avg_lvc_ox': avg_lvc_ox,
            'avg_real_ox': avg_real_ox,
            'avg_disp_red': avg_disp_red,
            'avg_disp_ox': avg_disp_ox,
            'avg_dusch_red': avg_dusch_red,
            'avg_dusch_ox': avg_dusch_ox,
            'avg_morse_red': avg_morse_red,
            'avg_morse_ox': avg_morse_ox,
            'avg_qvc_red': avg_qvc_red,
            'avg_qvc_ox': avg_qvc_ox,
            'avg_vpt2_red': avg_vpt2_red,
            'avg_vpt2_ox': avg_vpt2_ox,
            'avg_lvc2_red': avg_lvc2_red,
            'avg_lvc2_ox': avg_lvc2_ox,
            'weights': weights,
            'rmsd_data': rmsd_data,
            **dh_avgs,
        }

    # Générer les rapports
    print_section("Génération des rapports")

    # Rapport LVC original
    plot_data_lvc = generate_comparison_report(all_results, output_dir)

    # Rapport comparatif de toutes les méthodes
    plot_data_all = generate_all_methods_comparison(all_results, output_dir)

    # Générer les graphiques
    if not args.no_plots:
        if plot_data_lvc['molecules']:
            generate_plots(plot_data_lvc, output_dir)
        if plot_data_all['molecules']:
            generate_all_methods_plots(plot_data_all, output_dir)

    # Export CSV si pandas disponible
    if HAS_PANDAS and plot_data_all['molecules']:
        csv_data = {
            'molecule': plot_data_all['molecules'],
            'E_def_real_red': plot_data_all['real_red'],
            'E_def_LVC_red': plot_data_all['lvc_red'],
            'E_def_LVC2_red': plot_data_all['lvc2_red'],
            'E_def_Disp_red': plot_data_all['disp_red'],
            'E_def_Dusch_red': plot_data_all['dusch_red'],
            'E_def_Morse_red': plot_data_all['morse_red'],
            'E_def_VPT2_red': plot_data_all['vpt2_red'],
            'E_def_QVC_red': plot_data_all['qvc_red'],
        }
        for dh_name in ['DHI', 'MCRC', 'GHC', 'ODHD', 'ESSC']:
            csv_data[f'E_def_{dh_name}_red'] = plot_data_all[f'{dh_name.lower()}_red']
        csv_data.update({
            'E_def_real_ox': plot_data_all['real_ox'],
            'E_def_LVC_ox': plot_data_all['lvc_ox'],
            'E_def_LVC2_ox': plot_data_all['lvc2_ox'],
            'E_def_Disp_ox': plot_data_all['disp_ox'],
            'E_def_Dusch_ox': plot_data_all['dusch_ox'],
            'E_def_Morse_ox': plot_data_all['morse_ox'],
            'E_def_VPT2_ox': plot_data_all['vpt2_ox'],
            'E_def_QVC_ox': plot_data_all['qvc_ox'],
        })
        for dh_name in ['DHI', 'MCRC', 'GHC', 'ODHD', 'ESSC']:
            csv_data[f'E_def_{dh_name}_ox'] = plot_data_all[f'{dh_name.lower()}_ox']
        csv_data.update({
            'RMSD_red': plot_data_all['rmsd_red'],
            'RMSD_ox': plot_data_all['rmsd_ox'],
        })
        df = pd.DataFrame(csv_data)
        csv_path = output_dir / "all_methods_comparison_data.csv"
        df.to_csv(csv_path, index=False)
        print_success(f"Données exportées: {csv_path}")

    print_header("ANALYSE TERMINÉE")
    print(f"Résultats dans: {output_dir}")


if __name__ == '__main__':
    main()
