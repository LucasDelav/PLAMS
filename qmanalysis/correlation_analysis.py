"""
Analyse des corrélations entre descripteurs de fréquences et propriétés thermodynamiques.

Ce module charge les données Edef/ΔΔU/TΔS depuis les résultats redox, calcule les moyennes
de Boltzmann des descripteurs, et analyse les corrélations focalisées :
- ΔΔU ↔ descripteurs haute fréquence (>2000 cm⁻¹)
- TΔS ↔ descripteurs basse fréquence (0-500 cm⁻¹)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .common import (
    print_header, print_section, print_success, print_warning, print_error,
    boltzmann_weights, hartree_to_kj,
    HIGH_FREQ_DESCRIPTORS, LOW_FREQ_DESCRIPTORS
)
from .frequency_extraction import find_neutral_rkf_files, extract_all_frequency_data
from .frequency_descriptors import compute_frequency_descriptors, get_descriptor_names


def load_edef_data(workdir, molecule_name: str) -> Dict[int, Dict[str, float]]:
    """
    Charge les données Edef depuis les résultats d'analyse redox.

    Les données sont lues depuis:
    <workdir>/redox/results/<mol>_redox_conformers.csv

    Args:
        workdir (str or Path): Répertoire de travail principal
        molecule_name (str): Nom de la molécule

    Returns:
        Dict[int, Dict]: {conf_num: {'Edef_red_kJ': float, 'Edef_ox_kJ': float, 'weight': float}}
    """
    workdir = Path(workdir)
    edef_data = {}

    # Fichier CSV des conformères redox
    conformers_csv = workdir / "redox" / "results" / f"{molecule_name}_redox_conformers.csv"

    if not conformers_csv.exists():
        print_warning(f"Fichier conformères non trouvé: {conformers_csv}")
        return edef_data

    try:
        if HAS_PANDAS:
            df = pd.read_csv(conformers_csv)
            for _, row in df.iterrows():
                conf_num = int(row['conformer'])
                edef_data[conf_num] = {
                    'Edef_red_kJ': float(row.get('Edef_red_kJ', np.nan)),
                    'Edef_ox_kJ': float(row.get('Edef_ox_kJ', np.nan)),
                    'weight': float(row.get('weight', 0.0)),
                }
        else:
            # Lecture manuelle sans pandas
            with open(conformers_csv, 'r') as f:
                header = f.readline().strip().split(',')
                conf_idx = header.index('conformer')
                weight_idx = header.index('weight') if 'weight' in header else -1
                edef_red_idx = header.index('Edef_red_kJ') if 'Edef_red_kJ' in header else -1
                edef_ox_idx = header.index('Edef_ox_kJ') if 'Edef_ox_kJ' in header else -1

                for line in f:
                    parts = line.strip().split(',')
                    conf_num = int(parts[conf_idx])
                    edef_data[conf_num] = {
                        'Edef_red_kJ': float(parts[edef_red_idx]) if edef_red_idx >= 0 else np.nan,
                        'Edef_ox_kJ': float(parts[edef_ox_idx]) if edef_ox_idx >= 0 else np.nan,
                        'weight': float(parts[weight_idx]) if weight_idx >= 0 else 0.0,
                    }

        print_success(f"Données Edef chargées: {len(edef_data)} conformères")

    except Exception as e:
        print_error(f"Erreur lecture {conformers_csv}: {e}")

    return edef_data


def load_redox_thermodynamic_data(
    workdir,
    molecule_name: str
) -> Dict[int, Dict[str, float]]:
    """
    Charge ΔΔU, TΔS, Edef et poids depuis le CSV conformères redox.

    Tente d'abord de lire les colonnes ddU_red_kJ, TdS_red_kJ, etc. depuis le CSV.
    Si ces colonnes n'existent pas (anciens résultats), utilise un fallback qui parse
    le fichier _redox_analysis.txt.

    Args:
        workdir (str or Path): Répertoire de travail principal
        molecule_name (str): Nom de la molécule

    Returns:
        Dict[int, Dict]: {conf_num: {
            'Edef_red_kJ', 'Edef_ox_kJ',
            'ddU_red_kJ', 'ddU_ox_kJ',
            'TdS_red_kJ', 'TdS_ox_kJ',
            'weight'
        }}
    """
    workdir = Path(workdir)
    thermo_data = {}

    conformers_csv = workdir / "redox" / "results" / f"{molecule_name}_redox_conformers.csv"

    if not conformers_csv.exists():
        print_warning(f"Fichier conformères non trouvé: {conformers_csv}")
        return thermo_data

    # Liste des colonnes thermodynamiques attendues
    thermo_cols = ['ddU_red_kJ', 'TdS_red_kJ', 'ddU_ox_kJ', 'TdS_ox_kJ']

    try:
        if HAS_PANDAS:
            df = pd.read_csv(conformers_csv)
            has_thermo = all(col in df.columns for col in thermo_cols)

            for _, row in df.iterrows():
                conf_num = int(row['conformer'])
                entry = {
                    'Edef_red_kJ': float(row.get('Edef_red_kJ', np.nan)),
                    'Edef_ox_kJ': float(row.get('Edef_ox_kJ', np.nan)),
                    'weight': float(row.get('weight', 0.0)),
                }
                if has_thermo:
                    for col in thermo_cols:
                        entry[col] = float(row.get(col, np.nan))
                else:
                    for col in thermo_cols:
                        entry[col] = np.nan
                thermo_data[conf_num] = entry
        else:
            with open(conformers_csv, 'r') as f:
                header = f.readline().strip().split(',')
                col_indices = {}
                for col in ['conformer', 'weight', 'Edef_red_kJ', 'Edef_ox_kJ'] + thermo_cols:
                    col_indices[col] = header.index(col) if col in header else -1

                has_thermo = all(col_indices[col] >= 0 for col in thermo_cols)

                for line in f:
                    parts = line.strip().split(',')
                    conf_num = int(parts[col_indices['conformer']])
                    entry = {
                        'Edef_red_kJ': float(parts[col_indices['Edef_red_kJ']]) if col_indices['Edef_red_kJ'] >= 0 else np.nan,
                        'Edef_ox_kJ': float(parts[col_indices['Edef_ox_kJ']]) if col_indices['Edef_ox_kJ'] >= 0 else np.nan,
                        'weight': float(parts[col_indices['weight']]) if col_indices['weight'] >= 0 else 0.0,
                    }
                    if has_thermo:
                        for col in thermo_cols:
                            idx = col_indices[col]
                            entry[col] = float(parts[idx]) if idx >= 0 and parts[idx] else np.nan
                    else:
                        for col in thermo_cols:
                            entry[col] = np.nan
                    thermo_data[conf_num] = entry

        if has_thermo:
            print_success(f"Données thermodynamiques (ΔΔU/TΔS) chargées depuis CSV: {len(thermo_data)} conformères")
        else:
            print_warning("Colonnes ΔΔU/TΔS absentes du CSV, tentative de fallback depuis _redox_analysis.txt")
            thermo_data = _fallback_parse_thermo_from_txt(workdir, molecule_name, thermo_data)

    except Exception as e:
        print_error(f"Erreur lecture {conformers_csv}: {e}")

    return thermo_data


def _fallback_parse_thermo_from_txt(
    workdir: Path,
    molecule_name: str,
    existing_data: Dict[int, Dict[str, float]]
) -> Dict[int, Dict[str, float]]:
    """
    Parse ΔΔU et TΔS depuis le fichier _redox_analysis.txt (fallback pour anciens résultats).

    Args:
        workdir: Répertoire de travail
        molecule_name: Nom de la molécule
        existing_data: Données déjà chargées depuis CSV (Edef, poids)

    Returns:
        Dict mis à jour avec ddU et TdS
    """
    analysis_file = workdir / "redox" / "results" / f"{molecule_name}_redox_analysis.txt"

    if not analysis_file.exists():
        print_warning(f"Fichier d'analyse non trouvé: {analysis_file}")
        return existing_data

    current_conf = None
    current_section = None

    with open(analysis_file, 'r') as f:
        for line in f:
            # Détecter le conformère
            if 'Conformère' in line and 'poids' in line:
                try:
                    conf_match = re.search(r'Conformère\s+(\d+)', line)
                    if conf_match:
                        current_conf = int(conf_match.group(1))
                except (ValueError, AttributeError):
                    current_conf = None

            elif 'RÉDUCTION:' in line or 'REDUCTION:' in line:
                current_section = 'red'
            elif 'OXYDATION:' in line or 'OXIDATION:' in line:
                current_section = 'ox'

            elif current_conf is not None and current_section is not None:
                # Parser ΔΔU
                if 'ΔΔU' in line and '=' in line and 'kJ/mol' in line:
                    try:
                        val = float(line.split('=')[1].strip().split()[0])
                        if current_conf not in existing_data:
                            existing_data[current_conf] = {'weight': 0.0}
                        key = f'ddU_{current_section}_kJ'
                        existing_data[current_conf][key] = val
                    except (ValueError, IndexError):
                        pass

                # Parser TΔS
                elif 'TΔS' in line and '=' in line and 'kJ/mol' in line:
                    try:
                        val = float(line.split('=')[1].strip().split()[0])
                        if current_conf not in existing_data:
                            existing_data[current_conf] = {'weight': 0.0}
                        key = f'TdS_{current_section}_kJ'
                        existing_data[current_conf][key] = val
                    except (ValueError, IndexError):
                        pass

    # Vérifier que les données ont bien été parsées
    has_any = any(
        'ddU_red_kJ' in data or 'TdS_red_kJ' in data
        for data in existing_data.values()
    )
    if has_any:
        print_success(f"ΔΔU/TΔS parsés depuis {analysis_file.name}")
    else:
        print_warning("Aucune donnée ΔΔU/TΔS trouvée dans le fichier d'analyse")

    return existing_data


def _read_conformer_weights_from_summary(workdir) -> Optional[Dict[int, float]]:
    """
    Lit les poids de Boltzmann depuis le fichier conformers_summary.txt.

    C'est une copie locale de la fonction de qmredox/analysis.py pour éviter
    les dépendances circulaires.

    Args:
        workdir (Path or str): Répertoire de travail principal

    Returns:
        dict: {conf_num: weight} ou None si le fichier n'existe pas
    """
    workdir = Path(workdir)
    summary_file = workdir / "conformers" / "results" / "conformers_summary.txt"

    if not summary_file.exists():
        return None

    weights = {}
    in_conformers_section = False

    with open(summary_file, 'r') as f:
        for line in f:
            if "CONFORMÈRES FINAUX" in line:
                in_conformers_section = True
                continue

            if in_conformers_section:
                stripped = line.strip()

                # Fin de section
                if stripped.startswith("Fichiers XYZ") or stripped.startswith("Note:"):
                    break

                # Ignorer les séparateurs
                if not stripped or stripped.startswith("===") or stripped.startswith("---"):
                    continue
                if "#" in stripped and "ΔE" in stripped:
                    continue

                # Parser la ligne de données
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        conf_idx = int(parts[0])
                        weight = float(parts[2])
                        weights[conf_idx] = weight
                    except (ValueError, IndexError):
                        continue

    return weights if weights else None


def compute_boltzmann_averaged_descriptors(
    descriptors_by_conf: Dict[int, Dict[str, float]],
    weights: Dict[int, float]
) -> Dict[str, float]:
    """
    Calcule les descripteurs moyennés par les poids de Boltzmann.

    Args:
        descriptors_by_conf (Dict): {conf_num: {descriptor_name: value}}
        weights (Dict): {conf_num: weight}

    Returns:
        Dict[str, float]: Descripteurs moyennés

    Example:
        >>> descriptors = {1: {'zpe_kj': 100.0}, 2: {'zpe_kj': 105.0}}
        >>> weights = {1: 0.7, 2: 0.3}
        >>> avg = compute_boltzmann_averaged_descriptors(descriptors, weights)
        >>> print(f"ZPE moyenne: {avg['zpe_kj']:.2f}")  # 101.5 kJ/mol
    """
    if not descriptors_by_conf or not weights:
        return {}

    # Obtenir les noms de descripteurs
    first_conf = next(iter(descriptors_by_conf.values()))
    descriptor_names = list(first_conf.keys())

    # Initialiser les accumulateurs
    averaged = {name: 0.0 for name in descriptor_names}
    total_weight = 0.0

    # Calculer les moyennes pondérées
    for conf_num, descriptors in descriptors_by_conf.items():
        weight = weights.get(conf_num, 0.0)
        if weight <= 0:
            continue

        total_weight += weight
        for name in descriptor_names:
            value = descriptors.get(name, np.nan)
            if not np.isnan(value):
                averaged[name] += weight * value

    # Normaliser
    if total_weight > 0:
        for name in descriptor_names:
            averaged[name] /= total_weight

    return averaged


def compute_correlations(
    descriptors: Dict[str, Dict[str, float]],
    edef_values: Dict[str, Dict[str, float]],
    molecule_names: List[str]
) -> Tuple[Any, Any]:
    """
    Calcule les corrélations entre descripteurs de fréquences et Edef.

    Args:
        descriptors (Dict): {molecule_name: {descriptor: value}}
        edef_values (Dict): {molecule_name: {'Edef_red_kJ': float, 'Edef_ox_kJ': float}}
        molecule_names (List): Liste des noms de molécules

    Returns:
        Tuple[DataFrame, DataFrame] si pandas disponible:
            - df_correlations: Corrélations triées par R²
            - df_descriptors: Descripteurs par molécule
        Ou Tuple[Dict, Dict] sinon

    Note:
        Les corrélations sont calculées pour Edef_red_kJ et Edef_ox_kJ séparément.
    """
    from scipy.stats import pearsonr

    # Construire les tableaux de données
    descriptor_names = get_descriptor_names()

    # Matrices de données
    X = {name: [] for name in descriptor_names}  # Descripteurs
    Y_red = []  # Edef réduction
    Y_ox = []   # Edef oxydation

    for mol_name in molecule_names:
        mol_desc = descriptors.get(mol_name, {})
        mol_edef = edef_values.get(mol_name, {})

        for name in descriptor_names:
            X[name].append(mol_desc.get(name, np.nan))

        Y_red.append(mol_edef.get('Edef_red_kJ', np.nan))
        Y_ox.append(mol_edef.get('Edef_ox_kJ', np.nan))

    # Calculer les corrélations
    correlations = []

    for desc_name in descriptor_names:
        x = np.array(X[desc_name])

        # Corrélation avec Edef réduction
        y_red = np.array(Y_red)
        mask_red = ~np.isnan(x) & ~np.isnan(y_red)
        if np.sum(mask_red) >= 3:
            r_red, p_red = pearsonr(x[mask_red], y_red[mask_red])
            r2_red = r_red ** 2
        else:
            r_red, r2_red, p_red = np.nan, np.nan, np.nan

        # Corrélation avec Edef oxydation
        y_ox = np.array(Y_ox)
        mask_ox = ~np.isnan(x) & ~np.isnan(y_ox)
        if np.sum(mask_ox) >= 3:
            r_ox, p_ox = pearsonr(x[mask_ox], y_ox[mask_ox])
            r2_ox = r_ox ** 2
        else:
            r_ox, r2_ox, p_ox = np.nan, np.nan, np.nan

        correlations.append({
            'descriptor': desc_name,
            'r_reduction': r_red,
            'r2_reduction': r2_red,
            'p_reduction': p_red,
            'n_reduction': int(np.sum(mask_red)),
            'r_oxidation': r_ox,
            'r2_oxidation': r2_ox,
            'p_oxidation': p_ox,
            'n_oxidation': int(np.sum(mask_ox)),
        })

    # Construire le DataFrame des descripteurs
    descriptor_data = {'molecule': molecule_names}
    for name in descriptor_names:
        descriptor_data[name] = X[name]
    descriptor_data['Edef_red_kJ'] = Y_red
    descriptor_data['Edef_ox_kJ'] = Y_ox

    if HAS_PANDAS:
        df_corr = pd.DataFrame(correlations)
        # Trier par R² max (réduction ou oxydation)
        df_corr['r2_max'] = df_corr[['r2_reduction', 'r2_oxidation']].max(axis=1)
        df_corr = df_corr.sort_values('r2_max', ascending=False)

        df_desc = pd.DataFrame(descriptor_data)

        return df_corr, df_desc
    else:
        # Trier par R² max
        correlations.sort(key=lambda x: max(
            x['r2_reduction'] if not np.isnan(x['r2_reduction']) else 0,
            x['r2_oxidation'] if not np.isnan(x['r2_oxidation']) else 0
        ), reverse=True)

        return correlations, descriptor_data


def _compute_single_correlation_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Calcule les statistiques de corrélation complètes pour une paire (x, y).

    Tests statistiques :
    - Pearson r + p-value
    - Spearman rho + p-value
    - Kendall tau + p-value
    - R² et R² ajusté
    - Bootstrap 95% CI sur R² (1000 itérations)
    - R² LOOCV (leave-one-out cross-validation)
    - Shapiro-Wilk sur les résidus (test de normalité)

    Args:
        x: Tableau des valeurs du descripteur (sans NaN)
        y: Tableau des valeurs cibles (sans NaN)

    Returns:
        Dict avec toutes les statistiques
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau, shapiro, linregress

    n = len(x)
    result = {
        'n': n,
        'pearson_r': np.nan, 'pearson_p': np.nan,
        'spearman_rho': np.nan, 'spearman_p': np.nan,
        'kendall_tau': np.nan, 'kendall_p': np.nan,
        'r2': np.nan, 'r2_adj': np.nan,
        'r2_loocv': np.nan,
        'r2_bootstrap_lo': np.nan, 'r2_bootstrap_hi': np.nan,
        'shapiro_w': np.nan, 'shapiro_p': np.nan,
        'slope': np.nan, 'intercept': np.nan,
    }

    if n < 3:
        return result

    # Pearson
    r_pearson, p_pearson = pearsonr(x, y)
    result['pearson_r'] = float(r_pearson)
    result['pearson_p'] = float(p_pearson)

    # R² et R² ajusté
    r2 = r_pearson ** 2
    result['r2'] = float(r2)
    result['r2_adj'] = float(1.0 - (1.0 - r2) * (n - 1) / (n - 2)) if n > 2 else np.nan

    # Spearman
    rho, p_spearman = spearmanr(x, y)
    result['spearman_rho'] = float(rho)
    result['spearman_p'] = float(p_spearman)

    # Kendall
    tau, p_kendall = kendalltau(x, y)
    result['kendall_tau'] = float(tau)
    result['kendall_p'] = float(p_kendall)

    # Régression linéaire
    slope, intercept, _, _, _ = linregress(x, y)
    result['slope'] = float(slope)
    result['intercept'] = float(intercept)

    # Résidus et Shapiro-Wilk
    y_pred = slope * x + intercept
    residuals = y - y_pred
    if n >= 3:
        w_stat, p_shapiro = shapiro(residuals)
        result['shapiro_w'] = float(w_stat)
        result['shapiro_p'] = float(p_shapiro)

    # R² LOOCV (leave-one-out)
    if n >= 4:
        ss_loocv = 0.0
        ss_total = 0.0
        y_mean = np.mean(y)
        for i in range(n):
            x_train = np.delete(x, i)
            y_train = np.delete(y, i)
            s, ic, _, _, _ = linregress(x_train, y_train)
            y_pred_i = s * x[i] + ic
            ss_loocv += (y[i] - y_pred_i) ** 2
            ss_total += (y[i] - y_mean) ** 2
        result['r2_loocv'] = float(1.0 - ss_loocv / ss_total) if ss_total > 0 else np.nan

    # Bootstrap 95% CI sur R²
    if n >= 5:
        rng = np.random.default_rng(42)
        n_bootstrap = 1000
        r2_boot = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            x_b = x[idx]
            y_b = y[idx]
            # Vérifier qu'il y a de la variance
            if np.std(x_b) > 0 and np.std(y_b) > 0:
                r_b, _ = pearsonr(x_b, y_b)
                r2_boot[b] = r_b ** 2
            else:
                r2_boot[b] = np.nan
        valid_boot = r2_boot[~np.isnan(r2_boot)]
        if len(valid_boot) > 0:
            result['r2_bootstrap_lo'] = float(np.percentile(valid_boot, 2.5))
            result['r2_bootstrap_hi'] = float(np.percentile(valid_boot, 97.5))

    return result


def compute_focused_correlations(
    descriptors: Dict[str, Dict[str, float]],
    target_values: Dict[str, Dict[str, float]],
    molecule_names: List[str],
    target_descriptor_map: Optional[Dict[str, List[str]]] = None
) -> Any:
    """
    Calcule les corrélations focalisées entre descripteurs ciblés et cibles thermodynamiques.

    Par défaut, utilise le ciblage physique :
    - Cibles ddU_* → HIGH_FREQ_DESCRIPTORS
    - Cibles TdS_* → LOW_FREQ_DESCRIPTORS
    - Cibles Edef_* → tous les descripteurs

    Chaque paire (descripteur, cible) est analysée avec des tests statistiques extensifs.

    Args:
        descriptors: {molecule_name: {descriptor: value}}
        target_values: {molecule_name: {'ddU_red_kJ': float, 'TdS_red_kJ': float, ...}}
        molecule_names: Liste des noms de molécules
        target_descriptor_map: Dict optionnel {target_name: [descriptors]}.
                               Si None, utilise le ciblage physique par défaut.

    Returns:
        DataFrame (si pandas) ou list de dicts avec les corrélations complètes
    """
    # Déterminer les cibles disponibles
    all_targets = set()
    for mol_data in target_values.values():
        all_targets.update(mol_data.keys())

    # Ciblage physique par défaut
    if target_descriptor_map is None:
        all_desc_names = get_descriptor_names()
        target_descriptor_map = {}
        for target in all_targets:
            if 'ddU' in target:
                target_descriptor_map[target] = HIGH_FREQ_DESCRIPTORS
            elif 'TdS' in target:
                target_descriptor_map[target] = LOW_FREQ_DESCRIPTORS
            elif 'Edef' in target:
                target_descriptor_map[target] = all_desc_names

    # Construire les vecteurs de données
    X = {}
    for mol_name in molecule_names:
        mol_desc = descriptors.get(mol_name, {})
        for name, val in mol_desc.items():
            if name not in X:
                X[name] = []
            X[name].append(val)

    Y = {}
    for target in all_targets:
        Y[target] = []
        for mol_name in molecule_names:
            Y[target].append(target_values.get(mol_name, {}).get(target, np.nan))

    # Calculer les corrélations focalisées
    correlations = []

    for target_name, desc_list in target_descriptor_map.items():
        y = np.array(Y.get(target_name, []))

        for desc_name in desc_list:
            if desc_name not in X:
                continue

            x = np.array(X[desc_name])

            # Filtrer les NaN
            mask = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(mask) < 3:
                continue

            x_valid = x[mask]
            y_valid = y[mask]

            # Calculer les stats
            stats = _compute_single_correlation_stats(x_valid, y_valid)
            stats['descriptor'] = desc_name
            stats['target'] = target_name
            correlations.append(stats)

    if HAS_PANDAS:
        df = pd.DataFrame(correlations)
        if not df.empty:
            df = df.sort_values('r2', ascending=False)
        return df
    else:
        correlations.sort(key=lambda x: x.get('r2', 0) if not np.isnan(x.get('r2', 0)) else 0, reverse=True)
        return correlations


def run_full_analysis(
    workdirs: List[Path],
    output_dir: Path
) -> Tuple[Any, Any]:
    """
    Exécute l'analyse complète fréquence-Edef sur plusieurs molécules.

    Inclut désormais les corrélations focalisées ΔΔU/TΔS en plus des corrélations Edef.

    Args:
        workdirs (List[Path]): Liste des répertoires de travail des molécules
        output_dir (Path): Répertoire de sortie pour les résultats

    Returns:
        Tuple[df_correlations, df_descriptors] si pandas disponible

    Process:
        1. Pour chaque molécule:
           a. Trouver les fichiers RKF neutres
           b. Extraire les données de fréquences
           c. Calculer les descripteurs
           d. Charger les données Edef/ΔΔU/TΔS et poids de Boltzmann
           e. Calculer les descripteurs moyennés
        2. Calculer les corrélations globales (Edef)
        3. Calculer les corrélations focalisées (ΔΔU/TΔS)
        4. Exporter les résultats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("ANALYSE FRÉQUENCE-EDEF / ΔΔU / TΔS")

    # Collecter les données de toutes les molécules
    all_descriptors = {}        # {mol_name: {descriptor: value}}
    all_edef = {}              # {mol_name: {'Edef_red_kJ': float, 'Edef_ox_kJ': float}}
    all_thermo = {}            # {mol_name: {'ddU_red_kJ': ..., 'TdS_red_kJ': ..., etc.}}
    molecule_names = []

    for workdir in workdirs:
        workdir = Path(workdir)

        # Extraire le nom de molécule depuis le nom du workdir
        mol_name = workdir.name.replace('_workdir', '')
        molecule_names.append(mol_name)

        print_section(f"Traitement de {mol_name}")

        # 1. Trouver les fichiers RKF neutres
        rkf_files = find_neutral_rkf_files(workdir, mol_name)

        if not rkf_files:
            print_warning(f"Aucun fichier RKF neutre trouvé pour {mol_name}")
            all_descriptors[mol_name] = {}
            all_edef[mol_name] = {'Edef_red_kJ': np.nan, 'Edef_ox_kJ': np.nan}
            all_thermo[mol_name] = {
                'ddU_red_kJ': np.nan, 'ddU_ox_kJ': np.nan,
                'TdS_red_kJ': np.nan, 'TdS_ox_kJ': np.nan,
                'Edef_red_kJ': np.nan, 'Edef_ox_kJ': np.nan,
            }
            continue

        print(f"  {len(rkf_files)} fichiers RKF neutres trouvés")

        # 2. Extraire les données de fréquences pour chaque conformère
        descriptors_by_conf = {}
        for conf_num, rkf_path in sorted(rkf_files.items()):
            freq_data = extract_all_frequency_data(rkf_path)
            if freq_data['success']:
                desc = compute_frequency_descriptors(freq_data)
                descriptors_by_conf[conf_num] = desc

        print(f"  {len(descriptors_by_conf)} conformères analysés avec succès")

        # 3. Charger les poids de Boltzmann
        weights = _read_conformer_weights_from_summary(workdir)

        if weights is None:
            # Fallback: charger depuis les données redox
            thermo_data = load_redox_thermodynamic_data(workdir, mol_name)
            weights = {conf_num: data.get('weight', 0.0)
                      for conf_num, data in thermo_data.items()}
        else:
            thermo_data = load_redox_thermodynamic_data(workdir, mol_name)
            print_success(f"Poids de Boltzmann lus depuis conformers_summary.txt")

        # 4. Calculer les descripteurs moyennés
        avg_descriptors = compute_boltzmann_averaged_descriptors(descriptors_by_conf, weights)
        all_descriptors[mol_name] = avg_descriptors

        # 5. Charger les données thermodynamiques moyennées (pondérées)
        if thermo_data:
            total_weight = sum(weights.values())
            if total_weight > 0:
                avg_edef_red = sum(weights.get(conf, 0) * data.get('Edef_red_kJ', 0)
                                   for conf, data in thermo_data.items()) / total_weight
                avg_edef_ox = sum(weights.get(conf, 0) * data.get('Edef_ox_kJ', 0)
                                  for conf, data in thermo_data.items()) / total_weight
                all_edef[mol_name] = {
                    'Edef_red_kJ': avg_edef_red,
                    'Edef_ox_kJ': avg_edef_ox,
                }

                # Moyenner ΔΔU et TΔS
                thermo_keys = ['ddU_red_kJ', 'ddU_ox_kJ', 'TdS_red_kJ', 'TdS_ox_kJ']
                mol_thermo = {'Edef_red_kJ': avg_edef_red, 'Edef_ox_kJ': avg_edef_ox}
                for key in thermo_keys:
                    vals_weights = [
                        (data.get(key, np.nan), weights.get(conf, 0))
                        for conf, data in thermo_data.items()
                    ]
                    valid = [(v, w) for v, w in vals_weights if not np.isnan(v) and w > 0]
                    if valid:
                        w_sum = sum(w for _, w in valid)
                        mol_thermo[key] = sum(v * w for v, w in valid) / w_sum
                    else:
                        mol_thermo[key] = np.nan
                all_thermo[mol_name] = mol_thermo
            else:
                all_edef[mol_name] = {'Edef_red_kJ': np.nan, 'Edef_ox_kJ': np.nan}
                all_thermo[mol_name] = {k: np.nan for k in
                    ['ddU_red_kJ', 'ddU_ox_kJ', 'TdS_red_kJ', 'TdS_ox_kJ',
                     'Edef_red_kJ', 'Edef_ox_kJ']}
        else:
            all_edef[mol_name] = {'Edef_red_kJ': np.nan, 'Edef_ox_kJ': np.nan}
            all_thermo[mol_name] = {k: np.nan for k in
                ['ddU_red_kJ', 'ddU_ox_kJ', 'TdS_red_kJ', 'TdS_ox_kJ',
                 'Edef_red_kJ', 'Edef_ox_kJ']}

        # Afficher un aperçu
        if avg_descriptors:
            print(f"  ZPE moyen: {avg_descriptors.get('zpe_kj', np.nan):.2f} kJ/mol")
            print(f"  Flexibility index: {avg_descriptors.get('flexibility_index', np.nan):.3f}")
        print(f"  Edef(red): {all_edef[mol_name]['Edef_red_kJ']:.2f} kJ/mol")
        print(f"  Edef(ox): {all_edef[mol_name]['Edef_ox_kJ']:.2f} kJ/mol")
        ddU_red = all_thermo[mol_name].get('ddU_red_kJ', np.nan)
        TdS_red = all_thermo[mol_name].get('TdS_red_kJ', np.nan)
        if not np.isnan(ddU_red):
            print(f"  ΔΔU(red): {ddU_red:.2f} kJ/mol")
        if not np.isnan(TdS_red):
            print(f"  TΔS(red): {TdS_red:.2f} kJ/mol")

    # 6. Calculer les corrélations Edef (rétro-compatibilité)
    print_section("Calcul des corrélations Edef")

    df_corr, df_desc = compute_correlations(all_descriptors, all_edef, molecule_names)

    # 7. Calculer les corrélations focalisées ΔΔU/TΔS
    print_section("Calcul des corrélations focalisées ΔΔU/TΔS")

    df_focused = compute_focused_correlations(
        all_descriptors, all_thermo, molecule_names
    )

    # 8. Exporter les résultats
    print_section("Export des résultats")

    if HAS_PANDAS:
        # Export CSV des corrélations Edef
        corr_csv = output_dir / "frequency_correlations.csv"
        df_corr.to_csv(corr_csv, index=False)
        print_success(f"Corrélations Edef exportées: {corr_csv}")

        # Export CSV des descripteurs
        desc_csv = output_dir / "frequency_descriptors.csv"
        df_desc.to_csv(desc_csv, index=False)
        print_success(f"Descripteurs exportés: {desc_csv}")

        # Export CSV des corrélations focalisées
        if isinstance(df_focused, pd.DataFrame) and not df_focused.empty:
            # Séparer ΔΔU et TΔS
            ddU_mask = df_focused['target'].str.contains('ddU')
            TdS_mask = df_focused['target'].str.contains('TdS')

            if ddU_mask.any():
                ddU_csv = output_dir / "focused_correlations_ddU.csv"
                df_focused[ddU_mask].to_csv(ddU_csv, index=False)
                print_success(f"Corrélations ΔΔU exportées: {ddU_csv}")

            if TdS_mask.any():
                TdS_csv = output_dir / "focused_correlations_TdS.csv"
                df_focused[TdS_mask].to_csv(TdS_csv, index=False)
                print_success(f"Corrélations TΔS exportées: {TdS_csv}")

        # Export CSV des cibles thermodynamiques
        thermo_records = []
        for mol_name in molecule_names:
            record = {'molecule': mol_name}
            record.update(all_thermo.get(mol_name, {}))
            thermo_records.append(record)
        df_thermo = pd.DataFrame(thermo_records)
        thermo_csv = output_dir / "thermodynamic_targets.csv"
        df_thermo.to_csv(thermo_csv, index=False)
        print_success(f"Cibles thermodynamiques exportées: {thermo_csv}")

        # Afficher les meilleures corrélations Edef
        print("\nMeilleures corrélations avec Edef:")
        print("-" * 60)
        for _, row in df_corr.head(10).iterrows():
            r2_red = row['r2_reduction']
            r2_ox = row['r2_oxidation']
            print(f"  {row['descriptor']:25s}  R²(red)={r2_red:.3f}  R²(ox)={r2_ox:.3f}")

        # Afficher les meilleures corrélations focalisées
        if isinstance(df_focused, pd.DataFrame) and not df_focused.empty:
            print("\nMeilleures corrélations focalisées ΔΔU/TΔS:")
            print("-" * 80)
            for _, row in df_focused.head(10).iterrows():
                print(f"  {row['descriptor']:25s} → {row['target']:15s}  "
                      f"R²={row['r2']:.3f}  Pearson p={row['pearson_p']:.3e}  "
                      f"Spearman ρ={row['spearman_rho']:.3f}")

    else:
        # Export sans pandas (format texte)
        corr_txt = output_dir / "frequency_correlations.txt"
        with open(corr_txt, 'w') as f:
            f.write("descriptor,r_reduction,r2_reduction,p_reduction,n_reduction,"
                   "r_oxidation,r2_oxidation,p_oxidation,n_oxidation\n")
            for row in df_corr:
                f.write(f"{row['descriptor']},{row['r_reduction']:.6f},{row['r2_reduction']:.6f},"
                       f"{row['p_reduction']:.6f},{row['n_reduction']},"
                       f"{row['r_oxidation']:.6f},{row['r2_oxidation']:.6f},"
                       f"{row['p_oxidation']:.6f},{row['n_oxidation']}\n")
        print_success(f"Corrélations exportées: {corr_txt}")

        print("\nMeilleures corrélations avec Edef:")
        print("-" * 60)
        for row in df_corr[:10]:
            r2_red = row['r2_reduction']
            r2_ox = row['r2_oxidation']
            print(f"  {row['descriptor']:25s}  R²(red)={r2_red:.3f}  R²(ox)={r2_ox:.3f}")

    return df_corr, df_desc
