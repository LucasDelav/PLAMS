"""
Module pour la consolidation de données multi-molécules.

Ce module collecte et consolide les données moyennées de Boltzmann pour plusieurs
molécules, permettant les comparaisons inter-molécules dans les visualisations.

Chaque molécule est représentée par une ligne dans le DataFrame final, avec :
- Propriétés redox moyennées (E_reduction, E_oxidation, EA, IP)
- Orbitales frontières moyennées (HOMO, LUMO, gap)
- Charges de Bader moyennées: C1_bader, C2_bader, ..., O7_bader, O8_bader
- Énergies IQA moyennées: C1_iqa, C2_iqa, ..., O7_iqa, O8_iqa

Note: Seuls les 8 atomes du squelette commun (C1-C6, O7-O8) sont inclus.

Usage:
    from common.multi_molecule_data import collect_all_molecules
    from pathlib import Path

    workdirs = [Path("mol1_workdir"), Path("mol2_workdir")]
    df = collect_all_molecules(workdirs)
    print(df)  # DataFrame avec 1 ligne par molécule
"""

from pathlib import Path
from typing import Dict, List, Optional
import warnings
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("Pandas non disponible - ce module ne fonctionnera pas")


def read_boltzmann_weights(workdir: Path) -> Optional[Dict[str, float]]:
    """
    Lit les poids de Boltzmann depuis conformers_summary.txt.

    Args:
        workdir: Répertoire de travail de la molécule

    Returns:
        Dictionnaire {conformer_id: weight} ou None si fichier absent

    Example:
        weights = read_boltzmann_weights(Path("DHI_workdir"))
        # {'conf_1': 0.9755, 'conf_2': 0.0245}
    """
    summary_file = workdir / "conformers" / "results" / "conformers_summary.txt"

    if not summary_file.exists():
        warnings.warn(f"Fichier conformers_summary.txt non trouvé dans {workdir}")
        return None

    weights = {}
    in_table = False

    with open(summary_file, 'r') as f:
        for line in f:
            # Détecter début de la table des conformères
            if "CONFORMÈRES FINAUX" in line or "FINAL CONFORMERS" in line:
                in_table = True
                continue

            # Sauter les séparateurs
            if line.strip().startswith("===") or line.strip().startswith("---"):
                continue

            # Parser les lignes de conformères
            if in_table and line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        conf_num = int(parts[0])
                        weight = float(parts[2])
                        weights[f"conf_{conf_num}"] = weight
                    except (ValueError, IndexError):
                        pass

    if not weights:
        warnings.warn(f"Aucun poids de Boltzmann trouvé dans {summary_file}")
        return None

    return weights


def _load_bader_for_charge_state(
    qtaim_data: Dict,
    weights: Dict[str, float],
    charge_state: str
) -> Dict[int, float]:
    """
    Charge les charges de Bader pour un état de charge spécifique.

    Args:
        qtaim_data: Données QTAIM chargées depuis JSON
        weights: Poids de Boltzmann par conformère
        charge_state: État de charge ("0", "-1", "1")

    Returns:
        Dictionnaire {atom_idx: averaged_charge} pour les atomes 1-8
    """
    # Mapping des indices canoniques (on ne garde que 1-8)
    valid_atoms = {1, 2, 3, 4, 5, 6, 7, 8}

    bader_by_atom = {}  # {atom_idx: [(value, weight), ...]}

    for conf_id, conf_data in qtaim_data.items():
        if charge_state not in conf_data:
            continue

        state_data = conf_data[charge_state]
        if "bader_charges" not in state_data:
            continue

        weight = weights.get(conf_id, 0.0)
        if weight == 0:
            continue

        for atom_idx_str, charge_value in state_data["bader_charges"].items():
            atom_idx = int(atom_idx_str)
            if atom_idx not in valid_atoms:
                continue
            if atom_idx not in bader_by_atom:
                bader_by_atom[atom_idx] = []
            bader_by_atom[atom_idx].append((charge_value, weight))

    # Moyennes pondérées
    averaged = {}
    for atom_idx, values_weights in bader_by_atom.items():
        total_weighted = sum(v * w for v, w in values_weights)
        total_weight = sum(w for _, w in values_weights)
        if total_weight > 0:
            averaged[atom_idx] = total_weighted / total_weight

    return averaged


def load_and_average_bader_charges(
    workdir: Path,
    molecule_name: str
) -> Dict[str, float]:
    """
    Moyenne les charges de Bader avec les poids de Boltzmann.

    Lit les données depuis qtaim_complete_data.json (état neutre, charge=0).
    Limité aux 8 atomes du squelette commun (C1-C6, O7-O8).

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        Dictionnaire {C1_bader: value, C2_bader: value, ..., O7_bader: value, O8_bader: value}
        Retourne {} si QTAIM non disponible

    Note:
        Utilise les poids de Boltzmann du pipeline conformères pour pondérer
        les charges de Bader de chaque conformère.
    """
    # Lire les poids de Boltzmann
    weights = read_boltzmann_weights(workdir)
    if weights is None:
        return {}

    # Chercher le fichier JSON QTAIM
    qtaim_json = workdir / "qtaim" / "results" / "qtaim_complete_data.json"

    if not qtaim_json.exists():
        warnings.warn(f"Fichier qtaim_complete_data.json non trouvé pour {molecule_name}")
        return {}

    try:
        with open(qtaim_json, 'r') as f:
            qtaim_data = json.load(f)
    except Exception as e:
        warnings.warn(f"Erreur lors de la lecture de {qtaim_json}: {e}")
        return {}

    # Mapping des indices canoniques vers labels
    atom_labels = {
        1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
        7: 'O7', 8: 'O8'
    }

    # Charger les charges de l'état neutre
    neutral_charges = _load_bader_for_charge_state(qtaim_data, weights, "0")

    # Construire le dictionnaire avec labels
    averaged_charges = {}
    for atom_idx, charge in neutral_charges.items():
        label = atom_labels[atom_idx]
        averaged_charges[f"{label}_bader"] = charge

    return averaged_charges


def compute_delta_bader(workdir: Path, molecule_name: str) -> Dict[str, float]:
    """
    Calcule les Δ-descripteurs de transfert de charge lors du redox.

    ΔBader_red(atom) = Bader(réduit) - Bader(neutre)
    ΔBader_ox(atom)  = Bader(oxydé) - Bader(neutre)

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        Dictionnaire avec les Δ-descripteurs :
        - delta_C1_red, delta_C2_red, ..., delta_O8_red (réduction)
        - delta_C1_ox, delta_C2_ox, ..., delta_O8_ox (oxydation)

    Note:
        Valeur négative = gain d'électrons (densité augmente)
        Valeur positive = perte d'électrons (densité diminue)
    """
    # Lire les poids de Boltzmann
    weights = read_boltzmann_weights(workdir)
    if weights is None:
        return {}

    # Chercher le fichier JSON QTAIM
    qtaim_json = workdir / "qtaim" / "results" / "qtaim_complete_data.json"

    if not qtaim_json.exists():
        return {}

    try:
        with open(qtaim_json, 'r') as f:
            qtaim_data = json.load(f)
    except Exception:
        return {}

    # Mapping des indices canoniques vers labels
    atom_labels = {
        1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
        7: 'O7', 8: 'O8'
    }

    # Charger les charges pour les 3 états
    neutral = _load_bader_for_charge_state(qtaim_data, weights, "0")
    reduced = _load_bader_for_charge_state(qtaim_data, weights, "-1")
    oxidized = _load_bader_for_charge_state(qtaim_data, weights, "1")

    if not neutral:
        return {}

    deltas = {}

    # Calculer les Δ pour chaque atome du squelette commun
    for atom_idx, label in atom_labels.items():
        if atom_idx in neutral:
            # Δ réduction (réduit - neutre)
            if atom_idx in reduced:
                deltas[f"delta_{label}_red"] = reduced[atom_idx] - neutral[atom_idx]

            # Δ oxydation (oxydé - neutre)
            if atom_idx in oxidized:
                deltas[f"delta_{label}_ox"] = oxidized[atom_idx] - neutral[atom_idx]

    return deltas


def _load_iqa_for_charge_state(
    qtaim_data: Dict,
    weights: Dict[str, float],
    charge_state: str
) -> Dict[int, float]:
    """
    Charge les énergies IQA pour un état de charge spécifique.

    Args:
        qtaim_data: Données QTAIM chargées depuis JSON
        weights: Poids de Boltzmann par conformère
        charge_state: État de charge ("0", "-1", "1")

    Returns:
        Dictionnaire {atom_idx: averaged_iqa_energy} pour les atomes 1-8
    """
    # Mapping des indices canoniques (on ne garde que 1-8)
    valid_atoms = {1, 2, 3, 4, 5, 6, 7, 8}

    iqa_by_atom = {}  # {atom_idx: [(value, weight), ...]}

    for conf_id, conf_data in qtaim_data.items():
        if charge_state not in conf_data:
            continue

        state_data = conf_data[charge_state]
        if "iqa_EaddIQA" not in state_data:
            continue

        weight = weights.get(conf_id, 0.0)
        if weight == 0:
            continue

        # Récupérer le reindex_map si disponible
        reindex_map = state_data.get("reindex_map", {})

        for atom_idx_str, iqa_energy in state_data["iqa_EaddIQA"].items():
            # L'atom_idx_str est l'indice AMS, on cherche l'indice canonique
            ams_idx = int(atom_idx_str)

            # Chercher l'indice canonique via le reindex_map inverse
            canonical_idx = None
            for ams_str, canon in reindex_map.items():
                if int(ams_str) == ams_idx:
                    canonical_idx = canon
                    break

            if canonical_idx is None or canonical_idx not in valid_atoms:
                continue

            if canonical_idx not in iqa_by_atom:
                iqa_by_atom[canonical_idx] = []
            iqa_by_atom[canonical_idx].append((iqa_energy, weight))

    # Moyennes pondérées
    averaged = {}
    for atom_idx, values_weights in iqa_by_atom.items():
        total_weighted = sum(v * w for v, w in values_weights)
        total_weight = sum(w for _, w in values_weights)
        if total_weight > 0:
            averaged[atom_idx] = total_weighted / total_weight

    return averaged


def compute_delta_iqa(workdir: Path, molecule_name: str) -> Dict[str, float]:
    """
    Calcule les Δ-descripteurs IQA lors du redox.

    ΔIQA_red(atom) = IQA(réduit) - IQA(neutre)
    ΔIQA_ox(atom)  = IQA(oxydé) - IQA(neutre)

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        Dictionnaire avec les Δ-descripteurs IQA :
        - delta_C1_iqa_red, delta_C2_iqa_red, ..., delta_O8_iqa_red (réduction)
        - delta_C1_iqa_ox, delta_C2_iqa_ox, ..., delta_O8_iqa_ox (oxydation)
    """
    # Lire les poids de Boltzmann
    weights = read_boltzmann_weights(workdir)
    if weights is None:
        return {}

    # Chercher le fichier JSON QTAIM
    qtaim_json = workdir / "qtaim" / "results" / "qtaim_complete_data.json"

    if not qtaim_json.exists():
        return {}

    try:
        with open(qtaim_json, 'r') as f:
            qtaim_data = json.load(f)
    except Exception:
        return {}

    # Mapping des indices canoniques vers labels
    atom_labels = {
        1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
        7: 'O7', 8: 'O8'
    }

    # Charger les énergies IQA pour les 3 états
    neutral = _load_iqa_for_charge_state(qtaim_data, weights, "0")
    reduced = _load_iqa_for_charge_state(qtaim_data, weights, "-1")
    oxidized = _load_iqa_for_charge_state(qtaim_data, weights, "1")

    if not neutral:
        return {}

    deltas = {}

    # Calculer les Δ pour chaque atome du squelette commun
    for atom_idx, label in atom_labels.items():
        if atom_idx in neutral:
            # Δ réduction (réduit - neutre)
            if atom_idx in reduced:
                deltas[f"delta_{label}_iqa_red"] = reduced[atom_idx] - neutral[atom_idx]

            # Δ oxydation (oxydé - neutre)
            if atom_idx in oxidized:
                deltas[f"delta_{label}_iqa_ox"] = oxidized[atom_idx] - neutral[atom_idx]

    return deltas


def load_redox_decomposition_data(workdir: Path, molecule_name: str) -> Dict[str, float]:
    """
    Charge les 6 termes de décomposition énergétique depuis *_redox_conformers.csv.

    Effectue une moyenne pondérée par les poids de Boltzmann.

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        Dictionnaire avec les 6 termes de décomposition :
        - dG_reduction_kJ, EA_kJ, Edef_red_kJ (réduction)
        - dG_oxidation_kJ, IP_kJ, Edef_ox_kJ (oxydation)
    """
    if not HAS_PANDAS:
        return {}

    # Chercher le fichier CSV redox conformers
    csv_file = workdir / "redox" / "results" / f"{molecule_name}_redox_conformers.csv"

    if not csv_file.exists():
        return {}

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return {}

    if df.empty:
        return {}

    # Colonnes de décomposition
    decomp_cols = [
        'dG_reduction_kJ', 'EA_kJ', 'Edef_red_kJ',
        'dG_oxidation_kJ', 'IP_kJ', 'Edef_ox_kJ'
    ]

    # Vérifier que les colonnes existent
    missing = [col for col in decomp_cols if col not in df.columns]
    if missing:
        return {}

    # Calculer les moyennes pondérées par Boltzmann
    if 'weight' in df.columns:
        weights = df['weight'].values
        total_weight = weights.sum()
        if total_weight == 0:
            return {}
    else:
        # Poids uniformes si pas de colonne weight
        weights = np.ones(len(df)) / len(df)
        total_weight = 1.0

    result = {}
    for col in decomp_cols:
        values = df[col].values
        weighted_avg = (values * weights).sum() / total_weight
        result[col] = weighted_avg

    return result


def load_and_average_iqa_energies(workdir: Path, molecule_name: str) -> Dict[str, float]:
    """
    Charge et moyenne les énergies IQA avec les poids de Boltzmann.

    Lit les données depuis qtaim_complete_data.json (état neutre, charge=0).

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        Dictionnaire {C1_iqa: energie, C2_iqa: energie, ...}
        Ex: {'C1_iqa': -37.82, 'O7_iqa': -75.36, ...}

    Note:
        Retourne un dictionnaire vide si les données IQA ne sont pas disponibles
    """
    # Lire les poids de Boltzmann
    weights = read_boltzmann_weights(workdir)
    if weights is None:
        return {}

    # Chercher le fichier JSON QTAIM
    qtaim_json = workdir / "qtaim" / "results" / "qtaim_complete_data.json"

    if not qtaim_json.exists():
        # QTAIM non disponible
        return {}

    try:
        with open(qtaim_json, 'r') as f:
            qtaim_data = json.load(f)
    except Exception:
        return {}

    # Collecter les énergies IQA pour chaque conformère (état neutre, charge=0)
    iqa_by_atom = {}  # {atom_idx: [(value, weight), ...]}

    for conf_id, conf_data in qtaim_data.items():
        # État neutre (charge = 0)
        if "0" not in conf_data:
            continue

        neutral_data = conf_data["0"]
        if "iqa_EaddIQA" not in neutral_data:
            # IQA non disponible pour ce conformère
            continue

        # Poids de Boltzmann pour ce conformère
        weight = weights.get(conf_id, 0.0)
        if weight == 0:
            continue

        # Collecter les énergies IQA
        for atom_idx_str, iqa_energy in neutral_data["iqa_EaddIQA"].items():
            atom_idx = int(atom_idx_str)
            if atom_idx not in iqa_by_atom:
                iqa_by_atom[atom_idx] = []
            iqa_by_atom[atom_idx].append((iqa_energy, weight))

    # Calculer les moyennes pondérées
    averaged_iqa = {}

    # Mapping des indices canoniques vers labels (C1-C6, O7-O8)
    # Limité aux 8 atomes du squelette commun (pas les hydrogènes)
    atom_labels = {
        1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
        7: 'O7', 8: 'O8'
    }

    for atom_idx, values_weights in iqa_by_atom.items():
        # Filtrer : uniquement les 8 atomes du squelette commun
        if atom_idx not in atom_labels:
            continue

        total_weighted = sum(v * w for v, w in values_weights)
        total_weight = sum(w for _, w in values_weights)

        if total_weight > 0:
            label = atom_labels[atom_idx]
            averaged_iqa[f"{label}_iqa"] = total_weighted / total_weight

    return averaged_iqa


def compute_derived_descriptors(
    bader_charges: Dict[str, float],
    delta_bader: Dict[str, float],
    delta_iqa: Dict[str, float]
) -> Dict[str, float]:
    """
    Calcule les descripteurs dérivés de densité électronique.

    Ces descripteurs combinent les données atomiques pour créer des indicateurs
    plus pertinents pour l'analyse structure-propriété.

    Args:
        bader_charges: Charges de Bader par atome (C1_bader, O7_bader, etc.)
        delta_bader: Δ charges Bader (delta_O7_red, delta_O8_ox, etc.)
        delta_iqa: Δ énergies IQA (delta_O7_iqa_red, etc.)

    Returns:
        Dictionnaire avec les descripteurs dérivés :
        - q_O_total: Charge totale sur les oxygènes (O7 + O8)
        - q_O_asymmetry: Asymétrie de charge |O7 - O8|
        - delta_q_O_total_red: Δ charge totale O (réduction)
        - delta_q_O_total_ox: Δ charge totale O (oxydation)
        - delta_iqa_O_total_red: Δ IQA totale O (réduction)
        - delta_iqa_O_total_ox: Δ IQA totale O (oxydation)
    """
    derived = {}

    # 1. Charge totale sur les oxygènes (état neutre)
    o7_bader = bader_charges.get('O7_bader')
    o8_bader = bader_charges.get('O8_bader')

    if o7_bader is not None and o8_bader is not None:
        derived['q_O_total'] = o7_bader + o8_bader
        derived['q_O_asymmetry'] = abs(o7_bader - o8_bader)

    # 2. Δ charge totale sur les oxygènes (réduction)
    delta_o7_red = delta_bader.get('delta_O7_red')
    delta_o8_red = delta_bader.get('delta_O8_red')

    if delta_o7_red is not None and delta_o8_red is not None:
        derived['delta_q_O_total_red'] = delta_o7_red + delta_o8_red

    # 3. Δ charge totale sur les oxygènes (oxydation)
    delta_o7_ox = delta_bader.get('delta_O7_ox')
    delta_o8_ox = delta_bader.get('delta_O8_ox')

    if delta_o7_ox is not None and delta_o8_ox is not None:
        derived['delta_q_O_total_ox'] = delta_o7_ox + delta_o8_ox

    # 4. Δ IQA totale sur les oxygènes (réduction)
    delta_o7_iqa_red = delta_iqa.get('delta_O7_iqa_red')
    delta_o8_iqa_red = delta_iqa.get('delta_O8_iqa_red')

    if delta_o7_iqa_red is not None and delta_o8_iqa_red is not None:
        derived['delta_iqa_O_total_red'] = delta_o7_iqa_red + delta_o8_iqa_red

    # 5. Δ IQA totale sur les oxygènes (oxydation)
    delta_o7_iqa_ox = delta_iqa.get('delta_O7_iqa_ox')
    delta_o8_iqa_ox = delta_iqa.get('delta_O8_iqa_ox')

    if delta_o7_iqa_ox is not None and delta_o8_iqa_ox is not None:
        derived['delta_iqa_O_total_ox'] = delta_o7_iqa_ox + delta_o8_iqa_ox

    return derived


def load_molecule_summary(workdir: Path) -> Dict:
    """
    Charge les données moyennées pour UNE molécule.

    Args:
        workdir: Répertoire de travail de la molécule (e.g. "DHI_workdir")

    Returns:
        Dictionnaire avec toutes les propriétés moyennées :
        - molecule_name: str
        - E_reduction_V, E_oxidation_V: float (potentiels redox)
        - EA_V, IP_V: float (affinité électronique, potentiel d'ionisation)
        - HOMO_eV, LUMO_eV, gap_eV: float (orbitales frontières)
        - bader_charge_1...8: float (charges de Bader moyennées)

    Raises:
        FileNotFoundError: Si redox_summary.csv n'existe pas
        ValueError: Si le fichier redox est vide ou mal formaté

    Example:
        data = load_molecule_summary(Path("DHI_workdir"))
        print(data['molecule_name'])  # "DHI"
        print(data['LUMO_eV'])        # -0.017684
    """
    if not HAS_PANDAS:
        raise ImportError("Pandas requis pour charger les données")

    workdir = Path(workdir)
    molecule_name = workdir.name.replace('_workdir', '')

    # 1. Charger les données redox (déjà moyennées par Boltzmann)
    redox_file = workdir / "redox" / "results" / f"{molecule_name}_redox_summary.csv"

    if not redox_file.exists():
        raise FileNotFoundError(f"Fichier redox_summary.csv non trouvé: {redox_file}")

    try:
        redox_df = pd.read_csv(redox_file)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture de {redox_file}: {e}")

    if redox_df.empty:
        raise ValueError(f"Fichier redox_summary.csv vide pour {molecule_name}")

    # Extraire la première ligne (une seule ligne par molécule)
    redox_row = redox_df.iloc[0]

    # 2. Charger et moyenner les charges QTAIM (optionnel)
    bader_charges = load_and_average_bader_charges(workdir, molecule_name)

    # 3. Charger et moyenner les énergies IQA (optionnel)
    iqa_energies = load_and_average_iqa_energies(workdir, molecule_name)

    # 4. Calculer les Δ-descripteurs de transfert de charge (optionnel)
    delta_bader = compute_delta_bader(workdir, molecule_name)

    # 5. Calculer les Δ-descripteurs IQA (optionnel)
    delta_iqa = compute_delta_iqa(workdir, molecule_name)

    # 6. Charger les données de décomposition redox (optionnel)
    decomp_data = load_redox_decomposition_data(workdir, molecule_name)

    # 7. Calculer les descripteurs dérivés de densité électronique
    derived_descriptors = compute_derived_descriptors(
        bader_charges, delta_bader, delta_iqa
    )

    # 8. Construire le dictionnaire de données
    data = {
        'molecule_name': molecule_name,
        'E_reduction_V': redox_row['E_reduction_V'],
        'E_oxidation_V': redox_row['E_oxidation_V'],
        'EA_V': redox_row['EA_V'],
        'IP_V': redox_row['IP_V'],
        'HOMO_eV': redox_row['HOMO_eV'],
        'LUMO_eV': redox_row['LUMO_eV'],
        'gap_eV': redox_row['gap_eV'],
    }

    # Ajouter les charges de Bader si disponibles
    data.update(bader_charges)

    # Ajouter les énergies IQA si disponibles
    data.update(iqa_energies)

    # Ajouter les Δ-descripteurs Bader si disponibles
    data.update(delta_bader)

    # Ajouter les Δ-descripteurs IQA si disponibles
    data.update(delta_iqa)

    # Ajouter les données de décomposition redox si disponibles
    data.update(decomp_data)

    # Ajouter les descripteurs dérivés si disponibles
    data.update(derived_descriptors)

    return data


def load_conformer_redox_data(
    workdir: Path,
    molecule_name: str
) -> List[Dict]:
    """
    Lit les données redox per-conformère (sans moyenner).

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        Liste de dicts, un par conformère, avec les clés :
        conformer, weight, dG_reduction_kJ, EA_kJ, Edef_red_kJ,
        dG_oxidation_kJ, IP_kJ, Edef_ox_kJ, RMSD_red_A, RMSD_ox_A
    """
    if not HAS_PANDAS:
        return []

    csv_file = workdir / "redox" / "results" / f"{molecule_name}_redox_conformers.csv"
    if not csv_file.exists():
        return []

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return []

    if df.empty:
        return []

    rows = []
    for _, row in df.iterrows():
        d = {'conformer': int(row['conformer'])}
        for col in ['weight', 'dG_reduction_kJ', 'EA_kJ', 'Edef_red_kJ',
                     'dG_oxidation_kJ', 'IP_kJ', 'Edef_ox_kJ',
                     'RMSD_red_A', 'RMSD_ox_A']:
            if col in df.columns:
                d[col] = row[col]
        rows.append(d)
    return rows


def load_conformer_orbital_data(
    workdir: Path,
    molecule_name: str
) -> Dict[int, Dict[str, float]]:
    """
    Lit les données orbitales per-conformère.

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        {conf_num: {HOMO_eV, LUMO_eV, gap_eV}}
    """
    if not HAS_PANDAS:
        return {}

    csv_file = workdir / "redox" / "results" / f"{molecule_name}_orbital_data.csv"
    if not csv_file.exists():
        return {}

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return {}

    result = {}
    for _, row in df.iterrows():
        conf_num = int(row['conformer'])
        result[conf_num] = {
            'HOMO_eV': row.get('HOMO_eV', float('nan')),
            'LUMO_eV': row.get('LUMO_eV', float('nan')),
            'gap_eV': row.get('gap_eV', float('nan')),
        }
    return result


def _load_bader_single_conformer(
    conf_data: Dict,
    charge_state: str
) -> Dict[int, float]:
    """
    Extrait les charges Bader pour UN conformère et UN état de charge.

    Args:
        conf_data: Données d'un conformère depuis le JSON QTAIM
        charge_state: État de charge ("0", "-1", "1")

    Returns:
        {canonical_atom_idx: charge} pour les atomes 1-8
    """
    valid_atoms = {1, 2, 3, 4, 5, 6, 7, 8}

    if charge_state not in conf_data:
        return {}

    state_data = conf_data[charge_state]
    if "bader_charges" not in state_data:
        return {}

    charges = {}
    for atom_idx_str, charge_value in state_data["bader_charges"].items():
        atom_idx = int(atom_idx_str)
        if atom_idx in valid_atoms:
            charges[atom_idx] = charge_value

    return charges


def load_conformer_bader_charges(
    workdir: Path,
    molecule_name: str
) -> Dict[int, Dict[str, float]]:
    """
    Lit les charges de Bader per-conformère (état neutre).

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        {conf_num: {C1_bader: val, ..., O8_bader: val}}
    """
    qtaim_json = workdir / "qtaim" / "results" / "qtaim_complete_data.json"
    if not qtaim_json.exists():
        return {}

    try:
        with open(qtaim_json, 'r') as f:
            qtaim_data = json.load(f)
    except Exception:
        return {}

    atom_labels = {
        1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
        7: 'O7', 8: 'O8'
    }

    result = {}
    for conf_id, conf_data in qtaim_data.items():
        conf_num = int(conf_id.replace('conf_', ''))
        neutral_charges = _load_bader_single_conformer(conf_data, "0")
        if neutral_charges:
            bader_dict = {}
            for atom_idx, charge in neutral_charges.items():
                label = atom_labels[atom_idx]
                bader_dict[f"{label}_bader"] = charge
            result[conf_num] = bader_dict

    return result


def _load_iqa_single_conformer(
    conf_data: Dict,
    charge_state: str
) -> Dict[int, float]:
    """
    Extrait les énergies IQA pour UN conformère et UN état de charge.

    Args:
        conf_data: Données d'un conformère depuis le JSON QTAIM
        charge_state: État de charge ("0", "-1", "1")

    Returns:
        {canonical_atom_idx: iqa_energy} pour les atomes 1-8
    """
    valid_atoms = {1, 2, 3, 4, 5, 6, 7, 8}

    if charge_state not in conf_data:
        return {}

    state_data = conf_data[charge_state]
    if "iqa_EaddIQA" not in state_data:
        return {}

    reindex_map = state_data.get("reindex_map", {})

    # Build inverse map: AMS index -> canonical index
    ams_to_canonical = {}
    for ams_str, canon in reindex_map.items():
        ams_to_canonical[int(ams_str)] = canon

    iqa = {}
    for atom_idx_str, iqa_energy in state_data["iqa_EaddIQA"].items():
        ams_idx = int(atom_idx_str)
        canonical_idx = ams_to_canonical.get(ams_idx)
        if canonical_idx is not None and canonical_idx in valid_atoms:
            iqa[canonical_idx] = iqa_energy

    return iqa


def load_conformer_iqa_energies(
    workdir: Path,
    molecule_name: str
) -> Dict[int, Dict[str, float]]:
    """
    Lit les énergies IQA per-conformère (état neutre).

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule

    Returns:
        {conf_num: {C1_iqa: val, ..., O8_iqa: val}}
    """
    qtaim_json = workdir / "qtaim" / "results" / "qtaim_complete_data.json"
    if not qtaim_json.exists():
        return {}

    try:
        with open(qtaim_json, 'r') as f:
            qtaim_data = json.load(f)
    except Exception:
        return {}

    atom_labels = {
        1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
        7: 'O7', 8: 'O8'
    }

    result = {}
    for conf_id, conf_data in qtaim_data.items():
        conf_num = int(conf_id.replace('conf_', ''))
        neutral_iqa = _load_iqa_single_conformer(conf_data, "0")
        if neutral_iqa:
            iqa_dict = {}
            for atom_idx, energy in neutral_iqa.items():
                label = atom_labels[atom_idx]
                iqa_dict[f"{label}_iqa"] = energy
            result[conf_num] = iqa_dict

    return result


def compute_conformer_deltas(
    workdir: Path,
    molecule_name: str
) -> Dict[int, Dict[str, float]]:
    """
    Calcule les delta Bader et delta IQA per-conformère.

    Pour chaque conformère :
    - delta_X_red = Bader(réduit) - Bader(neutre)
    - delta_X_ox = Bader(oxydé) - Bader(neutre)
    - delta_X_iqa_red = IQA(réduit) - IQA(neutre)
    - delta_X_iqa_ox = IQA(oxydé) - IQA(neutre)

    Returns:
        {conf_num: {delta_C1_red, ..., delta_O8_iqa_ox, ...}}
    """
    qtaim_json = workdir / "qtaim" / "results" / "qtaim_complete_data.json"
    if not qtaim_json.exists():
        return {}

    try:
        with open(qtaim_json, 'r') as f:
            qtaim_data = json.load(f)
    except Exception:
        return {}

    atom_labels = {
        1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
        7: 'O7', 8: 'O8'
    }

    result = {}
    for conf_id, conf_data in qtaim_data.items():
        conf_num = int(conf_id.replace('conf_', ''))
        deltas = {}

        # Delta Bader
        neutral_bader = _load_bader_single_conformer(conf_data, "0")
        reduced_bader = _load_bader_single_conformer(conf_data, "-1")
        oxidized_bader = _load_bader_single_conformer(conf_data, "1")

        for atom_idx, label in atom_labels.items():
            if atom_idx in neutral_bader:
                if atom_idx in reduced_bader:
                    deltas[f"delta_{label}_red"] = reduced_bader[atom_idx] - neutral_bader[atom_idx]
                if atom_idx in oxidized_bader:
                    deltas[f"delta_{label}_ox"] = oxidized_bader[atom_idx] - neutral_bader[atom_idx]

        # Delta IQA
        neutral_iqa = _load_iqa_single_conformer(conf_data, "0")
        reduced_iqa = _load_iqa_single_conformer(conf_data, "-1")
        oxidized_iqa = _load_iqa_single_conformer(conf_data, "1")

        for atom_idx, label in atom_labels.items():
            if atom_idx in neutral_iqa:
                if atom_idx in reduced_iqa:
                    deltas[f"delta_{label}_iqa_red"] = reduced_iqa[atom_idx] - neutral_iqa[atom_idx]
                if atom_idx in oxidized_iqa:
                    deltas[f"delta_{label}_iqa_ox"] = oxidized_iqa[atom_idx] - neutral_iqa[atom_idx]

        if deltas:
            result[conf_num] = deltas

    return result


def compute_conformer_derived(
    bader: Dict[str, float],
    delta_bader: Dict[str, float],
    delta_iqa: Dict[str, float]
) -> Dict[str, float]:
    """
    Calcule les descripteurs dérivés pour UN conformère.

    Même logique que compute_derived_descriptors() mais per-conformer.
    """
    derived = {}

    o7_bader = bader.get('O7_bader')
    o8_bader = bader.get('O8_bader')

    if o7_bader is not None and o8_bader is not None:
        derived['q_O_total'] = o7_bader + o8_bader
        derived['q_O_asymmetry'] = abs(o7_bader - o8_bader)

    delta_o7_red = delta_bader.get('delta_O7_red')
    delta_o8_red = delta_bader.get('delta_O8_red')
    if delta_o7_red is not None and delta_o8_red is not None:
        derived['delta_q_O_total_red'] = delta_o7_red + delta_o8_red

    delta_o7_ox = delta_bader.get('delta_O7_ox')
    delta_o8_ox = delta_bader.get('delta_O8_ox')
    if delta_o7_ox is not None and delta_o8_ox is not None:
        derived['delta_q_O_total_ox'] = delta_o7_ox + delta_o8_ox

    delta_o7_iqa_red = delta_iqa.get('delta_O7_iqa_red')
    delta_o8_iqa_red = delta_iqa.get('delta_O8_iqa_red')
    if delta_o7_iqa_red is not None and delta_o8_iqa_red is not None:
        derived['delta_iqa_O_total_red'] = delta_o7_iqa_red + delta_o8_iqa_red

    delta_o7_iqa_ox = delta_iqa.get('delta_O7_iqa_ox')
    delta_o8_iqa_ox = delta_iqa.get('delta_O8_iqa_ox')
    if delta_o7_iqa_ox is not None and delta_o8_iqa_ox is not None:
        derived['delta_iqa_O_total_ox'] = delta_o7_iqa_ox + delta_o8_iqa_ox

    return derived


def collect_all_conformers(
    workdirs: List[Path],
    freq_csv: Optional[str] = None
) -> 'pd.DataFrame':
    """
    Collecte les données de TOUS les conformères individuellement.

    Contrairement à collect_all_molecules() qui retourne 1 ligne par molécule
    (moyennée par Boltzmann), cette fonction retourne 1 ligne par conformère.

    Args:
        workdirs: Liste de répertoires de travail
        freq_csv: Chemin optionnel vers un CSV de descripteurs fréquentiels
                  per-conformer (avec colonnes molecule_name, conformer, ...)

    Returns:
        DataFrame avec 1 ligne par conformère et les colonnes :
        molecule_name, conformer, weight, + toutes les propriétés redox,
        orbitales, Bader, IQA, deltas, dérivées, et optionnellement freq.
    """
    if not HAS_PANDAS:
        raise ImportError("Pandas requis pour collecter les données")

    # Charger les fréquences per-conformer si fournies
    freq_df = None
    if freq_csv and Path(freq_csv).exists():
        freq_df = pd.read_csv(freq_csv)
        if 'molecule_name' not in freq_df.columns or 'conformer' not in freq_df.columns:
            warnings.warn(f"CSV fréquences {freq_csv} sans colonnes molecule_name/conformer")
            freq_df = None

    all_rows = []

    for workdir in workdirs:
        workdir = Path(workdir)
        mol_name = workdir.name.replace('_workdir', '')

        # 1. Données redox per-conformer (base)
        redox_rows = load_conformer_redox_data(workdir, mol_name)
        if not redox_rows:
            print(f"  {mol_name}: pas de donnees redox per-conformer")
            continue

        # 2. Orbitales per-conformer
        orbital_data = load_conformer_orbital_data(workdir, mol_name)

        # 3. Bader per-conformer
        bader_data = load_conformer_bader_charges(workdir, mol_name)

        # 4. IQA per-conformer
        iqa_data = load_conformer_iqa_energies(workdir, mol_name)

        # 5. Deltas per-conformer
        delta_data = compute_conformer_deltas(workdir, mol_name)

        conf_count = 0
        for redox_row in redox_rows:
            conf_num = redox_row['conformer']
            row = {'molecule_name': mol_name}
            row.update(redox_row)

            # Merge orbitales
            if conf_num in orbital_data:
                row.update(orbital_data[conf_num])

            # Merge Bader
            bader = bader_data.get(conf_num, {})
            row.update(bader)

            # Merge IQA
            iqa = iqa_data.get(conf_num, {})
            row.update(iqa)

            # Merge deltas
            delta = delta_data.get(conf_num, {})
            row.update(delta)

            # Extract delta_bader and delta_iqa subsets for derived computation
            delta_bader_sub = {k: v for k, v in delta.items()
                               if k.startswith('delta_') and '_iqa_' not in k}
            delta_iqa_sub = {k: v for k, v in delta.items()
                             if '_iqa_' in k}

            # Derived descriptors
            derived = compute_conformer_derived(bader, delta_bader_sub, delta_iqa_sub)
            row.update(derived)

            all_rows.append(row)
            conf_count += 1

        print(f"  {mol_name}: {conf_count} conformeres charges")

    if not all_rows:
        raise ValueError("Aucun conformère n'a pu être chargé")

    df = pd.DataFrame(all_rows)

    # Merger les fréquences si disponibles
    if freq_df is not None:
        freq_df['conformer'] = freq_df['conformer'].astype(int)
        df = df.merge(freq_df, on=['molecule_name', 'conformer'], how='left',
                       suffixes=('', '_freq'))
        # Supprimer les colonnes dupliquées
        dup_cols = [c for c in df.columns if c.endswith('_freq')]
        if dup_cols:
            df = df.drop(columns=dup_cols)

    df = df.sort_values(['molecule_name', 'conformer']).reset_index(drop=True)
    return df


def collect_all_molecules(workdirs: List[Path]) -> 'pd.DataFrame':
    """
    Collecte les données de toutes les molécules.

    Args:
        workdirs: Liste de répertoires de travail (e.g. [Path("DHI_workdir"), ...])

    Returns:
        DataFrame avec 1 ligne par molécule et les colonnes :
        - molecule_name
        - E_reduction_V, E_oxidation_V, EA_V, IP_V
        - HOMO_eV, LUMO_eV, gap_eV
        - C1_bader, C2_bader, ..., O7_bader, O8_bader (si QTAIM disponible)
        - C1_iqa, C2_iqa, ..., O7_iqa, O8_iqa (si IQA disponible)

    Note:
        Les molécules avec des données manquantes génèrent des warnings
        mais ne bloquent pas le traitement des autres molécules.
        Seuls les 8 atomes du squelette commun sont inclus.

    Example:
        from pathlib import Path
        from common.multi_molecule_data import collect_all_molecules

        workdirs = [Path("DHI_workdir"), Path("O-DHI_workdir")]
        df = collect_all_molecules(workdirs)
        print(len(df))  # 2
        print(df.columns.tolist())  # Toutes les propriétés
    """
    if not HAS_PANDAS:
        raise ImportError("Pandas requis pour collecter les données")

    molecules_data = []

    for workdir in workdirs:
        try:
            data = load_molecule_summary(workdir)
            molecules_data.append(data)
            print(f"✓ {data['molecule_name']}: données chargées")
        except FileNotFoundError as e:
            print(f"⚠ {workdir.name}: fichier manquant - {e}")
        except ValueError as e:
            print(f"⚠ {workdir.name}: données invalides - {e}")
        except Exception as e:
            print(f"✗ {workdir.name}: erreur inattendue - {e}")

    if not molecules_data:
        raise ValueError("Aucune molécule n'a pu être chargée avec succès")

    # Créer le DataFrame
    df = pd.DataFrame(molecules_data)

    # Trier par nom de molécule
    df = df.sort_values('molecule_name').reset_index(drop=True)

    return df
