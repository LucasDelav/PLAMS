#!/usr/bin/env amspython
"""
Extraction des donnees depuis les fichiers RKF vers JSON.

Supporte deux workflows :
  - DFT : workflow redox complet (7 calculs/conformere + neutre DZP)
  - prediction : workflow predict-dg (3 SP/conformere)

Usage:
    amspython extract_predict_dg.py --solvent Acetonitrile [--data-dir .] [--output-dir dataset]

Modes speciaux:
    amspython extract_predict_dg.py --discover <path/to/adf.rkf>
        Liste toutes les sections et cles disponibles dans un fichier RKF.
"""

import os
import sys
import argparse
import json
import re

import numpy as np
from scm.plams import KFReader

from common.constants import (
    EH_TO_EV, EH_TO_KJ, R
)


# =============================================================================
# CONSTANTES
# =============================================================================

# 1 cm-1 en Hartree (hc * 1 cm-1)
CM1_TO_HARTREE = 4.556335252e-6

# kB en Hartree/K
KB_HARTREE = 3.166811563e-6

# Bohr -> Angstrom
BOHR_TO_ANG = 0.529177249

# Masses atomiques (amu)
ATOM_MASSES = {
    'H': 1.00794, 'He': 4.00260, 'Li': 6.941, 'Be': 9.01218,
    'B': 10.811, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.065,
    'Cl': 35.453, 'Ar': 39.948, 'Br': 79.904, 'I': 126.904,
}

# 1 amu en masse electronique
AMU_TO_ME = 1822.888486

# Atom types a agreger
AGGREGATE_TYPES = ['C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'P']

# Etats attendus par type de workflow
DFT_STATES = [
    'neutral_dzp', 'neutral_tz2p',
    'reduced_sp', 'reduced_opt', 'reduced_tz2p',
    'oxidized_sp', 'oxidized_opt', 'oxidized_tz2p',
]
PREDICTION_STATES = ['neutral_sp', 'reduced_sp', 'oxidized_sp']

# Mapping symlinks redox/results/rkf/ -> cle d'etat
DFT_RKF_MAP = {
    'neutral_DZP': 'neutral_dzp',
    'neutral_TZ2P': 'neutral_tz2p',
    'reduced_sp_DZP': 'reduced_sp',
    'reduced_opt_DZP': 'reduced_opt',
    'reduced_TZ2P': 'reduced_tz2p',
    'oxidized_sp_DZP': 'oxidized_sp',
    'oxidized_opt_DZP': 'oxidized_opt',
    'oxidized_TZ2P': 'oxidized_tz2p',
}

# Ordre de tri des etats : neutre -> reduit -> oxyde
_ALL_STATES_ORDERED = [
    'neutral_sp', 'neutral_dzp', 'neutral_tz2p',
    'reduced_sp', 'reduced_opt', 'reduced_tz2p',
    'oxidized_sp', 'oxidized_opt', 'oxidized_tz2p',
]
_STATE_ORDER = {s: i for i, s in enumerate(_ALL_STATES_ORDERED)}


def _sort_states(states_dict):
    """Trie un dict d'etats selon l'ordre: neutre, reduit, oxyde."""
    def _key(item):
        return _STATE_ORDER.get(item[0], 999)
    from collections import OrderedDict
    return OrderedDict(sorted(states_dict.items(), key=_key))


# =============================================================================
# JSON COMPACT AVEC ARRAYS INLINE
# =============================================================================

def _numpy_convert(obj):
    """Convertit les types numpy en types Python natifs."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _is_numeric_list(obj):
    """Verifie si obj est une liste de nombres (vecteur 1D)."""
    if not isinstance(obj, list) or not obj:
        return False
    return all(isinstance(x, (int, float)) or x is None for x in obj)


def _is_numeric_matrix(obj):
    """Verifie si obj est une liste de listes de nombres (matrice 2D)."""
    if not isinstance(obj, list) or not obj:
        return False
    # Toutes les entrees doivent etre des listes numeriques
    return all(isinstance(row, list) and _is_numeric_list(row) for row in obj)


def _is_block_matrix(obj):
    """Verifie si obj est une matrice de blocs 3x3 (Hessien: list[list[list[list]]])."""
    if not isinstance(obj, list) or not obj:
        return False
    if not isinstance(obj[0], list) or not obj[0]:
        return False
    return all(
        isinstance(row, list) and all(_is_numeric_matrix(block) for block in row)
        for row in obj
    )


def _format_number(x):
    """Formate un nombre de maniere compacte."""
    if x is None:
        return "null"
    if isinstance(x, int):
        return str(x)
    # Eviter la notation scientifique pour les petits nombres, garder la precision
    if abs(x) < 1e-15:
        return "0.0"
    return repr(x)


def _inline_vector(vec):
    """Serialise un vecteur numerique sur une seule ligne."""
    return "[" + ", ".join(_format_number(x) for x in vec) + "]"


def _inline_matrix(mat):
    """Serialise une matrice 2D avec une ligne par row."""
    rows = [_inline_vector(row) for row in mat]
    return rows


def compact_json_dumps(obj, indent=2):
    """
    Serialise en JSON avec indentation pour la structure mais arrays numeriques inline.

    - Dicts : indentes normalement
    - Vecteurs numeriques [1.2, 3.4, ...] : sur une seule ligne
    - Matrices [[...], [...]] : une ligne par row
    - Matrices de blocs 3x3 (Hessien) : un bloc par ligne
    - Listes de dicts (modes normaux) : indentees normalement
    - Strings multi-lignes (XYZ) : preservees telles quelles
    """

    def _serialize(obj, level=0):
        pad = " " * (indent * level)
        pad_inner = " " * (indent * (level + 1))

        obj = _numpy_convert(obj)

        if obj is None:
            return "null"

        if isinstance(obj, bool):
            return "true" if obj else "false"

        if isinstance(obj, (int, float)):
            return _format_number(obj)

        if isinstance(obj, str):
            return json.dumps(obj)

        # Vecteur numerique 1D -> une seule ligne
        if isinstance(obj, list) and _is_numeric_list(obj):
            return _inline_vector(obj)

        # Matrice de blocs 3x3 (Hessien) -> un bloc par ligne
        if isinstance(obj, list) and _is_block_matrix(obj):
            lines = [pad + "["]
            n = len(obj)
            for i, row_of_blocks in enumerate(obj):
                row_lines = []
                for block in row_of_blocks:
                    # Chaque bloc 3x3 sur une seule ligne
                    block_str = "[" + ", ".join(_inline_vector(r) for r in block) + "]"
                    row_lines.append(block_str)
                row_str = pad_inner + "[" + ", ".join(row_lines) + "]"
                if i < n - 1:
                    row_str += ","
                lines.append(row_str)
            lines.append(pad + "]")
            return "\n".join(lines)

        # Matrice numerique 2D -> une ligne par row
        if isinstance(obj, list) and _is_numeric_matrix(obj):
            lines = [pad + "["]
            rows = _inline_matrix(obj)
            for i, row_str in enumerate(rows):
                suffix = "," if i < len(rows) - 1 else ""
                lines.append(pad_inner + row_str + suffix)
            lines.append(pad + "]")
            return "\n".join(lines)

        # Liste generique (modes normaux, etc.)
        if isinstance(obj, list):
            if not obj:
                return "[]"
            lines = [pad + "["]
            for i, item in enumerate(obj):
                suffix = "," if i < len(obj) - 1 else ""
                val = _serialize(item, level + 1)
                # Si la valeur serialisee commence par du padding, l'utiliser tel quel
                if val.startswith(" "):
                    lines.append(val + suffix)
                else:
                    lines.append(pad_inner + val + suffix)
            lines.append(pad + "]")
            return "\n".join(lines)

        # Dict
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            lines = [pad + "{"]
            items = list(obj.items())
            for i, (key, val) in enumerate(items):
                suffix = "," if i < len(items) - 1 else ""
                val_str = _serialize(val, level + 1)
                key_str = json.dumps(key)
                # Si la valeur est multi-ligne, la mettre sur la ligne suivante
                if "\n" in val_str:
                    lines.append(pad_inner + key_str + ":")
                    lines.append(val_str + suffix)
                else:
                    lines.append(pad_inner + key_str + ": " + val_str + suffix)
            lines.append(pad + "}")
            return "\n".join(lines)

        return json.dumps(obj)

    return _serialize(obj)


# =============================================================================
# DECOUVERTE DES SECTIONS RKF
# =============================================================================

def discover_rkf(rkf_path):
    """Liste toutes les sections et cles d'un fichier RKF."""
    kf = KFReader(rkf_path)
    sections = {}
    for sec_var in kf:
        sec, var = sec_var
        if sec not in sections:
            sections[sec] = []
        try:
            val = kf.read(sec, var)
            if isinstance(val, np.ndarray):
                dtype = f"array({val.shape}, {val.dtype})"
            elif isinstance(val, (list, tuple)):
                dtype = f"list(len={len(val)})"
            else:
                dtype = f"{type(val).__name__}: {repr(val)[:80]}"
        except Exception:
            dtype = "???"
        sections[sec].append((var, dtype))
    return sections


def print_discovery(rkf_path):
    """Affiche les sections/cles d'un RKF."""
    print(f"\n=== DISCOVERY: {rkf_path} ===\n")
    sections = discover_rkf(rkf_path)
    for sec in sorted(sections.keys()):
        print(f"\n[{sec}]")
        for var, dtype in sorted(sections[sec]):
            print(f"  {var}: {dtype}")


# =============================================================================
# EXTRACTION DES ATOMES
# =============================================================================

def read_atoms_from_xyz(xyz_path):
    """
    Lit les atomes depuis un fichier XYZ.

    Returns:
        list of str: Symboles atomiques dans l'ordre (ex: ['C', 'C', 'O', 'H', ...])
    """
    atoms = []
    with open(xyz_path) as f:
        n_atoms = int(f.readline().strip())
        f.readline()  # commentaire
        for _ in range(n_atoms):
            parts = f.readline().split()
            atoms.append(parts[0])
    return atoms


def read_atoms_from_rkf(rkf_path):
    """
    Lit les atomes depuis un fichier adf.rkf (fallback si pas de XYZ).

    Returns:
        list of str: Symboles atomiques
    """
    kf = KFReader(rkf_path)
    try:
        n_atoms = kf.read("Geometry", "nr of atoms")
        atom_order = kf.read("Geometry", "atom order input")
        atomtype_labels = kf.read("Geometry", "atomtype").split()
        ntype = kf.read("Geometry", "nr of atomtypes")
        nqptr = kf.read("Geometry", "nqptr")

        atoms_internal = []
        for i in range(ntype):
            start = nqptr[i] - 1
            end = nqptr[i + 1] - 1
            for _ in range(start, end):
                atoms_internal.append(atomtype_labels[i])

        atoms = [''] * n_atoms
        for internal_idx in range(n_atoms):
            input_idx = atom_order[internal_idx] - 1
            atoms[input_idx] = atoms_internal[internal_idx]

        return atoms
    except Exception:
        try:
            labels = str(kf.read("Geometry", "atomtype")).split()
            return labels
        except Exception:
            return []


# =============================================================================
# EXTRACTION DES ENERGIES ET THERMODYNAMIQUE
# =============================================================================

def extract_thermo(rkf_path):
    """
    Extrait Ee, G, H, TS, U depuis adf.rkf.

    Returns:
        dict: Cles Ee, G, H, TS, U (en Hartree), ou None si indisponible
    """
    data = {}
    try:
        kf = KFReader(rkf_path)
        data['Ee'] = kf.read("Energy", "SCF Bond Energy")
    except Exception:
        data['Ee'] = None
        return data

    try:
        data['G'] = kf.read("Thermodynamics", "Gibbs free Energy")
        data['H'] = kf.read("Thermodynamics", "Enthalpy")
        data['TS'] = kf.read("Thermodynamics", "TS")
        try:
            data['U'] = kf.read("Thermodynamics", "Internal Energy total")
        except (KeyError, Exception):
            T = kf.read("Thermodynamics", "Temperature")
            R_ha = R / 4.184 / 627.509  # R en Hartree/K
            data['U'] = data['H'] - R_ha * T
    except (KeyError, Exception):
        data['G'] = None
        data['H'] = None
        data['TS'] = None
        data['U'] = None

    return data


# =============================================================================
# EXTRACTION DES ORBITALES (+-4 autour de HOMO/LUMO)
# =============================================================================

def extract_orbitals(rkf_path, n_neighbors=4):
    """
    Extrait les energies orbitales de HOMO-n a LUMO+n.

    Returns:
        dict: 'HOMO', 'LUMO', 'gap', 'orb_H-4'...'orb_H', 'orb_L'...'orb_L+4' (en eV)
    """
    data = {}
    kf = KFReader(rkf_path)

    try:
        data['HOMO'] = kf.read("Properties", "HOMO") * EH_TO_EV
        data['LUMO'] = kf.read("Properties", "LUMO") * EH_TO_EV
        data['gap'] = data['LUMO'] - data['HOMO']
    except Exception:
        data['HOMO'] = None
        data['LUMO'] = None
        data['gap'] = None

    all_eps = None
    occupations = None

    try:
        all_eps_raw = np.array(kf.read("AMSResults", "orbitalEnergies"))
        all_occ_raw = np.array(kf.read("AMSResults", "orbitalOccupations"))
        n_spin = int(kf.read("AMSResults", "nSpin"))
        n_orb = int(kf.read("AMSResults", "nOrbitals"))

        if n_spin == 2:
            all_eps = all_eps_raw[:n_orb] * EH_TO_EV
            occupations = all_occ_raw[:n_orb]
        else:
            all_eps = all_eps_raw * EH_TO_EV
            occupations = all_occ_raw
    except Exception:
        for eps_key, occ_key in [("eps_A", "froc_A"), ("eps-A_A", "froc-A_A")]:
            try:
                all_eps = np.array(kf.read("A", eps_key)) * EH_TO_EV
                occupations = np.array(kf.read("A", occ_key))
                break
            except Exception:
                continue

    if all_eps is not None and occupations is not None:
        occupied = np.where(occupations > 0)[0]
        if len(occupied) > 0:
            homo_idx = occupied[-1]
            lumo_idx = homo_idx + 1

            for i in range(-n_neighbors, 0):
                idx = homo_idx + i
                key = f"orb_H{i}"
                data[key] = float(all_eps[idx]) if 0 <= idx < len(all_eps) else None

            data['orb_H'] = float(all_eps[homo_idx])

            for i in range(0, n_neighbors + 1):
                idx = lumo_idx + i
                key = f"orb_L+{i}" if i > 0 else "orb_L"
                data[key] = float(all_eps[idx]) if 0 <= idx < len(all_eps) else None
    else:
        for i in range(-n_neighbors, 0):
            data[f"orb_H{i}"] = None
        data['orb_H'] = data.get('HOMO')
        data['orb_L'] = data.get('LUMO')
        for i in range(1, n_neighbors + 1):
            data[f"orb_L+{i}"] = None

    return data


# =============================================================================
# EXTRACTION DES COORDONNEES XYZ
# =============================================================================

def extract_xyz_string(rkf_path, atoms, comment=""):
    """
    Extrait les coordonnees depuis adf.rkf et les formate en string XYZ standard.

    Args:
        rkf_path: chemin vers adf.rkf
        atoms: liste des symboles atomiques (dans l'ordre input)
        comment: ligne de commentaire

    Returns:
        str: contenu au format .xyz, ou None
    """
    try:
        kf = KFReader(rkf_path)
        n_atoms = kf.read("Geometry", "nr of atoms")
        xyz_bohr = np.array(kf.read("Geometry", "xyz")).reshape(n_atoms, 3)
        xyz_ang = xyz_bohr * BOHR_TO_ANG
    except Exception:
        return None

    lines = [str(n_atoms), comment]
    for i in range(n_atoms):
        sym = atoms[i] if i < len(atoms) else "X"
        lines.append(f"{sym:2s}  {xyz_ang[i, 0]:14.8f}  {xyz_ang[i, 1]:14.8f}  {xyz_ang[i, 2]:14.8f}")

    return "\n".join(lines)


# =============================================================================
# EXTRACTION DES FREQUENCES, MODES NORMAUX, GRADIENTS
# =============================================================================

def extract_vibrations(rkf_path):
    """
    Extrait frequences et modes normaux depuis adf.rkf.

    Returns:
        dict: 'frequencies' (cm-1), 'normal_modes' (list de vecteurs), 'n_modes'
    """
    data = {'frequencies': None, 'normal_modes': None, 'n_modes': 0}
    try:
        kf = KFReader(rkf_path)
        freqs = np.array(kf.read("Vibrations", "Frequencies[cm-1]"))
        data['frequencies'] = freqs
        data['n_modes'] = len(freqs)

        modes = []
        for i in range(1, len(freqs) + 1):
            try:
                mode = np.array(kf.read("Vibrations", f"NoWeightNormalMode({i})"))
                modes.append(mode)
            except Exception:
                break
        if modes:
            data['normal_modes'] = modes
    except Exception:
        pass
    return data


def extract_gradients(rkf_path):
    """
    Extrait les gradients depuis adf.rkf.

    Returns:
        np.array ou None: Gradient en Hartree/Bohr (vecteur 3N)
    """
    try:
        kf = KFReader(rkf_path)
    except Exception:
        return None

    for section, key in [
        ("AMSResults", "Gradients"),
        ("GeoOpt", "Gradients_InputOrder"),
        ("GeoOpt", "Gradients_CART"),
    ]:
        try:
            grad = np.array(kf.read(section, key))
            return grad
        except Exception:
            continue

    return None


# =============================================================================
# EXTRACTION DU HESSIEN
# =============================================================================

def extract_hessian(rkf_path, n_atoms):
    """
    Extrait la matrice Hessienne cartesienne (3N x 3N) depuis adf.rkf.

    Returns:
        np.array (3N, 3N) en Hartree/Bohr^2, ou None
    """
    try:
        kf = KFReader(rkf_path)
        hess_flat = np.array(kf.read("AMSResults", "Hessian"), dtype=float)
        dof = 3 * n_atoms
        expected = dof * dof
        if len(hess_flat) != expected:
            return None
        return hess_flat.reshape(dof, dof)
    except Exception:
        return None


# =============================================================================
# CALCUL LVC2
# =============================================================================

def compute_zpe(frequencies_cm1):
    """Calcule la ZPE en Hartree depuis les frequences en cm-1."""
    if frequencies_cm1 is None:
        return None
    positive = frequencies_cm1[frequencies_cm1 > 0]
    return 0.5 * np.sum(positive * CM1_TO_HARTREE)


def compute_ts_vib(frequencies_cm1, temperature=298.15):
    """
    Calcule TS_vib en Hartree (approximation RRHO).

    S_vib = kB * sum [u/(e^u-1) - ln(1-e^{-u})] pour w > 10 cm-1
    """
    if frequencies_cm1 is None:
        return None
    ts = 0.0
    for freq in frequencies_cm1:
        if freq <= 10.0:
            continue
        u = freq * CM1_TO_HARTREE / (KB_HARTREE * temperature)
        if u > 500:
            continue
        exp_u = np.exp(u)
        s_i = u / (exp_u - 1.0) - np.log(1.0 - np.exp(-u))
        ts += s_i
    return KB_HARTREE * temperature * ts


LVC2_FREQ_MIN_CM1 = 50.0
"""Seuil de coupure des modes basse frequence pour le calcul LVC2.

Les modes positifs sous ce seuil sont des artefacts numeriques du hessien
ionique non-stationnaire calcule a la geometrie neutre : avec omega tres
faible, le terme F^2/(2k) = F^2/(2*mu*omega^2) diverge artificiellement.
Les modes physiques pertinents (torsions, rotations de groupes) commencent
typiquement vers 50-100 cm-1. Ce seuil reproduit le R^2 = 0.84 de la sec.
3.2 du rapport (CSV lvc_summary.csv) la ou freq>0 donne R^2 = 0.45.
"""


def compute_lvc2(gradients, frequencies, normal_modes, atoms,
                 freq_min_cm1=LVC2_FREQ_MIN_CM1):
    """
    Calcule E_relax^LVC2 = -sum F'_j^2 / (2k'_j) en Hartree.

    Args:
        gradients: np.array (3N,) en Hartree/Bohr
        frequencies: np.array (n_modes,) en cm-1
        normal_modes: list de np.array (3N,), normalises en cartesien
        atoms: list de str (symboles atomiques)
        freq_min_cm1: seuil de coupure (defaut LVC2_FREQ_MIN_CM1 = 50 cm-1)

    Returns:
        float ou None: E_relax en Hartree (negatif)
    """
    if gradients is None or frequencies is None or normal_modes is None:
        return None

    n_atoms = len(atoms)
    if len(gradients) != 3 * n_atoms:
        return None

    masses_3n = []
    for sym in atoms:
        m = ATOM_MASSES.get(sym, 12.0) * AMU_TO_ME
        masses_3n.extend([m, m, m])
    masses_3n = np.array(masses_3n)

    e_relax = 0.0
    for j, freq in enumerate(frequencies):
        if freq < freq_min_cm1 or j >= len(normal_modes):
            continue

        mode = np.array(normal_modes[j])
        if len(mode) != 3 * n_atoms:
            continue

        f_j = np.dot(gradients, mode)
        mu_j = np.sum(masses_3n * mode**2)
        omega_j = freq * CM1_TO_HARTREE
        k_j = mu_j * omega_j**2

        if k_j > 1e-12:
            e_relax -= f_j**2 / (2.0 * k_j)

    return e_relax


# =============================================================================
# EXTRACTION ConceptualDFT
# =============================================================================

def extract_conceptual_dft(rkf_path, n_atoms):
    """
    Extrait les donnees ConceptualDFT depuis adf.rkf.

    Returns:
        dict: Globaux (mu, eta, omega) + par atome (fplus, fminus, dual)
    """
    data = {'mu': None, 'eta': None, 'omega': None,
            'fplus': None, 'fminus': None, 'f0': None, 'dual': None}

    try:
        kf = KFReader(rkf_path)
    except Exception:
        return data

    for key_name, data_key in [
        ("Fukui Fplus", "fplus"),
        ("Fukui Fminus", "fminus"),
        ("Koopmans DD", "dual"),
    ]:
        try:
            val = kf.read("Properties", key_name)
            if isinstance(val, (list, np.ndarray)):
                arr = np.array(val)
                if len(arr) >= n_atoms:
                    data[data_key] = arr[:n_atoms]
        except Exception:
            pass

    if data['fplus'] is not None and data['fminus'] is not None:
        data['f0'] = (data['fplus'] + data['fminus']) / 2.0

    if data['dual'] is None and data['fplus'] is not None and data['fminus'] is not None:
        data['dual'] = data['fplus'] - data['fminus']

    try:
        homo = kf.read("Properties", "HOMO")
        lumo = kf.read("Properties", "LUMO")
        data['mu'] = (homo + lumo) / 2.0 * EH_TO_EV
        data['eta'] = (lumo - homo) * EH_TO_EV
        if data['eta'] > 0:
            data['omega'] = (data['mu'] ** 2) / (2.0 * data['eta'])
    except Exception:
        pass

    return data


# =============================================================================
# EXTRACTION CHARGES (Hirshfeld, Mulliken, Voronoi) + DIPOLE + DENSITE/POTENTIEL
# =============================================================================

def extract_charges_and_dipole(rkf_path, n_atoms):
    """
    Extrait les charges atomiques, le dipole, la densite et le potentiel aux noyaux.

    Returns:
        dict: hirshfeld_charges, mulliken_charges, voronoi_charges,
              electron_density_at_nuclei, electrostatic_potential_at_nuclei,
              dipole (vecteur 3D en Debye)
    """
    data = {
        'hirshfeld_charges': None,
        'mulliken_charges': None,
        'voronoi_charges': None,
        'electron_density_at_nuclei': None,
        'electrostatic_potential_at_nuclei': None,
        'dipole': None,
    }

    try:
        kf = KFReader(rkf_path)
    except Exception:
        return data

    # Charges atomiques
    charge_keys = [
        ("FragmentCharge Hirshfeld", "hirshfeld_charges"),
        ("AtomCharge Mulliken", "mulliken_charges"),
        ("AtomCharge_SCF Voronoi", "voronoi_charges"),
    ]
    for key_name, data_key in charge_keys:
        try:
            val = kf.read("Properties", key_name)
            if isinstance(val, (list, np.ndarray)):
                arr = np.array(val, dtype=float)
                if len(arr) >= n_atoms:
                    data[data_key] = arr[:n_atoms]
        except Exception:
            pass

    # Densite electronique aux noyaux
    try:
        val = kf.read("Properties", "Electron Density at Nuclei")
        if isinstance(val, (list, np.ndarray)):
            arr = np.array(val, dtype=float)
            if len(arr) >= n_atoms:
                data['electron_density_at_nuclei'] = arr[:n_atoms]
    except Exception:
        pass

    # Potentiel electrostatique aux noyaux
    try:
        val = kf.read("Properties", "Electrostatic Pot.at Nuclei")
        if isinstance(val, (list, np.ndarray)):
            arr = np.array(val, dtype=float)
            if len(arr) >= n_atoms:
                data['electrostatic_potential_at_nuclei'] = arr[:n_atoms]
    except Exception:
        pass

    # Dipole (vecteur 3D)
    try:
        val = kf.read("Properties", "Dipole")
        if isinstance(val, (list, np.ndarray)):
            arr = np.array(val, dtype=float)
            if len(arr) >= 3:
                data['dipole'] = arr[:3]
    except Exception:
        pass

    return data


# =============================================================================
# EXTRACTION QTAIM
# =============================================================================

def extract_qtaim(rkf_path, n_atoms):
    """
    Extrait les charges QTAIM (Bader) depuis adf.rkf.

    Returns:
        dict: 'bader_charges' (np.array par atome)
    """
    data = {'bader_charges': None}

    try:
        kf = KFReader(rkf_path)
    except Exception:
        return data

    for key_name, data_key in [
        ("Bader atomic charges", "bader_charges"),
    ]:
        try:
            val = kf.read("Properties", key_name)
            if isinstance(val, (list, np.ndarray)):
                arr = np.array(val)
                if len(arr) >= n_atoms:
                    data[data_key] = arr[:n_atoms]
        except Exception:
            pass

    return data


# =============================================================================
# EXTRACTION MATRICES (Linear Response + LI-DI)
# =============================================================================

def extract_matrices(rkf_path, n_atoms):
    """
    Extrait les matrices condensees depuis adf.rkf.

    Returns:
        dict: 'linear_response' (np.array packed), 'lidi' (np.array packed),
              'LI' (np.array diag), 'DI_total' (np.array)
    """
    n_packed = n_atoms * (n_atoms + 1) // 2
    data = {'linear_response': None, 'lidi': None, 'LI': None, 'DI_total': None}

    try:
        kf = KFReader(rkf_path)
    except Exception:
        return data

    try:
        val = kf.read("Properties", "Cond Linear Response")
        arr = np.array(val)
        if len(arr) >= n_packed:
            data['linear_response'] = arr[:n_packed]
    except Exception:
        pass

    try:
        val = kf.read("Properties", "LI-DI packed")
        arr = np.array(val)
        if len(arr) >= n_packed:
            data['lidi'] = arr[:n_packed]

            li = np.zeros(n_atoms)
            di_total = np.zeros(n_atoms)
            idx = 0
            for i in range(n_atoms):
                for j in range(i + 1):
                    if i == j:
                        li[i] = arr[idx]
                    else:
                        di_total[i] += arr[idx]
                        di_total[j] += arr[idx]
                    idx += 1
            data['LI'] = li
            data['DI_total'] = di_total
    except Exception:
        pass

    return data


# =============================================================================
# EXTRACTION POPULATIONS SFO (LUMO)
# =============================================================================

def extract_sfo_lumo_populations(rkf_path, n_atoms):
    """
    Extrait la population de la LUMO par atome.

    Returns:
        dict: 'lumo_pop' (np.array par atome ou None)
    """
    data = {'lumo_pop': None}

    try:
        kf = KFReader(rkf_path)
    except Exception:
        return data

    for sec in ["SFO", "Properties", "Gross Populations"]:
        for key in ["SFO popul", "LUMO populations", "Gross Populations",
                     "OrbPop", "SFO_Popul"]:
            try:
                val = kf.read(sec, key)
                if isinstance(val, (list, np.ndarray)):
                    data['lumo_pop_raw'] = np.array(val)
                    return data
            except Exception:
                pass

    return data


# =============================================================================
# AGREGATION PAR TYPE D'ATOME
# =============================================================================

def aggregate_by_type(values, atoms):
    """
    Agrege des valeurs par-atome par type d'atome (somme).

    Returns:
        dict: {'O': sum_O, 'C': sum_C, ...}
    """
    if values is None:
        return {}
    agg = {}
    for i, sym in enumerate(atoms):
        if i < len(values):
            agg[sym] = agg.get(sym, 0.0) + float(values[i])
    return agg


# =============================================================================
# DEPAQUETAGE DES MATRICES TRIANGULAIRES
# =============================================================================

def unpack_triangle_to_full(packed, n, triangle='lower'):
    """
    Depaquete un triangle en matrice 2D symetrique complete.

    Args:
        packed: np.array de taille n*(n+1)/2
        n: dimension de la matrice
        triangle: 'lower' (LI-DI) ou 'upper' (Linear Response)

    Returns:
        list[list[float]]: matrice n x n
    """
    if packed is None:
        return None

    mat = np.zeros((n, n))
    idx = 0

    if triangle == 'lower':
        for i in range(n):
            for j in range(i + 1):
                mat[i, j] = packed[idx]
                mat[j, i] = packed[idx]
                idx += 1
    else:  # upper
        for i in range(n):
            for j in range(i, n):
                mat[i, j] = packed[idx]
                mat[j, i] = packed[idx]
                idx += 1

    return mat.tolist()


# =============================================================================
# EXTRACTION COMPLETE D'UN JOB
# =============================================================================

def extract_all_from_job(job_dir, rkf_name="adf.rkf", atoms=None, state_label=""):
    """
    Extrait toutes les donnees d'un job.

    Args:
        job_dir: Chemin vers le repertoire du job
        rkf_name: Nom du fichier RKF engine
        atoms: Liste des symboles atomiques (si None, tente de lire depuis RKF)
        state_label: label pour la ligne de commentaire XYZ

    Returns:
        dict: Toutes les donnees extraites
    """
    adf_rkf = os.path.join(job_dir, rkf_name)
    if not os.path.exists(adf_rkf):
        print(f"  ! RKF non trouve: {adf_rkf}")
        return None

    if atoms is None:
        atoms = read_atoms_from_rkf(adf_rkf)
    n_atoms = len(atoms)

    data = {}

    # 0. Coordonnees XYZ
    data['xyz'] = extract_xyz_string(adf_rkf, atoms, comment=state_label)

    # 1. Thermo
    data.update(extract_thermo(adf_rkf))

    # 2. Orbitales
    data.update(extract_orbitals(adf_rkf))

    # 3. Vibrations
    vib = extract_vibrations(adf_rkf)
    data['frequencies'] = vib['frequencies']
    data['normal_modes'] = vib['normal_modes']
    data['n_modes'] = vib['n_modes']

    # ZPE
    try:
        kf_zpe = KFReader(adf_rkf)
        data['ZPE'] = kf_zpe.read("Vibrations", "ZeroPointEnergy")
    except Exception:
        data['ZPE'] = compute_zpe(vib['frequencies'])
    data['TS_vib'] = compute_ts_vib(vib['frequencies'])

    # 4. Gradients
    data['gradients'] = extract_gradients(adf_rkf)

    # 5. Hessien
    data['hessian'] = extract_hessian(adf_rkf, n_atoms)

    # 6. Charges (Hirshfeld, Mulliken, Voronoi) + dipole + densite/potentiel
    data['charges'] = extract_charges_and_dipole(adf_rkf, n_atoms)

    # 7. ConceptualDFT
    data['cdft'] = extract_conceptual_dft(adf_rkf, n_atoms)

    # 8. QTAIM
    data['qtaim'] = extract_qtaim(adf_rkf, n_atoms)

    # 9. SFO populations
    data['sfo'] = extract_sfo_lumo_populations(adf_rkf, n_atoms)

    # 10. Matrices (Linear Response + LI-DI)
    data['matrices'] = extract_matrices(adf_rkf, n_atoms)

    return data


def extract_from_rkf_file(rkf_path, atoms=None, state_label=""):
    """
    Extrait toutes les donnees directement depuis un fichier RKF (pas un repertoire).

    Args:
        rkf_path: Chemin vers adf.rkf
        atoms: Liste des symboles atomiques
        state_label: label pour la ligne de commentaire XYZ

    Returns:
        dict: Toutes les donnees extraites
    """
    if not os.path.exists(rkf_path):
        print(f"  ! RKF non trouve: {rkf_path}")
        return None

    if atoms is None:
        atoms = read_atoms_from_rkf(rkf_path)
    n_atoms = len(atoms)

    data = {}

    data['xyz'] = extract_xyz_string(rkf_path, atoms, comment=state_label)
    data.update(extract_thermo(rkf_path))
    data.update(extract_orbitals(rkf_path))

    vib = extract_vibrations(rkf_path)
    data['frequencies'] = vib['frequencies']
    data['normal_modes'] = vib['normal_modes']
    data['n_modes'] = vib['n_modes']

    try:
        kf_zpe = KFReader(rkf_path)
        data['ZPE'] = kf_zpe.read("Vibrations", "ZeroPointEnergy")
    except Exception:
        data['ZPE'] = compute_zpe(vib['frequencies'])
    data['TS_vib'] = compute_ts_vib(vib['frequencies'])

    data['gradients'] = extract_gradients(rkf_path)
    data['hessian'] = extract_hessian(rkf_path, n_atoms)
    data['charges'] = extract_charges_and_dipole(rkf_path, n_atoms)
    data['cdft'] = extract_conceptual_dft(rkf_path, n_atoms)
    data['qtaim'] = extract_qtaim(rkf_path, n_atoms)
    data['sfo'] = extract_sfo_lumo_populations(rkf_path, n_atoms)
    data['matrices'] = extract_matrices(rkf_path, n_atoms)

    return data


# =============================================================================
# CONSTRUCTION DU JSON PAR ETAT
# =============================================================================

def build_state_json(state_data, atoms, n_atoms):
    """
    Convertit la sortie de extract_all_from_job en dict JSON structure.

    Les matrices packed sont depaquetees en matrices 2D completes.
    Les modes normaux ne sont PAS inclus (trop volumineux, utilises uniquement pour LVC2).
    """
    s = {}

    # Coordonnees XYZ (string format .xyz)
    s['xyz'] = state_data.get('xyz')

    # Thermo
    s['thermo'] = {
        'Ee': state_data.get('Ee'),
        'G': state_data.get('G'),
        'H': state_data.get('H'),
        'TS': state_data.get('TS'),
        'U': state_data.get('U'),
    }

    # Orbitales
    s['orbitals'] = {
        'HOMO': state_data.get('HOMO'),
        'LUMO': state_data.get('LUMO'),
        'gap': state_data.get('gap'),
        'spectrum': {}
    }
    for i in range(-4, 0):
        s['orbitals']['spectrum'][f'H{i}'] = state_data.get(f'orb_H{i}')
    s['orbitals']['spectrum']['H'] = state_data.get('orb_H')
    s['orbitals']['spectrum']['L'] = state_data.get('orb_L')
    for i in range(1, 5):
        s['orbitals']['spectrum'][f'L+{i}'] = state_data.get(f'orb_L+{i}')

    # Vibrations : modes normaux avec frequences et deplacements
    freqs = state_data.get('frequencies')
    normal_modes = state_data.get('normal_modes')
    n_atoms = len(atoms)

    modes_list = []
    if freqs is not None:
        for j, freq in enumerate(freqs):
            mode_entry = {
                'frequency_cm1': float(freq),
            }
            if normal_modes is not None and j < len(normal_modes):
                mode_vec = np.array(normal_modes[j])
                # Reformater en (n_atoms, 3) : [[dx1,dy1,dz1], [dx2,dy2,dz2], ...]
                if len(mode_vec) == 3 * n_atoms:
                    mode_entry['displacements'] = mode_vec.reshape(n_atoms, 3).tolist()
                else:
                    mode_entry['displacements'] = mode_vec.tolist()
            modes_list.append(mode_entry)

    s['vibrations'] = {
        'n_modes': state_data.get('n_modes', 0),
        'ZPE': state_data.get('ZPE'),
        'TS_vib': state_data.get('TS_vib'),
        'modes': modes_list if modes_list else None,
    }

    # Gradients : (N, 3) - force par atome [dE/dx, dE/dy, dE/dz]
    grad = state_data.get('gradients')
    if isinstance(grad, np.ndarray) and len(grad) == 3 * n_atoms:
        s['gradients'] = grad.reshape(n_atoms, 3).tolist()
    else:
        s['gradients'] = grad.tolist() if isinstance(grad, np.ndarray) else grad

    # Hessien : (N, N) blocs de (3, 3) - couplage atome-atome
    hess = state_data.get('hessian')
    if isinstance(hess, np.ndarray) and hess.shape == (3 * n_atoms, 3 * n_atoms):
        # Restructurer (3N, 3N) -> liste[N][N] de matrices 3x3
        hess_blocks = []
        for i in range(n_atoms):
            row = []
            for j in range(n_atoms):
                block = hess[3*i:3*i+3, 3*j:3*j+3]
                row.append(block.tolist())
            hess_blocks.append(row)
        s['hessian'] = hess_blocks
    else:
        s['hessian'] = hess.tolist() if isinstance(hess, np.ndarray) else hess

    # Charges atomiques + dipole + densite/potentiel aux noyaux
    charges = state_data.get('charges', {})
    s['charges'] = {
        'hirshfeld': charges.get('hirshfeld_charges'),
        'mulliken': charges.get('mulliken_charges'),
        'voronoi': charges.get('voronoi_charges'),
        'electron_density_at_nuclei': charges.get('electron_density_at_nuclei'),
        'electrostatic_potential_at_nuclei': charges.get('electrostatic_potential_at_nuclei'),
        'dipole': charges.get('dipole'),
    }

    # ConceptualDFT
    cdft = state_data.get('cdft', {})
    s['cdft'] = {
        'mu': cdft.get('mu'),
        'eta': cdft.get('eta'),
        'omega': cdft.get('omega'),
        'fplus': cdft.get('fplus'),
        'fminus': cdft.get('fminus'),
        'f0': cdft.get('f0'),
        'dual': cdft.get('dual'),
    }

    # QTAIM
    qtaim = state_data.get('qtaim', {})
    s['qtaim'] = {
        'bader_charges': qtaim.get('bader_charges'),
    }

    # Matrices - depaquetees en 2D
    matrices = state_data.get('matrices', {})
    lr_packed = matrices.get('linear_response')
    lidi_packed = matrices.get('lidi')
    s['matrices'] = {
        'linear_response': unpack_triangle_to_full(lr_packed, n_atoms, 'upper'),
        'lidi': unpack_triangle_to_full(lidi_packed, n_atoms, 'lower'),
        'LI': matrices.get('LI'),
        'DI_total': matrices.get('DI_total'),
    }

    # SFO populations
    sfo = state_data.get('sfo', {})
    s['sfo'] = {
        'lumo_pop': sfo.get('lumo_pop'),
        'lumo_pop_raw': sfo.get('lumo_pop_raw'),
    }

    return s


def build_aggregated(states_json, atoms):
    """
    Construit la section 'aggregated' avec les sommes par type d'atome.
    """
    agg = {}

    for state_label, state_data in states_json.items():
        state_agg = {}

        # Charges par type
        charges = state_data.get('charges', {})
        for prop_name in ['hirshfeld', 'mulliken', 'voronoi']:
            values = charges.get(prop_name)
            if values is not None:
                arr = np.array(values) if not isinstance(values, np.ndarray) else values
                state_agg[prop_name] = aggregate_by_type(arr, atoms)

        # CDFT par type
        cdft = state_data.get('cdft', {})
        for prop_name in ['fplus', 'fminus', 'f0', 'dual']:
            values = cdft.get(prop_name)
            if values is not None:
                arr = np.array(values) if not isinstance(values, np.ndarray) else values
                state_agg[prop_name] = aggregate_by_type(arr, atoms)

        # QTAIM par type
        qtaim = state_data.get('qtaim', {})
        for prop_name in ['bader_charges']:
            values = qtaim.get(prop_name)
            if values is not None:
                arr = np.array(values) if not isinstance(values, np.ndarray) else values
                state_agg[prop_name] = aggregate_by_type(arr, atoms)

        # LI/DI par type
        matrices = state_data.get('matrices', {})
        for prop_name in ['LI', 'DI_total']:
            values = matrices.get(prop_name)
            if values is not None:
                arr = np.array(values) if not isinstance(values, np.ndarray) else values
                state_agg[prop_name] = aggregate_by_type(arr, atoms)

        if state_agg:
            agg[state_label] = state_agg

    return agg


# =============================================================================
# DECOUVERTE DES MOLECULES ET DETECTION DU WORKFLOW
# =============================================================================

def discover_molecules(data_dir):
    """
    Auto-detecte les *_workdir/ dans le repertoire donne.

    Returns:
        list of (mol_name, workdir_path)
    """
    molecules = []
    if not os.path.isdir(data_dir):
        print(f"  ! Repertoire non trouve: {data_dir}")
        return molecules

    for entry in sorted(os.listdir(data_dir)):
        if entry.endswith('_workdir') and os.path.isdir(os.path.join(data_dir, entry)):
            mol_name = entry[:-len('_workdir')]
            molecules.append((mol_name, os.path.join(data_dir, entry)))

    return molecules


def detect_workflow_type(workdir, mol_name):
    """
    Detecte le type de workflow (DFT ou prediction).

    DFT : presence de *_neutral_tz2p ou de symlinks TZ2P dans redox/results/rkf/
    prediction : seulement *_sp dans plams_workdir
    """
    rkf_dir = os.path.join(workdir, "redox", "results", "rkf")
    if os.path.isdir(rkf_dir):
        for f in os.listdir(rkf_dir):
            if 'TZ2P' in f:
                return 'DFT'

    calc_dir = os.path.join(workdir, "redox", "calculations", "plams_workdir")
    if os.path.isdir(calc_dir):
        for entry in os.listdir(calc_dir):
            if f"{mol_name}_conf_" in entry and entry.endswith('_neutral_tz2p'):
                return 'DFT'
        for entry in os.listdir(calc_dir):
            if f"{mol_name}_conf_" in entry and entry.endswith('_neutral_sp'):
                return 'prediction'

    return None


# =============================================================================
# DECOUVERTE DES JOBS PAR WORKFLOW
# =============================================================================

def _resolve_rkf_symlink(link_path, workdir, mol_name):
    """
    Resout un symlink RKF, y compris les symlinks brises pointant vers le cluster.

    Si le symlink pointe vers un chemin absolu du cluster (ex: /dlocal/run/...),
    on extrait la partie relative apres {mol_name}_workdir/ et on la reconstruit
    par rapport au workdir local.
    """
    # Si c'est un fichier normal ou un symlink valide
    real = os.path.realpath(link_path)
    if os.path.exists(real):
        return real

    # Symlink brise - tenter de remapper
    if os.path.islink(link_path):
        target = os.readlink(link_path)
        marker = f"{mol_name}_workdir/"
        idx = target.find(marker)
        if idx >= 0:
            relative = target[idx + len(marker):]
            local_path = os.path.join(workdir, relative)
            if os.path.exists(local_path):
                return local_path

    return None


def find_dft_conformer_jobs(workdir, mol_name):
    """
    Trouve les RKF pour chaque conformere du workflow DFT.

    Utilise les symlinks dans redox/results/rkf/ avec remapping des
    chemins brises du cluster vers les chemins locaux.

    Returns:
        dict: {conf_num: {state_key: rkf_path}}
    """
    rkf_dir = os.path.join(workdir, "redox", "results", "rkf")
    if not os.path.isdir(rkf_dir):
        return {}

    jobs = {}
    pattern = re.compile(r'^conf_(\d+)_(.+)\.rkf$')

    for filename in os.listdir(rkf_dir):
        m = pattern.match(filename)
        if not m:
            continue

        conf_num = int(m.group(1))
        suffix = m.group(2)  # ex: neutral_DZP, reduced_sp_DZP, etc.

        state_key = DFT_RKF_MAP.get(suffix)
        if state_key is None:
            continue

        link_path = os.path.join(rkf_dir, filename)
        rkf_path = _resolve_rkf_symlink(link_path, workdir, mol_name)

        if rkf_path is None:
            continue

        if conf_num not in jobs:
            jobs[conf_num] = {}
        jobs[conf_num][state_key] = rkf_path

    return jobs


def find_prediction_conformer_jobs(workdir, mol_name):
    """
    Trouve les jobs predict-dg pour chaque conformere.

    Returns:
        dict: {conf_num: {'neutral_sp': job_dir, 'reduced_sp': job_dir, 'oxidized_sp': job_dir}}
    """
    calc_dir = os.path.join(workdir, "redox", "calculations", "plams_workdir")
    if not os.path.exists(calc_dir):
        return {}

    jobs = {}
    for entry in os.listdir(calc_dir):
        full_path = os.path.join(calc_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if f"{mol_name}_conf_" not in entry:
            continue

        parts = entry.split('_')
        try:
            conf_idx = parts.index('conf')
            conf_num = int(parts[conf_idx + 1])
        except (ValueError, IndexError):
            continue

        if entry.endswith('_neutral_sp'):
            job_type = 'neutral_sp'
        elif entry.endswith('_reduced_sp'):
            job_type = 'reduced_sp'
        elif entry.endswith('_oxidized_sp'):
            job_type = 'oxidized_sp'
        else:
            continue

        if conf_num not in jobs:
            jobs[conf_num] = {}
        jobs[conf_num][job_type] = full_path

    return jobs


def read_boltzmann_weights(workdir):
    """Lit les poids de Boltzmann depuis conformers_summary.txt."""
    summary_file = os.path.join(workdir, "conformers", "results", "conformers_summary.txt")
    if not os.path.exists(summary_file):
        return {}

    weights = {}
    in_section = False

    with open(summary_file) as f:
        for line in f:
            stripped = line.strip()
            if "CONFORMERES FINAUX" in stripped or "CONFORMÈRES FINAUX" in stripped:
                in_section = True
                continue
            if in_section:
                if stripped.startswith("Fichiers") or stripped.startswith("Note:"):
                    break
                if not stripped or stripped.startswith("===") or stripped.startswith("---"):
                    continue
                if "#" in stripped and ("ΔE" in stripped or "dE" in stripped):
                    continue
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        conf_num = int(parts[0])
                        weight = float(parts[2])
                        weights[conf_num] = weight
                    except (ValueError, IndexError):
                        continue
    return weights


def find_xyz_file(workdir, conf_num):
    """Trouve le fichier XYZ d'un conformere."""
    results_dir = os.path.join(workdir, "conformers", "results")
    xyz_path = os.path.join(results_dir, f"conf_{conf_num}.xyz")
    if os.path.exists(xyz_path):
        return xyz_path
    return None


# =============================================================================
# GRANDEURS DERIVEES
# =============================================================================

def _make_unit_dict(value_ha):
    """Cree un dict avec ha, kj, ev a partir d'une valeur en Hartree."""
    if value_ha is None:
        return None
    return {
        'ha': float(value_ha),
        'kj': float(value_ha * EH_TO_KJ),
        'ev': float(value_ha * EH_TO_EV),
    }


def _make_unit_dict_no_ev(value_ha):
    """Cree un dict avec ha, kj a partir d'une valeur en Hartree."""
    if value_ha is None:
        return None
    return {
        'ha': float(value_ha),
        'kj': float(value_ha * EH_TO_KJ),
    }


def compute_prediction_derived(states, atoms):
    """
    Calcule les grandeurs derivees pour le workflow prediction.

    Args:
        states: dict {state_key: state_data} (sortie de extract_all_from_job)
        atoms: list de str

    Returns:
        dict: grandeurs derivees
    """
    n = states.get('neutral_sp', {})
    r = states.get('reduced_sp', {})
    o = states.get('oxidized_sp', {})
    derived = {}

    # EA vertical = Ee(neutral) - Ee(reduced)
    if n.get('Ee') is not None and r.get('Ee') is not None:
        derived['EA_vert'] = _make_unit_dict(n['Ee'] - r['Ee'])

    # IP vertical = Ee(oxidized) - Ee(neutral)
    if n.get('Ee') is not None and o.get('Ee') is not None:
        derived['IP_vert'] = _make_unit_dict(o['Ee'] - n['Ee'])

    # Delta G
    if n.get('G') is not None and r.get('G') is not None:
        derived['delta_G_red'] = _make_unit_dict_no_ev(r['G'] - n['G'])

    if n.get('G') is not None and o.get('G') is not None:
        derived['delta_G_ox'] = _make_unit_dict_no_ev(o['G'] - n['G'])

    # LVC2
    for ion_label, ion_key in [('red', 'reduced_sp'), ('ox', 'oxidized_sp')]:
        ion = states.get(ion_key, {})
        lvc2 = compute_lvc2(
            ion.get('gradients'), ion.get('frequencies'),
            ion.get('normal_modes'), atoms
        )
        derived[f'E_relax_LVC2_{ion_label}'] = _make_unit_dict_no_ev(lvc2)

    return derived


def compute_dft_derived(states):
    """
    Calcule les grandeurs derivees pour le workflow DFT.

    Schema composite : G_composite = Ee(TZ2P) + [G(DZP) - Ee(DZP)]

    Args:
        states: dict {state_key: state_data} (sortie de extract_from_rkf_file)

    Returns:
        dict: grandeurs derivees
    """
    derived = {}

    n_dzp = states.get('neutral_dzp', {})
    n_tz2p = states.get('neutral_tz2p', {})
    r_sp = states.get('reduced_sp', {})
    r_opt = states.get('reduced_opt', {})
    r_tz2p = states.get('reduced_tz2p', {})
    o_sp = states.get('oxidized_sp', {})
    o_opt = states.get('oxidized_opt', {})
    o_tz2p = states.get('oxidized_tz2p', {})

    # EA vertical (TZ2P coherent)
    # Le reduced_sp est DZP sur geom neutre, on n'a pas de TZ2P SP sur geom neutre en charge -1
    # Mais on a neutral_tz2p et reduced_sp. EA_vert_DZP = Ee(n_dzp) - Ee(r_sp)
    if n_dzp.get('Ee') is not None and r_sp.get('Ee') is not None:
        derived['EA_vert_DZP'] = _make_unit_dict(n_dzp['Ee'] - r_sp['Ee'])

    if n_dzp.get('Ee') is not None and o_sp.get('Ee') is not None:
        derived['IP_vert_DZP'] = _make_unit_dict(o_sp['Ee'] - n_dzp['Ee'])

    # Edef (deformation) en TZ2P : Ee(ion_tz2p) - Ee(ion_sp)
    # Note: reduced_sp est DZP, pas TZ2P. On ne peut pas calculer Edef en base coherente TZ2P
    # sans un SP TZ2P sur geom neutre en charge -1/+1.
    # On calcule Edef en DZP: Ee(ion_opt_DZP) - Ee(ion_sp_DZP)
    if r_opt.get('Ee') is not None and r_sp.get('Ee') is not None:
        derived['Edef_red_DZP'] = _make_unit_dict_no_ev(r_opt['Ee'] - r_sp['Ee'])

    if o_opt.get('Ee') is not None and o_sp.get('Ee') is not None:
        derived['Edef_ox_DZP'] = _make_unit_dict_no_ev(o_opt['Ee'] - o_sp['Ee'])

    # Edef en TZ2P (si les SP TZ2P existent sur les deux geometries)
    # reduced_tz2p = TZ2P SP sur geom red opt, reduced_sp est DZP...
    # On n'a pas de TZ2P SP sur geom neutre en charge -1
    # Mais on peut utiliser Edef_TZ2P = Ee(reduced_tz2p) - Ee(reduced_sp_tz2p_on_neutral)
    # Ce dernier n'existe pas dans les fichiers. On utilise donc DZP pour Edef.

    # G_composite = Ee(TZ2P) + [G(DZP) - Ee(DZP)]
    g_composite = {}
    if n_tz2p.get('Ee') is not None and n_dzp.get('G') is not None and n_dzp.get('Ee') is not None:
        g_composite['neutral'] = float(n_tz2p['Ee'] + (n_dzp['G'] - n_dzp['Ee']))

    if r_tz2p.get('Ee') is not None and r_opt.get('G') is not None and r_opt.get('Ee') is not None:
        g_composite['reduced'] = float(r_tz2p['Ee'] + (r_opt['G'] - r_opt['Ee']))

    if o_tz2p.get('Ee') is not None and o_opt.get('G') is not None and o_opt.get('Ee') is not None:
        g_composite['oxidized'] = float(o_tz2p['Ee'] + (o_opt['G'] - o_opt['Ee']))

    if g_composite:
        derived['G_composite'] = g_composite

    # Delta G composite
    if 'neutral' in g_composite and 'reduced' in g_composite:
        dg = g_composite['reduced'] - g_composite['neutral']
        derived['delta_G_red'] = _make_unit_dict_no_ev(dg)

    if 'neutral' in g_composite and 'oxidized' in g_composite:
        dg = g_composite['oxidized'] - g_composite['neutral']
        derived['delta_G_ox'] = _make_unit_dict_no_ev(dg)

    # DDU et TDS (corrections DZP)
    if n_dzp.get('U') is not None and n_dzp.get('Ee') is not None:
        u_ee_n = n_dzp['U'] - n_dzp['Ee']
        if r_opt.get('U') is not None and r_opt.get('Ee') is not None:
            u_ee_r = r_opt['U'] - r_opt['Ee']
            derived['ddU_red'] = _make_unit_dict_no_ev(u_ee_r - u_ee_n)
        if o_opt.get('U') is not None and o_opt.get('Ee') is not None:
            u_ee_o = o_opt['U'] - o_opt['Ee']
            derived['ddU_ox'] = _make_unit_dict_no_ev(u_ee_o - u_ee_n)

    if n_dzp.get('TS') is not None:
        if r_opt.get('TS') is not None:
            derived['TdS_red'] = _make_unit_dict_no_ev(r_opt['TS'] - n_dzp['TS'])
        if o_opt.get('TS') is not None:
            derived['TdS_ox'] = _make_unit_dict_no_ev(o_opt['TS'] - n_dzp['TS'])

    return derived


# =============================================================================
# TRAITEMENT DES MOLECULES
# =============================================================================

def process_molecule_prediction(workdir, mol_name, out_dir, solvent=""):
    """
    Traite une molecule en mode prediction (3 SP/conformere).

    Returns:
        list of dict: donnees JSON par conformere
    """
    print(f"\n  Traitement de {mol_name} (prediction)...")

    conf_jobs = find_prediction_conformer_jobs(workdir, mol_name)
    if not conf_jobs:
        print(f"  ! Aucun job predict-dg trouve pour {mol_name}")
        return []

    weights = read_boltzmann_weights(workdir)
    all_conformer_data = []

    for conf_num in sorted(conf_jobs.keys()):
        jobs = conf_jobs[conf_num]
        if not all(k in jobs for k in PREDICTION_STATES):
            print(f"  ! Conformere {conf_num} incomplet, ignore")
            continue

        # Lire les atomes
        xyz_path = find_xyz_file(workdir, conf_num)
        atoms = read_atoms_from_xyz(xyz_path) if xyz_path else None

        # Extraire les donnees des 3 etats
        raw_states = {}
        for state in PREDICTION_STATES:
            comment = f"{mol_name} conf{conf_num} {state}"
            state_data = extract_all_from_job(jobs[state], atoms=atoms, state_label=comment)
            if state_data is None:
                break
            raw_states[state] = state_data

        if len(raw_states) != 3:
            print(f"  ! Extraction incomplete pour conformere {conf_num}")
            continue

        if atoms is None:
            atoms = read_atoms_from_rkf(os.path.join(jobs['neutral_sp'], 'adf.rkf'))

        n_atoms = len(atoms)

        # Construire le JSON (etats tries : neutre, reduit, oxyde)
        states_json = {}
        for state_key, state_data in raw_states.items():
            states_json[state_key] = build_state_json(state_data, atoms, n_atoms)
        states_json = _sort_states(states_json)

        derived = compute_prediction_derived(raw_states, atoms)
        aggregated = build_aggregated(states_json, atoms)

        conformer_json = {
            'metadata': {
                'molecule': mol_name,
                'conformer': conf_num,
                'weight': weights.get(conf_num, 0.0),
                'n_atoms': n_atoms,
                'solvent': solvent,
                'workflow': 'prediction',
            },
            'states': states_json,
            'derived': derived,
            'aggregated': aggregated,
        }

        # Ecrire le fichier
        filename = f"{mol_name}_conf{conf_num}_prediction.json"
        write_json(conformer_json, os.path.join(out_dir, filename))
        all_conformer_data.append(conformer_json)

    print(f"  + {len(all_conformer_data)} conformere(s) traite(s) pour {mol_name}")
    return all_conformer_data


def process_molecule_dft(workdir, mol_name, out_dir, solvent=""):
    """
    Traite une molecule en mode DFT (8 RKF/conformere via symlinks).

    Returns:
        list of dict: donnees JSON par conformere
    """
    print(f"\n  Traitement de {mol_name} (DFT)...")

    conf_jobs = find_dft_conformer_jobs(workdir, mol_name)
    if not conf_jobs:
        print(f"  ! Aucun job DFT trouve pour {mol_name}")
        return []

    weights = read_boltzmann_weights(workdir)
    all_conformer_data = []

    for conf_num in sorted(conf_jobs.keys()):
        jobs = conf_jobs[conf_num]

        # Verifier qu'on a au minimum le neutre DZP
        if 'neutral_dzp' not in jobs:
            print(f"  ! Conformere {conf_num}: neutral_dzp manquant, ignore")
            continue

        # Lire les atomes depuis XYZ ou depuis le RKF neutre
        xyz_path = find_xyz_file(workdir, conf_num)
        atoms = read_atoms_from_xyz(xyz_path) if xyz_path else None
        if atoms is None:
            atoms = read_atoms_from_rkf(jobs['neutral_dzp'])

        n_atoms = len(atoms)

        # Extraire les donnees de chaque etat disponible
        raw_states = {}
        for state_key, rkf_path in jobs.items():
            comment = f"{mol_name} conf{conf_num} {state_key}"
            state_data = extract_from_rkf_file(rkf_path, atoms=atoms, state_label=comment)
            if state_data is not None:
                raw_states[state_key] = state_data

        if not raw_states:
            print(f"  ! Aucune extraction reussie pour conformere {conf_num}")
            continue

        # Construire le JSON (etats tries : neutre, reduit, oxyde)
        states_json = {}
        for state_key, state_data in raw_states.items():
            states_json[state_key] = build_state_json(state_data, atoms, n_atoms)
        states_json = _sort_states(states_json)

        derived = compute_dft_derived(raw_states)
        aggregated = build_aggregated(states_json, atoms)

        conformer_json = {
            'metadata': {
                'molecule': mol_name,
                'conformer': conf_num,
                'weight': weights.get(conf_num, 0.0),
                'n_atoms': n_atoms,
                'solvent': solvent,
                'workflow': 'DFT',
            },
            'states': states_json,
            'derived': derived,
            'aggregated': aggregated,
        }

        filename = f"{mol_name}_conf{conf_num}_DFT.json"
        write_json(conformer_json, os.path.join(out_dir, filename))
        all_conformer_data.append(conformer_json)

    print(f"  + {len(all_conformer_data)} conformere(s) traite(s) pour {mol_name}")
    return all_conformer_data


# =============================================================================
# MOYENNE DE BOLTZMANN (JSON)
# =============================================================================

def _deep_boltzmann_average(data_list, weights):
    """
    Moyenne de Boltzmann recursive sur des dicts imbriques.

    Gere scalaires, listes (arrays), et dicts imbriques.
    """
    if not data_list:
        return None

    first = data_list[0]

    # Strings (XYZ, etc.) : pas de moyenne, ignorer
    if isinstance(first, str):
        return None

    # Scalaire
    if isinstance(first, (int, float)):
        values = []
        ws = []
        for d, w in zip(data_list, weights):
            if d is not None and isinstance(d, (int, float)):
                values.append(d)
                ws.append(w)
        if not values:
            return None
        ws = np.array(ws)
        ws = ws / ws.sum() if ws.sum() > 0 else ws
        return float(np.average(values, weights=ws))

    # Liste (array 1D ou 2D numerique)
    if isinstance(first, list):
        # Ignorer les listes de dicts (ex: modes normaux) - pas moyennables
        if first and isinstance(first[0], dict):
            return None
        valid = [(d, w) for d, w in zip(data_list, weights) if d is not None and isinstance(d, list)]
        if not valid:
            return None
        try:
            arrays = [np.array(d, dtype=float) for d, _ in valid]
        except (ValueError, TypeError):
            return None
        ws = np.array([w for _, w in valid])
        ws = ws / ws.sum() if ws.sum() > 0 else ws
        # Verifier que toutes les shapes sont identiques
        shape = arrays[0].shape
        if not all(a.shape == shape for a in arrays):
            return None
        result = np.zeros(shape)
        for a, w in zip(arrays, ws):
            result += a * w
        return result.tolist()

    # Dict
    if isinstance(first, dict):
        result = {}
        all_keys = set()
        for d in data_list:
            if isinstance(d, dict):
                all_keys.update(d.keys())

        for key in all_keys:
            sub_values = []
            sub_weights = []
            for d, w in zip(data_list, weights):
                if isinstance(d, dict) and key in d:
                    sub_values.append(d[key])
                    sub_weights.append(w)

            if sub_values:
                avg = _deep_boltzmann_average(sub_values, sub_weights)
                if avg is not None:
                    result[key] = avg

        return result if result else None

    return None


def boltzmann_average_json(conformer_data_list):
    """
    Calcule la moyenne de Boltzmann sur les conformeres.

    Args:
        conformer_data_list: list of dict (JSON structures par conformere)

    Returns:
        dict: JSON structure moyennee
    """
    if not conformer_data_list:
        return {}

    weights = [d['metadata']['weight'] for d in conformer_data_list]
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)

    mol_name = conformer_data_list[0]['metadata']['molecule']
    workflow = conformer_data_list[0]['metadata']['workflow']
    solvent = conformer_data_list[0]['metadata'].get('solvent', '')

    summary = {
        'metadata': {
            'molecule': mol_name,
            'n_conformers': len(conformer_data_list),
            'n_atoms': conformer_data_list[0]['metadata']['n_atoms'],
            'solvent': solvent,
            'workflow': workflow,
            'conformer_weights': {
                d['metadata']['conformer']: d['metadata']['weight']
                for d in conformer_data_list
            },
        },
    }

    # Moyenner les etats
    all_state_keys = set()
    for d in conformer_data_list:
        all_state_keys.update(d.get('states', {}).keys())

    states_avg = {}
    for state_key in sorted(all_state_keys):
        state_list = [d['states'].get(state_key) for d in conformer_data_list]
        avg = _deep_boltzmann_average(state_list, weights)
        if avg:
            states_avg[state_key] = avg
    summary['states'] = states_avg

    # Moyenner les derives
    derived_list = [d.get('derived', {}) for d in conformer_data_list]
    derived_avg = _deep_boltzmann_average(derived_list, weights)
    summary['derived'] = derived_avg or {}

    # Moyenner les agreges
    agg_list = [d.get('aggregated', {}) for d in conformer_data_list]
    agg_avg = _deep_boltzmann_average(agg_list, weights)
    summary['aggregated'] = agg_avg or {}

    return summary


# =============================================================================
# ECRITURE JSON
# =============================================================================

def write_json(data, output_path):
    """Ecrit un dict en JSON compact (arrays inline, structure indentee)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(compact_json_dumps(data))


# =============================================================================
# MAIN
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extraction des donnees RKF vers JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  amspython extract_predict_dg.py --solvent Acetonitrile --data-dir data
  amspython extract_predict_dg.py --solvent Acetonitrile --data-dir 1844043
  amspython extract_predict_dg.py --discover path/to/adf.rkf
        """
    )

    parser.add_argument('--solvent', type=str,
                        help="Nom du solvant (ex: Acetonitrile)")
    parser.add_argument('--data-dir', type=str, default='.',
                        help="Repertoire contenant les *_workdir/ (defaut: .)")
    parser.add_argument('--output-dir', type=str, default='dataset',
                        help="Repertoire racine de sortie (defaut: dataset)")
    parser.add_argument('--discover', type=str,
                        help="Mode decouverte: liste les sections/cles d'un RKF")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Mode decouverte
    if args.discover:
        print_discovery(args.discover)
        ams_path = os.path.join(os.path.dirname(args.discover), "ams.rkf")
        if os.path.exists(ams_path):
            print_discovery(ams_path)
        return

    if not args.solvent:
        print("Erreur: --solvent requis (sauf en mode --discover)")
        sys.exit(1)

    print("=" * 80)
    print("EXTRACTION RKF -> JSON")
    print(f"  Solvant: {args.solvent}")
    print(f"  Donnees: {args.data_dir}")
    print(f"  Sortie:  {args.output_dir}")
    print("=" * 80)

    # Auto-decouverte des molecules
    molecules = discover_molecules(args.data_dir)
    if not molecules:
        print(f"\n  ! Aucun *_workdir/ trouve dans {args.data_dir}")
        sys.exit(1)

    print(f"\nMolecules detectees: {len(molecules)}")

    n_dft = 0
    n_pred = 0
    n_failed = 0

    for mol_name, workdir in molecules:
        # Detecter le type de workflow
        wf_type = detect_workflow_type(workdir, mol_name)
        if wf_type is None:
            print(f"\n  ! {mol_name}: workflow non reconnu, ignore")
            n_failed += 1
            continue

        # Creer le repertoire de sortie
        out_dir = os.path.join(args.output_dir, args.solvent, mol_name)
        os.makedirs(out_dir, exist_ok=True)

        # Traiter selon le type
        if wf_type == 'DFT':
            conformer_data = process_molecule_dft(workdir, mol_name, out_dir, solvent=args.solvent)
            n_dft += 1
        else:
            conformer_data = process_molecule_prediction(workdir, mol_name, out_dir, solvent=args.solvent)
            n_pred += 1

        # Ecrire le summary Boltzmann
        if conformer_data:
            summary = boltzmann_average_json(conformer_data)
            summary_name = f"{mol_name}_summary_{wf_type}.json"
            write_json(summary, os.path.join(out_dir, summary_name))

    print(f"\n{'='*80}")
    print(f"EXTRACTION TERMINEE")
    print(f"  Molecules DFT:        {n_dft}")
    print(f"  Molecules prediction: {n_pred}")
    print(f"  Echecs:               {n_failed}")
    print(f"  Sortie dans:          {args.output_dir}/{args.solvent}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
