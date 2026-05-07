# -*- coding: utf-8 -*-
"""
Module cluster-continuum pour solvants protiques.

Pour les solvants protiques (Water, Methanol), le COSMO implicite seul est
insuffisant. Ce module ajoute des molecules de solvant explicites autour du
solute aux sites de solvatation (paires libres, H labiles), puis le COSMO
implicite est applique par-dessus lors du calcul DFT.

Workflow :
1. Extraction des charges xTB (Mulliken) pour identifier les sites reactifs
2. Identification des sites de solvatation (paires libres + H labiles)
3. Placement geometrique des molecules de solvant
4. Fusion solute + solvant en un seul systeme
"""

import os
import numpy as np

from scm.plams import Molecule, Atom, AMSJob, Settings, KFReader

from .utils import DefaultJobManager


# ============================================================================
# CONSTANTES
# ============================================================================

CLUSTER_CONTINUUM_SOLVENTS = {"Water", "Methanol"}

LONE_PAIR_COUNTS = {
    'N': 1, 'O': 2, 'F': 3,
    'P': 1, 'S': 2, 'Cl': 3,
}

LABILE_H_CHARGE_THRESHOLD = 0.10  # e (Mulliken)

H_BOND_DISTANCE = 1.9  # Angstrom

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75,
}

SOLVENT_SMILES = {"Water": "O", "Methanol": "CO"}


# ============================================================================
# FONCTIONS PUBLIQUES
# ============================================================================

def needs_cluster_continuum(solvent_name):
    """
    Determine si le solvant necessite un traitement cluster-continuum.

    Args:
        solvent_name (str): Nom du solvant COSMO

    Returns:
        bool: True si le solvant est protique (Water, Methanol)
    """
    return solvent_name in CLUSTER_CONTINUUM_SOLVENTS


def get_solvent_template(solvent_name):
    """
    Retourne une molecule PLAMS template pour le solvant.

    Water : O a l'origine, 2 H en geometrie equilibre (angle 104.5 deg, d(OH)=0.9572 A).
    Methanol : O a l'origine, H(hydroxyl) et CH3 positionnes.

    Args:
        solvent_name (str): "Water" ou "Methanol"

    Returns:
        Molecule: Molecule PLAMS du solvant
    """
    mol = Molecule()

    if solvent_name == "Water":
        # O a l'origine, H1 et H2 en geometrie equilibre
        angle = np.radians(104.52 / 2)
        d_oh = 0.9572
        mol.add_atom(Atom(symbol='O', coords=(0.0, 0.0, 0.0)))
        mol.add_atom(Atom(symbol='H', coords=(d_oh * np.sin(angle), d_oh * np.cos(angle), 0.0)))
        mol.add_atom(Atom(symbol='H', coords=(-d_oh * np.sin(angle), d_oh * np.cos(angle), 0.0)))

    elif solvent_name == "Methanol":
        # O a l'origine, H hydroxyl et C positionnes
        d_oh = 0.9451
        d_co = 1.4300
        angle_coh = np.radians(108.5)
        mol.add_atom(Atom(symbol='O', coords=(0.0, 0.0, 0.0)))
        mol.add_atom(Atom(symbol='H', coords=(d_oh, 0.0, 0.0)))
        mol.add_atom(Atom(symbol='C', coords=(-d_co * np.cos(np.pi - angle_coh),
                                                d_co * np.sin(np.pi - angle_coh), 0.0)))
        # 3 H du CH3 (positions approximatives tetraedriques)
        c_pos = np.array([mol.atoms[2].x, mol.atoms[2].y, mol.atoms[2].z])
        d_ch = 1.09
        tet_angle = np.radians(109.5)
        for i in range(3):
            phi = 2 * np.pi * i / 3
            hx = c_pos[0] + d_ch * np.sin(tet_angle) * np.cos(phi)
            hy = c_pos[1] + d_ch * np.cos(tet_angle)
            hz = c_pos[2] + d_ch * np.sin(tet_angle) * np.sin(phi)
            mol.add_atom(Atom(symbol='H', coords=(hx, hy, hz)))
    else:
        raise ValueError(f"Solvant non supporte pour cluster-continuum: {solvent_name}")

    return mol


def extract_xtb_charges(molecule, calc_dir):
    """
    Lance un calcul GFN1-xTB SinglePoint et extrait les charges Mulliken.

    Args:
        molecule: Molecule PLAMS
        calc_dir (str): Repertoire pour le calcul

    Returns:
        list[float]: Charges Mulliken par atome
    """
    s = Settings()
    s.input.ams.Task = "SinglePoint"
    s.input.DFTB.Model = "GFN1-xTB"

    charges_dir = os.path.join(calc_dir, "00_charges")
    os.makedirs(charges_dir, exist_ok=True)

    with DefaultJobManager(charges_dir):
        job = AMSJob(molecule=molecule, settings=s, name="xtb_charges")
        job.run()

    if not job.ok():
        raise RuntimeError("Echec du calcul xTB pour extraction des charges")

    # Essayer plusieurs cles possibles pour les charges
    try:
        kf = KFReader(job.results.rkfpath(file='dftb'))
        charges = list(kf.read("Charges", "Mulliken Gross Charges"))
        return charges
    except Exception:
        pass

    try:
        kf = KFReader(job.results.rkfpath(file='ams'))
        charges = list(kf.read("AMSResults", "Charges"))
        return charges
    except Exception:
        pass

    # Dernier recours : methode PLAMS
    try:
        charges = job.results.get_charges()
        return list(charges)
    except Exception:
        raise RuntimeError(
            "Impossible de lire les charges xTB. "
            "Verifiez les cles RKF disponibles."
        )


def count_lone_pairs(molecule):
    """
    Compte les paires libres par atome.

    Args:
        molecule: Molecule PLAMS

    Returns:
        dict[int, int]: {atom_index (0-based): nombre_paires_libres}
    """
    result = {}
    for i, atom in enumerate(molecule.atoms):
        n_lp = LONE_PAIR_COUNTS.get(atom.symbol, 0)
        if n_lp > 0:
            result[i] = n_lp
    return result


def detect_labile_hydrogens(molecule, charges, threshold=LABILE_H_CHARGE_THRESHOLD):
    """
    Detecte les hydrogenes labiles (charge Mulliken >= seuil).

    Args:
        molecule: Molecule PLAMS
        charges (list[float]): Charges Mulliken par atome
        threshold (float): Seuil de charge (defaut: 0.10 e)

    Returns:
        list[int]: Indices (0-based) des H labiles
    """
    labile = []
    for i, atom in enumerate(molecule.atoms):
        if atom.symbol == 'H' and charges[i] >= threshold:
            labile.append(i)
    return labile


def identify_solvation_sites(molecule, charges):
    """
    Identifie les sites de solvatation (paires libres + H labiles).

    Args:
        molecule: Molecule PLAMS
        charges (list[float]): Charges Mulliken par atome

    Returns:
        list[dict]: Sites de solvatation avec cles:
            - atom_idx (int): Index de l'atome (0-based)
            - type (str): 'lone_pair' ou 'labile_h'
            - n_solvent (int): Nombre de molecules de solvant a placer
            - coords (ndarray): Coordonnees de l'atome
    """
    sites = []

    # Paires libres
    lp_dict = count_lone_pairs(molecule)
    for atom_idx, n_lp in lp_dict.items():
        atom = molecule.atoms[atom_idx]
        sites.append({
            'atom_idx': atom_idx,
            'type': 'lone_pair',
            'n_solvent': n_lp,
            'coords': np.array([atom.x, atom.y, atom.z]),
        })

    # H labiles
    labile_h = detect_labile_hydrogens(molecule, charges)
    for atom_idx in labile_h:
        atom = molecule.atoms[atom_idx]
        sites.append({
            'atom_idx': atom_idx,
            'type': 'labile_h',
            'n_solvent': 1,
            'coords': np.array([atom.x, atom.y, atom.z]),
        })

    return sites


def compute_placement_positions(com, target_pos, n_molecules, distance=H_BOND_DISTANCE):
    """
    Calcule les positions de placement des molecules de solvant.

    n=1 : placement sur l'axe COM->atome, a distance de l'atome.
    n>1 : distribution sur cone (demi-angle 60 deg), equidistants angulairement.

    Args:
        com (ndarray): Centre de masse du solute
        target_pos (ndarray): Position de l'atome cible
        n_molecules (int): Nombre de molecules a placer
        distance (float): Distance de placement (A)

    Returns:
        list[ndarray]: Positions de placement
    """
    axis = target_pos - com
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / norm

    if n_molecules == 1:
        return [target_pos + axis * distance]

    # Vecteur perpendiculaire
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp1 = np.cross(axis, ref)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    # Distribution sur cone (demi-angle 60 deg)
    cone_angle = np.radians(60)
    positions = []
    for i in range(n_molecules):
        phi = 2 * np.pi * i / n_molecules
        direction = (np.cos(cone_angle) * axis +
                     np.sin(cone_angle) * (np.cos(phi) * perp1 + np.sin(phi) * perp2))
        direction /= np.linalg.norm(direction)
        positions.append(target_pos + direction * distance)

    return positions


def orient_solvent_at_site(solvent_template, site_type, placement_pos, target_pos):
    """
    Oriente une molecule de solvant au site de solvatation.

    - lone_pair : H du solvant pointe vers l'atome cible (H-bond donor)
    - labile_h : O du solvant pointe vers l'atome cible (H-bond acceptor)

    Utilise la rotation de Rodrigues pour aligner le vecteur actif.

    Args:
        solvent_template: Molecule PLAMS template du solvant
        site_type (str): 'lone_pair' ou 'labile_h'
        placement_pos (ndarray): Position ou placer le solvant
        target_pos (ndarray): Position de l'atome cible dans le solute

    Returns:
        Molecule: Molecule de solvant orientee et positionnee
    """
    mol = solvent_template.copy()

    # Extraire les coordonnees
    coords = np.array([[a.x, a.y, a.z] for a in mol.atoms])

    # Determiner l'atome actif et le vecteur de reference
    o_idx = 0  # O est toujours le premier atome dans nos templates
    h_idx = 1  # Premier H

    if site_type == 'lone_pair':
        # H du solvant doit pointer vers le solute -> H en direction de target
        active_idx = h_idx
        ref_vec = coords[active_idx] - coords[o_idx]  # O->H dans le template
    else:
        # O du solvant doit pointer vers le solute -> O en direction de target
        active_idx = o_idx
        ref_vec = coords[o_idx] - coords[h_idx]  # H->O dans le template
        # Pour labile_h on veut que O soit le plus proche de target

    # Vecteur cible : de placement_pos vers target_pos
    target_vec = target_pos - placement_pos
    target_norm = np.linalg.norm(target_vec)
    if target_norm < 1e-10:
        target_vec = np.array([0.0, 0.0, 1.0])
    else:
        target_vec = target_vec / target_norm

    # Normaliser le vecteur de reference
    ref_norm = np.linalg.norm(ref_vec)
    if ref_norm < 1e-10:
        ref_vec = np.array([0.0, 0.0, 1.0])
    else:
        ref_vec = ref_vec / ref_norm

    # Rotation de Rodrigues pour aligner ref_vec sur target_vec
    rotation_matrix = _rodrigues_rotation(ref_vec, target_vec)

    # Appliquer la rotation autour de l'origine du template (O)
    origin = coords[o_idx].copy()
    rotated_coords = np.array([
        rotation_matrix @ (c - origin) + origin for c in coords
    ])

    # Translater pour que l'atome actif soit a placement_pos
    if site_type == 'lone_pair':
        # Placer le H du solvant pres du target, O plus loin
        # L'atome actif (H) doit etre le plus proche de target
        translation = placement_pos - rotated_coords[active_idx]
    else:
        # Placer le O du solvant pres du target
        translation = placement_pos - rotated_coords[active_idx]

    rotated_coords += translation

    # Mettre a jour les coordonnees
    for i, atom in enumerate(mol.atoms):
        atom.x, atom.y, atom.z = rotated_coords[i]

    return mol


def build_cluster_continuum_system(solute, charges, solvent_name):
    """
    Construit le systeme solute + solvant explicite.

    Orchestrateur principal : identifie les sites, place les molecules
    de solvant, et fusionne tout en une seule Molecule PLAMS.

    Args:
        solute: Molecule PLAMS du solute
        charges (list[float]): Charges Mulliken du solute
        solvent_name (str): Nom du solvant ("Water" ou "Methanol")

    Returns:
        Molecule: Systeme combine solute + N molecules de solvant
    """
    # Centre de masse du solute
    coords = np.array([[a.x, a.y, a.z] for a in solute.atoms])
    com = coords.mean(axis=0)

    # Identifier les sites de solvatation
    sites = identify_solvation_sites(solute, charges)

    # Template du solvant
    template = get_solvent_template(solvent_name)

    # Placer les molecules de solvant
    solvent_molecules = []
    total_solvent = 0

    for site in sites:
        positions = compute_placement_positions(
            com, site['coords'], site['n_solvent']
        )
        for pos in positions:
            oriented = orient_solvent_at_site(
                template, site['type'], pos, site['coords']
            )
            solvent_molecules.append(oriented)
            total_solvent += 1

    print(f"  Sites de solvatation: {len(sites)}")
    print(f"  Molecules de solvant ajoutees: {total_solvent}")

    # Fusionner
    return merge_molecules([solute] + solvent_molecules)


def merge_molecules(mol_list):
    """
    Fusionne plusieurs Molecule PLAMS en une seule.

    Args:
        mol_list (list[Molecule]): Liste de molecules a fusionner

    Returns:
        Molecule: Molecule combinee
    """
    combined = Molecule()
    for mol in mol_list:
        for atom in mol.atoms:
            combined.add_atom(Atom(symbol=atom.symbol, coords=(atom.x, atom.y, atom.z)))
    return combined


# ============================================================================
# FONCTIONS PRIVEES
# ============================================================================

def _rodrigues_rotation(v_from, v_to):
    """
    Calcule la matrice de rotation (Rodrigues) pour aligner v_from sur v_to.

    Args:
        v_from (ndarray): Vecteur source (normalise)
        v_to (ndarray): Vecteur cible (normalise)

    Returns:
        ndarray: Matrice de rotation 3x3
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    sin_angle = np.linalg.norm(cross)

    # Cas vecteurs paralleles
    if sin_angle < 1e-10:
        if dot > 0:
            return np.eye(3)
        else:
            # Rotation de 180 deg autour d'un axe perpendiculaire
            ref = np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            perp = np.cross(v_from, ref)
            perp /= np.linalg.norm(perp)
            # Matrice de rotation de 180 deg autour de perp
            return 2 * np.outer(perp, perp) - np.eye(3)

    # Matrice antisymetrique
    k = cross / sin_angle
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])

    # Formule de Rodrigues: R = I + sin(theta)*K + (1-cos(theta))*K^2
    R = np.eye(3) + sin_angle * K + (1 - dot) * (K @ K)
    return R
