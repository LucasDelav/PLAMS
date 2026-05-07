"""
Descripteurs étendus extraits des fichiers RKF.

Ce module ajoute des descripteurs supplémentaires non liés aux fréquences:
- Constantes de force et masses réduites (propriétés vibrationnelles avancées)
- Surface et volume COSMO (propriétés moléculaires)
- Moment dipolaire et quadrupolaire (polarité)
- HOMO/LUMO et gap (propriétés électroniques)
- Composantes d'énergie (structure électronique)
"""

from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np


def extract_extended_data(rkf_path) -> Dict[str, Any]:
    """
    Extrait les données étendues depuis un fichier RKF.

    Args:
        rkf_path: Chemin vers le fichier adf.rkf

    Returns:
        Dict contenant toutes les données étendues extraites
    """
    try:
        from scm.plams import KFFile
        kf = KFFile(str(rkf_path))

        data = {'success': True}

        # === Vibrations avancées ===
        try:
            data['force_constants'] = np.array(kf.read('Vibrations', 'ForceConstants'), dtype=float)
        except:
            data['force_constants'] = None

        try:
            data['reduced_masses'] = np.array(kf.read('Vibrations', 'ReducedMasses'), dtype=float)
        except:
            data['reduced_masses'] = None

        # === COSMO ===
        try:
            data['cosmo_area'] = float(kf.read('COSMO', 'Area'))
        except:
            data['cosmo_area'] = None

        try:
            data['cosmo_volume'] = float(kf.read('COSMO', 'Volume'))
        except:
            data['cosmo_volume'] = None

        # === Propriétés électroniques ===
        try:
            data['homo'] = float(kf.read('Properties', 'HOMO'))
        except:
            data['homo'] = None

        try:
            data['lumo'] = float(kf.read('Properties', 'LUMO'))
        except:
            data['lumo'] = None

        try:
            dipole = kf.read('Properties', 'Dipole')
            data['dipole'] = np.array(dipole, dtype=float)
        except:
            data['dipole'] = None

        try:
            quadrupole = kf.read('Properties', 'Quadrupole')
            data['quadrupole'] = np.array(quadrupole, dtype=float)
        except:
            data['quadrupole'] = None

        # === Composantes d'énergie ===
        try:
            data['kinetic_energy'] = float(kf.read('Energy', 'Kinetic Energy'))
        except:
            data['kinetic_energy'] = None

        try:
            data['xc_energy'] = float(kf.read('Energy', 'XC Energy'))
        except:
            data['xc_energy'] = None

        try:
            data['bond_energy'] = float(kf.read('Energy', 'Bond Energy'))
        except:
            data['bond_energy'] = None

        # === Géométrie ===
        try:
            data['n_atoms'] = int(kf.read('Geometry', 'nr of atoms'))
        except:
            data['n_atoms'] = None

        try:
            coords = np.array(kf.read('Geometry', 'xyz')).reshape(-1, 3)
            data['coords'] = coords
        except:
            data['coords'] = None

        return data

    except Exception as e:
        return {'success': False, 'error': str(e)}


def compute_force_constant_descriptors(force_constants: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Calcule les descripteurs basés sur les constantes de force.

    Les constantes de force (en mdyn/Å) reflètent la rigidité des liaisons.
    Plus elles sont grandes, plus les liaisons sont rigides.

    Args:
        force_constants: Tableau des constantes de force

    Returns:
        Dict de descripteurs
    """
    if force_constants is None or len(force_constants) == 0:
        return {
            'fc_max': np.nan,
            'fc_min': np.nan,
            'fc_mean': np.nan,
            'fc_std': np.nan,
            'fc_median': np.nan,
            'fc_sum': np.nan,
            'fc_q25': np.nan,
            'fc_q75': np.nan,
            'fc_iqr': np.nan,
            # Constantes de force par plage
            'fc_mean_soft': np.nan,  # fc < 0.1
            'fc_mean_medium': np.nan,  # 0.1 <= fc < 0.4
            'fc_mean_stiff': np.nan,  # fc >= 0.4
            'n_soft_modes': np.nan,
            'n_stiff_modes': np.nan,
            'ratio_stiff_soft': np.nan,
        }

    fc = force_constants

    # Plages de constantes de force
    soft_mask = fc < 0.1
    medium_mask = (fc >= 0.1) & (fc < 0.4)
    stiff_mask = fc >= 0.4

    soft_fc = fc[soft_mask]
    medium_fc = fc[medium_mask]
    stiff_fc = fc[stiff_mask]

    n_soft = len(soft_fc)
    n_stiff = len(stiff_fc)

    return {
        'fc_max': float(np.max(fc)),
        'fc_min': float(np.min(fc)),
        'fc_mean': float(np.mean(fc)),
        'fc_std': float(np.std(fc)),
        'fc_median': float(np.median(fc)),
        'fc_sum': float(np.sum(fc)),
        'fc_q25': float(np.percentile(fc, 25)),
        'fc_q75': float(np.percentile(fc, 75)),
        'fc_iqr': float(np.percentile(fc, 75) - np.percentile(fc, 25)),
        'fc_mean_soft': float(np.mean(soft_fc)) if len(soft_fc) > 0 else np.nan,
        'fc_mean_medium': float(np.mean(medium_fc)) if len(medium_fc) > 0 else np.nan,
        'fc_mean_stiff': float(np.mean(stiff_fc)) if len(stiff_fc) > 0 else np.nan,
        'n_soft_modes': n_soft,
        'n_stiff_modes': n_stiff,
        'ratio_stiff_soft': float(n_stiff) / float(n_soft) if n_soft > 0 else np.nan,
    }


def compute_reduced_mass_descriptors(reduced_masses: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Calcule les descripteurs basés sur les masses réduites.

    Les masses réduites (en u.m.a.) caractérisent les atomes impliqués dans chaque mode.
    - Masses faibles (~1) : mouvements H
    - Masses élevées (>3) : mouvements d'atomes lourds (C, N, O)

    Args:
        reduced_masses: Tableau des masses réduites

    Returns:
        Dict de descripteurs
    """
    if reduced_masses is None or len(reduced_masses) == 0:
        return {
            'rm_max': np.nan,
            'rm_min': np.nan,
            'rm_mean': np.nan,
            'rm_std': np.nan,
            'rm_median': np.nan,
            # Modes par type d'atome impliqué
            'n_h_dominated': np.nan,  # rm < 2 (hydrogènes)
            'n_heavy_dominated': np.nan,  # rm >= 3 (atomes lourds)
            'ratio_heavy_h': np.nan,
            'rm_weighted_freq_proxy': np.nan,  # proxy pour la fréquence moyenne
        }

    rm = reduced_masses

    h_mask = rm < 2.0
    heavy_mask = rm >= 3.0

    n_h = np.sum(h_mask)
    n_heavy = np.sum(heavy_mask)

    return {
        'rm_max': float(np.max(rm)),
        'rm_min': float(np.min(rm)),
        'rm_mean': float(np.mean(rm)),
        'rm_std': float(np.std(rm)),
        'rm_median': float(np.median(rm)),
        'n_h_dominated': int(n_h),
        'n_heavy_dominated': int(n_heavy),
        'ratio_heavy_h': float(n_heavy) / float(n_h) if n_h > 0 else np.nan,
        # Proxy: sqrt(fc/rm) ~ fréquence, donc 1/sqrt(rm) corrèle avec fréquence
        'rm_inverse_sqrt_mean': float(np.mean(1.0 / np.sqrt(rm))) if np.all(rm > 0) else np.nan,
    }


def compute_cosmo_descriptors(area: Optional[float], volume: Optional[float],
                               n_atoms: Optional[int]) -> Dict[str, float]:
    """
    Calcule les descripteurs COSMO (surface et volume moléculaires).

    Args:
        area: Surface moléculaire COSMO (Å²)
        volume: Volume moléculaire COSMO (Å³)
        n_atoms: Nombre d'atomes

    Returns:
        Dict de descripteurs
    """
    result = {
        'cosmo_area': area if area is not None else np.nan,
        'cosmo_volume': volume if volume is not None else np.nan,
    }

    # Rapport surface/volume (indicateur de compacité)
    if area is not None and volume is not None and volume > 0:
        result['surface_volume_ratio'] = area / volume
        # Sphéricité (pour une sphère parfaite = 1)
        # Sphericity = (pi^(1/3) * (6V)^(2/3)) / A
        result['sphericity'] = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / area
    else:
        result['surface_volume_ratio'] = np.nan
        result['sphericity'] = np.nan

    # Descripteurs normalisés par atome
    if n_atoms is not None and n_atoms > 0:
        result['area_per_atom'] = area / n_atoms if area is not None else np.nan
        result['volume_per_atom'] = volume / n_atoms if volume is not None else np.nan
    else:
        result['area_per_atom'] = np.nan
        result['volume_per_atom'] = np.nan

    return result


def compute_electronic_descriptors(homo: Optional[float], lumo: Optional[float],
                                    dipole: Optional[np.ndarray],
                                    quadrupole: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Calcule les descripteurs électroniques.

    Args:
        homo: Énergie HOMO (Hartree)
        lumo: Énergie LUMO (Hartree)
        dipole: Moment dipolaire [x, y, z] (Debye)
        quadrupole: Moment quadrupolaire (6 composantes)

    Returns:
        Dict de descripteurs
    """
    result = {}

    # HOMO/LUMO
    result['homo_ev'] = homo * 27.2114 if homo is not None else np.nan  # Hartree -> eV
    result['lumo_ev'] = lumo * 27.2114 if lumo is not None else np.nan

    if homo is not None and lumo is not None:
        result['homo_lumo_gap'] = (lumo - homo) * 27.2114  # eV
        # Indicateurs de réactivité
        result['chemical_potential'] = (homo + lumo) / 2 * 27.2114  # eV
        result['hardness'] = (lumo - homo) / 2 * 27.2114  # eV
        if homo != lumo:
            result['electrophilicity'] = result['chemical_potential']**2 / (2 * result['hardness'])
        else:
            result['electrophilicity'] = np.nan
    else:
        result['homo_lumo_gap'] = np.nan
        result['chemical_potential'] = np.nan
        result['hardness'] = np.nan
        result['electrophilicity'] = np.nan

    # Dipole
    if dipole is not None and len(dipole) == 3:
        result['dipole_x'] = float(dipole[0])
        result['dipole_y'] = float(dipole[1])
        result['dipole_z'] = float(dipole[2])
        result['dipole_magnitude'] = float(np.linalg.norm(dipole))
    else:
        result['dipole_x'] = np.nan
        result['dipole_y'] = np.nan
        result['dipole_z'] = np.nan
        result['dipole_magnitude'] = np.nan

    # Quadrupole
    if quadrupole is not None and len(quadrupole) >= 6:
        result['quadrupole_trace'] = float(quadrupole[0] + quadrupole[3] + quadrupole[5])
        result['quadrupole_anisotropy'] = float(np.std(quadrupole[:6]))
    else:
        result['quadrupole_trace'] = np.nan
        result['quadrupole_anisotropy'] = np.nan

    return result


def compute_energy_descriptors(kinetic: Optional[float], xc: Optional[float],
                                bond: Optional[float]) -> Dict[str, float]:
    """
    Calcule les descripteurs basés sur les composantes d'énergie.

    Args:
        kinetic: Énergie cinétique (Hartree)
        xc: Énergie d'échange-corrélation (Hartree)
        bond: Énergie de liaison (Hartree)

    Returns:
        Dict de descripteurs
    """
    # Conversion Hartree -> kJ/mol
    EH_TO_KJ = 2625.5

    result = {
        'kinetic_energy_kj': kinetic * EH_TO_KJ if kinetic is not None else np.nan,
        'xc_energy_kj': xc * EH_TO_KJ if xc is not None else np.nan,
        'bond_energy_kj': bond * EH_TO_KJ if bond is not None else np.nan,
    }

    # Ratios
    if kinetic is not None and xc is not None and xc != 0:
        result['kinetic_xc_ratio'] = abs(kinetic / xc)
    else:
        result['kinetic_xc_ratio'] = np.nan

    return result


def compute_geometry_descriptors(coords: Optional[np.ndarray], n_atoms: Optional[int]) -> Dict[str, float]:
    """
    Calcule les descripteurs géométriques basés sur les coordonnées.

    Args:
        coords: Coordonnées atomiques (N x 3)
        n_atoms: Nombre d'atomes

    Returns:
        Dict de descripteurs
    """
    if coords is None or len(coords) == 0:
        return {
            'mol_extent_x': np.nan,
            'mol_extent_y': np.nan,
            'mol_extent_z': np.nan,
            'mol_extent_max': np.nan,
            'radius_of_gyration': np.nan,
        }

    # Étendue moléculaire
    extent_x = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    extent_y = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    extent_z = float(np.max(coords[:, 2]) - np.min(coords[:, 2]))

    # Rayon de giration
    center = np.mean(coords, axis=0)
    distances_sq = np.sum((coords - center)**2, axis=1)
    rg = float(np.sqrt(np.mean(distances_sq)))

    return {
        'mol_extent_x': extent_x,
        'mol_extent_y': extent_y,
        'mol_extent_z': extent_z,
        'mol_extent_max': max(extent_x, extent_y, extent_z),
        'radius_of_gyration': rg,
    }


def compute_all_extended_descriptors(rkf_path) -> Dict[str, float]:
    """
    Calcule tous les descripteurs étendus depuis un fichier RKF.

    Args:
        rkf_path: Chemin vers le fichier adf.rkf

    Returns:
        Dict contenant tous les descripteurs étendus
    """
    # Extraire les données brutes
    data = extract_extended_data(rkf_path)

    if not data.get('success', False):
        return {}

    descriptors = {}

    # Constantes de force
    descriptors.update(compute_force_constant_descriptors(data.get('force_constants')))

    # Masses réduites
    descriptors.update(compute_reduced_mass_descriptors(data.get('reduced_masses')))

    # COSMO
    descriptors.update(compute_cosmo_descriptors(
        data.get('cosmo_area'),
        data.get('cosmo_volume'),
        data.get('n_atoms')
    ))

    # Électronique
    descriptors.update(compute_electronic_descriptors(
        data.get('homo'),
        data.get('lumo'),
        data.get('dipole'),
        data.get('quadrupole')
    ))

    # Énergie
    descriptors.update(compute_energy_descriptors(
        data.get('kinetic_energy'),
        data.get('xc_energy'),
        data.get('bond_energy')
    ))

    # Géométrie
    descriptors.update(compute_geometry_descriptors(
        data.get('coords'),
        data.get('n_atoms')
    ))

    return descriptors


def get_extended_descriptor_names() -> list:
    """
    Retourne la liste des noms de tous les descripteurs étendus.

    Returns:
        list: Liste des noms de descripteurs
    """
    return [
        # Constantes de force
        'fc_max', 'fc_min', 'fc_mean', 'fc_std', 'fc_median', 'fc_sum',
        'fc_q25', 'fc_q75', 'fc_iqr',
        'fc_mean_soft', 'fc_mean_medium', 'fc_mean_stiff',
        'n_soft_modes', 'n_stiff_modes', 'ratio_stiff_soft',
        # Masses réduites
        'rm_max', 'rm_min', 'rm_mean', 'rm_std', 'rm_median',
        'n_h_dominated', 'n_heavy_dominated', 'ratio_heavy_h', 'rm_inverse_sqrt_mean',
        # COSMO
        'cosmo_area', 'cosmo_volume', 'surface_volume_ratio', 'sphericity',
        'area_per_atom', 'volume_per_atom',
        # Électronique
        'homo_ev', 'lumo_ev', 'homo_lumo_gap',
        'chemical_potential', 'hardness', 'electrophilicity',
        'dipole_x', 'dipole_y', 'dipole_z', 'dipole_magnitude',
        'quadrupole_trace', 'quadrupole_anisotropy',
        # Énergie
        'kinetic_energy_kj', 'xc_energy_kj', 'bond_energy_kj', 'kinetic_xc_ratio',
        # Géométrie
        'mol_extent_x', 'mol_extent_y', 'mol_extent_z', 'mol_extent_max',
        'radius_of_gyration',
    ]
