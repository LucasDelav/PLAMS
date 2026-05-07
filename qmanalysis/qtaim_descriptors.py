"""
Calcul des descripteurs de densite electronique a partir des proprietes QTAIM.

Ce module transforme les proprietes de densite brutes (rho, laplacian, Ts, ELF,
charges Bader, BCP/RCP) en descripteurs numeriques utilisables pour les correlations
avec EA/IP verticaux.
"""

from typing import Dict, List, Any

import numpy as np



def _safe_stats(values: List[float], prefix: str) -> Dict[str, float]:
    """
    Calcule les statistiques (mean, std, min, max, sum) pour une liste de valeurs.

    Args:
        values: Liste de valeurs numeriques
        prefix: Prefixe pour les noms de descripteurs

    Returns:
        Dict des statistiques
    """
    if not values:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_min': np.nan,
            f'{prefix}_max': np.nan,
            f'{prefix}_sum': np.nan,
        }

    arr = np.array(values, dtype=float)
    return {
        f'{prefix}_mean': float(np.mean(arr)),
        f'{prefix}_std': float(np.std(arr)),
        f'{prefix}_min': float(np.min(arr)),
        f'{prefix}_max': float(np.max(arr)),
        f'{prefix}_sum': float(np.sum(arr)),
    }


def compute_density_descriptors(density_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcule ~50 descripteurs a partir des proprietes de densite extraites.

    Categories de descripteurs:
    1. Statistiques globales (rho, laplacian, Ts, ELF, charge)
    2. Par type d'atome (C, O, H, X=heteroatomes)
    3. BCP (bond critical points)
    4. RCP (ring critical points)
    5. Ratios derives (inspires de Politzer)

    Args:
        density_data: Resultat de extract_density_properties()

    Returns:
        Dict[str, float]: Descripteurs numeriques
    """
    desc = {}

    if not density_data.get('success', False):
        return desc

    symbols = density_data.get('atom_symbols', [])
    n_atoms = density_data.get('n_atoms', 0)

    # Identifier les types d'atomes par indice input (1-indexed)
    heavy_indices = []
    c_indices = []
    o_indices = []
    h_indices = []
    x_indices = []  # Heteroatomes hors C/O/H

    for i, sym in enumerate(symbols):
        input_idx = i + 1
        if sym == 'H':
            h_indices.append(input_idx)
        elif sym == 'C':
            c_indices.append(input_idx)
            heavy_indices.append(input_idx)
        elif sym == 'O':
            o_indices.append(input_idx)
            heavy_indices.append(input_idx)
        else:
            x_indices.append(input_idx)
            heavy_indices.append(input_idx)

    # Helper: extraire les valeurs pour un sous-ensemble d'indices
    def _get_values(data_dict: Dict[int, float], indices: List[int]) -> List[float]:
        return [data_dict[idx] for idx in indices if idx in data_dict]

    # ========================================================================
    # Categorie 1: Statistiques globales
    # ========================================================================
    all_indices = list(range(1, n_atoms + 1))

    property_map = {
        'rho': density_data.get('rho_nuclei', {}),
        'laplacian': density_data.get('laplacian', {}),
        'Ts': density_data.get('Ts', {}),
        'ELF': density_data.get('ELF', {}),
        'charge': density_data.get('bader_charges', {}),
    }

    for prop_name, prop_data in property_map.items():
        if not prop_data:
            continue

        # Tous les atomes
        all_vals = _get_values(prop_data, all_indices)
        desc.update(_safe_stats(all_vals, f'{prop_name}_all'))

        # Atomes lourds (non-H)
        heavy_vals = _get_values(prop_data, heavy_indices)
        stats = _safe_stats(heavy_vals, f'{prop_name}_heavy')
        desc[f'{prop_name}_heavy_mean'] = stats[f'{prop_name}_heavy_mean']
        desc[f'{prop_name}_heavy_std'] = stats[f'{prop_name}_heavy_std']

    # ========================================================================
    # Categorie 2: Par type d'atome
    # ========================================================================
    for prop_name, prop_data in property_map.items():
        if not prop_data:
            continue

        # Carbones
        c_vals = _get_values(prop_data, c_indices)
        if c_vals:
            desc[f'{prop_name}_C_mean'] = float(np.mean(c_vals))
            desc[f'{prop_name}_C_sum'] = float(np.sum(c_vals))
        else:
            desc[f'{prop_name}_C_mean'] = np.nan
            desc[f'{prop_name}_C_sum'] = np.nan

        # Oxygenes
        o_vals = _get_values(prop_data, o_indices)
        if o_vals:
            desc[f'{prop_name}_O_mean'] = float(np.mean(o_vals))
            desc[f'{prop_name}_O_sum'] = float(np.sum(o_vals))
        else:
            desc[f'{prop_name}_O_mean'] = np.nan
            desc[f'{prop_name}_O_sum'] = np.nan

        # Hydrogenes
        h_vals = _get_values(prop_data, h_indices)
        if h_vals:
            desc[f'{prop_name}_H_mean'] = float(np.mean(h_vals))
            desc[f'{prop_name}_H_sum'] = float(np.sum(h_vals))
        else:
            desc[f'{prop_name}_H_mean'] = np.nan
            desc[f'{prop_name}_H_sum'] = np.nan

        # Heteroatomes (F, Cl, Br, S, etc.)
        x_vals = _get_values(prop_data, x_indices)
        if x_vals:
            desc[f'{prop_name}_X_mean'] = float(np.mean(x_vals))
        else:
            desc[f'{prop_name}_X_mean'] = np.nan

    # ========================================================================
    # Categorie 3: BCP (bond critical points)
    # ========================================================================
    cp_data = density_data.get('cp_data', {})
    bcp_densities = cp_data.get('bcp_densities', [])
    bcp_atoms = cp_data.get('bcp_atoms', [])

    if bcp_densities:
        arr = np.array(bcp_densities)
        desc['rho_BCP_mean'] = float(np.mean(arr))
        desc['rho_BCP_min'] = float(np.min(arr))
        desc['rho_BCP_max'] = float(np.max(arr))
        desc['rho_BCP_std'] = float(np.std(arr))
        desc['n_BCP'] = float(len(arr))
    else:
        desc['rho_BCP_mean'] = np.nan
        desc['rho_BCP_min'] = np.nan
        desc['rho_BCP_max'] = np.nan
        desc['rho_BCP_std'] = np.nan
        desc['n_BCP'] = 0.0

    # BCP par type de liaison
    cc_bcp = []
    co_bcp = []
    ch_bcp = []

    c_set = set(c_indices)
    o_set = set(o_indices)
    h_set = set(h_indices)

    for i, (a1, a2) in enumerate(bcp_atoms):
        if i >= len(bcp_densities):
            break
        rho_bcp = bcp_densities[i]

        if a1 in c_set and a2 in c_set:
            cc_bcp.append(rho_bcp)
        elif (a1 in c_set and a2 in o_set) or (a1 in o_set and a2 in c_set):
            co_bcp.append(rho_bcp)
        elif (a1 in c_set and a2 in h_set) or (a1 in h_set and a2 in c_set):
            ch_bcp.append(rho_bcp)

    desc['rho_BCP_CC_mean'] = float(np.mean(cc_bcp)) if cc_bcp else np.nan
    desc['rho_BCP_CO_mean'] = float(np.mean(co_bcp)) if co_bcp else np.nan
    desc['rho_BCP_CH_mean'] = float(np.mean(ch_bcp)) if ch_bcp else np.nan

    # ========================================================================
    # Categorie 4: RCP (ring critical points)
    # ========================================================================
    rcp_densities = cp_data.get('rcp_densities', [])
    if rcp_densities:
        arr = np.array(rcp_densities)
        desc['rho_RCP_mean'] = float(np.mean(arr))
        desc['rho_RCP_max'] = float(np.max(arr))
    else:
        desc['rho_RCP_mean'] = np.nan
        desc['rho_RCP_max'] = np.nan

    # ========================================================================
    # Categorie 5: Ratios derives (inspires de Politzer)
    # ========================================================================
    rho_data = density_data.get('rho_nuclei', {})
    ts_data = density_data.get('Ts', {})
    lap_data = density_data.get('laplacian', {})
    elf_data = density_data.get('ELF', {})
    charge_data = density_data.get('bader_charges', {})

    # Ts/rho pour atomes lourds -> proxy de la temperature cinetique locale
    ts_over_rho = []
    for idx in heavy_indices:
        if idx in ts_data and idx in rho_data and rho_data[idx] != 0:
            ts_over_rho.append(ts_data[idx] / rho_data[idx])
    desc['Ts_over_rho_heavy_mean'] = float(np.mean(ts_over_rho)) if ts_over_rho else np.nan

    # Laplacian/rho pour atomes lourds -> concentration de charge
    lap_over_rho = []
    for idx in heavy_indices:
        if idx in lap_data and idx in rho_data and rho_data[idx] != 0:
            lap_over_rho.append(lap_data[idx] / rho_data[idx])
    desc['laplacian_over_rho_heavy_mean'] = float(np.mean(lap_over_rho)) if lap_over_rho else np.nan

    # ELF moyen pour atomes lourds -> localisation electronique
    elf_heavy = _get_values(elf_data, heavy_indices)
    desc['ELF_heavy_mean'] = float(np.mean(elf_heavy)) if elf_heavy else np.nan

    # Span et somme des charges absolues -> polarisation
    if charge_data:
        charges = list(charge_data.values())
        desc['charge_span'] = float(max(charges) - min(charges))
        desc['sum_abs_charge'] = float(sum(abs(c) for c in charges))
    else:
        desc['charge_span'] = np.nan
        desc['sum_abs_charge'] = np.nan

    # ESP au noyau (si disponible)
    esp_data = density_data.get('esp_nuclei', {})
    if esp_data:
        esp_heavy = _get_values(esp_data, heavy_indices)
        if esp_heavy:
            desc['esp_heavy_mean'] = float(np.mean(esp_heavy))
            desc['esp_heavy_std'] = float(np.std(esp_heavy))
            desc['esp_heavy_min'] = float(np.min(esp_heavy))
            desc['esp_heavy_max'] = float(np.max(esp_heavy))
            desc['esp_span'] = float(max(esp_heavy) - min(esp_heavy))
        else:
            for k in ['esp_heavy_mean', 'esp_heavy_std', 'esp_heavy_min',
                       'esp_heavy_max', 'esp_span']:
                desc[k] = np.nan
    else:
        for k in ['esp_heavy_mean', 'esp_heavy_std', 'esp_heavy_min',
                   'esp_heavy_max', 'esp_span']:
            desc[k] = np.nan

    return desc


def get_density_descriptor_names() -> List[str]:
    """
    Retourne la liste ordonnee des noms de descripteurs de densite.

    Returns:
        List[str]: Noms des descripteurs dans un ordre fixe
    """
    names = []

    # Categorie 1: Statistiques globales
    for prop in ['rho', 'laplacian', 'Ts', 'ELF', 'charge']:
        for stat in ['mean', 'std', 'min', 'max', 'sum']:
            names.append(f'{prop}_all_{stat}')
        names.append(f'{prop}_heavy_mean')
        names.append(f'{prop}_heavy_std')

    # Categorie 2: Par type d'atome
    for prop in ['rho', 'laplacian', 'Ts', 'ELF', 'charge']:
        for atom_type in ['C', 'O', 'H', 'X']:
            if atom_type == 'X':
                names.append(f'{prop}_{atom_type}_mean')
            else:
                names.append(f'{prop}_{atom_type}_mean')
                names.append(f'{prop}_{atom_type}_sum')

    # Categorie 3: BCP
    names.extend([
        'rho_BCP_mean', 'rho_BCP_min', 'rho_BCP_max', 'rho_BCP_std', 'n_BCP',
        'rho_BCP_CC_mean', 'rho_BCP_CO_mean', 'rho_BCP_CH_mean',
    ])

    # Categorie 4: RCP
    names.extend(['rho_RCP_mean', 'rho_RCP_max'])

    # Categorie 5: Ratios derives
    names.extend([
        'Ts_over_rho_heavy_mean',
        'laplacian_over_rho_heavy_mean',
        # ELF_heavy_mean already in category 1
        'charge_span',
        'sum_abs_charge',
    ])

    # ESP
    names.extend([
        'esp_heavy_mean', 'esp_heavy_std', 'esp_heavy_min', 'esp_heavy_max',
        'esp_span',
    ])

    return names
