"""
Extraction des proprietes de densite electronique locale depuis les fichiers RKF QTAIM.

Ce module fournit des fonctions pour localiser les fichiers RKF des calculs QTAIM neutres
et extraire les proprietes de densite (rho aux noyaux, Bader Laplacian, Ts, ELF, etc.)
ainsi que les donnees des points critiques (BCP, RCP).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from .common import print_warning


def find_qtaim_neutral_rkf_files(workdir, molecule_name: str) -> Dict[int, Path]:
    """
    Localise les fichiers adf.rkf des calculs QTAIM neutres dans un workdir.

    Les fichiers sont recherches dans:
    <workdir>/qtaim/calculations/plams_workdir/<mol>_conf_N_neutral/adf.rkf

    Args:
        workdir (str or Path): Repertoire de travail principal de la molecule
        molecule_name (str): Nom de la molecule

    Returns:
        Dict[int, Path]: Dictionnaire {conf_num: chemin_rkf}
    """
    workdir = Path(workdir)
    rkf_files = {}

    plams_workdir = workdir / "qtaim" / "calculations" / "plams_workdir"

    if not plams_workdir.exists():
        print_warning(f"Repertoire plams_workdir QTAIM non trouve: {plams_workdir}")
        return rkf_files

    # Pattern: <mol>_conf_<N>_neutral ou <mol>_conf_<N>_neutral.XXX
    neutral_pattern = re.compile(
        rf"^{re.escape(molecule_name)}_conf_(\d+)_neutral(?:\.\d+)?$"
    )

    for job_dir in plams_workdir.iterdir():
        if not job_dir.is_dir():
            continue

        match = neutral_pattern.match(job_dir.name)
        if match:
            conf_num = int(match.group(1))
            rkf_path = job_dir / "adf.rkf"

            if rkf_path.exists():
                # Garder la version la plus recente si plusieurs suffixes
                if conf_num not in rkf_files or job_dir.name > rkf_files[conf_num].parent.name:
                    rkf_files[conf_num] = rkf_path

    return rkf_files


def _get_atom_order_mapping(kf) -> List[int]:
    """
    Obtient le mapping ordre interne -> ordre input depuis un KFFile.

    AMS reordonne les atomes en interne (typiquement par type).
    Le tableau 'atom order index' a la forme [n_atoms, 2]:
      - Premiere moitie: input_idx -> internal_idx
      - Seconde moitie: internal_idx -> input_idx

    On utilise la seconde moitie pour convertir les proprietes stockees
    en ordre interne vers l'ordre d'entree.

    Args:
        kf: KFFile ou KFReader object

    Returns:
        List[int]: mapping[internal_idx] = input_idx
    """
    try:
        atom_order = kf.read("Geometry", "atom order index")
        n_atoms = len(atom_order) // 2
        return [int(x) for x in atom_order[n_atoms:2 * n_atoms]]
    except KeyError:
        return []


def _read_property_in_input_order(kf, section: str, variable: str,
                                  atom_order: List[int]) -> Optional[Dict[int, float]]:
    """
    Lit une propriete atomique et la convertit en ordre input.

    Args:
        kf: KFFile ou KFReader
        section: Section RKF (ex: 'Properties')
        variable: Nom de la variable (ex: 'Bader atomic charges')
        atom_order: Mapping internal_idx -> input_idx

    Returns:
        Dict[int, float]: {input_idx: value} ou None si non disponible
    """
    try:
        values = kf.read(section, variable)
        if not hasattr(values, '__len__'):
            values = [values]

        result = {}
        for internal_idx, val in enumerate(values):
            if atom_order and internal_idx < len(atom_order):
                input_idx = atom_order[internal_idx]
            else:
                input_idx = internal_idx + 1
            result[input_idx] = float(val)

        return result
    except KeyError:
        return None


def _extract_cp_data(kf, atom_order: List[int]) -> Dict[str, Any]:
    """
    Extrait les donnees des points critiques (BCP et RCP).

    AMS CP code encoding:
      - Code 1 = NCP (nuclear critical point, rank=3, signature=-3)
      - Code 3 = BCP (bond critical point, rank=3, signature=-1)
      - Code 4 = RCP (ring critical point, rank=3, signature=+1)

    BP atoms format (column-major):
      [atom1_1..atom1_N, atom2_1..atom2_N, bcp_cp_idx_1..bcp_cp_idx_N]
      where atom1/atom2 are NCP indices (1-indexed, internal order)
      and bcp_cp_idx is the CP index of the BCP.

    Args:
        kf: KFFile ou KFReader
        atom_order: Mapping internal_idx -> input_idx

    Returns:
        Dict avec n_cp, bcp_densities, bcp_atoms, rcp_densities
    """
    result = {
        'n_cp': 0,
        'bcp_densities': [],
        'bcp_atoms': [],
        'rcp_densities': [],
    }

    try:
        cp_codes = kf.read("Properties", "CP code number for (Rank,Signatu")
        if not hasattr(cp_codes, '__len__'):
            cp_codes = [cp_codes]

        n_cp = len(cp_codes)
        result['n_cp'] = n_cp

        # Lire les densites aux CP
        try:
            cp_densities = kf.read("Properties", "CP density at")
            if not hasattr(cp_densities, '__len__'):
                cp_densities = [cp_densities]
        except KeyError:
            cp_densities = None

        # Classer par type de CP
        # Code 3 = BCP (not 2), Code 4 = RCP
        bcp_indices = []
        rcp_indices = []
        for i, code in enumerate(cp_codes):
            code = int(code)
            if code == 3:  # BCP
                bcp_indices.append(i)
            elif code == 4:  # RCP
                rcp_indices.append(i)

        # Densites aux BCP et RCP
        if cp_densities is not None:
            for idx in bcp_indices:
                if idx < len(cp_densities):
                    result['bcp_densities'].append(float(cp_densities[idx]))
            for idx in rcp_indices:
                if idx < len(cp_densities):
                    result['rcp_densities'].append(float(cp_densities[idx]))

        # Paires d'atomes pour les BCP via bond paths
        # BP atoms is column-major: [atom1_1..atom1_N, atom2_1..atom2_N, bcp_idx_1..bcp_idx_N]
        # atom1/atom2 are NCP indices (1-indexed) which map to internal atom indices
        try:
            bp_atoms_raw = kf.read("Properties", "BP atoms")
            if not hasattr(bp_atoms_raw, '__len__'):
                bp_atoms_raw = [bp_atoms_raw]

            bp_num = int(kf.read("Properties", "BP number of"))

            if len(bp_atoms_raw) >= 2 * bp_num:
                atom1_list = [int(x) for x in bp_atoms_raw[:bp_num]]
                atom2_list = [int(x) for x in bp_atoms_raw[bp_num:2 * bp_num]]

                for i in range(bp_num):
                    # NCP indices are 1-indexed internal atom indices
                    a1_internal = atom1_list[i] - 1  # Convert to 0-indexed
                    a2_internal = atom2_list[i] - 1

                    # Convert to input order
                    if atom_order:
                        a1 = atom_order[a1_internal] if a1_internal < len(atom_order) else a1_internal + 1
                        a2 = atom_order[a2_internal] if a2_internal < len(atom_order) else a2_internal + 1
                    else:
                        a1 = a1_internal + 1
                        a2 = a2_internal + 1

                    result['bcp_atoms'].append((a1, a2))

        except KeyError:
            pass

    except KeyError:
        pass

    return result


def extract_density_properties(rkf_path) -> Dict[str, Any]:
    """
    Extrait TOUTES les proprietes de densite locale depuis un RKF QTAIM.

    Lit depuis la section 'Properties' (ordre INTERNE AMS) puis convertit
    en ordre INPUT via 'Geometry'/'atom order index'.

    Args:
        rkf_path (str or Path): Chemin vers le fichier adf.rkf

    Returns:
        Dict contenant:
            'success': bool
            'n_atoms': int
            'atom_symbols': List[str] (depuis Geometry/labels, ordre input)
            'atom_numbers': List[int] (Z, ordre input)
            'rho_nuclei': Dict[int, float] (Electron Density at Nuclei)
            'laplacian': Dict[int, float] (Bader Laplacian)
            'Ts': Dict[int, float] (Bader Ts)
            'ELF': Dict[int, float] (Bader ELF)
            'Tc': Dict[int, float] (Bader Tc)
            'bader_charges': Dict[int, float] (Bader atomic charges)
            'esp_nuclei': Dict[int, float] (Electrostatic Pot. at Nuclei)
            'cp_data': Dict (points critiques)
    """
    result = {
        'success': False,
        'n_atoms': 0,
        'atom_symbols': [],
        'atom_numbers': [],
        'rho_nuclei': {},
        'laplacian': {},
        'Ts': {},
        'ELF': {},
        'Tc': {},
        'bader_charges': {},
        'esp_nuclei': {},
        'cp_data': {},
    }

    try:
        from scm.plams import KFFile
        kf = KFFile(str(rkf_path))

        # Nombre d'atomes
        n_atoms = int(kf.read("Geometry", "nr of atoms"))
        result['n_atoms'] = n_atoms

        # Mapping ordre interne -> ordre input
        atom_order = _get_atom_order_mapping(kf)

        # Labels des atomes (en ordre input)
        try:
            # atomtype: chaine avec les types uniques (ex: "O C H")
            atomtype = kf.read("Geometry", "atomtype")
            if isinstance(atomtype, str):
                atomtype = atomtype.split()

            # cum nr of atoms: [0, n_type1, n_type1+n_type2, ..., n_total]
            cum_nr = kf.read("Geometry", "cum nr of atoms")
            if not hasattr(cum_nr, '__len__'):
                cum_nr = [cum_nr]

            # Construire la liste des symboles en ordre interne
            symbols_internal = []
            for i in range(len(atomtype)):
                start = int(cum_nr[i])
                end = int(cum_nr[i + 1])
                count = end - start
                sym = atomtype[i].strip() if i < len(atomtype) else 'X'
                symbols_internal.extend([sym] * count)

            # Convertir en ordre input
            symbols_input = [''] * n_atoms
            for internal_idx in range(n_atoms):
                if atom_order and internal_idx < len(atom_order):
                    input_idx = atom_order[internal_idx]
                    symbols_input[input_idx - 1] = symbols_internal[internal_idx]
                else:
                    symbols_input[internal_idx] = symbols_internal[internal_idx]

            result['atom_symbols'] = symbols_input

            # Numeros atomiques
            Z_MAP = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
                'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
                'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'Br': 35, 'I': 53,
            }
            result['atom_numbers'] = [Z_MAP.get(s, 0) for s in symbols_input]

        except (KeyError, Exception) as e:
            print_warning(f"Impossible de lire les labels atomiques: {e}")

        # Extraire les proprietes de densite (toutes en ordre interne)
        properties = {
            'rho_nuclei': 'Electron Density at Nuclei',
            'laplacian': 'Bader Laplacian',
            'Ts': 'Bader Ts',
            'ELF': 'Bader ELF',
            'Tc': 'Bader Tc',
            'bader_charges': 'Bader atomic charges',
        }

        for key, var_name in properties.items():
            data = _read_property_in_input_order(kf, "Properties", var_name, atom_order)
            if data is not None:
                result[key] = data

        # Potentiel electrostatique aux noyaux
        esp = _read_property_in_input_order(
            kf, "Properties", "Electrostatic Pot.at Nuclei", atom_order
        )
        if esp is not None:
            result['esp_nuclei'] = esp

        # Points critiques
        result['cp_data'] = _extract_cp_data(kf, atom_order)

        result['success'] = True

    except Exception as e:
        print_warning(f"Erreur extraction densite depuis {rkf_path}: {e}")

    return result
