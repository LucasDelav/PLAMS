"""
Calcul de l'ALIE (Average Local Ionization Energy) de Politzer & Murray.

Ce module effectue le post-traitement DENSF sur les fichiers RKF neutres existants
pour calculer l'ALIE sur l'isosurface de densité 0.001 a.u., sans nouveau calcul SCF.

Pipeline :
  1. Extraction des métadonnées orbitales (occupations, énergies) depuis adf.rkf
  2. DENSF Phase 1 : densité SCF sur grille 3D → identification de l'isosurface
  3. Extraction des points de l'isosurface ρ = 0.001 a.u. (interpolation linéaire)
  4. DENSF Phase 2 : orbitales SCF + densité + potentiel coulombien aux points de surface
  5. Calcul de l'ALIE : Ī(r) = Σ_i [n_i × ψ_i(r)² × |ε_i|] / ρ(r)
  6. Extraction des descripteurs : Ī_S,min, V_S,min, etc.

Référence :
  Politzer, P.; Murray, J. S. "Average local ionization energy: A review"
  J. Mol. Model. 2018, 24, 1-10.
"""

import os
import subprocess
from typing import Dict, Any

import numpy as np

from .common import (
    EH_TO_EV, EH_TO_KJ,
    print_success, print_warning, print_error
)


# ============================================================================
# CONSTANTES
# ============================================================================

# Seuil d'isodensité pour la surface moléculaire (a.u.)
ISODENSITY_THRESHOLD = 0.001

# Conversion bohr <-> angstrom
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


# ============================================================================
# PHASE 0 : EXTRACTION DES MÉTADONNÉES ORBITALES
# ============================================================================

def extract_orbital_metadata(adf_rkf_path: str) -> Dict[str, Any]:
    """
    Extrait les occupations et énergies de toutes les orbitales occupées depuis adf.rkf.

    Conformément à la formulation originale de Politzer & Murray, toutes les
    orbitales occupées (coeur + valence) sont incluses dans le calcul ALIE.
    Les orbitales de coeur ont une amplitude négligeable à la surface
    d'isodensité 0.001 a.u. et n'affectent donc pas le résultat, mais les
    inclure est plus fidèle à la théorie.

    Args:
        adf_rkf_path: Chemin vers le fichier adf.rkf

    Returns:
        Dict avec:
            - occupations: np.ndarray des occupations de toutes les orbitales occupées
            - orbital_energies: np.ndarray des énergies en Hartree
            - orbital_indices: indices 1-based des orbitales occupées (pour DENSF)
            - n_occupied: nombre d'orbitales occupées
            - n_total: nombre total d'orbitales
    """
    from scm.plams import KFReader
    kf = KFReader(str(adf_rkf_path))

    occupations = np.array(kf.read("A", "froc_A"), dtype=float)
    orbital_energies = np.array(kf.read("A", "eps_A"), dtype=float)
    n_occupied = int(np.sum(occupations > 0))

    # Indices 1-based de toutes les orbitales occupées (pour DENSF ORBITALS)
    orbital_indices = list(range(1, n_occupied + 1))

    return {
        'occupations': occupations[:n_occupied],
        'orbital_energies': orbital_energies[:n_occupied],
        'n_occupied': n_occupied,
        'orbital_indices': orbital_indices,
        'n_total': len(occupations),
    }


# ============================================================================
# PHASE 1 : DENSF DENSITÉ SUR GRILLE 3D
# ============================================================================

def _find_densf_executable() -> str:
    """Trouve le chemin de l'exécutable densf."""
    # Essayer AMSBIN d'abord
    amsbin = os.environ.get('AMSBIN', '')
    if amsbin:
        candidate = os.path.join(amsbin, 'densf')
        if os.path.isfile(candidate):
            return candidate

    # Essayer dans le PATH
    import shutil
    path = shutil.which('densf')
    if path:
        return path

    raise FileNotFoundError(
        "densf non trouvé. Vérifiez que l'environnement AMS est chargé "
        "(source amsbashrc.sh)"
    )


def run_densf_density_grid(adf_rkf_path: str, work_dir: str,
                           grid_quality: str = "medium") -> str:
    """
    Lance DENSF pour calculer la densité SCF sur une grille 3D.

    Args:
        adf_rkf_path: Chemin vers adf.rkf (sera converti en absolu)
        work_dir: Répertoire de travail pour les fichiers temporaires
        grid_quality: Qualité de la grille ("coarse", "medium", "fine")

    Returns:
        Chemin vers le fichier TAPE41 résultant (format KF)
    """
    densf = _find_densf_executable()
    os.makedirs(work_dir, exist_ok=True)

    # Chemin absolu obligatoire car DENSF tourne avec cwd=work_dir
    abs_rkf = os.path.abspath(adf_rkf_path)

    # Syntaxe DENSF : GRID est un bloc qui nécessite END
    input_text = (
        f"ADFFILE {abs_rkf}\n"
        f"GRID {grid_quality}\n"
        f"END\n"
        f"DENSITY SCF\n"
        f"END INPUT\n"
    )

    input_file = os.path.join(work_dir, "densf_phase1.inp")
    with open(input_file, 'w') as f:
        f.write(input_text)

    # DENSF écrit TAPE41 dans le cwd
    result = subprocess.run(
        [densf],
        input=input_text,
        capture_output=True,
        text=True,
        cwd=work_dir,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DENSF Phase 1 a échoué (code={result.returncode}):\n"
            f"STDERR: {result.stderr[:500]}"
        )

    # DENSF écrit TAPE41 dans le cwd (work_dir)
    tape41 = os.path.join(work_dir, "TAPE41")
    if not os.path.exists(tape41):
        raise FileNotFoundError(
            f"Fichier TAPE41 non trouvé dans {work_dir} après DENSF Phase 1"
        )

    return tape41


# ============================================================================
# LECTURE DU FICHIER TAPE41 (GRILLE 3D)
# ============================================================================

def read_t41_density(t41_path: str) -> Dict[str, Any]:
    """
    Lit un fichier TAPE41 (format KF) pour extraire la densité et les paramètres de grille.

    Variables TAPE41 (vérifiées empiriquement sur AMS 2025.102) :
      Grid / Start_point         : [3] float — origine en bohr
      Grid / nr of points x      : int
      Grid / nr of points y      : int
      Grid / nr of points z      : int
      Grid / x-vector            : [3] float — vecteur PAS selon x (bohr/point)
      Grid / y-vector            : [3] float — vecteur PAS selon y (bohr/point)
      Grid / z-vector            : [3] float — vecteur PAS selon z (bohr/point)
      SCF / Density              : [nx*ny*nz] float — densité, ordre Fortran

    Args:
        t41_path: Chemin vers le fichier TAPE41

    Returns:
        Dict avec:
            - origin: np.ndarray [3] (bohr)
            - step_x, step_y, step_z: np.ndarray [3] — vecteurs pas par point (bohr)
            - shape: tuple (nx, ny, nz)
            - density: np.ndarray [nx, ny, nz] — densité en a.u.
    """
    from scm.plams import KFReader
    kf = KFReader(str(t41_path))

    # Paramètres de grille
    start_point = np.array(kf.read("Grid", "Start_point"), dtype=float)
    nx = int(kf.read("Grid", "nr of points x"))
    ny = int(kf.read("Grid", "nr of points y"))
    nz = int(kf.read("Grid", "nr of points z"))

    # Vecteurs PAS par point de grille (en bohr)
    step_x = np.array(kf.read("Grid", "x-vector"), dtype=float)
    step_y = np.array(kf.read("Grid", "y-vector"), dtype=float)
    step_z = np.array(kf.read("Grid", "z-vector"), dtype=float)

    # Densité SCF sur la grille (ordre Fortran : x varie le plus vite)
    density_flat = np.array(kf.read("SCF", "Density"), dtype=float)
    density = density_flat.reshape((nx, ny, nz), order='F')

    return {
        'origin': start_point,
        'step_x': step_x,
        'step_y': step_y,
        'step_z': step_z,
        'shape': (nx, ny, nz),
        'density': density,
    }


# ============================================================================
# PHASE 2 : EXTRACTION DE L'ISOSURFACE
# ============================================================================

def extract_isosurface_points(grid_data: Dict[str, Any],
                              threshold: float = ISODENSITY_THRESHOLD
                              ) -> np.ndarray:
    """
    Extrait les points de l'isosurface ρ = threshold par interpolation linéaire.

    Parcourt toutes les arêtes de la grille 3D et identifie celles où la densité
    traverse le seuil. Effectue une interpolation linéaire pour obtenir les
    coordonnées exactes du point de traversée.

    Args:
        grid_data: Dictionnaire retourné par read_t41_density()
        threshold: Seuil de densité (défaut: 0.001 a.u.)

    Returns:
        np.ndarray de shape (N, 3) — coordonnées en bohr des points de surface
    """
    origin = grid_data['origin']
    dx = grid_data['step_x']  # Vecteur pas (déjà par point)
    dy = grid_data['step_y']
    dz = grid_data['step_z']
    nx, ny, nz = grid_data['shape']
    density = grid_data['density']

    surface_points = []

    def _grid_to_coord(ix, iy, iz):
        return origin + ix * dx + iy * dy + iz * dz

    def _check_edge(i1, j1, k1, i2, j2, k2):
        """Vérifie si une arête traverse le seuil et ajoute le point interpolé."""
        rho1 = density[i1, j1, k1]
        rho2 = density[i2, j2, k2]

        if (rho1 - threshold) * (rho2 - threshold) < 0:
            # Interpolation linéaire
            t = (threshold - rho1) / (rho2 - rho1)
            p1 = _grid_to_coord(i1, j1, k1)
            p2 = _grid_to_coord(i2, j2, k2)
            surface_points.append(p1 + t * (p2 - p1))

    # Parcourir les arêtes selon x
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx - 1):
                _check_edge(ix, iy, iz, ix + 1, iy, iz)

    # Parcourir les arêtes selon y
    for iz in range(nz):
        for iy in range(ny - 1):
            for ix in range(nx):
                _check_edge(ix, iy, iz, ix, iy + 1, iz)

    # Parcourir les arêtes selon z
    for iz in range(nz - 1):
        for iy in range(ny):
            for ix in range(nx):
                _check_edge(ix, iy, iz, ix, iy, iz + 1)

    if not surface_points:
        print_warning(f"Aucun point de surface trouvé pour le seuil {threshold}")
        return np.array([]).reshape(0, 3)

    return np.array(surface_points)


# ============================================================================
# PHASE 3 : DENSF ORBITALES AUX POINTS DE SURFACE (GRID INLINE)
# ============================================================================

def run_densf_orbitals_at_points(
    adf_rkf_path: str,
    surface_points_bohr: np.ndarray,
    orbital_indices: list,
    work_dir: str,
) -> str:
    """
    Lance DENSF avec Grid Inline pour calculer orbitales + densité + potentiel
    aux points de surface spécifiés.

    Args:
        adf_rkf_path: Chemin vers adf.rkf (sera converti en absolu)
        surface_points_bohr: np.ndarray (N, 3) en bohr
        orbital_indices: Liste d'indices 1-based des orbitales occupées à calculer
        work_dir: Répertoire de travail

    Returns:
        Chemin vers le fichier de sortie TAPE41
    """
    densf = _find_densf_executable()
    os.makedirs(work_dir, exist_ok=True)

    abs_rkf = os.path.abspath(adf_rkf_path)

    # Construire l'input DENSF avec Grid Inline
    # Syntaxe vérifiée : GRID Inline ... END, ORBITALS SCF A ... END
    lines = [
        f"ADFFILE {abs_rkf}",
        "UNITS",
        "  Length bohr",
        "END",
        "GRID Inline",
    ]

    for pt in surface_points_bohr:
        lines.append(f"  {pt[0]:.10f} {pt[1]:.10f} {pt[2]:.10f}")

    lines.append("END")
    lines.append("ORBITALS SCF")

    # Découper la liste d'orbitales en lignes de max 20 indices
    # pour éviter l'erreur DENSF "Free block line would be truncated"
    # quand la ligne dépasse ~160 caractères (molécules avec atomes lourds)
    all_indices = list(orbital_indices)
    chunk_size = 20
    for i in range(0, len(all_indices), chunk_size):
        chunk = all_indices[i:i + chunk_size]
        lines.append("  A " + " ".join(str(idx) for idx in chunk))

    lines.append("END")
    lines.append("DENSITY SCF")
    lines.append("POTENTIAL coul SCF")
    lines.append("END INPUT")

    input_text = "\n".join(lines)

    input_file = os.path.join(work_dir, "densf_phase2.inp")
    with open(input_file, 'w') as f:
        f.write(input_text)

    result = subprocess.run(
        [densf],
        input=input_text,
        capture_output=True,
        text=True,
        cwd=work_dir,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DENSF Phase 2 a échoué (code={result.returncode}):\n"
            f"STDERR: {result.stderr[:500]}"
        )

    tape41 = os.path.join(work_dir, "TAPE41")
    if not os.path.exists(tape41):
        raise FileNotFoundError(
            f"Fichier TAPE41 non trouvé dans {work_dir} après DENSF Phase 2"
        )

    return tape41


def read_t41_inline_results(
    t41_path: str,
    orbital_indices: list,
    n_points: int,
) -> Dict[str, np.ndarray]:
    """
    Lit les résultats DENSF inline (orbitales, densité, potentiel) depuis TAPE41.

    Variables TAPE41 inline (vérifiées empiriquement) :
      SCF / Density    : [n_points] float — ρ(r)
      SCF / Coulpot    : [n_points] float — V_coul(r)
      SCF_A / <idx>    : [n_points] float — ψ_idx(r)
      ...

    Args:
        t41_path: Chemin vers le fichier TAPE41 Phase 2
        orbital_indices: Liste d'indices 1-based des orbitales demandées
        n_points: Nombre de points de surface

    Returns:
        Dict avec:
            - orbitals: np.ndarray (n_orbitals, n_points) — ψ_i(r) pour chaque orbitale
            - density: np.ndarray (n_points,) — ρ_total(r) (SCF, coeur+valence)
            - potential: np.ndarray (n_points,) — V_coul(r)
    """
    from scm.plams import KFReader
    kf = KFReader(str(t41_path))

    # Densité SCF aux points inline
    density = np.array(kf.read("SCF", "Density"), dtype=float)

    # Potentiel coulombien (variable = "Coulpot")
    try:
        potential = np.array(kf.read("SCF", "Coulpot"), dtype=float)
    except (KeyError, Exception):
        potential = np.zeros(n_points)

    # Orbitales demandées : section SCF_A, variable = str(idx)
    n_orb = len(orbital_indices)
    orbitals = np.zeros((n_orb, n_points))
    for j, idx in enumerate(orbital_indices):
        try:
            orb_data = np.array(kf.read("SCF_A", str(idx)), dtype=float)
            orbitals[j] = orb_data
        except (KeyError, Exception):
            print_warning(f"Impossible de lire l'orbitale {idx} dans SCF_A")

    return {
        'orbitals': orbitals,
        'density': density,
        'potential': potential,
    }


# ============================================================================
# PHASE 4 : CALCUL DE L'ALIE ET EXTRACTION DES DESCRIPTEURS
# ============================================================================

def compute_alie_at_points(
    orbitals: np.ndarray,
    occupations: np.ndarray,
    orbital_energies: np.ndarray,
) -> np.ndarray:
    """
    Calcule l'ALIE (Average Local Ionization Energy) à chaque point de surface.

    Ī(r) = Σ_i [n_i × ψ_i(r)² × |ε_i|] / ρ(r)

    où ρ(r) = Σ_i n_i × ψ_i(r)² est la densité reconstituée à partir de
    toutes les orbitales occupées fournies.

    Args:
        orbitals: (n_occupied, n_points) — valeurs des orbitales occupées
        occupations: (n_occupied,) — occupations des orbitales
        orbital_energies: (n_occupied,) — énergies en Hartree

    Returns:
        np.ndarray (n_points,) — ALIE en Hartree
    """
    n_orb = orbitals.shape[0]
    n_points = orbitals.shape[1]
    numerator = np.zeros(n_points)
    rho = np.zeros(n_points)

    for i in range(n_orb):
        n_i = occupations[i]
        eps_i = abs(orbital_energies[i])
        psi_sq = orbitals[i] ** 2
        numerator += n_i * psi_sq * eps_i
        rho += n_i * psi_sq

    # Diviser par la densité (avec protection division par zéro)
    alie = np.full(n_points, np.nan)
    mask = rho > 1e-12
    alie[mask] = numerator[mask] / rho[mask]

    return alie


def extract_alie_descriptors(
    alie: np.ndarray,
    potential: np.ndarray,
    density: np.ndarray,
    surface_points: np.ndarray,
) -> Dict[str, Any]:
    """
    Extrait les descripteurs ALIE et potentiel électrostatique de surface.

    Args:
        alie: (n_points,) — ALIE en Hartree
        potential: (n_points,) — Potentiel coulombien en Hartree
        density: (n_points,) — Densité aux points de surface
        surface_points: (n_points, 3) — Coordonnées en bohr

    Returns:
        Dict avec les descripteurs :
            - I_S_min_Ha, I_S_min_eV, I_S_min_kJ : minimum de l'ALIE
            - I_S_max_Ha, I_S_mean_Ha, I_S_std_Ha : stats ALIE
            - V_S_min_Ha, V_S_min_eV, V_S_min_kJ : minimum du potentiel
            - V_S_max_Ha, V_S_mean_Ha : stats potentiel
            - n_surface_points : nombre de points de surface
            - rho_surface_mean : densité moyenne sur la surface (diagnostic)
    """
    # Filtrer les NaN (points où ρ ≈ 0)
    valid = ~np.isnan(alie)
    alie_valid = alie[valid]
    pot_valid = potential[valid]
    rho_valid = density[valid]

    if len(alie_valid) == 0:
        print_error("Aucun point ALIE valide")
        return {
            'I_S_min_Ha': np.nan, 'I_S_min_eV': np.nan, 'I_S_min_kJ': np.nan,
            'V_S_min_Ha': np.nan, 'V_S_min_eV': np.nan, 'V_S_min_kJ': np.nan,
            'n_surface_points': 0,
        }

    i_min = np.min(alie_valid)
    i_max = np.max(alie_valid)
    i_mean = np.mean(alie_valid)
    i_std = np.std(alie_valid)

    v_min = np.min(pot_valid)
    v_max = np.max(pot_valid)
    v_mean = np.mean(pot_valid)

    return {
        # ALIE descriptors
        'I_S_min_Ha': float(i_min),
        'I_S_min_eV': float(i_min * EH_TO_EV),
        'I_S_min_kJ': float(i_min * EH_TO_KJ),
        'I_S_max_Ha': float(i_max),
        'I_S_max_eV': float(i_max * EH_TO_EV),
        'I_S_mean_Ha': float(i_mean),
        'I_S_mean_eV': float(i_mean * EH_TO_EV),
        'I_S_std_Ha': float(i_std),
        'I_S_variance_Ha': float(i_std ** 2),
        # Electrostatic potential descriptors
        'V_S_min_Ha': float(v_min),
        'V_S_min_eV': float(v_min * EH_TO_EV),
        'V_S_min_kJ': float(v_min * EH_TO_KJ),
        'V_S_max_Ha': float(v_max),
        'V_S_max_eV': float(v_max * EH_TO_EV),
        'V_S_mean_Ha': float(v_mean),
        'V_S_mean_eV': float(v_mean * EH_TO_EV),
        # Diagnostics
        'n_surface_points': int(np.sum(valid)),
        'rho_surface_mean': float(np.mean(rho_valid)),
    }


# ============================================================================
# FONCTION ORCHESTRATRICE
# ============================================================================

def compute_alie_for_conformer(
    adf_rkf_path: str,
    work_dir: str,
    name: str = "alie",
    grid_quality: str = "medium",
    isodensity: float = ISODENSITY_THRESHOLD,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline complet de calcul ALIE pour un conformère.

    Phase 1 : DENSF densité sur grille 3D → identifier l'isosurface
    Phase 2 : Extraction des points de surface (interpolation linéaire)
    Phase 3 : DENSF orbitales + densité + potentiel aux points de surface
    Phase 4 : Calcul ALIE et extraction des descripteurs

    Args:
        adf_rkf_path: Chemin vers le fichier adf.rkf du calcul neutre
        work_dir: Répertoire de travail pour les fichiers DENSF temporaires
        name: Nom de base pour les fichiers
        grid_quality: Qualité de la grille DENSF ("coarse", "medium", "fine")
        isodensity: Seuil d'isodensité pour la surface (défaut: 0.001 a.u.)
        verbose: Afficher les messages de progression

    Returns:
        Dict avec tous les descripteurs ALIE + métadonnées :
            - I_S_min_Ha, I_S_min_eV, I_S_min_kJ
            - V_S_min_Ha, V_S_min_eV, V_S_min_kJ
            - I_S_mean_Ha, I_S_max_Ha, I_S_std_Ha, I_S_variance_Ha
            - V_S_mean_Ha, V_S_max_Ha
            - n_surface_points, rho_surface_mean
            - n_occupied, n_total_orbitals
            - success: bool
    """
    adf_rkf_path = str(adf_rkf_path)
    work_dir = str(work_dir)

    try:
        # --- Phase 0 : Métadonnées orbitales ---
        if verbose:
            print(f"  Phase 0: Extraction des métadonnées orbitales...")
        meta = extract_orbital_metadata(adf_rkf_path)
        if verbose:
            print(f"    {meta['n_occupied']} occupées / {meta['n_total']} total")

        # --- Phase 1 : DENSF densité sur grille 3D ---
        if verbose:
            print(f"  Phase 1: DENSF densité sur grille {grid_quality}...")
        phase1_dir = os.path.join(work_dir, f"{name}_phase1")
        t41_file = run_densf_density_grid(adf_rkf_path, phase1_dir, grid_quality)

        # --- Lecture de la grille 3D ---
        if verbose:
            print(f"  Lecture de la grille 3D...")
        grid_data = read_t41_density(t41_file)
        if verbose:
            print(f"    Grille: {grid_data['shape']}")

        # --- Phase 2 : Extraction de l'isosurface ---
        if verbose:
            print(f"  Phase 2: Extraction isosurface rho = {isodensity}...")
        surface_points = extract_isosurface_points(grid_data, threshold=isodensity)
        n_points = surface_points.shape[0]

        if n_points == 0:
            print_error(f"Aucun point de surface trouvé pour {name}")
            return {'success': False, 'error': 'no_surface_points'}

        if verbose:
            print(f"    {n_points} points de surface trouvés")

        # --- Phase 3 : DENSF toutes les orbitales occupées aux points de surface ---
        if verbose:
            print(f"  Phase 3: DENSF {meta['n_occupied']} orbitales occupées "
                  f"aux {n_points} points de surface...")
        phase2_dir = os.path.join(work_dir, f"{name}_phase2")
        t41_orb = run_densf_orbitals_at_points(
            adf_rkf_path, surface_points, meta['orbital_indices'], phase2_dir
        )

        inline_data = read_t41_inline_results(
            t41_orb, meta['orbital_indices'], n_points
        )

        # --- Phase 4 : Calcul ALIE (toutes orbitales occupées) ---
        if verbose:
            print(f"  Phase 4: Calcul de l'ALIE...")
        alie = compute_alie_at_points(
            inline_data['orbitals'],
            meta['occupations'],
            meta['orbital_energies'],
        )

        # --- Extraction des descripteurs ---
        descriptors = extract_alie_descriptors(
            alie,
            inline_data['potential'],
            inline_data['density'],
            surface_points,
        )

        # Ajouter les métadonnées
        descriptors['n_occupied'] = meta['n_occupied']
        descriptors['n_total_orbitals'] = meta['n_total']
        descriptors['success'] = True

        if verbose:
            print_success(
                f"ALIE calculé: I_S,min = {descriptors['I_S_min_Ha']:.4f} Ha "
                f"({descriptors['I_S_min_eV']:.2f} eV), "
                f"V_S,min = {descriptors['V_S_min_Ha']:.4f} Ha "
                f"({descriptors['V_S_min_eV']:.2f} eV), "
                f"{n_points} pts"
            )

        return descriptors

    except Exception as e:
        print_error(f"Erreur ALIE pour {name}: {e}")
        return {
            'success': False,
            'error': str(e),
            'I_S_min_Ha': np.nan, 'I_S_min_eV': np.nan, 'I_S_min_kJ': np.nan,
            'V_S_min_Ha': np.nan, 'V_S_min_eV': np.nan, 'V_S_min_kJ': np.nan,
            'n_surface_points': 0,
        }
