"""
Dual-Hessian anharmonic correction strategies for E_relax prediction.

Uses two Hessians at the same geometry (H_neutral and H_ion from SP) to capture
anharmonic effects beyond the quadratic LVC2 approximation.

Five strategies, each with a single tunable parameter:
  1. DHI  — Dual-Hessian Interpolation (mode-by-mode)
  2. MCRC — Mode-resolved Curvature Ratio Correction
  3. GHC  — Gradient-Hessian Consistency (mode-by-mode weighted ratio)
  4. ODHD — Off-Diagonal Hessian Difference
  5. ESSC — Eigenvalue Spectral Shift Correction

IMPORTANT: All strategies are implemented in mode-by-mode form to properly
handle imaginary frequency modes. Matrix inversion (pinv) is NOT used because
it includes contributions from imaginary/soft modes that corrupt the result.

All energies are returned in kJ/mol (negative = stabilisation).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

from common.constants import EH_TO_KJ
from common.utils import print_warning, print_error


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_hessian(rkf_path: Path) -> Optional[np.ndarray]:
    """
    Extract the Cartesian Hessian matrix from AMSResults/Hessian.

    Args:
        rkf_path: Path to adf.rkf file.

    Returns:
        (3N, 3N) numpy array in Hartree/Bohr², or None on failure.
    """
    try:
        from scm.plams import KFReader
        kf = KFReader(str(rkf_path))

        hess_flat = np.array(kf.read("AMSResults", "Hessian"), dtype=float)
        n_atoms = int(kf.read("Molecule", "nAtoms"))
        dof = 3 * n_atoms

        expected = dof * dof
        if len(hess_flat) != expected:
            print_warning(
                f"Hessian size mismatch in {rkf_path}: "
                f"got {len(hess_flat)}, expected {expected} (3N={dof})"
            )
            return None

        H = hess_flat.reshape(dof, dof)
        return H

    except KeyError:
        print_warning(f"Hessian non trouvé dans {rkf_path}")
        return None
    except Exception as e:
        print_error(f"Erreur extraction Hessian depuis {rkf_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Helper: mode-by-mode LVC2
# ---------------------------------------------------------------------------

def _compute_lvc2_mode_by_mode(
    g: np.ndarray,
    ionic_modes: List[np.ndarray],
    ionic_fc: np.ndarray,
    frequencies: np.ndarray,
    freq_threshold: float = 50.0,
) -> float:
    """
    Compute LVC2 = Σ_j -(F'_j²)/(2k'_j) in mode-by-mode form.

    Skips modes with k_j <= 0 or |freq| < freq_threshold.
    Returns energy in Hartree.
    """
    e_total = 0.0
    for j, mode in enumerate(ionic_modes):
        if mode is None:
            continue
        k_j = ionic_fc[j]
        if k_j <= 0 or abs(frequencies[j]) < freq_threshold:
            continue
        F_j = np.dot(g, mode)
        e_total += -(F_j ** 2) / (2.0 * k_j)
    return e_total


# ---------------------------------------------------------------------------
# Strategy 1: DHI — Dual-Hessian Interpolation (mode-by-mode)
# ---------------------------------------------------------------------------

def compute_dhi_edef(
    g: np.ndarray,
    H_neutral: np.ndarray,
    H_ion: np.ndarray,
    alpha: float,
    ionic_modes: Optional[List[np.ndarray]] = None,
    ionic_fc: Optional[np.ndarray] = None,
    frequencies: Optional[np.ndarray] = None,
    freq_threshold: float = 50.0,
) -> Optional[float]:
    """
    Mode-by-mode DHI: for each ionic mode j,
      k_eff_j = (1-α)·k'_j + α·κ_j   where κ_j = N'_j^T·H_neutral·N'_j
      E_j = -(F'_j²) / (2·k_eff_j)

    E_relax^DHI = Σ_j E_j  (skip if k_eff_j <= 0)

    α=0 recovers LVC2. α=1 uses neutral curvature only.

    Returns E_relax in kJ/mol, or None on failure.
    """
    if ionic_modes is None or ionic_fc is None or frequencies is None:
        return None
    try:
        e_total = 0.0
        for j, mode in enumerate(ionic_modes):
            if mode is None:
                continue
            k_j = ionic_fc[j]
            if k_j <= 0 or abs(frequencies[j]) < freq_threshold:
                continue
            kappa_j = float(mode @ H_neutral @ mode)
            k_eff = (1.0 - alpha) * k_j + alpha * kappa_j
            if k_eff <= 0:
                continue
            F_j = np.dot(g, mode)
            e_total += -(F_j ** 2) / (2.0 * k_eff)

        return e_total * EH_TO_KJ
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Strategy 2: MCRC — Mode-resolved Curvature Ratio Correction
# ---------------------------------------------------------------------------

def compute_curvature_ratios(
    H_neutral: np.ndarray,
    ionic_modes: List[np.ndarray],
    ionic_fc: np.ndarray,
    freq_threshold: float = 50.0,
    frequencies: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute κ_j / k'_j for each ionic mode j.

    κ_j = N'_j^T · H_neutral · N'_j  (neutral curvature along ionic mode)
    r_j = κ_j / k'_j                  (curvature ratio)

    Returns array of ratios (same length as ionic_modes).
    Modes below freq_threshold or with k'_j <= 0 get ratio = 1.0 (no correction).
    """
    n_modes = len(ionic_modes)
    ratios = np.ones(n_modes)

    for j in range(n_modes):
        mode = ionic_modes[j]
        if mode is None:
            continue
        k_j = ionic_fc[j]
        if k_j <= 0:
            continue
        if frequencies is not None and abs(frequencies[j]) < freq_threshold:
            continue

        kappa_j = mode @ H_neutral @ mode
        ratios[j] = kappa_j / k_j

    return ratios


def compute_mcrc_edef(
    g: np.ndarray,
    H_neutral: np.ndarray,
    ionic_modes: List[np.ndarray],
    ionic_fc: np.ndarray,
    frequencies: np.ndarray,
    beta: float,
    freq_threshold: float = 50.0,
) -> Optional[float]:
    """
    E_relax^MCRC = Σ_j E_j  where
    E_j = -(F'_j²)/(2k'_j) × (1 + β·(r_j - 1)·|Δq'_j|)

    r_j = κ_j / k'_j  (curvature ratio neutral/ion)
    Δq'_j = F'_j / k'_j (harmonic displacement along mode j)

    Returns E_relax in kJ/mol, or None on failure.
    """
    try:
        ratios = compute_curvature_ratios(
            H_neutral, ionic_modes, ionic_fc,
            freq_threshold, frequencies
        )

        e_total = 0.0
        for j, mode in enumerate(ionic_modes):
            if mode is None:
                continue
            k_j = ionic_fc[j]
            if k_j <= 0 or abs(frequencies[j]) < freq_threshold:
                continue

            F_j = np.dot(g, mode)
            dq_j = F_j / k_j
            e_harm_j = -(F_j ** 2) / (2.0 * k_j)
            r_j = ratios[j]
            correction = 1.0 + beta * (r_j - 1.0) * abs(dq_j)
            e_total += e_harm_j * correction

        return e_total * EH_TO_KJ

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Strategy 3: GHC — Gradient-Hessian Consistency (mode-by-mode)
# ---------------------------------------------------------------------------

def compute_ghc_edef(
    g: np.ndarray,
    H_neutral: np.ndarray,
    H_ion: np.ndarray,
    lvc2_kj: float,
    gamma: float,
    ionic_modes: Optional[List[np.ndarray]] = None,
    ionic_fc: Optional[np.ndarray] = None,
    frequencies: Optional[np.ndarray] = None,
    freq_threshold: float = 50.0,
) -> Optional[float]:
    """
    Mode-by-mode GHC: compute weighted-average curvature ratio A along
    gradient direction, using mode decomposition:
      A = Σ_j (F'_j² · r_j) / Σ_j F'_j²
    where r_j = κ_j / k'_j (curvature ratio for mode j).

    E_relax^GHC = LVC2 × (1 + γ·(A - 1))

    Returns E_relax in kJ/mol, or None on failure.
    """
    if ionic_modes is None or ionic_fc is None or frequencies is None:
        return None
    try:
        num = 0.0
        den = 0.0
        for j, mode in enumerate(ionic_modes):
            if mode is None:
                continue
            k_j = ionic_fc[j]
            if k_j <= 0 or abs(frequencies[j]) < freq_threshold:
                continue
            F_j = np.dot(g, mode)
            F_j_sq = F_j ** 2
            kappa_j = float(mode @ H_neutral @ mode)
            r_j = kappa_j / k_j
            num += F_j_sq * r_j
            den += F_j_sq

        if den < 1e-30:
            return lvc2_kj

        A = num / den
        return lvc2_kj * (1.0 + gamma * (A - 1.0))

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Strategy 4: ODHD — Off-Diagonal Hessian Difference
# ---------------------------------------------------------------------------

def compute_odhd_correction(
    g: np.ndarray,
    H_neutral: np.ndarray,
    H_ion: np.ndarray,
    ionic_modes: List[np.ndarray],
    ionic_fc: np.ndarray,
    frequencies: np.ndarray,
    lvc2_kj: float,
    eta: float,
    freq_threshold: float = 50.0,
) -> Optional[float]:
    """
    ΔH = H_ion - H_neutral
    ΔH'_ij = N'_i^T · ΔH · N'_j          (in ionic mode basis)
    Δq'_i = (g · N'_i) / k'_i             (harmonic displacement)
    C = Σ_{i≠j} ΔH'_ij · Δq'_i · Δq'_j   (inter-mode coupling)
    E_relax^ODHD = LVC2 + η · C

    Returns E_relax in kJ/mol, or None on failure.
    """
    try:
        delta_H = H_ion - H_neutral

        # Build list of valid modes with their displacements
        valid = []
        for j, mode in enumerate(ionic_modes):
            if mode is None:
                continue
            k_j = ionic_fc[j]
            if k_j <= 0 or abs(frequencies[j]) < freq_threshold:
                continue
            F_j = np.dot(g, mode)
            dq_j = F_j / k_j
            valid.append((j, mode, dq_j))

        n_valid = len(valid)
        if n_valid < 2:
            return lvc2_kj

        # Compute off-diagonal coupling C
        C = 0.0
        for a in range(n_valid):
            _, mode_i, dq_i = valid[a]
            dH_mode_i = delta_H @ mode_i     # ΔH · N'_i
            for b in range(a + 1, n_valid):
                _, mode_j, dq_j = valid[b]
                dH_ij = np.dot(mode_j, dH_mode_i)  # N'_j^T · ΔH · N'_i
                C += 2.0 * dH_ij * dq_i * dq_j     # factor 2 for i<j + j<i

        # C is in Hartree, convert to kJ/mol
        return lvc2_kj + eta * C * EH_TO_KJ

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Strategy 5: ESSC — Eigenvalue Spectral Shift Correction
# ---------------------------------------------------------------------------

def compute_essc_correction(
    g: np.ndarray,
    H_neutral: np.ndarray,
    H_ion: np.ndarray,
    lvc2_kj: float,
    mu: float,
) -> Optional[float]:
    """
    ΔH = H_ion - H_neutral, diagonalize: ΔH·u_k = λ_k·u_k
    g_k = g · u_k
    S = Σ_k λ_k · g_k² / |g|²
    E_relax^ESSC = LVC2 × (1 + μ · S / ‖H_ion‖_F)

    Returns E_relax in kJ/mol, or None on failure.
    """
    try:
        g_norm_sq = np.dot(g, g)
        if g_norm_sq < 1e-30:
            return lvc2_kj

        delta_H = H_ion - H_neutral
        eigenvalues, eigenvectors = np.linalg.eigh(delta_H)

        # g_k = g · u_k for each eigenvector k
        g_proj = eigenvectors.T @ g                # (n, ) array
        S = np.sum(eigenvalues * g_proj ** 2) / g_norm_sq

        H_ion_frob = np.linalg.norm(H_ion, 'fro')
        if H_ion_frob < 1e-30:
            return lvc2_kj

        return lvc2_kj * (1.0 + mu * S / H_ion_frob)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-conformer runner
# ---------------------------------------------------------------------------

# Default parameter values (optimized on n=11 molecules, reduction)
DEFAULT_PARAMS = {
    'alpha_dhi': -0.026,
    'beta_mcrc': 0.277,
    'gamma_ghc': 0.332,
    'eta_odhd': 0.287,
    'mu_essc': 1.0,
}

# Method display names
METHOD_NAMES = {
    'DHI': 'DHI (Dual-Hessian Interp.)',
    'MCRC': 'MCRC (Curvature Ratio)',
    'GHC': 'GHC (Gradient-Hessian)',
    'ODHD': 'ODHD (Off-Diag ΔH)',
    'ESSC': 'ESSC (Spectral Shift)',
}


def run_dual_hessian_for_conformer(
    neutral_rkf: Path,
    ion_sp_rkf: Path,
    params: Optional[Dict[str, float]] = None,
    freq_threshold: float = 50.0,
) -> Dict[str, Any]:
    """
    Compute all 5 dual-Hessian E_relax predictions for one conformer.

    Args:
        neutral_rkf: Path to the neutral optimized adf.rkf (source of H_neutral).
        ion_sp_rkf:  Path to the ion single-point adf.rkf (source of g, H_ion, modes).
        params:      Dict of parameter values (keys: alpha_dhi, beta_mcrc, gamma_ghc,
                     eta_odhd, mu_essc). Missing keys use DEFAULT_PARAMS.
        freq_threshold: Minimum frequency in cm⁻¹ for mode inclusion.

    Returns:
        Dict with keys:
            success: bool
            E_def_lvc2_kj: float (LVC2 baseline for reference)
            E_def_DHI_kj, E_def_MCRC_kj, E_def_GHC_kj, E_def_ODHD_kj, E_def_ESSC_kj: float
            params: dict of parameter values used
            error: str if failure
    """
    from .lvc_analysis import extract_gradients, extract_normal_modes_and_fc

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    result = {
        'success': False,
        'E_def_lvc2_kj': np.nan,
        'E_def_DHI_kj': np.nan,
        'E_def_MCRC_kj': np.nan,
        'E_def_GHC_kj': np.nan,
        'E_def_ODHD_kj': np.nan,
        'E_def_ESSC_kj': np.nan,
        'params': p,
        'error': None,
    }

    # --- Extract data ---
    g = extract_gradients(ion_sp_rkf)
    if g is None:
        result['error'] = "Gradient extraction failed"
        return result

    H_ion = extract_hessian(ion_sp_rkf)
    if H_ion is None:
        result['error'] = "Ionic Hessian extraction failed"
        return result

    H_neutral = extract_hessian(neutral_rkf)
    if H_neutral is None:
        result['error'] = "Neutral Hessian extraction failed"
        return result

    ionic_modes, ionic_fc, frequencies = extract_normal_modes_and_fc(ion_sp_rkf)
    if ionic_modes is None:
        result['error'] = "Ionic modes extraction failed"
        return result

    # Dimension check
    dof = len(g)
    if H_ion.shape != (dof, dof) or H_neutral.shape != (dof, dof):
        result['error'] = (
            f"Dimension mismatch: g={dof}, "
            f"H_ion={H_ion.shape}, H_neutral={H_neutral.shape}"
        )
        return result

    # --- LVC2 baseline (mode-by-mode, skips imaginary modes) ---
    lvc2_hartree = _compute_lvc2_mode_by_mode(
        g, ionic_modes, ionic_fc, frequencies, freq_threshold
    )
    lvc2_kj = lvc2_hartree * EH_TO_KJ
    result['E_def_lvc2_kj'] = lvc2_kj

    # --- Strategy 1: DHI (mode-by-mode) ---
    result['E_def_DHI_kj'] = compute_dhi_edef(
        g, H_neutral, H_ion, p['alpha_dhi'],
        ionic_modes, ionic_fc, frequencies, freq_threshold
    )

    # --- Strategy 2: MCRC ---
    result['E_def_MCRC_kj'] = compute_mcrc_edef(
        g, H_neutral, ionic_modes, ionic_fc, frequencies,
        p['beta_mcrc'], freq_threshold
    )

    # --- Strategy 3: GHC (mode-by-mode) ---
    result['E_def_GHC_kj'] = compute_ghc_edef(
        g, H_neutral, H_ion, lvc2_kj, p['gamma_ghc'],
        ionic_modes, ionic_fc, frequencies, freq_threshold
    )

    # --- Strategy 4: ODHD ---
    result['E_def_ODHD_kj'] = compute_odhd_correction(
        g, H_neutral, H_ion, ionic_modes, ionic_fc, frequencies,
        lvc2_kj, p['eta_odhd'], freq_threshold
    )

    # --- Strategy 5: ESSC ---
    result['E_def_ESSC_kj'] = compute_essc_correction(
        g, H_neutral, H_ion, lvc2_kj, p['mu_essc']
    )

    result['success'] = True
    return result


# ---------------------------------------------------------------------------
# Molecule-level runner
# ---------------------------------------------------------------------------

def run_dual_hessian_for_molecule(
    workdir: Path,
    molecule_name: str,
    params: Optional[Dict[str, float]] = None,
    freq_threshold: float = 50.0,
) -> Dict[str, Any]:
    """
    Run dual-Hessian analysis for all conformers of a molecule (reduction + oxidation).

    Args:
        workdir: Molecule workdir (e.g., DA_workdir/).
        molecule_name: Molecule name (e.g., "DA").
        params: Parameter dict (see run_dual_hessian_for_conformer).
        freq_threshold: Minimum frequency in cm⁻¹.

    Returns:
        Dict with keys:
            conformers: {conf_num: {'reduction': result_dict, 'oxidation': result_dict}}
    """
    from .lvc_analysis import find_sp_rkf_files

    results = {'conformers': {}}

    # Find neutral RKFs
    plams_dir = workdir / "redox" / "calculations" / "plams_workdir"
    if not plams_dir.exists():
        print_warning(f"Pas de plams_workdir dans {workdir}")
        return results

    # Find SP RKF files for reduction and oxidation
    reduced_sp_files = find_sp_rkf_files(workdir, molecule_name, 'reduced')
    oxidized_sp_files = find_sp_rkf_files(workdir, molecule_name, 'oxidized')

    # Find neutral RKF files
    all_conf_nums = set(reduced_sp_files.keys()) | set(oxidized_sp_files.keys())

    for conf_num in sorted(all_conf_nums):
        # Find neutral RKF for this conformer
        neutral_dir = plams_dir / f"{molecule_name}_conf_{conf_num}_neutral"
        neutral_rkf = neutral_dir / "adf.rkf"
        if not neutral_rkf.exists():
            continue

        conf_results = {}

        # Reduction
        if conf_num in reduced_sp_files:
            ion_sp_rkf = reduced_sp_files[conf_num]
            conf_results['reduction'] = run_dual_hessian_for_conformer(
                neutral_rkf, ion_sp_rkf, params, freq_threshold
            )

        # Oxidation
        if conf_num in oxidized_sp_files:
            ion_sp_rkf = oxidized_sp_files[conf_num]
            conf_results['oxidation'] = run_dual_hessian_for_conformer(
                neutral_rkf, ion_sp_rkf, params, freq_threshold
            )

        results['conformers'][conf_num] = conf_results

    return results


# ---------------------------------------------------------------------------
# Parameter optimization
# ---------------------------------------------------------------------------

def _preextract_data(
    workdirs: List[Path],
    molecule_names: List[str],
    freq_threshold: float = 50.0,
) -> List[Dict[str, Any]]:
    """
    Pre-extract all Hessian/gradient/mode data for all conformers.
    Returns a list of dicts, each containing the extracted arrays.
    """
    from .lvc_analysis import (
        extract_gradients, extract_normal_modes_and_fc,
        find_sp_rkf_files,
    )

    entries = []

    for workdir, mol_name in zip(workdirs, molecule_names):
        plams_dir = workdir / "redox" / "calculations" / "plams_workdir"
        if not plams_dir.exists():
            continue

        for process, state in [('reduction', 'reduced'), ('oxidation', 'oxidized')]:
            sp_files = find_sp_rkf_files(workdir, mol_name, state)

            for conf_num, ion_sp_rkf in sp_files.items():
                neutral_dir = plams_dir / f"{mol_name}_conf_{conf_num}_neutral"
                neutral_rkf = neutral_dir / "adf.rkf"
                if not neutral_rkf.exists():
                    continue

                g = extract_gradients(ion_sp_rkf)
                H_ion = extract_hessian(ion_sp_rkf)
                H_neutral = extract_hessian(neutral_rkf)
                modes, fc, freqs = extract_normal_modes_and_fc(ion_sp_rkf)

                if g is None or H_ion is None or H_neutral is None:
                    continue
                if modes is None or fc is None or freqs is None:
                    continue

                dof = len(g)
                if H_ion.shape != (dof, dof) or H_neutral.shape != (dof, dof):
                    continue

                # Compute LVC2 baseline (mode-by-mode, skip imaginary modes)
                lvc2_h = _compute_lvc2_mode_by_mode(
                    g, modes, fc, freqs, freq_threshold
                )
                lvc2_kj = lvc2_h * EH_TO_KJ

                entries.append({
                    'mol_name': mol_name,
                    'conf_num': conf_num,
                    'process': process,
                    'g': g,
                    'H_ion': H_ion,
                    'H_neutral': H_neutral,
                    'modes': modes,
                    'fc': fc,
                    'freqs': freqs,
                    'lvc2_kj': lvc2_kj,
                    'freq_threshold': freq_threshold,
                })

    return entries


def optimize_parameters(
    workdirs: List[Path],
    molecule_names: List[str],
    real_edef: Dict[str, Dict[int, Dict[str, float]]],
    process: str = 'reduction',
    freq_threshold: float = 50.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Optimize each method's single parameter by minimizing MAE on the dataset.

    Args:
        workdirs: List of molecule workdirs.
        molecule_names: Corresponding molecule names.
        real_edef: {mol_name: {conf_num: {'E_def_red': float, 'E_def_ox': float}}}.
        process: 'reduction' or 'oxidation'.
        freq_threshold: Minimum frequency threshold.

    Returns:
        Dict mapping method name to (optimal_param, min_MAE).
        Keys: 'alpha_dhi', 'beta_mcrc', 'gamma_ghc', 'eta_odhd', 'mu_essc'.
    """
    from scipy.optimize import minimize_scalar

    edef_key = 'E_def_red' if process == 'reduction' else 'E_def_ox'

    # Pre-extract data
    entries = _preextract_data(workdirs, molecule_names, freq_threshold)
    # Filter to the requested process
    entries = [e for e in entries if e['process'] == process]

    if not entries:
        print_warning("Aucune donnée pour l'optimisation")
        return {}

    # Attach real E_def values
    for e in entries:
        mol = e['mol_name']
        conf = e['conf_num']
        if mol in real_edef and conf in real_edef[mol]:
            e['real_edef'] = real_edef[mol][conf].get(edef_key)
        else:
            e['real_edef'] = None

    entries = [e for e in entries if e['real_edef'] is not None]
    if not entries:
        print_warning("Aucune valeur réelle de E_def pour l'optimisation")
        return {}

    optimized = {}

    # --- DHI (mode-by-mode) ---
    def mae_dhi(alpha):
        errors = []
        for e in entries:
            val = compute_dhi_edef(
                e['g'], e['H_neutral'], e['H_ion'], alpha,
                e['modes'], e['fc'], e['freqs'], e['freq_threshold']
            )
            if val is not None:
                errors.append(abs(val - e['real_edef']))
        return np.mean(errors) if errors else 1e10

    res = minimize_scalar(mae_dhi, bounds=(-1.0, 2.0), method='bounded')
    optimized['alpha_dhi'] = (res.x, res.fun)

    # --- MCRC ---
    def mae_mcrc(beta):
        errors = []
        for e in entries:
            val = compute_mcrc_edef(
                e['g'], e['H_neutral'], e['modes'], e['fc'],
                e['freqs'], beta, e['freq_threshold']
            )
            if val is not None:
                errors.append(abs(val - e['real_edef']))
        return np.mean(errors) if errors else 1e10

    res = minimize_scalar(mae_mcrc, bounds=(-5.0, 5.0), method='bounded')
    optimized['beta_mcrc'] = (res.x, res.fun)

    # --- GHC (mode-by-mode) ---
    def mae_ghc(gamma):
        errors = []
        for e in entries:
            val = compute_ghc_edef(
                e['g'], e['H_neutral'], e['H_ion'],
                e['lvc2_kj'], gamma,
                e['modes'], e['fc'], e['freqs'], e['freq_threshold']
            )
            if val is not None:
                errors.append(abs(val - e['real_edef']))
        return np.mean(errors) if errors else 1e10

    res = minimize_scalar(mae_ghc, bounds=(-5.0, 5.0), method='bounded')
    optimized['gamma_ghc'] = (res.x, res.fun)

    # --- ODHD ---
    def mae_odhd(eta):
        errors = []
        for e in entries:
            val = compute_odhd_correction(
                e['g'], e['H_neutral'], e['H_ion'],
                e['modes'], e['fc'], e['freqs'],
                e['lvc2_kj'], eta, e['freq_threshold']
            )
            if val is not None:
                errors.append(abs(val - e['real_edef']))
        return np.mean(errors) if errors else 1e10

    res = minimize_scalar(mae_odhd, bounds=(-5.0, 5.0), method='bounded')
    optimized['eta_odhd'] = (res.x, res.fun)

    # --- ESSC ---
    def mae_essc(mu):
        errors = []
        for e in entries:
            val = compute_essc_correction(
                e['g'], e['H_neutral'], e['H_ion'],
                e['lvc2_kj'], mu
            )
            if val is not None:
                errors.append(abs(val - e['real_edef']))
        return np.mean(errors) if errors else 1e10

    res = minimize_scalar(mae_essc, bounds=(-10.0, 10.0), method='bounded')
    optimized['mu_essc'] = (res.x, res.fun)

    return optimized
