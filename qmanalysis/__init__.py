"""
Module d'analyse fréquence-Edef.

Ce module fournit des outils pour extraire les données de fréquences vibrationnelles
depuis les fichiers RKF, calculer des descripteurs, et chercher des corrélations
avec les énergies de déformation (Edef).
"""

from .frequency_extraction import (
    find_neutral_rkf_files,
    extract_all_frequency_data,
    extract_ir_intensities
)

from .frequency_descriptors import (
    compute_frequency_descriptors,
    compute_zpe,
    FREQUENCY_RANGES
)

from .common import (
    HIGH_FREQ_DESCRIPTORS,
    LOW_FREQ_DESCRIPTORS
)

from .correlation_analysis import (
    load_edef_data,
    load_redox_thermodynamic_data,
    compute_boltzmann_averaged_descriptors,
    compute_correlations,
    compute_focused_correlations,
    run_full_analysis
)

from .bilinear_analysis import (
    bilinear_regression,
    compute_all_bilinear_correlations,
    run_bilinear_analysis,
    format_bilinear_results
)

from .extended_descriptors import (
    extract_extended_data,
    compute_all_extended_descriptors,
    get_extended_descriptor_names
)

from .lvc_analysis import (
    extract_gradients,
    extract_normal_modes_and_fc,
    compute_lvc_edef,
    compute_lvc_edef_from_rkf,
    compute_lvc2_edef_from_rkf,
    find_sp_rkf_files,
    run_lvc_analysis_for_molecule,
    load_real_edef,
    extract_coordinates_from_rkf,
    compute_rmsd,
    find_opt_rkf_files,
    compute_rmsd_for_molecule,
)

from .duschinsky_analysis import (
    extract_atomic_masses,
    extract_coordinates_bohr,
    compute_displacement_edef_neutral_modes,
    compute_duschinsky_edef,
    compute_duschinsky_matrix,
    solve_morse_a,
    morse_energy,
    compute_morse_duschinsky_edef,
    run_duschinsky_analysis_for_molecule,
    # Legacy (deprecated)
    compute_displacement_vector,
    compute_huang_rhys_factors,
    compute_reorganization_energy,
)

from .qvc_analysis import (
    estimate_frequency_shift_ratio,
    estimate_all_frequency_shifts,
    compute_qvc_correction,
    compute_qvc_edef,
    run_qvc_analysis_for_molecule,
    optimize_alpha,
)

from .vpt2_analysis import (
    compute_vpt2_cubic_edef,
    run_vpt2_analysis_for_molecule,
)

from .dual_hessian_analysis import (
    extract_hessian,
    compute_dhi_edef,
    compute_mcrc_edef,
    compute_ghc_edef,
    compute_odhd_correction,
    compute_essc_correction,
    compute_curvature_ratios,
    run_dual_hessian_for_conformer,
    run_dual_hessian_for_molecule,
    optimize_parameters,
    DEFAULT_PARAMS,
    METHOD_NAMES,
)

from .qtaim_extraction import (
    find_qtaim_neutral_rkf_files,
    extract_density_properties,
)

from .qtaim_descriptors import (
    compute_density_descriptors,
    get_density_descriptor_names,
)

from .qtaim_correlation import (
    load_ea_ip_targets,
    load_orbital_data,
    run_density_analysis,
)

from .common import (
    HIGH_DENSITY_DESCRIPTORS,
    CHARGE_DESCRIPTORS,
)

from .alie_extraction import (
    compute_alie_for_conformer,
    extract_orbital_metadata,
    extract_isosurface_points,
    compute_alie_at_points,
    extract_alie_descriptors,
    ISODENSITY_THRESHOLD,
)

__all__ = [
    # Extraction
    'find_neutral_rkf_files',
    'extract_all_frequency_data',
    'extract_ir_intensities',

    # Descripteurs
    'compute_frequency_descriptors',
    'compute_zpe',
    'FREQUENCY_RANGES',
    'HIGH_FREQ_DESCRIPTORS',
    'LOW_FREQ_DESCRIPTORS',

    # Corrélations monolinéaires
    'load_edef_data',
    'load_redox_thermodynamic_data',
    'compute_boltzmann_averaged_descriptors',
    'compute_correlations',
    'compute_focused_correlations',
    'run_full_analysis',

    # Corrélations bilinéaires
    'bilinear_regression',
    'compute_all_bilinear_correlations',
    'run_bilinear_analysis',
    'format_bilinear_results',

    # Descripteurs étendus
    'extract_extended_data',
    'compute_all_extended_descriptors',
    'get_extended_descriptor_names',

    # Analyse LVC (Linear Vibronic Coupling)
    'extract_gradients',
    'extract_normal_modes_and_fc',
    'compute_lvc_edef',
    'compute_lvc_edef_from_rkf',
    'compute_lvc2_edef_from_rkf',
    'find_sp_rkf_files',
    'run_lvc_analysis_for_molecule',
    'load_real_edef',
    'extract_coordinates_from_rkf',
    'compute_rmsd',
    'find_opt_rkf_files',
    'compute_rmsd_for_molecule',

    # Analyse Duschinsky
    'extract_atomic_masses',
    'extract_coordinates_bohr',
    'compute_displacement_edef_neutral_modes',
    'compute_duschinsky_edef',
    'compute_duschinsky_matrix',
    'solve_morse_a',
    'morse_energy',
    'compute_morse_duschinsky_edef',
    'run_duschinsky_analysis_for_molecule',
    # Legacy (deprecated)
    'compute_displacement_vector',
    'compute_huang_rhys_factors',
    'compute_reorganization_energy',

    # Analyse QVC (Quadratic Vibronic Coupling)
    'estimate_frequency_shift_ratio',
    'estimate_all_frequency_shifts',
    'compute_qvc_correction',
    'compute_qvc_edef',
    'run_qvc_analysis_for_molecule',
    'optimize_alpha',

    # Analyse VPT2-cubique
    'compute_vpt2_cubic_edef',
    'run_vpt2_analysis_for_molecule',

    # Analyse Dual-Hessian (corrections anharmoniques)
    'extract_hessian',
    'compute_dhi_edef',
    'compute_mcrc_edef',
    'compute_ghc_edef',
    'compute_odhd_correction',
    'compute_essc_correction',
    'compute_curvature_ratios',
    'run_dual_hessian_for_conformer',
    'run_dual_hessian_for_molecule',
    'optimize_parameters',
    'DEFAULT_PARAMS',
    'METHOD_NAMES',

    # Analyse densite QTAIM vs EA/IP
    'find_qtaim_neutral_rkf_files',
    'extract_density_properties',
    'compute_density_descriptors',
    'get_density_descriptor_names',
    'load_ea_ip_targets',
    'load_orbital_data',
    'run_density_analysis',
    'HIGH_DENSITY_DESCRIPTORS',
    'CHARGE_DESCRIPTORS',

    # Analyse ALIE de Politzer
    'compute_alie_for_conformer',
    'extract_orbital_metadata',
    'extract_isosurface_points',
    'compute_alie_at_points',
    'extract_alie_descriptors',
    'ISODENSITY_THRESHOLD',
]
