"""
Utilitaires spécifiques au pipeline QTAIM/IQA.

Ce module contient les fonctions spécifiques aux calculs QTAIM et IQA.
"""

from pathlib import Path
from typing import Dict
from scm.plams import Settings

# Importation des utilitaires communs
from common.functional import configure_functional


class CalculationError(Exception):
    """Custom exception for calculation errors."""
    pass


def get_conformer_files(workdir, state="neutral"):
    """
    Get conformer XYZ files from conformers results directory.

    Args:
        workdir (Path or str): Working directory
        state (str): Oxidation state (neutral, oxidized, reduced) - currently only "neutral" supported

    Returns:
        Dict[str, Path]: Dictionary mapping conformer_id -> xyz_file_path
    """
    workdir = Path(workdir)
    conformers_results_dir = workdir / "conformers" / "results"

    conformer_files = {}

    if not conformers_results_dir.exists():
        raise FileNotFoundError(f"Conformers results directory not found: {conformers_results_dir}")

    # Find all conformer XYZ files (format: conf_1.xyz, conf_2.xyz, etc.)
    for xyz_file in conformers_results_dir.glob("conf_*.xyz"):
        # Extract conformer ID from filename
        # Example: conf_1.xyz -> conf_1
        conf_id = xyz_file.stem  # Gets filename without extension
        conformer_files[conf_id] = xyz_file

    return conformer_files


def create_adf_settings(
    functional="BLYP",
    basis="TZP",
    quality="Good",
    integration_accuracy=5.0,
    run_iqa=True,
    charge=0,
    spin_polarization=0
):
    """
    Create ADF settings for QTAIM/IQA calculations with default values.

    Args:
        functional (str): DFT functional (default: BLYP)
        basis (str): Basis set (default: TZP)
        quality (str): Numerical quality (default: Good)
        integration_accuracy (float): QTAIM integration accuracy (default: 5.0)
        run_iqa (bool): Enable IQA calculations (default: True)
        charge (int): Molecular charge (default: 0)
        spin_polarization (int): Spin polarization (default: 0)

    Returns:
        Settings: PLAMS Settings object
    """
    s = Settings()

    # AMS driver settings
    s.input.AMS.Task = 'SinglePoint'

    # System charge and spin
    s.input.AMS.System.Charge = charge
    if spin_polarization > 0:
        s.input.adf.SpinPolarization = spin_polarization
        s.input.adf.Unrestricted = 'Yes'

    # Engine ADF block
    s.input.adf.Basis.Type = basis
    s.input.adf.Basis.Core = 'None'

    # Disable ZORA - QTAIM/IQA is incompatible with relativistic corrections
    s.input.adf.Relativity.Level = 'None'

    # XC functional using common function
    configure_functional(s, functional)

    # Numerical quality
    s.input.adf.NumericalQuality = quality

    # QTAIM analysis
    s.input.adf.QTAIM.Enabled = 'Yes'
    s.input.adf.QTAIM.AnalysisLevel = 'Full'
    s.input.adf.QTAIM.Source = 'Yes'

    # IQA analysis
    if run_iqa:
        s.input.adf.IQA.Enabled = 'Yes'

    return s
