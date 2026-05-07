"""
Module commun pour tous les pipelines PlamsV4.

Ce module contient les fonctionnalités partagées entre les différents
pipelines de calcul (conformers, redox, qtaim).
"""

from .constants import *
from .functional import configure_functional, ALL_FUNCTIONALS
from .workspace import setup_workspace
from .utils import boltzmann_weights, DefaultJobManager
from .frequency import (
    extract_frequencies_from_rkf,
    has_imaginary_frequencies,
    check_frequencies,
    run_frequency_calculation,
    check_and_fix_frequencies
)

__all__ = [
    # Constants
    'R', 'F', 'TEMPERATURE', 'UNIT',
    'EH_TO_EV', 'EH_TO_KJ', 'EV_TO_KJ', 'KJ_TO_EV', 'KCAL_TO_KJ',

    # Functional configuration
    'configure_functional', 'ALL_FUNCTIONALS',

    # Workspace management
    'setup_workspace',

    # Utilities
    'boltzmann_weights', 'DefaultJobManager',

    # Frequency verification
    'extract_frequencies_from_rkf',
    'has_imaginary_frequencies',
    'check_frequencies',
    'run_frequency_calculation',
    'check_and_fix_frequencies',
]
