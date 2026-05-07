# -*- coding: utf-8 -*-
"""
Utilitaires spécifiques au pipeline de génération de conformères.

Ce module contient les constantes et fonctions spécifiques au pipeline
de génération et optimisation de conformères.
"""

# Importation des utilitaires communs
from common.constants import TEMPERATURE, R, UNIT, KCAL_TO_KJ
from common.functional import configure_functional, ALL_FUNCTIONALS
from common.utils import DefaultJobManager, boltzmann_weights

# Re-export pour compatibilité avec le code existant
__all__ = [
    'TEMPERATURE', 'R', 'UNIT', 'KCAL_TO_KJ',
    'configure_functional', 'ALL_FUNCTIONALS',
    'DefaultJobManager', 'boltzmann_weights',
]
