"""
Utilitaires partagés entre les différents pipelines.

Ce module contient des fonctions et classes utilitaires utilisées
par plusieurs pipelines (conformers, redox, qtaim).
"""

import os
import numpy as np
from scm.plams import config

from .constants import R, TEMPERATURE


class DefaultJobManager:
    """
    Gestionnaire de contexte pour rediriger temporairement le répertoire de travail PLAMS.

    Cette classe permet d'exécuter des jobs PLAMS dans un répertoire spécifique
    sans affecter la configuration globale de manière permanente.

    Examples:
        >>> with DefaultJobManager("/path/to/job/dir"):
        ...     job = AMSJob(...)
        ...     result = job.run()
    """

    def __init__(self, job_dir):
        """
        Initialise le gestionnaire de contexte.

        Args:
            job_dir (str or Path): Répertoire cible pour les jobs PLAMS
        """
        self.job_dir = str(job_dir)
        self._orig_mgr = None
        self._orig_workdir = None

    def __enter__(self):
        """Entre dans le contexte : sauvegarde et modifie le workdir."""
        from scm.plams import JobManager

        if hasattr(config, 'default_jobmanager'):
            self._orig_mgr = config.default_jobmanager
            if hasattr(self._orig_mgr, 'workdir'):
                self._orig_workdir = self._orig_mgr.workdir

        os.makedirs(self.job_dir, exist_ok=True)

        # Create a new JobManager with the desired workdir instead of trying to modify the existing one
        if hasattr(config, 'default_jobmanager') and config.default_jobmanager:
            new_mgr = JobManager(
                settings=config.default_jobmanager.settings if hasattr(config.default_jobmanager, 'settings') else {},
                path=self.job_dir,
                folder=config.default_jobmanager.foldername if hasattr(config.default_jobmanager, 'foldername') else None
            )
            config.default_jobmanager = new_mgr

        return self

    def __exit__(self, exc_type, exc, tb):
        """Sort du contexte : restaure le workdir original."""
        if self._orig_mgr:
            config.default_jobmanager = self._orig_mgr


def boltzmann_weights(energies_kj, temperature=TEMPERATURE):
    """
    Calcule les poids de Boltzmann à partir d'énergies relatives.

    Les poids sont calculés selon la formule :
    w_i = exp(-β * ΔE_i) / Z
    où β = 1/(RT) et Z = Σ exp(-β * ΔE_i)

    Args:
        energies_kj (array-like): Énergies relatives en kJ/mol
        temperature (float): Température en Kelvin (défaut: 298.15 K)

    Returns:
        tuple: (weights, Z)
               - weights (np.ndarray): Poids de Boltzmann normalisés (somme = 1)
               - Z (float): Fonction de partition

    Examples:
        >>> energies = [0.0, 5.0, 10.0]  # kJ/mol
        >>> weights, Z = boltzmann_weights(energies)
        >>> print(f"Poids: {weights}")
        >>> print(f"Fonction de partition: {Z}")
    """
    # Constante β = 1/(RT) avec R en kJ/(mol·K)
    beta = 1.0 / ((R / 1000.0) * temperature)

    # Calcul des facteurs de Boltzmann
    energies_array = np.array(energies_kj, dtype=float)
    exp_values = np.exp(-beta * energies_array)

    # Fonction de partition
    Z = exp_values.sum()

    # Normalisation des poids
    if Z > 0:
        weights = exp_values / Z
    else:
        weights = np.zeros_like(exp_values)

    return weights, Z


def print_header(title, width=80):
    """
    Affiche un header formaté de manière standardisée.

    Args:
        title (str): Titre à afficher
        width (int): Largeur totale du header (défaut: 80)

    Examples:
        >>> print_header("CALCUL DE CONFORMÈRES")
        ================================================================================
        CALCUL DE CONFORMÈRES
        ================================================================================
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width + "\n")


def print_section(title, width=80):
    """
    Affiche un séparateur de section formaté.

    Args:
        title (str): Titre de la section
        width (int): Largeur totale (défaut: 80)

    Examples:
        >>> print_section("Optimisation des conformères")
        --------------------------------------------------------------------------------
        Optimisation des conformères
        --------------------------------------------------------------------------------
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def print_success(message):
    """Affiche un message de succès avec symbole ✓."""
    print(f"✓ {message}")


def print_warning(message):
    """Affiche un message d'avertissement avec symbole ⚠."""
    print(f"⚠ {message}")


def print_error(message):
    """Affiche un message d'erreur avec symbole ✗."""
    print(f"✗ {message}")


def format_parameter_summary(params_dict, indent=2):
    """
    Formate un dictionnaire de paramètres pour l'affichage.

    Args:
        params_dict (dict): Dictionnaire clé-valeur des paramètres
        indent (int): Nombre d'espaces pour l'indentation

    Returns:
        str: Texte formaté des paramètres

    Examples:
        >>> params = {"Functional": "PBE0", "Basis": "DZP", "Temperature": 298.15}
        >>> print(format_parameter_summary(params))
          Functional:  PBE0
          Basis:       DZP
          Temperature: 298.15
    """
    if not params_dict:
        return ""

    # Trouver la longueur maximale des clés
    max_key_len = max(len(str(k)) for k in params_dict.keys())

    lines = []
    prefix = " " * indent
    for key, value in params_dict.items():
        lines.append(f"{prefix}{str(key):<{max_key_len}}: {value}")

    return "\n".join(lines)
