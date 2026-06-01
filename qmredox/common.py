"""
Utilitaires spécifiques au pipeline de calculs redox.

Ce module contient les fonctions et configurations spécifiques aux calculs
de potentiels redox par DFT.
"""

from scm.plams import Settings

# Importation des utilitaires communs
from common.constants import (
    R, F, TEMPERATURE,
    EH_TO_EV, EH_TO_KJ, EV_TO_KJ, KJ_TO_EV,
    hartree_to_ev, hartree_to_kj, ev_to_volt, kj_to_volt
)
from common.functional import (
    configure_functional,
    ALL_FUNCTIONALS,
    LDA_FUNCTIONALS, GGA_FUNCTIONALS, METAGGA_FUNCTIONALS,
    HYBRID_FUNCTIONALS, METAHYBRID_FUNCTIONALS
)
from common.workspace import setup_workspace as common_setup_workspace

# Re-export pour compatibilité avec le code existant
__all__ = [
    # Constantes
    'R', 'F', 'TEMPERATURE',
    'EH_TO_EV', 'EH_TO_KJ', 'EV_TO_KJ', 'KJ_TO_EV',

    # Conversions
    'hartree_to_ev', 'hartree_to_kj', 'ev_to_volt', 'kj_to_volt',

    # Fonctionnelles
    'configure_functional', 'ALL_FUNCTIONALS',
    'LDA_FUNCTIONALS', 'GGA_FUNCTIONALS', 'METAGGA_FUNCTIONALS',
    'HYBRID_FUNCTIONALS', 'METAHYBRID_FUNCTIONALS',

    # Configuration ADF
    'setup_adf_settings',

    # Workspace
    'setup_workspace',

    # Utilitaires géométriques
    'kabsch_rmsd',
]


def setup_workspace(name):
    """
    Configure l'environnement de travail pour les calculs redox.

    Crée la structure :
        <name>_workdir/redox/calculations/
        <name>_workdir/redox/results/

    Args:
        name (str): Nom de la molécule

    Returns:
        Path: Répertoire principal <name>_workdir
    """
    workdir, _, _ = common_setup_workspace(name=name, subfolder="redox", init_plams=True)
    return workdir


def setup_adf_settings(task='GeometryOptimization', charge=0, spin_polarization=0,
                       solvent="Acetonitrile", functional="PBE0", basis="DZP",
                       normal_modes=None, gradients=False, extra_adf_blocks=None):
    """
    Configure un objet Settings pour un calcul ADF.

    Args:
        task (str): Tâche AMS à exécuter (défaut: 'GeometryOptimization')
        charge (int): Charge moléculaire (défaut: 0)
        spin_polarization (float): Polarisation de spin (défaut: 0)
        solvent (str): Nom du solvant pour solvatation COSMO (défaut: "Acetonitrile")
        functional (str): Fonctionnelle DFT (défaut: "PBE0")
        basis (str): Base atomique (défaut: "DZP")
        normal_modes (bool or None): Activer les modes normaux (Hessien).
            None = auto (Yes pour GeomOpt, No pour SinglePoint).
        gradients (bool): Demander explicitement le gradient nucléaire analytique.
            Utile pour un SinglePoint sans NormalModes : le gradient (analytique,
            quasi gratuit) suffit au descripteur vibronique LVC, qui n'utilise que
            le Hessien du neutre. Évite les ~6N évaluations de gradient qu'impose
            un Hessien ionique numérique (PBE0+COSMO).
        extra_adf_blocks (dict or None): Blocs ADF supplémentaires à ajouter.
            Ex: {'ConceptualDFT': {'Enabled': 'Yes', 'AnalysisLevel': 'Full'}}

    Returns:
        Settings: Objet Settings configuré pour le calcul
    """
    s = Settings()

    # Configuration de la tâche AMS
    s.input.ams.Task = task

    # Configuration des modes normaux
    if normal_modes is None:
        # Auto: Yes pour GeomOpt (besoin thermo), No pour SinglePoint
        if task == 'GeometryOptimization':
            s.input.ams.Properties.NormalModes = "Yes"
        # Pour SinglePoint, pas de NormalModes par défaut
    elif normal_modes:
        s.input.ams.Properties.NormalModes = "Yes"

    # Gradient nucléaire analytique (sans le coût d'un Hessien)
    if gradients:
        s.input.ams.Properties.Gradients = "Yes"

    # Configuration du moteur ADF
    s.input.adf.Basis.Type = basis
    s.input.adf.Basis.Core = "None"

    # Configuration de la fonctionnelle
    configure_functional(s, functional)

    # Relativité
    s.input.adf.Relativity.Level = "None"

    # Paramètres pour les molécules chargées
    if charge != 0:
        s.input.ams.System.Charge = charge

        # Aides à la convergence SCF pour les états ioniques (open-shell)
        # Empêche les divergences observées sur les cations TZ2P (ex. Mol_3_ox)
        s.input.adf.SCF.Iterations = 500
        s.input.adf.SCF.Mixing = 0.05
        s.input.adf.SCF.Converge = "1.0e-6 1.0e-6"

    # Paramètres pour les systèmes à couche ouverte
    if spin_polarization > 0:
        s.input.adf.SpinPolarization = spin_polarization
        s.input.adf.Unrestricted = "Yes"

    # Configuration de la solvatation
    from common.frequency import _apply_solvation
    _apply_solvation(s, solvent, charge=charge)

    # Blocs ADF supplémentaires (ConceptualDFT, QTAIM, etc.)
    if extra_adf_blocks:
        for block_name, block_settings in extra_adf_blocks.items():
            for key, value in block_settings.items():
                s.input.adf[block_name][key] = value

    return s


def kabsch_rmsd(mol1, mol2):
    """
    Calcule le RMSD entre deux structures moléculaires après alignement optimal
    utilisant l'algorithme de Kabsch.

    Args:
        mol1, mol2 (Molecule): Objets Molecule de PLAMS à comparer

    Returns:
        float: RMSD après alignement optimal en Å
    """
    import numpy as np

    # Vérifier que les molécules ont le même nombre d'atomes
    if len(mol1) != len(mol2):
        print(f"Erreur : les molécules ont un nombre différent d'atomes "
              f"({len(mol1)} vs {len(mol2)})")
        return float('inf')

    # Extraire les coordonnées sous forme de matrices NumPy
    coords1 = np.array([atom.coords for atom in mol1.atoms])
    coords2 = np.array([atom.coords for atom in mol2.atoms])

    # Centrer les coordonnées (translation au centre de masse)
    centroid1 = coords1.mean(axis=0)
    centroid2 = coords2.mean(axis=0)

    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Matrice de covariance
    H = coords1_centered.T @ coords2_centered

    # Décomposition en valeurs singulières (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Matrice de rotation optimale
    R = Vt.T @ U.T

    # Correction pour éviter les réflexions (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Appliquer la rotation à coords1
    coords1_aligned = coords1_centered @ R

    # Calculer le RMSD
    diff = coords1_aligned - coords2_centered
    rmsd = np.sqrt((diff ** 2).sum() / len(mol1))

    return rmsd
