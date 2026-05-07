"""
Gestion unifiée des espaces de travail (workdirs) pour tous les pipelines.

Ce module fournit une fonction standardisée pour créer et organiser
les répertoires de travail utilisés par les différents pipelines.
"""

import os
from pathlib import Path


def setup_workspace(name, subfolder=None, init_plams=False):
    """
    Configure l'environnement de travail pour un pipeline de calcul.

    Cette fonction crée une structure de répertoires standardisée :

    <name>_workdir/
    ├── conformers/
    │   ├── calculations/
    │   └── results/
    ├── redox/
    │   ├── calculations/
    │   └── results/
    └── qtaim/
        ├── calculations/
        └── results/

    Args:
        name (str): Nom de la molécule (utilisé pour créer <name>_workdir)
        subfolder (str, optional): Sous-dossier spécifique à créer
                                   ('conformers', 'redox', 'qtaim').
                                   Si None, crée la structure complète.
        init_plams (bool): Si True, initialise PLAMS dans calculations/
                          (défaut: False)

    Returns:
        tuple: (workdir, calc_dir, results_dir)
               - workdir (Path): Répertoire principal <name>_workdir
               - calc_dir (Path): Répertoire calculations/
               - results_dir (Path): Répertoire results/

    Examples:
        >>> # Pour le pipeline conformers
        >>> workdir, calc_dir, results_dir = setup_workspace("o-Quinone", "conformers")

        >>> # Pour créer toute la structure
        >>> workdir, _, _ = setup_workspace("o-Quinone")
    """
    # Nettoyer le nom de la molécule pour le nom de dossier
    clean_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in name)

    # Créer le répertoire principal
    workdir_name = f"{clean_name}_workdir"

    # Si on est déjà dans un dossier qui se termine par _workdir, on l'utilise
    current_dir = Path.cwd()
    if current_dir.name.endswith('_workdir'):
        workdir = current_dir
    else:
        workdir = current_dir / workdir_name

    # Créer le répertoire principal s'il n'existe pas
    workdir.mkdir(exist_ok=True)

    # Si subfolder est spécifié, créer uniquement ce sous-dossier
    if subfolder:
        subfolder_dir = workdir / subfolder
        calc_dir = subfolder_dir / "calculations"
        results_dir = subfolder_dir / "results"

        calc_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Créer toute la structure pour tous les pipelines
        for pipeline in ['conformers', 'redox', 'qtaim', 'screening']:
            pipeline_dir = workdir / pipeline
            (pipeline_dir / "calculations").mkdir(parents=True, exist_ok=True)
            (pipeline_dir / "results").mkdir(parents=True, exist_ok=True)

        # Par défaut, retourner les chemins conformers
        calc_dir = workdir / "conformers" / "calculations"
        results_dir = workdir / "conformers" / "results"

    # Initialiser PLAMS si demandé
    if init_plams:
        from scm.plams import init
        init(path=str(calc_dir))

    return workdir, calc_dir, results_dir


def get_pipeline_dirs(workdir, pipeline):
    """
    Récupère les répertoires calculations/ et results/ d'un pipeline spécifique.

    Args:
        workdir (Path or str): Répertoire principal <name>_workdir
        pipeline (str): Nom du pipeline ('conformers', 'redox', 'qtaim')

    Returns:
        tuple: (calc_dir, results_dir) en tant que Path objects

    Raises:
        ValueError: Si le pipeline n'est pas reconnu

    Examples:
        >>> calc_dir, results_dir = get_pipeline_dirs("o-Quinone_workdir", "redox")
    """
    workdir = Path(workdir)

    if pipeline not in ['conformers', 'redox', 'qtaim', 'screening']:
        raise ValueError(f"Pipeline '{pipeline}' non reconnu. "
                        f"Utilisez 'conformers', 'redox', 'qtaim', ou 'screening'.")

    pipeline_dir = workdir / pipeline
    calc_dir = pipeline_dir / "calculations"
    results_dir = pipeline_dir / "results"

    # Créer les répertoires s'ils n'existent pas
    calc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return calc_dir, results_dir


def print_workspace_info(workdir, calc_dir, results_dir, pipeline_name):
    """
    Affiche les informations sur l'espace de travail de manière standardisée.

    Args:
        workdir (Path): Répertoire principal
        calc_dir (Path): Répertoire de calculs
        results_dir (Path): Répertoire de résultats
        pipeline_name (str): Nom du pipeline pour l'affichage
    """
    print(f"✓ Environnement de travail configuré:")
    print(f"  Workdir principal:      {workdir}")
    print(f"  Calculs {pipeline_name:10s}: {calc_dir}")
    print(f"  Résultats {pipeline_name:10s}: {results_dir}")
