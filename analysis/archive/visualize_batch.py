#!/usr/bin/env python3
"""
Génération de visualisations comparatives pour plusieurs molécules.

Ce script génère des graphiques comparant plusieurs molécules entre elles,
où chaque point représente une molécule différente (moyennes de Boltzmann).

Usage:
    # Workdirs explicites
    ./visualize_batch.py --workdirs mol1_workdir mol2_workdir mol3_workdir

    # Auto-découverte
    ./visualize_batch.py --batch-dir /path/to/job_dir

    # Avec répertoire de sortie personnalisé
    ./visualize_batch.py --batch-dir . --output-dir my_viz

Examples:
    # Comparer DHI et O-DHI
    ./visualize_batch.py --workdirs 1592750/DHI_workdir 1592750/O-DHI_workdir

    # Auto-découverte dans un répertoire de job
    ./visualize_batch.py --batch-dir 1592750

    # Sortie personnalisée
    ./visualize_batch.py --batch-dir . --output-dir custom_batch_viz
"""

import argparse
import sys
from pathlib import Path

# Ajouter la racine du projet au path pour les imports
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

# Import direct pour éviter common/__init__.py qui charge PLAMS
import importlib.util
import os

# Import direct de multi_molecule_data
spec_data = importlib.util.spec_from_file_location(
    "multi_molecule_data",
    os.path.join(_project_root, "common", "multi_molecule_data.py")
)
if spec_data is None or spec_data.loader is None:
    print("✗ Impossible de charger common/multi_molecule_data.py")
    sys.exit(1)
multi_molecule_data = importlib.util.module_from_spec(spec_data)
spec_data.loader.exec_module(multi_molecule_data)

collect_all_molecules = multi_molecule_data.collect_all_molecules

# Import direct de batch_correlations
spec_viz = importlib.util.spec_from_file_location(
    "batch_correlations",
    os.path.join(_project_root, "visualization", "batch_correlations.py")
)
if spec_viz is None or spec_viz.loader is None:
    print("✗ Impossible de charger visualization/batch_correlations.py")
    sys.exit(1)
batch_correlations = importlib.util.module_from_spec(spec_viz)
spec_viz.loader.exec_module(batch_correlations)

generate_batch_plots = batch_correlations.generate_batch_plots


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def print_header(title, width=80):
    """Affiche un header formaté."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width + "\n")


def print_section(title, width=80):
    """Affiche un séparateur de section formaté."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def print_success(message):
    """Affiche un message de succès avec symbole ✓."""
    print(f"✓ {message}")


def print_error(message):
    """Affiche un message d'erreur avec symbole ✗."""
    print(f"✗ {message}")


def print_warning(message):
    """Affiche un avertissement avec symbole ⚠."""
    print(f"⚠ {message}")


def discover_workdirs(batch_dir: Path):
    """
    Trouve tous les *_workdir dans batch_dir.

    Args:
        batch_dir: Répertoire contenant les workdirs

    Returns:
        Liste de Path des workdirs trouvés
    """
    workdirs = [d for d in batch_dir.glob("*_workdir") if d.is_dir()]
    return sorted(workdirs)


def parse_arguments():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(
        description="Génération de visualisations batch PlamsV4 (comparaisons inter-molécules)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  1. Workdirs explicites:
     ./visualize_batch.py --workdirs DHI_workdir O-DHI_workdir

  2. Auto-découverte:
     ./visualize_batch.py --batch-dir 1592750

  3. Sortie personnalisée:
     ./visualize_batch.py --batch-dir . --output-dir my_visualizations

Graphiques générés:
  - batch_lumo_vs_reduction.png      : LUMO vs potentiel de réduction
  - batch_bader_o7_vs_reduction.png  : Charge O7 vs réduction
  - batch_gap_vs_reactivity.png      : Gap HOMO-LUMO vs EA/IP
  - batch_correlation_heatmap.png    : Matrice de corrélation complète
  - molecules_summary.csv            : Données consolidées (CSV)

Note:
  Chaque point sur les graphiques représente UNE MOLÉCULE (pas un conformère).
  Les valeurs sont les moyennes de Boltzmann sur les conformères de chaque molécule.

Dépendances requises:
  - matplotlib (obligatoire)
  - pandas (obligatoire)
  - scipy (pour régressions linéaires)
  - seaborn (optionnel, pour heatmap améliorée)
        """
    )

    # Mode de sélection des workdirs
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--workdirs',
        nargs='+',
        help="Liste explicite de workdirs à comparer"
    )
    group.add_argument(
        '--batch-dir',
        type=str,
        help="Répertoire contenant les workdirs (auto-découverte de *_workdir)"
    )

    # Options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='batch_visualization',
        help="Répertoire de sortie (défaut: batch_visualization)"
    )

    args = parser.parse_args()

    # Convertir en Path
    if args.workdirs:
        args.workdirs = [Path(w) for w in args.workdirs]
    if args.batch_dir:
        args.batch_dir = Path(args.batch_dir)
    args.output_dir = Path(args.output_dir)

    return args


def check_dependencies():
    """
    Vérifie la présence des dépendances nécessaires.

    Returns:
        dict: Status des dépendances {package: bool}
    """
    dependencies = {
        'matplotlib': False,
        'pandas': False,
        'scipy': False,
        'seaborn': False
    }

    for package in dependencies.keys():
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False

    return dependencies


def main():
    """Fonction principale du programme."""
    args = parse_arguments()

    print_header("GÉNÉRATION DE VISUALISATIONS BATCH PLAMSV4")

    # Vérifier les dépendances
    print_section("VÉRIFICATION DES DÉPENDANCES")
    deps = check_dependencies()

    for package, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {package}")

    # Vérifier les dépendances critiques
    if not deps['matplotlib']:
        print_error("Matplotlib est requis pour générer les visualisations")
        print("  Installation: pip install matplotlib")
        sys.exit(1)

    if not deps['pandas']:
        print_error("Pandas est requis pour la manipulation des données")
        print("  Installation: pip install pandas")
        sys.exit(1)

    if not deps['scipy']:
        print_error("SciPy est requis pour les régressions linéaires")
        print("  Installation: pip install scipy")
        sys.exit(1)

    # Avertissements pour dépendances optionnelles
    if not deps['seaborn']:
        print("\n⚠ Seaborn non disponible: heatmap utilisera matplotlib pur")
        print("  Installation: pip install seaborn")

    print()

    # Déterminer la liste des workdirs
    if args.workdirs:
        workdirs = args.workdirs
        print_section("WORKDIRS SPÉCIFIÉS")
        for w in workdirs:
            if not w.exists():
                print_error(f"Workdir non trouvé: {w}")
                sys.exit(1)
            print(f"  → {w}")
    else:
        print_section("AUTO-DÉCOUVERTE DES WORKDIRS")
        print(f"Répertoire: {args.batch_dir}")
        workdirs = discover_workdirs(args.batch_dir)

        if not workdirs:
            print_error(f"Aucun workdir trouvé dans {args.batch_dir}")
            sys.exit(1)

        print(f"\n✓ {len(workdirs)} workdir(s) trouvé(s):")
        for w in workdirs:
            print(f"  → {w.name}")

    print()
    print_section("COLLECTE DES DONNÉES")
    print(f"Nombre de molécules: {len(workdirs)}\n")

    # Collecter les données de toutes les molécules
    try:
        df = collect_all_molecules(workdirs)
    except Exception as e:
        print_error(f"Erreur lors de la collecte des données: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n✓ {len(df)} molécule(s) chargée(s) avec succès")
    print(f"  Colonnes disponibles: {len(df.columns)}")
    print(f"  Molécules: {', '.join(df['molecule_name'].tolist())}")

    # Générer les visualisations
    print()
    try:
        generate_batch_plots(df, args.output_dir)
    except Exception as e:
        print_error(f"Erreur lors de la génération: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print_success(f"Résultats sauvegardés dans: {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
