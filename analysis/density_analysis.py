#!/usr/bin/env amspython
"""
Analyse de correlation densite electronique locale (QTAIM) vs EA/IP vertical.

Ce script extrait les proprietes de densite electronique locale depuis les
calculs QTAIM neutres, calcule des descripteurs et analyse leurs correlations
avec les affinites electroniques (EA) et potentiels d'ionisation (IP) verticaux.

Usage:
    ./density_analysis.py --batch-dir 1594174/dir_workdir --output-dir density_analysis
    ./density_analysis.py --workdirs mol1_workdir mol2_workdir --top-n 15
    ./density_analysis.py --batch-dir workdirs/ --no-plots
"""

import sys
import os
import argparse
from pathlib import Path

# Ajouter la racine du projet au path pour les imports
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

from qmanalysis.qtaim_correlation import run_density_analysis
from common.utils import print_header, print_success, print_error


def find_workdirs_in_batch(batch_dir: Path) -> list:
    """
    Trouve tous les repertoires *_workdir dans un repertoire batch.

    Args:
        batch_dir: Repertoire contenant les workdirs

    Returns:
        Liste triee des chemins vers les workdirs
    """
    workdirs = []
    for entry in sorted(batch_dir.iterdir()):
        if entry.is_dir() and entry.name.endswith('_workdir'):
            # Verifier qu'il y a un sous-repertoire qtaim
            qtaim_dir = entry / "qtaim"
            if qtaim_dir.exists():
                workdirs.append(entry)
    return workdirs


def main():
    parser = argparse.ArgumentParser(
        description="Analyse de correlation densite QTAIM vs EA/IP vertical"
    )

    # Sources de donnees (mutuellement exclusifs)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--batch-dir',
        type=str,
        help="Repertoire contenant les *_workdir (mode batch)"
    )
    source.add_argument(
        '--workdirs',
        type=str,
        nargs='+',
        help="Liste explicite de repertoires de travail"
    )

    # Options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='density_analysis',
        help="Repertoire de sortie (defaut: density_analysis)"
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help="Nombre de top correlations a afficher (defaut: 10)"
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Desactiver la generation de graphiques"
    )

    args = parser.parse_args()

    # Determiner les workdirs
    if args.batch_dir:
        batch_path = Path(args.batch_dir)
        if not batch_path.exists():
            print_error(f"Repertoire batch non trouve: {batch_path}")
            sys.exit(1)
        workdirs = find_workdirs_in_batch(batch_path)
        if not workdirs:
            print_error(f"Aucun *_workdir avec qtaim/ trouve dans {batch_path}")
            sys.exit(1)
        print(f"Mode batch: {len(workdirs)} molecules trouvees dans {batch_path}")
    else:
        workdirs = [Path(w) for w in args.workdirs]
        for w in workdirs:
            if not w.exists():
                print_error(f"Workdir non trouve: {w}")
                sys.exit(1)

    output_dir = Path(args.output_dir)

    # Lancer l'analyse
    print_header("ANALYSE DENSITE QTAIM vs EA/IP")
    print(f"  Molecules: {len(workdirs)}")
    print(f"  Sortie: {output_dir}")
    print(f"  Top-N: {args.top_n}")
    print(f"  Plots: {'Non' if args.no_plots else 'Oui'}")
    print()

    try:
        df_corr_ea, df_desc = run_density_analysis(workdirs, output_dir)

        if not args.no_plots:
            # Import visualization lazily to avoid transitive pandas dependency
            import importlib.util
            _vis_spec = importlib.util.spec_from_file_location(
                "density_correlations",
                os.path.join(str(Path(os.path.abspath(__file__)).parent.parent),
                             "visualization", "density_correlations.py")
            )
            _vis_mod = importlib.util.module_from_spec(_vis_spec)
            _vis_spec.loader.exec_module(_vis_mod)

            generate_all_density_plots = _vis_mod.generate_all_density_plots

            # Charger aussi les correlations IP
            ip_csv = output_dir / "density_correlations_IP.csv"
            try:
                import pandas as pd
                if ip_csv.exists():
                    df_corr_ip = pd.read_csv(ip_csv)
                else:
                    df_corr_ip = df_corr_ea  # Fallback
            except ImportError:
                df_corr_ip = df_corr_ea

            generate_all_density_plots(
                df_corr_ea, df_corr_ip, df_desc,
                output_dir, top_n=args.top_n
            )

        print_success(f"\nAnalyse terminee. Resultats dans {output_dir}/")

    except Exception as e:
        print_error(f"Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
