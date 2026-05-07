#!/usr/bin/env amspython
"""
QTAIM and IQA Analysis Pipeline

Performs QTAIM and IQA calculations on conformers and extracts topological data.

Usage:
    ./qtaim_calc.py --name <molecule_name> [options]
"""

import sys
import argparse
from pathlib import Path

from scm.plams import init, finish

from common.utils import print_header, print_success, print_error, print_section
from qmqtaim.pipeline import QTAIMPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QTAIM/IQA analysis pipeline for conformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  1. Analyse QTAIM/IQA standard:
     ./qtaim_calc.py --name o-Quinone-Meth \\
                     --functional BLYP --basis TZP

  2. Analyse sur conformères spécifiques:
     ./qtaim_calc.py --name molecule \\
                     --conformers conf_1 conf_2 \\
                     --charges 0 -1

  3. QTAIM seul (sans IQA):
     ./qtaim_calc.py --name molecule --no-iqa

  4. Re-analyser des calculs existants (sans refaire les calculs DFT):
     ./qtaim_calc.py --name molecule --analysis-only
        """
    )

    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Nom de la molécule (workdir: <name>_workdir)'
    )

    parser.add_argument(
        '--conformers',
        type=str,
        nargs='+',
        default=None,
        help='Conformères spécifiques à traiter (défaut: tous)'
    )

    parser.add_argument(
        '--charges',
        type=int,
        nargs='+',
        default=[0, -1, +1],
        help='États de charge à calculer (défaut: 0 -1 +1)'
    )

    parser.add_argument(
        '--functional',
        type=str,
        default='BLYP',
        help='Fonctionnelle DFT (défaut: BLYP)'
    )

    parser.add_argument(
        '--basis',
        type=str,
        default='TZP',
        help='Base atomique (défaut: TZP)'
    )

    parser.add_argument(
        '--quality',
        type=str,
        default='Good',
        choices=['Basic', 'Normal', 'Good', 'VeryGood', 'Excellent'],
        help='Qualité numérique (défaut: Good)'
    )

    parser.add_argument(
        '--integration-accuracy',
        type=float,
        default=5.0,
        help='Précision d\'intégration QTAIM (défaut: 5.0)'
    )

    parser.add_argument(
        '--no-iqa',
        action='store_true',
        help='Désactiver les calculs IQA'
    )

    parser.add_argument(
        '--nproc',
        type=int,
        default=1,
        help='Nombre de processus parallèles (défaut: 1)'
    )

    parser.add_argument(
        '--workdir',
        type=str,
        default=None,
        help='Chemin vers le workdir (défaut: ./<name>_workdir)'
    )

    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Analyser des calculs existants sans lancer de nouveaux calculs'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print_header("QTAIM/IQA ANALYSIS PIPELINE")

    # Déterminer le workdir
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        workdir = Path.cwd() / f"{args.name}_workdir"

    if not workdir.exists():
        print_error(f"Le workdir {workdir} n'existe pas!")
        print(f"  Vous devez d'abord exécuter conformers_gen.py et redox_calc.py")
        sys.exit(1)

    print(f"Workdir: {workdir}\n")

    if args.analysis_only:
        print_section("MODE ANALYSE SEULE")
        print("Les RKF existants seront analysés sans nouveaux calculs DFT\n")
    else:
        print("Paramètres de calcul:")
        print(f"  Molécule:              {args.name}")
        print(f"  Fonctionnelle:         {args.functional}")
        print(f"  Base:                  {args.basis}")
        print(f"  Qualité:               {args.quality}")
        print(f"  Précision intégration: {args.integration_accuracy}")
        print(f"  IQA activé:            {not args.no_iqa}")

    print(f"  États de charge:       {args.charges}")
    if args.conformers:
        print(f"  Conformères:           {', '.join(args.conformers)}")
    else:
        print(f"  Conformères:           tous")

    # Initialize PLAMS avec le bon workdir
    from common.workspace import get_pipeline_dirs
    calc_dir, results_dir = get_pipeline_dirs(workdir, "qtaim")

    # PLAMS init() - needed even in analysis mode for KFReader
    init(path=str(calc_dir), folder="plams_workdir")
    print_success("PLAMS initialisé")

    try:
        # Initialize pipeline
        pipeline = QTAIMPipeline(
            workdir=workdir,
            functional=args.functional,
            basis=args.basis,
            quality=args.quality,
            integration_accuracy=args.integration_accuracy,
            run_iqa=not args.no_iqa,
            nproc=args.nproc,
            molecule_name=args.name
        )

        # Run pipeline with analysis_only flag
        results = pipeline.run_pipeline(
            conformer_ids=args.conformers,
            charges=args.charges,
            analysis_only=args.analysis_only
        )

        # Export results
        print_section("Exportation des résultats")
        pipeline.analyzer.export_results(
            results,
            pipeline.results_dir,
            workdir=pipeline.workdir
        )

        print("\n" + "=" * 80)
        print_success("Pipeline terminé avec succès!")
        print(f"Résultats sauvegardés dans: {pipeline.results_dir}/")
        print("=" * 80)

    except Exception as e:
        print_error(f"Le pipeline a échoué: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        finish()
        print_success("PLAMS finalisé")


if __name__ == "__main__":
    sys.exit(main())
