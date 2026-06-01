#!/usr/bin/env amspython
"""
Script principal pour le calcul de potentiels redox par DFT (V5).

Changements V5 :
- Le neutre n'est plus recalculé : réutilisation du RKF de l'étape 4 conformères
- Schéma composite TZ2P+DZP pour les énergies
- Lecture depuis neutral_rkf_paths.json

Usage:
    # Mode calculs complets:
    ./redox_calc.py --input-dir /path/to/conformers/results --name molecule_name [options]

    # Mode analyse seule:
    ./redox_calc.py --analysis-only --workdir /path/to/molecule_workdir --name molecule_name
"""

import os
import sys
import json
import argparse
import glob

from scm.plams import Molecule, AMSJob, finish

from common.utils import print_header, print_success, print_error, print_section
from qmredox.common import setup_workspace, ALL_FUNCTIONALS
from qmredox.pipeline import run_redox_pipeline, run_predict_dg_pipeline
from qmredox.analysis import analyze_molecule_redox, analyze_molecule_predict_dg


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Calcul de potentiels redox par DFT (V5 - composite TZ2P+DZP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  1. Calculs complets à partir de conformères:
     ./redox_calc.py --input-dir ./molecule_workdir/conformers/results \\
                     --name molecule --functional PBE0 --basis DZP

  2. Analyse seule de calculs existants:
     ./redox_calc.py --analysis-only --workdir ./molecule_workdir \\
                     --name molecule --u-method direct

  3. Avec base de raffinement différente:
     ./redox_calc.py --input-dir ./molecule_workdir/conformers/results \\
                     --name molecule --refinement-basis QZ4P
        """
    )

    parser.add_argument('--input-dir', type=str,
                        help="Répertoire conformers/results/ contenant XYZ et neutral_rkf_paths.json")
    parser.add_argument('--name', type=str, required=True,
                        help="Nom de la molécule")

    parser.add_argument('--predict-dg', action='store_true',
                        help="Mode criblage rapide : 3 SP sur géom neutre (NormalModes sur "
                             "le neutre, gradient seul sur les ions) avec ConceptualDFT et "
                             "QTAIM (au lieu des 7 calculs complets)")
    parser.add_argument('--analysis-only', action='store_true',
                        help="Analyser des calculs existants sans lancer de nouveaux calculs")
    parser.add_argument('--workdir', type=str,
                        help="Répertoire de travail (requis en mode --analysis-only)")

    parser.add_argument('--functional', type=str, default='PBE0',
                        choices=ALL_FUNCTIONALS,
                        help="Fonctionnelle DFT (défaut: PBE0)")
    parser.add_argument('--basis', type=str, default='DZP',
                        choices=['SZ', 'DZ', 'DZP', 'TZP', 'TZ2P', 'QZ4P'],
                        help="Base atomique DZP pour opt+SP (défaut: DZP)")
    parser.add_argument('--refinement-basis', type=str, default='TZ2P',
                        choices=['TZP', 'TZ2P', 'QZ4P'],
                        help="Base de raffinement pour SP composites (défaut: TZ2P)")
    parser.add_argument('--solvent', type=str, default='Acetonitrile',
                        help="Solvant pour COSMO (défaut: Acetonitrile)")
    parser.add_argument('--temperature', type=float, default=298.15,
                        help="Température pour Boltzmann (K, défaut: 298.15)")
    parser.add_argument('--u-method', type=str, default='direct',
                        choices=['direct', 'h_minus_rt'],
                        help="Méthode de calcul de U (défaut: direct)")
    parser.add_argument('--max-freq-attempts', type=int, default=3,
                        help="Max tentatives correction fréquences (défaut: 3)")

    args = parser.parse_args()

    if args.analysis_only:
        if not args.workdir:
            parser.error("--workdir est requis en mode --analysis-only")
    else:
        if not args.input_dir:
            parser.error("--input-dir est requis pour lancer des calculs")

    return args


def load_conformers_from_neutral_data(input_dir):
    """
    Charge les conformères depuis conformers/results/ avec neutral_rkf_paths.json.

    Args:
        input_dir (str): Répertoire conformers/results/

    Returns:
        dict: {conf_num: (Molecule, neutral_rkf_path)}
    """
    json_file = os.path.join(input_dir, "neutral_rkf_paths.json")

    if not os.path.exists(json_file):
        raise ValueError(f"neutral_rkf_paths.json non trouvé dans {input_dir}")

    with open(json_file, 'r') as f:
        rkf_paths = json.load(f)

    xyz_files = glob.glob(os.path.join(input_dir, "conf_*.xyz"))

    if not xyz_files:
        raise ValueError(f"Aucun fichier conf_*.xyz trouvé dans {input_dir}")

    conformers = {}

    for xyz_file in xyz_files:
        basename = os.path.basename(xyz_file).replace('.xyz', '')

        try:
            parts = basename.split('_')
            if 'conf' in parts:
                conf_idx = parts.index('conf')
                conf_num = int(parts[conf_idx + 1])
            else:
                continue
        except (ValueError, IndexError):
            continue

        rkf_path = rkf_paths.get(str(conf_num))
        if rkf_path is None:
            print(f"⚠ Pas de RKF neutre pour conformère {conf_num}, ignoré")
            continue

        if not os.path.exists(rkf_path):
            print(f"⚠ RKF neutre introuvable pour conformère {conf_num}: {rkf_path}")
            continue

        mol = Molecule(xyz_file)
        conformers[conf_num] = (mol, rkf_path)

    print_success(f"{len(conformers)} conformère(s) chargé(s) avec RKF neutres depuis {input_dir}")

    return conformers


def load_existing_jobs(workdir, name, predict_dg=False):
    """
    Charge les jobs existants depuis un répertoire de calculs V5.

    Args:
        workdir (str): Répertoire de travail principal
        name (str): Nom de la molécule
        predict_dg (bool): Si True, cherche les jobs du mode predict-dg

    Returns:
        dict: {conf_num: {'neutral_rkf': str, 'neutral_tz2p': job, ...}}
    """
    from pathlib import Path

    calculations_dir = os.path.join(workdir, "redox", "calculations", "plams_workdir")

    if not os.path.exists(calculations_dir):
        raise ValueError(f"Le répertoire de calculs {calculations_dir} n'existe pas")

    rkf_json = Path(workdir) / "conformers" / "results" / "neutral_rkf_paths.json"
    neutral_rkf_paths = {}
    if rkf_json.exists():
        with open(rkf_json, 'r') as f:
            neutral_rkf_paths = json.load(f)

    job_dirs = [d for d in os.listdir(calculations_dir)
                if os.path.isdir(os.path.join(calculations_dir, d)) and
                f"{name}_conf_" in d]

    if not job_dirs:
        raise ValueError(f"Aucun calcul trouvé pour {name} dans {calculations_dir}")

    jobs_data = {}

    for job_dir in job_dirs:
        try:
            parts = job_dir.split('_')
            if 'conf' not in parts:
                continue

            conf_idx = parts.index('conf')
            conf_num = int(parts[conf_idx + 1])

            if predict_dg:
                # Mode predict-dg : neutral_sp, reduced_sp, oxidized_sp
                if 'neutral_sp' in job_dir:
                    job_type = 'neutral_sp'
                elif 'reduced_sp' in job_dir:
                    job_type = 'reduced_sp'
                elif 'oxidized_sp' in job_dir:
                    job_type = 'oxidized_sp'
                else:
                    continue
            else:
                # Mode complet : 7 calculs
                if 'neutral_tz2p' in job_dir:
                    job_type = 'neutral_tz2p'
                elif 'reduced_tz2p' in job_dir:
                    job_type = 'reduced_tz2p'
                elif 'oxidized_tz2p' in job_dir:
                    job_type = 'oxidized_tz2p'
                elif 'reduced' in job_dir:
                    if 'opt' in job_dir:
                        job_type = 'reduced_opt'
                    else:
                        job_type = 'reduced_sp'
                elif 'oxidized' in job_dir:
                    if 'opt' in job_dir:
                        job_type = 'oxidized_opt'
                    else:
                        job_type = 'oxidized_sp'
                else:
                    continue

            job_path = os.path.join(calculations_dir, job_dir)

            if not os.path.exists(os.path.join(job_path, 'ams.rkf')):
                print(f"⚠ Job incomplet ignoré: {job_dir}")
                continue

            job = AMSJob.load_external(job_path)

            if conf_num not in jobs_data:
                jobs_data[conf_num] = {}
                if str(conf_num) in neutral_rkf_paths:
                    jobs_data[conf_num]['neutral_rkf'] = neutral_rkf_paths[str(conf_num)]

            jobs_data[conf_num][job_type] = job

        except (ValueError, IndexError) as e:
            print(f"⚠ Impossible de parser {job_dir}: {str(e)}")
            continue

    print_success(f"{len(jobs_data)} conformère(s) avec calculs existants chargé(s)")

    return jobs_data


def main():
    """Fonction principale du programme."""
    args = parse_arguments()

    print_header("CALCUL DE POTENTIELS REDOX PAR DFT (V5)")

    # MODE ANALYSE SEULE
    if args.analysis_only:
        print_section("MODE ANALYSE SEULE")
        print(f"Répertoire de travail: {args.workdir}")
        print(f"Température: {args.temperature} K")
        if not args.predict_dg:
            print(f"Méthode U: {args.u_method}")
        print(f"Mode predict-dg: {'Oui' if args.predict_dg else 'Non'}\n")

        try:
            jobs_data = load_existing_jobs(args.workdir, args.name,
                                           predict_dg=args.predict_dg)

            if args.predict_dg:
                results = analyze_molecule_predict_dg(
                    mol_name=args.name,
                    jobs_data=jobs_data,
                    workdir=args.workdir,
                    temperature=args.temperature
                )
            else:
                results = analyze_molecule_redox(
                    mol_name=args.name,
                    jobs_data=jobs_data,
                    workdir=args.workdir,
                    temperature=args.temperature,
                    u_method=args.u_method
                )

            print("\n" + "=" * 80)
            print_success("Analyse terminée avec succès")
            print(f"Résultats sauvegardés dans: {args.workdir}/redox/results/")
            print("=" * 80)
            return

        except Exception as e:
            print_error(f"Erreur lors de l'analyse: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # MODE CALCULS
    if args.predict_dg:
        print_section("MODE PREDICT-DG (criblage rapide)")
    else:
        print_section("MODE CALCULS COMPLETS")

    print(f"\nParamètres de calcul:")
    print(f"  Molécule:           {args.name}")
    print(f"  Mode:               {'predict-dg' if args.predict_dg else 'complet (composite TZ2P+DZP)'}")
    print(f"  Fonctionnelle:      {args.functional}")
    print(f"  Base:               {args.basis}")
    if not args.predict_dg:
        print(f"  Base raffinement:   {args.refinement_basis}")
    print(f"  Solvant:            {args.solvent}")
    print(f"  Température:        {args.temperature} K")
    if not args.predict_dg:
        print(f"  Méthode U:          {args.u_method}")
        print(f"  Max freq attempts:  {args.max_freq_attempts}")
    if args.predict_dg:
        print(f"  ConceptualDFT:      Activé (Full)")
        print(f"  QTAIM:              Activé (Full)")

    # Configuration de l'environnement
    workdir = setup_workspace(args.name)
    print_success(f"Environnement de travail configuré: {workdir}")

    # Chargement des conformères avec RKF neutres
    conformers = load_conformers_from_neutral_data(args.input_dir)

    # Exécution du pipeline pour chaque conformère
    all_jobs = {}

    for conf_num, (mol_neutral, neutral_rkf_path) in sorted(conformers.items()):
        print(f"\n{'='*80}")
        print(f"TRAITEMENT DU CONFORMÈRE {conf_num}")
        print(f"{'='*80}")

        try:
            job_name = f"{args.name}_conf_{conf_num}"

            if args.predict_dg:
                jobs = run_predict_dg_pipeline(
                    mol_neutral=mol_neutral,
                    neutral_rkf_path=neutral_rkf_path,
                    name=job_name,
                    solvent=args.solvent,
                    functional=args.functional,
                    basis=args.basis
                )
            else:
                jobs = run_redox_pipeline(
                    mol_neutral=mol_neutral,
                    neutral_rkf_path=neutral_rkf_path,
                    name=job_name,
                    solvent=args.solvent,
                    functional=args.functional,
                    basis=args.basis,
                    refinement_basis=args.refinement_basis,
                    max_attempts=args.max_freq_attempts
                )

            all_jobs[conf_num] = jobs

        except Exception as e:
            print_error(f"Erreur lors du traitement du conformère {conf_num}")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Finaliser PLAMS
    finish()

    if not all_jobs:
        print_error("Aucun conformère n'a été calculé avec succès")
        return

    # Lancer l'analyse
    print_header("ANALYSE DES RÉSULTATS")

    if args.predict_dg:
        results = analyze_molecule_predict_dg(
            mol_name=args.name,
            jobs_data=all_jobs,
            workdir=workdir,
            temperature=args.temperature
        )
    else:
        results = analyze_molecule_redox(
            mol_name=args.name,
            jobs_data=all_jobs,
            workdir=workdir,
            temperature=args.temperature,
            u_method=args.u_method
        )

    print("\n" + "=" * 80)
    print_success("PROGRAMME TERMINÉ AVEC SUCCÈS")
    print("=" * 80)

    print(f"\nRésultats disponibles dans: {os.path.join(workdir, 'redox', 'results')}")


if __name__ == "__main__":
    main()
