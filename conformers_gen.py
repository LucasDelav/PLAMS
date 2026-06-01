#!/usr/bin/env amspython
# -*- coding: utf-8 -*-
"""
Pipeline de génération et optimisation de conformères.

Pipeline simplifié en 4 étapes :
1. Generate RDKit+UFF
2. Optimize GFN1-xTB
3. Boltzmann filter (seuil 0.05)
4. DFT opt + freq + solvant (obligatoire) + Boltzmann DFT filter (seuil 0.10)

Usage:
    ./conformers_gen.py --name <molecule_name> --smiles "<SMILES>" [options]
"""

import os
import sys
import json
import argparse

from scm.plams import init, finish, from_smiles

from common.utils import print_header, print_success, print_error, print_warning, boltzmann_weights
from common.constants import EH_TO_KJ
from qmconf.common import UNIT, TEMPERATURE
from qmconf.pipeline import (
    setup_workspace,
    generate_conformers,
    optimize_conformers,
    print_results,
    filter_boltzmann_weights,
    dft_optimize_with_freq,
    optimize_molecules_xtb,
    compute_dft_boltzmann_weights,
)


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline de génération et optimisation de conformères",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  1. Génération standard:
     ./conformers_gen.py --name molecule --smiles "CC(=O)O"

  2. Avec paramètres personnalisés:
     ./conformers_gen.py --name molecule --smiles "CC(=O)O" \\
                         --functional PBE0 --basis TZP \\
                         --boltzmann-min-weight 0.05 --solvent Water

  3. Re-analyser des conformères existants (sans refaire calculs):
     ./conformers_gen.py --name molecule --analysis-only --workdir ./molecule_workdir
        """
    )

    # Identité molécule
    parser.add_argument("--name", required=True, help="Nom de la molécule")
    parser.add_argument("--smiles", required=False,
                        help="SMILES d'entrée (requis pour mode calcul complet)")

    # Mode analyse seule
    parser.add_argument("--analysis-only", action="store_true",
                        help="Ré-analyser des conformères existants sans refaire les calculs")
    parser.add_argument("--workdir", type=str,
                        help="Répertoire de travail (requis en mode --analysis-only)")
    parser.add_argument("--subfolder", type=str, default="boltzmann_filtered",
                        help="Sous-dossier source pour --analysis-only (défaut: boltzmann_filtered)")

    # Paramètres de calcul DFT
    parser.add_argument("--functional", default="PBE0",
                        help="Fonctionnelle DFT (défaut: PBE0)")
    parser.add_argument("--basis", choices=["SZ", "DZ", "DZP", "TZP", "TZ2P"],
                        default="DZP", help="Base ADF (défaut: DZP)")

    # Boltzmann
    parser.add_argument("--boltzmann-min-weight", type=float, default=0.05,
                        help="Poids de Boltzmann minimal après xTB (défaut: 0.05)")
    parser.add_argument("--boltzmann-min-weight-dft", type=float, default=0.10,
                        help="Poids de Boltzmann minimal après DFT (défaut: 0.10)")

    # Solvant
    parser.add_argument("--solvent", type=str, default="Acetonitrile",
                        help="Solvant COSMO pour l'étape DFT (défaut: Acetonitrile)")

    # Cluster-continuum
    parser.add_argument("--cluster-continuum", action="store_true",
                        help="Activer le mode cluster-continuum pour solvants protiques (Water, Methanol)")

    # Paramètres d'analyse
    parser.add_argument("--temperature", type=float, default=298.15,
                        help="Température pour poids de Boltzmann (K, défaut: 298.15)")

    args = parser.parse_args()

    # Validation
    if args.analysis_only:
        if not args.workdir:
            parser.error("--workdir est requis en mode --analysis-only")
    else:
        if not args.smiles:
            parser.error("--smiles est requis pour le mode calcul complet")

    return args


def main():
    """Fonction principale du programme."""
    args = parse_arguments()

    print_header("PIPELINE DE GÉNÉRATION DE CONFORMÈRES")

    # =======================================================================
    # MODE ANALYSE SEULE
    # =======================================================================
    if args.analysis_only:
        from pathlib import Path
        from common.utils import print_section
        from qmconf.analysis import (
            analyze_existing_conformers,
            export_conformers_summary,
            export_conformers_csv
        )

        print_section("MODE ANALYSE SEULE")
        print(f"Répertoire de travail: {args.workdir}")
        print(f"Sous-dossier source: {args.subfolder}")
        print(f"Température: {args.temperature} K\n")

        workdir = Path(args.workdir)

        if not workdir.exists():
            print_error(f"Le workdir {workdir} n'existe pas!")
            sys.exit(1)

        try:
            results = analyze_existing_conformers(
                workdir,
                temperature=args.temperature,
                subfolder=args.subfolder
            )

            print("\n" + "=" * 80)
            print("RÉSULTATS DE L'ANALYSE")
            print("=" * 80)
            print(f"Température: {results['temperature']:.2f} K")
            print(f"Fonction de partition: Z = {results['partition_function']:.6e}")
            print(f"Nombre de conformères: {len(results['conformers'])}\n")

            print(f"{'#':<5} {'ΔE [kJ/mol]':<15} {'Poids':<12} {'%':<10}")
            print("-" * 80)

            for conf_num in sorted(results['weights'].keys()):
                energy = results['relative_energies_kj'][conf_num]
                weight = results['weights'][conf_num]
                percentage = weight * 100
                print(f"{conf_num:<5} {energy:>10.3f}     {weight:>10.6f}  {percentage:>8.2f}%")

            print("=" * 80)

            results_dir = workdir / "conformers" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            summary_file = results_dir / f"conformers_reanalysis_{args.subfolder}.txt"
            export_conformers_summary(results, summary_file, args.name)

            csv_file = results_dir / f"conformers_data_{args.subfolder}.csv"
            export_conformers_csv(results, csv_file)

            print("\n" + "=" * 80)
            print_success("Analyse terminée avec succès!")
            print(f"Résultats sauvegardés dans: {results_dir}/")
            print("=" * 80)

            return

        except Exception as e:
            print_error(f"Erreur lors de l'analyse: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # =======================================================================
    # MODE CALCULS COMPLETS
    # =======================================================================

    # Determiner si le mode cluster-continuum est actif
    use_cluster_continuum = False
    if args.cluster_continuum:
        from common.cluster_continuum import needs_cluster_continuum
        if needs_cluster_continuum(args.solvent):
            use_cluster_continuum = True
            print_success(f"Mode cluster-continuum activé pour {args.solvent}")
        else:
            print_warning(f"--cluster-continuum ignoré: {args.solvent} n'est pas un solvant protique")

    # Configuration de l'espace de travail
    workdir, calc_dir, results_dir = setup_workspace(args.name)

    print(f"Dossier de travail: {workdir}\n")
    print("Paramètres de calcul:")
    print(f"  Molécule:         {args.name}")
    print(f"  SMILES:           {args.smiles}")
    print(f"  Fonctionnelle:    {args.functional}")
    print(f"  Base:             {args.basis}")
    print(f"  Solvant:          {args.solvent}")
    print(f"  Cluster-cont.:    {'Oui' if use_cluster_continuum else 'Non'}")
    print(f"  Boltz. xTB:       {args.boltzmann_min_weight}")
    print(f"  Boltz. DFT:       {args.boltzmann_min_weight_dft}")
    print(f"  Temperature:      {TEMPERATURE} K")
    print(f"  Unit interne:     {UNIT}")

    # Initialisation PLAMS
    init(path=str(calc_dir))

    try:
        # Construction de la molécule depuis SMILES
        try:
            mol = from_smiles(args.smiles)
            mol.properties.name = args.name
            print_success("Molécule créée depuis SMILES")
        except Exception as e:
            print_error(f"Erreur lors de la création de la molécule: {e}")
            sys.exit(1)

        if use_cluster_continuum:
            _run_cluster_continuum_pipeline(args, mol, calc_dir, results_dir)
        else:
            _run_standard_pipeline(args, mol, calc_dir, results_dir)

    except RuntimeError as e:
        print_error(f"Erreur d'exécution: {e}")
        sys.exit(2)
    except Exception as e:
        import traceback
        print_error(f"Erreur inattendue: {e}")
        print(traceback.format_exc())
        sys.exit(3)
    finally:
        finish()
        print_success("PLAMS finalisé")


def _run_standard_pipeline(args, mol, calc_dir, results_dir):
    """Pipeline standard en 4 étapes (sans cluster-continuum)."""

    # Étape 1: Génération des conformères
    gen_job = generate_conformers(mol, calc_dir)
    print_results(gen_job)

    # Étape 2: Optimisation GFN1-xTB
    opt_job = optimize_conformers(gen_job, calc_dir)
    print_results(opt_job)

    # Étape 3: Filtrage Boltzmann xTB
    _, selected_conformers = filter_boltzmann_weights(
        opt_job, calc_dir, weight_threshold=args.boltzmann_min_weight
    )

    # Export des conformères retenus par Boltzmann
    export_dir = os.path.join(calc_dir, "boltzmann_filtered")
    os.makedirs(export_dir, exist_ok=True)

    for j, conf in enumerate(selected_conformers):
        xyz_file = os.path.join(export_dir, f"conf_{j+1}.xyz")
        conf.write(xyz_file)

    print_success(f"Conformères exportés dans: {export_dir}")

    # Étape 4: Optimisation DFT + fréquences + solvant (obligatoire)
    valid_conformers, valid_conf_indices, rkf_paths = dft_optimize_with_freq(
        selected_conformers, args.name, calc_dir,
        functional=args.functional,
        basis=args.basis,
        solvent=args.solvent,
        max_attempts=3
    )

    # Recalculer les poids de Boltzmann au niveau DFT et filtrer
    filtered = _filter_dft_boltzmann(
        valid_conformers, valid_conf_indices, rkf_paths,
        args.boltzmann_min_weight_dft
    )

    _export_results(
        args, results_dir, **filtered,
        n_after_xtb=len(opt_job.results.get_conformers()),
        n_after_boltzmann=len(selected_conformers),
    )


def _run_cluster_continuum_pipeline(args, mol, calc_dir, results_dir):
    """Pipeline cluster-continuum en 3 phases pour solvants protiques."""
    from common.cluster_continuum import (
        extract_xtb_charges,
        build_cluster_continuum_system,
    )
    from common.utils import print_section

    # ===================================================================
    # PHASE 1 : Phase gazeuse (étapes 1-2-3)
    # ===================================================================
    print_section("PHASE 1 : Génération de conformères en phase gazeuse")

    phase1_dir = os.path.join(calc_dir, "phase1")
    os.makedirs(phase1_dir, exist_ok=True)

    # Étape 1: Génération des conformères
    gen_job = generate_conformers(mol, phase1_dir)
    print_results(gen_job)

    # Étape 2: Optimisation GFN1-xTB
    opt_job = optimize_conformers(gen_job, phase1_dir)
    print_results(opt_job)

    # Étape 3: Filtrage Boltzmann xTB
    _, selected_conformers = filter_boltzmann_weights(
        opt_job, phase1_dir, weight_threshold=args.boltzmann_min_weight
    )

    n_after_xtb = len(opt_job.results.get_conformers())
    n_after_boltzmann = len(selected_conformers)

    # ===================================================================
    # INTERMEDIAIRE : Charges xTB + placement solvant
    # ===================================================================
    print_section("INTERMEDIAIRE : Extraction charges et placement solvant")

    phase2_dir = os.path.join(calc_dir, "phase2")
    os.makedirs(phase2_dir, exist_ok=True)

    # Charges xTB du conformère le plus stable
    print("  Extraction des charges xTB (conformère le plus stable)...")
    charges = extract_xtb_charges(selected_conformers[0], phase2_dir)
    print_success(f"  {len(charges)} charges extraites")

    # Construction des systèmes augmentés
    print(f"\n  Construction des systèmes augmentés ({args.solvent})...")
    augmented = []
    for i, conf in enumerate(selected_conformers):
        print(f"\n  → Conformère {i+1}/{len(selected_conformers)}:")
        aug = build_cluster_continuum_system(conf, charges, args.solvent)
        augmented.append(aug)

    # Export XYZ des systèmes augmentés (pour vérification visuelle)
    aug_export_dir = os.path.join(phase2_dir, "augmented_xyz")
    os.makedirs(aug_export_dir, exist_ok=True)
    for i, aug in enumerate(augmented):
        aug.write(os.path.join(aug_export_dir, f"augmented_{i+1}.xyz"))
    print_success(f"  Systèmes augmentés exportés dans: {aug_export_dir}")

    # ===================================================================
    # PHASE 2 : Pipeline sur systèmes augmentés (étapes 2-3-4)
    # ===================================================================
    print_section("PHASE 2 : Optimisation des systèmes augmentés")

    # Étape 2 (Phase 2): xTB opt (jobs individuels, pas ConformersJob)
    optimized = optimize_molecules_xtb(augmented, phase2_dir)

    if not optimized:
        raise RuntimeError("Aucun système augmenté n'a pu être optimisé")

    # Étape 3 (Phase 2): Boltzmann filter manuel (seuil xTB)
    print(f"\n[Phase 2 - Étape 3] Filtrage Boltzmann des systèmes augmentés")
    e_min = optimized[0][1]  # Déjà trié par énergie
    energies_kj = [(e - e_min) * EH_TO_KJ for _, e in optimized]
    weights, _ = boltzmann_weights(energies_kj, TEMPERATURE)

    selected_phase2 = []
    for (opt_mol, _e), w in zip(optimized, weights):
        if w >= args.boltzmann_min_weight:
            selected_phase2.append(opt_mol)

    print(f"  {len(selected_phase2)} systèmes retenus sur {len(optimized)} (w >= {args.boltzmann_min_weight})")

    if not selected_phase2:
        raise RuntimeError("Aucun système augmenté ne passe le filtre Boltzmann")

    # Étape 4 (Phase 2): DFT opt+freq+COSMO (fonction existante)
    valid_conformers, valid_conf_indices, rkf_paths = dft_optimize_with_freq(
        selected_phase2, args.name, phase2_dir,
        functional=args.functional,
        basis=args.basis,
        solvent=args.solvent,
        max_attempts=3
    )

    # Recalculer les poids de Boltzmann au niveau DFT et filtrer
    filtered = _filter_dft_boltzmann(
        valid_conformers, valid_conf_indices, rkf_paths,
        args.boltzmann_min_weight_dft
    )

    _export_results(
        args, results_dir, **filtered,
        n_after_xtb=n_after_xtb,
        n_after_boltzmann=n_after_boltzmann,
        cluster_continuum=True,
        n_augmented=len(augmented),
        n_selected_phase2=len(selected_phase2),
    )


def _filter_dft_boltzmann(valid_conformers, valid_conf_indices, rkf_paths,
                          dft_weight_threshold):
    """Recalcule les poids Boltzmann DFT et filtre selon le seuil.

    Returns:
        dict avec clés: final_conformers, final_rkf_paths, dft_weights, dft_energies
    """
    # Construire le mapping rkf pour Boltzmann
    rkf_for_boltzmann = {}
    for j, vi in enumerate(valid_conf_indices):
        conf_num_in_boltzmann = vi + 1
        if conf_num_in_boltzmann in rkf_paths:
            rkf_for_boltzmann[j + 1] = rkf_paths[conf_num_in_boltzmann]

    dft_weights, dft_energies = compute_dft_boltzmann_weights(
        rkf_for_boltzmann, temperature=TEMPERATURE
    )

    if not dft_weights:
        return dict(
            final_conformers=valid_conformers,
            final_rkf_paths=rkf_for_boltzmann,
            dft_weights=dft_weights,
            dft_energies=dft_energies,
        )

    # Filtrer selon le seuil DFT
    n_before = len(valid_conformers)
    kept_conformers = []
    kept_rkf = {}
    kept_weights = {}
    kept_energies = {}

    new_idx = 1
    for old_idx in sorted(dft_weights.keys()):
        if dft_weights[old_idx] >= dft_weight_threshold:
            kept_conformers.append(valid_conformers[old_idx - 1])
            if old_idx in rkf_for_boltzmann:
                kept_rkf[new_idx] = rkf_for_boltzmann[old_idx]
            kept_weights[new_idx] = dft_weights[old_idx]
            kept_energies[new_idx] = dft_energies[old_idx]
            new_idx += 1

    print(f"\n  Filtrage Boltzmann DFT (w >= {dft_weight_threshold}): "
          f"{len(kept_conformers)}/{n_before} conformères retenus")

    return dict(
        final_conformers=kept_conformers,
        final_rkf_paths=kept_rkf,
        dft_weights=kept_weights,
        dft_energies=kept_energies,
    )


def _export_results(args, results_dir, final_conformers,
                    final_rkf_paths, dft_weights, dft_energies,
                    n_after_xtb, n_after_boltzmann,
                    cluster_continuum=False,
                    n_augmented=0, n_selected_phase2=0):
    """Exporte les résultats finaux (commun aux deux pipelines)."""
    os.makedirs(results_dir, exist_ok=True)

    for j, conf in enumerate(final_conformers):
        dst_xyz = os.path.join(results_dir, f"conf_{j+1}.xyz")
        conf.write(dst_xyz)

    # Créer le fichier conformers_summary.txt
    summary_file = os.path.join(results_dir, "conformers_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"{'='*80}\n")
        mode_label = "CLUSTER-CONTINUUM" if cluster_continuum else "STANDARD"
        f.write(f"RÉSUMÉ DE GÉNÉRATION DE CONFORMÈRES ({mode_label})\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Molécule:          {args.name}\n")
        f.write(f"SMILES:            {args.smiles}\n")
        f.write(f"Fonctionnelle:     {args.functional}\n")
        f.write(f"Base:              {args.basis}\n")
        f.write(f"Solvant:           {args.solvent}\n")
        if cluster_continuum:
            f.write(f"Cluster-continuum: Oui\n")
        f.write(f"Température:       {TEMPERATURE} K\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"STATISTIQUES\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Conformères après GFN1-xTB:        {n_after_xtb}\n")
        f.write(f"Après Boltzmann xTB (w>={args.boltzmann_min_weight}): {n_after_boltzmann}\n")
        if cluster_continuum:
            f.write(f"Systèmes augmentés:                {n_augmented}\n")
            f.write(f"Après Boltzmann Phase 2:           {n_selected_phase2}\n")
        f.write(f"Après Boltzmann DFT (w>={args.boltzmann_min_weight_dft}): {len(final_conformers)}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"CONFORMÈRES FINAUX\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'#':>4}  {'ΔE [kJ/mol]':>12}  {'Poids Boltzmann':>16}\n")
        f.write(f"{'-'*40}\n")

        for j in range(1, len(final_conformers) + 1):
            dE = dft_energies.get(j, 0.0)
            w = dft_weights.get(j, 0.0)
            f.write(f"{j:>4}  {dE:>12.3f}  {w:>16.6f}\n")

        f.write(f"\nFichiers XYZ exportés dans: {results_dir}/\n")
        f.write(f"\nNote: Seuls les conformères sans fréquences imaginaires sont inclus.\n")
        f.write(f"Note: Poids de Boltzmann calculés sur énergies de Gibbs DFT ({args.functional}/{args.basis}).\n")

    print_success(f"Résumé exporté dans: {summary_file}")

    # Créer neutral_rkf_paths.json (pont vers le pipeline redox)
    rkf_json_file = os.path.join(results_dir, "neutral_rkf_paths.json")
    with open(rkf_json_file, 'w') as f:
        json.dump(final_rkf_paths, f, indent=2)

    print_success(f"Chemins RKF neutres exportés dans: {rkf_json_file}")

    # Récapitulatif final
    print("\n" + "=" * 80)
    print("RÉCAPITULATIF FINAL")
    print("=" * 80)
    print(f"  Conformères après GFN1-xTB: {n_after_xtb}")
    print(f"  Après Boltzmann xTB (w>={args.boltzmann_min_weight}): {n_after_boltzmann}")
    if cluster_continuum:
        print(f"  Systèmes augmentés:      {n_augmented}")
        print(f"  Après Boltzmann Phase 2: {n_selected_phase2}")
    print(f"  Après Boltzmann DFT (w>={args.boltzmann_min_weight_dft}): {len(final_conformers)}")
    print(f"  Exportés dans results/:  {len(final_conformers)}")
    print(f"\nRésultats disponibles dans: {results_dir}/")
    print(f"Pont vers redox: {rkf_json_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
