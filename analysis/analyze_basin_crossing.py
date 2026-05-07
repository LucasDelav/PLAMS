#!/usr/bin/env amspython
"""
Analyse de changement de bassin conformationnel lors de l'optimisation ionique.

Deux approches de décomposition de E_relax :

1. Décomposition LVC (robuste, pas de matching conformère) :
   E_relax(A) = E_LVC(A) + ΔE_beyond(A)
   Où E_LVC(A) est la relaxation harmonique dans le bassin A,
   et ΔE_beyond capture anharmonicité + changement conformationnel.

2. Décomposition RMSD (quand un conformère cible est identifiable) :
   E_relax(A) = E_relax_local(B) + E_vert_shift(A→B)
   Avec validation : E_relax_local(B) ≈ E_LVC(B)

Diagnostic supplémentaire :
   - RMSD croisé entre géométries ioniques optimisées (détecte convergence
     vers le même bassin ionique)
   - Seuil de confiance sur le matching RMSD

Usage:
  amspython analyze_basin_crossing.py --batch-dir 1594174/dir_workdir \
      --output-dir thermodynamic_models
"""

import sys
import math
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmanalysis.lvc_analysis import (
    extract_coordinates_from_rkf, compute_rmsd,
    find_sp_rkf_files, find_opt_rkf_files,
    compute_lvc_edef_from_rkf, compute_lvc2_edef_from_rkf,
)
from qmanalysis.frequency_extraction import find_neutral_rkf_files

EH_TO_KJ = 2625.499638

# Seuils de confiance pour le matching RMSD
RMSD_CONFIDENT = 0.30   # Å : matching fiable
RMSD_UNCERTAIN = 0.50   # Å : matching incertain


def extract_energy_from_rkf(rkf_path):
    """Extrait l'énergie SCF Bond Energy depuis un RKF."""
    from scm.plams import KFReader
    try:
        kf = KFReader(str(rkf_path))
        return float(kf.read("Energy", "Bond Energy"))
    except Exception:
        return None


def analyze_basin_crossing_molecule(workdir, mol_name, process='reduced'):
    """
    Analyse le changement de bassin pour une molécule.

    Approche en 3 couches :
    1. Calcul LVC/LVC2 pour chaque conformère (E_relax_local harmonique)
    2. Matching RMSD avec seuils de confiance
    3. Décomposition : E_relax = E_LVC + ΔE_beyond

    Returns:
        dict avec les résultats par conformère
    """
    workdir = Path(workdir)

    # Localiser les fichiers RKF
    neutral_rkfs = find_neutral_rkf_files(workdir, mol_name)
    sp_rkfs = find_sp_rkf_files(workdir, mol_name, process)
    opt_rkfs = find_opt_rkf_files(workdir, mol_name, process)

    if not neutral_rkfs or not sp_rkfs or not opt_rkfs:
        return None

    # Extraire toutes les géométries neutres
    neutral_coords = {}
    for conf_num, rkf in sorted(neutral_rkfs.items()):
        coords = extract_coordinates_from_rkf(rkf)
        if coords is not None:
            neutral_coords[conf_num] = coords

    # Extraire les énergies SP ioniques (à géométrie neutre)
    sp_energies = {}
    for conf_num, rkf in sorted(sp_rkfs.items()):
        e = extract_energy_from_rkf(rkf)
        if e is not None:
            sp_energies[conf_num] = e

    # Calculer E_LVC et E_LVC2 pour chaque conformère
    lvc_results = {}
    lvc2_results = {}
    for conf_num in sorted(neutral_rkfs.keys()):
        neutral_rkf = neutral_rkfs.get(conf_num)
        sp_rkf = sp_rkfs.get(conf_num)
        if neutral_rkf and sp_rkf:
            lvc_res = compute_lvc_edef_from_rkf(neutral_rkf, sp_rkf)
            if lvc_res['success']:
                lvc_results[conf_num] = lvc_res['E_def_lvc_kj']
            lvc2_res = compute_lvc2_edef_from_rkf(sp_rkf)
            if lvc2_res['success']:
                lvc2_results[conf_num] = lvc2_res.get('E_def_lvc2_kj') or lvc2_res.get('E_def_lvc_kj')

    # Extraire les géométries ioniques optimisées et calculer le RMSD croisé
    ion_opt_coords = {}
    ion_opt_energies = {}
    for conf_num, rkf in sorted(opt_rkfs.items()):
        coords = extract_coordinates_from_rkf(rkf)
        e = extract_energy_from_rkf(rkf)
        if coords is not None:
            ion_opt_coords[conf_num] = coords
        if e is not None:
            ion_opt_energies[conf_num] = e

    # RMSD croisé entre géométries ioniques optimisées
    ion_cross_rmsd = {}
    ion_confs = sorted(ion_opt_coords.keys())
    for i, c_i in enumerate(ion_confs):
        for c_j in ion_confs[i+1:]:
            if ion_opt_coords[c_i].shape == ion_opt_coords[c_j].shape:
                try:
                    r = compute_rmsd(ion_opt_coords[c_i], ion_opt_coords[c_j])
                    ion_cross_rmsd[(c_i, c_j)] = r
                except Exception:
                    pass

    # Analyser chaque conformère
    results = {}

    for conf_A in sorted(opt_rkfs.keys()):
        if conf_A not in neutral_coords or conf_A not in sp_energies:
            continue
        if conf_A not in ion_opt_coords or conf_A not in ion_opt_energies:
            continue

        coords_ion = ion_opt_coords[conf_A]
        E_opt = ion_opt_energies[conf_A]
        E_sp_A = sp_energies[conf_A]
        E_relax_total = (E_opt - E_sp_A) * EH_TO_KJ  # kJ/mol

        # === Couche 1 : Décomposition LVC ===
        E_lvc_A = lvc_results.get(conf_A)
        E_lvc2_A = lvc2_results.get(conf_A)

        # ΔE_beyond = tout ce qui n'est pas capturé par l'harmonique
        dE_beyond_lvc = (E_relax_total - E_lvc_A) if E_lvc_A is not None else None
        dE_beyond_lvc2 = (E_relax_total - E_lvc2_A) if E_lvc2_A is not None else None

        # === Couche 2 : Matching RMSD ===
        rmsd_to_neutrals = {}
        for conf_j, coords_j in neutral_coords.items():
            if coords_ion.shape != coords_j.shape:
                continue
            try:
                rmsd_to_neutrals[conf_j] = compute_rmsd(coords_ion, coords_j)
            except Exception:
                continue

        rmsd_self = rmsd_to_neutrals.get(conf_A, float('nan'))

        if rmsd_to_neutrals:
            conf_B = min(rmsd_to_neutrals, key=rmsd_to_neutrals.get)
            rmsd_best = rmsd_to_neutrals[conf_B]
        else:
            conf_B = conf_A
            rmsd_best = float('nan')

        # Confiance du matching
        if rmsd_best < RMSD_CONFIDENT:
            confidence = 'high'
        elif rmsd_best < RMSD_UNCERTAIN:
            confidence = 'medium'
        else:
            confidence = 'low'

        crossed = (conf_B != conf_A)

        # === Couche 3 : Décomposition RMSD (si confiance suffisante) ===
        E_vert_shift = 0.0
        E_relax_local_rmsd = E_relax_total

        if crossed and conf_B in sp_energies:
            E_sp_B = sp_energies[conf_B]
            E_vert_shift = (E_sp_B - E_sp_A) * EH_TO_KJ
            E_relax_local_rmsd = (E_opt - E_sp_B) * EH_TO_KJ

        # Validation croisée : E_relax_local(B) devrait ≈ E_LVC(B)
        E_lvc_B = lvc_results.get(conf_B)
        validation_ok = None
        if E_lvc_B is not None and confidence != 'low':
            # Si E_local ≈ E_LVC(B), le matching est cohérent
            validation_ok = abs(E_relax_local_rmsd - E_lvc_B) < max(10.0, 0.5 * abs(E_lvc_B))

        # Conformères ioniques qui convergent vers le même bassin
        same_basin_confs = []
        for (ci, cj), r in ion_cross_rmsd.items():
            if r < RMSD_CONFIDENT:
                if ci == conf_A:
                    same_basin_confs.append(cj)
                elif cj == conf_A:
                    same_basin_confs.append(ci)

        results[conf_A] = {
            # DFT direct
            'E_relax_total': E_relax_total,
            # Décomposition LVC (robuste)
            'E_lvc': E_lvc_A,
            'E_lvc2': E_lvc2_A,
            'dE_beyond_lvc': dE_beyond_lvc,
            'dE_beyond_lvc2': dE_beyond_lvc2,
            # Matching RMSD
            'rmsd_self': rmsd_self,
            'rmsd_best': rmsd_best,
            'conf_target': conf_B,
            'crossed': crossed,
            'confidence': confidence,
            # Décomposition RMSD
            'E_relax_local_rmsd': E_relax_local_rmsd,
            'E_vert_shift': E_vert_shift,
            'E_lvc_target': E_lvc_B,
            'validation_ok': validation_ok,
            # Diagnostic
            'rmsd_all': rmsd_to_neutrals,
            'same_basin_confs': same_basin_confs,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyse de changement de bassin conformationnel")
    parser.add_argument('--batch-dir', required=True, help='Répertoire batch des workdirs')
    parser.add_argument('--output-dir',
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'thermodynamic_models'),
                        help='Répertoire de sortie')
    parser.add_argument('--molecules', nargs='*', default=None, help='Molécules spécifiques')
    args = parser.parse_args()

    batch_path = Path(args.batch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Découvrir les molécules
    if args.molecules:
        mol_names = args.molecules
    else:
        mol_names = sorted([
            d.name.replace('_workdir', '')
            for d in batch_path.iterdir()
            if d.is_dir() and d.name.endswith('_workdir')
        ])

    print("=" * 100)
    print("ANALYSE DE CHANGEMENT DE BASSIN CONFORMATIONNEL (v2)")
    print("=" * 100)
    print(f"\n  Seuils RMSD : confiant < {RMSD_CONFIDENT} Å, incertain < {RMSD_UNCERTAIN} Å")

    all_results = {}

    for process, label in [('reduced', 'RÉDUCTION'), ('oxidized', 'OXYDATION')]:
        print(f"\n{'='*100}")
        print(f"  {label}")
        print(f"{'='*100}")

        for mol_name in mol_names:
            workdir = batch_path / f"{mol_name}_workdir"
            if not workdir.exists():
                continue

            results = analyze_basin_crossing_molecule(workdir, mol_name, process)
            if not results:
                continue

            all_results[(mol_name, process)] = results

            # Boltzmann weights
            from generate_thermodynamic_models import read_conformer_weights
            weights = read_conformer_weights(workdir)

            print(f"\n  --- {mol_name} ---")

            # Header compact
            print(f"  {'Conf':>4s}  {'E_relax':>8s}  {'E_LVC':>8s}  {'E_LVC2':>8s}  "
                  f"{'ΔE_byd':>8s}  {'RMSD_A':>7s}  {'RMSD_B':>7s}  "
                  f"{'Target':>6s}  {'Conf?':>5s}  {'w':>5s}")
            print(f"  {'':>4s}  {'kJ/mol':>8s}  {'kJ/mol':>8s}  {'kJ/mol':>8s}  "
                  f"{'kJ/mol':>8s}  {'Å':>7s}  {'Å':>7s}")
            print("  " + "-" * 90)

            # Boltzmann-weighted averages
            sum_w = 0.0
            avg_total = 0.0
            avg_lvc = 0.0
            avg_lvc2 = 0.0
            avg_beyond = 0.0
            n_crossed = 0
            n_problematic = 0

            for conf_num in sorted(results.keys()):
                r = results[conf_num]
                w = weights.get(conf_num, 1.0 / len(results)) if weights else 1.0 / len(results)

                # ΔE_beyond : utiliser LVC2 si disponible, sinon LVC
                dE_beyond = r['dE_beyond_lvc2'] if r['dE_beyond_lvc2'] is not None else r['dE_beyond_lvc']

                # Indicateur de confiance
                conf_marker = ''
                if r['crossed']:
                    if r['confidence'] == 'high':
                        conf_marker = f"→{r['conf_target']}"
                    elif r['confidence'] == 'medium':
                        conf_marker = f"→{r['conf_target']}?"
                    else:
                        conf_marker = "???"
                else:
                    conf_marker = "same"

                # Flag problèmes
                flags = []
                if r['E_relax_total'] > 0:
                    flags.append('E>0!')
                if r['confidence'] == 'low' and r['crossed']:
                    flags.append('low')
                if r['validation_ok'] is False:
                    flags.append('!val')
                flag_str = ','.join(flags) if flags else ''

                e_lvc_str = f"{r['E_lvc']:8.2f}" if r['E_lvc'] is not None else '    N/A '
                e_lvc2_str = f"{r['E_lvc2']:8.2f}" if r['E_lvc2'] is not None else '    N/A '
                de_str = f"{dE_beyond:8.2f}" if dE_beyond is not None else '    N/A '

                print(f"  {conf_num:4d}  {r['E_relax_total']:8.2f}  {e_lvc_str}  {e_lvc2_str}  "
                      f"{de_str}  {r['rmsd_self']:7.3f}  {r['rmsd_best']:7.3f}  "
                      f"{conf_marker:>6s}  {flag_str:>5s}  {w:5.3f}")

                sum_w += w
                avg_total += w * r['E_relax_total']
                if r['E_lvc'] is not None:
                    avg_lvc += w * r['E_lvc']
                if r['E_lvc2'] is not None:
                    avg_lvc2 += w * r['E_lvc2']
                if dE_beyond is not None:
                    avg_beyond += w * dE_beyond
                if r['crossed']:
                    n_crossed += 1
                if r['E_relax_total'] > 0 or (dE_beyond is not None and abs(dE_beyond) > 30):
                    n_problematic += 1

            if sum_w > 0:
                avg_total /= sum_w
                avg_lvc /= sum_w
                avg_lvc2 /= sum_w
                avg_beyond /= sum_w

            print("  " + "-" * 90)
            print(f"  {'<w>':>4s}  {avg_total:8.2f}  {avg_lvc:8.2f}  {avg_lvc2:8.2f}  "
                  f"{avg_beyond:8.2f}  {'':>7s}  {'':>7s}  "
                  f"{'':>6s}  {n_crossed:2d}/{len(results):2d}")

            # Détail pour les conformères problématiques
            for conf_num in sorted(results.keys()):
                r = results[conf_num]
                show = (r['crossed'] or r['rmsd_self'] > 0.15 or r['E_relax_total'] > 0)
                if not show:
                    continue

                print(f"\n    Conf {conf_num}:")

                # RMSD vers neutres
                print(f"      RMSD vers neutres:", end="")
                for j, rmsd_j in sorted(r['rmsd_all'].items()):
                    marker = " ←" if j == r['conf_target'] else ""
                    self_marker = " (self)" if j == conf_num else ""
                    print(f"  {j}={rmsd_j:.3f}{marker}{self_marker}", end="")
                print()

                # Validation de la décomposition
                if r['crossed'] and r['E_lvc_target'] is not None:
                    print(f"      Validation: E_local(RMSD)={r['E_relax_local_rmsd']:.1f}, "
                          f"E_LVC(B={r['conf_target']})={r['E_lvc_target']:.1f}  "
                          f"→ {'OK' if r['validation_ok'] else 'INCOHÉRENT'}")

                # Convergence ionique croisée
                if r['same_basin_confs']:
                    print(f"      Même bassin ionique que: conf {r['same_basin_confs']}")

                # Interprétation
                dE = r['dE_beyond_lvc2'] if r['dE_beyond_lvc2'] is not None else r['dE_beyond_lvc']
                if dE is not None:
                    if dE < -10:
                        print(f"      → Changement conformationnel (ΔE_beyond={dE:.1f} << 0)")
                    elif dE > 10:
                        print(f"      → Optimisation problématique (ΔE_beyond={dE:.1f} >> 0)")
                    else:
                        print(f"      → Relaxation essentiellement harmonique (ΔE_beyond={dE:.1f})")

    # === Export summary ===
    summary_path = output_dir / 'basin_crossing_analysis.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("ANALYSE DE CHANGEMENT DE BASSIN CONFORMATIONNEL (v2)\n")
        f.write("=" * 100 + "\n\n")

        f.write("Décomposition : E_relax = E_LVC + ΔE_beyond\n")
        f.write("  E_LVC = relaxation harmonique dans le bassin du conformère de départ\n")
        f.write("  ΔE_beyond = anharmonicité + changement conformationnel\n")
        f.write(f"  Seuils RMSD : confiant < {RMSD_CONFIDENT} Å, incertain < {RMSD_UNCERTAIN} Å\n\n")

        # Table résumé par molécule
        f.write(f"{'Molécule':12s} {'Proc':>4s} {'nConf':>5s} {'nCross':>6s} "
                f"{'<E_relax>':>10s} {'<E_LVC>':>10s} {'<E_LVC2>':>10s} {'<ΔE_byd>':>10s} "
                f"{'nProb':>5s}\n")
        f.write("-" * 80 + "\n")

        for (mol_name, process), results in sorted(all_results.items()):
            from generate_thermodynamic_models import read_conformer_weights
            workdir = batch_path / f"{mol_name}_workdir"
            weights = read_conformer_weights(workdir)

            sum_w = 0.0
            avg_total = 0.0
            avg_lvc = 0.0
            avg_lvc2 = 0.0
            avg_beyond = 0.0
            n_crossed = 0
            n_prob = 0

            for conf_num, r in results.items():
                w = weights.get(conf_num, 1.0 / len(results)) if weights else 1.0 / len(results)
                sum_w += w
                avg_total += w * r['E_relax_total']
                if r['E_lvc'] is not None:
                    avg_lvc += w * r['E_lvc']
                if r['E_lvc2'] is not None:
                    avg_lvc2 += w * r['E_lvc2']
                dE = r['dE_beyond_lvc2'] if r['dE_beyond_lvc2'] is not None else r['dE_beyond_lvc']
                if dE is not None:
                    avg_beyond += w * dE
                if r['crossed']:
                    n_crossed += 1
                if r['E_relax_total'] > 0 or (dE is not None and abs(dE) > 30):
                    n_prob += 1

            if sum_w > 0:
                avg_total /= sum_w
                avg_lvc /= sum_w
                avg_lvc2 /= sum_w
                avg_beyond /= sum_w

            proc_label = 'red' if process == 'reduced' else 'ox'
            f.write(f"{mol_name:12s} {proc_label:>4s} {len(results):5d} {n_crossed:6d} "
                    f"{avg_total:10.2f} {avg_lvc:10.2f} {avg_lvc2:10.2f} {avg_beyond:10.2f} "
                    f"{n_prob:5d}\n")

        # Détail par conformère
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("DÉTAIL PAR CONFORMÈRE\n")
        f.write("=" * 100 + "\n")

        for (mol_name, process), results in sorted(all_results.items()):
            proc_label = 'Réduction' if process == 'reduced' else 'Oxydation'
            f.write(f"\n--- {mol_name} ({proc_label}) ---\n")

            for conf_num in sorted(results.keys()):
                r = results[conf_num]
                dE = r['dE_beyond_lvc2'] if r['dE_beyond_lvc2'] is not None else r['dE_beyond_lvc']
                dE_str = f"{dE:.2f}" if dE is not None else "N/A"

                lvc_str = f"{r['E_lvc']:.2f}" if r['E_lvc'] is not None else "N/A"
                lvc2_str = f"{r['E_lvc2']:.2f}" if r['E_lvc2'] is not None else "N/A"

                f.write(f"  Conf {conf_num}: E_relax={r['E_relax_total']:.2f}, "
                        f"E_LVC={lvc_str}, "
                        f"E_LVC2={lvc2_str}, "
                        f"ΔE_beyond={dE_str}, "
                        f"RMSD_self={r['rmsd_self']:.3f}, "
                        f"target=conf_{r['conf_target']}, "
                        f"confidence={r['confidence']}")

                if r['same_basin_confs']:
                    f.write(f", same_basin={r['same_basin_confs']}")
                f.write("\n")

    print(f"\n  Saved: {summary_path}")
    print("\nTerminé.")


if __name__ == '__main__':
    main()
