"""
Pipeline de calculs redox V5.

Le neutre n'est plus recalculé : on réutilise le RKF de l'étape 4 des conformères.
Nouveau schéma composite TZ2P+DZP pour les énergies.

Pipeline par conformère (7 calculs) :
1. neutral_tz2p   : SP TZ2P sur géom neutre opt (Ee cohérent)
2. reduced_sp     : SP TZ2P sur géom neutre (EA vertical)
3. reduced_opt    : GeomOpt DZP + freq + solvant (COSMO CONJ CORR)
4. reduced_tz2p   : SP TZ2P sur géom red opt
5. oxidized_sp    : SP TZ2P sur géom neutre (IP vertical)
6. oxidized_opt   : GeomOpt DZP + freq + solvant (COSMO CONJ CORR)
7. oxidized_tz2p  : SP TZ2P sur géom ox opt
"""

from scm.plams import AMSJob

from .common import setup_adf_settings
from common.frequency import check_and_fix_frequencies


def _run_single_point(mol, name, charge, spin_polarization, solvent, functional, basis,
                      normal_modes=False, gradients=False, extra_adf_blocks=None):
    """Run a SinglePoint calculation and return the job."""
    settings = setup_adf_settings(
        task='SinglePoint',
        charge=charge,
        spin_polarization=spin_polarization,
        solvent=solvent,
        functional=functional,
        basis=basis,
        normal_modes=normal_modes,
        gradients=gradients,
        extra_adf_blocks=extra_adf_blocks
    )

    job = AMSJob(settings=settings, name=name, molecule=mol)
    job.run()

    if not job.check():
        raise RuntimeError(f"Le calcul single-point {name} a échoué")

    print(f"  ✓ {name} terminé")
    return job


def _run_optimization(mol, name, charge, spin_polarization, solvent, functional, basis,
                      max_attempts=3):
    """Run a GeometryOptimization + freq and return the job."""
    settings = setup_adf_settings(
        task='GeometryOptimization',
        charge=charge,
        spin_polarization=spin_polarization,
        solvent=solvent,
        functional=functional,
        basis=basis,
        normal_modes=True
    )

    job = AMSJob(settings=settings, name=name, molecule=mol)
    job.run()

    if not job.check():
        from pathlib import Path
        log_file = Path(job.path) / "ams.log"
        is_ts = False
        if log_file.exists():
            with open(log_file, 'r') as f:
                if "transition state" in f.read().lower():
                    is_ts = True
        if not is_ts:
            raise RuntimeError(f"L'optimisation {name} a échoué")
        print(f"  ✓ {name} terminé (état de transition détecté)")
    else:
        print(f"  ✓ {name} terminé")

    # Vérification et correction des fréquences
    job, freq_ok = check_and_fix_frequencies(
        job=job,
        functional=functional,
        basis=basis,
        job_base_name=name,
        charge=charge,
        spin_polarization=spin_polarization,
        quality="Good",
        max_attempts=max_attempts,
        solvent=solvent
    )

    if not freq_ok:
        print(f"  ⚠ Fréquences imaginaires subsistent pour {name}")

    return job


def sp_neutral_tz2p(mol_neutral, name, solvent="Acetonitrile", functional="PBE0",
                    refinement_basis="TZ2P"):
    """
    SP TZ2P sur la géométrie neutre optimisée (pour Ee cohérent avec ions).

    Args:
        mol_neutral (Molecule): Molécule neutre optimisée
        name (str): Nom du calcul
        solvent (str): Solvant COSMO
        functional (str): Fonctionnelle DFT
        refinement_basis (str): Base de raffinement (défaut: TZ2P)

    Returns:
        AMSJob: Job SP TZ2P
    """
    print(f"\n  === SP NEUTRE TZ2P : {name} ===")
    return _run_single_point(
        mol_neutral, f"{name}_neutral_tz2p",
        charge=0, spin_polarization=0,
        solvent=solvent, functional=functional,
        basis=refinement_basis    )


def sp_reduced(mol_neutral, name, solvent="Acetonitrile", functional="PBE0",
               basis="TZ2P"):
    """
    SP TZ2P de la molécule réduite (charge -1) sur géométrie neutre (EA vertical).

    Args:
        mol_neutral (Molecule): Molécule neutre optimisée
        name (str): Nom du calcul
    """
    print(f"\n  === SP RÉDUIT : {name} ===")
    return _run_single_point(
        mol_neutral, f"{name}_reduced_sp",
        charge=-1, spin_polarization=1,
        solvent=solvent, functional=functional,
        basis=basis    )


def optimize_reduced(mol_neutral, name, solvent="Acetonitrile", functional="PBE0",
                     basis="DZP", max_attempts=3):
    """
    GeomOpt DZP + freq de la molécule réduite (charge -1).

    Args:
        mol_neutral (Molecule): Molécule neutre comme point de départ
        name (str): Nom du calcul
    """
    print(f"\n  === OPTIMISATION RÉDUITE : {name} ===")
    return _run_optimization(
        mol_neutral, f"{name}_reduced_opt",
        charge=-1, spin_polarization=1,
        solvent=solvent, functional=functional,
        basis=basis, max_attempts=max_attempts
    )


def sp_reduced_tz2p(job_reduced_opt, name, solvent="Acetonitrile", functional="PBE0",
                    refinement_basis="TZ2P"):
    """
    SP TZ2P sur la géométrie réduite optimisée DZP.

    Args:
        job_reduced_opt (AMSJob): Job de l'optimisation réduite DZP
        name (str): Nom du calcul
    """
    print(f"\n  === SP RÉDUIT TZ2P : {name} ===")
    mol_reduced = job_reduced_opt.results.get_main_molecule()
    return _run_single_point(
        mol_reduced, f"{name}_reduced_tz2p",
        charge=-1, spin_polarization=1,
        solvent=solvent, functional=functional,
        basis=refinement_basis    )


def sp_oxidized(mol_neutral, name, solvent="Acetonitrile", functional="PBE0",
                basis="TZ2P"):
    """
    SP TZ2P de la molécule oxydée (charge +1) sur géométrie neutre (IP vertical).

    Args:
        mol_neutral (Molecule): Molécule neutre optimisée
        name (str): Nom du calcul
    """
    print(f"\n  === SP OXYDÉ : {name} ===")
    return _run_single_point(
        mol_neutral, f"{name}_oxidized_sp",
        charge=1, spin_polarization=1,
        solvent=solvent, functional=functional,
        basis=basis    )


def optimize_oxidized(mol_neutral, name, solvent="Acetonitrile", functional="PBE0",
                      basis="DZP", max_attempts=3):
    """
    GeomOpt DZP + freq de la molécule oxydée (charge +1).

    Args:
        mol_neutral (Molecule): Molécule neutre comme point de départ
        name (str): Nom du calcul
    """
    print(f"\n  === OPTIMISATION OXYDÉE : {name} ===")
    return _run_optimization(
        mol_neutral, f"{name}_oxidized_opt",
        charge=1, spin_polarization=1,
        solvent=solvent, functional=functional,
        basis=basis, max_attempts=max_attempts
    )


def sp_oxidized_tz2p(job_oxidized_opt, name, solvent="Acetonitrile", functional="PBE0",
                     refinement_basis="TZ2P"):
    """
    SP TZ2P sur la géométrie oxydée optimisée DZP.

    Args:
        job_oxidized_opt (AMSJob): Job de l'optimisation oxydée DZP
        name (str): Nom du calcul
    """
    print(f"\n  === SP OXYDÉ TZ2P : {name} ===")
    mol_oxidized = job_oxidized_opt.results.get_main_molecule()
    return _run_single_point(
        mol_oxidized, f"{name}_oxidized_tz2p",
        charge=1, spin_polarization=1,
        solvent=solvent, functional=functional,
        basis=refinement_basis    )


def run_redox_pipeline(mol_neutral, neutral_rkf_path, name,
                       solvent="Acetonitrile", functional="PBE0",
                       basis="DZP", refinement_basis="TZ2P",
                       max_attempts=3):
    """
    Exécute le pipeline complet de calculs redox pour un conformère.

    Le neutre n'est PAS ré-optimisé : on réutilise le RKF de l'étape 4 des conformères.

    Args:
        mol_neutral (Molecule): Molécule neutre optimisée (depuis conformères)
        neutral_rkf_path (str): Chemin vers le adf.rkf du neutre (conformères)
        name (str): Nom de base pour les calculs
        solvent (str): Solvant pour COSMO
        functional (str): Fonctionnelle DFT
        basis (str): Base atomique pour les calculs DZP
        refinement_basis (str): Base de raffinement pour SP (défaut: TZ2P)
        max_attempts (int): Tentatives de correction de fréquences

    Returns:
        dict: Dictionnaire avec les jobs/paths de chaque étape
              {'neutral_rkf': str, 'neutral_tz2p': job,
               'reduced_sp': job, 'reduced_opt': job, 'reduced_tz2p': job,
               'oxidized_sp': job, 'oxidized_opt': job, 'oxidized_tz2p': job}
    """
    results = {'neutral_rkf': neutral_rkf_path}

    try:
        print(f"\n{'='*60}")
        print(f"PIPELINE REDOX : {name}")
        print(f"{'='*60}")

        # 1. SP TZ2P neutre
        results['neutral_tz2p'] = sp_neutral_tz2p(
            mol_neutral, name, solvent, functional, refinement_basis
        )

        # 2. SP réduit TZ2P (sur géom neutre, pour EA vertical)
        results['reduced_sp'] = sp_reduced(
            mol_neutral, name, solvent, functional, refinement_basis
        )

        # 3. Optimisation réduite DZP
        results['reduced_opt'] = optimize_reduced(
            mol_neutral, name, solvent, functional, basis, max_attempts
        )

        # 4. SP réduit TZ2P
        results['reduced_tz2p'] = sp_reduced_tz2p(
            results['reduced_opt'], name, solvent, functional, refinement_basis
        )

        # 5. SP oxydé TZ2P (sur géom neutre, pour IP vertical)
        results['oxidized_sp'] = sp_oxidized(
            mol_neutral, name, solvent, functional, refinement_basis
        )

        # 6. Optimisation oxydée DZP
        results['oxidized_opt'] = optimize_oxidized(
            mol_neutral, name, solvent, functional, basis, max_attempts
        )

        # 7. SP oxydé TZ2P
        results['oxidized_tz2p'] = sp_oxidized_tz2p(
            results['oxidized_opt'], name, solvent, functional, refinement_basis
        )

        print(f"\n{'='*60}")
        print(f"✓ Pipeline complet terminé pour {name}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERREUR dans le pipeline pour {name}")
        print(f"{'='*60}")
        print(f"Message d'erreur : {str(e)}")
        import traceback
        traceback.print_exc()

    return results


def run_predict_dg_pipeline(mol_neutral, neutral_rkf_path, name,
                            solvent="Acetonitrile", functional="PBE0",
                            basis="DZP"):
    """
    Pipeline predict-dg : 3 single points sur géométrie neutre avec ConceptualDFT et QTAIM.

    Génère les données nécessaires à la prédiction de DeltaG :
    - EA/IP verticaux et Koopmans (HOMO/LUMO)
    - Hessien NEUTRE (NormalModes sur le seul SP neutre) pour le descripteur
      vibronique LVC : E_relax = -1/2 g_ion^T H_neutre^{-1} g_ion
    - Gradients ioniques (analytiques) pour LVC, sans Hessien ionique
    - Descripteurs ConceptualDFT et QTAIM (sur les 3 états de charge)

    Note : seul le SP neutre porte NormalModes (Hessien neutre, réutilisé aussi
    par la thermochimie). Les SP ioniques ne demandent que le gradient analytique
    (quasi gratuit) : LVC n'a pas besoin du Hessien ionique, ce qui évite ~6N
    évaluations de gradient par état ionique (Hessien numérique PBE0+COSMO).

    Args:
        mol_neutral (Molecule): Molécule neutre optimisée (depuis conformères)
        neutral_rkf_path (str): Chemin vers le adf.rkf du neutre (conformères)
        name (str): Nom de base pour les calculs
        solvent (str): Solvant pour COSMO
        functional (str): Fonctionnelle DFT
        basis (str): Base atomique

    Returns:
        dict: {'neutral_rkf': str, 'neutral_sp': job, 'reduced_sp': job,
               'oxidized_sp': job, 'mode': 'predict_dg'}
    """
    extra_adf_blocks = {
        'ConceptualDFT': {'Enabled': 'Yes', 'AnalysisLevel': 'Full'},
        'QTAIM': {'Enabled': 'Yes', 'AnalysisLevel': 'Full'}
    }

    results = {'neutral_rkf': neutral_rkf_path, 'mode': 'predict_dg'}

    try:
        print(f"\n{'='*60}")
        print(f"PIPELINE PREDICT-DG : {name}")
        print(f"{'='*60}")

        # 1. SP + NormalModes neutre (Hessien neutre pour LVC + ConceptualDFT + QTAIM)
        print(f"\n  === SP NEUTRE + NormalModes : {name} ===")
        results['neutral_sp'] = _run_single_point(
            mol_neutral, f"{name}_neutral_sp",
            charge=0, spin_polarization=0,
            solvent=solvent, functional=functional, basis=basis,
            normal_modes=True, extra_adf_blocks=extra_adf_blocks
        )

        # 2. SP réduit : gradient seul (pour LVC) + ConceptualDFT + QTAIM, sans Hessien
        print(f"\n  === SP RÉDUIT (gradient) : {name} ===")
        results['reduced_sp'] = _run_single_point(
            mol_neutral, f"{name}_reduced_sp",
            charge=-1, spin_polarization=1,
            solvent=solvent, functional=functional, basis=basis,
            normal_modes=False, gradients=True, extra_adf_blocks=extra_adf_blocks
        )

        # 3. SP oxydé : gradient seul (pour LVC) + ConceptualDFT + QTAIM, sans Hessien
        print(f"\n  === SP OXYDÉ (gradient) : {name} ===")
        results['oxidized_sp'] = _run_single_point(
            mol_neutral, f"{name}_oxidized_sp",
            charge=1, spin_polarization=1,
            solvent=solvent, functional=functional, basis=basis,
            normal_modes=False, gradients=True, extra_adf_blocks=extra_adf_blocks
        )

        print(f"\n{'='*60}")
        print(f"✓ Pipeline predict-dg terminé pour {name}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERREUR dans le pipeline predict-dg pour {name}")
        print(f"{'='*60}")
        print(f"Message d'erreur : {str(e)}")
        import traceback
        traceback.print_exc()

    return results
