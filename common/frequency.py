"""
Module de vérification de fréquences vibrationnelles.

Ce module fournit des fonctions pour vérifier la présence de fréquences
imaginaires dans les résultats de calculs DFT, permettant d'identifier
si une géométrie correspond à un vrai minimum énergétique.
"""

import os
import numpy as np
from pathlib import Path
from scm.plams import Settings, AMSJob, KFReader

from .functional import configure_functional


def extract_frequencies_from_rkf(rkf_path, section="Vibrations", key="Frequencies[cm-1]"):
    """
    Extrait les fréquences vibrationnelles depuis un fichier RKF.

    Args:
        rkf_path (str or Path): Chemin vers le fichier adf.rkf
        section (str): Section du fichier RKF (défaut: "Vibrations")
        key (str): Clé des fréquences (défaut: "Frequencies[cm-1]")

    Returns:
        np.ndarray: Tableau des fréquences en cm⁻¹, ou tableau vide si échec

    Examples:
        >>> freqs = extract_frequencies_from_rkf("path/to/adf.rkf")
        >>> print(f"Nombre de modes: {len(freqs)}")
        >>> neg_freqs = [f for f in freqs if f < 0]
        >>> print(f"Fréquences imaginaires: {neg_freqs}")
    """
    try:
        kf = KFReader(str(rkf_path))
        freqs = np.array(kf.read(section, key), dtype=float)
        return freqs
    except Exception as e:
        print(f"  ⚠ Erreur lors de l'extraction des fréquences depuis {rkf_path}: {e}")
        return np.array([])


def has_imaginary_frequencies(frequencies, threshold=0.0):
    """
    Vérifie si des fréquences imaginaires sont présentes.

    Pour garantir un vrai minimum énergétique, TOUTES les fréquences doivent
    être positives. Par défaut, aucune fréquence négative n'est tolérée.

    Args:
        frequencies (array-like): Tableau des fréquences en cm⁻¹
        threshold (float): Seuil en cm⁻¹ (défaut: 0.0)
                          Fréquences < threshold sont considérées imaginaires
                          Défaut 0.0 = aucune fréquence négative tolérée

    Returns:
        tuple: (has_imaginary, imaginary_freqs)
               - has_imaginary (bool): True si fréquences imaginaires présentes
               - imaginary_freqs (list): Liste des fréquences < threshold

    Examples:
        >>> freqs = np.array([45.2, -5.1, 123.4, 456.7])
        >>> has_imag, imag_list = has_imaginary_frequencies(freqs)
        >>> print(f"Fréquences imaginaires: {has_imag}")  # True
        >>> print(f"Valeurs: {imag_list}")  # [-5.1]
    """
    frequencies = np.array(frequencies, dtype=float)
    imaginary = frequencies[frequencies < threshold]
    return len(imaginary) > 0, imaginary.tolist()


def is_transition_state(frequencies, threshold=0.0):
    """
    Détecte si une géométrie correspond à un état de transition.

    Un état de transition (point selle de premier ordre) est caractérisé par
    EXACTEMENT UNE fréquence imaginaire. Les minima ont 0 fréquences imaginaires,
    les points selles d'ordre supérieur ont 2 ou plus.

    Args:
        frequencies (array-like): Tableau des fréquences en cm⁻¹
        threshold (float): Seuil en cm⁻¹ (défaut: 0.0)
                          Fréquences < threshold sont considérées imaginaires

    Returns:
        bool: True si géométrie = état de transition (exactement 1 fréquence imaginaire)

    Examples:
        >>> freqs_min = np.array([45.2, 23.1, 123.4, 456.7])
        >>> is_transition_state(freqs_min)  # False (0 imaginaires = minimum)

        >>> freqs_ts = np.array([-123.4, 45.2, 123.4, 456.7])
        >>> is_transition_state(freqs_ts)  # True (1 imaginaire = TS)

        >>> freqs_saddle = np.array([-50.1, -23.4, 123.4, 456.7])
        >>> is_transition_state(freqs_saddle)  # False (2 imaginaires = point selle ordre 2)
    """
    from common.constants import FREQUENCY_IMAGINARY_THRESHOLD

    if threshold is None:
        threshold = FREQUENCY_IMAGINARY_THRESHOLD

    frequencies = np.array(frequencies, dtype=float)
    imaginary_count = np.sum(frequencies < threshold)

    return imaginary_count == 1


def check_frequencies(job_result, threshold=0.0, verbose=True):
    """
    Vérifie les fréquences d'un résultat de calcul PLAMS.

    Cette fonction est la fonction principale pour vérifier si une géométrie
    optimisée correspond à un vrai minimum énergétique.

    Args:
        job_result: Résultat d'un AMSJob avec calcul de fréquences
        threshold (float): Seuil pour fréquences imaginaires (défaut: 0.0 cm⁻¹)
                          Par défaut, aucune fréquence négative n'est tolérée
        verbose (bool): Afficher les messages (défaut: True)

    Returns:
        tuple: (is_minimum, frequencies, imaginary_freqs)
               - is_minimum (bool): True si minimum (pas de fréquences imaginaires)
               - frequencies (np.ndarray): Toutes les fréquences
               - imaginary_freqs (list): Fréquences imaginaires détectées

    Examples:
        >>> job = AMSJob(...)
        >>> result = job.run()
        >>> is_min, freqs, imag = check_frequencies(result)
        >>> if is_min:
        ...     print("Géométrie est un minimum !")
        ... else:
        ...     print(f"Attention: {len(imag)} fréquence(s) imaginaire(s)")
    """
    # Essayer d'extraire les fréquences depuis le RKF
    try:
        rkf_path = job_result.rkfpath(file='adf')
        frequencies = extract_frequencies_from_rkf(rkf_path)

        if len(frequencies) == 0:
            if verbose:
                print(f"  ⚠ Impossible d'extraire les fréquences")
            return False, np.array([]), []

        # Vérifier les fréquences imaginaires
        has_imag, imag_list = has_imaginary_frequencies(frequencies, threshold)

        if verbose:
            if has_imag:
                print(f"  ✗ {len(imag_list)} fréquence(s) imaginaire(s) détectée(s):")
                for freq in imag_list:
                    print(f"      {freq:.2f} cm⁻¹")
            else:
                print(f"  ✓ Aucune fréquence imaginaire (minimum énergétique)")

        return not has_imag, frequencies, imag_list

    except Exception as e:
        if verbose:
            print(f"  ✗ Erreur lors de la vérification des fréquences: {e}")
        return False, np.array([]), []


def perturb_geometry_along_mode(mol, mode_displacements, scale_factor=0.1):
    """
    Perturbe une géométrie le long d'un mode normal de vibration.

    Args:
        mol: Objet Molecule PLAMS
        mode_displacements (array): Vecteur des déplacements [x1,y1,z1,x2,y2,z2,...]
        scale_factor (float): Facteur d'échelle de la perturbation en Å (défaut: 0.1)

    Returns:
        Molecule: Nouvelle molécule avec géométrie perturbée
    """
    from scm.plams import Molecule

    perturbed_mol = mol.copy()
    n_atoms = len(perturbed_mol.atoms)

    if len(mode_displacements) != 3 * n_atoms:
        print(f"  ⚠ Erreur: le vecteur de déplacement a {len(mode_displacements)} composantes "
              f"mais la molécule a {n_atoms} atomes (attendu: {3*n_atoms})")
        return perturbed_mol

    # Appliquer la perturbation à chaque atome
    for i, atom in enumerate(perturbed_mol.atoms):
        atom.x += scale_factor * mode_displacements[3*i]
        atom.y += scale_factor * mode_displacements[3*i + 1]
        atom.z += scale_factor * mode_displacements[3*i + 2]

    return perturbed_mol


def extract_imaginary_mode_displacements(rkf_path):
    """
    Extrait le vecteur de déplacements du mode imaginaire le plus négatif.

    Args:
        rkf_path (str or Path): Chemin vers adf.rkf

    Returns:
        tuple: (mode_number, frequency, displacements) ou (None, None, None) si aucun
    """
    try:
        kf = KFReader(str(rkf_path))
        freqs = np.array(kf.read("Vibrations", "Frequencies[cm-1]"), dtype=float)

        # Trouver le mode le plus négatif
        negative_indices = np.where(freqs < 0)[0]
        if len(negative_indices) == 0:
            return None, None, None

        # Mode le plus négatif
        worst_idx = negative_indices[np.argmin(freqs[negative_indices])]
        mode_num = worst_idx + 1  # Fortran indexing
        frequency = freqs[worst_idx]

        # Lire le vecteur de déplacements
        try:
            displacements = np.array(
                kf.read("Vibrations", f"NoWeightNormalMode({mode_num})"),
                dtype=float
            )
            return mode_num, frequency, displacements
        except KeyError:
            print(f"  ⚠ Impossible de lire le vecteur du mode {mode_num}")
            return mode_num, frequency, None

    except Exception as e:
        print(f"  ⚠ Erreur lors de l'extraction du mode imaginaire: {e}")
        return None, None, None


def _apply_solvation(settings, solvent, charge=0):
    """
    Configure la solvatation COSMO sur un objet Settings.

    Args:
        settings: Objet Settings PLAMS
        solvent (str or None): Nom du solvant. None = pas de solvant.
        charge (int): Charge moléculaire. Détermine la méthode Charged
                      (CONJ pour neutre, CONJ CORR pour ions).
    """
    if not solvent:
        return

    settings.input.adf.Solvation.Surf = "Delley"
    settings.input.adf.Solvation.Solv = f"name={solvent}"
    if charge != 0:
        settings.input.adf.Solvation.Charged = "method=CONJ CORR"
    else:
        settings.input.adf.Solvation.Charged = "method=CONJ"
    settings.input.adf.Solvation['C-Mat'] = "POT"
    settings.input.adf.Solvation.SCF = "VAR ALL"


def run_frequency_calculation(
    mol,
    functional,
    basis,
    job_name,
    charge=0,
    spin_polarization=0,
    quality="Good",
    optimize=True,
    allow_transition_state=False,
    solvent=None
):
    """
    Lance un calcul de fréquences avec optimisation de géométrie.

    Cette fonction effectue une optimisation de géométrie DFT suivie d'un
    calcul de modes normaux pour déterminer si la structure est un minimum.

    Args:
        mol: Objet Molecule PLAMS
        functional (str): Fonctionnelle DFT (ex: "PBE0", "BLYP")
        basis (str): Base atomique (ex: "DZP", "TZP")
        job_name (str): Nom du job PLAMS
        charge (int): Charge moléculaire (défaut: 0)
        spin_polarization (int): Polarisation de spin (défaut: 0)
        quality (str): Qualité numérique (défaut: "Good")
        optimize (bool): Optimiser avant calcul de fréquences (défaut: True)
        allow_transition_state (bool): Retourner le job même si c'est un état de transition (défaut: False)
        solvent (str or None): Solvant COSMO (ex: "Acetonitrile"). None = pas de solvant.

    Returns:
        AMSJob: Job exécuté avec résultats, ou None si échec critique
    """
    # Créer les settings
    s = Settings()

    # Task: GeometryOptimization avec NormalModes
    if optimize:
        s.input.ams.Task = "GeometryOptimization"
    else:
        s.input.ams.Task = "SinglePoint"

    # IMPORTANT: Activer le calcul des modes normaux
    s.input.ams.Properties.NormalModes = "Yes"

    # System settings
    if charge != 0:
        s.input.ams.System.Charge = charge

    # ADF settings
    s.input.adf.Basis.Type = basis
    s.input.adf.Basis.Core = "None"
    s.input.adf.NumericalQuality = quality
    s.input.adf.Relativity.Level = "None"

    # Paramètres pour les systèmes à couche ouverte
    if spin_polarization != 0:
        s.input.adf.SpinPolarization = spin_polarization
        s.input.adf.Unrestricted = "Yes"

    # Fonctionnelle DFT
    configure_functional(s, functional)

    # Solvant COSMO
    _apply_solvation(s, solvent, charge=charge)

    # Créer et lancer le job
    try:
        job = AMSJob(molecule=mol, settings=s, name=job_name)
        result = job.run()

        # Vérifier si c'est un échec à cause d'un état de transition
        if not result.ok():
            # Lire le fichier de log pour vérifier si c'est un état de transition
            log_file = Path(result.job.path) / "ams.log"
            is_transition_state = False

            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read().lower()
                    if "transition state" in log_content:
                        is_transition_state = True

            if is_transition_state and allow_transition_state:
                # C'est un état de transition mais on veut quand même retourner le job
                print(f"  ⚠ Job {job_name} a convergé vers un état de transition")
                return job
            elif is_transition_state:
                print(f"  ⚠ Job {job_name} a convergé vers un état de transition (sera corrigé)")
                return job  # Retourner quand même pour permettre la correction
            else:
                # Vrai échec (crash)
                print(f"  ✗ Job {job_name} a échoué (crash)")
                return None

        return job

    except Exception as e:
        print(f"  ✗ Erreur lors du lancement du job {job_name}: {e}")
        return None


def _get_scf_bond_energy(job):
    """Extract SCF Bond Energy (Hartree) from a PLAMS job's adf.rkf."""
    try:
        rkf_path = job.results.rkfpath(file='adf')
        kf = KFReader(rkf_path)
        return kf.read('Energy', 'SCF Bond Energy')
    except Exception:
        return None


def check_and_fix_frequencies(
    job,
    functional,
    basis,
    job_base_name,
    charge=0,
    spin_polarization=0,
    quality="Good",
    max_attempts=3,
    perturbation_scale=0.2,
    threshold=0.0,
    solvent=None
):
    """
    Vérifie les fréquences et tente de corriger si des fréquences imaginaires sont trouvées.

    Stratégie de correction :
    1. Détecte le mode imaginaire le plus négatif
    2. Perturbe la géométrie le long de ce mode (+ et -)
    3. Ré-optimise avec calcul de fréquences
    4. Répète jusqu'à élimination ou max_attempts

    Args:
        job: Job PLAMS initial (peut avoir des fréquences imaginaires)
        functional (str): Fonctionnelle DFT
        basis (str): Base atomique
        job_base_name (str): Nom de base pour les jobs de correction
        charge (int): Charge moléculaire
        spin_polarization (int): Polarisation de spin
        quality (str): Qualité numérique
        max_attempts (int): Nombre maximum de tentatives (défaut: 3)
        perturbation_scale (float): Échelle de perturbation de base en Å (défaut: 0.2)
        threshold (float): Seuil pour fréquences imaginaires (défaut: 0.0)
        solvent (str or None): Solvant COSMO (ex: "Acetonitrile"). None = pas de solvant.

    Returns:
        tuple: (best_job, success)
               - best_job: Meilleur job obtenu
               - success: True si toutes les fréquences imaginaires ont été éliminées
    """
    # Vérifier les fréquences initiales
    is_minimum, frequencies, imag_freqs = check_frequencies(
        job.results, threshold=threshold, verbose=False
    )

    if is_minimum:
        print(f"  ✓ Aucune fréquence imaginaire")
        return job, True

    print(f"  ⚠ {len(imag_freqs)} fréquence(s) imaginaire(s) détectée(s)")
    for freq in imag_freqs:
        print(f"      {freq:.2f} cm⁻¹")

    # Énergie de référence du job original (pour éviter de converger vers un
    # minimum local plus haut en énergie, cf. bug I-HMC conf_1 DEA)
    original_energy = _get_scf_bond_energy(job)
    if original_energy is not None:
        print(f"  Énergie de référence (SCF Bond Energy) : {original_energy:.6f} Eh")

    # Tentatives de correction
    current_job = job
    best_job = job
    best_imag_count = len(imag_freqs)

    for attempt in range(1, max_attempts + 1):
        print(f"\n  → Tentative de correction {attempt}/{max_attempts}...")

        # Calcul du scale factor progressif : 0.2, 0.4, 0.6, ...
        current_scale = perturbation_scale * attempt

        # Extraire le mode imaginaire le plus négatif
        rkf_path = current_job.results.rkfpath(file='adf')
        mode_num, frequency, displacements = extract_imaginary_mode_displacements(rkf_path)

        if displacements is None:
            print(f"    ✗ Impossible d'extraire le mode imaginaire")
            break

        print(f"    Perturbation le long du mode {mode_num} ({frequency:.2f} cm⁻¹)")
        print(f"    Scale factor: ±{current_scale:.2f} Å")

        # Obtenir la molécule optimisée actuelle
        current_mol = current_job.results.get_main_molecule()

        # Tester perturbation POSITIVE
        print(f"    → Perturbation positive (+{current_scale:.2f} Å)...")
        pos_mol = perturb_geometry_along_mode(current_mol, displacements, +current_scale)

        pos_job = run_frequency_calculation(
            pos_mol, functional, basis,
            f"{job_base_name}_corr{attempt}_pos",
            charge=charge,
            spin_polarization=spin_polarization,
            quality=quality,
            optimize=True,
            allow_transition_state=True,
            solvent=solvent
        )

        pos_imag_count = best_imag_count
        pos_accepted = False
        if pos_job is not None:
            is_min, _, imag = check_frequencies(pos_job.results, threshold=threshold, verbose=False)
            pos_imag_count = len(imag)

            if is_min:
                # Vérifier que l'énergie est inférieure ou égale à l'original
                pos_energy = _get_scf_bond_energy(pos_job)
                if original_energy is not None and pos_energy is not None and pos_energy > original_energy + 1e-6:
                    print(f"    ⚠ Correction positive : minimum trouvé mais énergie PLUS HAUTE "
                          f"({pos_energy:.6f} vs {original_energy:.6f} Eh, "
                          f"Δ = {(pos_energy - original_energy) * 2625.5:+.1f} kJ/mol)")
                    print(f"    → Correction rejetée (minimum local parasite)")
                else:
                    print(f"    ✓ Correction réussie (perturbation positive)")
                    pos_accepted = True

        # Tester perturbation NÉGATIVE
        print(f"    → Perturbation négative (-{current_scale:.2f} Å)...")
        neg_mol = perturb_geometry_along_mode(current_mol, displacements, -current_scale)

        neg_job = run_frequency_calculation(
            neg_mol, functional, basis,
            f"{job_base_name}_corr{attempt}_neg",
            charge=charge,
            spin_polarization=spin_polarization,
            quality=quality,
            optimize=True,
            allow_transition_state=True,
            solvent=solvent
        )

        neg_imag_count = best_imag_count
        neg_accepted = False
        if neg_job is not None:
            is_min, _, imag = check_frequencies(neg_job.results, threshold=threshold, verbose=False)
            neg_imag_count = len(imag)

            if is_min:
                neg_energy = _get_scf_bond_energy(neg_job)
                if original_energy is not None and neg_energy is not None and neg_energy > original_energy + 1e-6:
                    print(f"    ⚠ Correction négative : minimum trouvé mais énergie PLUS HAUTE "
                          f"({neg_energy:.6f} vs {original_energy:.6f} Eh, "
                          f"Δ = {(neg_energy - original_energy) * 2625.5:+.1f} kJ/mol)")
                    print(f"    → Correction rejetée (minimum local parasite)")
                else:
                    print(f"    ✓ Correction réussie (perturbation négative)")
                    neg_accepted = True

        # Si une correction valide a été acceptée, retourner la meilleure
        if pos_accepted and neg_accepted:
            pos_e = _get_scf_bond_energy(pos_job) or 0.0
            neg_e = _get_scf_bond_energy(neg_job) or 0.0
            if neg_e < pos_e:
                return neg_job, True
            return pos_job, True
        elif pos_accepted:
            return pos_job, True
        elif neg_accepted:
            return neg_job, True

        # Garder le meilleur résultat et mettre à jour current_job pour la prochaine tentative
        if pos_imag_count < best_imag_count:
            best_job = pos_job
            best_imag_count = pos_imag_count
            current_job = pos_job
            print(f"    → Meilleur résultat (pos): {best_imag_count} mode(s) imaginaire(s)")
        elif neg_imag_count < best_imag_count:
            best_job = neg_job
            best_imag_count = neg_imag_count
            current_job = neg_job
            print(f"    → Meilleur résultat (neg): {best_imag_count} mode(s) imaginaire(s)")
        else:
            # Aucune amélioration cette fois, mais continuer avec un scale factor plus grand
            # Choisir le job avec le moins de fréquences imaginaires (même si égal)
            if pos_imag_count <= neg_imag_count and pos_job is not None:
                current_job = pos_job
                print(f"    → Pas d'amélioration (tentative {attempt}/{max_attempts}), on continue avec pos")
            elif neg_job is not None:
                current_job = neg_job
                print(f"    → Pas d'amélioration (tentative {attempt}/{max_attempts}), on continue avec neg")
            else:
                # Les deux jobs ont échoué
                print(f"    ✗ Les deux perturbations ont échoué, arrêt")
                break

    # Résultat final
    if best_imag_count == 0:
        print(f"\n  ✓ Toutes les fréquences imaginaires ont été éliminées")
        return best_job, True
    else:
        print(f"\n  ⚠ {best_imag_count} fréquence(s) imaginaire(s) subsiste(nt)")
        return best_job, False


