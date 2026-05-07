"""
Analyse VPT2-cubique pour le calcul de l'énergie de déformation.

Ce module implémente une correction anharmonique cubique (VPT2) pour E_relax,
complémentaire à l'approche Morse-Duschinsky. Au lieu d'ajuster un potentiel
de Morse, on ajuste un potentiel cubique par mode :

    V(q) = F_j·q + ½ b_j·q² + (1/6) c_j·q³

avec trois contraintes indépendantes :
    - V'(0) = F_j : gradient de la PES ionique à la géométrie neutre (depuis le SP)
    - V'(Δq_j) = 0 : le minimum ionique est un extremum
    - V''(Δq_j) = k_j : courbure au minimum ionique (depuis le calcul de fréquences)

Note : k_j est la constante de force au minimum ionique V''(Δq), et NON le
coefficient quadratique dans l'expansion autour de q=0. Le coefficient b_j = V''(0)
est déterminé par les contraintes : b_j = k_j - c_j·Δq_j.

Résolution :
    c_j = 2(F_j + k_j·Δq_j) / Δq_j²
    b_j = k_j - c_j·Δq_j

L'énergie de relaxation par mode se simplifie :
    E_j^VPT2 = V(Δq_j) - V(0) = (1/3) F_j·Δq_j - (1/6) k_j·Δq_j²

Propriétés :
    - Pour un mode parfaitement harmonique (F_j = -k_j·Δq_j), on retrouve
      exactement le résultat Duschinsky : E_j = -½ k_j·Δq_j²
    - Le terme -(1/6) k_j·Δq² est toujours négatif (relaxation)
    - Le terme (1/3) F_j·Δq est négatif pour un gradient physique (F_j·Δq < 0)
    - Pour grands déplacements, E ≈ -(1/6) k_j·Δq² (1/3 du résultat harmonique),
      donc VPT2 est plus modéré que Duschinsky mais ne sature pas comme Morse

Convention d'unités (identique à Duschinsky/LVC) :
    - Coordonnées : Bohr
    - Modes normaux : NoWeightNormalMode(i), normalisés à 1 (sans masse)
    - Constantes de force : Hartree/Bohr²
    - Gradients : Hartree/Bohr
    - Masses atomiques : AMU
    - Résultats : Hartree, convertis en kJ/mol
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np

from .common import (
    EH_TO_KJ,
)
from .duschinsky_analysis import (
    extract_coordinates_bohr,
    extract_atomic_masses,
)
from .lvc_analysis import (
    extract_normal_modes_and_fc,
    extract_gradients,
    find_opt_rkf_files,
    find_sp_rkf_files,
)
from .frequency_extraction import find_neutral_rkf_files


def compute_vpt2_cubic_edef(
    neutral_rkf: Path,
    ion_opt_rkf: Path,
    ion_sp_rkf: Path,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Calcule E_def par la méthode VPT2-cubique (correction anharmonique polynomiale).

    Pour chaque mode j de l'ion :
    1. Calcule dQ_j par projection mass-pondérée du déplacement (identique à Duschinsky)
    2. Projette le gradient SP : F_j = g · N'_j (dot product simple, comme LVC)
    3. Détermine c_j = 2(F_j + k_j·Δq) / Δq² (contrainte V'(Δq)=0 et V''(Δq)=k_j)
    4. Calcule E_j^VPT2 = (1/3) F_j·Δq - (1/6) k_j·Δq²

    Args:
        neutral_rkf: RKF du calcul neutre optimisé
        ion_opt_rkf: RKF de l'ion optimisé (modes, FC, coordonnées)
        ion_sp_rkf: RKF du single-point ionique sur géométrie neutre (gradients)
        freq_threshold: Seuil minimal de fréquence (cm⁻¹)
        top_n_modes: Nombre de modes dominants à retourner

    Returns:
        Dict avec 'success', 'E_def_kJ', 'E_def_hartree',
        'E_def_harmonic_kJ' (pour comparaison), 'contributions',
        'n_modes_used', 'n_modes_total', 'error'
    """
    result = {
        'success': False,
        'E_def_kJ': None,
        'E_def_hartree': None,
        'E_def_harmonic_kJ': None,
        'contributions': [],
        'n_modes_used': 0,
        'n_modes_total': 0,
        'error': None,
    }

    # 1. Coordonnées en Bohr
    coords_neutral = extract_coordinates_bohr(neutral_rkf)
    coords_ion = extract_coordinates_bohr(ion_opt_rkf)
    if coords_neutral is None:
        result['error'] = f"Coordonnées non extraites depuis {neutral_rkf}"
        return result
    if coords_ion is None:
        result['error'] = f"Coordonnées non extraites depuis {ion_opt_rkf}"
        return result
    if coords_neutral.shape != coords_ion.shape:
        result['error'] = f"Dimensions incompatibles: {coords_neutral.shape} vs {coords_ion.shape}"
        return result

    # 2. Modes normaux et constantes de force de l'ION
    ion_modes, ion_fc, ion_freq = extract_normal_modes_and_fc(ion_opt_rkf)
    if ion_modes is None:
        result['error'] = f"Modes normaux non extraits depuis {ion_opt_rkf}"
        return result

    # 3. Masses atomiques de l'ion (pour projection mass-weighted)
    masses = extract_atomic_masses(ion_opt_rkf)
    if masses is None:
        result['error'] = f"Masses atomiques non extraites depuis {ion_opt_rkf}"
        return result
    masses_3N = np.repeat(masses, 3)

    # 4. Gradient SP (ion PES à géométrie neutre)
    gradients = extract_gradients(ion_sp_rkf)
    if gradients is None:
        result['error'] = f"Gradients non extraits depuis {ion_sp_rkf}"
        return result

    # 5. Déplacement neutre → ion (même convention que Duschinsky)
    delta_R = coords_neutral - coords_ion  # (3N,) en Bohr

    # 6. Projection et calcul E_def VPT2-cubique
    E_def_total_hartree = 0.0
    E_def_harmonic_hartree = 0.0
    contributions = []

    for j in range(len(ion_modes)):
        mode = ion_modes[j]
        freq = ion_freq[j]
        k_j = ion_fc[j]

        if mode is None or freq < freq_threshold or k_j <= 0:
            continue

        # Masse réduite et projection mass-weighted du déplacement
        mu_j = np.sum(masses_3N * mode ** 2)
        W_j = np.sum(masses_3N * mode * delta_R)
        dQ_j = W_j / mu_j

        # Gradient projeté sur mode ionique (dot product simple, comme LVC)
        F_j = np.dot(gradients, mode)

        # Énergie harmonique Duschinsky (pour comparaison)
        E_harm_j = -0.5 * k_j * dQ_j ** 2

        # VPT2-cubique avec k_j = V''(Δq) (courbure au minimum ionique)
        # Convention code : dQ_j = ion→neutre (delta_R = coords_neutral - coords_ion)
        # Donc Δq = neutre→ion = -dQ_j
        #
        # Formule exacte : E = (1/3) F_j·Δq - (1/6) k_j·Δq²
        # En substituant Δq = -dQ : E = -(1/3) F_j·dQ - (1/6) k_j·dQ²
        # Limite harmonique (F_j = k_j·dQ) → -(1/3)k·dQ² - (1/6)k·dQ² = -½k·dQ²  ✓
        if abs(dQ_j) > 1e-10:
            # c_j = 2(F_j + k_j·Δq) / Δq²  avec Δq = -dQ
            f3_j = 2.0 * (F_j - k_j * dQ_j) / (dQ_j ** 2)

            E_vpt2_j = -(1.0 / 3.0) * F_j * dQ_j - (1.0 / 6.0) * k_j * dQ_j ** 2

            # Harmonicité : F_j/(k_j·dQ_j) = 1 pour un mode parfaitement harmonique
            h_j = F_j / (k_j * dQ_j) if abs(k_j * dQ_j) > 1e-15 else 1.0
        else:
            f3_j = 0.0
            E_vpt2_j = E_harm_j
            h_j = 1.0

        E_def_total_hartree += E_vpt2_j
        E_def_harmonic_hartree += E_harm_j

        contributions.append({
            'mode_num': j + 1,
            'frequency': freq,
            'delta_Q': dQ_j,
            'k_j': k_j,
            'F_j': F_j,
            'f3_j': f3_j,
            'harmonicity': h_j,
            'contribution_hartree': E_vpt2_j,
            'contribution_kj': E_vpt2_j * EH_TO_KJ,
            'contribution_harmonic_kj': E_harm_j * EH_TO_KJ,
        })

    contributions.sort(key=lambda x: abs(x['contribution_kj']), reverse=True)

    result['success'] = True
    result['E_def_kJ'] = E_def_total_hartree * EH_TO_KJ
    result['E_def_hartree'] = E_def_total_hartree
    result['E_def_harmonic_kJ'] = E_def_harmonic_hartree * EH_TO_KJ
    result['contributions'] = contributions
    result['top_contributions'] = contributions[:top_n_modes]
    result['n_modes_used'] = len(contributions)
    result['n_modes_total'] = len(ion_modes)

    return result


def run_vpt2_analysis_for_molecule(
    workdir: Path,
    molecule_name: str,
    freq_threshold: float = 50.0,
    top_n_modes: int = 10
) -> Dict[str, Any]:
    """
    Exécute l'analyse VPT2-cubique complète pour une molécule.

    Pour chaque conformère, calcule E_def par VPT2-cubique pour réduction et oxydation.

    Args:
        workdir: Répertoire de travail de la molécule
        molecule_name: Nom de la molécule
        freq_threshold: Seuil minimal de fréquence (cm⁻¹)
        top_n_modes: Nombre de modes dominants à rapporter

    Returns:
        Dict contenant les résultats par conformère pour réduction et oxydation
    """
    workdir = Path(workdir)

    # Localiser les fichiers RKF
    neutral_rkfs = find_neutral_rkf_files(workdir, molecule_name)
    reduced_opt_rkfs = find_opt_rkf_files(workdir, molecule_name, 'reduced')
    oxidized_opt_rkfs = find_opt_rkf_files(workdir, molecule_name, 'oxidized')
    reduced_sp_rkfs = find_sp_rkf_files(workdir, molecule_name, 'reduced')
    oxidized_sp_rkfs = find_sp_rkf_files(workdir, molecule_name, 'oxidized')

    results = {
        'molecule': molecule_name,
        'conformers': {},
    }

    for conf_num in sorted(neutral_rkfs.keys()):
        neutral_rkf = neutral_rkfs.get(conf_num)
        reduced_opt_rkf = reduced_opt_rkfs.get(conf_num)
        oxidized_opt_rkf = oxidized_opt_rkfs.get(conf_num)
        reduced_sp_rkf = reduced_sp_rkfs.get(conf_num)
        oxidized_sp_rkf = oxidized_sp_rkfs.get(conf_num)

        conf_results = {
            'reduction': None,
            'oxidation': None,
        }

        # Réduction
        if neutral_rkf and reduced_opt_rkf and reduced_sp_rkf:
            conf_results['reduction'] = compute_vpt2_cubic_edef(
                neutral_rkf, reduced_opt_rkf, reduced_sp_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )

        # Oxydation
        if neutral_rkf and oxidized_opt_rkf and oxidized_sp_rkf:
            conf_results['oxidation'] = compute_vpt2_cubic_edef(
                neutral_rkf, oxidized_opt_rkf, oxidized_sp_rkf,
                freq_threshold=freq_threshold, top_n_modes=top_n_modes
            )

        results['conformers'][conf_num] = conf_results

    return results
