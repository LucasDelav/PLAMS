#!/usr/bin/env amspython
"""
Diagnostic approfondi des erreurs du Morse-Duschinsky.

Pour chaque molécule, analyse :
1. La répartition des contributions par mode (harmonique vs Morse)
2. L'erreur cumulative mode par mode
3. Le signe du gradient vs le signe du déplacement (cohérence)
4. La saturation Morse (modes où E_Morse ≈ D = dissociation)
5. L'effet de l'approximation de modes indépendants
"""

import numpy as np
from pathlib import Path
from scipy.optimize import brentq
from scm.plams import KFReader

EH_TO_KJ = 2625.499638

base = Path(__file__).resolve().parent.parent / "data" / "1594174" / "dir_workdir"

def find_rkf(mol_name, state):
    plams_dir = base / f"{mol_name}_workdir/redox/calculations/plams_workdir"
    if not plams_dir.exists():
        return None
    for d in plams_dir.iterdir():
        if d.is_dir() and f"{mol_name}_conf_1_{state}" in d.name and "_OLD" not in d.name:
            rkf = d / "adf.rkf"
            if rkf.exists():
                return rkf
    return None


def solve_morse_a(k, dQ, F_abs, a_min=1e-6, a_max=50.0):
    dQ_abs = abs(dQ)
    if dQ_abs < 1e-10 or F_abs < 1e-15 or k <= 0:
        return None
    F_harmonic = k * dQ_abs
    if F_abs >= F_harmonic * 0.99:
        return None
    def func(a):
        u = np.exp(-a * dQ_abs)
        return (k / a) * (u - u**2) - F_abs
    try:
        f_low = func(a_min)
        f_high = func(a_max)
        if f_low * f_high > 0:
            f_high = func(100.0)
            if f_low * f_high > 0:
                return None
            a_max = 100.0
        return brentq(func, a_min, a_max, xtol=1e-10)
    except (ValueError, RuntimeError):
        return None


def morse_energy(k, dQ, a):
    D = k / (2 * a**2)
    return D * (1 - np.exp(-a * abs(dQ)))**2


def analyze_molecule_detailed(mol_name, process='reduction'):
    neutral_rkf = find_rkf(mol_name, "neutral")
    if process == 'reduction':
        ion_opt_rkf = find_rkf(mol_name, "reduced_opt")
        ion_sp_rkf = find_rkf(mol_name, "reduced_sp")
    else:
        ion_opt_rkf = find_rkf(mol_name, "oxidized_opt")
        ion_sp_rkf = find_rkf(mol_name, "oxidized_sp")

    if not all([neutral_rkf, ion_opt_rkf, ion_sp_rkf]):
        return None

    try:
        kf_n = KFReader(str(neutral_rkf))
        kf_ion = KFReader(str(ion_opt_rkf))
        kf_sp = KFReader(str(ion_sp_rkf))

        coords_n = np.array(kf_n.read("Molecule", "Coords"), dtype=float)
        coords_ion = np.array(kf_ion.read("Molecule", "Coords"), dtype=float)
        ion_masses = np.array(kf_ion.read("Molecule", "AtomMasses"), dtype=float)
        ion_masses_3N = np.repeat(ion_masses, 3)

        gradients = np.array(kf_sp.read("AMSResults", "Gradients"), dtype=float)
        ion_freq = np.array(kf_ion.read("Vibrations", "Frequencies[cm-1]"), dtype=float)
        ion_fc = np.array(kf_ion.read("Vibrations", "ForceConstants"), dtype=float)
        n_modes = len(ion_freq)

        E_sp = kf_sp.read("Energy", "Bond Energy")
        E_opt = kf_ion.read("Energy", "Bond Energy")
        E_def_real = (E_opt - E_sp) * EH_TO_KJ

        delta_R = coords_n - coords_ion

        modes = []
        for j in range(n_modes):
            if ion_freq[j] < 50.0 or ion_fc[j] <= 0:
                continue

            mode_ion = np.array(kf_ion.read("Vibrations", f"NoWeightNormalMode({j+1})"), dtype=float)
            k_j = ion_fc[j]

            mu_j = np.sum(ion_masses_3N * mode_ion**2)
            W_j = np.sum(ion_masses_3N * mode_ion * delta_R)
            dQ_j = W_j / mu_j

            F_j = np.dot(gradients, mode_ion)

            E_harm_j = -0.5 * k_j * dQ_j**2
            h_j = F_j / (-k_j * dQ_j) if abs(dQ_j) > 1e-10 else 1.0

            a_j = solve_morse_a(k_j, dQ_j, abs(F_j))

            if a_j is not None and abs(dQ_j) > 0.05:
                E_morse_j = -morse_energy(k_j, dQ_j, a_j)
                D_j = k_j / (2 * a_j**2)
                saturation = morse_energy(k_j, dQ_j, a_j) / D_j  # ratio E/D
            else:
                E_morse_j = E_harm_j
                D_j = None
                saturation = None

            # Cohérence gradient-déplacement :
            # Sur la PES de l'ion, à la géométrie neutre :
            #   - F_j < 0 et dQ_j > 0 : gradient pousse vers l'ion (cohérent)
            #   - F_j > 0 et dQ_j < 0 : idem (cohérent)
            #   - Sinon : incohérent (le gradient ne pointe pas vers le minimum ion)
            coherent = (F_j * dQ_j < 0) if abs(dQ_j) > 1e-10 else True

            modes.append({
                'mode': j + 1,
                'freq': ion_freq[j],
                'k': k_j,
                'dQ': dQ_j,
                'F': F_j,
                'h': h_j,
                'a': a_j,
                'D': D_j,
                'saturation': saturation,
                'E_harm': E_harm_j * EH_TO_KJ,
                'E_morse': E_morse_j * EH_TO_KJ,
                'coherent': coherent,
            })

        return {
            'mol': mol_name,
            'E_real': E_def_real,
            'modes': sorted(modes, key=lambda x: abs(x['E_harm']), reverse=True),
        }

    except Exception as e:
        return {'mol': mol_name, 'error': str(e)}


# =====================================================
# Analyse détaillée
# =====================================================

molecules = [
    "H2C-CA", "Al-CA", "CA", "DHI", "Cl-DA", "F-DA",
    "HO-CA", "Cl-HMC",
    "DA", "F-HMC", "HMC",
]
molecules = list(dict.fromkeys(molecules))

print("=" * 160)
print("DIAGNOSTIC DES ERREURS MORSE-DUSCHINSKY — RÉDUCTION")
print("=" * 160)

for mol_name in molecules:
    r = analyze_molecule_detailed(mol_name, 'reduction')
    if r is None or 'error' in r:
        continue

    modes = r['modes']
    E_real = r['E_real']
    E_harm_total = sum(m['E_harm'] for m in modes)
    E_morse_total = sum(m['E_morse'] for m in modes)
    err_morse = E_morse_total - E_real

    # Décomposer l'erreur
    n_incoherent = sum(1 for m in modes if not m['coherent'])
    n_saturated = sum(1 for m in modes if m['saturation'] is not None and m['saturation'] > 0.9)
    n_morse_applied = sum(1 for m in modes if m['a'] is not None and abs(m['dQ']) > 0.05)
    E_incoherent = sum(m['E_morse'] for m in modes if not m['coherent'])

    # Somme des contributions "cohérentes" vs "incohérentes"
    E_coherent_morse = sum(m['E_morse'] for m in modes if m['coherent'])
    E_coherent_harm = sum(m['E_harm'] for m in modes if m['coherent'])

    print(f"\n{'='*80}")
    print(f"{mol_name}  |  Réel={E_real:.2f}  Harm={E_harm_total:.2f}  Morse={E_morse_total:.2f}  Err_Morse={err_morse:+.2f} kJ/mol")
    print(f"  Modes: {len(modes)} total, {n_morse_applied} Morse appliqué, {n_incoherent} incohérents (F et dQ même signe), {n_saturated} saturés (E/D > 0.9)")
    print(f"  E_incohérent = {E_incoherent:.2f} kJ/mol (contributions des modes avec gradient incohérent)")
    print(f"  E_cohérent_Morse = {E_coherent_morse:.2f}, E_cohérent_Harm = {E_coherent_harm:.2f}")
    print(f"  Si on exclut les incohérents: E_Morse_filtre = {E_coherent_morse:.2f} (Δ={E_coherent_morse - E_real:+.2f})")

    if abs(err_morse) > 5.0:
        print(f"\n  {'Mode':>5} {'Freq':>8} {'dQ':>10} {'F':>12} {'h':>8} {'a':>8} {'D(kJ)':>8} {'sat':>6} {'E_harm':>10} {'E_morse':>10} {'coh':>5}")
        for m in modes[:20]:
            a_str = f"{m['a']:.4f}" if m['a'] is not None else "harm"
            D_str = f"{m['D']*EH_TO_KJ:.1f}" if m['D'] is not None else "---"
            sat_str = f"{m['saturation']:.3f}" if m['saturation'] is not None else "---"
            coh_str = "OK" if m['coherent'] else "BAD"
            print(f"  {m['mode']:5d} {m['freq']:8.1f} {m['dQ']:10.4f} {m['F']:12.6f} {m['h']:8.4f} {a_str:>8} {D_str:>8} {sat_str:>6} {m['E_harm']:10.2f} {m['E_morse']:10.2f} {coh_str:>5}")


# =====================================================
# Résumé : test du filtrage par cohérence
# =====================================================
print("\n\n" + "=" * 160)
print("RÉSUMÉ : EFFET DU FILTRAGE PAR COHÉRENCE GRADIENT-DÉPLACEMENT")
print("=" * 160)

print(f"\n{'Molécule':<12} {'Réel':>10} {'Morse':>10} {'Morse_filt':>12} {'Δ_Morse':>10} {'Δ_filt':>10} {'n_incoh':>8}")
print("-" * 80)

for mol_name in molecules:
    r = analyze_molecule_detailed(mol_name, 'reduction')
    if r is None or 'error' in r:
        continue
    modes = r['modes']
    E_real = r['E_real']
    E_morse_total = sum(m['E_morse'] for m in modes)
    # Filtrage : exclure les modes incohérents (gradient dans la mauvaise direction)
    E_morse_filtered = sum(m['E_morse'] for m in modes if m['coherent'])
    n_incoh = sum(1 for m in modes if not m['coherent'])

    d_morse = E_morse_total - E_real
    d_filt = E_morse_filtered - E_real

    flag = " ✓" if abs(d_filt) < abs(d_morse) * 0.7 else ""
    print(f"{mol_name:<12} {E_real:10.2f} {E_morse_total:10.2f} {E_morse_filtered:12.2f} {d_morse:10.2f} {d_filt:10.2f} {n_incoh:8d}{flag}")


# =====================================================
# Test : limiter dQ max pour éviter les modes trop déplacés
# =====================================================
print("\n\n" + "=" * 160)
print("TEST : EFFET D'UN CAP SUR |dQ| (modes avec |dQ| > seuil → E = E_LVC_mode)")
print("=" * 160)

# Pour les modes très déplacés, utiliser la contribution LVC au lieu de Morse
# L'idée : si dQ est trop grand, le mode est probablement dans un nouveau bassin
# et ni Morse ni harmonique n'a de sens. Le LVC (gradient seul) est alors plus fiable.

for dQ_cap in [0.2, 0.5, 1.0, 2.0]:
    print(f"\n--- Cap |dQ| = {dQ_cap} Bohr ---")
    print(f"{'Molécule':<12} {'Réel':>10} {'Morse':>10} {'Morse_cap':>12} {'Δ_Morse':>10} {'Δ_cap':>10}")
    print("-" * 70)

    for mol_name in molecules:
        r = analyze_molecule_detailed(mol_name, 'reduction')
        if r is None or 'error' in r:
            continue
        modes = r['modes']
        E_real = r['E_real']
        E_morse_total = sum(m['E_morse'] for m in modes)

        # Pour modes avec |dQ| > cap, utiliser contribution LVC : -F²/(2k)
        E_capped = 0.0
        for m in modes:
            if abs(m['dQ']) > dQ_cap:
                # LVC contribution pour ce mode
                E_capped += -(m['F']**2) / (2.0 * m['k']) * EH_TO_KJ
            else:
                E_capped += m['E_morse']

        d_morse = E_morse_total - E_real
        d_cap = E_capped - E_real

        print(f"{mol_name:<12} {E_real:10.2f} {E_morse_total:10.2f} {E_capped:12.2f} {d_morse:10.2f} {d_cap:10.2f}")


# =====================================================
# Test : Morse avec fallback LVC pour modes incohérents
# =====================================================
print("\n\n" + "=" * 160)
print("TEST : MORSE + FALLBACK LVC POUR MODES INCOHÉRENTS")
print("=" * 160)

print(f"\n{'Molécule':<12} {'Réel':>10} {'Morse':>10} {'LVC':>10} {'Morse+LVC':>12} {'Δ_Morse':>10} {'Δ_M+L':>10}")
print("-" * 80)

for mol_name in molecules:
    r = analyze_molecule_detailed(mol_name, 'reduction')
    if r is None or 'error' in r:
        continue
    modes = r['modes']
    E_real = r['E_real']
    E_morse_total = sum(m['E_morse'] for m in modes)

    # Hybrid : Morse pour modes cohérents, LVC pour modes incohérents
    E_hybrid = 0.0
    E_lvc_total = 0.0
    for m in modes:
        lvc_contrib = -(m['F']**2) / (2.0 * m['k']) * EH_TO_KJ
        E_lvc_total += lvc_contrib
        if m['coherent']:
            E_hybrid += m['E_morse']
        else:
            E_hybrid += lvc_contrib

    d_morse = E_morse_total - E_real
    d_hybrid = E_hybrid - E_real

    flag = " ✓" if abs(d_hybrid) < abs(d_morse) * 0.7 else ""
    print(f"{mol_name:<12} {E_real:10.2f} {E_morse_total:10.2f} {E_lvc_total:10.2f} {E_hybrid:12.2f} {d_morse:10.2f} {d_hybrid:10.2f}{flag}")
