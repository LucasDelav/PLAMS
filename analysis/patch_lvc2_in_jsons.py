#!/usr/bin/env python3
"""Mise a jour in-place des champs derived.E_relax_LVC2_red/ox dans les JSON
predict-dg, en appliquant le seuil FREQ_MIN_CM1 = 50 cm-1.

Lit les modes/gradients deja stockes dans chaque JSON (pas de calcul DFT
necessaire), recalcule LVC2 propre, et reecrit le fichier en preservant
toutes les autres donnees.

Sauvegarde l'ancienne valeur dans 'derived.E_relax_LVC2_X_legacy' (freq>0)
au cas ou.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDG_DIR = PROJECT_ROOT / "dataset_predict_dg"

EH_TO_KJ = 2625.499638
EV_TO_HARTREE = 1.0 / 27.211386245988
AMU_TO_ME = 1822.888486209
CM1_TO_HARTREE = 1.0 / 219474.6313705
FREQ_MIN_CM1 = 50.0

ATOM_MASSES = {'H': 1.00794, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994,
               'F': 18.9984, 'P': 30.9738, 'S': 32.065, 'Cl': 35.453,
               'Br': 79.904, 'I': 126.9045}


def safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def parse_atoms(xyz_str):
    return [line.split()[0] for line in xyz_str.strip().split('\n')[2:]
            if line.split()]


def compute_lvc2_kj(state_block, freq_min):
    """Returns LVC2 in Ha or None."""
    grad = state_block.get('gradients')
    modes = state_block.get('vibrations', {}).get('modes')
    xyz = state_block.get('xyz', '')
    if not grad or not modes or not xyz:
        return None
    g = np.array(grad).flatten()
    atoms = parse_atoms(xyz)
    if len(atoms) * 3 != len(g):
        return None
    masses_3n = np.array([ATOM_MASSES.get(s, 12.0) * AMU_TO_ME
                          for s in atoms for _ in range(3)])
    e_relax = 0.0
    for m in modes:
        freq = m.get('frequency_cm1')
        disp = m.get('displacements')
        if freq is None or disp is None or freq < freq_min:
            continue
        mode = np.array(disp).flatten()
        if mode.size != g.size:
            continue
        f_j = float(g @ mode)
        mu_j = float(np.sum(masses_3n * mode ** 2))
        omega_j = freq * CM1_TO_HARTREE
        k_j = mu_j * omega_j ** 2
        if k_j > 1e-12:
            e_relax -= f_j ** 2 / (2.0 * k_j)
    return e_relax  # Hartree


def make_unit_dict(value_ha):
    return {
        'ha': value_ha,
        'kj': value_ha * EH_TO_KJ,
    }


def patch_file(jf):
    try:
        d = json.load(open(jf))
    except Exception as e:
        return f"PARSE_ERROR: {e}"
    derived = d.setdefault('derived', {})
    states = d.get('states', {})

    changed = False
    for ion_state, key in [('reduced_sp', 'E_relax_LVC2_red'),
                           ('oxidized_sp', 'E_relax_LVC2_ox')]:
        sb = states.get(ion_state)
        if sb is None:
            continue
        new_ha = compute_lvc2_kj(sb, FREQ_MIN_CM1)
        if new_ha is None:
            continue
        # Save legacy if not already done
        legacy_key = key + '_legacy_freq0'
        if key in derived and legacy_key not in derived:
            derived[legacy_key] = derived[key]
        new_block = make_unit_dict(new_ha)
        # Compare with old
        old = derived.get(key, {}) or {}
        old_ha = old.get('ha') if isinstance(old, dict) else None
        if old_ha is None or abs(old_ha - new_ha) > 1e-9:
            derived[key] = new_block
            changed = True

    if changed:
        # Atomic write
        tmp = jf.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(d, f, indent=2)
        tmp.replace(jf)
        return "UPDATED"
    return "UNCHANGED"


def main():
    if not PDG_DIR.exists():
        print(f"ERROR: {PDG_DIR} not found")
        sys.exit(1)
    files = list(PDG_DIR.glob("*/*/*_conf*_prediction.json"))
    print(f"Found {len(files)} prediction JSONs")
    n_upd = n_unch = n_err = 0
    for i, jf in enumerate(files):
        if i % 200 == 0:
            print(f"  [{i}/{len(files)}] {jf.relative_to(PDG_DIR)}")
        status = patch_file(jf)
        if status == "UPDATED":
            n_upd += 1
        elif status == "UNCHANGED":
            n_unch += 1
        else:
            n_err += 1
            print(f"  WARN {jf.name}: {status}")
    print(f"\nDone. Updated={n_upd}, Unchanged={n_unch}, Errors={n_err}")


if __name__ == '__main__':
    main()
