#!/usr/bin/env python3
"""Section 4.2 -- Erelax avec LVC2 RECALCULE (seuil freq > 50 cm-1).

Re-calcule le LVC2 a la volee depuis les modes/gradients bruts du
dataset_predict_dg/ en appliquant un seuil de 50 cm-1 sur les frequences
positives, ce qui elimine les modes mous artefactuels qui faisaient diverger
la formule F^2/(2k). Voir test /tmp/test_lvc2_threshold.py qui montre que
ce seuil reproduit le R^2=0.84 du CSV de la sec. 3.2 du rapport.

Pipeline:
  1. Boucle sur les 10 solvants
  2. Pour chaque conformere : recalcule LVC2_red et LVC2_ox avec freq > 50
  3. Filtre EA<0, DEA, etc.
  4. Test par solvant (15 catechols + tous) et global
  5. Compare aux figures 22-23 du rapport
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

_rapport_dir = Path(__file__).resolve().parent.parent / "rapport"
sys.path.insert(0, str(_rapport_dir))
from style import (
    apply_style, setup_grid,
    C_RED_DARK, C_BLUE_DARK, C_GRAY,
)
apply_style()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_PDG_DIR = PROJECT_ROOT / "dataset_predict_dg"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "section42_recalc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EH_TO_KJ = 2625.499638
AMU_TO_ME = 1822.888486209
CM1_TO_HARTREE = 1.0 / 219474.6313705
FREQ_MIN_CM1 = 50.0  # Seuil de coupure pour LVC2

ATOM_MASSES = {'H': 1.00794, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994,
               'F': 18.9984, 'P': 30.9738, 'S': 32.065, 'Cl': 35.453,
               'Br': 79.904, 'I': 126.9045}

SOLVENT_EPSILON = {
    "Water": 78.4, "DMSO": 47.24, "Acetonitrile": 35.69, "Methanol": 32.66,
    "Acetone": 20.7, "Dichloromethane": 8.93, "Tetrahydrofuran": 7.58,
    "Chloroform": 4.81, "Toluene": 2.38, "Hexane": 1.88,
}
SOLVENT_ORDER = sorted(SOLVENT_EPSILON.keys(),
                       key=lambda s: SOLVENT_EPSILON[s], reverse=True)

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC', 'L-DOPA'}
CATECHOLS_15 = {
    'Al-CA', 'CA', 'Cl-DA', 'Cl-HMC', 'DA', 'DHI', 'F-DA', 'F-HMC',
    'H2C-CA', 'HMC', 'HO-CA', 'L-DA', 'O-DHI', 'P-DHI', 'S-DHI',
}


def safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def recompute_lvc2(jf_pdg, ion_state, freq_min=FREQ_MIN_CM1):
    """Recompute LVC2 in kJ/mol from raw data with frequency threshold."""
    try:
        d = json.load(open(jf_pdg))
    except Exception:
        return None
    ion = d.get('states', {}).get(ion_state, {})
    grad = ion.get('gradients')
    modes = ion.get('vibrations', {}).get('modes')
    xyz = ion.get('xyz', '')
    if not grad or not modes or not xyz:
        return None
    g = np.array(grad).flatten()
    atoms = []
    for line in xyz.strip().split('\n')[2:]:
        parts = line.split()
        if parts:
            atoms.append(parts[0])
    if len(atoms) * 3 != len(g):
        return None
    masses_3n = np.array([ATOM_MASSES.get(s, 12.0) * AMU_TO_ME
                          for s in atoms for _ in range(3)])
    e_relax = 0.0
    for m in modes:
        freq = m.get('frequency_cm1')
        disp = m.get('displacements')
        if freq is None or disp is None:
            continue
        if freq < freq_min:
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
    return e_relax * EH_TO_KJ


def get_oxygen_indices(xyz_str):
    O_idx = []
    for i, line in enumerate(xyz_str.strip().split('\n')[2:]):
        parts = line.split()
        if parts and parts[0].strip().upper() == 'O':
            O_idx.append(i)
    return O_idx


def parse_predict_dg(jf):
    """Returns recomputed LVC2 + Fukui descriptors, or None."""
    try:
        d = json.load(open(jf))
    except Exception:
        return None
    lvc2_r = recompute_lvc2(jf, 'reduced_sp')
    lvc2_o = recompute_lvc2(jf, 'oxidized_sp')
    if lvc2_r is None or lvc2_o is None:
        return None
    if abs(lvc2_r) > 150 or abs(lvc2_o) > 150:
        return None
    ns = d.get('states', {}).get('neutral_sp', {})
    cdft = ns.get('cdft', {})
    fplus_at = cdft.get('fplus')
    fminus_at = cdft.get('fminus')
    if not fplus_at or not fminus_at:
        return None
    O_idx = get_oxygen_indices(ns.get('xyz', ''))
    if not O_idx:
        return None
    fplus_O = np.array(fplus_at)[O_idx]
    fminus_O = np.array(fminus_at)[O_idx]
    return {
        'lvc2_red': lvc2_r, 'lvc2_ox': lvc2_o,
        'fplus_sumO': float(fplus_O.sum()),
        'fminus_sumO': float(fminus_O.sum()),
        'fplus_maxO': float(np.max(np.abs(fplus_O))),
        'fminus_maxO': float(np.max(np.abs(fminus_O))),
        'mu': cdft.get('mu'), 'eta': cdft.get('eta'), 'omega': cdft.get('omega'),
    }


def parse_dft(jf):
    try:
        d = json.load(open(jf))
    except Exception:
        return None
    s = d.get('states', {})
    ee_n_tz = safe_get(s, 'neutral_tz2p', 'thermo', 'Ee')
    ee_o_sp = safe_get(s, 'oxidized_sp', 'thermo', 'Ee')
    ee_o_tz = safe_get(s, 'oxidized_tz2p', 'thermo', 'Ee')
    ee_r_sp = safe_get(s, 'reduced_sp', 'thermo', 'Ee')
    ee_r_tz = safe_get(s, 'reduced_tz2p', 'thermo', 'Ee')
    if None in (ee_n_tz, ee_o_sp, ee_o_tz, ee_r_sp, ee_r_tz):
        return None
    EA = (ee_n_tz - ee_r_sp) * EH_TO_KJ
    return {
        'real_ox': (ee_o_tz - ee_o_sp) * EH_TO_KJ,
        'real_red': (ee_r_tz - ee_r_sp) * EH_TO_KJ,
        'EA': EA,
        'lumo_eV': safe_get(s, 'neutral_dzp', 'orbitals', 'LUMO'),
    }


def load_dataset():
    records = []
    for solv_dir in sorted(DATASET_DIR.iterdir()):
        if not solv_dir.is_dir() or solv_dir.name not in SOLVENT_EPSILON:
            continue
        solv = solv_dir.name
        # Build PDG lookup
        lookup = {}
        pdg_solv = DATASET_PDG_DIR / solv
        if pdg_solv.exists():
            for mol_dir in pdg_solv.iterdir():
                if not mol_dir.is_dir():
                    continue
                for jf in mol_dir.glob(f"{mol_dir.name}_conf*_prediction.json"):
                    name = jf.stem
                    try:
                        conf = int(name.split("_conf")[-1].split("_")[0])
                    except ValueError:
                        continue
                    info = parse_predict_dg(jf)
                    if info is not None:
                        lookup[(mol_dir.name, conf)] = info
        for mol_dir in sorted(solv_dir.iterdir()):
            if not mol_dir.is_dir() or mol_dir.name in DEA_EXCLUDED:
                continue
            mol = mol_dir.name
            for cf in sorted(mol_dir.glob(f"{mol}_conf*_DFT.json")):
                name = cf.stem
                try:
                    conf = int(name.split("_conf")[-1].split("_")[0])
                except ValueError:
                    continue
                dft = parse_dft(cf)
                if dft is None or dft['EA'] < 0:
                    continue
                info = lookup.get((mol, conf))
                if info is None:
                    continue
                if abs(dft['real_ox']) > 200 or abs(dft['real_red']) > 200:
                    continue
                records.append({
                    'molecule': mol, 'conformer': conf, 'solvent': solv,
                    'epsilon': SOLVENT_EPSILON[solv],
                    'is_catechol_15': mol in CATECHOLS_15,
                    'real_ox': dft['real_ox'], 'real_red': dft['real_red'],
                    'lumo_eV': dft['lumo_eV'],
                    **info,
                })
    return records


def calibrated_metrics(x, y):
    s, i, r, _, _ = linregress(x, y)
    return float(r ** 2), float(np.mean(np.abs(s * x + i - y))), float(s), float(i)


def fit_a(x_lvc2, fmax, y_real):
    def neg_r2(a):
        x_corr = x_lvc2 * (1 + a * fmax ** 2)
        s, i, r, _, _ = linregress(x_corr, y_real)
        return -r ** 2
    res = minimize_scalar(neg_r2, bounds=(0.0, 50.0), method='bounded')
    return float(res.x)


def evaluate_models(recs, key_real, key_lvc2, key_fmax):
    if len(recs) < 8:
        return None
    x_lvc = np.array([r[key_lvc2] for r in recs])
    y_real = np.array([r[key_real] for r in recs])
    fmax = np.array([r[key_fmax] for r in recs])
    out = {}
    out['brut'] = calibrated_metrics(x_lvc, y_real)
    x_c = x_lvc * (1 + np.abs(fmax) ** 1.5)
    out['fmax15'] = calibrated_metrics(x_c, y_real)
    a_opt = fit_a(x_lvc, fmax, y_real)
    x_d = x_lvc * (1 + a_opt * fmax ** 2)
    r2, mae, s, i = calibrated_metrics(x_d, y_real)
    out['fmax_a'] = (r2, mae, a_opt)
    return out


def run_per_solvent(records):
    print("\n" + "=" * 100)
    print(f"{'Solvent':<18} {'Subset':<10} {'n':>5} | {'Process':<5} | "
          f"{'brut R²':>9} {'fmax15 R²':>11} {'fmax_a R²':>11} {'a':>6}")
    print("=" * 100)
    summary = []
    for solv in SOLVENT_ORDER:
        recs_solv = [r for r in records if r['solvent'] == solv]
        for subset_name, recs in [('15 cat', [r for r in recs_solv if r['is_catechol_15']]),
                                  ('all', recs_solv)]:
            for proc, kr, kl, kf in [('ox', 'real_ox', 'lvc2_ox', 'fminus_maxO'),
                                     ('red', 'real_red', 'lvc2_red', 'fplus_maxO')]:
                res = evaluate_models(recs, kr, kl, kf)
                if res is None:
                    continue
                a = res['fmax_a'][2]
                print(f"{solv:<18} {subset_name:<10} {len(recs):>5} | {proc:<5} | "
                      f"{res['brut'][0]:>9.3f} {res['fmax15'][0]:>11.3f} "
                      f"{res['fmax_a'][0]:>11.3f} {a:>6.2f}")
                summary.append({'solvent': solv, 'subset': subset_name,
                                'n': len(recs), 'process': proc, **res})
    return summary


def plot_persolvent_15cat(records, key_real, key_lvc2, key_fmax,
                          process, fname):
    color = C_RED_DARK if process == 'ox' else C_BLUE_DARK
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    for ax, solv in zip(axes, SOLVENT_ORDER):
        recs = [r for r in records if r['solvent'] == solv and r['is_catechol_15']]
        if len(recs) < 5:
            ax.set_title(f"{solv} (n={len(recs)})", fontsize=9)
            ax.text(0.5, 0.5, "n<5", ha='center', va='center',
                    transform=ax.transAxes)
            continue
        x_lvc = np.array([r[key_lvc2] for r in recs])
        y_real = np.array([r[key_real] for r in recs])
        fmax = np.array([r[key_fmax] for r in recs])
        x_c = x_lvc * (1 + np.abs(fmax) ** 1.5)
        s, i, rval, _, _ = linregress(x_c, y_real)
        y_cal = s * x_c + i
        r2 = rval ** 2
        mae = float(np.mean(np.abs(y_cal - y_real)))
        ax.scatter(y_real, y_cal, s=24, alpha=0.7, color=color,
                   edgecolors='white', linewidths=0.3)
        all_v = np.concatenate([y_real, y_cal])
        margin = (all_v.max() - all_v.min()) * 0.12
        lims = [all_v.min() - margin, all_v.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("DFT (kJ/mol)", fontsize=8)
        ax.set_ylabel("LVC2×(1+|f|^1.5) cal. (kJ/mol)", fontsize=8)
        ax.set_title(f"{solv} (ε={SOLVENT_EPSILON[solv]:.1f})\n"
                     f"n={len(recs)} R²={r2:.2f} MAE={mae:.1f}",
                     fontsize=9)
        setup_grid(ax)
    fig.suptitle(f"$E_\\mathrm{{relax}}^\\mathrm{{{process}}}$ : 15 catéchols par solvant "
                 f"(LVC2 recalculé, freq>{FREQ_MIN_CM1:.0f} cm⁻¹)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → {fname}")


def plot_global(records, key_real, key_lvc2, key_fmax, process, fname):
    """Global multi-solvent parity, color by solvent."""
    x_lvc = np.array([r[key_lvc2] for r in records])
    y_real = np.array([r[key_real] for r in records])
    fmax = np.array([r[key_fmax] for r in records])
    color = C_RED_DARK if process == 'ox' else C_BLUE_DARK

    SOLVENT_COLOR = plt.cm.RdYlBu(np.linspace(0.05, 0.95, 10))
    cmap = {s: SOLVENT_COLOR[i] for i, s in enumerate(SOLVENT_ORDER)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ['LVC2 brut', 'LVC2 × (1+|f_max,O|^1.5)', 'LVC2 × (1+a·f_max,O²)']
    x_models = [
        x_lvc,
        x_lvc * (1 + np.abs(fmax) ** 1.5),
        None,  # filled below
    ]
    a_opt = fit_a(x_lvc, fmax, y_real)
    x_models[2] = x_lvc * (1 + a_opt * fmax ** 2)
    for ax, x_used, title in zip(axes, x_models, titles):
        s, i, rval, _, _ = linregress(x_used, y_real)
        y_cal = s * x_used + i
        r2 = rval ** 2
        mae = float(np.mean(np.abs(y_cal - y_real)))
        for solv in SOLVENT_ORDER:
            mask = np.array([r['solvent'] == solv for r in records])
            if mask.sum() == 0: continue
            ax.scatter(y_real[mask], y_cal[mask], color=cmap[solv],
                       s=10, alpha=0.55, edgecolors='white', linewidths=0.15,
                       label=f"{solv} (ε={SOLVENT_EPSILON[solv]:.1f})")
        all_v = np.concatenate([y_real, y_cal])
        margin = (all_v.max() - all_v.min()) * 0.10
        lims = [all_v.min() - margin, all_v.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"E_relax_{process} DFT (kJ/mol)")
        ax.set_ylabel(f"E_relax_{process} prédit calibré")
        suffix = f"\na={a_opt:.2f}" if 'a·' in title else ""
        ax.set_title(f"{title}\nR²={r2:.3f}  MAE={mae:.2f}{suffix}", fontsize=10)
        setup_grid(ax)
    axes[2].legend(fontsize=6, loc='upper left', framealpha=0.85, ncol=1)
    fig.suptitle(f"$E_\\mathrm{{relax}}^\\mathrm{{{process}}}$ multi-solvant (n={len(records)}, "
                 f"LVC2 recalculé freq>{FREQ_MIN_CM1:.0f} cm⁻¹)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → {fname}")


def main():
    print("=" * 70)
    print(f"SECTION 4.2 -- LVC2 RECALCULE (freq > {FREQ_MIN_CM1:.0f} cm-1)")
    print("=" * 70)
    records = load_dataset()
    print(f"\n{len(records)} conformeres apres filtres")
    n_15cat = sum(1 for r in records if r['is_catechol_15'])
    print(f"  dont 15 catechols: {n_15cat}")

    summary = run_per_solvent(records)

    print("\n" + "=" * 70)
    print("=== Figures par solvant (15 catechols) ===")
    print("=" * 70)
    plot_persolvent_15cat(records, 'real_ox', 'lvc2_ox', 'fminus_maxO',
                          'ox', 's42r_persolvent_ox_15cat.png')
    plot_persolvent_15cat(records, 'real_red', 'lvc2_red', 'fplus_maxO',
                          'red', 's42r_persolvent_red_15cat.png')

    print("\n" + "=" * 70)
    print("=== Figures globales multi-solvant (tous conformeres) ===")
    print("=" * 70)
    plot_global(records, 'real_ox', 'lvc2_ox', 'fminus_maxO',
                'ox', 's42r_global_ox.png')
    plot_global(records, 'real_red', 'lvc2_red', 'fplus_maxO',
                'red', 's42r_global_red.png')

    # Global summary line
    x_lvc_red = np.array([r['lvc2_red'] for r in records])
    y_red = np.array([r['real_red'] for r in records])
    s, i, r, _, _ = linregress(x_lvc_red, y_red)
    print(f"\nGLOBAL multi-solvant n={len(records)}:")
    print(f"  Reduction  : R² brut = {r**2:.3f}  MAE = {np.mean(np.abs(s*x_lvc_red+i-y_red)):.2f}")
    x_lvc_ox = np.array([r['lvc2_ox'] for r in records])
    y_ox = np.array([r['real_ox'] for r in records])
    s, i, r, _, _ = linregress(x_lvc_ox, y_ox)
    print(f"  Oxydation  : R² brut = {r**2:.3f}  MAE = {np.mean(np.abs(s*x_lvc_ox+i-y_ox)):.2f}")


if __name__ == '__main__':
    main()
