#!/usr/bin/env python3
"""Section 4.2 -- Erelax modele Fukui SOLVANT PAR SOLVANT.

Objectif : reproduire le resultat de la section 3.2 (R^2 = 0.76-0.89 sur 13
catechols Acetonitrile) en filtrant sur Acetonitrile uniquement, puis comparer
sur les autres solvants. Permet de detecter si la perte de R^2 multi-solvant
vient de la diversite chimique ou du melange de solvants.

Pour chaque solvant, on teste:
  - LVC2 brut calibre
  - LVC2 * (1 + |f+/-_max,O|^1.5)        Fig. 15 du rapport
  - LVC2 * (1 + a |f+/-_max,O|^2)        Fig. 16 du rapport
  - LVC2 + epsilon_LUMO (baseline)

Sur deux jeux:
  - 13 catechols stables (CA, HMC, DA, DHI families, sans DEA)
  - Tous les conformeres du solvant (toutes familles)
"""

import json
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "section42_persolvent"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EH_TO_KJ = 2625.499638

SOLVENT_EPSILON = {
    "Water": 78.4, "DMSO": 47.24, "Acetonitrile": 35.69, "Methanol": 32.66,
    "Acetone": 20.7, "Dichloromethane": 8.93, "Tetrahydrofuran": 7.58,
    "Chloroform": 4.81, "Toluene": 2.38, "Hexane": 1.88,
}
SOLVENT_ORDER = sorted(SOLVENT_EPSILON.keys(),
                       key=lambda s: SOLVENT_EPSILON[s], reverse=True)

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC', 'L-DOPA'}

# 15 catechols stables (sec. 3.2)
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


def get_oxygen_indices(xyz_str):
    lines = xyz_str.strip().split('\n')
    if len(lines) < 3:
        return []
    O_idx = []
    for i, line in enumerate(lines[2:]):
        parts = line.split()
        if parts and parts[0].strip().upper() == 'O':
            O_idx.append(i)
    return O_idx


def parse_predict_dg(jf):
    try:
        d = json.load(open(jf))
    except Exception:
        return None
    derived = d.get('derived', {})
    lvc2_red = safe_get(derived, 'E_relax_LVC2_red', 'kj')
    lvc2_ox = safe_get(derived, 'E_relax_LVC2_ox', 'kj')
    if lvc2_red is None or lvc2_ox is None:
        return None
    if abs(lvc2_red) > 150 or abs(lvc2_ox) > 150:
        return None
    ns = d.get('states', {}).get('neutral_sp', {})
    cdft = ns.get('cdft', {})
    fplus_at = cdft.get('fplus')
    fminus_at = cdft.get('fminus')
    if not fplus_at or not fminus_at:
        return None
    xyz = ns.get('xyz', '')
    O_idx = get_oxygen_indices(xyz)
    if not O_idx:
        return None
    fplus_at = np.array(fplus_at)
    fminus_at = np.array(fminus_at)
    fplus_O = fplus_at[O_idx]
    fminus_O = fminus_at[O_idx]
    return {
        'lvc2_red': lvc2_red, 'lvc2_ox': lvc2_ox,
        'fplus_sumO': float(fplus_O.sum()),
        'fminus_sumO': float(fminus_O.sum()),
        'fplus_maxO': float(np.max(np.abs(fplus_O))),
        'fminus_maxO': float(np.max(np.abs(fminus_O))),
        'mu': cdft.get('mu'),
        'eta': cdft.get('eta'),
    }


def parse_dft(jf):
    try:
        d = json.load(open(jf))
    except Exception:
        return None
    states = d.get('states', {})
    ee_n_tz = safe_get(states, 'neutral_tz2p', 'thermo', 'Ee')
    ee_o_sp = safe_get(states, 'oxidized_sp', 'thermo', 'Ee')
    ee_o_tz = safe_get(states, 'oxidized_tz2p', 'thermo', 'Ee')
    ee_r_sp = safe_get(states, 'reduced_sp', 'thermo', 'Ee')
    ee_r_tz = safe_get(states, 'reduced_tz2p', 'thermo', 'Ee')
    if None in (ee_n_tz, ee_o_sp, ee_o_tz, ee_r_sp, ee_r_tz):
        return None
    EA = (ee_n_tz - ee_r_sp) * EH_TO_KJ
    return {
        'real_ox': (ee_o_tz - ee_o_sp) * EH_TO_KJ,
        'real_red': (ee_r_tz - ee_r_sp) * EH_TO_KJ,
        'EA': EA,
        'lumo_eV': safe_get(states, 'neutral_dzp', 'orbitals', 'LUMO'),
    }


def load_dataset():
    records = []
    for solv_dir in sorted(DATASET_DIR.iterdir()):
        if not solv_dir.is_dir() or solv_dir.name not in SOLVENT_EPSILON:
            continue
        solv = solv_dir.name
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
    lumo = np.array([r['lumo_eV'] if r['lumo_eV'] is not None else np.nan
                     for r in recs])
    valid = ~np.isnan(lumo)

    out = {}
    out['brut'] = calibrated_metrics(x_lvc, y_real)
    # +lumo: multivar linear
    if valid.sum() > 5:
        A = np.vstack([x_lvc[valid], lumo[valid], np.ones(valid.sum())]).T
        coefs, *_ = np.linalg.lstsq(A, y_real[valid], rcond=None)
        pred = A @ coefs
        ss_res = np.sum((y_real[valid] - pred) ** 2)
        ss_tot = np.sum((y_real[valid] - y_real[valid].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        mae = float(np.mean(np.abs(pred - y_real[valid])))
        out['lumo'] = (r2, mae, None, None)
    else:
        out['lumo'] = (np.nan, np.nan, None, None)
    # *fmax^1.5
    x_c = x_lvc * (1 + np.abs(fmax) ** 1.5)
    out['fmax15'] = calibrated_metrics(x_c, y_real)
    # *(1 + a*fmax^2)
    a_opt = fit_a(x_lvc, fmax, y_real)
    x_d = x_lvc * (1 + a_opt * fmax ** 2)
    r2, mae, s, i = calibrated_metrics(x_d, y_real)
    out['fmax_a'] = (r2, mae, a_opt, None)
    return out


def run_per_solvent(records):
    print("\n" + "=" * 95)
    print(f"{'Solvent':<18} {'Subset':<12} {'n':>5} | {'Process':<5} | "
          f"{'brut R²':>9} {'+LUMO':>9} {'fmax15':>9} {'fmax_a':>11}")
    print("=" * 95)

    summary = []
    for solv in SOLVENT_ORDER:
        recs_solv = [r for r in records if r['solvent'] == solv]
        for subset_name, recs in [
            ('all', recs_solv),
            ('15 cat', [r for r in recs_solv if r['is_catechol_15']]),
        ]:
            for proc, kr, kl, kf in [
                ('ox', 'real_ox', 'lvc2_ox', 'fminus_maxO'),
                ('red', 'real_red', 'lvc2_red', 'fplus_maxO'),
            ]:
                res = evaluate_models(recs, kr, kl, kf)
                if res is None:
                    print(f"{solv:<18} {subset_name:<12} {len(recs):>5} | {proc:<5} | (n trop faible)")
                    continue
                a_str = f"a={res['fmax_a'][2]:.1f}"
                print(f"{solv:<18} {subset_name:<12} {len(recs):>5} | {proc:<5} | "
                      f"{res['brut'][0]:>9.3f} {res['lumo'][0]:>9.3f} "
                      f"{res['fmax15'][0]:>9.3f} {res['fmax_a'][0]:>6.3f} {a_str:>5}")
                summary.append({
                    'solvent': solv, 'subset': subset_name, 'n': len(recs),
                    'process': proc, **{k: v for k, v in res.items()},
                })
    return summary


def plot_summary_heatmap(summary):
    """Heatmap: R^2 of best model (LVC2*(1+a*fmax^2)) per solvent x process,
    for the 15 catechols subset."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, proc, label in [(axes[0], 'ox', 'Oxydation'),
                            (axes[1], 'red', 'Réduction')]:
        rows_15 = [s for s in summary if s['subset'] == '15 cat' and s['process'] == proc]
        rows_all = [s for s in summary if s['subset'] == 'all' and s['process'] == proc]
        # Sort by SOLVENT_ORDER
        order = SOLVENT_ORDER
        # Build matrix: rows = subset (all, 15 cat), cols = solvent, value = R^2 fmax_a
        matrix = np.full((4, len(order)), np.nan)
        models = ['brut', 'lumo', 'fmax15', 'fmax_a']
        labels = ['LVC2 brut', '+ε_LUMO', '×(1+|f|^1.5)', '×(1+a·f²)']
        # Use 15-catechol subset
        for j, solv in enumerate(order):
            row = next((r for r in rows_15 if r['solvent'] == solv), None)
            if row is None:
                continue
            for i, m in enumerate(models):
                v = row.get(m)
                if v is not None:
                    matrix[i, j] = v[0]  # R^2
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([f"{s}\nε={SOLVENT_EPSILON[s]:.1f}" for s in order],
                           rotation=30, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{label} — R² (15 catéchols par solvant)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    color = 'black' if matrix[i, j] > 0.5 else 'white'
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center',
                            color=color, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 's42p_r2_heatmap_15cat.png', dpi=150)
    plt.close(fig)
    print(f"\n  → s42p_r2_heatmap_15cat.png")


def plot_per_solvent_15cat(records, key_real, key_lvc2, key_fmax,
                           process, fname):
    """One panel per solvent: parity for LVC2*(1+|fmax|^1.5) on 15 catechols."""
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
    fig.suptitle(f"$E_\\mathrm{{relax}}^\\mathrm{{{process}}}$ : 15 catéchols, "
                 f"un panneau par solvant — modèle LVC2×(1+|f$_{{\\max,O}}$|^1.5)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → {fname}")


def main():
    print("=" * 70)
    print("SECTION 4.2 -- Erelax PAR SOLVANT (modeles Fukui)")
    print("=" * 70)
    records = load_dataset()
    print(f"\n{len(records)} conformeres apres filtres")

    summary = run_per_solvent(records)

    # Heatmap
    plot_summary_heatmap(summary)
    # Per-solvent parity for 15 catechols
    plot_per_solvent_15cat(records, 'real_ox', 'lvc2_ox', 'fminus_maxO',
                           'ox', 's42p_persolvent_ox_15cat.png')
    plot_per_solvent_15cat(records, 'real_red', 'lvc2_red', 'fplus_maxO',
                           'red', 's42p_persolvent_red_15cat.png')


if __name__ == '__main__':
    main()
