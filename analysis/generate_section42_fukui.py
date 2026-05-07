#!/usr/bin/env python3
"""Section 4.2 -- Erelax avec descripteurs Fukui multi-solvant.

Reproduction du modele symetrique Fukui (Fig. 15-16 du rapport, sec. 3.2.3)
sur le jeu multi-solvant complet, en utilisant les descripteurs Fukui agreges
depuis dataset_predict_dg/ (au lieu de la baseline LVC2+epsilon_LUMO).

Modeles testes:
  - LVC2 brut calibre (baseline)
  - LVC2 + epsilon_LUMO (baseline rapport actuel)
  - LVC2 * (1 + |f+/-_sumO|^1.5)        sans parametre
  - LVC2 * (1 + |f+/-_max,O|^1.5)       sans parametre
  - LVC2 * (1 + a |f+/-_max,O|^2)       1 parametre par processus

Filtre EA<0 applique pour coherence avec sec. 4 (n=1402).
Stratification par famille moleculaire en bonus.
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
from scipy.stats import linregress, pearsonr

_rapport_dir = Path(__file__).resolve().parent.parent / "rapport"
sys.path.insert(0, str(_rapport_dir))
from style import (
    apply_style, setup_grid, stats_text,
    C_RED_DARK, C_BLUE_DARK, C_GREEN, C_GRAY,
    SCATTER_KW, STATS_BOX, FIGSIZE_DOUBLE_H,
)
apply_style()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_PDG_DIR = PROJECT_ROOT / "dataset_predict_dg"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "section42_fukui"
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


def safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def molecule_family(mol):
    if 'CA' in mol or 'HMC' in mol or 'CATECHOL' in mol or 'EUGENOL' in mol:
        return 'CA/HMC'
    if 'DA' in mol or 'DOPA' in mol:
        return 'DA/DOPA'
    if 'DHI' in mol:
        return 'DHI'
    if 'IQ' in mol or 'HMQ' in mol or 'CQ' in mol or 'QUINONE' in mol:
        return 'quinone'
    return 'other'


def get_oxygen_indices(xyz_str):
    """Return list of indices of O atoms in the molecule (0-indexed)."""
    lines = xyz_str.strip().split('\n')
    if len(lines) < 3:
        return []
    O_idx = []
    for i, line in enumerate(lines[2:]):  # skip count + comment
        parts = line.split()
        if parts and parts[0].strip().upper() == 'O':
            O_idx.append(i)
    return O_idx


def parse_predict_dg(jf):
    """Extract LVC2, atomic Fukui, aggregated Fukui, plus xyz to find O indices."""
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
    # Atomic Fukui restricted to O atoms
    fplus_O = fplus_at[O_idx]
    fminus_O = fminus_at[O_idx]
    # Aggregated (sum on O)
    fplus_sumO = float(fplus_O.sum())
    fminus_sumO = float(fminus_O.sum())
    # Max on O
    fplus_maxO = float(np.max(np.abs(fplus_O)))
    fminus_maxO = float(np.max(np.abs(fminus_O)))
    return {
        'lvc2_red': lvc2_red, 'lvc2_ox': lvc2_ox,
        'fplus_sumO': fplus_sumO, 'fminus_sumO': fminus_sumO,
        'fplus_maxO': fplus_maxO, 'fminus_maxO': fminus_maxO,
        'eps_LUMO': cdft.get('mu'),  # placeholder, replace below
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
        # Build LVC2 / Fukui lookup for this solvent
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
                if dft is None:
                    continue
                if dft['EA'] < 0:
                    continue  # filter EA<0 for consistency with sec.4
                info = lookup.get((mol, conf))
                if info is None:
                    continue
                # Reject pathological E_relax_DFT (failed opt)
                if abs(dft['real_ox']) > 200 or abs(dft['real_red']) > 200:
                    continue
                records.append({
                    'molecule': mol, 'conformer': conf, 'solvent': solv,
                    'epsilon': SOLVENT_EPSILON[solv],
                    'family': molecule_family(mol),
                    'real_ox': dft['real_ox'], 'real_red': dft['real_red'],
                    'lumo_eV': dft['lumo_eV'],
                    **info,
                })
    return records


# ============================================================================
# Models
# ============================================================================

def calibrated_metrics(x, y):
    """Linear calibration y_cal = s*x + i, return R^2 and MAE."""
    s, i, r, _, _ = linregress(x, y)
    y_cal = s * x + i
    return float(r ** 2), float(np.mean(np.abs(y_cal - y))), s, i


def model_fmax_no_param(x_lvc2, fmax):
    return x_lvc2 * (1 + np.abs(fmax) ** 1.5)


def model_fmax_one_param(x_lvc2, fmax, a):
    return x_lvc2 * (1 + a * fmax ** 2)


def fit_a(x_lvc2, fmax, y_real):
    """Fit single parameter a in LVC2 * (1 + a*f^2) to maximize R^2 calibrated."""
    def neg_r2(a):
        x_corr = model_fmax_one_param(x_lvc2, fmax, a)
        s, i, r, _, _ = linregress(x_corr, y_real)
        return -r ** 2
    res = minimize_scalar(neg_r2, bounds=(0.0, 20.0), method='bounded')
    return float(res.x)


# ============================================================================
# Figures
# ============================================================================

def plot_4models_parity(records, key_real, key_lvc2, key_fmax, key_fsum,
                        process, fname):
    """4-panel parity: brut, +eps_LUMO, *fmax_no_param, *fmax_with_param."""
    x_lvc = np.array([r[key_lvc2] for r in records])
    y_real = np.array([r[key_real] for r in records])
    fmax = np.array([r[key_fmax] for r in records])
    lumo = np.array([r['lumo_eV'] if r['lumo_eV'] is not None else np.nan
                     for r in records])

    # Filter NaN in lumo
    valid = ~np.isnan(lumo)
    x_lvc_v = x_lvc[valid]
    y_real_v = y_real[valid]
    fmax_v = fmax[valid]
    lumo_v = lumo[valid]

    # Model 1: LVC2 brut
    r2_a, mae_a, _, _ = calibrated_metrics(x_lvc, y_real)
    # Model 2: LVC2 + lumo (multivariate calibration)
    A = np.vstack([x_lvc_v, lumo_v, np.ones_like(x_lvc_v)]).T
    coefs, *_ = np.linalg.lstsq(A, y_real_v, rcond=None)
    pred_b = A @ coefs
    r2_b = 1.0 - np.sum((y_real_v - pred_b) ** 2) / np.sum((y_real_v - y_real_v.mean()) ** 2)
    mae_b = np.mean(np.abs(pred_b - y_real_v))
    # Model 3: LVC2 * (1 + |fmax|^1.5)
    x_c = model_fmax_no_param(x_lvc, fmax)
    r2_c, mae_c, _, _ = calibrated_metrics(x_c, y_real)
    # Model 4: LVC2 * (1 + a fmax^2) with fitted a
    a_opt = fit_a(x_lvc, fmax, y_real)
    x_d = model_fmax_one_param(x_lvc, fmax, a_opt)
    r2_d, mae_d, _, _ = calibrated_metrics(x_d, y_real)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    titles = [
        f"LVC2 brut\nR²={r2_a:.3f} MAE={mae_a:.1f}",
        f"LVC2 + ε_LUMO\nR²={r2_b:.3f} MAE={mae_b:.1f}",
        f"LVC2×(1+|f_max,O|^1.5)\nR²={r2_c:.3f} MAE={mae_c:.1f}",
        f"LVC2×(1+{a_opt:.2f}·f_max,O²)\nR²={r2_d:.3f} MAE={mae_d:.1f}",
    ]
    pred_data = [
        # brut: y_cal = s*x + i
        (x_lvc, y_real, *calibrated_metrics(x_lvc, y_real)[2:]),
        # +lumo: needs special handling
        (None, None, None, None),
        (x_c, y_real, *calibrated_metrics(x_c, y_real)[2:]),
        (x_d, y_real, *calibrated_metrics(x_d, y_real)[2:]),
    ]

    color = C_RED_DARK if process == 'ox' else C_BLUE_DARK

    for k, (ax, title) in enumerate(zip(axes, titles)):
        if k == 1:
            # +lumo: use multivariate prediction directly
            ax.scatter(y_real_v, pred_b, s=10, alpha=0.5, color=color,
                       edgecolors='white', linewidths=0.15)
            all_v = np.concatenate([y_real_v, pred_b])
        else:
            x_used, y_used, slope, intercept = pred_data[k]
            y_cal = slope * x_used + intercept
            ax.scatter(y_used, y_cal, s=10, alpha=0.5, color=color,
                       edgecolors='white', linewidths=0.15)
            all_v = np.concatenate([y_used, y_cal])
        margin = (all_v.max() - all_v.min()) * 0.10
        lims = [all_v.min() - margin, all_v.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"E_relax_{process} DFT (kJ/mol)")
        if k == 0:
            ax.set_ylabel(f"E_relax_{process} prédit (kJ/mol)")
        ax.set_title(title, fontsize=9)
        setup_grid(ax)

    fig.suptitle(
        f"$E_\\mathrm{{relax}}^\\mathrm{{{process}}}$ : 4 modèles, n={len(records)}",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → {fname}")
    return {'brut': (r2_a, mae_a), 'lumo': (r2_b, mae_b),
            'fmax15': (r2_c, mae_c), 'fmax_a': (r2_d, mae_d, a_opt)}


def plot_per_family(records, key_real, key_lvc2, key_fmax, process, fname):
    """Stratified parity: 1 panel per family, with LVC2*fmax model."""
    families = sorted(set(r['family'] for r in records))
    fig, axes = plt.subplots(1, len(families), figsize=(3.2 * len(families), 3.5),
                             squeeze=False)
    axes = axes[0]
    color = C_RED_DARK if process == 'ox' else C_BLUE_DARK
    for ax, fam in zip(axes, families):
        recs = [r for r in records if r['family'] == fam]
        if len(recs) < 5:
            ax.set_title(f"{fam} (n={len(recs)})", fontsize=9)
            ax.text(0.5, 0.5, "n trop faible", ha='center', va='center',
                    transform=ax.transAxes)
            continue
        x_lvc = np.array([r[key_lvc2] for r in recs])
        y_real = np.array([r[key_real] for r in recs])
        fmax = np.array([r[key_fmax] for r in recs])
        x_c = model_fmax_no_param(x_lvc, fmax)
        r2, mae, slope, intercept = calibrated_metrics(x_c, y_real)
        y_cal = slope * x_c + intercept
        ax.scatter(y_real, y_cal, s=12, alpha=0.55, color=color,
                   edgecolors='white', linewidths=0.2)
        all_v = np.concatenate([y_real, y_cal])
        margin = (all_v.max() - all_v.min()) * 0.10
        lims = [all_v.min() - margin, all_v.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"DFT (kJ/mol)")
        ax.set_title(f"{fam} (n={len(recs)})\nR²={r2:.3f} MAE={mae:.1f}",
                     fontsize=9)
        setup_grid(ax)
    axes[0].set_ylabel(f"LVC2×(1+|f_max,O|^1.5) calibré (kJ/mol)")
    fig.suptitle(f"$E_\\mathrm{{relax}}^\\mathrm{{{process}}}$ : stratification par famille",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → {fname}")


def main():
    print("=" * 70)
    print("SECTION 4.2 (Fukui multi-solvant) -- E_relax avec descripteurs CDFT")
    print("=" * 70)
    records = load_dataset()
    n = len(records)
    print(f"\n{n} conformeres apres filtres (DEA, EA<0, |E_relax|>200, LVC2 path)")
    if n < 50:
        print("ABORT: trop peu de records")
        return

    # By family count
    fam_counts = defaultdict(int)
    for r in records:
        fam_counts[r['family']] += 1
    print(f"  Familles: {dict(fam_counts)}")

    print("\n=== OXYDATION ===")
    res_ox = plot_4models_parity(
        records, 'real_ox', 'lvc2_ox', 'fminus_maxO', 'fminus_sumO',
        'ox', 's42f_4models_ox.png')
    print(f"  brut    : R^2={res_ox['brut'][0]:.3f}  MAE={res_ox['brut'][1]:.2f}")
    print(f"  + ε_LUMO: R^2={res_ox['lumo'][0]:.3f}  MAE={res_ox['lumo'][1]:.2f}")
    print(f"  *fmax15 : R^2={res_ox['fmax15'][0]:.3f}  MAE={res_ox['fmax15'][1]:.2f}")
    print(f"  *fmax_a : R^2={res_ox['fmax_a'][0]:.3f}  MAE={res_ox['fmax_a'][1]:.2f}  a={res_ox['fmax_a'][2]:.2f}")
    plot_per_family(records, 'real_ox', 'lvc2_ox', 'fminus_maxO',
                    'ox', 's42f_perfamily_ox.png')

    print("\n=== REDUCTION ===")
    res_red = plot_4models_parity(
        records, 'real_red', 'lvc2_red', 'fplus_maxO', 'fplus_sumO',
        'red', 's42f_4models_red.png')
    print(f"  brut    : R^2={res_red['brut'][0]:.3f}  MAE={res_red['brut'][1]:.2f}")
    print(f"  + ε_LUMO: R^2={res_red['lumo'][0]:.3f}  MAE={res_red['lumo'][1]:.2f}")
    print(f"  *fmax15 : R^2={res_red['fmax15'][0]:.3f}  MAE={res_red['fmax15'][1]:.2f}")
    print(f"  *fmax_a : R^2={res_red['fmax_a'][0]:.3f}  MAE={res_red['fmax_a'][1]:.2f}  a={res_red['fmax_a'][2]:.2f}")
    plot_per_family(records, 'real_red', 'lvc2_red', 'fplus_maxO',
                    'red', 's42f_perfamily_red.png')

    # Summary table
    summary = OUTPUT_DIR / 's42f_summary.txt'
    with open(summary, 'w') as f:
        f.write(f"Multi-solvent E_relax (n={n}, after EA>=0 + DEA + non-pathological filters)\n")
        f.write("=" * 70 + "\n\n")
        for proc, res in [('ox', res_ox), ('red', res_red)]:
            f.write(f"\n--- {proc.upper()} ---\n")
            for k, v in res.items():
                if k == 'fmax_a':
                    f.write(f"  {k:10}: R^2={v[0]:.3f}  MAE={v[1]:.2f}  a={v[2]:.2f}\n")
                else:
                    f.write(f"  {k:10}: R^2={v[0]:.3f}  MAE={v[1]:.2f}\n")
    print(f"\n  Resume: {summary}")


if __name__ == '__main__':
    main()
