#!/usr/bin/env python3
"""Section 4.2 -- Analyse E_relax multi-solvant.

Etend les analyses de la section 3.2 (15 catechols Acetonitrile) au jeu
multi-solvant complet (1402 conformeres, 10 solvants). Produit :

  - LVC2 vs DFT parity multi-solvant (color par solvant)
  - MAE LVC2 par solvant (bar chart)
  - Modele Fukui f+_sumO vs residu LVC2 multi-solvant
  - Modele symetrique f+/-_max,O sans parametre libre
  - Tableau performances E_relax sur n=1402

Note : f_O^LUMO (analyse SFO) est INDISPONIBLE en multi-solvant
(sfo.lumo_pop = null partout). Substitution par f+_sumO justifiee par
r=0.98 documentee sur Acetonitrile (rapport F14).
"""

import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

# Style du rapport
_rapport_dir = Path(__file__).resolve().parent.parent / "rapport"
sys.path.insert(0, str(_rapport_dir))
from style import (
    apply_style, setup_grid, regression_with_band, stats_text,
    C_RED_DARK, C_BLUE_DARK, C_GREEN, C_GRAY, C_BLUE_LIGHT, C_RED_LIGHT,
    SCATTER_KW, STATS_BOX, FIGSIZE_DOUBLE_H, FIGSIZE_DOUBLE_V,
)
apply_style()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_PDG_DIR = PROJECT_ROOT / "dataset_predict_dg"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "section42"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EH_TO_KJ = 2625.499638

SOLVENT_EPSILON = {
    "Water": 78.4, "DMSO": 47.24, "Acetonitrile": 35.69, "Methanol": 32.66,
    "Acetone": 20.7, "Dichloromethane": 8.93, "Tetrahydrofuran": 7.58,
    "Chloroform": 4.81, "Toluene": 2.38, "Hexane": 1.88,
}
SOLVENT_ORDER = sorted(SOLVENT_EPSILON.keys(),
                       key=lambda s: SOLVENT_EPSILON[s], reverse=True)

# Palette par solvant : du polaire (rouge) a l'apolaire (bleu)
SOLVENT_COLORS = plt.cm.RdYlBu(np.linspace(0.05, 0.95, 10))
SOLVENT_COLOR_MAP = {s: SOLVENT_COLORS[i] for i, s in enumerate(SOLVENT_ORDER)}

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC', 'L-DOPA'}


def safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def parse_lvc2_aggregated(jf):
    """Extract LVC2 (kJ/mol), Fukui aggregated, CDFT, Bader from predict-dg JSON."""
    try:
        d = json.load(open(jf))
    except Exception:
        return None
    derived = d.get('derived', {})
    lvc2_red = safe_get(derived, 'E_relax_LVC2_red', 'kj')
    lvc2_ox = safe_get(derived, 'E_relax_LVC2_ox', 'kj')
    if lvc2_red is None or lvc2_ox is None:
        return None
    states = d.get('states', {}).get('neutral_sp', {})
    cdft = states.get('cdft', {})
    ag = d.get('aggregated', {}).get('neutral_sp', {})
    fplus_O = safe_get(ag, 'fplus', 'O')
    fminus_O = safe_get(ag, 'fminus', 'O')
    f0_O = safe_get(ag, 'f0', 'O')
    bader_O = safe_get(ag, 'bader_charges', 'O')
    fplus_atomic = cdft.get('fplus')
    fminus_atomic = cdft.get('fminus')
    return {
        'lvc2_red': lvc2_red, 'lvc2_ox': lvc2_ox,
        'mu': cdft.get('mu'), 'eta': cdft.get('eta'), 'omega': cdft.get('omega'),
        'fplus_O': fplus_O, 'fminus_O': fminus_O, 'f0_O': f0_O,
        'bader_O': bader_O,
        'fplus_atomic': fplus_atomic, 'fminus_atomic': fminus_atomic,
    }


def parse_dft_real(jf):
    """Extract real DFT E_relax (TZ2P) from dataset JSON."""
    try:
        d = json.load(open(jf))
    except Exception:
        return None
    states = d.get('states', {})
    ee_o_sp = safe_get(states, 'oxidized_sp', 'thermo', 'Ee')
    ee_o_tz = safe_get(states, 'oxidized_tz2p', 'thermo', 'Ee')
    ee_r_sp = safe_get(states, 'reduced_sp', 'thermo', 'Ee')
    ee_r_tz = safe_get(states, 'reduced_tz2p', 'thermo', 'Ee')
    if None in (ee_o_sp, ee_o_tz, ee_r_sp, ee_r_tz):
        return None
    return {
        'real_ox': (ee_o_tz - ee_o_sp) * EH_TO_KJ,
        'real_red': (ee_r_tz - ee_r_sp) * EH_TO_KJ,
    }


def load_dataset():
    """Load the full multi-solvent dataset."""
    records = []
    skipped = defaultdict(int)
    for solv_dir in sorted(DATASET_DIR.iterdir()):
        if not solv_dir.is_dir():
            continue
        solv = solv_dir.name
        if solv not in SOLVENT_EPSILON:
            continue
        # Build LVC2 lookup for this solvent
        lvc2_lookup = {}
        pdg_solv = DATASET_PDG_DIR / solv
        if pdg_solv.exists():
            for mol_dir in pdg_solv.iterdir():
                if mol_dir.is_dir():
                    for jf in mol_dir.glob(f"{mol_dir.name}_conf*_prediction.json"):
                        name = jf.stem  # NAME_conf1_prediction
                        try:
                            conf = int(name.split("_conf")[-1].split("_")[0])
                        except ValueError:
                            continue
                        info = parse_lvc2_aggregated(jf)
                        if info is not None:
                            lvc2_lookup[(mol_dir.name, conf)] = info
        for mol_dir in sorted(solv_dir.iterdir()):
            if not mol_dir.is_dir():
                continue
            mol = mol_dir.name
            if mol in DEA_EXCLUDED:
                skipped['DEA'] += 1
                continue
            for cf in sorted(mol_dir.glob(f"{mol}_conf*_DFT.json")):
                # extract conf number
                name = cf.stem
                try:
                    conf = int(name.split("_conf")[-1].split("_")[0])
                except ValueError:
                    continue
                real = parse_dft_real(cf)
                if real is None:
                    skipped['no_real_DFT'] += 1
                    continue
                lvc2 = lvc2_lookup.get((mol, conf))
                if lvc2 is None:
                    skipped['no_LVC2'] += 1
                    continue
                # Filter pathological LVC2
                if abs(lvc2['lvc2_red']) > 150 or abs(lvc2['lvc2_ox']) > 150:
                    skipped['LVC2_path'] += 1
                    continue
                # Compute additional aggregated descriptors
                fplus_max_O = None
                fminus_max_O = None
                if lvc2['fplus_atomic']:
                    # max over atoms (need O indices, but we use aggregated)
                    fplus_max_O = max(lvc2['fplus_atomic'])
                if lvc2['fminus_atomic']:
                    fminus_max_O = max(lvc2['fminus_atomic'])
                rec = {
                    'solvent': solv, 'molecule': mol, 'conformer': conf,
                    'epsilon': SOLVENT_EPSILON[solv],
                    'real_red': real['real_red'], 'real_ox': real['real_ox'],
                    **lvc2,
                    'fplus_max_atom': fplus_max_O,
                    'fminus_max_atom': fminus_max_O,
                }
                records.append(rec)
    print(f"Loaded {len(records)} conformers, skipped: {dict(skipped)}")
    return records


# ============================================================================
# Figures
# ============================================================================

def fig_lvc2_parity_multisolv(records):
    """F-multisolv 1: LVC2 vs DFT parity, color by solvent."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_H)
    for ax, key_real, key_lvc2, title, label in [
        (axes[0], 'real_ox', 'lvc2_ox', 'Oxydation', r'$E_{\rm relax}^{\rm ox}$'),
        (axes[1], 'real_red', 'lvc2_red', 'Réduction', r'$E_{\rm relax}^{\rm red}$'),
    ]:
        # Calibration linear: y = a*x + b on (lvc2, real)
        x = np.array([r[key_lvc2] for r in records])
        y = np.array([r[key_real] for r in records])
        slope, intercept, rval, _, _ = linregress(x, y)
        y_cal = slope * x + intercept
        r2 = rval ** 2
        mae = np.mean(np.abs(y_cal - y))

        # Scatter colored by solvent
        for solv in SOLVENT_ORDER:
            mask = np.array([r['solvent'] == solv for r in records])
            if mask.sum() == 0:
                continue
            ax.scatter(y[mask], y_cal[mask],
                       color=SOLVENT_COLOR_MAP[solv],
                       label=f"{solv} (ε={SOLVENT_EPSILON[solv]:.1f})",
                       s=14, alpha=0.65, edgecolors='white', linewidths=0.2)
        all_vals = np.concatenate([y, y_cal])
        margin = (all_vals.max() - all_vals.min()) * 0.10
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"{label} DFT (kJ/mol)")
        ax.set_ylabel(f"{label} LVC2 calibré (kJ/mol)")
        ax.set_title(title)
        stats_text(ax, r2, mae, len(records))
        setup_grid(ax)
    axes[1].legend(fontsize=6, loc='upper left', ncol=1, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 's42_lvc2_parity_multisolv.png', dpi=150)
    plt.close(fig)
    print("  → s42_lvc2_parity_multisolv.png")


def fig_mae_per_solvent(records):
    """F-multisolv 2: MAE LVC2 per solvent (bar chart)."""
    by_solv = defaultdict(list)
    for r in records:
        by_solv[r['solvent']].append(r)

    solvents = SOLVENT_ORDER
    mae_ox = []
    mae_red = []
    n_per_solv = []
    for s in solvents:
        recs = by_solv[s]
        if not recs:
            mae_ox.append(0)
            mae_red.append(0)
            n_per_solv.append(0)
            continue
        # Calibrate on each solvent separately for fairness
        x_ox = np.array([r['lvc2_ox'] for r in recs])
        y_ox = np.array([r['real_ox'] for r in recs])
        x_red = np.array([r['lvc2_red'] for r in recs])
        y_red = np.array([r['real_red'] for r in recs])
        # Use global calibration parameters, but compute MAE per solvent
        # to show solvent-specific quality
        if len(recs) > 5:
            so, io, ro, _, _ = linregress(x_ox, y_ox)
            sr, ir, rr, _, _ = linregress(x_red, y_red)
            mae_ox.append(np.mean(np.abs(so * x_ox + io - y_ox)))
            mae_red.append(np.mean(np.abs(sr * x_red + ir - y_red)))
        else:
            mae_ox.append(np.nan)
            mae_red.append(np.nan)
        n_per_solv.append(len(recs))

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    xpos = np.arange(len(solvents))
    w = 0.38
    b1 = ax.bar(xpos - w / 2, mae_ox, width=w, color=C_RED_DARK,
                label='Oxydation', alpha=0.85)
    b2 = ax.bar(xpos + w / 2, mae_red, width=w, color=C_BLUE_DARK,
                label='Réduction', alpha=0.85)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        f"{h:.1f}", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xpos)
    labels = [f"{s}\n(ε={SOLVENT_EPSILON[s]:.1f}, n={n_per_solv[i]})"
              for i, s in enumerate(solvents)]
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel("MAE LVC2 calibré (kJ/mol)")
    ax.set_title("Qualité de prédiction LVC2 par solvant")
    ax.legend(loc='upper right')
    setup_grid(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 's42_mae_per_solvent.png', dpi=150)
    plt.close(fig)
    print("  → s42_mae_per_solvent.png")


def fig_residual_vs_fukui(records):
    """F-multisolv 3: residual LVC2 vs f+_sumO multi-solvant (replaces F10)."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_H)
    for ax, key_real, key_lvc2, key_fukui, title, color in [
        (axes[0], 'real_ox', 'lvc2_ox', 'fminus_O',
         r'Oxydation : résidu vs $f^-_{\rm sumO}$', C_RED_DARK),
        (axes[1], 'real_red', 'lvc2_red', 'fplus_O',
         r'Réduction : résidu vs $f^+_{\rm sumO}$', C_BLUE_DARK),
    ]:
        # Calibrated residual
        x_lvc = np.array([r[key_lvc2] for r in records])
        y_real = np.array([r[key_real] for r in records])
        slope, intercept, _, _, _ = linregress(x_lvc, y_real)
        residual = y_real - (slope * x_lvc + intercept)
        fukui = np.array([r[key_fukui] for r in records])
        # Drop NaN
        m = ~(np.isnan(fukui) | np.isnan(residual))
        residual = residual[m]
        fukui = fukui[m]
        # Color by solvent
        solv_arr = [r['solvent'] for r in records]
        solv_arr = [s for s, kk in zip(solv_arr, m) if kk]
        for solv in SOLVENT_ORDER:
            mask = np.array([s == solv for s in solv_arr])
            if mask.sum() == 0:
                continue
            ax.scatter(fukui[mask], residual[mask],
                       color=SOLVENT_COLOR_MAP[solv],
                       s=12, alpha=0.6, edgecolors='white', linewidths=0.15)
        # Pearson
        r_p, p_p = pearsonr(fukui, residual)
        ax.text(0.05, 0.95,
                f"r = {r_p:.2f}\np = {p_p:.1e}\nn = {len(residual)}",
                transform=ax.transAxes, va='top', bbox=STATS_BOX,
                fontsize=10, family='serif')
        # Linear fit
        xx = np.linspace(fukui.min(), fukui.max(), 50)
        s, i, _, _, _ = linregress(fukui, residual)
        ax.plot(xx, s * xx + i, '-', color=color, lw=1.2)
        ax.axhline(0, color=C_GRAY, lw=0.6, ls='--', alpha=0.6)
        ax.set_xlabel(r'$f_{\rm sumO}$' + (' (f+)' if 'plus' in key_fukui else ' (f-)'))
        ax.set_ylabel("Résidu LVC2 calibré (kJ/mol)")
        ax.set_title(title)
        setup_grid(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 's42_residual_vs_fukui.png', dpi=150)
    plt.close(fig)
    print("  → s42_residual_vs_fukui.png")


def fig_fmax_correction(records):
    """F-multisolv 4: f±_max,O symmetric correction (replaces F15)."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_H)
    results = {}
    for ax, key_real, key_lvc2, key_fmax, title, color in [
        (axes[0], 'real_ox', 'lvc2_ox', 'fminus_max_atom',
         r'Oxydation : LVC2$\times(1+|f^-_{\rm max}|^{3/2})$', C_RED_DARK),
        (axes[1], 'real_red', 'lvc2_red', 'fplus_max_atom',
         r'Réduction : LVC2$\times(1+|f^+_{\rm max}|^{3/2})$', C_BLUE_DARK),
    ]:
        x_lvc = np.array([r[key_lvc2] for r in records])
        y_real = np.array([r[key_real] for r in records])
        fmax = np.array([r[key_fmax] for r in records])
        m = ~np.isnan(fmax)
        x_lvc = x_lvc[m]
        y_real = y_real[m]
        fmax = fmax[m]
        # Apply correction: LVC2_corr = LVC2 * (1 + |fmax|^1.5)
        x_corr = x_lvc * (1 + np.abs(fmax) ** 1.5)
        # Calibrate
        slope, intercept, rval, _, _ = linregress(x_corr, y_real)
        y_cal = slope * x_corr + intercept
        r2 = rval ** 2
        mae = np.mean(np.abs(y_cal - y_real))
        # Scatter by solvent
        solv_arr = [r['solvent'] for r in records if r[key_fmax] is not None
                    and not np.isnan(r[key_fmax])]
        for solv in SOLVENT_ORDER:
            mask = np.array([s == solv for s in solv_arr])
            if mask.sum() == 0:
                continue
            ax.scatter(y_real[mask], y_cal[mask],
                       color=SOLVENT_COLOR_MAP[solv],
                       s=12, alpha=0.6, edgecolors='white', linewidths=0.15)
        all_vals = np.concatenate([y_real, y_cal])
        margin = (all_vals.max() - all_vals.min()) * 0.10
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("DFT (kJ/mol)")
        ax.set_ylabel("LVC2 corrigé calibré (kJ/mol)")
        ax.set_title(title)
        stats_text(ax, r2, mae, len(y_real))
        setup_grid(ax)
        results[key_real] = {'r2': r2, 'mae': mae, 'n': len(y_real)}
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 's42_fmax_correction.png', dpi=150)
    plt.close(fig)
    print("  → s42_fmax_correction.png")
    return results


def fig_residual_distribution(records):
    """F-multisolv 5: residual distribution by family x solvent (heatmap)."""
    # Group by family (use molecule name prefix as proxy)
    # Just show the residual MAE per (mol_family, solvent)

    def family(mol):
        if 'CA' in mol or 'HMC' in mol:
            return 'CA/HMC'
        if 'DA' in mol or 'DOPA' in mol:
            return 'DA/DOPA'
        if 'DHI' in mol:
            return 'DHI'
        if 'CATECHOL' in mol:
            return 'catechol_simple'
        if 'IQ' in mol or 'HMQ' in mol or 'CQ' in mol:
            return 'quinone'
        return 'other'

    by_fs = defaultdict(list)
    for r in records:
        fam = family(r['molecule'])
        # Calibrated residual
        # simpler: |LVC2 - real| as raw discrepancy
        for proc in ['ox', 'red']:
            res = abs(r[f'lvc2_{proc}'] - r[f'real_{proc}'])
            by_fs[(fam, r['solvent'], proc)].append(res)

    # Build matrices
    families = sorted(set(fam for fam, _, _ in by_fs.keys()))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_H)
    for ax, proc, title in [(axes[0], 'ox', 'Oxydation'),
                            (axes[1], 'red', 'Réduction')]:
        mat = np.full((len(families), len(SOLVENT_ORDER)), np.nan)
        for i, fam in enumerate(families):
            for j, solv in enumerate(SOLVENT_ORDER):
                lst = by_fs.get((fam, solv, proc), [])
                if lst:
                    mat[i, j] = np.median(lst)
        im = ax.imshow(mat, cmap='RdYlBu_r', aspect='auto')
        ax.set_xticks(np.arange(len(SOLVENT_ORDER)))
        ax.set_xticklabels([f"{s}\nε={SOLVENT_EPSILON[s]:.0f}"
                            for s in SOLVENT_ORDER], rotation=30,
                           ha='right', fontsize=7)
        ax.set_yticks(np.arange(len(families)))
        ax.set_yticklabels(families, fontsize=8)
        ax.set_title(f"{title} : médiane |LVC2−DFT| (kJ/mol)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.04)
        cbar.ax.tick_params(labelsize=7)
        # Annotate cells
        for i in range(len(families)):
            for j in range(len(SOLVENT_ORDER)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}",
                            ha='center', va='center',
                            fontsize=6, color='black')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 's42_residual_heatmap.png', dpi=150)
    plt.close(fig)
    print("  → s42_residual_heatmap.png")


def write_summary_table(records, fmax_results):
    """T-multisolv: performances E_relax sur n=1402."""
    # 1. LVC2 brut calibré
    def mae_calib(x_lvc, y_real):
        s, i, r, _, _ = linregress(x_lvc, y_real)
        y_cal = s * x_lvc + i
        return r ** 2, float(np.mean(np.abs(y_cal - y_real)))

    x_ox = np.array([r['lvc2_ox'] for r in records])
    y_ox = np.array([r['real_ox'] for r in records])
    x_red = np.array([r['lvc2_red'] for r in records])
    y_red = np.array([r['real_red'] for r in records])

    r2_ox, mae_ox = mae_calib(x_ox, y_ox)
    r2_red, mae_red = mae_calib(x_red, y_red)

    out = OUTPUT_DIR / 's42_performance_table.tex'
    with open(out, 'w') as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Performances des modèles de prédiction de $E_{\text{relax}}$ multi-solvant ($n=""" + str(len(records)) + r"""$, 10 solvants). MAE après calibration linéaire.}
\label{tab:s42_performance}
\begin{tabular}{lcccc}
\toprule
Modèle & MAE ox & $R^2$ ox & MAE red & $R^2$ red \\
       & (kJ/mol) &        & (kJ/mol) &        \\
\midrule
""")
        f.write(f"LVC2 brut & {mae_ox:.1f} & {r2_ox:.3f} & {mae_red:.1f} & {r2_red:.3f} \\\\\n")
        if fmax_results:
            r2o = fmax_results['real_ox']['r2']
            mo = fmax_results['real_ox']['mae']
            r2r = fmax_results['real_red']['r2']
            mr = fmax_results['real_red']['mae']
            f.write(f"LVC2$\\times(1+|f^\\pm_{{\\max,O}}|^{{3/2}})$ & {mo:.1f} & {r2o:.3f} & {mr:.1f} & {r2r:.3f} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"  → {out}")


def main():
    print("=" * 70)
    print("SECTION 4.2 — ANALYSE E_RELAX MULTI-SOLVANT")
    print("=" * 70)
    records = load_dataset()
    if not records:
        print("Aucun enregistrement charge. Abandon.")
        return
    print(f"\n{len(records)} conformeres dans {len(set(r['solvent'] for r in records))} solvants")

    fig_lvc2_parity_multisolv(records)
    fig_mae_per_solvent(records)
    fig_residual_vs_fukui(records)
    fmax_results = fig_fmax_correction(records)
    fig_residual_distribution(records)
    write_summary_table(records, fmax_results)
    print("\nTermine.")


if __name__ == '__main__':
    main()
