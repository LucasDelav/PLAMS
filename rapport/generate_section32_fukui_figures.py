#!/usr/bin/env python3
"""Generate Fukui figures for section 3.2.3.

All panels are individual files at FIGSIZE_DOUBLE_V,
assembled by LaTeX \\subfigure. Single color per process.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).parent))
from style import (
    apply_style, setup_grid, regression_with_band,
    SCATTER_KW, ANNOT_KW, STATS_BOX,
    C_RED_DARK, C_BLUE_DARK, C_GRAY,
    FIGSIZE_DOUBLE_V,
)
import matplotlib.pyplot as plt

apply_style()

OUTPUT_DIR = Path(__file__).parent / 'figures'
DATA_DIR = Path(__file__).parent / 'data'
PREDICT_DIR = Path(__file__).parent.parent / 'predict_dg_results'

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}


def load_data():
    import csv
    lvc_rows = list(csv.DictReader(open(DATA_DIR / 'lvc_summary.csv')))
    lvc_data = {}
    for r in lvc_rows:
        mol = r['molecule_name']
        if mol in DEA_EXCLUDED:
            continue
        try:
            lvc_data[mol] = {
                'real_red': float(r['real_red']), 'lvc2_red': float(r['lvc2_red']),
                'real_ox': float(r.get('real_ox', 'nan')),
                'lvc2_ox': float(r.get('lvc2_ox', 'nan')),
            }
        except (ValueError, TypeError):
            continue

    # Read summed Fukui + all per-atom dual descriptors for min/max
    needed_exact = {
        'molecule', 'conformer', 'weight',
        'fplus_sumO_n', 'fminus_sumO_n', 'dual_sumO_n',
    }
    df = pd.read_csv(PREDICT_DIR / 'predict_dg_conformers.csv',
                     usecols=lambda c: c in needed_exact or
                     (c.startswith('dual_') and c.endswith('_n') and 'sum' not in c) or
                     (c.startswith(('fplus_O', 'fminus_O')) and c.endswith('_n')))

    fukui_cols = [c for c in df.columns if c not in ('molecule', 'conformer', 'weight')]
    # Identify per-atom dual columns (exclude sum columns)
    dual_atom_cols = [c for c in df.columns
                      if c.startswith('dual_') and c.endswith('_n') and 'sum' not in c]

    grouped = {}
    for mol, grp in df.groupby('molecule'):
        if mol in DEA_EXCLUDED:
            continue
        w = grp['weight'].values.astype(float)
        w = w / w.sum()
        avg = {}
        for col in fukui_cols:
            vals = pd.to_numeric(grp[col], errors='coerce').values
            if np.all(np.isfinite(vals)):
                avg[col] = float(np.sum(w * vals))
            else:
                avg[col] = np.nan

        # Δf_min and Δf_max per conformer, then Boltzmann average
        mol_dual_cols = [c for c in dual_atom_cols if c in grp.columns]
        if mol_dual_cols:
            dual_df = grp[mol_dual_cols].apply(pd.to_numeric, errors='coerce')
            dual_min_per_conf = dual_df.min(axis=1).values
            dual_max_per_conf = dual_df.max(axis=1).values
            if np.all(np.isfinite(dual_min_per_conf)):
                avg['dual_min'] = float(np.sum(w * dual_min_per_conf))
            else:
                avg['dual_min'] = np.nan
            if np.all(np.isfinite(dual_max_per_conf)):
                avg['dual_max'] = float(np.sum(w * dual_max_per_conf))
            else:
                avg['dual_max'] = np.nan

        # Max f+ and f- on O atoms
        fp_o = [c for c in grp.columns if c.startswith('fplus_O') and c.endswith('_n')]
        fm_o = [c for c in grp.columns if c.startswith('fminus_O') and c.endswith('_n')]
        if fp_o:
            avg['fplus_maxO'] = float(np.sum(w * grp[fp_o].apply(pd.to_numeric, errors='coerce').max(axis=1).values))
        if fm_o:
            avg['fminus_maxO'] = float(np.sum(w * grp[fm_o].apply(pd.to_numeric, errors='coerce').max(axis=1).values))

        grouped[mol] = avg

    common_mols = sorted(set(lvc_data.keys()) & set(grouped.keys()))
    result = {
        'molecules': common_mols,
        'residual_red': np.array([lvc_data[m]['real_red'] - lvc_data[m]['lvc2_red'] for m in common_mols]),
        'residual_ox': np.array([lvc_data[m]['real_ox'] - lvc_data[m]['lvc2_ox'] for m in common_mols]),
    }
    for col in fukui_cols:
        result[col] = np.array([grouped[m].get(col, np.nan) for m in common_mols])
    result['dual_min'] = np.array([grouped[m].get('dual_min', np.nan) for m in common_mols])
    result['dual_max'] = np.array([grouped[m].get('dual_max', np.nan) for m in common_mols])
    result['fplus_maxO'] = np.array([grouped[m].get('fplus_maxO', np.nan) for m in common_mols])
    result['fminus_maxO'] = np.array([grouped[m].get('fminus_maxO', np.nan) for m in common_mols])
    return result


def _residual_single(x, y, names, color, xlabel, ylabel, fname):
    """Generate a single residual scatter panel at FIGSIZE_DOUBLE_V."""
    ok = np.isfinite(x) & np.isfinite(y)
    x_ok, y_ok = x[ok], y[ok]
    names_ok = [n for n, k in zip(names, ok) if k]

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    ax.scatter(x_ok, y_ok, c=color, **SCATTER_KW)
    for i, name in enumerate(names_ok):
        ax.annotate(name, (x_ok[i], y_ok[i]), **ANNOT_KW)

    if len(x_ok) >= 3:
        regression_with_band(ax, x_ok, y_ok, color=color)
        _, _, r, p, _ = linregress(x_ok, y_ok)
        ax.text(0.05, 0.95,
                f"$r = {r:.2f}$\n$p = {p:.3f}$\n$n = {len(x_ok)}$",
                transform=ax.transAxes, va='top',
                fontsize=14, bbox=STATS_BOX, family='serif')

    ax.axhline(0, color=C_GRAY, ls='--', lw=0.6, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    setup_grid(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname)
    plt.close()
    print(f"  [+] {fname}")


def plot_fukui_panels(data):
    mols = data['molecules']

    # Reduction: f+ and dual
    _residual_single(
        data['fplus_sumO_n'], data['residual_red'], mols, C_BLUE_DARK,
        '$f^+_{\\mathrm{sumO}}$',
        'Résidu réduction (kJ/mol)',
        'fukui_fplus_sumO_red.png')

    _residual_single(
        data['dual_sumO_n'], data['residual_red'], mols, C_BLUE_DARK,
        '$\\Delta f_{\\mathrm{sumO}}$',
        'Résidu réduction (kJ/mol)',
        'fukui_dual_sumO_red.png')

    # Reduction: dual_max (most nucleophilic site)
    _residual_single(
        data['dual_max'], data['residual_red'], mols, C_BLUE_DARK,
        '$\\Delta f_{\\max}$',
        'Résidu réduction (kJ/mol)',
        'fukui_dual_max_red.png')

    # Oxidation: f- and dual
    _residual_single(
        data['fminus_sumO_n'], data['residual_ox'], mols, C_RED_DARK,
        '$f^-_{\\mathrm{sumO}}$',
        'Résidu oxydation (kJ/mol)',
        'fukui_fminus_sumO_ox.png')

    _residual_single(
        data['dual_sumO_n'], data['residual_ox'], mols, C_RED_DARK,
        '$\\Delta f_{\\mathrm{sumO}}$',
        'Résidu oxydation (kJ/mol)',
        'fukui_dual_sumO_ox.png')

    # Oxidation: dual_min (most electrophilic site)
    _residual_single(
        data['dual_min'], data['residual_ox'], mols, C_RED_DARK,
        '$\\Delta f_{\\min}$',
        'Résidu oxydation (kJ/mol)',
        'fukui_dual_min_ox.png')


def plot_fukui_vs_lumo(data):
    import csv
    desc_rows = list(csv.DictReader(open(DATA_DIR / 'descriptors_per_molecule.csv')))
    lumo_O = {}
    for r in desc_rows:
        try:
            lumo_O[r['molecule']] = float(r['LUMO_O_frac'])
        except (ValueError, TypeError, KeyError):
            continue

    mols = data['molecules']
    fplus = data['fplus_sumO_n']
    fO = np.array([lumo_O.get(m, np.nan) for m in mols])
    ok = np.isfinite(fplus) & np.isfinite(fO)
    x, y = fO[ok], fplus[ok]
    names = [m for m, k in zip(mols, ok) if k]

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    ax.scatter(x, y, c=C_BLUE_DARK, **SCATTER_KW)
    for i, name in enumerate(names):
        ax.annotate(name, (x[i], y[i]), **ANNOT_KW)

    regression_with_band(ax, x, y, color=C_BLUE_DARK)

    _, _, r, p, _ = linregress(x, y)
    ax.text(0.05, 0.95,
            f"$r = {r:.2f}$\n$p = {p:.1e}$\n$n = {len(x)}$",
            transform=ax.transAxes, va='top',
            fontsize=14, bbox=STATS_BOX, family='serif')

    ax.set_xlabel('$f_{\\mathrm{O}}^{\\mathrm{LUMO}}$ (analyse SFO)')
    ax.set_ylabel('$f^+_{\\mathrm{sumO}}$ (Fukui condensé)')
    setup_grid(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fukui_vs_lumo_comparison.png')
    plt.close()
    print(f"  [+] fukui_vs_lumo_comparison.png  (r={r:.3f})")


def plot_fmaxO_parity(data):
    """Generate parity plots for the symmetric f_maxO correction model.

    4 individual panels: brut/corrigé × ox/red, assembled by LaTeX.
    Non-parametric: LVC2*(1+|f_maxO|^1.5)
    Parametric: LVC2*(1+a*|f_maxO|^2), a fitted per process.
    """
    from style import stats_text, regression_with_band, SCATTER_KW, ANNOT_KW, FIGSIZE_DOUBLE_V, C_GRAY
    from scipy.optimize import minimize_scalar

    molecules, is_dea, real_red, real_ox, lvc2_red, lvc2_ox = _load_lvc_full()
    fmaxO_plus, fmaxO_minus = _load_fmaxO(data)

    configs = [
        # (fname, real, lvc2, fmaxO, color, power, fit_a)
        ('fmaxO_noparam_ox', real_ox, lvc2_ox, fmaxO_minus, C_RED_DARK, 1.5, False),
        ('fmaxO_noparam_red', real_red, lvc2_red, fmaxO_plus, C_BLUE_DARK, 1.5, False),
        ('fmaxO_param_ox', real_ox, lvc2_ox, fmaxO_minus, C_RED_DARK, 2.0, True),
        ('fmaxO_param_red', real_red, lvc2_red, fmaxO_plus, C_BLUE_DARK, 2.0, True),
    ]

    for fname, real, lvc2, fmax, color, power, fit_a in configs:
        valid = np.isfinite(real) & np.isfinite(lvc2) & np.isfinite(fmax)
        xs = real[valid]
        d = fmax[valid]
        lvc = lvc2[valid]
        names = [m for m, k in zip(data['molecules_full'], valid) if k]

        if fit_a:
            def obj(a):
                corr = lvc * (1 + a * np.abs(d)**power)
                s, i, r, _, _ = linregress(corr, xs)
                cal = s * corr + i
                return np.mean(np.abs(xs - cal))
            res = minimize_scalar(obj, bounds=(-5, 5), method='bounded')
            predicted = lvc * (1 + res.x * np.abs(d)**power)
            a_val = res.x
        else:
            predicted = lvc * (1 + np.abs(d)**power)
            a_val = None

        # Calibrated parity
        s_cal, i_cal, r_val, _, _ = linregress(predicted, xs)
        y_cal = s_cal * predicted + i_cal
        r2 = r_val**2
        mae = np.mean(np.abs(xs - y_cal))

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        ax.scatter(xs, y_cal, c=color, **SCATTER_KW)
        for i, name in enumerate(names):
            ax.annotate(name, (xs[i], y_cal[i]), **ANNOT_KW)
        regression_with_band(ax, xs, y_cal, color=color)

        all_vals = np.concatenate([xs, y_cal])
        margin = (all_vals.max() - all_vals.min()) * 0.12
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        stats_text(ax, r2, mae, len(xs))

        ax.set_xlabel('$E_{\\mathrm{relax}}^{\\mathrm{DFT}}$ (kJ/mol)')
        ax.set_ylabel('$E_{\\mathrm{relax}}$ calibré (kJ/mol)')
        setup_grid(ax)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{fname}.png')
        plt.close()

        param_str = f' a={a_val:.2f}' if a_val is not None else ''
        print(f"  [+] {fname}.png  (R2={r2:.3f}, MAE={mae:.1f}{param_str})")


def _load_lvc_full():
    """Load LVC data for all 15 stable molecules."""
    import csv
    lvc_rows = list(csv.DictReader(open(DATA_DIR / 'lvc_summary.csv')))
    molecules, real_red, real_ox, lvc2_red, lvc2_ox = [], [], [], [], []
    for r in lvc_rows:
        mol = r['molecule_name']
        if mol in DEA_EXCLUDED:
            continue
        molecules.append(mol)
        real_red.append(float(r.get('real_red', 'nan')))
        real_ox.append(float(r.get('real_ox', 'nan')))
        lvc2_red.append(float(r.get('lvc2_red', 'nan')))
        lvc2_ox.append(float(r.get('lvc2_ox', 'nan')))
    is_dea = np.zeros(len(molecules), dtype=bool)
    return (molecules, is_dea,
            np.array(real_red), np.array(real_ox),
            np.array(lvc2_red), np.array(lvc2_ox))


def _load_fmaxO(data):
    """Get fplus_maxO and fminus_maxO aligned with the full 15-mol LVC list."""
    molecules_full, *_ = _load_lvc_full()
    predict_mols = data['molecules']

    fplus = np.full(len(molecules_full), np.nan)
    fminus = np.full(len(molecules_full), np.nan)
    predict_idx = {m: i for i, m in enumerate(predict_mols)}
    for i, m in enumerate(molecules_full):
        if m in predict_idx:
            j = predict_idx[m]
            fplus[i] = data.get('fplus_maxO', np.full(len(predict_mols), np.nan))[j]
            fminus[i] = data.get('fminus_maxO', np.full(len(predict_mols), np.nan))[j]
    return fplus, fminus


if __name__ == '__main__':
    print("=== Generating section 3.2.3 Fukui figures ===")
    data = load_data()
    print(f"  {len(data['molecules'])} molecules")

    # Store full molecule list for parity plots
    mols_full, *_ = _load_lvc_full()
    data['molecules_full'] = mols_full

    plot_fukui_panels(data)
    plot_fukui_vs_lumo(data)
    plot_fmaxO_parity(data)
    print("=== Done ===")
