#!/usr/bin/env python3
"""Generate figures for section 3.2 (E_relax correlations).

All scatter/parity panels are individual files at FIGSIZE_DOUBLE_V,
assembled into multi-panel figures by LaTeX \\subfigure.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).parent))
from style import (
    apply_style, setup_grid, regression_with_band, stats_text,
    SCATTER_KW, ANNOT_KW, STATS_BOX,
    C_RED_DARK, C_BLUE_DARK, C_GRAY,
    FIGSIZE_DOUBLE_V,
)
import matplotlib.pyplot as plt

apply_style()

OUTPUT_DIR = Path(__file__).parent / 'figures'
DATA_DIR = Path(__file__).parent / 'data'

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}


def read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def _sf(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def load_lvc_data():
    rows = read_csv(DATA_DIR / 'lvc_summary.csv')
    molecules = [r['molecule_name'] for r in rows]
    is_dea = np.array([m in DEA_EXCLUDED for m in molecules])
    real_red = np.array([_sf(r.get('real_red')) for r in rows])
    real_ox = np.array([_sf(r.get('real_ox')) for r in rows])
    lvc2_red = np.array([_sf(r.get('lvc2_red')) for r in rows])
    lvc2_ox = np.array([_sf(r.get('lvc2_ox')) for r in rows])
    return molecules, is_dea, real_red, real_ox, lvc2_red, lvc2_ox


def load_descriptors():
    desc_rows = read_csv(DATA_DIR / 'descriptors_per_molecule.csv')
    return {r['molecule']: float(r['LUMO_O_frac'])
            for r in desc_rows if r.get('LUMO_O_frac', '') not in ('', 'nan', 'NaN')}


def _calibrated_parity_panel(ax, x_dft, y_raw, names, color):
    """Calibrated parity: single panel with regression + CI band."""
    slope_cal, intercept_cal, r_val, _, _ = linregress(y_raw, x_dft)
    y_cal = slope_cal * y_raw + intercept_cal
    r2 = r_val ** 2
    mae = np.mean(np.abs(y_cal - x_dft))

    ax.scatter(x_dft, y_cal, c=color, **SCATTER_KW)
    for i, mol in enumerate(names):
        ax.annotate(mol, (x_dft[i], y_cal[i]), **ANNOT_KW)

    regression_with_band(ax, x_dft, y_cal, color=color)

    all_vals = np.concatenate([x_dft, y_cal])
    margin = (all_vals.max() - all_vals.min()) * 0.12
    lims = [all_vals.min() - margin, all_vals.max() + margin]
    ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    stats_text(ax, r2, mae, len(x_dft))
    return r2, mae


def _residual_panel(ax, x, y, names, color):
    """Single residual scatter with regression + CI band, r and p."""
    ok = np.isfinite(x) & np.isfinite(y)
    x_ok, y_ok = x[ok], y[ok]
    names_ok = [n for n, k in zip(names, ok) if k]

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
    setup_grid(ax)


def _save_single(fname):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname)
    plt.close()
    print(f"  [+] {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8b,d: LVC2 calibrated parity (individual panels)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lvc2_brut_parity():
    molecules, is_dea, real_red, real_ox, lvc2_red, lvc2_ox = load_lvc_data()

    for suffix, real, lvc2, color in [
        ('oxidation', real_ox, lvc2_ox, C_RED_DARK),
        ('reduction', real_red, lvc2_red, C_BLUE_DARK),
    ]:
        valid = np.isfinite(real) & np.isfinite(lvc2) & ~is_dea
        xs, ys = real[valid], lvc2[valid]
        names = [m for m, k in zip(molecules, valid) if k]

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        r2, mae = _calibrated_parity_panel(ax, xs, ys, names, color)
        ax.set_xlabel('$E_{\\mathrm{relax}}^{\\mathrm{DFT}}$ (kJ/mol)')
        ax.set_ylabel('$E_{\\mathrm{relax}}^{\\mathrm{LVC2}}$ calibré (kJ/mol)')
        setup_grid(ax)
        _save_single(f'lvc2_brut_{suffix}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9: Grouped bars LVC2 vs real
# ═══════════════════════════════════════════════════════════════════════════════

def plot_grouped_bars():
    molecules, is_dea, real_red, real_ox, lvc2_red, lvc2_ox = load_lvc_data()

    for suffix, real, lvc2, c_real, c_lvc2 in [
        ('ox', real_ox, lvc2_ox, C_RED_DARK, '#f4a582'),
        ('red', real_red, lvc2_red, C_BLUE_DARK, '#92c5de'),
    ]:
        valid = np.isfinite(real) & np.isfinite(lvc2) & ~is_dea
        names = [m for m, k in zip(molecules, valid) if k]
        r_vals, l_vals = real[valid], lvc2[valid]
        order = np.argsort(r_vals)
        names = [names[i] for i in order]
        r_vals, l_vals = r_vals[order], l_vals[order]

        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        ax.bar(x - width/2, r_vals, width, color=c_real, alpha=0.85,
               edgecolor='white', linewidth=0.3, label='DFT')
        ax.bar(x + width/2, l_vals, width, color=c_lvc2, alpha=0.85,
               edgecolor='white', linewidth=0.3, label='LVC2')
        ax.set_ylabel('$E_{\\mathrm{relax}}$ (kJ/mol)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='gray', lw=0.6)
        ax.legend(fontsize=10, loc='lower left')
        setup_grid(ax, limit_ticks=False)
        mae = np.mean(np.abs(r_vals - l_vals))
        ax.text(0.98, 0.05, f'MAE = {mae:.1f} kJ/mol',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, bbox=STATS_BOX)
        _save_single(f'lvc_vs_real_bars_{suffix}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10: Residual vs LUMO O fraction (individual panels: ox, red)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_residual_vs_fO():
    molecules, is_dea, real_red, real_ox, lvc2_red, lvc2_ox = load_lvc_data()
    fO_map = load_descriptors()

    for suffix, real, lvc2, color in [
        ('ox', real_ox, lvc2_ox, C_RED_DARK),
        ('red', real_red, lvc2_red, C_BLUE_DARK),
    ]:
        valid = np.isfinite(real) & np.isfinite(lvc2) & ~is_dea
        residual = real - lvc2
        names = [m for m, k in zip(molecules, valid) if k]
        res = residual[valid]
        fO = np.array([fO_map.get(m, np.nan) for m in names])
        ok = np.isfinite(fO)
        names_ok = [n for n, k in zip(names, ok) if k]
        res_ok, fO_ok = res[ok], fO[ok]

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        _residual_panel(ax, fO_ok, res_ok, names_ok, color)
        ax.set_xlabel('$f_{\\mathrm{O}}^{\\mathrm{LUMO}}$')
        ax.set_ylabel('Résidu $E_{\\mathrm{relax}} - E_{\\mathrm{relax}}^{\\mathrm{LVC2}}$ (kJ/mol)')
        _save_single(f'residual_lvc2_vs_fO_{suffix}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11: LVC2 brut vs corrigé (4 individual panels)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lvc2_corrected():
    molecules, is_dea, real_red, real_ox, lvc2_red, lvc2_ox = load_lvc_data()
    fO_map = load_descriptors()
    lumo_O = np.array([fO_map.get(m, np.nan) for m in molecules])

    configs = [
        ('lvc2_brut_ox', real_ox, lvc2_ox, C_RED_DARK),
        ('lvc2_corr_ox', real_ox, lvc2_ox * (1 + lumo_O**2), C_RED_DARK),
        ('lvc2_brut_red', real_red, lvc2_red, C_BLUE_DARK),
        ('lvc2_corr_red', real_red, lvc2_red * (1 + lumo_O**2), C_BLUE_DARK),
    ]

    for fname_base, real, predicted, color in configs:
        valid = np.isfinite(real) & np.isfinite(predicted) & ~is_dea
        xs, ys = real[valid], predicted[valid]
        names = [m for m, k in zip(molecules, valid) if k]

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        _calibrated_parity_panel(ax, xs, ys, names, color)
        ax.set_xlabel('$E_{\\mathrm{relax}}^{\\mathrm{DFT}}$ (kJ/mol)')
        ax.set_ylabel('$E_{\\mathrm{relax}}$ calibré (kJ/mol)')
        setup_grid(ax)
        _save_single(f'{fname_base}.png')


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=== Generating section 3.2 figures ===")
    plot_lvc2_brut_parity()
    plot_grouped_bars()
    plot_residual_vs_fO()
    plot_lvc2_corrected()
    print("=== Done ===")
