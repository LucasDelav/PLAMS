#!/usr/bin/env amspython
"""
Regenerate beyond_koopmans figures (IP_3models_parity, EA_3models_parity,
residuals_koopmans_vs_complement) with R²_adj and consistent style
(regression line + 95% confidence band).
"""

import csv
import numpy as np
from pathlib import Path
from scipy.stats import linregress, t as t_dist, pearsonr
from scm.plams import KFFile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import (
    apply_style, setup_grid, get_color_map, add_parity_line,
    regression_with_band, stats_text,
    C_RED_DARK, C_BLUE_DARK, C_GRAY,
    SCATTER_KW, ANNOT_KW, STATS_BOX,
)
import matplotlib.pyplot as plt

apply_style()

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Acetonitrile_marais"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "beyond_koopmans"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}
QUINONES = {
    'CQ', 'HMQ', 'Al-CQ', 'HO-CQ', 'H2C-CQ',
    'Cl-HMQ', 'F-HMQ', 'Br-HMQ', 'I-HMQ',
    'DOPA', 'Cl-DOPA', 'F-DOPA', 'Br-DOPA', 'I-DOPA', 'L-DOPA',
    'O-IQ', 'P-IQ', 'S-IQ', 'IQ',
}
EH_TO_EV = 27.211386245988
EV_TO_KJ = 96.4853321


def load_data():
    """Load per-conformer IP/EA + extended orbital energies (kJ/mol) from RKFs."""
    records = []
    for wdir in sorted(DATA_DIR.iterdir()):
        if not wdir.is_dir() or not wdir.name.endswith('_workdir'):
            continue
        mol = wdir.name.replace('_workdir', '')
        if mol in QUINONES or mol in DEA_EXCLUDED:
            continue

        csv_path = wdir / 'redox' / 'results' / f'{mol}_redox_conformers.csv'
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                conf_num = int(row['conformer'])
                rkf_path = (wdir / 'redox' / 'calculations' / 'plams_workdir'
                            / f'{mol}_conf_{conf_num}_neutral_tz2p' / 'adf.rkf')
                if not rkf_path.exists():
                    continue
                try:
                    kf = KFFile(str(rkf_path))
                    eps = np.array(kf.read('A', 'eps_A'), dtype=float)
                    froc = np.array(kf.read('A', 'froc_A'), dtype=float)
                    n_occ = int(np.sum(froc > 0))
                    if n_occ < 2 or len(eps) - n_occ < 2:
                        continue
                    records.append({
                        'molecule': mol,
                        'conformer': conf_num,
                        'IP_kJ': float(row['IP_kJ']),
                        'EA_kJ': float(row['EA_kJ']),
                        'eH': eps[n_occ - 1] * EH_TO_EV * EV_TO_KJ,
                        'eHm1': eps[n_occ - 2] * EH_TO_EV * EV_TO_KJ,
                        'eL': eps[n_occ] * EH_TO_EV * EV_TO_KJ,
                        'eLp1': eps[n_occ + 1] * EH_TO_EV * EV_TO_KJ,
                    })
                except Exception as e:
                    print(f"  [!] {mol} conf_{conf_num}: {e}")
    return records


def r2_adj(y, pred, p):
    """Adjusted R² with p predictors (excluding intercept)."""
    n = len(y)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def regression_band_multi(ax, y_actual, y_pred, color):
    """Plot regression line + 95% CI band on a parity plot (actual vs predicted)."""
    n = len(y_actual)
    slope, intercept, _, _, _ = linregress(y_actual, y_pred)
    x_fit = np.linspace(y_actual.min() - 0.02 * np.ptp(y_actual),
                        y_actual.max() + 0.02 * np.ptp(y_actual), 200)
    y_fit = slope * x_fit + intercept

    residuals = y_pred - (slope * y_actual + intercept)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = np.mean(y_actual)
    ss_x = np.sum((y_actual - x_mean) ** 2)
    t_val = t_dist.ppf(0.975, n - 2)
    ci = t_val * s_err * np.sqrt(1.0 / n + (x_fit - x_mean) ** 2 / ss_x)

    ax.plot(x_fit, y_fit, '-', color=color, lw=1.5, alpha=0.8)
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.15)


def plot_3models_parity(records, prop, y_key, primary_sign, orb_primary, orb_secondary,
                        orb_comp, orb_comp2, color, fname):
    """Generate 3-panel parity plot (Koopmans, bilinear, 4 orbitals)."""
    molecules = [r['molecule'] for r in records]
    color_map = get_color_map(molecules)
    n = len(records)

    y = np.array([r[y_key] for r in records])
    x1 = np.array([primary_sign * r[orb_primary] for r in records])
    x2 = np.array([primary_sign * r[orb_secondary] for r in records])
    xc = np.array([r[orb_comp] for r in records])
    xc2 = np.array([r[orb_comp2] for r in records])

    # Model 1: Koopmans
    X1 = np.column_stack([np.ones(n), x1])
    beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    pred1 = X1 @ beta1

    # Model 2: Bilinear
    X2 = np.column_stack([np.ones(n), x1, xc])
    beta2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    pred2 = X2 @ beta2

    # Model 3: 4 orbitals
    X3 = np.column_stack([np.ones(n), x2, x1, xc, xc2])
    beta3 = np.linalg.lstsq(X3, y, rcond=None)[0]
    pred3 = X3 @ beta3

    models = [
        (pred1, 1, 'Koopmans ($\\varepsilon_{\\mathrm{ref}}$)'),
        (pred2, 2, 'Bilin. ($\\varepsilon_H + \\varepsilon_L$)'),
        (pred3, 4, '4 orbitales'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (pred, p_count, title) in zip(axes, models):
        r2a = r2_adj(y, pred, p_count)
        mae = np.mean(np.abs(y - pred))

        # Scatter with colors per molecule
        plotted = set()
        for i, mol in enumerate(molecules):
            label = mol if mol not in plotted else None
            ax.scatter(y[i], pred[i], c=[color_map[mol]],
                       label=label, s=50, alpha=0.7,
                       edgecolors='white', linewidths=0.4, zorder=3)
            plotted.add(mol)

        # Regression line + 95% CI band
        regression_band_multi(ax, y, pred, color)

        # Parity line
        add_parity_line(ax)

        # Stats box
        stats_text(ax, r2a, mae, n)

        ax.set_xlabel(f'{prop} DFT (kJ/mol)')
        ax.set_ylabel(f'{prop} prédit (kJ/mol)')
        ax.set_title(title)
        setup_grid(ax)

    axes[0].legend(loc='lower right', fontsize=5, ncol=2, markerscale=0.7)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(OUTPUT_DIR / f'{fname}.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [+] {fname}.png  (n={n})")


def plot_residuals(records):
    """Residuals of Koopmans vs complement orbital — two separate figures."""
    molecules = [r['molecule'] for r in records]
    color_map = get_color_map(molecules)

    # Common data
    y_ip = np.array([r['IP_kJ'] for r in records])
    x_homo = np.array([-r['eH'] for r in records])
    x_lumo = np.array([r['eL'] for r in records])
    y_ea = np.array([r['EA_kJ'] for r in records])
    x_lumo_ea = np.array([-r['eL'] for r in records])
    x_homo_ea = np.array([r['eH'] for r in records])

    fit_ip = linregress(x_homo, y_ip)
    resid_ip = y_ip - (fit_ip.slope * x_homo + fit_ip.intercept)
    r_corr_ip, p_corr_ip = pearsonr(x_lumo, resid_ip)

    fit_ea = linregress(x_lumo_ea, y_ea)
    resid_ea = y_ea - (fit_ea.slope * x_lumo_ea + fit_ea.intercept)
    r_corr_ea, p_corr_ea = pearsonr(x_homo_ea, resid_ea)

    panels = [
        ('residuals_ip_vs_lumo', x_lumo, resid_ip, r_corr_ip, p_corr_ip,
         '$\\varepsilon_L$ (kJ/mol)', 'Résidu IP Koopmans (kJ/mol)',
         C_RED_DARK, True),
        ('residuals_ea_vs_homo', x_homo_ea, resid_ea, r_corr_ea, p_corr_ea,
         '$\\varepsilon_H$ (kJ/mol)', 'Résidu EA Koopmans (kJ/mol)',
         C_BLUE_DARK, False),
    ]

    for fname, x_vals, resid, r_corr, p_corr, xlabel, ylabel, color, show_legend in panels:
        from style import FIGSIZE_DOUBLE_V
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)

        # Scatter colored by molecule
        plotted = set()
        for i, mol in enumerate(molecules):
            label = mol if mol not in plotted else None
            ax.scatter(x_vals[i], resid[i], c=[color_map[mol]],
                       label=label, s=50, alpha=0.7,
                       edgecolors='white', linewidths=0.4, zorder=3)
            plotted.add(mol)

        regression_with_band(ax, x_vals, resid, color=color)
        ax.axhline(0, color=C_GRAY, ls='--', lw=0.6)

        # Stats box with r and p
        stats = f"$r = {r_corr:.2f}$\n$p = {p_corr:.1e}$"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top',
                fontsize=14, bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                       alpha=0.85, edgecolor='gray', linewidth=0.6),
                family='serif')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show_legend:
            ax.legend(loc='lower right', fontsize=7, ncol=2, markerscale=0.8)
        setup_grid(ax)
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            plt.savefig(OUTPUT_DIR / f'{fname}.{ext}', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [+] {fname}.png")


def main():
    print("Loading data from RKF files...")
    records = load_data()
    mols = set(r['molecule'] for r in records)
    print(f"Loaded {len(records)} conformers from {len(mols)} molecules\n")

    print("Generating IP 3-model parity plot...")
    plot_3models_parity(records, 'IP', 'IP_kJ',
                        primary_sign=-1,
                        orb_primary='eH', orb_secondary='eHm1',
                        orb_comp='eL', orb_comp2='eLp1',
                        color=C_RED_DARK, fname='IP_3models_parity')

    print("Generating EA 3-model parity plot...")
    plot_3models_parity(records, 'EA', 'EA_kJ',
                        primary_sign=-1,
                        orb_primary='eL', orb_secondary='eLp1',
                        orb_comp='eH', orb_comp2='eHm1',
                        color=C_BLUE_DARK, fname='EA_3models_parity')

    print("Generating residuals plot...")
    plot_residuals(records)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
