#!/usr/bin/env amspython
"""
Generate all publication-quality figures for the report.

Produces all figures from:
  - Redox CSV data in data/Acetonitrile/ (always available)
  - Politzer CSVs in rapport/data/ (optional, from analysis scripts)
  - LVC summary CSV in rapport/data/ (optional, from analysis scripts)

Run from project root:
    python3 rapport/generate_figures.py

To generate optional data files:
    python3 -m analysis.generate_ip_ea_comparison --output-dir rapport/data
    python3 -m analysis.generate_lvc_edef_report --batch-dir data/Acetonitrile --output-dir rapport/data
"""

import csv
from pathlib import Path
import numpy as np

from style import (
    apply_style, setup_grid, get_color_map, add_parity_line,
    regression_with_band, stats_text, calibrated_parity,
    C_RED_DARK, C_RED_LIGHT, C_BLUE_DARK, C_BLUE_LIGHT, C_GREEN, C_GRAY,
    FIGSIZE_SINGLE, FIGSIZE_DOUBLE_H, FIGSIZE_DOUBLE_V,
    SCATTER_KW, ANNOT_KW, STATS_BOX,
)

import matplotlib.pyplot as plt
from scipy.stats import linregress

apply_style()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Acetonitrile_marais"
OPTIONAL_DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Molecule filters ─────────────────────────────────────────────────────────
DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}
C_ORANGE = '#e08214'
QUINONES = {
    'CQ', 'HMQ', 'Al-CQ', 'HO-CQ', 'H2C-CQ',
    'Cl-HMQ', 'F-HMQ', 'Br-HMQ', 'I-HMQ',
    'DOPA', 'Cl-DOPA', 'F-DOPA', 'Br-DOPA', 'I-DOPA', 'L-DOPA',
    'O-IQ', 'P-IQ', 'S-IQ', 'IQ',
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def read_csv(path):
    """Read a CSV and return a list of dicts with numeric conversion."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = {}
            for k, v in row.items():
                try:
                    c[k] = float(v)
                except (ValueError, TypeError):
                    c[k] = v
            rows.append(c)
    return rows


def _is_valid(val):
    """Check if a value is numeric and non-NaN."""
    try:
        return not np.isnan(float(val))
    except (ValueError, TypeError):
        return False


def find_molecule_workdirs():
    """Find all catechol workdirs in data/Acetonitrile."""
    mols = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.endswith('_workdir'):
            mol = d.name.replace('_workdir', '')
            if mol not in QUINONES:
                mols.append((mol, d))
    return mols


def load_all_redox_data():
    """Load redox data for all catechol molecules.

    Returns:
        per_mol: dict {mol_name: Boltzmann-averaged data dict}
        per_conf: list of per-conformer data dicts
    """
    per_mol = {}
    per_conf = []

    for mol, wdir in find_molecule_workdirs():
        csv_conf = wdir / "redox" / "results" / f"{mol}_redox_conformers.csv"
        csv_orb = wdir / "redox" / "results" / f"{mol}_orbital_data.csv"

        if not csv_conf.exists():
            continue

        rows = read_csv(csv_conf)
        if not rows:
            continue

        for r in rows:
            r['molecule_name'] = mol
            r['is_dea'] = mol in DEA_EXCLUDED
            per_conf.append(r)

        weights = [r['weight'] for r in rows]
        sw = sum(weights)
        if sw <= 0:
            continue

        avg = {
            'molecule_name': mol,
            'n_conf': len(rows),
            'is_dea': mol in DEA_EXCLUDED,
        }
        numeric_keys = [
            k for k in rows[0]
            if isinstance(rows[0][k], float) and k not in ('weight', 'conformer')
        ]
        for k in numeric_keys:
            vals = [r[k] for r in rows if isinstance(r.get(k), float)]
            ws = [r['weight'] for r in rows if isinstance(r.get(k), float)]
            if vals and sum(ws) > 0:
                avg[k] = sum(v * w for v, w in zip(vals, ws)) / sum(ws)

        if csv_orb.exists():
            orb_rows = read_csv(csv_orb)
            if orb_rows:
                homo_vals = [
                    r.get('HOMO_eV', r.get('homo_eV'))
                    for r in orb_rows
                    if r.get('HOMO_eV') is not None or r.get('homo_eV') is not None
                ]
                lumo_vals = [
                    r.get('LUMO_eV', r.get('lumo_eV'))
                    for r in orb_rows
                    if r.get('LUMO_eV') is not None or r.get('lumo_eV') is not None
                ]
                orb_weights = [r.get('weight', 1.0) for r in orb_rows]
                osw = sum(orb_weights[:len(homo_vals)])
                if homo_vals and osw > 0:
                    avg['HOMO_eV'] = sum(
                        v * w for v, w in zip(homo_vals, orb_weights)
                    ) / osw
                if lumo_vals and osw > 0:
                    avg['LUMO_eV'] = sum(
                        v * w for v, w in zip(lumo_vals, orb_weights)
                    ) / osw

                for orb_r, conf_r in zip(orb_rows, rows):
                    for k in ('HOMO_eV', 'homo_eV', 'LUMO_eV', 'lumo_eV'):
                        if k in orb_r:
                            for pc in per_conf:
                                if (pc['molecule_name'] == mol
                                        and pc.get('conformer') == conf_r.get('conformer')):
                                    pc[k] = orb_r[k]

        per_mol[mol] = avg

    return per_mol, per_conf


def load_politzer_data():
    """Load Politzer CSV data if available.

    Returns:
        (mol_rows, conf_rows) or (None, None) if not found.
    """
    mol_csv = OPTIONAL_DATA_DIR / "politzer_per_molecule.csv"
    conf_csv = OPTIONAL_DATA_DIR / "politzer_per_conformer.csv"

    mol_rows = read_csv(mol_csv) if mol_csv.exists() else None
    conf_rows = read_csv(conf_csv) if conf_csv.exists() else None

    return mol_rows, conf_rows


def load_lvc_data():
    """Load LVC summary CSV if available.

    Expected columns: molecule_name, lvc_red, real_red, lvc_ox, real_ox,
                      lvc2_red, lvc2_ox

    Returns:
        list of dicts or None if not found.
    """
    lvc_csv = OPTIONAL_DATA_DIR / "lvc_summary.csv"
    if not lvc_csv.exists():
        return None
    rows = read_csv(lvc_csv)
    return rows or None


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: DECOMPOSITION MAGNITUDE BARS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_decomposition_bars(per_mol):
    """Stacked bars: relative contribution of |IP/EA|, |Erelax|,
    |ddU - TdS| to |dG|, 2 panels (ox/red)."""
    mols = {k: v for k, v in per_mol.items() if k not in DEA_EXCLUDED}
    names = sorted(mols.keys())

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_DOUBLE_V, sharex=False)
    bar_colors = [C_BLUE_DARK, C_RED_DARK, C_GREEN]

    for ax_idx, (process, title) in enumerate([
        ('oxidation', 'Oxidation'),
        ('reduction', 'Reduction'),
    ]):
        ax = axes[ax_idx]
        mol_names, ip_ea, erelax, thermal = [], [], [], []

        for mol in names:
            d = mols[mol]
            if process == 'oxidation':
                dg = abs(d.get('dG_oxidation_kJ', 0))
                ie = abs(d.get('IP_kJ', 0))
                er = abs(d.get('Edef_ox_kJ', d.get('Erelax_ox_kJ', 0)))
                th = abs(d.get('ddU_ox_kJ', 0) - d.get('TdS_ox_kJ', 0))
            else:
                dg = abs(d.get('dG_reduction_kJ', 0))
                ie = abs(d.get('EA_kJ', 0))
                er = abs(d.get('Edef_red_kJ', d.get('Erelax_red_kJ', 0)))
                th = abs(d.get('ddU_red_kJ', 0) - d.get('TdS_red_kJ', 0))

            if dg > 0:
                mol_names.append(mol)
                ip_ea.append(ie / dg * 100)
                erelax.append(er / dg * 100)
                thermal.append(th / dg * 100)

        x = np.arange(len(mol_names))
        w = 0.55

        ax.bar(x, ip_ea, w, label='|IP| or |EA|',
               color=bar_colors[0], alpha=0.85)
        ax.bar(x, erelax, w, bottom=ip_ea, label='|$E_{relax}$|',
               color=bar_colors[1], alpha=0.85)
        bottoms = [a + b for a, b in zip(ip_ea, erelax)]
        ax.bar(x, thermal, w, bottom=bottoms,
               label='$|\\Delta\\Delta U - T\\Delta S|$',
               color=bar_colors[2], alpha=0.85)

        ax.set_ylabel('Contribution to |$\\Delta G$| (%)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(mol_names, rotation=60, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.axhline(100, color='gray', ls='--', lw=0.6, alpha=0.5)
        setup_grid(ax, limit_ticks=False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'decomposition_magnitude_bars.png')
    plt.close()
    print("  [+] decomposition_magnitude_bars.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 2-5: KOOPMANS CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_koopmans(per_mol, per_conf):
    """IP vs HOMO and EA vs LUMO, per-molecule and per-conformer."""
    mols = {
        k: v for k, v in per_mol.items()
        if 'HOMO_eV' in v and k not in DEA_EXCLUDED
    }
    if not mols:
        print("  [x] No orbital data for Koopmans plots")
        return

    # --- Per-molecule plots (figs 2 & 3) ---
    # Calibrated parity: x = actual DFT, y = a*(-eps) + b
    EV_TO_KJ = 96.4853321  # eV -> kJ/mol
    for (x_key, y_key, prop_label, fname, sign, color) in [
        ('HOMO_eV', 'IP_kJ', 'IP',
         'IP_vs_HOMO_sans_outliers.png', -1, C_RED_DARK),
        ('LUMO_eV', 'EA_kJ', 'EA',
         'EA_vs_LUMO_sans_outliers.png', -1, C_BLUE_DARK),
    ]:
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        koop_raw, dft_vals, labels = [], [], []
        for mol, d in sorted(mols.items()):
            xv = d.get(x_key)
            yv = d.get(y_key)
            if xv is not None and yv is not None:
                koop_raw.append(sign * xv * EV_TO_KJ)
                dft_vals.append(abs(yv))
                labels.append(mol)

        if len(koop_raw) < 3:
            plt.close()
            continue

        koop_raw = np.array(koop_raw)
        dft_vals = np.array(dft_vals)
        n = len(dft_vals)

        # Calibration: fit dft = a * koop + b, then calibrated = a * koop + b
        slope, intercept, r_val, _, _ = linregress(koop_raw, dft_vals)
        r2 = r_val ** 2
        calibrated = slope * koop_raw + intercept
        mae = np.mean(np.abs(dft_vals - calibrated))

        # Parity plot: x = DFT, y = calibrated
        ax.scatter(dft_vals, calibrated, c=color, **SCATTER_KW)
        for i, mol in enumerate(labels):
            ax.annotate(mol, (dft_vals[i], calibrated[i]), **ANNOT_KW)

        regression_with_band(ax, dft_vals, calibrated, color=color)
        add_parity_line(ax)
        stats_text(ax, r2, mae, n)

        ax.set_xlabel(f'{prop_label} DFT (kJ/mol)')
        ax.set_ylabel(f'{prop_label} Koopmans calibré (kJ/mol)')
        setup_grid(ax)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / fname)
        plt.close()
        print(f"  [+] {fname}  (R2={r2:.3f}, MAE={mae:.1f})")

    # --- Per-conformer plots (figs 4 & 5) ---
    conf_data = [c for c in per_conf if c['molecule_name'] not in DEA_EXCLUDED]
    color_map = get_color_map([c['molecule_name'] for c in conf_data])

    for (x_key, y_key, prop_label, fname, sign, reg_color) in [
        ('HOMO_eV', 'IP_kJ', 'IP',
         'IP_vs_HOMO_per_conformer_sans_outliers.png', -1, C_RED_DARK),
        ('LUMO_eV', 'EA_kJ', 'EA',
         'EA_vs_LUMO_per_conformer_sans_outliers.png', -1, C_BLUE_DARK),
    ]:
        alt_x = 'homo_eV' if 'HOMO' in x_key else 'lumo_eV'

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        koop_raw, dft_vals, mol_names = [], [], []

        for c in conf_data:
            xv = c.get(x_key, c.get(alt_x))
            yv = c.get(y_key)
            if xv is not None and yv is not None:
                koop_raw.append(sign * xv * EV_TO_KJ)
                dft_vals.append(abs(yv))
                mol_names.append(c['molecule_name'])

        if len(koop_raw) < 5:
            plt.close()
            continue

        koop_raw = np.array(koop_raw)
        dft_vals = np.array(dft_vals)
        n = len(dft_vals)

        # Calibration: fit dft = a * koop + b
        slope, intercept, r_val, _, _ = linregress(koop_raw, dft_vals)
        calibrated = slope * koop_raw + intercept
        mae = np.mean(np.abs(dft_vals - calibrated))
        # R²_adj with p=1 predictor
        ss_res = np.sum((dft_vals - calibrated) ** 2)
        ss_tot = np.sum((dft_vals - np.mean(dft_vals)) ** 2)
        r2_raw = 1 - ss_res / ss_tot
        r2a = 1 - (1 - r2_raw) * (n - 1) / (n - 2)

        # Parity plot: x = DFT, y = calibrated
        plotted = set()
        for i, mol in enumerate(mol_names):
            label = mol if mol not in plotted else None
            ax.scatter(dft_vals[i], calibrated[i], c=[color_map[mol]],
                       label=label, **SCATTER_KW)
            plotted.add(mol)

        regression_with_band(ax, dft_vals, calibrated, color=reg_color)
        add_parity_line(ax)
        stats_text(ax, r2a, mae, n, adjusted=True)

        ax.set_xlabel(f'{prop_label} DFT (kJ/mol)')
        ax.set_ylabel(f'{prop_label} Koopmans calibré (kJ/mol)')
        # Legend only on IP plot (c), not on EA (d)
        if prop_label == 'IP':
            ax.legend(loc='lower right', fontsize=7, ncol=2,
                      markerscale=0.8)
        setup_grid(ax)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / fname)
        plt.close()
        print(f"  [+] {fname}  (R2adj={r2a:.3f}, MAE={mae:.1f}, n={n})")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 6-7: POLITZER PARITY (per-molecule)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_politzer_parity(mol_rows):
    """IP and EA parity plots: Koopmans vs Politzer (calibrated), one per file."""
    politzer_dir = OUTPUT_DIR / 'politzer'
    politzer_dir.mkdir(exist_ok=True)

    # --- Collect data ---
    x_ip_k, y_ip_k, lbl_ip_k = [], [], []
    x_ip_p, y_ip_p, lbl_ip_p = [], [], []
    x_ea_k, y_ea_k, lbl_ea_k = [], [], []
    x_ea_v, y_ea_v, lbl_ea_v = [], [], []
    for row in mol_rows:
        mol = row['molecule_name']
        ip_dft = row.get('IP_kJ')
        ip_koop = row.get('IP_Koopmans_kJ')
        ip_pol = row.get('IP_Politzer_kJ')
        ea_dft = row.get('EA_kJ')
        ea_koop = row.get('EA_Koopmans_kJ')
        v_s_min_kj = row.get('V_S_min_kJ')
        if _is_valid(ip_dft) and _is_valid(ip_koop):
            x_ip_k.append(float(ip_dft)); y_ip_k.append(float(ip_koop)); lbl_ip_k.append(mol)
        if _is_valid(ip_dft) and _is_valid(ip_pol):
            x_ip_p.append(float(ip_dft)); y_ip_p.append(float(ip_pol)); lbl_ip_p.append(mol)
        if _is_valid(ea_dft) and _is_valid(ea_koop):
            x_ea_k.append(float(ea_dft)); y_ea_k.append(float(ea_koop)); lbl_ea_k.append(mol)
        if _is_valid(ea_dft) and _is_valid(v_s_min_kj):
            x_ea_v.append(float(ea_dft)); y_ea_v.append(-float(v_s_min_kj)); lbl_ea_v.append(mol)

    # --- Generate one figure per panel (calibrated parity) ---
    panels = [
        (x_ip_k, y_ip_k, lbl_ip_k, C_RED_DARK,
         'IP DFT (kJ/mol)', 'IP Koopmans calibré (kJ/mol)',
         'IP_Koopmans_per_molecule.png'),
        (x_ip_p, y_ip_p, lbl_ip_p, C_RED_DARK,
         'IP DFT (kJ/mol)', 'IP Politzer calibré (kJ/mol)',
         'IP_Politzer_per_molecule.png'),
        (x_ea_k, y_ea_k, lbl_ea_k, C_BLUE_DARK,
         'EA DFT (kJ/mol)', 'EA Koopmans calibré (kJ/mol)',
         'EA_Koopmans_per_molecule.png'),
        (x_ea_v, y_ea_v, lbl_ea_v, C_BLUE_DARK,
         'EA DFT (kJ/mol)', 'EA Politzer calibré (kJ/mol)',
         'EA_Politzer_per_molecule.png'),
    ]
    for (xd, yd, lbls, color, xlabel, ylabel, fname) in panels:
        if len(xd) < 2:
            continue
        xd = np.asarray(xd, dtype=float)
        yd = np.asarray(yd, dtype=float)

        # Calibration: fit x_dft = a * y_raw + b
        slope_cal, intercept_cal, r_val, _, _ = linregress(yd, xd)
        y_cal = slope_cal * yd + intercept_cal
        r2 = r_val ** 2
        mae = np.mean(np.abs(y_cal - xd))
        n = len(xd)

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        ax.scatter(xd, y_cal, c=color, **SCATTER_KW)
        for i, mol in enumerate(lbls):
            ax.annotate(mol, (xd[i], y_cal[i]), **ANNOT_KW)

        regression_with_band(ax, xd, y_cal, color=color)
        add_parity_line(ax)
        stats_text(ax, r2, mae, n)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        setup_grid(ax)
        plt.tight_layout()
        plt.savefig(politzer_dir / fname)
        plt.close()
        print(f"  [+] politzer/{fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 8-9: POLITZER PER-CONFORMER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_politzer_per_conformer(conf_rows):
    """IP vs ALIE and EA vs VS per-conformer: calibrated parity plots."""
    politzer_dir = OUTPUT_DIR / 'politzer'
    politzer_dir.mkdir(exist_ok=True)
    EV_TO_KJ = 96.4853321  # eV -> kJ/mol

    color_map = get_color_map([r['molecule_name'] for r in conf_rows])

    for (raw_col, dft_col, prop_label, fname, sign_raw) in [
        ('I_S_min_eV', 'IP_kJ', 'IP',
         'IP_vs_ALIE_per_conformer.png', 1),
        ('V_S_min_eV', 'EA_kJ', 'EA',
         'EA_vs_VS_per_conformer.png', -1),
    ]:
        # Collect valid data
        raw_vals, dft_vals, mol_names = [], [], []
        for r in conf_rows:
            rv = r.get(raw_col)
            dv = r.get(dft_col)
            if _is_valid(rv) and _is_valid(dv):
                raw_vals.append(sign_raw * float(rv) * EV_TO_KJ)
                dft_vals.append(abs(float(dv)))
                mol_names.append(r['molecule_name'])

        if len(raw_vals) < 5:
            print(f"  [x] No valid data for {fname}")
            continue

        raw_vals = np.array(raw_vals)
        dft_vals = np.array(dft_vals)
        n = len(dft_vals)

        # Calibration: fit dft = a * raw + b
        slope_cal, intercept_cal, r_val, _, _ = linregress(raw_vals, dft_vals)
        calibrated = slope_cal * raw_vals + intercept_cal
        mae = np.mean(np.abs(dft_vals - calibrated))
        # R²_adj with p=1 predictor
        ss_res = np.sum((dft_vals - calibrated) ** 2)
        ss_tot = np.sum((dft_vals - np.mean(dft_vals)) ** 2)
        r2_raw = 1 - ss_res / ss_tot
        r2a = 1 - (1 - r2_raw) * (n - 1) / (n - 2)

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)

        # Scatter by molecule: x = DFT, y = calibrated
        plotted = set()
        for i, mol in enumerate(mol_names):
            label = mol if mol not in plotted else None
            ax.scatter(dft_vals[i], calibrated[i], c=[color_map[mol]],
                       label=label, **SCATTER_KW)
            plotted.add(mol)

        regression_with_band(ax, dft_vals, calibrated,
                             color=C_RED_DARK if 'IP' in prop_label else C_BLUE_DARK)
        add_parity_line(ax)
        stats_text(ax, r2a, mae, n, adjusted=True)

        ax.set_xlabel(f'{prop_label} DFT (kJ/mol)')
        ax.set_ylabel(f'{prop_label} Politzer calibré (kJ/mol)')
        # Legend only on IP plot (c), not on EA (d)
        if prop_label == 'IP':
            ax.legend(loc='lower right', fontsize=7, ncol=2, markerscale=0.8)
        setup_grid(ax)

        plt.tight_layout()
        plt.savefig(politzer_dir / fname)
        plt.close()
        print(f"  [+] politzer/{fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 10-11: LVC vs REAL (PARITY SCATTER)
# ═══════════════════════════════════════════════════════════════════════════════

def _lvc_parity(ax, x_real, y_raw, mol_names, color, title,
                calibrate=False):
    """Unified LVC parity scatter with regression + parity line.

    If calibrate=True, plot calibrated values (a*y_raw+b) as a parity plot
    with x=real, y=calibrated. Otherwise plot raw values.
    """
    def _safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan
    x_real = np.array([_safe_float(v) for v in x_real])
    y_raw = np.array([_safe_float(v) for v in y_raw])
    mask = np.isfinite(x_real) & np.isfinite(y_raw) & (x_real != 0) & (y_raw != 0)

    if np.sum(mask) < 3:
        return

    xm, ym_raw = x_real[mask], y_raw[mask]
    names_m = [m for m, keep in zip(mol_names, mask) if keep]

    if calibrate:
        # Calibration: fit real = a * raw + b
        slope_cal, intercept_cal, r_val, _, _ = linregress(ym_raw, xm)
        ym = slope_cal * ym_raw + intercept_cal
        r2 = r_val ** 2
        mae = np.mean(np.abs(xm - ym))
        n = len(xm)

        ax.scatter(xm, ym, c=color, **SCATTER_KW)
        for i, name in enumerate(names_m):
            ax.annotate(name, (xm[i], ym[i]), **ANNOT_KW)

        regression_with_band(ax, xm, ym, color=color)
        add_parity_line(ax)
        stats_text(ax, r2, mae, n)

        ax.set_xlabel('$E_{relax}$ réel (kJ/mol)')
        ax.set_ylabel('$E_{relax}$ LVC calibré (kJ/mol)')
    else:
        ym = ym_raw
        ax.scatter(xm, ym, c=color, **SCATTER_KW)
        for i, name in enumerate(names_m):
            ax.annotate(name, (xm[i], ym[i]), **ANNOT_KW)

        # Parity line
        all_vals = np.concatenate([xm, ym])
        margin = (all_vals.max() - all_vals.min()) * 0.1
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)

        if len(xm) >= 3:
            slope, intercept, r2, mae, n = regression_with_band(
                ax, xm, ym, color=color)
            stats_text(ax, r2, mae, n)

        ax.set_xlabel('$E_{relax}$ réel (kJ/mol)')
        ax.set_ylabel('$E_{relax}$ prédit (kJ/mol)')

    setup_grid(ax)


def plot_lvc_vs_real(lvc_data):
    """LVC vs real Erelax: oxidation and reduction parity plots."""
    def _sf(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan
    molecules = [r['molecule_name'] for r in lvc_data]
    lvc_red = np.array([_sf(r.get('lvc_red')) for r in lvc_data])
    real_red = np.array([_sf(r.get('real_red')) for r in lvc_data])
    lvc_ox = np.array([_sf(r.get('lvc_ox')) for r in lvc_data])
    real_ox = np.array([_sf(r.get('real_ox')) for r in lvc_data])

    # Oxidation
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    _lvc_parity(ax, real_ox, lvc_ox, molecules, C_RED_DARK, 'Oxydation',
                calibrate=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lvc_vs_real_oxidation.png')
    plt.close()
    print("  [+] lvc_vs_real_oxidation.png")

    # Reduction
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    _lvc_parity(ax, real_red, lvc_red, molecules, C_BLUE_DARK, 'Réduction',
                calibrate=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lvc_vs_real_reduction.png')
    plt.close()
    print("  [+] lvc_vs_real_reduction.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: ERROR vs RMSD (reduction)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_vs_rmsd(per_mol):
    """|Erelax| vs RMSD scatter with threshold lines."""
    mols = {k: v for k, v in per_mol.items() if k not in DEA_EXCLUDED}

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    x, y, labels = [], [], []
    for mol, d in sorted(mols.items()):
        rmsd = d.get('RMSD_red_A')
        erelax = d.get('Edef_red_kJ', d.get('Erelax_red_kJ'))
        if rmsd is not None and erelax is not None:
            x.append(rmsd)
            y.append(abs(erelax))
            labels.append(mol)

    if len(x) < 3:
        plt.close()
        print("  [x] Not enough data for error_vs_rmsd_reduction")
        return

    ax.scatter(x, y, c=C_BLUE_DARK, **SCATTER_KW)
    for i, mol in enumerate(labels):
        ax.annotate(mol, (x[i], y[i]), **ANNOT_KW)

    ax.axvline(0.15, color=C_RED_LIGHT, ls='--', lw=1.0, alpha=0.7,
               label='RMSD = 0.15 \u00c5')
    ax.axvline(0.30, color=C_RED_DARK, ls='--', lw=1.0, alpha=0.7,
               label='RMSD = 0.30 \u00c5')

    ax.set_xlabel('RMSD neutral $\\rightarrow$ reduced (\u00c5)')
    ax.set_ylabel('|$E_{\\mathrm{relax}}^{\\mathrm{red}}$| (kJ/mol)')
    ax.legend(fontsize=7)
    setup_grid(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_vs_rmsd_reduction.png')
    plt.close()
    print("  [+] error_vs_rmsd_reduction.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: ERELAX COMPARISON BARS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_erelax_comparison_bars(per_mol):
    """Erelax per molecule, one file per process (ox/red)."""
    mols = {k: v for k, v in per_mol.items() if k not in DEA_EXCLUDED}
    names = sorted(mols.keys())

    for suffix, key, color_pos, color_neg in [
        ('ox', 'Edef_ox_kJ', C_RED_DARK, C_RED_LIGHT),
        ('red', 'Edef_red_kJ', C_BLUE_DARK, C_BLUE_LIGHT),
    ]:
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        vals = [mols[m].get(key, 0) for m in names]
        x = np.arange(len(names))

        colors = [color_pos if v >= 0 else color_neg for v in vals]
        ax.bar(x, vals, 0.55, color=colors, alpha=0.85,
               edgecolor='white', linewidth=0.3)
        ax.set_ylabel('$E_{\\mathrm{relax}}$ (kJ/mol)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='gray', lw=0.6)
        setup_grid(ax, limit_ticks=False)

        fname = f'lvc_vs_real_comparison_bars_{suffix}.png'
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / fname)
        plt.close()
        print(f"  [+] {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 14-15: LVC vs LVC2
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lvc_vs_lvc2(lvc_data):
    """LVC (calibrated) vs LVC2 corrected (1+fO^2): one file per model per process.

    DEA molecules (Br-DA, Br-HMC, I-DA, I-HMC) are overlaid with distinct
    markers but excluded from the regression / stats of the stable set.
    """
    # Load fO descriptors
    desc_csv = OPTIONAL_DATA_DIR / "descriptors_per_molecule.csv"
    if not desc_csv.exists():
        print("  [!] descriptors_per_molecule.csv not found, skipping LVC vs LVC2")
        return
    desc_rows = read_csv(desc_csv)
    fO_map = {r['molecule']: float(r['LUMO_O_frac'])
              for r in desc_rows if _is_valid(r.get('LUMO_O_frac'))}

    molecules = [r['molecule_name'] for r in lvc_data]
    is_dea = np.array([m in DEA_EXCLUDED for m in molecules])
    lumo_O = np.array([fO_map.get(m, np.nan) for m in molecules])

    for suffix, title, real_key, lvc_key, lvc2_key, color in [
        ('oxidation', 'Oxydation', 'real_ox', 'lvc_ox', 'lvc2_ox', C_RED_DARK),
        ('reduction', 'R\u00e9duction', 'real_red', 'lvc_red', 'lvc2_red', C_BLUE_DARK),
    ]:
        def _sf(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan
        real = np.array([_sf(r.get(real_key)) for r in lvc_data])
        lvc = np.array([_sf(r.get(lvc_key)) for r in lvc_data])
        lvc2_raw = np.array([_sf(r.get(lvc2_key)) for r in lvc_data])

        # LVC panel (calibrated) — stable molecules only (DEA have no LVC values)
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        _lvc_parity(ax, real, lvc, molecules, color, f'LVC ({title})',
                    calibrate=True)
        plt.tight_layout()
        fname_lvc = f'lvc_vs_lvc2_lvc_{suffix}.png'
        plt.savefig(OUTPUT_DIR / fname_lvc)
        plt.close()
        print(f"  [+] {fname_lvc}")

        # LVC2 panel: corrected with (1 + fO^2), stable molecules only
        lvc2_corr = lvc2_raw * (1 + lumo_O ** 2)
        valid = np.isfinite(real) & np.isfinite(lvc2_corr) & ~is_dea

        if np.sum(valid) < 3:
            continue

        xs, ys = real[valid], lvc2_corr[valid]
        names_s = [m for m, k in zip(molecules, valid) if k]

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        ax.scatter(xs, ys, c=color, **SCATTER_KW)
        for i, name in enumerate(names_s):
            ax.annotate(name, (xs[i], ys[i]), **ANNOT_KW)

        regression_with_band(ax, xs, ys, color=color)
        add_parity_line(ax)

        r2 = 1 - np.sum((xs - ys)**2) / np.sum((xs - np.mean(xs))**2)
        mae = np.mean(np.abs(xs - ys))
        stats_text(ax, r2, mae, len(xs))

        ax.set_xlabel('$E_{relax}$ réel (kJ/mol)')
        ax.set_ylabel('$E_{relax}$ LVC2 corrigé (kJ/mol)')
        setup_grid(ax)
        plt.tight_layout()
        fname_lvc2 = f'lvc_vs_lvc2_lvc2_{suffix}.png'
        plt.savefig(OUTPUT_DIR / fname_lvc2)
        plt.close()
        print(f"  [+] {fname_lvc2}  (R2={r2:.3f}, MAE={mae:.1f}, n={len(xs)})")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 15b: LVC2 CORRECTED (reduction only)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lvc2_corrected(lvc_data):
    """Parity plot: LVC2 brut vs LVC2 corrigé (LUMO_O) for reduction."""
    # Load descriptors
    desc_csv = OPTIONAL_DATA_DIR / "descriptors_per_molecule.csv"
    if not desc_csv.exists():
        print("  [!] descriptors_per_molecule.csv not found, skipping LVC2 corrected")
        return

    desc_rows = read_csv(desc_csv)
    lumo_O_by_mol = {r['molecule']: float(r['LUMO_O_frac'])
                     for r in desc_rows if _is_valid(r.get('LUMO_O_frac'))}

    def _sf(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan
    molecules = [r['molecule_name'] for r in lvc_data]
    is_dea = np.array([m in DEA_EXCLUDED for m in molecules])
    real_red = np.array([_sf(r.get('real_red')) for r in lvc_data])
    lvc2_red = np.array([_sf(r.get('lvc2_red')) for r in lvc_data])

    # Parameter-free correction: E_real = LVC2 * (1 + fO^2)
    lumo_O = np.array([lumo_O_by_mol.get(m, np.nan) for m in molecules])
    corrected = lvc2_red * (1 + lumo_O ** 2)

    valid = np.isfinite(real_red) & np.isfinite(corrected)
    mask_stable = valid & ~is_dea

    if np.sum(mask_stable) < 3:
        print("  [!] Not enough data for LVC2 corrected plot")
        return

    # Panel (a): LVC2 brut (stable molecules only)
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    xs_brut = real_red[mask_stable]
    ys_brut = lvc2_red[mask_stable]
    names_s_brut = [m for m, k in zip(molecules, mask_stable) if k]
    ax.scatter(xs_brut, ys_brut, c=C_BLUE_DARK, **SCATTER_KW)
    for i, name in enumerate(names_s_brut):
        ax.annotate(name, (xs_brut[i], ys_brut[i]), **ANNOT_KW)
    if len(xs_brut) >= 3:
        regression_with_band(ax, xs_brut, ys_brut, color=C_BLUE_DARK)
        all_vals = np.concatenate([xs_brut, ys_brut])
        margin = (all_vals.max() - all_vals.min()) * 0.1
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
        r2_brut = 1 - np.sum((xs_brut - ys_brut)**2) / np.sum((xs_brut - np.mean(xs_brut))**2)
        mae_brut = np.mean(np.abs(xs_brut - ys_brut))
        stats_text(ax, r2_brut, mae_brut, len(xs_brut))
    ax.set_xlabel('$E_{relax}$ réel (kJ/mol)')
    ax.set_ylabel('$E_{relax}$ LVC2 brut (kJ/mol)')
    setup_grid(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lvc2_brut_reduction.png')
    plt.close()
    print("  [+] lvc2_brut_reduction.png")

    # Panel (b): LVC2 corrigé (stable molecules only)
    xs = real_red[mask_stable]
    ys = corrected[mask_stable]
    names_s = [m for m, k in zip(molecules, mask_stable) if k]

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    ax.scatter(xs, ys, c=C_BLUE_DARK, **SCATTER_KW)
    for i, name in enumerate(names_s):
        ax.annotate(name, (xs[i], ys[i]), **ANNOT_KW)

    all_vals = np.concatenate([xs, ys])
    margin = (all_vals.max() - all_vals.min()) * 0.1
    lims = [all_vals.min() - margin, all_vals.max() + margin]
    ax.plot(lims, lims, '--', color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)

    mae = np.mean(np.abs(xs - ys))
    maxerr = np.max(np.abs(xs - ys))
    from scipy.stats import linregress as lr
    fit = lr(xs, ys)
    r2 = fit.rvalue ** 2

    stats_text(ax, r2, mae, len(xs))
    ax.set_xlabel('$E_{relax}$ réel (kJ/mol)')
    ax.set_ylabel('$E_{relax}$ corrigé (kJ/mol)')
    setup_grid(ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lvc2_corrected_reduction.png')
    plt.close()
    print(f"  [+] lvc2_corrected_reduction.png  (MAE={mae:.1f}, Max|err|={maxerr:.1f})")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 16-18: THERMODYNAMIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_thermodynamic_models(per_mol):
    """ddU distribution, TdS distribution, and dG parity plot."""
    mols = {k: v for k, v in per_mol.items() if k not in DEA_EXCLUDED}
    thermo_dir = OUTPUT_DIR / 'thermodynamic_models'
    thermo_dir.mkdir(exist_ok=True)

    # ── Figure 16: ddU distribution (one file per process) ──
    for suffix, ddu_key, color in [
        ('ox', 'ddU_ox_kJ', C_RED_DARK),
        ('red', 'ddU_red_kJ', C_BLUE_DARK),
    ]:
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        y_vals, labels = [], []
        for mol, d in sorted(mols.items()):
            ddu = d.get(ddu_key)
            if ddu is not None:
                y_vals.append(ddu)
                labels.append(mol)

        if not y_vals:
            ax.text(0.5, 0.5, 'No data available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color=C_GRAY)
            plt.close()
            continue

        x_pos = np.arange(len(y_vals))
        ax.scatter(x_pos, y_vals, c=color, **SCATTER_KW)
        for i, mol in enumerate(labels):
            ax.annotate(mol, (x_pos[i], y_vals[i]),
                        xytext=(3, 4), textcoords='offset points',
                        fontsize=6, rotation=40, color=C_GRAY)

        mean_val = np.mean(y_vals)
        std_val = np.std(y_vals)
        ax.axhline(mean_val, color=C_GREEN, ls='--', lw=1.2, alpha=0.7,
                   label=f'Mean = {mean_val:.1f} kJ/mol')
        ax.axhspan(mean_val - std_val, mean_val + std_val,
                   color=C_GREEN, alpha=0.08)

        ax.set_ylabel('$\\Delta\\Delta U$ (kJ/mol)')
        ax.set_xticks([])
        ax.legend(fontsize=8)
        setup_grid(ax)

        fname = f'ddU_{suffix}.png'
        plt.tight_layout()
        plt.savefig(thermo_dir / fname)
        plt.close()
        print(f"  [+] thermodynamic_models/{fname}")

    # ── Figure 17: TdS distribution (one file per process) ──
    for suffix, tds_key, color in [
        ('ox', 'TdS_ox_kJ', C_RED_DARK),
        ('red', 'TdS_red_kJ', C_BLUE_DARK),
    ]:
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        y_vals, labels = [], []
        for mol, d in sorted(mols.items()):
            tds = d.get(tds_key)
            if tds is not None:
                y_vals.append(tds)
                labels.append(mol)

        x_pos = np.arange(len(y_vals))
        ax.scatter(x_pos, y_vals, c=color, **SCATTER_KW)
        for i, mol in enumerate(labels):
            ax.annotate(mol, (x_pos[i], y_vals[i]),
                        xytext=(3, 4), textcoords='offset points',
                        fontsize=6, rotation=40, color=C_GRAY)

        mean_val = np.mean(y_vals)
        std_val = np.std(y_vals)
        ax.axhline(mean_val, color=C_GREEN, ls='--', lw=1.2, alpha=0.7,
                   label=f'Mean = {mean_val:.1f} kJ/mol')
        ax.axhspan(mean_val - std_val, mean_val + std_val,
                   color=C_GREEN, alpha=0.08)

        ax.set_ylabel('$T\\Delta S$ (kJ/mol)')
        ax.set_xticks([])
        ax.legend(fontsize=8)
        setup_grid(ax)

        fname = f'TdS_{suffix}.png'
        plt.tight_layout()
        plt.savefig(thermo_dir / fname)
        plt.close()
        print(f"  [+] thermodynamic_models/{fname}")

    # ── Figure 18: dG parity (DFT vs predicted) ──
    # Prediction: Koopmans (IP/EA) + LVC2 (Erelax) + constant (ddU, TdS)
    # Load LVC2 data
    lvc_data = load_lvc_data()
    lvc2_by_mol = {}
    if lvc_data:
        for r in lvc_data:
            lvc2_by_mol[r['molecule_name']] = r

    # Compute mean ddU and TdS as constant models
    ddU_ox_vals = [d.get('ddU_ox_kJ') for d in mols.values()
                   if d.get('ddU_ox_kJ') is not None]
    ddU_red_vals = [d.get('ddU_red_kJ') for d in mols.values()
                    if d.get('ddU_red_kJ') is not None]
    TdS_ox_vals = [d.get('TdS_ox_kJ') for d in mols.values()
                   if d.get('TdS_ox_kJ') is not None]
    TdS_red_vals = [d.get('TdS_red_kJ') for d in mols.values()
                    if d.get('TdS_red_kJ') is not None]
    mean_ddU_ox = np.mean(ddU_ox_vals) if ddU_ox_vals else 0
    mean_ddU_red = np.mean(ddU_red_vals) if ddU_red_vals else 0
    mean_TdS_ox = np.mean(TdS_ox_vals) if TdS_ox_vals else 0
    mean_TdS_red = np.mean(TdS_red_vals) if TdS_red_vals else 0

    # Koopmans calibration: fit IP/EA from HOMO/LUMO
    homo_vals, ip_vals = [], []
    lumo_vals, ea_vals = [], []
    for mol, d in mols.items():
        if d.get('HOMO_eV') is not None and d.get('IP_kJ') is not None:
            homo_vals.append(-d['HOMO_eV'])
            ip_vals.append(d['IP_kJ'])
        if d.get('LUMO_eV') is not None and d.get('EA_kJ') is not None:
            lumo_vals.append(-d['LUMO_eV'])
            ea_vals.append(abs(d['EA_kJ']))

    ip_fit = linregress(homo_vals, ip_vals) if len(homo_vals) >= 3 else None
    ea_fit = linregress(lumo_vals, ea_vals) if len(lumo_vals) >= 3 else None

    for suffix, dg_key, color, sign_ea, erelax_key, \
            mean_ddU, mean_TdS, orb_key, orb_sign, orb_fit in [
        ('ox', 'dG_oxidation_kJ', C_RED_DARK, +1, 'lvc2_ox',
         mean_ddU_ox, mean_TdS_ox, 'HOMO_eV', -1, ip_fit),
        ('red', 'dG_reduction_kJ', C_BLUE_DARK, -1, 'lvc2_red',
         mean_ddU_red, mean_TdS_red, 'LUMO_eV', -1, ea_fit),
    ]:
        if orb_fit is None:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
        x_dft, y_pred, labels = [], [], []

        for mol, d in sorted(mols.items()):
            dg = d.get(dg_key)
            orb = d.get(orb_key)
            lvc2_mol = lvc2_by_mol.get(mol, {})
            erelax = lvc2_mol.get(erelax_key)

            if dg is None or orb is None:
                continue
            if not _is_valid(erelax):
                continue

            erelax = float(erelax)
            # Koopmans predicted IP or EA
            ip_ea_pred = orb_fit.slope * (orb_sign * orb) + orb_fit.intercept

            # dG predicted = sign * IP/EA + Erelax + ddU - TdS
            pred = sign_ea * ip_ea_pred + erelax + mean_ddU - mean_TdS

            x_dft.append(dg)
            y_pred.append(pred)
            labels.append(mol)

        if len(x_dft) < 3:
            plt.close()
            continue

        x_arr = np.array(x_dft)
        y_arr = np.array(y_pred)
        n = len(x_arr)

        ax.scatter(x_arr, y_arr, c=color, **SCATTER_KW)
        for i, mol in enumerate(labels):
            ax.annotate(mol, (x_arr[i], y_arr[i]), **ANNOT_KW)

        regression_with_band(ax, x_arr, y_arr, color=color)
        add_parity_line(ax)

        mae_res = np.mean(np.abs(x_arr - y_arr))
        r2_res = linregress(x_arr, y_arr).rvalue ** 2
        stats_text(ax, r2_res, mae_res, n)
        ax.set_xlabel('$\\Delta G^\\circ$ DFT (kJ/mol)')
        ax.set_ylabel('$\\Delta G^\\circ$ pr\u00e9dit (kJ/mol)')
        setup_grid(ax)

        fname = f'dG_parity_{suffix}.png'
        plt.tight_layout()
        plt.savefig(thermo_dir / fname)
        plt.close()
        print(f"  [+] thermodynamic_models/{fname}  (R2={r2_res:.3f}, MAE={mae_res:.1f})")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES: BEYOND KOOPMANS (3-model regression)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_extended_orbitals(per_conf):
    """Extract HOMO-1, HOMO, LUMO, LUMO+1 from neutral_tz2p RKFs.

    Returns list of dicts with extended orbital data merged with redox data.
    """
    from scm.plams import KFFile

    EH_TO_EV = 27.211386245988
    records = []

    for c in per_conf:
        mol = c['molecule_name']
        if mol in DEA_EXCLUDED:
            continue

        conf_num = c.get('conformer')
        if conf_num is None:
            continue

        wdir = DATA_DIR / f"{mol}_workdir"
        plams_dir = wdir / "redox" / "calculations" / "plams_workdir"
        rkf_path = plams_dir / f"{mol}_conf_{int(conf_num)}_neutral_tz2p" / "adf.rkf"

        if not rkf_path.exists():
            continue

        try:
            kf = KFFile(str(rkf_path))
            eps = np.array(kf.read('A', 'eps_A'), dtype=float)
            froc = np.array(kf.read('A', 'froc_A'), dtype=float)
            n_occ = int(np.sum(froc > 0))
            n_virt = len(eps) - n_occ

            if n_occ < 2 or n_virt < 2:
                continue

            rec = dict(c)
            rec['HOMO_tz2p_eV'] = eps[n_occ - 1] * EH_TO_EV
            rec['HOMOm1_tz2p_eV'] = eps[n_occ - 2] * EH_TO_EV
            rec['LUMO_tz2p_eV'] = eps[n_occ] * EH_TO_EV
            rec['LUMOp1_tz2p_eV'] = eps[n_occ + 1] * EH_TO_EV
            records.append(rec)
        except Exception as e:
            print(f"    [!] {mol} conf_{conf_num}: {e}")

    return records


def plot_beyond_koopmans(per_conf):
    """Generate 3-model parity plots (Koopmans, bilinear, 4-orbital) for IP and EA."""
    bk_dir = OUTPUT_DIR / 'beyond_koopmans'
    bk_dir.mkdir(exist_ok=True)

    records = extract_extended_orbitals(per_conf)
    if len(records) < 5:
        print("  [!] Not enough data for beyond_koopmans plots")
        return

    EV_TO_KJ = 96.4853321
    color_map = get_color_map([r['molecule_name'] for r in records])

    for prop, y_key, orb_keys, sign, color, fname_parity, fname_resid in [
        ('IP', 'IP_kJ',
         ['HOMO_tz2p_eV', 'HOMOm1_tz2p_eV', 'LUMO_tz2p_eV', 'LUMOp1_tz2p_eV'],
         -1, C_RED_DARK,
         'IP_3models_parity', 'residuals_koopmans_vs_complement'),
        ('EA', 'EA_kJ',
         ['LUMO_tz2p_eV', 'LUMOp1_tz2p_eV', 'HOMO_tz2p_eV', 'HOMOm1_tz2p_eV'],
         -1, C_BLUE_DARK,
         'EA_3models_parity', 'residuals_koopmans_vs_complement'),
    ]:
        # Filter valid records
        valid = []
        for r in records:
            y = r.get(y_key)
            if y is None or not _is_valid(y):
                continue
            if all(_is_valid(r.get(k)) for k in orb_keys):
                valid.append(r)

        if len(valid) < 5:
            continue

        y_arr = np.array([abs(float(r[y_key])) for r in valid])
        mol_names = [r['molecule_name'] for r in valid]

        # Primary orbital (HOMO for IP, LUMO for EA)
        x1 = np.array([sign * float(r[orb_keys[0]]) * EV_TO_KJ for r in valid])
        # Secondary orbital (HOMO-1 for IP, LUMO+1 for EA)
        x2 = np.array([sign * float(r[orb_keys[1]]) * EV_TO_KJ for r in valid])
        # Complement orbital (LUMO for IP, HOMO for EA)
        x_comp = np.array([float(r[orb_keys[2]]) * EV_TO_KJ for r in valid])
        # Complement+1 (LUMO+1 for IP, HOMO-1 for EA)
        x_comp2 = np.array([float(r[orb_keys[3]]) * EV_TO_KJ for r in valid])

        n = len(y_arr)

        # Model 1: Koopmans calibrated
        fit1 = linregress(x1, y_arr)
        pred1 = fit1.slope * x1 + fit1.intercept

        # Model 2: Bilinear (primary + complement)
        X2 = np.column_stack([np.ones(n), x1, x_comp])
        beta2 = np.linalg.lstsq(X2, y_arr, rcond=None)[0]
        pred2 = X2 @ beta2

        # Model 3: 4 orbitals
        X3 = np.column_stack([np.ones(n), x2, x1, x_comp, x_comp2])
        beta3 = np.linalg.lstsq(X3, y_arr, rcond=None)[0]
        pred3 = X3 @ beta3

        # Stats
        models = [
            (pred1, f'Koopmans ($\\varepsilon_{{ref}}$)'),
            (pred2, f'Bilin. ($\\varepsilon_H + \\varepsilon_L$)'),
            (pred3, f'4 orbitales'),
        ]

        # --- 3-panel parity plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (pred, title) in zip(axes, models):
            plotted = set()
            for i, mol in enumerate(mol_names):
                label = mol if mol not in plotted else None
                ax.scatter(y_arr[i], pred[i], c=[color_map[mol]],
                           label=label, s=50, alpha=0.7,
                           edgecolors='white', linewidths=0.4, zorder=3)
                plotted.add(mol)

            add_parity_line(ax)
            r2 = linregress(y_arr, pred).rvalue ** 2
            mae = np.mean(np.abs(y_arr - pred))
            rmse = np.sqrt(np.mean((y_arr - pred)**2))

            ax.set_xlabel(f'{prop} DFT (kJ/mol)')
            ax.set_ylabel(f'{prop} prédit (kJ/mol)')
            ax.set_title(f'{title}\n$R^2={r2:.4f}$, MAE={mae:.2f}')
            setup_grid(ax)

        axes[0].legend(loc='lower right', fontsize=5, ncol=2, markerscale=0.7)
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            plt.savefig(bk_dir / f'{fname_parity}.{ext}', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [+] beyond_koopmans/{fname_parity}.png  (n={n})")

    # --- Residuals vs complement orbital ---
    # Only generate once (combined IP and EA)
    valid_ip = [r for r in records
                if _is_valid(r.get('IP_kJ')) and _is_valid(r.get('HOMO_tz2p_eV'))
                and _is_valid(r.get('LUMO_tz2p_eV'))]
    valid_ea = [r for r in records
                if _is_valid(r.get('EA_kJ')) and _is_valid(r.get('LUMO_tz2p_eV'))
                and _is_valid(r.get('HOMO_tz2p_eV'))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if len(valid_ip) >= 5:
        ax = axes[0]
        y_ip = np.array([float(r['IP_kJ']) for r in valid_ip])
        x_homo = np.array([-float(r['HOMO_tz2p_eV']) * EV_TO_KJ for r in valid_ip])
        x_lumo = np.array([float(r['LUMO_tz2p_eV']) * EV_TO_KJ for r in valid_ip])
        fit_ip = linregress(x_homo, y_ip)
        resid_ip = y_ip - (fit_ip.slope * x_homo + fit_ip.intercept)
        r_corr, p_corr = np.corrcoef(x_lumo, resid_ip)[0, 1], 0
        from scipy.stats import pearsonr
        r_corr, p_corr = pearsonr(x_lumo, resid_ip)
        ax.scatter(x_lumo, resid_ip, c=C_RED_DARK, **SCATTER_KW)
        for i, r in enumerate(valid_ip):
            ax.annotate(r['molecule_name'], (x_lumo[i], resid_ip[i]), **ANNOT_KW)
        regression_with_band(ax, x_lumo, resid_ip, color=C_RED_DARK)
        ax.axhline(0, color=C_GRAY, ls='--', lw=0.6)
        ax.set_xlabel('$\\varepsilon_L$ (kJ/mol)')
        ax.set_ylabel('Résidu IP Koopmans (kJ/mol)')
        ax.set_title(f'(a) Résidu IP vs $\\varepsilon_L$\n$r={r_corr:.2f}$, $p={p_corr:.1e}$')
        setup_grid(ax)

    if len(valid_ea) >= 5:
        ax = axes[1]
        y_ea = np.array([abs(float(r['EA_kJ'])) for r in valid_ea])
        x_lumo_ea = np.array([-float(r['LUMO_tz2p_eV']) * EV_TO_KJ for r in valid_ea])
        x_homo_ea = np.array([float(r['HOMO_tz2p_eV']) * EV_TO_KJ for r in valid_ea])
        fit_ea = linregress(x_lumo_ea, y_ea)
        resid_ea = y_ea - (fit_ea.slope * x_lumo_ea + fit_ea.intercept)
        from scipy.stats import pearsonr
        r_corr_ea, p_corr_ea = pearsonr(x_homo_ea, resid_ea)
        ax.scatter(x_homo_ea, resid_ea, c=C_BLUE_DARK, **SCATTER_KW)
        for i, r in enumerate(valid_ea):
            ax.annotate(r['molecule_name'], (x_homo_ea[i], resid_ea[i]), **ANNOT_KW)
        regression_with_band(ax, x_homo_ea, resid_ea, color=C_BLUE_DARK)
        ax.axhline(0, color=C_GRAY, ls='--', lw=0.6)
        ax.set_xlabel('$\\varepsilon_H$ (kJ/mol)')
        ax.set_ylabel('Résidu |EA| Koopmans (kJ/mol)')
        ax.set_title(f'(b) Résidu |EA| vs $\\varepsilon_H$\n$r={r_corr_ea:.2f}$, $p={p_corr_ea:.1e}$')
        setup_grid(ax)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(bk_dir / f'residuals_koopmans_vs_complement.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print("  [+] beyond_koopmans/residuals_koopmans_vs_complement.png")

    # --- MAE comparison bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_data = []
    for prop_label, y_key, orb_keys_list, s in [
        ('IP', 'IP_kJ',
         ['HOMO_tz2p_eV', 'HOMOm1_tz2p_eV', 'LUMO_tz2p_eV', 'LUMOp1_tz2p_eV'], -1),
        ('EA', 'EA_kJ',
         ['LUMO_tz2p_eV', 'LUMOp1_tz2p_eV', 'HOMO_tz2p_eV', 'HOMOm1_tz2p_eV'], -1),
    ]:
        v = [r for r in records if _is_valid(r.get(y_key))
             and all(_is_valid(r.get(k)) for k in orb_keys_list)]
        if len(v) < 5:
            continue
        y = np.array([abs(float(r[y_key])) for r in v])
        x_p = np.array([s * float(r[orb_keys_list[0]]) * EV_TO_KJ for r in v])
        x_s = np.array([s * float(r[orb_keys_list[1]]) * EV_TO_KJ for r in v])
        x_c = np.array([float(r[orb_keys_list[2]]) * EV_TO_KJ for r in v])
        x_c2 = np.array([float(r[orb_keys_list[3]]) * EV_TO_KJ for r in v])
        nn = len(y)

        f1 = linregress(x_p, y)
        mae1 = np.mean(np.abs(y - (f1.slope * x_p + f1.intercept)))

        X2 = np.column_stack([np.ones(nn), x_p, x_c])
        b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
        mae2 = np.mean(np.abs(y - X2 @ b2))

        X3 = np.column_stack([np.ones(nn), x_s, x_p, x_c, x_c2])
        b3 = np.linalg.lstsq(X3, y, rcond=None)[0]
        mae3 = np.mean(np.abs(y - X3 @ b3))

        bar_data.append((prop_label, [mae1, mae2, mae3]))

    if bar_data:
        x_pos = np.arange(len(bar_data))
        w = 0.25
        model_labels = ['Koopmans', 'Bilinéaire', '4 orbitales']
        colors = [C_BLUE_DARK, C_RED_DARK, C_GREEN]
        for i, (lbl, c) in enumerate(zip(model_labels, colors)):
            vals = [d[1][i] for d in bar_data]
            ax.bar(x_pos + i * w, vals, w, label=lbl, color=c, alpha=0.85)

        ax.set_xticks(x_pos + w)
        ax.set_xticklabels([d[0] for d in bar_data])
        ax.set_ylabel('MAE (kJ/mol)')
        ax.legend()
        setup_grid(ax, limit_ticks=False)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(bk_dir / f'MAE_comparison_bar.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print("  [+] beyond_koopmans/MAE_comparison_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FIGURE GENERATION - RAPPORT")
    print("=" * 60)
    print(f"Data:     {DATA_DIR}")
    print(f"Optional: {OPTIONAL_DATA_DIR}")
    print(f"Output:   {OUTPUT_DIR}")
    print()

    # ── Load redox data (always available) ──
    per_mol, per_conf = load_all_redox_data()
    print(f"Molecules loaded: {len(per_mol)} (catechols)")
    print(f"Total conformers: {len(per_conf)}")
    print()

    # ── Redox-based figures (always generated) ──
    print("Generating redox figures...")
    plot_decomposition_bars(per_mol)
    plot_koopmans(per_mol, per_conf)
    plot_beyond_koopmans(per_conf)
    plot_error_vs_rmsd(per_mol)
    plot_erelax_comparison_bars(per_mol)
    plot_thermodynamic_models(per_mol)
    print()

    # ── Politzer figures (optional) ──
    # Filter to stable molecules only (those in per_mol)
    all_mol_names = set(per_mol.keys())
    pol_mol, pol_conf = load_politzer_data()
    if pol_mol:
        pol_mol = [r for r in pol_mol if r['molecule_name'] in all_mol_names]
        if pol_conf:
            pol_conf = [r for r in pol_conf if r['molecule_name'] in all_mol_names]
        print("Generating Politzer figures...")
        plot_politzer_parity(pol_mol)
        if pol_conf:
            plot_politzer_per_conformer(pol_conf)
        print()
    else:
        print("[!] Politzer data not found in rapport/data/")
        print("    Expected: politzer_per_molecule.csv, politzer_per_conformer.csv")
        print("    Generate with: python3 -m analysis.generate_ip_ea_comparison")
        print()

    # ── LVC figures (optional) ──
    lvc_data = load_lvc_data()
    if lvc_data:
        print("Generating LVC figures...")
        plot_lvc_vs_real(lvc_data)
        plot_lvc_vs_lvc2(lvc_data)
        plot_lvc2_corrected(lvc_data)
        print()
    else:
        print("[!] LVC data not found in rapport/data/")
        print("    Expected: lvc_summary.csv")
        print("    Generate with: python3 -m analysis.generate_lvc_edef_report")
        print()

    print("=" * 60)
    print(f"All figures saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
