#!/usr/bin/env amspython
"""
Comparaison Screening (GFN1-xTB geom opt + PBE0/TZ2P SP) vs Full DFT (PBE0/DZP).

Quantifie le compromis précision/vitesse du workflow screening par rapport au
pipeline DFT complet sur 15 molécules. Génère parity plots, statistiques et
CSV de comparaison.

Sortie → results/screening_comparison/

Usage:
    amspython analysis/compare_screening_vs_dft.py
    amspython analysis/compare_screening_vs_dft.py --screening-root /path/to/root
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from analysis import (
    PROJECT_ROOT, BATCH_DIR, RESULTS_DIR,
    MOLECULES_15, DEA_EXCLUDED,
)
from common.constants import EH_TO_EV, EV_TO_KJ
from common.utils import print_header, print_section, print_success, print_error


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_csv(path):
    """Lit un CSV et retourne une liste de dicts avec conversion numérique."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_screening_data(screening_root, molecules):
    """Charge les données screening (summary + conformers) pour chaque molécule.

    Returns:
        dict[str, dict] avec clés EA_kJ, IP_kJ, Erelax_red_kJ, Erelax_ox_kJ,
        HOMO_eV, LUMO_eV (déjà en eV).
    """
    data = {}
    for mol in molecules:
        summary_path = screening_root / f"{mol}_workdir" / "screening" / "results" / f"{mol}_screening_summary.csv"
        if not summary_path.exists():
            print_error(f"Screening summary introuvable : {summary_path}")
            continue
        rows = read_csv(summary_path)
        if not rows:
            continue
        r = rows[0]
        data[mol] = {
            'EA_kJ': r['EA_kJ'],
            'IP_kJ': r['IP_kJ'],
            'Erelax_red_kJ': r['Erelax_red_kJ'],
            'Erelax_ox_kJ': r['Erelax_ox_kJ'],
            'HOMO_eV': r['HOMO_eV'],
            'LUMO_eV': r['LUMO_eV'],
        }
    return data


def load_full_dft_data(batch_dir, molecules):
    """Charge les données DFT complètes (summary + conformers) pour chaque molécule.

    Returns:
        dict[str, dict] avec clés EA_kJ, IP_kJ, Edef_red_kJ, Edef_ox_kJ,
        HOMO_eV, LUMO_eV.
    """
    data = {}
    for mol in molecules:
        base = batch_dir / f"{mol}_workdir" / "redox" / "results"
        summary_path = base / f"{mol}_redox_summary.csv"
        conf_path = base / f"{mol}_redox_conformers.csv"

        if not summary_path.exists():
            print_error(f"DFT summary introuvable : {summary_path}")
            continue

        # Summary : EA_V, IP_V sont en convention électrochimique (négatifs)
        # Screening utilise la convention positive (EA = Ee_n - Ee_a > 0)
        # → il faut prendre la valeur absolue pour comparer
        sr = read_csv(summary_path)[0]
        ea_kj = abs(sr['EA_V']) * EV_TO_KJ
        ip_kj = abs(sr['IP_V']) * EV_TO_KJ
        homo_ev = sr['HOMO_eV'] * EH_TO_EV   # Hartree → eV
        lumo_ev = sr['LUMO_eV'] * EH_TO_EV

        # Edef : moyenne de Boltzmann depuis conformers CSV
        edef_red = np.nan
        edef_ox = np.nan
        if conf_path.exists():
            conf_rows = read_csv(conf_path)
            if conf_rows:
                weights = np.array([r['weight'] for r in conf_rows])
                edef_r = np.array([r['Edef_red_kJ'] for r in conf_rows])
                edef_o = np.array([r['Edef_ox_kJ'] for r in conf_rows])
                w_sum = weights.sum()
                if w_sum > 0:
                    edef_red = np.sum(weights * edef_r) / w_sum
                    edef_ox = np.sum(weights * edef_o) / w_sum

        data[mol] = {
            'EA_kJ': ea_kj,
            'IP_kJ': ip_kj,
            'Edef_red_kJ': edef_red,
            'Edef_ox_kJ': edef_ox,
            'HOMO_eV': homo_ev,
            'LUMO_eV': lumo_ev,
        }
    return data


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(screening, dft):
    """Construit la table de comparaison pour les molécules communes."""
    table = []
    common = sorted(set(screening.keys()) & set(dft.keys()))
    for mol in common:
        s = screening[mol]
        d = dft[mol]
        row = {
            'molecule': mol,
            'is_DEA': mol in DEA_EXCLUDED,
            'EA_scr': s['EA_kJ'],
            'EA_dft': d['EA_kJ'],
            'IP_scr': s['IP_kJ'],
            'IP_dft': d['IP_kJ'],
            'Erelax_red_scr': s['Erelax_red_kJ'],
            'Edef_red_dft': d['Edef_red_kJ'],
            'Erelax_ox_scr': s['Erelax_ox_kJ'],
            'Edef_ox_dft': d['Edef_ox_kJ'],
            'HOMO_scr': s['HOMO_eV'],
            'HOMO_dft': d['HOMO_eV'],
            'LUMO_scr': s['LUMO_eV'],
            'LUMO_dft': d['LUMO_eV'],
        }
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(x, y):
    """Calcule R², MAE, RMSE, biais, pente, intercept entre deux vecteurs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {'n': n, 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                'bias': np.nan, 'max_err': np.nan, 'slope': np.nan, 'intercept': np.nan}
    slope, intercept, r_val, p_val, _ = linregress(x, y)
    residuals = y - x
    return {
        'n': n,
        'R2': r_val ** 2,
        'MAE': np.mean(np.abs(residuals)),
        'RMSE': np.sqrt(np.mean(residuals ** 2)),
        'bias': np.mean(residuals),
        'max_err': np.max(np.abs(residuals)),
        'slope': slope,
        'intercept': intercept,
        'p_value': p_val,
    }


def compute_all_stats(table):
    """Calcule les stats pour 6 propriétés × 3 sous-ensembles."""
    properties = [
        ('EA', 'EA_scr', 'EA_dft', 'kJ/mol'),
        ('IP', 'IP_scr', 'IP_dft', 'kJ/mol'),
        ('Erelax_red', 'Erelax_red_scr', 'Edef_red_dft', 'kJ/mol'),
        ('Erelax_ox', 'Erelax_ox_scr', 'Edef_ox_dft', 'kJ/mol'),
        ('HOMO', 'HOMO_scr', 'HOMO_dft', 'eV'),
        ('LUMO', 'LUMO_scr', 'LUMO_dft', 'eV'),
    ]
    subsets = [
        ('all_15', lambda r: True),
        ('non_DEA_11', lambda r: not r['is_DEA']),
        ('DEA_4', lambda r: r['is_DEA']),
    ]

    results = {}
    for prop_name, col_scr, col_dft, unit in properties:
        results[prop_name] = {'unit': unit}
        for subset_name, filt in subsets:
            rows = [r for r in table if filt(r)]
            x = [r[col_scr] for r in rows]
            y = [r[col_dft] for r in rows]
            results[prop_name][subset_name] = compute_stats(x, y)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _parity_scatter(ax, table, col_scr, col_dft, unit, title,
                    color_normal='#2980b9', color_dea='#e74c3c'):
    """Scatter parity plot avec DEA en rouge, annotations, et régression."""
    x_all, y_all = [], []
    for row in table:
        xs, xd = row[col_scr], row[col_dft]
        if not (np.isfinite(xs) and np.isfinite(xd)):
            continue
        c = color_dea if row['is_DEA'] else color_normal
        ax.scatter(xs, xd, s=80, color=c, edgecolors='white', linewidths=0.5, zorder=3)
        ax.annotate(row['molecule'], (xs, xd), xytext=(5, 5),
                    textcoords='offset points', fontsize=6, alpha=0.8)
        x_all.append(xs)
        y_all.append(xd)

    if len(x_all) < 2:
        return

    x_all, y_all = np.array(x_all), np.array(y_all)
    vmin = min(x_all.min(), y_all.min())
    vmax = max(x_all.max(), y_all.max())
    margin = 0.08 * (vmax - vmin)
    vmin -= margin
    vmax += margin

    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, alpha=0.5)

    # Régression non-DEA
    non_dea = [r for r in table
               if not r['is_DEA'] and np.isfinite(r[col_scr]) and np.isfinite(r[col_dft])]
    if len(non_dea) >= 3:
        xn = np.array([r[col_scr] for r in non_dea])
        yn = np.array([r[col_dft] for r in non_dea])
        sl, ic, rv, pv, _ = linregress(xn, yn)
        residuals = yn - xn
        r2 = rv ** 2
        mae = np.mean(np.abs(residuals))
        x_fit = np.linspace(vmin, vmax, 100)
        ax.plot(x_fit, sl * x_fit + ic, '-', color=color_normal, linewidth=1.5, alpha=0.7)
        ax.text(0.05, 0.95,
                f'non-DEA (n={len(xn)}):\n$R^2$={r2:.4f}\nMAE={mae:.2f} {unit}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)


def plot_ea_parity(table, output_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    _parity_scatter(ax, table, 'EA_scr', 'EA_dft', 'kJ/mol',
                    'EA : Screening vs Full DFT')
    ax.set_xlabel('EA screening (kJ/mol)', fontsize=11)
    ax.set_ylabel('EA full DFT (kJ/mol)', fontsize=11)
    out = Path(output_dir) / 'ea_parity.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print_success(f"  {out.name}")


def plot_ip_parity(table, output_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    _parity_scatter(ax, table, 'IP_scr', 'IP_dft', 'kJ/mol',
                    'IP : Screening vs Full DFT')
    ax.set_xlabel('IP screening (kJ/mol)', fontsize=11)
    ax.set_ylabel('IP full DFT (kJ/mol)', fontsize=11)
    out = Path(output_dir) / 'ip_parity.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print_success(f"  {out.name}")


def plot_erelax_edef_parity(table, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    _parity_scatter(axes[0], table, 'Erelax_red_scr', 'Edef_red_dft', 'kJ/mol',
                    '(a) Réduction : $E_{relax}$ vs $E_{def}$',
                    color_normal='#2980b9', color_dea='#e74c3c')
    axes[0].set_xlabel('$E_{relax}$ screening (kJ/mol)', fontsize=11)
    axes[0].set_ylabel('$E_{def}$ full DFT (kJ/mol)', fontsize=11)

    _parity_scatter(axes[1], table, 'Erelax_ox_scr', 'Edef_ox_dft', 'kJ/mol',
                    '(b) Oxydation : $E_{relax}$ vs $E_{def}$',
                    color_normal='#e67e22', color_dea='#e74c3c')
    axes[1].set_xlabel('$E_{relax}$ screening (kJ/mol)', fontsize=11)
    axes[1].set_ylabel('$E_{def}$ full DFT (kJ/mol)', fontsize=11)

    out = Path(output_dir) / 'erelax_edef_parity.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print_success(f"  {out.name}")


def plot_homo_lumo_parity(table, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    _parity_scatter(axes[0], table, 'HOMO_scr', 'HOMO_dft', 'eV',
                    '(a) HOMO : Screening vs Full DFT',
                    color_normal='#8e44ad', color_dea='#e74c3c')
    axes[0].set_xlabel('HOMO screening (eV)', fontsize=11)
    axes[0].set_ylabel('HOMO full DFT (eV)', fontsize=11)

    _parity_scatter(axes[1], table, 'LUMO_scr', 'LUMO_dft', 'eV',
                    '(b) LUMO : Screening vs Full DFT',
                    color_normal='#27ae60', color_dea='#e74c3c')
    axes[1].set_xlabel('LUMO screening (eV)', fontsize=11)
    axes[1].set_ylabel('LUMO full DFT (eV)', fontsize=11)

    out = Path(output_dir) / 'homo_lumo_parity.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print_success(f"  {out.name}")


def plot_metrics_summary(all_stats, output_dir):
    """Bar chart R², MAE, RMSE par propriété (non-DEA seulement)."""
    props = ['EA', 'IP', 'Erelax_red', 'Erelax_ox', 'HOMO', 'LUMO']
    prop_labels = ['EA', 'IP', '$E_{relax}$ red', '$E_{relax}$ ox', 'HOMO', 'LUMO']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['R2', 'MAE', 'RMSE']
    metric_labels = ['$R^2$', 'MAE', 'RMSE']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    x = np.arange(len(props))
    width = 0.6

    for i, (metric, mlabel, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = []
        for p in props:
            s = all_stats[p]['non_DEA_11']
            vals.append(s.get(metric, 0) if not np.isnan(s.get(metric, np.nan)) else 0)

        bars = axes[i].bar(x, vals, width, color=color, alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, vals):
            fmt = f'{val:.4f}' if metric == 'R2' else f'{val:.2f}'
            axes[i].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                         fmt, ha='center', va='bottom', fontsize=8)

        unit_str = ''
        if metric != 'R2':
            # Mixed units — annotate individually
            unit_str = ' (kJ/mol or eV)'
        axes[i].set_ylabel(f'{mlabel}{unit_str}', fontsize=11)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(prop_labels, rotation=30, ha='right', fontsize=9)
        axes[i].grid(True, alpha=0.3, axis='y')
        if metric == 'R2':
            axes[i].set_ylim(0, 1.1)

    fig.suptitle('Screening vs Full DFT — Métriques (n=11, non-DEA)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path(output_dir) / 'metrics_summary.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print_success(f"  {out.name}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_summary_text(all_stats, table, output_dir):
    """Écrit le rapport texte avec statistiques complètes."""
    out = Path(output_dir) / 'screening_vs_dft_statistics.txt'
    lines = []
    lines.append("=" * 72)
    lines.append("COMPARAISON SCREENING vs FULL DFT")
    lines.append("Screening : GFN1-xTB geom opt + PBE0/TZ2P single-point")
    lines.append("Full DFT  : PBE0/DZP full optimization + fréquences")
    lines.append("=" * 72)
    lines.append("")

    n_total = len(table)
    n_dea = sum(1 for r in table if r['is_DEA'])
    lines.append(f"Molécules : {n_total} total, {n_total - n_dea} non-DEA, {n_dea} DEA")
    lines.append(f"DEA exclues : {', '.join(sorted(DEA_EXCLUDED))}")
    lines.append("")

    for prop_name in ['EA', 'IP', 'Erelax_red', 'Erelax_ox', 'HOMO', 'LUMO']:
        pdata = all_stats[prop_name]
        unit = pdata['unit']
        lines.append("-" * 72)
        lines.append(f"  {prop_name} [{unit}]")
        lines.append("-" * 72)

        for subset_name in ['all_15', 'non_DEA_11', 'DEA_4']:
            s = pdata[subset_name]
            label = {'all_15': 'Toutes (15)', 'non_DEA_11': 'Non-DEA (11)',
                     'DEA_4': 'DEA (4)'}[subset_name]
            if np.isnan(s.get('R2', np.nan)):
                lines.append(f"  {label:20s} : n={s['n']} — données insuffisantes")
                continue
            lines.append(
                f"  {label:20s} : n={s['n']:2d}  "
                f"R²={s['R2']:.4f}  MAE={s['MAE']:.2f}  RMSE={s['RMSE']:.2f}  "
                f"biais={s['bias']:+.2f}  max_err={s['max_err']:.2f} {unit}"
            )
            lines.append(
                f"  {'':20s}   pente={s['slope']:.4f}  intercept={s['intercept']:.2f}"
            )
        lines.append("")

    # Per-molecule table
    lines.append("=" * 72)
    lines.append("DÉTAIL PAR MOLÉCULE")
    lines.append("=" * 72)
    header = (f"{'Molécule':12s} {'DEA':>4s}  "
              f"{'EA_scr':>8s} {'EA_dft':>8s} {'ΔEA':>7s}  "
              f"{'IP_scr':>8s} {'IP_dft':>8s} {'ΔIP':>7s}  "
              f"{'Er_r_scr':>8s} {'Ed_r_dft':>8s} {'Δ':>7s}  "
              f"{'Er_o_scr':>8s} {'Ed_o_dft':>8s} {'Δ':>7s}")
    lines.append(header)
    lines.append("-" * len(header))
    for row in sorted(table, key=lambda r: r['molecule']):
        dea = 'Yes' if row['is_DEA'] else ''
        def fmt(v):
            return f'{v:8.2f}' if np.isfinite(v) else f'{"N/A":>8s}'
        def delta(a, b):
            d = b - a
            return f'{d:+7.2f}' if np.isfinite(d) else f'{"N/A":>7s}'
        lines.append(
            f"{row['molecule']:12s} {dea:>4s}  "
            f"{fmt(row['EA_scr'])} {fmt(row['EA_dft'])} {delta(row['EA_scr'], row['EA_dft'])}  "
            f"{fmt(row['IP_scr'])} {fmt(row['IP_dft'])} {delta(row['IP_scr'], row['IP_dft'])}  "
            f"{fmt(row['Erelax_red_scr'])} {fmt(row['Edef_red_dft'])} {delta(row['Erelax_red_scr'], row['Edef_red_dft'])}  "
            f"{fmt(row['Erelax_ox_scr'])} {fmt(row['Edef_ox_dft'])} {delta(row['Erelax_ox_scr'], row['Edef_ox_dft'])}"
        )

    lines.append("")
    lines.append("=" * 72)
    lines.append("CONCLUSION")
    lines.append("=" * 72)

    # Auto-generate conclusion from stats
    ea_r2 = all_stats['EA']['non_DEA_11'].get('R2', 0)
    ip_r2 = all_stats['IP']['non_DEA_11'].get('R2', 0)
    er_red_r2 = all_stats['Erelax_red']['non_DEA_11'].get('R2', 0)
    er_ox_r2 = all_stats['Erelax_ox']['non_DEA_11'].get('R2', 0)
    homo_r2 = all_stats['HOMO']['non_DEA_11'].get('R2', 0)

    lines.append(f"  EA  R²={ea_r2:.4f} — {'Excellent' if ea_r2 > 0.99 else 'Bon' if ea_r2 > 0.95 else 'Modéré'}")
    lines.append(f"  IP  R²={ip_r2:.4f} — {'Excellent' if ip_r2 > 0.99 else 'Bon' if ip_r2 > 0.95 else 'Modéré'}")
    lines.append(f"  Erelax_red R²={er_red_r2:.4f} — {'Excellent' if er_red_r2 > 0.99 else 'Bon' if er_red_r2 > 0.95 else 'Modéré' if er_red_r2 > 0.8 else 'Faible'}")
    lines.append(f"  Erelax_ox  R²={er_ox_r2:.4f} — {'Excellent' if er_ox_r2 > 0.99 else 'Bon' if er_ox_r2 > 0.95 else 'Modéré' if er_ox_r2 > 0.8 else 'Faible'}")
    lines.append(f"  HOMO R²={homo_r2:.4f} — {'Excellent' if homo_r2 > 0.99 else 'Bon' if homo_r2 > 0.95 else 'Modéré'}")
    lines.append("")
    ea_mae = all_stats['EA']['non_DEA_11'].get('MAE', 0)
    ip_mae = all_stats['IP']['non_DEA_11'].get('MAE', 0)
    lines.append(f"  Le screening reproduit EA (MAE={ea_mae:.1f} kJ/mol) et IP (MAE={ip_mae:.1f} kJ/mol)")
    lines.append(f"  avec une précision compatible avec un pré-screening rapide.")
    er_red_mae = all_stats['Erelax_red']['non_DEA_11'].get('MAE', 0)
    lines.append(f"  Erelax/Edef (MAE_red={er_red_mae:.1f} kJ/mol) montre les écarts attendus")
    lines.append(f"  entre relaxation DFTB et DFT complète.")
    lines.append("")

    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print_success(f"  {out.name}")


def write_comparison_csv(table, output_dir):
    """Écrit le CSV de comparaison par molécule."""
    out = Path(output_dir) / 'screening_vs_dft_comparison.csv'
    fieldnames = [
        'molecule', 'is_DEA',
        'EA_scr_kJ', 'EA_dft_kJ', 'dEA_kJ',
        'IP_scr_kJ', 'IP_dft_kJ', 'dIP_kJ',
        'Erelax_red_scr_kJ', 'Edef_red_dft_kJ', 'dErelax_red_kJ',
        'Erelax_ox_scr_kJ', 'Edef_ox_dft_kJ', 'dErelax_ox_kJ',
        'HOMO_scr_eV', 'HOMO_dft_eV', 'dHOMO_eV',
        'LUMO_scr_eV', 'LUMO_dft_eV', 'dLUMO_eV',
    ]
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(table, key=lambda r: r['molecule']):
            writer.writerow({
                'molecule': row['molecule'],
                'is_DEA': row['is_DEA'],
                'EA_scr_kJ': f"{row['EA_scr']:.4f}",
                'EA_dft_kJ': f"{row['EA_dft']:.4f}",
                'dEA_kJ': f"{row['EA_dft'] - row['EA_scr']:.4f}",
                'IP_scr_kJ': f"{row['IP_scr']:.4f}",
                'IP_dft_kJ': f"{row['IP_dft']:.4f}",
                'dIP_kJ': f"{row['IP_dft'] - row['IP_scr']:.4f}",
                'Erelax_red_scr_kJ': f"{row['Erelax_red_scr']:.4f}",
                'Edef_red_dft_kJ': f"{row['Edef_red_dft']:.4f}",
                'dErelax_red_kJ': f"{row['Edef_red_dft'] - row['Erelax_red_scr']:.4f}",
                'Erelax_ox_scr_kJ': f"{row['Erelax_ox_scr']:.4f}",
                'Edef_ox_dft_kJ': f"{row['Edef_ox_dft']:.4f}",
                'dErelax_ox_kJ': f"{row['Edef_ox_dft'] - row['Erelax_ox_scr']:.4f}",
                'HOMO_scr_eV': f"{row['HOMO_scr']:.6f}",
                'HOMO_dft_eV': f"{row['HOMO_dft']:.6f}",
                'dHOMO_eV': f"{row['HOMO_dft'] - row['HOMO_scr']:.6f}",
                'LUMO_scr_eV': f"{row['LUMO_scr']:.6f}",
                'LUMO_dft_eV': f"{row['LUMO_dft']:.6f}",
                'dLUMO_eV': f"{row['LUMO_dft'] - row['LUMO_scr']:.6f}",
            })
    print_success(f"  {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comparaison Screening vs Full DFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--screening-root', type=str,
                        default=str(PROJECT_ROOT),
                        help="Racine contenant les <mol>_workdir/screening/ (défaut: PROJECT_ROOT)")
    parser.add_argument('--batch-dir', type=str,
                        default=str(BATCH_DIR),
                        help="Répertoire contenant les <mol>_workdir/redox/ DFT (défaut: BATCH_DIR)")
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'screening_comparison'),
                        help="Répertoire de sortie")
    parser.add_argument('--molecules', nargs='+', default=None,
                        help="Liste de molécules (défaut: MOLECULES_15)")

    args = parser.parse_args()
    screening_root = Path(args.screening_root)
    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    molecules = args.molecules or MOLECULES_15

    print_header("COMPARAISON SCREENING vs FULL DFT")

    # 1. Charger les données
    print_section("Chargement des données")
    screening = load_screening_data(screening_root, molecules)
    print_success(f"Screening : {len(screening)} molécules chargées")
    dft = load_full_dft_data(batch_dir, molecules)
    print_success(f"Full DFT  : {len(dft)} molécules chargées")

    # 2. Construire la table de comparaison
    table = build_comparison_table(screening, dft)
    n_common = len(table)
    n_dea = sum(1 for r in table if r['is_DEA'])
    print_success(f"Comparaison : {n_common} molécules communes ({n_common - n_dea} non-DEA, {n_dea} DEA)")

    if n_common == 0:
        print_error("Aucune molécule commune trouvée !")
        sys.exit(1)

    # 3. Statistiques
    print_section("Statistiques")
    all_stats = compute_all_stats(table)
    for prop in ['EA', 'IP', 'Erelax_red', 'Erelax_ox', 'HOMO', 'LUMO']:
        s = all_stats[prop]['non_DEA_11']
        unit = all_stats[prop]['unit']
        if np.isfinite(s.get('R2', np.nan)):
            print(f"  {prop:12s} (non-DEA) : R²={s['R2']:.4f}  MAE={s['MAE']:.2f} {unit}")
        else:
            print(f"  {prop:12s} (non-DEA) : données insuffisantes")

    # 4. Graphiques
    print_section("Graphiques")
    plot_ea_parity(table, output_dir)
    plot_ip_parity(table, output_dir)
    plot_erelax_edef_parity(table, output_dir)
    plot_homo_lumo_parity(table, output_dir)
    plot_metrics_summary(all_stats, output_dir)

    # 5. Fichiers de sortie
    print_section("Fichiers de sortie")
    write_comparison_csv(table, output_dir)
    write_summary_text(all_stats, table, output_dir)

    print_success(f"\nTerminé — résultats dans {output_dir}")


if __name__ == '__main__':
    main()
