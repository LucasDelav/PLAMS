"""
Shared plotting style for all report figures.

Provides a unified color palette, rcParams, and helper functions
to ensure visual consistency across all figures in the rapport.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.stats import linregress, t as t_dist

# ── Color palette (divergent red/blue) ──────────────────────────────────────
C_RED_DARK = '#b2182b'
C_RED_LIGHT = '#d6604d'
C_BLUE_DARK = '#2166ac'
C_BLUE_LIGHT = '#4393c3'
C_GREEN = '#1b7837'
C_GRAY = '#636363'

# ── Standard figure sizes (inches, matched to LaTeX textwidth ~6.5in) ──────
FIGSIZE_SINGLE = (3.5, 3.0)       # 1 panel (scatter, parity)
FIGSIZE_DOUBLE_H = (7.0, 3.5)     # 2 panels side by side
FIGSIZE_DOUBLE_V = (7.0, 5.5)     # 2 panels stacked / per-conformer
FIGSIZE_WIDE = (7.0, 3.0)         # wide single panel (bars)

# ── Stats box style ─────────────────────────────────────────────────────────
STATS_BOX = dict(
    boxstyle='round,pad=0.4', facecolor='white', alpha=0.85,
    edgecolor='gray', linewidth=0.6,
)

# ── Standard scatter kwargs ─────────────────────────────────────────────────
SCATTER_KW = dict(s=50, edgecolors='white', linewidths=0.4, zorder=3, alpha=0.85)
ANNOT_KW = dict(fontsize=9, color=C_GRAY, textcoords='offset points', xytext=(4, 4))


def apply_style():
    """Apply global rcParams for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 15,
        'axes.labelsize': 17,
        'axes.titlesize': 17,
        'axes.linewidth': 0.8,
        'legend.fontsize': 12,
        'legend.framealpha': 0.85,
        'legend.edgecolor': 'gray',
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def setup_grid(ax, limit_ticks=True):
    """Add subtle grid and minor ticks, limit major ticks to avoid crowding."""
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    if limit_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def get_color_map(names):
    """Return a deterministic color mapping for molecule names (tab20)."""
    unique = sorted(set(names))
    cmap = plt.cm.get_cmap('tab20', max(len(unique), 20))
    return {n: cmap(i) for i, n in enumerate(unique)}


def add_parity_line(ax):
    """Draw a thin gray dashed parity line y=x."""
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, '--', color=C_GRAY, alpha=0.5, lw=0.8, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def regression_with_band(ax, x, y, color=C_RED_DARK, label=None):
    """Plot regression line with 95% confidence band.

    Returns:
        slope, intercept, r2, mae, n
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    slope, intercept, r, p, se = linregress(x, y)
    r2 = r ** 2
    y_pred = slope * x + intercept
    mae = np.mean(np.abs(y - y_pred))

    x_fit = np.linspace(x.min() - 0.02 * np.ptp(x),
                        x.max() + 0.02 * np.ptp(x), 200)
    y_fit = slope * x_fit + intercept
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)
    t_val = t_dist.ppf(0.975, n - 2)
    ci = t_val * s_err * np.sqrt(1.0 / n + (x_fit - x_mean) ** 2 / ss_x)

    ax.plot(x_fit, y_fit, '-', color=color, lw=1.5, alpha=0.8, label=label)
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.15)

    return slope, intercept, r2, mae, n


def stats_text(ax, r2, mae, n, unit='kJ/mol', pos='upper left', adjusted=False):
    """Place a stats box on the axes."""
    if adjusted:
        text = f"$R^2_{{\\mathrm{{adj}}}} = {r2:.3f}$\nMAE = {mae:.1f} {unit}\n$n = {n}$"
    else:
        text = f"$R^2 = {r2:.3f}$\nMAE = {mae:.1f} {unit}\n$n = {n}$"
    if pos == 'upper left':
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top',
                fontsize=14, bbox=STATS_BOX, family='serif')
    else:
        ax.text(0.95, 0.05, text, transform=ax.transAxes, va='bottom',
                ha='right', fontsize=14, bbox=STATS_BOX, family='serif')


def calibrated_parity(ax, x_dft, y_raw, labels, color, method_label):
    """Calibrated parity plot: fit y_raw -> x_dft, scatter corrected values.

    Returns:
        dict with R2, MAE, RMSE, n, cal_slope, cal_intercept
    """
    x_dft = np.asarray(x_dft, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    n = len(x_dft)
    if n < 2:
        return {}

    slope_cal, intercept_cal, r_val, p_val, _ = linregress(y_raw, x_dft)
    y_cal = slope_cal * y_raw + intercept_cal
    r2 = r_val ** 2
    mae = np.mean(np.abs(y_cal - x_dft))
    rmse = np.sqrt(np.mean((y_cal - x_dft) ** 2))

    all_vals = np.concatenate([x_dft, y_cal])
    vmin, vmax = np.min(all_vals) - 10, np.max(all_vals) + 10
    ax.plot([vmin, vmax], [vmin, vmax], '--', color=C_GRAY, lw=0.8,
            alpha=0.5, zorder=0)

    ax.scatter(x_dft, y_cal, c=color, **SCATTER_KW)
    for i, mol in enumerate(labels):
        ax.annotate(mol, (x_dft[i], y_cal[i]), **ANNOT_KW)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    stats_text(ax, r2, mae, n,
               pos='upper left' if slope_cal > 0 else 'lower right')

    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'n': n,
            'cal_slope': slope_cal, 'cal_intercept': intercept_cal}


