#!/usr/bin/env python3
"""Generate figures for sections 3.3 (ΔΔU vs η) and 3.4 (TΔS parity).

Uses rapport/style.py for visual consistency with sections 3.1-3.2.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from style import (
    apply_style, setup_grid, regression_with_band, stats_text, add_parity_line,
    SCATTER_KW, ANNOT_KW, STATS_BOX,
    C_RED_DARK, C_BLUE_DARK, C_GREEN, C_GRAY,
    FIGSIZE_DOUBLE_V,
)
import matplotlib.pyplot as plt

apply_style()

OUTPUT_DIR = Path(__file__).parent / 'figures' / 'thermodynamic_models'
DATA_DIR = Path(__file__).parent / 'data'
PREDICT_DIR = Path(__file__).parent.parent / 'predict_dg_results'
BASE = Path(__file__).parent.parent

DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}
RAPPORT15 = ['Al-CA', 'CA', 'Cl-DA', 'Cl-HMC', 'DA', 'DHI', 'F-DA', 'F-HMC',
             'H2C-CA', 'HMC', 'HO-CA', 'L-DA', 'O-DHI', 'P-DHI', 'S-DHI']


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    import glob
    # Redox decomposition (Boltzmann-weighted)
    files = glob.glob(str(BASE / 'data' / 'Acetonitrile' / '*' / 'redox' / 'results' / '*_redox_conformers.csv'))
    rows = []
    for f in sorted(files):
        mol = Path(f).stem.replace('_redox_conformers', '')
        df = pd.read_csv(f)
        w = df['weight'].values; w = w / w.sum()
        row = {'molecule': mol}
        for col in ['ddU_red_kJ', 'ddU_ox_kJ', 'TdS_red_kJ', 'TdS_ox_kJ', 'EA_kJ', 'IP_kJ']:
            if col in df.columns:
                row[col] = float(np.sum(w * df[col].values))
        rows.append(row)
    redox = pd.DataFrame(rows)
    data = redox[redox['molecule'].isin(RAPPORT15)].copy()
    data['eta_vert'] = (data['IP_kJ'] - data['EA_kJ']) / 2
    data['mu_vert'] = -(data['IP_kJ'] + data['EA_kJ']) / 2

    # predict_dg for bader_sumO
    all_cols = pd.read_csv(PREDICT_DIR / 'predict_dg_conformers.csv', nrows=0).columns.tolist()
    keep = ['molecule', 'conformer', 'weight', 'bader_sumO_n']
    keep = [c for c in keep if c in all_cols]
    pdg = pd.read_csv(PREDICT_DIR / 'predict_dg_conformers.csv', usecols=keep)
    pdg = pdg[~pdg['molecule'].isin(DEA_EXCLUDED)].copy()

    def bw_avg(g):
        w = g['weight'].values; w = w / w.sum()
        result = {}
        for c in ['bader_sumO_n']:
            if c in g.columns:
                vals = g[c].values
                if np.all(np.isfinite(vals)):
                    result[c] = float(np.sum(w * vals))
                else:
                    result[c] = np.nan
        return pd.Series(result)

    bw = pdg.groupby('molecule').apply(bw_avg).reset_index()
    data = data.merge(bw, on='molecule', how='left')
    data = data.sort_values('molecule').reset_index(drop=True)
    return data


# ============================================================
# FIGURE 1: η vs ΔΔU — 2 individual panels (assembled by LaTeX)
# ============================================================

def plot_ddU_vs_eta(data):
    configs = [
        ('ddU_eta_ox', 'ddU_ox_kJ', C_RED_DARK,
         r'$\Delta\Delta U_{\mathrm{ox}}$ (kJ/mol)', '(a) Oxydation'),
        ('ddU_eta_red', 'ddU_red_kJ', C_BLUE_DARK,
         r'$\Delta\Delta U_{\mathrm{red}}$ (kJ/mol)', '(b) Réduction'),
    ]

    for fname, target_col, color, ylabel, subtitle in configs:
        x = data['eta_vert'].values
        y = data[target_col].values
        names = data['molecule'].values

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)

        # Regression with confidence band
        slope, intercept, r2, mae, n = regression_with_band(ax, x, y, color=color)

        # Constant model
        cst = np.mean(y)
        mae_cst = np.mean(np.abs(y - cst))
        ax.axhline(cst, color=C_GREEN, linestyle='--', alpha=0.6, lw=1.2, zorder=1,
                   label=f'Constant: MAE = {mae_cst:.2f}')

        # Scatter points (on top)
        ax.scatter(x, y, c=color, **SCATTER_KW)
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), **ANNOT_KW)

        # Stats box
        _, _, r_val, p_val, _ = linregress(x, y)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2)
        stats_text(ax, r2_adj, mae, n, pos='upper left', adjusted=True)

        ax.set_xlabel(r'$\eta = (\mathrm{IP} - \mathrm{EA})/2$ (kJ/mol)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=12, loc='lower left')
        setup_grid(ax)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{fname}.png')
        plt.close()
        print(f'  [+] {fname}.png  (R2_adj={r2_adj:.3f}, MAE={mae:.2f}, p={p_val:.1e})')


# ============================================================
# FIGURE 2: TΔS parity plots — 2 individual panels
# ============================================================

def plot_TdS_parity(data):
    configs = [
        ('TdS_bader_mu_ox', 'TdS_ox_kJ', C_RED_DARK,
         r'$T\Delta S_{\mathrm{ox}}$', '(a) Oxydation'),
        ('TdS_bader_mu_red', 'TdS_red_kJ', C_BLUE_DARK,
         r'$T\Delta S_{\mathrm{red}}$', '(b) Réduction'),
    ]

    for fname, target_col, color, label_prefix, subtitle in configs:
        x1 = data['bader_sumO_n'].values.astype(float)
        x2 = data['mu_vert'].values.astype(float)
        y = data[target_col].values
        names = data['molecule'].values
        mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
        n = int(mask.sum())

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)

        if n >= 6:
            X = np.column_stack([x1[mask], x2[mask]])
            yv = y[mask]
            reg = LinearRegression().fit(X, yv)
            y_pred = reg.predict(X)
            r2 = r2_score(yv, y_pred)
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - 3)
            mae_2d = mean_absolute_error(yv, y_pred)

            # Parity: real (x-axis) vs predicted (y-axis)
            regression_with_band(ax, yv, y_pred, color=color)
            ax.scatter(yv, y_pred, c=color, **SCATTER_KW)
            names_v = names[mask]
            for i, name in enumerate(names_v):
                ax.annotate(name, (yv[i], y_pred[i]), **ANNOT_KW)

            # Parity line
            add_parity_line(ax)

            # Molecules without QTAIM data (constant fallback)
            cst = np.mean(y[np.isfinite(y)])
            mae_cst = np.mean(np.abs(y[np.isfinite(y)] - cst))
            no_data = ~mask & np.isfinite(y)
            if no_data.sum() > 0:
                ax.scatter(y[no_data], np.full(int(no_data.sum()), cst),
                           c=C_GRAY, s=40, alpha=0.5, marker='x', zorder=4,
                           linewidths=1.5, label='Sans QTAIM')
                for i in np.where(no_data)[0]:
                    ax.annotate(names[i], (y[i], cst), fontsize=9, color=C_GRAY,
                                textcoords='offset points', xytext=(4, -8))
                ax.legend(fontsize=12, loc='best')

            stats_text(ax, r2_adj, mae_2d, n, pos='upper left', adjusted=True)

        ax.set_xlabel(f'{label_prefix} DFT (kJ/mol)')
        ax.set_ylabel(f'{label_prefix} prédit (kJ/mol)')
        setup_grid(ax)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{fname}.png')
        plt.close()
        cst_str = f', const MAE={mae_cst:.2f}' if n >= 6 else ''
        print(f'  [+] {fname}.png  (R2_adj={r2_adj:.3f}, MAE={mae_2d:.2f}{cst_str})')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    data = load_data()
    print(f'Molecules: {len(data)} ({data["bader_sumO_n"].notna().sum()} with QTAIM)')
    plot_ddU_vs_eta(data)
    plot_TdS_parity(data)
    print('Done.')
