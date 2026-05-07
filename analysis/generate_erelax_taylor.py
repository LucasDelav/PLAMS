#!/usr/bin/env amspython
"""
Section 3.2 — Développement de Taylor pour la correction E_relax.

Modèles multiplicatifs testés (réduction et oxydation) :

  Actuel (sans paramètre) : E_relax^corr = E_relax^LVC2 × (1 + (f_O)^2)

  Ordre 2 : E_relax^corr = E_relax^LVC2 × (c_0 + c_1·f_O + c_2·f_O^2)
  Ordre 3 : E_relax^corr = E_relax^LVC2 × (c_0 + c_1·f_O + c_2·f_O^2 + c_3·f_O^3)

Les coefficients proches de 0, 1 ou -1 valident le fondement physique du modèle.

Données : descriptors_per_molecule.csv (15 catéchols stables, per-molécule).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_project_root = Path(__file__).resolve().parent.parent
DATA_DIR = _project_root / "rapport" / "data"
FIGURES_DIR = _project_root / "rapport" / "figures"
RESULTS_DIR = _project_root / "results" / "figures"

DEA_MOLECULES = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}


# ======================================================================
# FITTING
# ======================================================================

def fit_taylor(E_lvc2, f_O, E_real, order=2):
    """Fit E_real = E_lvc2 × (c_0 + c_1·f_O + ... + c_n·f_O^n). Sans intercept."""
    features = [E_lvc2 * (f_O ** k) for k in range(order + 1)]
    labels = [f'c_{k}' for k in range(order + 1)]
    X = np.column_stack(features)
    beta, _, _, _ = np.linalg.lstsq(X, E_real, rcond=None)
    return beta, labels, X @ beta


def compute_metrics(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    max_err = np.max(np.abs(y - y_pred))
    return r2, mae, rmse, max_err


def leave_one_out_cv(E_lvc2, f_O, E_real, order):
    n = len(E_real)
    y_pred_cv = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta, _, _ = fit_taylor(E_lvc2[mask], f_O[mask], E_real[mask], order)
        x_test = np.array([E_lvc2[i] * (f_O[i] ** k) for k in range(order + 1)])
        y_pred_cv[i] = x_test @ beta
    return y_pred_cv


def current_model(E_lvc2, f_O):
    return E_lvc2 * (1 + f_O ** 2)


# ======================================================================
# FIGURES
# ======================================================================

def plot_comparison(process_name, mol_names, E_real, models_dict, output_path):
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for idx, (model_name, info) in enumerate(models_dict.items()):
        ax = axes[idx]
        y_pred = info['y_pred']
        r2, mae, rmse, max_err = info['metrics']
        ax.scatter(E_real, y_pred, s=60, alpha=0.8, color='#2196F3',
                   edgecolors='white', linewidths=0.5, zorder=3)
        for i, mol in enumerate(mol_names):
            ax.annotate(mol, (E_real[i], y_pred[i]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=5, alpha=0.7)
        vmin = min(E_real.min(), y_pred.min()) * 1.1
        vmax = max(E_real.max(), y_pred.max()) * 0.9
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, alpha=0.5, zorder=1)
        ax.set_xlabel(r'$E_{\mathrm{relax}}$ DFT (kJ/mol)', fontsize=10)
        ax.set_ylabel(r'$E_{\mathrm{relax}}$ prédit (kJ/mol)', fontsize=10)
        ax.set_title(f'{process_name} — {model_name}', fontsize=10, fontweight='bold')
        textstr = (f'$R^2$ = {r2:.4f}\n'
                   f'MAE = {mae:.2f} kJ/mol\n'
                   f'max|err| = {max_err:.2f} kJ/mol')
        if 'coeffs_str' in info:
            textstr += f'\n{info["coeffs_str"]}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure : {output_path}")


def plot_correction_factor(mol_names, f_O, E_real, E_lvc2, models_dict, output_path):
    ratio = E_real / E_lvc2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(f_O, ratio, s=70, alpha=0.8, color='#333333',
               edgecolors='white', linewidths=0.5, zorder=5, label='DFT')
    for i, mol in enumerate(mol_names):
        ax.annotate(mol, (f_O[i], ratio[i]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=5, alpha=0.7)
    f_O_curve = np.linspace(0, max(f_O) * 1.2, 100)
    colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']
    for idx, (model_name, info) in enumerate(models_dict.items()):
        if 'beta' in info:
            beta = info['beta']
            P = sum(beta[k] * f_O_curve ** k for k in range(len(beta)))
            ax.plot(f_O_curve, P, '-', color=colors[idx % len(colors)],
                    lw=2, alpha=0.8, label=model_name, zorder=3)
        elif 'Actuel' in model_name:
            P = 1 + f_O_curve ** 2
            ax.plot(f_O_curve, P, '--', color=colors[idx % len(colors)],
                    lw=2, alpha=0.8, label=model_name, zorder=3)
    ax.set_xlabel(r'$f_{\mathrm{O}}^{\mathrm{LUMO}}$', fontsize=12)
    ax.set_ylabel(r'$E_{\mathrm{relax}} / E_{\mathrm{relax}}^{\mathrm{LVC2}}$', fontsize=12)
    ax.set_title('Facteur de correction vs composition LUMO', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure : {output_path}")


# ======================================================================
# ANALYSE
# ======================================================================

def run_process(process_name, E_real, E_lvc2, f_O, mol_names, prefix):
    print(f"\n{'=' * 70}")
    print(f"  {process_name.upper()}")
    print(f"{'=' * 70}")

    models = {}

    # Modèle actuel (sans paramètre)
    y_pred_current = current_model(E_lvc2, f_O)
    metrics_current = compute_metrics(E_real, y_pred_current)
    models['Actuel: (1+f_O²)'] = {
        'y_pred': y_pred_current, 'metrics': metrics_current,
        'coeffs_str': 'Sans paramètre',
    }
    print(f"\n  Actuel (1 + f_O²) :")
    print(f"    R² = {metrics_current[0]:.4f}, MAE = {metrics_current[1]:.2f}, "
          f"max|err| = {metrics_current[3]:.2f} kJ/mol")

    # Ordre 2
    beta2, labels2, y_pred2 = fit_taylor(E_lvc2, f_O, E_real, order=2)
    metrics2 = compute_metrics(E_real, y_pred2)
    y_pred2_cv = leave_one_out_cv(E_lvc2, f_O, E_real, order=2)
    metrics2_cv = compute_metrics(E_real, y_pred2_cv)
    coeffs_str2 = ', '.join(f'{l}={v:.4f}' for l, v in zip(labels2, beta2))
    models['Ordre 2'] = {
        'y_pred': y_pred2, 'metrics': metrics2,
        'beta': beta2, 'coeffs_str': coeffs_str2,
    }
    print(f"\n  Ordre 2 : P(f_O) = c_0 + c_1·f_O + c_2·f_O²")
    for l, v in zip(labels2, beta2):
        print(f"    {l} = {v:+.4f}")
    print(f"    R² = {metrics2[0]:.4f}, MAE = {metrics2[1]:.2f}, max|err| = {metrics2[3]:.2f}")
    print(f"    LOO-CV : R² = {metrics2_cv[0]:.4f}, MAE = {metrics2_cv[1]:.2f}")

    # Ordre 3
    beta3, labels3, y_pred3 = fit_taylor(E_lvc2, f_O, E_real, order=3)
    metrics3 = compute_metrics(E_real, y_pred3)
    y_pred3_cv = leave_one_out_cv(E_lvc2, f_O, E_real, order=3)
    metrics3_cv = compute_metrics(E_real, y_pred3_cv)
    coeffs_str3 = ', '.join(f'{l}={v:.4f}' for l, v in zip(labels3, beta3))
    models['Ordre 3'] = {
        'y_pred': y_pred3, 'metrics': metrics3,
        'beta': beta3, 'coeffs_str': coeffs_str3,
    }
    print(f"\n  Ordre 3 : P(f_O) = c_0 + c_1·f_O + c_2·f_O² + c_3·f_O³")
    for l, v in zip(labels3, beta3):
        print(f"    {l} = {v:+.4f}")
    print(f"    R² = {metrics3[0]:.4f}, MAE = {metrics3[1]:.2f}, max|err| = {metrics3[3]:.2f}")
    print(f"    LOO-CV : R² = {metrics3_cv[0]:.4f}, MAE = {metrics3_cv[1]:.2f}")

    # Interprétation
    print(f"\n  Interprétation des coefficients (ordre 2) :")
    c0, c1, c2 = beta2
    for val, name, target in [(c0, 'c_0', 1.0), (c1, 'c_1', 0.0), (c2, 'c_2', 1.0)]:
        if abs(val - target) < 0.15:
            print(f"    {name} ≈ {target:.0f} ✓")
        else:
            print(f"    {name} = {val:+.4f}")

    # Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(process_name, mol_names, E_real, models,
                    FIGURES_DIR / f"erelax_taylor_{prefix}_parity.png")
    plot_correction_factor(mol_names, f_O, E_real, E_lvc2, models,
                           FIGURES_DIR / f"erelax_taylor_{prefix}_factor.png")

    # Tableau
    print(f"\n  {'─' * 70}")
    print(f"  TABLEAU RÉCAPITULATIF — {process_name}")
    print(f"  {'─' * 70}")
    print(f"  {'Modèle':<22s} {'R²':>8s} {'MAE':>8s} {'max|err|':>10s} {'CV-MAE':>8s}")
    print(f"  {'─' * 70}")
    for model_name, info in models.items():
        r2, mae, rmse, max_err = info['metrics']
        cv_str = '—'
        if model_name == 'Ordre 2':
            cv_str = f'{metrics2_cv[1]:.2f}'
        elif model_name == 'Ordre 3':
            cv_str = f'{metrics3_cv[1]:.2f}'
        print(f"  {model_name:<22s} {r2:>8.4f} {mae:>8.2f} {max_err:>10.2f} {cv_str:>8s}")

    return models, {
        'order2': {'beta': beta2, 'labels': labels2, 'metrics': metrics2, 'cv_metrics': metrics2_cv},
        'order3': {'beta': beta3, 'labels': labels3, 'metrics': metrics3, 'cv_metrics': metrics3_cv},
        'current': {'metrics': metrics_current},
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("DÉVELOPPEMENT DE TAYLOR POUR E_relax")
    print("Données : descriptors_per_molecule.csv — 15 catéchols stables")
    print("=" * 70)

    df = pd.read_csv(DATA_DIR / "descriptors_per_molecule.csv")
    print(f"\n{len(df)} molécules chargées")

    df = df[~df['molecule'].isin(DEA_MOLECULES)].copy()
    print(f"{len(df)} molécules après exclusion DEA")

    # --- Réduction ---
    df_red = df.dropna(subset=['E_real_red', 'LVC2_red', 'LUMO_O_frac']).copy()
    print(f"\nRéduction : {len(df_red)} molécules avec données complètes")

    details_red = None
    if len(df_red) >= 5:
        _, details_red = run_process(
            'Réduction',
            df_red['E_real_red'].values, df_red['LVC2_red'].values,
            df_red['LUMO_O_frac'].values, df_red['molecule'].values, 'red')

    # --- Oxydation ---
    df_ox = df.dropna(subset=['E_real_ox', 'LVC2_ox', 'LUMO_O_frac']).copy()
    print(f"\nOxydation : {len(df_ox)} molécules avec données complètes")

    details_ox = None
    if len(df_ox) >= 5:
        _, details_ox = run_process(
            'Oxydation',
            df_ox['E_real_ox'].values, df_ox['LVC2_ox'].values,
            df_ox['LUMO_O_frac'].values, df_ox['molecule'].values, 'ox')

    # --- Export CSV ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for process, details in [('reduction', details_red), ('oxidation', details_ox)]:
        if details is None:
            continue
        for order_key in ['current', 'order2', 'order3']:
            d = details[order_key]
            row = {'process': process, 'model': order_key}
            r2, mae, rmse, max_err = d['metrics']
            row.update({'R2': r2, 'MAE': mae, 'RMSE': rmse, 'max_err': max_err})
            if 'cv_metrics' in d:
                row['CV_R2'] = d['cv_metrics'][0]
                row['CV_MAE'] = d['cv_metrics'][1]
            if 'beta' in d:
                for l, v in zip(d['labels'], d['beta']):
                    row[l] = v
            rows.append(row)

    csv_out = RESULTS_DIR / "erelax_taylor_summary.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False, float_format='%.6f')
    print(f"\n  CSV : {csv_out}")
    print("\nDone.")


if __name__ == '__main__':
    main()
