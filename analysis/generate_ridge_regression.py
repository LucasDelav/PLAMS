#!/usr/bin/env amspython
"""
Section 3.1.3 — Régression ridge multi-orbitale + descripteur (f_O^LUMO)^2.

Modèles testés (IP et EA) :
  M1  Koopmans calibré           : Y = a·ε_ref + b                    (2 param)
  M2  Bilinéaire                 : Y = a·ε_H + b·ε_L + c             (3 param)
  M3  Bilinéaire + (f_O)^2       : Y = a·ε_H + b·ε_L + c·(f_O)^2 + d (4 param)
  M4  4 orbitales                : Y = a·ε_{H-1} + b·ε_H + c·ε_L + d·ε_{L+1} + e (5 param)
  M5  4 orbitales + (f_O)^2      : Y = ..+ f·(f_O)^2 + g             (6 param)

Toutes les énergies orbitales sont en kJ/mol (pas en eV).
λ optimisé par leave-one-molecule-out CV.

Données : data/Acetonitrile_marais, 34 molécules (hors 4 DEA), per-conformère.
"""

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Import collect_all_conformers ---
_project_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "multi_molecule_data",
    _project_root / "common" / "multi_molecule_data.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
collect_all_conformers = _mod.collect_all_conformers

BATCH_DIR = _project_root / "data" / "Acetonitrile_marais"
FIGURES_DIR = _project_root / "rapport" / "figures" / "beyond_koopmans"
DATA_DIR = _project_root / "results" / "figures"
EH_TO_EV = 27.211386245988
EV_TO_KJ = 96.4853329  # 1 eV = 96.485 kJ/mol

# 19 catéchols (exclut les quinones)
CATECHOLS = {
    'HMC', 'F-HMC', 'Cl-HMC', 'Br-HMC', 'I-HMC',
    'DA', 'F-DA', 'Cl-DA', 'Br-DA', 'I-DA', 'L-DA',
    'CA', 'Al-CA', 'HO-CA', 'H2C-CA',
    'DHI', 'O-DHI', 'S-DHI', 'P-DHI',
}

# 4 molécules DEA à exclure
DEA_MOLECULES = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}

# Catéchols stables = 15
STABLE_CATECHOLS = CATECHOLS - DEA_MOLECULES


# ======================================================================
# EXTRACTION ORBITALES + f_O^LUMO DEPUIS RKF
# ======================================================================

def extract_orbital_data(workdir, mol_name):
    """Extrait ε_{H-1}, ε_H, ε_L, ε_{L+1} (eV) et f_O^LUMO par conformère."""
    from scm.plams import KFReader

    plams_dir = workdir / "redox" / "calculations" / "plams_workdir"
    if not plams_dir.exists():
        return {}

    results = {}
    for rkf_dir in sorted(plams_dir.glob(f"{mol_name}_conf_*_neutral_tz2p")):
        rkf_path = rkf_dir / "adf.rkf"
        if not rkf_path.exists():
            continue
        try:
            conf_num = int(rkf_dir.name.split("_conf_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue

        try:
            kf = KFReader(str(rkf_path))

            # --- Orbital energies ---
            n_mo = kf.read('A', 'nmo_A')
            occupations = np.array(kf.read('A', 'froc_A'), dtype=float)
            energies = np.array(kf.read('A', 'eps_A'), dtype=float)
            n_occ = int(np.sum(occupations > 0.5))
            n_total = len(energies)

            if n_occ < 2 or n_occ + 1 >= n_total:
                continue

            homo_m1_eV = energies[n_occ - 2] * EH_TO_EV
            homo_eV = energies[n_occ - 1] * EH_TO_EV
            lumo_eV = energies[n_occ] * EH_TO_EV
            lumo_p1_eV = energies[n_occ + 1] * EH_TO_EV if n_occ + 1 < n_total else np.nan

            # --- f_O^LUMO via SFO populations ---
            n_atoms = kf.read('Molecule', 'nAtoms')
            atomic_numbers = list(kf.read('Molecule', 'AtomicNumbers'))
            n_sfo = kf.read('SFOs', 'number')
            sfo_fragments = list(kf.read('SFOs', 'fragment'))
            n_contrib = list(kf.read('SFO popul', 'number of contributions'))
            sfo_pop_idx = list(kf.read('SFO popul', 'sfo_index'))
            sfo_pop_val = list(kf.read('SFO popul', 'sfo_pop'))

            lumo_mo_idx = n_occ  # 0-indexed
            offset = 0
            atom_weights = np.zeros(n_atoms)
            for mo_idx in range(n_mo):
                nc = n_contrib[mo_idx]
                if mo_idx == lumo_mo_idx:
                    for sfo_idx, pop_val in zip(
                            sfo_pop_idx[offset:offset + nc],
                            sfo_pop_val[offset:offset + nc]):
                        if 1 <= sfo_idx <= n_sfo:
                            atom_idx = sfo_fragments[sfo_idx - 1] - 1
                            if 0 <= atom_idx < n_atoms:
                                atom_weights[atom_idx] += abs(pop_val)
                    break
                offset += nc

            total_w = np.sum(atom_weights)
            if total_w > 0:
                atom_fractions = atom_weights / total_w
            else:
                atom_fractions = atom_weights
            O_frac = sum(atom_fractions[i] for i in range(n_atoms)
                         if atomic_numbers[i] == 8)

            results[conf_num] = {
                'HOMOm1_eV': homo_m1_eV,
                'HOMO_eV_tz2p': homo_eV,
                'LUMO_eV_tz2p': lumo_eV,
                'LUMOp1_eV': lumo_p1_eV,
                'LUMO_O_frac': O_frac,
            }

        except Exception as e:
            print(f"  ! Erreur {mol_name} conf_{conf_num}: {e}")

    return results


# ======================================================================
# RIDGE REGRESSION + LEAVE-ONE-MOLECULE-OUT CV
# ======================================================================

def ridge_fit(X, y, lam):
    """Ridge regression (intercept non pénalisé)."""
    n, p = X.shape
    X_aug = np.column_stack([X, np.ones(n)])
    penalty = np.eye(p + 1) * lam
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(X_aug.T @ X_aug + penalty, X_aug.T @ y)
    return beta


def ridge_predict(X, beta):
    n = X.shape[0]
    X_aug = np.column_stack([X, np.ones(n)])
    return X_aug @ beta


def leave_one_mol_out_cv(X, y, mol_labels, lam):
    """Leave-one-molecule-out CV."""
    unique_mols = np.unique(mol_labels)
    y_pred_cv = np.full_like(y, np.nan)
    for mol in unique_mols:
        test_mask = mol_labels == mol
        train_mask = ~test_mask
        if np.sum(train_mask) < X.shape[1] + 1:
            continue
        beta = ridge_fit(X[train_mask], y[train_mask], lam)
        y_pred_cv[test_mask] = ridge_predict(X[test_mask], beta)
    valid = ~np.isnan(y_pred_cv)
    if np.sum(valid) == 0:
        return np.inf, np.inf
    residuals = y[valid] - y_pred_cv[valid]
    return np.mean(np.abs(residuals)), np.sqrt(np.mean(residuals ** 2))


def optimize_lambda(X, y, mol_labels, lambdas=None):
    if lambdas is None:
        lambdas = np.logspace(-4, 4, 50)
    best_lam, best_cv_mae = 0.0, np.inf
    results = []
    for lam in np.concatenate([[0.0], lambdas]):
        cv_mae, cv_rmse = leave_one_mol_out_cv(X, y, mol_labels, lam)
        results.append((lam, cv_mae, cv_rmse))
        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_lam = lam
    return best_lam, best_cv_mae, results


def compute_metrics(y, y_pred, n_params):
    n = len(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(ss_res / n)
    return r2, r2_adj, mae, rmse


# ======================================================================
# MODÈLES
# ======================================================================

def build_models_ip(df):
    models = {}
    models['M1: Koopmans (ε_H)'] = {
        'X': df[['HOMO_kJ']].values, 'labels': ['ε_H'],
    }
    models['M2: ε_H + ε_L'] = {
        'X': df[['HOMO_kJ', 'LUMO_kJ']].values, 'labels': ['ε_H', 'ε_L'],
    }
    models['M3: ε_H + ε_L + (f_O)²'] = {
        'X': df[['HOMO_kJ', 'LUMO_kJ', 'fO_sq']].values,
        'labels': ['ε_H', 'ε_L', '(f_O)²'],
    }
    models['M4: 4 orbitales'] = {
        'X': df[['HOMOm1_kJ', 'HOMO_kJ', 'LUMO_kJ', 'LUMOp1_kJ']].values,
        'labels': ['ε_{H-1}', 'ε_H', 'ε_L', 'ε_{L+1}'],
    }
    models['M5: 4 orb + (f_O)²'] = {
        'X': df[['HOMOm1_kJ', 'HOMO_kJ', 'LUMO_kJ', 'LUMOp1_kJ', 'fO_sq']].values,
        'labels': ['ε_{H-1}', 'ε_H', 'ε_L', 'ε_{L+1}', '(f_O)²'],
    }
    return models


def build_models_ea(df):
    models = {}
    models['M1: Koopmans (ε_L)'] = {
        'X': df[['LUMO_kJ']].values, 'labels': ['ε_L'],
    }
    models['M2: ε_H + ε_L'] = {
        'X': df[['HOMO_kJ', 'LUMO_kJ']].values, 'labels': ['ε_H', 'ε_L'],
    }
    models['M3: ε_H + ε_L + (f_O)²'] = {
        'X': df[['HOMO_kJ', 'LUMO_kJ', 'fO_sq']].values,
        'labels': ['ε_H', 'ε_L', '(f_O)²'],
    }
    models['M4: 4 orbitales'] = {
        'X': df[['HOMOm1_kJ', 'HOMO_kJ', 'LUMO_kJ', 'LUMOp1_kJ']].values,
        'labels': ['ε_{H-1}', 'ε_H', 'ε_L', 'ε_{L+1}'],
    }
    models['M5: 4 orb + (f_O)²'] = {
        'X': df[['HOMOm1_kJ', 'HOMO_kJ', 'LUMO_kJ', 'LUMOp1_kJ', 'fO_sq']].values,
        'labels': ['ε_{H-1}', 'ε_H', 'ε_L', 'ε_{L+1}', '(f_O)²'],
    }
    return models


# ======================================================================
# FIGURES
# ======================================================================

def plot_parity_grid(target_name, y, models_results, mol_labels, output_path):
    n_models = len(models_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    unique_mols = sorted(set(mol_labels))
    cmap = plt.cm.get_cmap('tab20', max(len(unique_mols), 20))
    color_map = {mol: cmap(i) for i, mol in enumerate(unique_mols)}

    for idx, (model_name, res) in enumerate(models_results.items()):
        ax = axes[idx]
        y_pred = res['y_pred']
        for mol in unique_mols:
            mask = mol_labels == mol
            ax.scatter(y[mask], y_pred[mask], s=30, alpha=0.7,
                       color=color_map[mol], edgecolors='white',
                       linewidths=0.3, zorder=3)
        vmin = min(y.min(), y_pred.min()) - 5
        vmax = max(y.max(), y_pred.max()) + 5
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, alpha=0.5, zorder=1)
        ax.set_xlabel(f'{target_name} DFT (kJ/mol)', fontsize=9)
        ax.set_ylabel(f'{target_name} prédit (kJ/mol)', fontsize=9)
        ax.set_title(model_name, fontsize=9, fontweight='bold')
        textstr = (f'$R^2_{{adj}}$ = {res["r2_adj"]:.4f}\n'
                   f'MAE = {res["mae"]:.2f}\n'
                   f'CV-MAE = {res["cv_mae"]:.2f}\n'
                   f'λ = {res["lambda"]:.3g}')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.grid(True, alpha=0.3, zorder=0)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure : {output_path}")


def plot_coefficients(target_name, models_results, output_path):
    fig, axes = plt.subplots(1, len(models_results),
                              figsize=(4 * len(models_results), 4), sharey=False)
    if len(models_results) == 1:
        axes = [axes]
    for idx, (model_name, res) in enumerate(models_results.items()):
        ax = axes[idx]
        labels = res['feature_labels'] + ['intercept']
        beta = res['beta']
        colors = ['#2196F3' if b >= 0 else '#F44336' for b in beta]
        bars = ax.barh(range(len(labels)), beta, color=colors, alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Coefficient', fontsize=9)
        ax.set_title(model_name, fontsize=9, fontweight='bold')
        ax.axvline(0, color='black', lw=0.5, alpha=0.5)
        for i, (_, val) in enumerate(zip(bars, beta)):
            x_pos = val + 0.02 * max(abs(b) for b in beta) * np.sign(val)
            ax.text(x_pos, i, f'{val:.3f}', va='center', fontsize=7,
                    ha='left' if val >= 0 else 'right')
        ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure : {output_path}")


def plot_lambda_curve(target_name, model_name, cv_results, best_lam, output_path):
    lambdas = [r[0] for r in cv_results if r[0] > 0]
    cv_maes = [r[1] for r in cv_results if r[0] > 0]
    ols_mae = [r[1] for r in cv_results if r[0] == 0.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(lambdas, cv_maes, 'o-', color='#2196F3', markersize=3, lw=1.2,
                label='Ridge CV-MAE')
    if ols_mae:
        ax.axhline(ols_mae[0], color='#F44336', ls='--', lw=1, alpha=0.7,
                    label=f'OLS CV-MAE = {ols_mae[0]:.2f}')
    if best_lam > 0:
        ax.axvline(best_lam, color='green', ls=':', lw=1.2, alpha=0.7,
                    label=f'λ* = {best_lam:.3g}')
    ax.set_xlabel('λ (paramètre de régularisation)', fontsize=10)
    ax.set_ylabel('CV-MAE (kJ/mol)', fontsize=10)
    ax.set_title(f'{target_name} — {model_name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# ======================================================================
# ANALYSE
# ======================================================================

def run_analysis(target_name, y, models, mol_labels, prefix):
    print(f"\n{'=' * 70}")
    print(f"  RÉGRESSION RIDGE — {target_name}")
    print(f"{'=' * 70}")

    results = {}
    for model_name, spec in models.items():
        X = spec['X']
        feature_labels = spec['labels']
        n_params = X.shape[1]

        best_lam, best_cv_mae, cv_results = optimize_lambda(X, y, mol_labels)
        beta = ridge_fit(X, y, best_lam)
        y_pred = ridge_predict(X, beta)
        r2, r2_adj, mae, rmse = compute_metrics(y, y_pred, n_params)

        beta_ols = ridge_fit(X, y, 0.0)
        y_pred_ols = ridge_predict(X, beta_ols)
        _, _, mae_ols, _ = compute_metrics(y, y_pred_ols, n_params)
        cv_mae_ols, _ = leave_one_mol_out_cv(X, y, mol_labels, 0.0)

        results[model_name] = {
            'beta': beta, 'feature_labels': feature_labels,
            'y_pred': y_pred,
            'r2': r2, 'r2_adj': r2_adj, 'mae': mae, 'rmse': rmse,
            'lambda': best_lam, 'cv_mae': best_cv_mae,
            'cv_results': cv_results,
            'mae_ols': mae_ols, 'cv_mae_ols': cv_mae_ols, 'beta_ols': beta_ols,
        }

        print(f"\n  {model_name}")
        print(f"  {'─' * 60}")
        print(f"  λ* = {best_lam:.4g}")
        print(f"  R²_adj = {r2_adj:.4f}   MAE = {mae:.2f}   RMSE = {rmse:.2f}   "
              f"CV-MAE = {best_cv_mae:.2f} kJ/mol")
        print(f"  (OLS: MAE = {mae_ols:.2f}, CV-MAE = {cv_mae_ols:.2f})")
        print(f"  Coefficients (ridge, ε en kJ/mol) :")
        all_labels = feature_labels + ['intercept']
        for i, lbl in enumerate(all_labels):
            marker = '  (non pénalisé)' if lbl == 'intercept' else ''
            print(f"    {lbl:<12s} = {beta[i]:>+10.4f}{marker}")
        print(f"  Coefficients (OLS) :")
        for i, lbl in enumerate(all_labels):
            print(f"    {lbl:<12s} = {beta_ols[i]:>+10.4f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_parity_grid(target_name, y, results, mol_labels,
                     FIGURES_DIR / f"{prefix}_ridge_parity.png")
    plot_coefficients(target_name, results,
                      FIGURES_DIR / f"{prefix}_ridge_coefficients.png")
    for model_name, res in results.items():
        if '4 orb' in model_name or 'f_O' in model_name:
            safe_name = model_name.replace(' ', '_').replace(':', '').replace('+', 'p')
            plot_lambda_curve(target_name, model_name, res['cv_results'], res['lambda'],
                              FIGURES_DIR / f"{prefix}_lambda_{safe_name}.png")

    print(f"\n  {'─' * 80}")
    print(f"  TABLEAU RÉCAPITULATIF — {target_name}")
    print(f"  {'─' * 80}")
    print(f"  {'Modèle':<28s} {'R²_adj':>8s} {'MAE':>8s} {'CV-MAE':>8s} {'λ*':>10s} {'n_param':>8s}")
    print(f"  {'─' * 80}")
    for model_name, res in results.items():
        n_p = len(res['feature_labels'])
        print(f"  {model_name:<28s} {res['r2_adj']:>8.4f} {res['mae']:>8.2f} "
              f"{res['cv_mae']:>8.2f} {res['lambda']:>10.3g} {n_p:>8d}")

    return results


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("SECTION 3.1.3 — RÉGRESSION RIDGE MULTI-ORBITALE + (f_O^LUMO)²")
    print(f"Données : data/Acetonitrile_marais — 34 molécules (hors 4 DEA)")
    print("=" * 70)

    # 1. Sélectionner les workdirs (toutes sauf DEA)
    workdirs = []
    for d in sorted(BATCH_DIR.glob("*_workdir")):
        mol_name = d.name.replace('_workdir', '')
        if mol_name not in DEA_MOLECULES:
            workdirs.append(d)
    print(f"\n{len(workdirs)} workdirs (38 - 4 DEA = 34)")

    # 2. Charger IP/EA per-conformère via collect_all_conformers
    print("\nChargement des données redox per-conformère...")
    df = collect_all_conformers(workdirs)
    print(f"  {len(df)} conformères, {df['molecule_name'].nunique()} molécules")

    # 3. Extraire ε_{H-1}, ε_L+1 et f_O^LUMO depuis les RKF TZ2P
    print("\nExtraction orbitales TZ2P + f_O^LUMO depuis les RKF...")
    orb_rows = []
    for workdir in workdirs:
        mol_name = workdir.name.replace('_workdir', '')
        orb_data = extract_orbital_data(workdir, mol_name)
        for conf_num, data in orb_data.items():
            orb_rows.append({'molecule_name': mol_name, 'conformer': conf_num, **data})
        if orb_data:
            print(f"    {mol_name}: {len(orb_data)} conformères")

    df_orb = pd.DataFrame(orb_rows)
    print(f"  {len(df_orb)} conformères avec orbitales TZ2P")

    # 4. Fusionner
    df = df.merge(df_orb, on=['molecule_name', 'conformer'], how='inner')
    print(f"  {len(df)} conformères après fusion")

    # 5. Convertir en kJ/mol
    df['HOMO_kJ'] = df['HOMO_eV_tz2p'] * EV_TO_KJ
    df['HOMOm1_kJ'] = df['HOMOm1_eV'] * EV_TO_KJ
    df['LUMO_kJ'] = df['LUMO_eV_tz2p'] * EV_TO_KJ
    df['LUMOp1_kJ'] = df['LUMOp1_eV'] * EV_TO_KJ
    df['fO_sq'] = df['LUMO_O_frac'] ** 2
    df['EA_kJ_abs'] = df['EA_kJ'].abs()

    print(f"\n  Plages d'énergies orbitales (kJ/mol) :")
    for col in ['HOMO_kJ', 'HOMOm1_kJ', 'LUMO_kJ', 'LUMOp1_kJ']:
        print(f"    {col:<14s}: [{df[col].min():.1f}, {df[col].max():.1f}]")
    print(f"    {'fO_sq':<14s}: [{df['fO_sq'].min():.4f}, {df['fO_sq'].max():.4f}]")

    # 6. Filtrer données valides
    required = ['IP_kJ', 'EA_kJ_abs', 'HOMO_kJ', 'HOMOm1_kJ', 'LUMO_kJ', 'LUMOp1_kJ', 'fO_sq']
    df_valid = df.dropna(subset=required).copy()
    print(f"\n  {len(df_valid)} conformères avec données complètes "
          f"({df_valid['molecule_name'].nunique()} molécules)")

    mol_labels = df_valid['molecule_name'].values

    # === IP ===
    y_ip = df_valid['IP_kJ'].values
    results_ip = run_analysis('IP', y_ip, build_models_ip(df_valid), mol_labels, 'IP')

    # === EA ===
    y_ea = df_valid['EA_kJ_abs'].values
    results_ea = run_analysis('|EA|', y_ea, build_models_ea(df_valid), mol_labels, 'EA')

    # === Export CSV ===
    summary_rows = []
    for target, results in [('IP', results_ip), ('|EA|', results_ea)]:
        for model_name, res in results.items():
            row = {
                'target': target, 'model': model_name,
                'n_features': len(res['feature_labels']),
                'R2_adj': res['r2_adj'], 'MAE': res['mae'],
                'RMSE': res['rmse'], 'CV_MAE': res['cv_mae'],
                'lambda_opt': res['lambda'],
                'MAE_OLS': res['mae_ols'], 'CV_MAE_OLS': res['cv_mae_ols'],
            }
            all_labels = res['feature_labels'] + ['intercept']
            for lbl, val in zip(all_labels, res['beta']):
                row[f'coef_{lbl}'] = val
            summary_rows.append(row)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = DATA_DIR / "ridge_regression_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_out, index=False, float_format='%.6f')
    print(f"\n  CSV : {csv_out}")
    print("\nDone.")


if __name__ == '__main__':
    main()
