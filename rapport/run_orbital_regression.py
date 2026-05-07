#!/usr/bin/env amspython
"""
Re-run the 3-model orbital regression (Koopmans, bilinear, 4 orbitals)
with orbital energies in kJ/mol, and print coefficients + full statistics.
"""

import csv
import numpy as np
from pathlib import Path
from scipy.stats import linregress, pearsonr, f as f_dist
from scm.plams import KFFile

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Acetonitrile_marais"
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
    """Load per-conformer IP/EA from redox CSVs, then extract orbital energies from RKFs."""
    records = []

    for wdir in sorted(DATA_DIR.iterdir()):
        if not wdir.is_dir() or not wdir.name.endswith('_workdir'):
            continue
        mol = wdir.name.replace('_workdir', '')
        if mol in QUINONES or mol in DEA_EXCLUDED:
            continue

        # Read redox conformers CSV
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
                        'weight': float(row['weight']),
                        'IP_kJ': float(row['IP_kJ']),
                        'EA_kJ': float(row['EA_kJ']),
                        # Orbital energies in kJ/mol
                        'eH_kJ': eps[n_occ - 1] * EH_TO_EV * EV_TO_KJ,
                        'eHm1_kJ': eps[n_occ - 2] * EH_TO_EV * EV_TO_KJ,
                        'eL_kJ': eps[n_occ] * EH_TO_EV * EV_TO_KJ,
                        'eLp1_kJ': eps[n_occ + 1] * EH_TO_EV * EV_TO_KJ,
                    })
                except Exception as e:
                    print(f"  [!] {mol} conf_{conf_num}: {e}")

    return records


def r2_adj(y, pred, p):
    n = len(y)
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def f_test_nested(y, pred_reduced, p_reduced, pred_full, p_full):
    """F-test: does the full model improve significantly over the reduced model?"""
    n = len(y)
    ss_red = np.sum((y - pred_reduced)**2)
    ss_full = np.sum((y - pred_full)**2)
    df_extra = p_full - p_reduced
    df_resid = n - p_full - 1
    if df_resid <= 0 or df_extra <= 0:
        return np.nan, np.nan
    f_stat = ((ss_red - ss_full) / df_extra) / (ss_full / df_resid)
    p_val = 1 - f_dist.cdf(f_stat, df_extra, df_resid)
    return f_stat, p_val


def leave_one_mol_out_cv(records, y, X, molecules):
    """Leave-one-molecule-out cross-validation MAE."""
    unique_mols = list(set(molecules))
    errors = []
    for mol in unique_mols:
        mask_train = np.array([m != mol for m in molecules])
        mask_test = ~mask_train
        if np.sum(mask_test) == 0 or np.sum(mask_train) < X.shape[1]:
            continue
        beta = np.linalg.lstsq(X[mask_train], y[mask_train], rcond=None)[0]
        pred = X[mask_test] @ beta
        errors.extend(np.abs(y[mask_test] - pred).tolist())
    return np.mean(errors) if errors else np.nan


def run_regression(name, y, X_dict, molecules):
    """Run all 3 models and print results."""
    print(f"\n{'='*70}")
    print(f"  {name} (n = {len(y)}, {len(set(molecules))} molécules)")
    print(f"{'='*70}")

    n = len(y)

    # Model 1: Koopmans calibré (primary orbital only)
    x_prim = X_dict['primary']
    X1 = np.column_stack([np.ones(n), x_prim])
    beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    pred1 = X1 @ beta1

    # Model 2: Bilinéaire (primary + complement)
    x_comp = X_dict['complement']
    X2 = np.column_stack([np.ones(n), x_prim, x_comp])
    beta2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    pred2 = X2 @ beta2

    # Model 3: 4 orbitales
    x_sec = X_dict['secondary']
    x_comp2 = X_dict['complement2']
    X3 = np.column_stack([np.ones(n), x_sec, x_prim, x_comp, x_comp2])
    beta3 = np.linalg.lstsq(X3, y, rcond=None)[0]
    pred3 = X3 @ beta3

    # Stats
    for label, beta, pred, X_mat, p_params in [
        ('Koopmans calibré', beta1, pred1, X1, 1),
        ('Bilinéaire', beta2, pred2, X2, 2),
        ('4 orbitales', beta3, pred3, X3, 4),
    ]:
        mae = np.mean(np.abs(y - pred))
        rmse = np.sqrt(np.mean((y - pred)**2))
        r2a = r2_adj(y, pred, p_params)
        cv_mae = leave_one_mol_out_cv(None, y, X_mat, molecules)

        print(f"\n  --- {label} ({p_params+1} paramètres) ---")
        print(f"  R²_adj = {r2a:.4f}  |  MAE = {mae:.2f} kJ/mol  |  RMSE = {rmse:.2f} kJ/mol  |  CV-MAE = {cv_mae:.2f} kJ/mol")
        print(f"  Coefficients: {', '.join(f'{b:.4f}' for b in beta)}")

        # t-test for each coefficient
        X_t = X_mat
        ss_res = np.sum((y - pred)**2)
        dof = n - X_t.shape[1]
        if dof > 0:
            mse = ss_res / dof
            try:
                cov = mse * np.linalg.inv(X_t.T @ X_t)
                se = np.sqrt(np.diag(cov))
                t_vals = beta / se
                from scipy.stats import t as t_dist
                p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), dof))
                print(f"  SE:     {', '.join(f'{s:.4f}' for s in se)}")
                print(f"  t-stat: {', '.join(f'{t:.2f}' for t in t_vals)}")
                print(f"  p-val:  {', '.join(f'{p:.2e}' for p in p_vals)}")
            except np.linalg.LinAlgError:
                print("  [!] Could not compute t-stats (singular matrix)")

    # F-tests vs Koopmans
    f2, p2 = f_test_nested(y, pred1, 1, pred2, 2)
    f3, p3 = f_test_nested(y, pred1, 1, pred3, 4)
    print(f"\n  F-test bilinéaire vs Koopmans: F = {f2:.1f}, p = {p2:.1e}")
    print(f"  F-test 4 orbitales vs Koopmans: F = {f3:.1f}, p = {p3:.1e}")

    # Print equations
    print(f"\n  === Équations (ε en kJ/mol) ===")
    orb_names = X_dict['names']
    print(f"  Koopmans:    {name} = {beta1[1]:.4f}·{orb_names[0]} + {beta1[0]:.1f}")
    print(f"  Bilinéaire:  {name} = {beta2[1]:.4f}·{orb_names[0]} + {beta2[2]:.4f}·{orb_names[1]} + {beta2[0]:.1f}")
    print(f"  4 orbitales: {name} = {beta3[1]:.4f}·{orb_names[2]} + {beta3[2]:.4f}·{orb_names[0]} + {beta3[3]:.4f}·{orb_names[1]} + {beta3[4]:.4f}·{orb_names[3]} + {beta3[0]:.1f}")


def main():
    print("Loading data from RKF files...")
    records = load_data()
    print(f"Loaded {len(records)} conformers from {len(set(r['molecule'] for r in records))} molecules")

    molecules = [r['molecule'] for r in records]

    # --- IP ---
    y_ip = np.array([r['IP_kJ'] for r in records])
    run_regression('IP', y_ip, {
        'primary': np.array([-r['eH_kJ'] for r in records]),
        'complement': np.array([r['eL_kJ'] for r in records]),
        'secondary': np.array([-r['eHm1_kJ'] for r in records]),
        'complement2': np.array([r['eLp1_kJ'] for r in records]),
        'names': ['-ε_H', 'ε_L', '-ε_{H-1}', 'ε_{L+1}'],
    }, molecules)

    # --- EA ---
    y_ea = np.array([r['EA_kJ'] for r in records])
    run_regression('EA', y_ea, {
        'primary': np.array([-r['eL_kJ'] for r in records]),
        'complement': np.array([r['eH_kJ'] for r in records]),
        'secondary': np.array([-r['eLp1_kJ'] for r in records]),
        'complement2': np.array([r['eHm1_kJ'] for r in records]),
        'names': ['-ε_L', 'ε_H', '-ε_{L+1}', 'ε_{H-1}'],
    }, molecules)


if __name__ == '__main__':
    main()
