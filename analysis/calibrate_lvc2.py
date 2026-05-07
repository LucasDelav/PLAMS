#!/usr/bin/env python3
"""
Calibration empirique du LVC2 et analyse des descripteurs pour prédire le RMSD.

Approche 1 : Régression linéaire E_relax_réel = a × E_relax_LVC2 + b
Approche 2 : Descripteurs (Huang-Rhys, n_imag) pour prédire/corriger l'erreur LVC2

Usage:
    python3 analysis/calibrate_lvc2.py
"""

import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "rapport" / "data" / "all_methods_comparison_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "rapport" / "data"

EH_TO_KJ = 2625.499638


def load_data():
    """Charge les données LVC2 et réelles."""
    with open(DATA_CSV) as f:
        rows = list(csv.DictReader(f))
    return rows


def approche1_calibration(rows):
    """Approche 1 : régression linéaire simple du LVC2."""
    print("=" * 70)
    print("APPROCHE 1 : CALIBRATION EMPIRIQUE DU LVC2")
    print("=" * 70)

    results = {}

    for process, real_key, lvc2_key, label in [
        ('ox', 'E_def_real_ox', 'E_def_LVC2_ox', 'Oxydation'),
        ('red', 'E_def_real_red', 'E_def_LVC2_red', 'Réduction'),
    ]:
        real = np.array([float(r[real_key]) for r in rows])
        lvc2 = np.array([float(r[lvc2_key]) for r in rows])
        mask = np.isfinite(real) & np.isfinite(lvc2) & (real != 0) & (lvc2 != 0)
        real, lvc2 = real[mask], lvc2[mask]
        names = [r['molecule'] for r, m in zip(rows, mask) if m]

        # Régression
        fit = linregress(lvc2, real)
        cal = fit.slope * lvc2 + fit.intercept

        mae_brut = np.mean(np.abs(real - lvc2))
        mae_cal = np.mean(np.abs(real - cal))
        rmse_brut = np.sqrt(np.mean((real - lvc2) ** 2))
        rmse_cal = np.sqrt(np.mean((real - cal) ** 2))

        print(f"\n  {label} (n={len(real)}):")
        print(f"    E_réel = {fit.slope:.3f} × LVC2 + ({fit.intercept:+.1f})")
        print(f"    R² = {fit.rvalue**2:.3f}")
        print(f"    MAE brut:   {mae_brut:.1f} kJ/mol")
        print(f"    MAE calibré:{mae_cal:.1f} kJ/mol  (amélioration {(1-mae_cal/mae_brut)*100:.0f}%)")
        print(f"    RMSE brut:  {rmse_brut:.1f} kJ/mol")
        print(f"    RMSE calibré:{rmse_cal:.1f} kJ/mol")

        # Leave-one-out cross-validation
        loo_errors = []
        for i in range(len(real)):
            train_x = np.delete(lvc2, i)
            train_y = np.delete(real, i)
            loo_fit = linregress(train_x, train_y)
            pred = loo_fit.slope * lvc2[i] + loo_fit.intercept
            loo_errors.append(abs(real[i] - pred))
        mae_loo = np.mean(loo_errors)
        print(f"    MAE LOO-CV: {mae_loo:.1f} kJ/mol")

        # Détail par molécule
        print(f"\n    {'Molécule':<15} {'Réel':>8} {'LVC2':>8} {'Calibré':>8} {'Err brut':>9} {'Err cal':>9}")
        print(f"    {'-'*58}")
        for j in range(len(real)):
            err_brut = real[j] - lvc2[j]
            err_cal = real[j] - cal[j]
            print(f"    {names[j]:<15} {real[j]:>8.1f} {lvc2[j]:>8.1f} {cal[j]:>8.1f} {err_brut:>+9.1f} {err_cal:>+9.1f}")

        results[process] = {
            'slope': float(fit.slope),
            'intercept': float(fit.intercept),
            'R2': float(fit.rvalue ** 2),
            'MAE_brut': float(mae_brut),
            'MAE_calibre': float(mae_cal),
            'MAE_LOO': float(mae_loo),
            'n': int(len(real)),
        }

    return results


def approche2_descripteurs(rows):
    """Approche 2 : descripteurs pour prédire le résidu LVC2."""
    print("\n" + "=" * 70)
    print("APPROCHE 2 : DESCRIPTEURS PRÉDICTIFS DU RMSD / RÉSIDU")
    print("=" * 70)

    real_red = np.array([float(r['E_def_real_red']) for r in rows])
    lvc2_red = np.array([float(r['E_def_LVC2_red']) for r in rows])
    rmsd_red = np.array([float(r['RMSD_red']) for r in rows])
    residual_red = real_red - lvc2_red
    names = [r['molecule'] for r in rows]

    real_ox = np.array([float(r['E_def_real_ox']) for r in rows])
    lvc2_ox = np.array([float(r['E_def_LVC2_ox']) for r in rows])
    rmsd_ox = np.array([float(r['RMSD_ox']) for r in rows])
    residual_ox = real_ox - lvc2_ox

    # --- 2a. |E_relax_LVC2| comme proxy de Huang-Rhys ---
    print("\n  2a. |E_relax_LVC2| comme proxy du facteur de Huang-Rhys")
    print("  " + "-" * 50)

    # |LVC2| vs RMSD
    for label, lvc2, rmsd, residual in [
        ('Réduction', lvc2_red, rmsd_red, residual_red),
        ('Oxydation', lvc2_ox, rmsd_ox, residual_ox),
    ]:
        fit_rmsd = linregress(np.abs(lvc2), rmsd)
        fit_res = linregress(np.abs(lvc2), residual)
        print(f"\n    {label}:")
        print(f"      |LVC2| vs RMSD:    R²={fit_rmsd.rvalue**2:.3f}, p={fit_rmsd.pvalue:.4f}")
        print(f"      |LVC2| vs résidu:  R²={fit_res.rvalue**2:.3f}, p={fit_res.pvalue:.4f}")

    # --- 2c. RMSD vs résidu directement ---
    print("\n  2c. RMSD vs résidu LVC2")
    print("  " + "-" * 50)

    for label, rmsd, residual, lvc2, real in [
        ('Réduction', rmsd_red, residual_red, lvc2_red, real_red),
        ('Oxydation', rmsd_ox, residual_ox, lvc2_ox, real_ox),
    ]:
        fit = linregress(rmsd, residual)
        print(f"\n    {label}:")
        print(f"      RMSD vs résidu: R²={fit.rvalue**2:.3f}, p={fit.pvalue:.4f}")
        print(f"      résidu = {fit.slope:.1f} × RMSD + ({fit.intercept:+.1f})")

        # Modèle combiné : LVC2 calibré + correction RMSD
        # Mais RMSD n'est pas connu a priori... sauf si on le prédit

    # --- Régression multilinéaire : LVC2 + |LVC2|² ---
    print("\n  2d. Régression quadratique : E_réel = a × LVC2 + b × LVC2² + c")
    print("  " + "-" * 50)

    for label, lvc2, real in [
        ('Réduction', lvc2_red, real_red),
        ('Oxydation', lvc2_ox, real_ox),
    ]:
        # Modèle quadratique
        X = np.column_stack([lvc2, lvc2 ** 2, np.ones_like(lvc2)])
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, real, rcond=None)
        pred = X @ coeffs
        mae = np.mean(np.abs(real - pred))
        ss_res = np.sum((real - pred) ** 2)
        ss_tot = np.sum((real - np.mean(real)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # LOO-CV
        loo_errors = []
        for i in range(len(real)):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(real, i)
            c_loo, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
            pred_i = X[i] @ c_loo
            loo_errors.append(abs(real[i] - pred_i))
        mae_loo = np.mean(loo_errors)

        print(f"\n    {label}:")
        print(f"      a={coeffs[0]:.4f}, b={coeffs[1]:.6f}, c={coeffs[2]:.1f}")
        print(f"      R² = {r2:.3f}, MAE = {mae:.1f} kJ/mol, MAE LOO = {mae_loo:.1f} kJ/mol")

    return {}


def main():
    rows = load_data()
    print(f"Données chargées: {len(rows)} molécules\n")

    # Approche 1
    cal_results = approche1_calibration(rows)

    # Sauvegarder les coefficients
    cal_file = OUTPUT_DIR / "lvc2_calibration.json"
    with open(cal_file, 'w') as f:
        json.dump(cal_results, f, indent=2)
    print(f"\n  Coefficients sauvegardés: {cal_file}")

    # Approche 2
    approche2_descripteurs(rows)

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == '__main__':
    main()
