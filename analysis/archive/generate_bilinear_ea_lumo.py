#!/usr/bin/env amspython
"""
Section 3.1.2 — Au-delà de Koopmans : régression bilinéaire EA = a·LUMO + b·LUMO+1 + c.

Extrait LUMO et LUMO+1 depuis les RKF neutres TZ2P, puis compare :
  - Modèle univarié  : EA = a·LUMO + b        (Koopmans classique)
  - Modèle bilinéaire : EA = a·LUMO + b·LUMO+1 + c

Produit :
  - EA_bilinear_LUMO_LUMOp1.png  (parity plots)
  - EA_vs_LUMO_colored_by_LUMOp1.png
  - bilinear_ea_lumo_data.csv
"""

import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# --- Import collect_all_conformers sans passer par common/__init__ ---
_project_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "multi_molecule_data",
    _project_root / "common" / "multi_molecule_data.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
collect_all_conformers = _mod.collect_all_conformers

BATCH_DIR = _project_root / "1787066"
OUTPUT_DIR = _project_root / "results" / "figures"
EH_TO_EV = 27.211386245988


def extract_lumo_levels(workdir, mol_name):
    """
    Extrait LUMO et LUMO+1 depuis les neutral_tz2p RKFs.

    Returns:
        Dict {conf_num: {'LUMO_tz2p_eV': float, 'LUMOp1_tz2p_eV': float, ...}}
    """
    from scm.plams import KFFile

    plams_dir = workdir / "redox" / "calculations" / "plams_workdir"
    if not plams_dir.exists():
        return {}

    results = {}
    for rkf_dir in sorted(plams_dir.glob(f"{mol_name}_conf_*_neutral_tz2p")):
        rkf_path = rkf_dir / "adf.rkf"
        if not rkf_path.exists():
            continue

        dirname = rkf_dir.name
        try:
            conf_num = int(dirname.split("_conf_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue

        try:
            kf = KFFile(str(rkf_path))
            eps = np.array(kf.read('A', 'eps_A'), dtype=float)
            froc = np.array(kf.read('A', 'froc_A'), dtype=float)
            n_occ = int(np.sum(froc > 0))
            n_total = len(eps)

            if n_occ < 2 or n_occ + 1 >= n_total:
                continue

            results[conf_num] = {
                'HOMO_tz2p_eV': eps[n_occ - 1] * EH_TO_EV,
                'LUMO_tz2p_eV': eps[n_occ] * EH_TO_EV,
                'LUMOp1_tz2p_eV': eps[n_occ + 1] * EH_TO_EV,
                'LUMOp2_tz2p_eV': eps[n_occ + 2] * EH_TO_EV if n_occ + 2 < n_total else np.nan,
            }
        except Exception as e:
            print(f"  ! Erreur {mol_name} conf_{conf_num}: {e}")

    return results


def main():
    print("=" * 70)
    print("RÉGRESSION BILINÉAIRE : EA = a·LUMO + b·LUMO+1 + c")
    print("=" * 70)

    # 1. Load per-conformer data
    workdirs = sorted([d for d in BATCH_DIR.glob("*_workdir") if d.is_dir()])
    if not workdirs:
        print(f"Aucun workdir trouvé dans {BATCH_DIR}")
        sys.exit(1)

    df = collect_all_conformers(workdirs)
    print(f"\n{len(df)} conformères, {df['molecule_name'].nunique()} molécules")

    # 2. Extract LUMO / LUMO+1 from neutral_tz2p RKFs
    print("\nExtraction LUMO / LUMO+1 depuis neutral_tz2p...")
    lumo_data = []
    for workdir in workdirs:
        mol_name = workdir.name.replace('_workdir', '')
        levels = extract_lumo_levels(workdir, mol_name)
        for conf_num, data in levels.items():
            lumo_data.append({
                'molecule_name': mol_name,
                'conformer': conf_num,
                **data,
            })

    df_lumo = pd.DataFrame(lumo_data)
    print(f"  {len(df_lumo)} conformères avec LUMO/LUMO+1 extraits")

    # 3. Merge
    df = df.merge(df_lumo, on=['molecule_name', 'conformer'], how='inner')
    print(f"  {len(df)} conformères après fusion")

    # On utilise |EA| pour avoir la même convention que Koopmans : EA > 0 si capture favorable
    # EA_kJ dans les données = EA signé (positif si exothermique)
    required = ['EA_kJ', 'LUMO_tz2p_eV', 'LUMOp1_tz2p_eV']
    df_valid = df.dropna(subset=required).copy()
    df_valid['EA_kJ_abs'] = df_valid['EA_kJ'].abs()
    print(f"  {len(df_valid)} conformères avec données complètes")

    if len(df_valid) < 5:
        print("Pas assez de données pour la régression.")
        sys.exit(1)

    y = df_valid['EA_kJ_abs'].values
    x_lumo = df_valid['LUMO_tz2p_eV'].values
    x_lumo_p1 = df_valid['LUMOp1_tz2p_eV'].values

    # ===================================================================
    # Modèle 1 : |EA| = a·LUMO + b  (univarié, Koopmans classique)
    # ===================================================================
    slope1, intercept1, r1, p1, _ = stats.linregress(x_lumo, y)
    r2_1 = r1 ** 2
    n = len(y)
    k1 = 1
    r2_adj_1 = 1 - (1 - r2_1) * (n - 1) / (n - k1 - 1)
    rss1 = np.sum((y - (slope1 * x_lumo + intercept1)) ** 2)
    aic1 = n * np.log(rss1 / n) + 2 * (k1 + 1)

    print(f"\n{'='*60}")
    print("MODÈLE 1 — Koopmans classique : |EA| = a·LUMO + b")
    print(f"  a = {slope1:.2f}, b = {intercept1:.2f}")
    print(f"  R² = {r2_1:.4f}, R² adj = {r2_adj_1:.4f}")
    print(f"  AIC = {aic1:.2f}")
    print(f"  p = {p1:.2e}")

    # ===================================================================
    # Modèle 2 : |EA| = a·LUMO + b·LUMO+1 + c  (bilinéaire)
    # ===================================================================
    from sklearn.linear_model import LinearRegression

    X_bi = np.column_stack([x_lumo, x_lumo_p1])
    model2 = LinearRegression().fit(X_bi, y)
    y_pred2 = model2.predict(X_bi)
    rss2 = np.sum((y - y_pred2) ** 2)

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_2 = 1 - rss2 / ss_tot
    k2 = 2
    r2_adj_2 = 1 - (1 - r2_2) * (n - 1) / (n - k2 - 1)
    aic2 = n * np.log(rss2 / n) + 2 * (k2 + 1)

    a2, b2 = model2.coef_
    c2 = model2.intercept_

    # F-test : modèle 2 vs modèle 1
    df_num = k2 - k1
    df_den = n - k2 - 1
    f_stat = ((rss1 - rss2) / df_num) / (rss2 / df_den)
    p_ftest = 1 - stats.f.cdf(f_stat, df_num, df_den)

    # MAE / RMSE
    y_pred1 = slope1 * x_lumo + intercept1
    mae1 = np.mean(np.abs(y - y_pred1))
    rmse1 = np.sqrt(rss1 / n)
    mae2 = np.mean(np.abs(y - y_pred2))
    rmse2 = np.sqrt(rss2 / n)

    print(f"\n{'='*60}")
    print("MODÈLE 2 — Bilinéaire : |EA| = a·LUMO + b·LUMO+1 + c")
    print(f"  a = {a2:.2f}  (LUMO)")
    print(f"  b = {b2:.2f}  (LUMO+1)")
    print(f"  c = {c2:.2f}  (intercept)")
    print(f"  R² = {r2_2:.4f}, R² adj = {r2_adj_2:.4f}")
    print(f"  AIC = {aic2:.2f}")

    print(f"\n{'='*60}")
    print("COMPARAISON")
    print(f"  {'Métrique':<20} {'Univarié':>12} {'Bilinéaire':>12} {'Δ':>10}")
    print(f"  {'-'*54}")
    print(f"  {'R²':<20} {r2_1:>12.4f} {r2_2:>12.4f} {r2_2-r2_1:>+10.4f}")
    print(f"  {'R² ajusté':<20} {r2_adj_1:>12.4f} {r2_adj_2:>12.4f} {r2_adj_2-r2_adj_1:>+10.4f}")
    print(f"  {'MAE (kJ/mol)':<20} {mae1:>12.2f} {mae2:>12.2f} {mae2-mae1:>+10.2f}")
    print(f"  {'RMSE (kJ/mol)':<20} {rmse1:>12.2f} {rmse2:>12.2f} {rmse2-rmse1:>+10.2f}")
    print(f"  {'AIC':<20} {aic1:>12.2f} {aic2:>12.2f} {aic2-aic1:>+10.2f}")
    print(f"\n  F-test (M2 vs M1): F = {f_stat:.3f}, p = {p_ftest:.4e}")
    print(f"  {'→ LUMO+1 significatif' if p_ftest < 0.05 else '→ LUMO+1 non significatif'} (α = 0.05)")

    # ===================================================================
    # Coefficients t-test pour le modèle bilinéaire
    # ===================================================================
    X_aug = np.column_stack([np.ones(n), x_lumo, x_lumo_p1])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_pred_manual = X_aug @ beta
    residuals = y - y_pred_manual
    s2 = np.sum(residuals**2) / (n - 3)
    cov_beta = s2 * np.linalg.inv(X_aug.T @ X_aug)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_vals = beta / se_beta
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), n - 3))

    print(f"\n  Coefficients détaillés du modèle bilinéaire :")
    print(f"  {'Terme':<15} {'Coeff':>10} {'SE':>10} {'t':>8} {'p':>12}")
    print(f"  {'-'*55}")
    labels = ['intercept', 'LUMO', 'LUMO+1']
    for i, lbl in enumerate(labels):
        sig = ''
        if p_vals[i] < 0.05:
            sig = '*'
        if p_vals[i] < 0.01:
            sig = '**'
        if p_vals[i] < 0.001:
            sig = '***'
        print(f"  {lbl:<15} {beta[i]:>10.3f} {se_beta[i]:>10.3f} {t_vals[i]:>8.3f} {p_vals[i]:>11.4e} {sig}")

    # ===================================================================
    # Corrélation entre LUMO et LUMO+1
    # ===================================================================
    corr_lumo = np.corrcoef(x_lumo, x_lumo_p1)[0, 1]
    print(f"\n  Corrélation LUMO / LUMO+1 : r = {corr_lumo:.4f}")
    if abs(corr_lumo) > 0.9:
        print(f"  ⚠ Forte colinéarité (|r| > 0.9) — interpréter les coefficients avec prudence")

    r2_lumo_vs_lumop1 = corr_lumo ** 2
    vif = 1 / (1 - r2_lumo_vs_lumop1) if r2_lumo_vs_lumop1 < 1 else float('inf')
    print(f"  VIF = {vif:.2f}")

    # ===================================================================
    # FIGURE : Parity plot EA_DFT vs EA_pred pour les deux modèles
    # ===================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mol_names = df_valid['molecule_name'].values
    unique_mols = sorted(set(mol_names))
    cmap = plt.cm.get_cmap('tab20', max(len(unique_mols), 20))
    color_map = {mol: cmap(i) for i, mol in enumerate(unique_mols)}

    for ax, y_pred, title, r2, r2_adj, eq in [
        (axes[0], y_pred1,
         'Koopmans : |EA| = a·LUMO + b',
         r2_1, r2_adj_1,
         f'$y = {slope1:.1f}x {intercept1:+.1f}$'),
        (axes[1], y_pred2,
         'Bilinéaire : |EA| = a·LUMO + b·LUMO+1 + c',
         r2_2, r2_adj_2,
         f'$y = {a2:.1f}\\epsilon_{{L}} {b2:+.1f}\\epsilon_{{L+1}} {c2:+.1f}$'),
    ]:
        for mol in unique_mols:
            mask = mol_names == mol
            ax.scatter(y[mask], y_pred[mask], s=50, alpha=0.7,
                      color=color_map[mol], edgecolors='white', linewidths=0.5,
                      label=mol, zorder=3)

        vmin = min(y.min(), y_pred.min()) - 5
        vmax = max(y.max(), y_pred.max()) + 5
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, alpha=0.5, zorder=1)

        ax.set_xlabel('|EA| DFT (kJ/mol)', fontsize=11)
        ax.set_ylabel('|EA| prédit (kJ/mol)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        textstr = (f'$R^2 = {r2:.4f}$\n'
                   f'$R^2_{{adj}} = {r2_adj:.4f}$\n'
                   f'MAE = {mae:.1f} kJ/mol\n'
                   f'RMSE = {rmse:.1f} kJ/mol\n'
                   f'{eq}')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.grid(True, alpha=0.3, zorder=0)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='center right', fontsize=7,
               bbox_to_anchor=(1.12, 0.5), ncol=1)

    plt.tight_layout()
    outfile = OUTPUT_DIR / "EA_bilinear_LUMO_LUMOp1.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure: {outfile}")

    # ===================================================================
    # FIGURE 2 : Scatter |EA| vs LUMO coloré par LUMO+1
    # ===================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(x_lumo, y, c=x_lumo_p1, cmap='coolwarm', s=60,
                     edgecolors='white', linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax2, label='LUMO+1 (eV)')

    for idx in range(len(df_valid)):
        row = df_valid.iloc[idx]
        ax2.annotate(f"{row['molecule_name'][:4]}", (x_lumo[idx], y[idx]),
                    xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)

    ax2.set_xlabel('LUMO (eV)', fontsize=11)
    ax2.set_ylabel('|EA| (kJ/mol)', fontsize=11)
    ax2.set_title('|EA| vs LUMO — couleur = LUMO+1', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    outfile2 = OUTPUT_DIR / "EA_vs_LUMO_colored_by_LUMOp1.png"
    plt.savefig(outfile2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {outfile2}")

    # ===================================================================
    # Export CSV
    # ===================================================================
    csv_out = OUTPUT_DIR / "bilinear_ea_lumo_data.csv"
    df_valid[['molecule_name', 'conformer', 'weight', 'EA_kJ', 'EA_kJ_abs',
              'LUMO_tz2p_eV', 'LUMOp1_tz2p_eV', 'LUMOp2_tz2p_eV']].to_csv(csv_out, index=False)
    print(f"  CSV: {csv_out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
