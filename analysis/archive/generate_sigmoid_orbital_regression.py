#!/usr/bin/env amspython
"""
Test de fonctions sigmoïdes pour la pondération orbitale (IP & EA).

Variantes testées :
  1. Sigmoïde 1p : w(ε) = σ(k·(ε - ε_HOMO))       [steepness k]
  2. Sigmoïde 2p : w(ε) = σ(k·(ε - ε_HOMO - μ))    [steepness k + shift μ]
  3. Sigmoïde généralisée : w(ε) = σ(k·(ε - ε_HOMO))^n  [steepness k + power n]
  4. Double sigmoïde (bande) : σ(k·(ε-a)) · σ(-k·(ε-b))  [band-pass]
  5. Softmax : w_i = exp(β·ε_i) / Σ exp(β·ε_j)      [température inverse β]
  6. Bilinéaire sigmoïde : régression sur 2 descripteurs pondérés-sigmoïde

Comparaison avec Koopmans et bilinéaire HOMO+HOMO-1 / LUMO+LUMO+1.
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
from scipy.optimize import minimize

# --- Import collect_all_conformers ---
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


# ======================================================================
# EXTRACTION DES ORBITALES
# ======================================================================

def extract_all_orbitals(workdir, mol_name):
    from scm.plams import KFFile
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
            kf = KFFile(str(rkf_path))
            eps = np.array(kf.read('A', 'eps_A'), dtype=float) * EH_TO_EV
            froc = np.array(kf.read('A', 'froc_A'), dtype=float)
            n_occ = int(np.sum(froc > 0))
            results[conf_num] = {
                'eps': eps, 'occ': froc,
                'n_occ': n_occ, 'n_total': len(eps),
            }
        except Exception as e:
            print(f"  ! Erreur {mol_name} conf_{conf_num}: {e}")
    return results


# ======================================================================
# FONCTIONS SIGMOÏDES
# ======================================================================

def sigmoid(x):
    """Sigmoïde standard σ(x) = 1/(1+exp(-x)), stable numériquement."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def desc_sigmoid_1p_ip(orb, k):
    """Sigmoïde 1-param pour IP : w(ε) = σ(k·(ε - ε_HOMO))"""
    eps = orb['eps'][:orb['n_occ']]
    eps_homo = eps[-1]
    w = sigmoid(k * (eps - eps_homo))
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_homo


def desc_sigmoid_2p_ip(orb, k, mu):
    """Sigmoïde 2-param : w(ε) = σ(k·(ε - ε_HOMO + μ))"""
    eps = orb['eps'][:orb['n_occ']]
    eps_homo = eps[-1]
    w = sigmoid(k * (eps - eps_homo + mu))
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_homo


def desc_sigmoid_power_ip(orb, k, n_pow):
    """Sigmoïde généralisée : w(ε) = σ(k·(ε - ε_HOMO))^n"""
    eps = orb['eps'][:orb['n_occ']]
    eps_homo = eps[-1]
    w = sigmoid(k * (eps - eps_homo)) ** n_pow
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_homo


def desc_double_sigmoid_ip(orb, k, width):
    """Double sigmoïde (bande) : σ(k·(ε-a))·σ(-k·(ε-b)), bande autour de HOMO."""
    eps = orb['eps'][:orb['n_occ']]
    eps_homo = eps[-1]
    a = eps_homo - width  # borne basse
    b = eps_homo + 0.1    # borne haute (juste au-dessus de HOMO)
    w = sigmoid(k * (eps - a)) * sigmoid(-k * (eps - b))
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_homo


def desc_softmax_ip(orb, beta):
    """Softmax : w_i = exp(β·ε_i) / Σ exp(β·ε_j)"""
    eps = orb['eps'][:orb['n_occ']]
    # Shift for numerical stability
    eps_shifted = beta * (eps - eps.max())
    w = np.exp(eps_shifted)
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps[-1]


# --- EA versions (virtuelles, inversées) ---

def desc_sigmoid_1p_ea(orb, k):
    """Sigmoïde 1-param pour EA : w(ε) = σ(-k·(ε - ε_LUMO))"""
    n_occ = orb['n_occ']
    eps = orb['eps'][n_occ:]
    eps_lumo = eps[0]
    w = sigmoid(-k * (eps - eps_lumo))
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_lumo


def desc_sigmoid_2p_ea(orb, k, mu):
    """Sigmoïde 2-param : w(ε) = σ(-k·(ε - ε_LUMO - μ))"""
    n_occ = orb['n_occ']
    eps = orb['eps'][n_occ:]
    eps_lumo = eps[0]
    w = sigmoid(-k * (eps - eps_lumo - mu))
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_lumo


def desc_sigmoid_power_ea(orb, k, n_pow):
    """Sigmoïde généralisée EA : w(ε) = σ(-k·(ε - ε_LUMO))^n"""
    n_occ = orb['n_occ']
    eps = orb['eps'][n_occ:]
    eps_lumo = eps[0]
    w = sigmoid(-k * (eps - eps_lumo)) ** n_pow
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_lumo


def desc_double_sigmoid_ea(orb, k, width):
    """Double sigmoïde EA : bande autour de LUMO."""
    n_occ = orb['n_occ']
    eps = orb['eps'][n_occ:]
    eps_lumo = eps[0]
    a = eps_lumo - 0.1
    b = eps_lumo + width
    w = sigmoid(k * (eps - a)) * sigmoid(-k * (eps - b))
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps_lumo


def desc_softmax_ea(orb, beta):
    """Softmax EA : w_i = exp(-β·ε_i) / Σ exp(-β·ε_j) (favorise basses énergies)"""
    n_occ = orb['n_occ']
    eps = orb['eps'][n_occ:]
    eps_shifted = -beta * (eps - eps.min())
    w = np.exp(eps_shifted)
    s = np.sum(w)
    return np.sum(w * eps) / s if s > 1e-30 else eps[0]


# ======================================================================
# RÉGRESSION & OPTIMISATION
# ======================================================================

def linreg_metrics(descriptors, targets):
    x = np.array(descriptors)
    y = np.array(targets)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {'r2': 0, 'mae': 999, 'rmse': 999, 'n': 0}
    slope, intercept, r, p, _ = stats.linregress(x, y)
    y_pred = slope * x + intercept
    res = y - y_pred
    n = len(x)
    r2 = r ** 2
    return {
        'r2': r2,
        'r2_adj': 1 - (1 - r2) * (n - 1) / (n - 2),
        'mae': np.mean(np.abs(res)),
        'rmse': np.sqrt(np.mean(res ** 2)),
        'slope': slope, 'intercept': intercept, 'p': p, 'n': n,
        'y_pred': y_pred, 'y': y, 'x': x,
    }


def bilinear_metrics(x1, x2, y):
    """Régression bilinéaire y = a·x1 + b·x2 + c."""
    from sklearn.linear_model import LinearRegression
    X = np.column_stack([x1, x2])
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    res = y - y_pred
    n = len(y)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(res ** 2)
    r2 = 1 - ss_res / ss_tot
    k = 2
    return {
        'r2': r2,
        'r2_adj': 1 - (1 - r2) * (n - 1) / (n - k - 1),
        'mae': np.mean(np.abs(res)),
        'rmse': np.sqrt(np.mean(res ** 2)),
        'coefs': (model.coef_[0], model.coef_[1], model.intercept_),
        'n': n, 'y_pred': y_pred, 'y': y,
        'aic': n * np.log(ss_res / n) + 2 * (k + 1),
    }


def optimize_1p(orb_list, targets, desc_func, bounds, n_grid=300):
    """Optimise un paramètre scalaire."""
    grid = np.linspace(bounds[0], bounds[1], n_grid)
    best_rmse, best_p = 999, grid[0]
    rmse_curve = []
    for p in grid:
        descs = [desc_func(o, p) for o in orb_list]
        m = linreg_metrics(descs, targets)
        rmse_curve.append(m['rmse'])
        if m['rmse'] < best_rmse:
            best_rmse = m['rmse']
            best_p = p
    # Refine
    from scipy.optimize import minimize_scalar
    delta = (bounds[1] - bounds[0]) / n_grid * 5
    res = minimize_scalar(
        lambda p: linreg_metrics([desc_func(o, p) for o in orb_list], targets)['rmse'],
        bounds=(max(bounds[0], best_p - delta), min(bounds[1], best_p + delta)),
        method='bounded')
    if res.fun < best_rmse:
        best_p = res.x
    descs = [desc_func(o, best_p) for o in orb_list]
    final = linreg_metrics(descs, targets)
    final['best_param'] = best_p
    final['rmse_curve'] = np.array(rmse_curve)
    final['param_grid'] = grid
    return final


def optimize_2p(orb_list, targets, desc_func, bounds, n_grid=40):
    """Optimise 2 paramètres par grid search + raffinement."""
    g1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
    g2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
    best_rmse, best_params = 999, (g1[0], g2[0])
    for p1 in g1:
        for p2 in g2:
            descs = [desc_func(o, p1, p2) for o in orb_list]
            m = linreg_metrics(descs, targets)
            if m['rmse'] < best_rmse:
                best_rmse = m['rmse']
                best_params = (p1, p2)
    # Refine
    def obj(params):
        descs = [desc_func(o, params[0], params[1]) for o in orb_list]
        return linreg_metrics(descs, targets)['rmse']
    res = minimize(obj, best_params, bounds=[bounds[0], bounds[1]], method='L-BFGS-B')
    if res.fun < best_rmse:
        best_params = tuple(res.x)
    descs = [desc_func(o, *best_params) for o in orb_list]
    final = linreg_metrics(descs, targets)
    final['best_param'] = best_params
    return final


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("SIGMOÏDES ORBITALES — IP & EA")
    print("=" * 70)

    # Load data
    workdirs = sorted([d for d in BATCH_DIR.glob("*_workdir") if d.is_dir()])
    df = collect_all_conformers(workdirs)
    print(f"\n{len(df)} conformères, {df['molecule_name'].nunique()} molécules")

    # Extract orbitals
    print("\nExtraction orbitales neutral_tz2p...")
    orb_rows = []
    for workdir in workdirs:
        mol_name = workdir.name.replace('_workdir', '')
        orbitals = extract_all_orbitals(workdir, mol_name)
        for conf_num, orb_data in orbitals.items():
            orb_rows.append({
                'molecule_name': mol_name, 'conformer': conf_num,
                'orbital_data': orb_data,
            })
    df_orb = pd.DataFrame(orb_rows)
    df = df.merge(df_orb, on=['molecule_name', 'conformer'], how='inner')

    df_ip = df.dropna(subset=['IP_kJ']).copy()
    df_ea = df.dropna(subset=['EA_kJ']).copy()
    df_ea['EA_kJ_abs'] = df_ea['EA_kJ'].abs()

    ip_targets = df_ip['IP_kJ'].values
    ea_targets = df_ea['EA_kJ_abs'].values
    ip_orbs = df_ip['orbital_data'].tolist()
    ea_orbs = df_ea['orbital_data'].tolist()

    print(f"  {len(ip_targets)} conformères IP, {len(ea_targets)} conformères EA")

    # ==================================================================
    # BASELINES
    # ==================================================================
    # Koopmans
    koop_ip = linreg_metrics([o['eps'][o['n_occ']-1] for o in ip_orbs], ip_targets)
    koop_ea = linreg_metrics([o['eps'][o['n_occ']] for o in ea_orbs], ea_targets)

    # Bilinéaire HOMO + HOMO-1
    x1_ip = np.array([o['eps'][o['n_occ']-1] for o in ip_orbs])
    x2_ip = np.array([o['eps'][o['n_occ']-2] for o in ip_orbs])
    bilin_ip = bilinear_metrics(x1_ip, x2_ip, ip_targets)

    # Bilinéaire LUMO + LUMO+1
    x1_ea = np.array([o['eps'][o['n_occ']] for o in ea_orbs])
    x2_ea = np.array([o['eps'][o['n_occ']+1] for o in ea_orbs])
    bilin_ea = bilinear_metrics(x1_ea, x2_ea, ea_targets)

    # ==================================================================
    # SIGMOÏDES — IP
    # ==================================================================
    print(f"\n{'='*70}")
    print("IP — SIGMOÏDES")
    print(f"{'='*70}")

    ip_models = {}
    ip_models['Koopmans (HOMO)'] = koop_ip
    ip_models['Bilinéaire H+H-1'] = bilin_ip

    print(f"\n  {'Koopmans (HOMO)':<35} R²={koop_ip['r2']:.6f}  MAE={koop_ip['mae']:.3f}  RMSE={koop_ip['rmse']:.3f}")
    a, b, c = bilin_ip['coefs']
    print(f"  {'Bilinéaire H+H-1':<35} R²={bilin_ip['r2']:.6f}  MAE={bilin_ip['mae']:.3f}  RMSE={bilin_ip['rmse']:.3f}  [{a:.1f}·H {b:+.1f}·H-1 {c:+.1f}]")

    # 1. Sigmoïde 1-param
    res = optimize_1p(ip_orbs, ip_targets, desc_sigmoid_1p_ip, (0.1, 50))
    ip_models['Sigmoïde 1p'] = res
    print(f"  {'Sigmoïde 1p (k)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  k={res['best_param']:.3f}")

    # 2. Sigmoïde 2-param (k + shift μ)
    res = optimize_2p(ip_orbs, ip_targets, desc_sigmoid_2p_ip, [(0.5, 50), (-3, 3)])
    ip_models['Sigmoïde 2p (k,μ)'] = res
    print(f"  {'Sigmoïde 2p (k,μ)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  params={res['best_param']}")

    # 3. Sigmoïde puissance (k + n)
    res = optimize_2p(ip_orbs, ip_targets, desc_sigmoid_power_ip, [(0.5, 50), (0.1, 5)])
    ip_models['Sigmoïde power (k,n)'] = res
    print(f"  {'Sigmoïde power (k,n)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  params={res['best_param']}")

    # 4. Double sigmoïde (bande)
    res = optimize_2p(ip_orbs, ip_targets, desc_double_sigmoid_ip, [(1, 50), (0.1, 5)])
    ip_models['Double sigmoïde'] = res
    print(f"  {'Double sigmoïde (k,w)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  params={res['best_param']}")

    # 5. Softmax
    res = optimize_1p(ip_orbs, ip_targets, desc_softmax_ip, (0.1, 30))
    ip_models['Softmax'] = res
    print(f"  {'Softmax (β)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  β={res['best_param']:.3f}")

    # ==================================================================
    # SIGMOÏDES — EA
    # ==================================================================
    print(f"\n{'='*70}")
    print("EA — SIGMOÏDES")
    print(f"{'='*70}")

    ea_models = {}
    ea_models['Koopmans (LUMO)'] = koop_ea
    ea_models['Bilinéaire L+L+1'] = bilin_ea

    print(f"\n  {'Koopmans (LUMO)':<35} R²={koop_ea['r2']:.6f}  MAE={koop_ea['mae']:.3f}  RMSE={koop_ea['rmse']:.3f}")
    a, b, c = bilin_ea['coefs']
    print(f"  {'Bilinéaire L+L+1':<35} R²={bilin_ea['r2']:.6f}  MAE={bilin_ea['mae']:.3f}  RMSE={bilin_ea['rmse']:.3f}  [{a:.1f}·L {b:+.1f}·L+1 {c:+.1f}]")

    # 1. Sigmoïde 1-param
    res = optimize_1p(ea_orbs, ea_targets, desc_sigmoid_1p_ea, (0.1, 50))
    ea_models['Sigmoïde 1p'] = res
    print(f"  {'Sigmoïde 1p (k)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  k={res['best_param']:.3f}")

    # 2. Sigmoïde 2-param
    res = optimize_2p(ea_orbs, ea_targets, desc_sigmoid_2p_ea, [(0.5, 50), (-2, 5)])
    ea_models['Sigmoïde 2p (k,μ)'] = res
    print(f"  {'Sigmoïde 2p (k,μ)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  params={res['best_param']}")

    # 3. Sigmoïde puissance
    res = optimize_2p(ea_orbs, ea_targets, desc_sigmoid_power_ea, [(0.5, 50), (0.1, 5)])
    ea_models['Sigmoïde power (k,n)'] = res
    print(f"  {'Sigmoïde power (k,n)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  params={res['best_param']}")

    # 4. Double sigmoïde
    res = optimize_2p(ea_orbs, ea_targets, desc_double_sigmoid_ea, [(1, 50), (0.1, 5)])
    ea_models['Double sigmoïde'] = res
    print(f"  {'Double sigmoïde (k,w)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  params={res['best_param']}")

    # 5. Softmax
    res = optimize_1p(ea_orbs, ea_targets, desc_softmax_ea, (0.1, 30))
    ea_models['Softmax'] = res
    print(f"  {'Softmax (β)':<35} R²={res['r2']:.6f}  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  β={res['best_param']:.3f}")

    # ==================================================================
    # TABLEAUX RÉCAPITULATIFS
    # ==================================================================
    for label, models, ref_key in [
        ('IP', ip_models, 'Koopmans (HOMO)'),
        ('EA', ea_models, 'Koopmans (LUMO)'),
    ]:
        ref_rmse = models[ref_key]['rmse']
        ref_mae = models[ref_key]['mae']
        print(f"\n{'='*85}")
        print(f"  RÉCAPITULATIF {label}")
        print(f"{'='*85}")
        print(f"  {'Modèle':<35} {'R²':>8} {'R² adj':>8} {'MAE':>8} {'RMSE':>8} {'ΔMAE':>8} {'ΔRMSE':>8}")
        print(f"  {'-'*84}")
        for name, res in sorted(models.items(), key=lambda x: x[1]['rmse']):
            r2a = res.get('r2_adj', res['r2'])
            dmae = res['mae'] - ref_mae
            drmse = res['rmse'] - ref_rmse
            marker = ' <<<' if res['rmse'] < ref_rmse and name != ref_key else ''
            print(f"  {name:<35} {res['r2']:>8.4f} {r2a:>8.4f} "
                  f"{res['mae']:>7.3f} {res['rmse']:>7.3f} "
                  f"{dmae:>+7.3f} {drmse:>+7.3f}{marker}")

    # ==================================================================
    # FIGURES
    # ==================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, models, ref_key in [
        ('IP', ip_models, 'Koopmans (HOMO)'),
        ('EA', ea_models, 'Koopmans (LUMO)'),
    ]:
        # Parity : best sigmoid vs Koopmans vs Bilinéaire
        bilin_key = 'Bilinéaire H+H-1' if label == 'IP' else 'Bilinéaire L+L+1'
        best_sig = min([k for k in models if k not in (ref_key, bilin_key)],
                       key=lambda k: models[k]['rmse'])

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, key in zip(axes, [ref_key, bilin_key, best_sig]):
            res = models[key]
            y_true = res['y']
            y_pred = res['y_pred']
            ax.scatter(y_true, y_pred, s=45, alpha=0.7, c='steelblue',
                      edgecolors='white', linewidths=0.5, zorder=3)
            vmin = min(y_true.min(), y_pred.min()) - 5
            vmax = max(y_true.max(), y_pred.max()) + 5
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, alpha=0.5)
            ax.set_xlabel(f'{label} DFT (kJ/mol)', fontsize=10)
            ax.set_ylabel(f'{label} prédit (kJ/mol)', fontsize=10)
            ax.set_title(key, fontsize=11, fontweight='bold')
            txt = (f"$R^2 = {res['r2']:.4f}$\n"
                   f"MAE = {res['mae']:.3f} kJ/mol\n"
                   f"RMSE = {res['rmse']:.3f} kJ/mol")
            if 'best_param' in res:
                p = res['best_param']
                if isinstance(p, tuple):
                    txt += f"\nparams = ({p[0]:.2f}, {p[1]:.2f})"
                else:
                    txt += f"\nparam = {p:.3f}"
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.grid(True, alpha=0.3, zorder=0)

        plt.suptitle(f'{label} — Koopmans vs Bilinéaire vs Meilleure Sigmoïde',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        outfile = OUTPUT_DIR / f"{label}_sigmoid_parity.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  Figure: {outfile}")

    # RMSE vs k pour sigmoïde 1-param
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, label, models, ref_key in [
        (axes[0], 'IP', ip_models, 'Koopmans (HOMO)'),
        (axes[1], 'EA', ea_models, 'Koopmans (LUMO)'),
    ]:
        res_sig = models['Sigmoïde 1p']
        if 'param_grid' in res_sig and 'rmse_curve' in res_sig:
            ax.plot(res_sig['param_grid'], res_sig['rmse_curve'], 'b-', linewidth=1.5)
            ax.axhline(models[ref_key]['rmse'], color='r', linestyle='--',
                      alpha=0.7, label=f'Koopmans: {models[ref_key]["rmse"]:.3f}')
            bilin_key = 'Bilinéaire H+H-1' if label == 'IP' else 'Bilinéaire L+L+1'
            ax.axhline(models[bilin_key]['rmse'], color='g', linestyle=':',
                      alpha=0.7, label=f'Bilinéaire: {models[bilin_key]["rmse"]:.3f}')
            ax.axvline(res_sig['best_param'], color='orange', linestyle='--',
                      alpha=0.7)
            ax.scatter([res_sig['best_param']], [res_sig['rmse']], c='orange',
                      s=100, zorder=5, marker='*',
                      label=f'Optimal k={res_sig["best_param"]:.2f}: {res_sig["rmse"]:.3f}')
            ax.set_xlabel('Steepness k', fontsize=10)
            ax.set_ylabel('RMSE (kJ/mol)', fontsize=10)
            ax.set_title(f'{label} — Sigmoïde σ(k·(ε−ε_ref))', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = OUTPUT_DIR / "sigmoid_rmse_vs_k.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {outfile}")

    # Bar chart final
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, label, models, ref_key in [
        (axes[0], 'IP', ip_models, 'Koopmans (HOMO)'),
        (axes[1], 'EA', ea_models, 'Koopmans (LUMO)'),
    ]:
        sorted_m = sorted(models.items(), key=lambda x: x[1]['rmse'])
        names = [n for n, _ in sorted_m]
        rmses = [r['rmse'] for _, r in sorted_m]
        maes = [r['mae'] for _, r in sorted_m]
        x_pos = np.arange(len(names))
        w = 0.35
        ax.barh(x_pos - w/2, rmses, w, label='RMSE', color='steelblue', alpha=0.8)
        ax.barh(x_pos + w/2, maes, w, label='MAE', color='coral', alpha=0.8)
        ax.axvline(models[ref_key]['rmse'], color='red', linestyle='--', alpha=0.4)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Erreur (kJ/mol)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    plt.suptitle('Sigmoïdes vs Koopmans vs Bilinéaire', fontsize=13, fontweight='bold')
    plt.tight_layout()
    outfile = OUTPUT_DIR / "sigmoid_comparison_bar.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {outfile}")

    print("\nDone.")


if __name__ == '__main__':
    main()
