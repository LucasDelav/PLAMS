#!/usr/bin/env amspython
"""
Section 3.1.2 — Au-delà de Koopmans : pondération orbitale optimisée.

Teste différentes distributions de pondération sur TOUTES les orbitales
pour prédire IP et EA :
  - Koopmans classique (HOMO ou LUMO seul)
  - Bilinéaire (HOMO + HOMO-1, LUMO + LUMO+1)
  - Fermi-Dirac : w(ε) = 1 / (1 + exp((ε_F - ε)/kT))
  - Exponentielle : w(ε) = exp(-α·|ε - ε_ref|)
  - Gaussienne : w(ε) = exp(-(ε - ε_ref)² / (2σ²))
  - Top-N : poids uniforme sur les N orbitales les plus proches de la frontière
  - Lorentzienne : w(ε) = 1 / (1 + ((ε - ε_ref)/γ)²)

Pour chaque distribution, on optimise le(s) paramètre(s) pour minimiser le RMSE
de la régression linéaire :  IP = a·descriptor + b  (idem EA).

Données : 21 catéchols du batch 1787066, toutes orbitales TZ2P.
"""

import sys
from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar, minimize

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
    """
    Extrait TOUTES les énergies orbitales + occupations depuis neutral_tz2p.

    Returns:
        Dict {conf_num: {
            'eps': np.ndarray (en eV),
            'occ': np.ndarray,
            'n_occ': int,
            'n_total': int
        }}
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
                'eps': eps,
                'occ': froc,
                'n_occ': n_occ,
                'n_total': len(eps),
            }
        except Exception as e:
            print(f"  ! Erreur {mol_name} conf_{conf_num}: {e}")

    return results


# ======================================================================
# DISTRIBUTIONS DE PONDÉRATION
# ======================================================================

def weights_fermi_dirac(eps_occupied, eps_homo, kT):
    """Fermi-Dirac : w(ε) = 1 / (1 + exp((ε_HOMO - ε) / kT))"""
    if kT <= 0:
        return np.zeros_like(eps_occupied)
    x = (eps_homo - eps_occupied) / kT
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(x))


def weights_exponential(eps_occupied, eps_homo, alpha):
    """Exponentielle décroissante depuis HOMO : w(ε) = exp(-α·(ε_HOMO - ε))"""
    if alpha <= 0:
        return np.ones_like(eps_occupied)
    return np.exp(-alpha * (eps_homo - eps_occupied))


def weights_gaussian(eps_occupied, eps_ref, sigma):
    """Gaussienne centrée sur ε_ref : w(ε) = exp(-(ε - ε_ref)² / (2σ²))"""
    if sigma <= 0:
        return np.zeros_like(eps_occupied)
    return np.exp(-0.5 * ((eps_occupied - eps_ref) / sigma) ** 2)


def weights_lorentzian(eps_occupied, eps_ref, gamma):
    """Lorentzienne : w(ε) = 1 / (1 + ((ε - ε_ref)/γ)²)"""
    if gamma <= 0:
        return np.zeros_like(eps_occupied)
    return 1.0 / (1.0 + ((eps_occupied - eps_ref) / gamma) ** 2)


def weights_top_n(n_orbitals, n_top):
    """Top-N : poids 1 pour les N orbitales les plus proches de la frontière."""
    w = np.zeros(n_orbitals)
    n_top = min(int(n_top), n_orbitals)
    if n_top > 0:
        w[-n_top:] = 1.0
    return w


def weights_top_n_virtual(n_orbitals, n_top):
    """Top-N virtuel : poids 1 pour les N premières virtuelles."""
    w = np.zeros(n_orbitals)
    n_top = min(int(n_top), n_orbitals)
    if n_top > 0:
        w[:n_top] = 1.0
    return w


# ======================================================================
# DESCRIPTEUR : MOYENNE PONDÉRÉE DES ÉNERGIES ORBITALES
# ======================================================================

def compute_weighted_descriptor_ip(orbital_data, weight_func, params):
    """
    Calcule le descripteur pondéré pour l'IP à partir des orbitales occupées.

    descriptor = Σ w_i * ε_i / Σ w_i  (moyenne pondérée)
    """
    eps_occ = orbital_data['eps'][:orbital_data['n_occ']]
    eps_homo = eps_occ[-1]

    if weight_func == 'fermi_dirac':
        w = weights_fermi_dirac(eps_occ, eps_homo, params[0])
    elif weight_func == 'exponential':
        w = weights_exponential(eps_occ, eps_homo, params[0])
    elif weight_func == 'gaussian':
        w = weights_gaussian(eps_occ, eps_homo, params[0])
    elif weight_func == 'lorentzian':
        w = weights_lorentzian(eps_occ, eps_homo, params[0])
    elif weight_func == 'top_n':
        w = weights_top_n(len(eps_occ), params[0])
    else:
        raise ValueError(f"Unknown weight function: {weight_func}")

    w_sum = np.sum(w)
    if w_sum < 1e-30:
        return eps_homo  # fallback to HOMO
    return np.sum(w * eps_occ) / w_sum


def compute_weighted_descriptor_ea(orbital_data, weight_func, params):
    """
    Calcule le descripteur pondéré pour l'EA à partir des orbitales virtuelles.
    """
    n_occ = orbital_data['n_occ']
    eps_virt = orbital_data['eps'][n_occ:]
    eps_lumo = eps_virt[0]

    if weight_func == 'fermi_dirac':
        # Inversé : w(ε) = 1 / (1 + exp((ε - ε_LUMO) / kT))
        kT = params[0]
        if kT <= 0:
            w = np.zeros_like(eps_virt)
        else:
            x = (eps_virt - eps_lumo) / kT
            x = np.clip(x, -500, 500)
            w = 1.0 / (1.0 + np.exp(x))
    elif weight_func == 'exponential':
        alpha = params[0]
        if alpha <= 0:
            w = np.ones_like(eps_virt)
        else:
            w = np.exp(-alpha * (eps_virt - eps_lumo))
    elif weight_func == 'gaussian':
        w = weights_gaussian(eps_virt, eps_lumo, params[0])
    elif weight_func == 'lorentzian':
        w = weights_lorentzian(eps_virt, eps_lumo, params[0])
    elif weight_func == 'top_n':
        w = weights_top_n_virtual(len(eps_virt), params[0])
    else:
        raise ValueError(f"Unknown weight function: {weight_func}")

    w_sum = np.sum(w)
    if w_sum < 1e-30:
        return eps_lumo
    return np.sum(w * eps_virt) / w_sum


# ======================================================================
# OPTIMISATION
# ======================================================================

def evaluate_model(descriptors, targets):
    """Régression linéaire et métriques."""
    x = np.array(descriptors)
    y = np.array(targets)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {'r2': 0, 'mae': 999, 'rmse': 999, 'n': 0}

    slope, intercept, r, p, se = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    n = len(x)
    r2 = r ** 2
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2

    return {
        'r2': r2, 'r2_adj': r2_adj, 'mae': mae, 'rmse': rmse,
        'slope': slope, 'intercept': intercept, 'p': p, 'n': n,
        'y_pred': y_pred, 'y': y, 'x': x,
    }


def optimize_distribution(all_orbital_data, targets, weight_func, param_bounds,
                           mode='ip'):
    """
    Optimise le paramètre de la distribution pour minimiser le RMSE.

    Args:
        all_orbital_data: liste de dicts orbitaux (1 par conformère)
        targets: array des IP ou |EA| en kJ/mol
        weight_func: nom de la distribution
        param_bounds: (min, max) pour le paramètre
        mode: 'ip' ou 'ea'
    """
    compute_func = (compute_weighted_descriptor_ip if mode == 'ip'
                    else compute_weighted_descriptor_ea)

    def objective(param):
        descriptors = []
        for orb_data in all_orbital_data:
            d = compute_func(orb_data, weight_func, [param])
            descriptors.append(d)
        result = evaluate_model(descriptors, targets)
        return result['rmse']

    # Grid search first, then refine
    grid = np.linspace(param_bounds[0], param_bounds[1], 200)
    rmse_grid = [objective(p) for p in grid]
    best_idx = np.argmin(rmse_grid)
    best_param_coarse = grid[best_idx]

    # Refine with scipy
    result = minimize_scalar(objective,
                             bounds=(max(param_bounds[0], best_param_coarse - (grid[1] - grid[0]) * 5),
                                     min(param_bounds[1], best_param_coarse + (grid[1] - grid[0]) * 5)),
                             method='bounded')

    best_param = result.x
    best_rmse = result.fun

    # Compute final descriptors
    descriptors = [compute_func(orb_data, weight_func, [best_param])
                   for orb_data in all_orbital_data]
    final = evaluate_model(descriptors, targets)
    final['best_param'] = best_param
    final['param_grid'] = grid
    final['rmse_grid'] = np.array(rmse_grid)

    return final


def optimize_top_n(all_orbital_data, targets, mode='ip', max_n=30):
    """Optimise le nombre d'orbitales pour le modèle Top-N."""
    compute_func = (compute_weighted_descriptor_ip if mode == 'ip'
                    else compute_weighted_descriptor_ea)

    best_rmse = 999
    best_n = 1
    results_by_n = {}

    for n in range(1, max_n + 1):
        descriptors = [compute_func(orb_data, 'top_n', [n])
                       for orb_data in all_orbital_data]
        res = evaluate_model(descriptors, targets)
        results_by_n[n] = res
        if res['rmse'] < best_rmse:
            best_rmse = res['rmse']
            best_n = n

    final = results_by_n[best_n]
    final['best_param'] = best_n
    final['results_by_n'] = results_by_n
    return final


# ======================================================================
# MODÈLE 2-PARAMÈTRES : a·desc1 + b·desc2 + c
# ======================================================================

def optimize_2param_distribution(all_orbital_data, targets, weight_func,
                                  param_bounds, mode='ip'):
    """
    Distribution à 2 paramètres : centre (μ) + largeur (σ/kT/γ).
    Le centre peut s'écarter de HOMO/LUMO.
    """
    n_occ_list = [orb['n_occ'] for orb in all_orbital_data]

    def compute_desc(mu_shift, width):
        descriptors = []
        for orb_data in all_orbital_data:
            n_occ = orb_data['n_occ']
            if mode == 'ip':
                eps = orb_data['eps'][:n_occ]
                ref = eps[-1] + mu_shift  # HOMO + shift
            else:
                eps = orb_data['eps'][n_occ:]
                ref = eps[0] + mu_shift  # LUMO + shift

            if weight_func == 'gaussian_2p':
                w = weights_gaussian(eps, ref, width)
            elif weight_func == 'lorentzian_2p':
                w = weights_lorentzian(eps, ref, width)
            else:
                w = weights_gaussian(eps, ref, width)

            w_sum = np.sum(w)
            if w_sum < 1e-30:
                descriptors.append(eps[-1] if mode == 'ip' else eps[0])
            else:
                descriptors.append(np.sum(w * eps) / w_sum)
        return descriptors

    def objective(params):
        mu_shift, width = params
        if width <= 0.01:
            return 999
        descriptors = compute_desc(mu_shift, width)
        res = evaluate_model(descriptors, targets)
        return res['rmse']

    # Grid search
    best_rmse = 999
    best_params = (0, 1)
    for mu_s in np.linspace(param_bounds[0][0], param_bounds[0][1], 30):
        for w in np.linspace(param_bounds[1][0], param_bounds[1][1], 30):
            rmse = objective([mu_s, w])
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (mu_s, w)

    # Refine
    result = minimize(objective, best_params,
                      bounds=[param_bounds[0], param_bounds[1]],
                      method='L-BFGS-B')
    if result.fun < best_rmse:
        best_params = tuple(result.x)

    descriptors = compute_desc(*best_params)
    final = evaluate_model(descriptors, targets)
    final['best_param'] = best_params
    final['param_names'] = ('μ_shift', 'width')
    return final


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("PONDÉRATION ORBITALE OPTIMISÉE — IP & EA")
    print("=" * 70)

    # 1. Load redox data
    workdirs = sorted([d for d in BATCH_DIR.glob("*_workdir") if d.is_dir()])
    df = collect_all_conformers(workdirs)
    print(f"\n{len(df)} conformères, {df['molecule_name'].nunique()} molécules")

    # 2. Extract all orbital data
    print("\nExtraction de toutes les orbitales depuis neutral_tz2p...")
    orb_rows = []
    for workdir in workdirs:
        mol_name = workdir.name.replace('_workdir', '')
        orbitals = extract_all_orbitals(workdir, mol_name)
        for conf_num, orb_data in orbitals.items():
            orb_rows.append({
                'molecule_name': mol_name,
                'conformer': conf_num,
                'orbital_data': orb_data,
            })

    df_orb = pd.DataFrame(orb_rows)
    print(f"  {len(df_orb)} conformères avec orbitales extraites")

    # 3. Merge
    df = df.merge(df_orb, on=['molecule_name', 'conformer'], how='inner')
    print(f"  {len(df)} conformères après fusion")

    # Prepare targets
    df_ip = df.dropna(subset=['IP_kJ']).copy()
    df_ea = df.dropna(subset=['EA_kJ']).copy()
    df_ea['EA_kJ_abs'] = df_ea['EA_kJ'].abs()

    ip_targets = df_ip['IP_kJ'].values
    ea_targets = df_ea['EA_kJ_abs'].values
    ip_orb_data = df_ip['orbital_data'].tolist()
    ea_orb_data = df_ea['orbital_data'].tolist()

    print(f"\n  IP: {len(ip_targets)} conformères")
    print(f"  EA: {len(ea_targets)} conformères")

    # ===================================================================
    # DISTRIBUTIONS À TESTER
    # ===================================================================
    distributions_1p = {
        'Fermi-Dirac': ('fermi_dirac', (0.05, 15.0)),
        'Exponentielle': ('exponential', (0.01, 10.0)),
        'Gaussienne': ('gaussian', (0.05, 20.0)),
        'Lorentzienne': ('lorentzian', (0.05, 20.0)),
    }

    distributions_2p = {
        'Gaussienne-2p': ('gaussian_2p', [(-5.0, 5.0), (0.05, 20.0)]),
        'Lorentzienne-2p': ('lorentzian_2p', [(-5.0, 5.0), (0.05, 20.0)]),
    }

    # ===================================================================
    # IP
    # ===================================================================
    print(f"\n{'='*70}")
    print("RÉSULTATS — IP (Potentiel d'Ionisation)")
    print(f"{'='*70}")

    ip_results = {}

    # Baseline : Koopmans (HOMO seul = Top-1)
    desc_homo = [orb['eps'][orb['n_occ'] - 1] for orb in ip_orb_data]
    res_koopmans_ip = evaluate_model(desc_homo, ip_targets)
    ip_results['Koopmans (HOMO)'] = res_koopmans_ip
    print(f"\n  {'Koopmans (HOMO)':<30} R²={res_koopmans_ip['r2']:.6f}  "
          f"MAE={res_koopmans_ip['mae']:.3f}  RMSE={res_koopmans_ip['rmse']:.3f} kJ/mol")

    # 1-parameter distributions
    for name, (func, bounds) in distributions_1p.items():
        res = optimize_distribution(ip_orb_data, ip_targets, func, bounds, mode='ip')
        ip_results[name] = res
        print(f"  {name:<30} R²={res['r2']:.6f}  "
              f"MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  "
              f"param={res['best_param']:.4f}")

    # Top-N
    res_topn = optimize_top_n(ip_orb_data, ip_targets, mode='ip', max_n=30)
    ip_results['Top-N'] = res_topn
    print(f"  {'Top-N':<30} R²={res_topn['r2']:.6f}  "
          f"MAE={res_topn['mae']:.3f}  RMSE={res_topn['rmse']:.3f}  "
          f"N={res_topn['best_param']}")

    # 2-parameter distributions
    for name, (func, bounds) in distributions_2p.items():
        res = optimize_2param_distribution(ip_orb_data, ip_targets, func, bounds, mode='ip')
        ip_results[name] = res
        print(f"  {name:<30} R²={res['r2']:.6f}  "
              f"MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  "
              f"params={res['best_param']}")

    # ===================================================================
    # EA
    # ===================================================================
    print(f"\n{'='*70}")
    print("RÉSULTATS — EA (Affinité Électronique)")
    print(f"{'='*70}")

    ea_results = {}

    # Baseline : Koopmans (LUMO seul)
    desc_lumo = [orb['eps'][orb['n_occ']] for orb in ea_orb_data]
    res_koopmans_ea = evaluate_model(desc_lumo, ea_targets)
    ea_results['Koopmans (LUMO)'] = res_koopmans_ea
    print(f"\n  {'Koopmans (LUMO)':<30} R²={res_koopmans_ea['r2']:.6f}  "
          f"MAE={res_koopmans_ea['mae']:.3f}  RMSE={res_koopmans_ea['rmse']:.3f} kJ/mol")

    for name, (func, bounds) in distributions_1p.items():
        res = optimize_distribution(ea_orb_data, ea_targets, func, bounds, mode='ea')
        ea_results[name] = res
        print(f"  {name:<30} R²={res['r2']:.6f}  "
              f"MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  "
              f"param={res['best_param']:.4f}")

    res_topn_ea = optimize_top_n(ea_orb_data, ea_targets, mode='ea', max_n=30)
    ea_results['Top-N'] = res_topn_ea
    print(f"  {'Top-N':<30} R²={res_topn_ea['r2']:.6f}  "
          f"MAE={res_topn_ea['mae']:.3f}  RMSE={res_topn_ea['rmse']:.3f}  "
          f"N={res_topn_ea['best_param']}")

    for name, (func, bounds) in distributions_2p.items():
        res = optimize_2param_distribution(ea_orb_data, ea_targets, func, bounds, mode='ea')
        ea_results[name] = res
        print(f"  {name:<30} R²={res['r2']:.6f}  "
              f"MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}  "
              f"params={res['best_param']}")

    # ===================================================================
    # TABLEAU RÉCAPITULATIF
    # ===================================================================
    print(f"\n\n{'='*90}")
    print("TABLEAU RÉCAPITULATIF")
    print(f"{'='*90}")

    print(f"\n  {'--- IP ---':^90}")
    print(f"  {'Modèle':<30} {'R²':>8} {'R² adj':>8} {'MAE':>10} {'RMSE':>10} {'ΔMAE':>8} {'ΔRMSE':>8}")
    print(f"  {'-'*82}")
    ref_mae_ip = res_koopmans_ip['mae']
    ref_rmse_ip = res_koopmans_ip['rmse']
    for name, res in sorted(ip_results.items(), key=lambda x: x[1]['rmse']):
        dmae = res['mae'] - ref_mae_ip
        drmse = res['rmse'] - ref_rmse_ip
        print(f"  {name:<30} {res['r2']:>8.4f} {res.get('r2_adj', res['r2']):>8.4f} "
              f"{res['mae']:>9.3f} {res['rmse']:>9.3f} {dmae:>+8.3f} {drmse:>+8.3f}")

    print(f"\n  {'--- EA ---':^90}")
    print(f"  {'Modèle':<30} {'R²':>8} {'R² adj':>8} {'MAE':>10} {'RMSE':>10} {'ΔMAE':>8} {'ΔRMSE':>8}")
    print(f"  {'-'*82}")
    ref_mae_ea = res_koopmans_ea['mae']
    ref_rmse_ea = res_koopmans_ea['rmse']
    for name, res in sorted(ea_results.items(), key=lambda x: x[1]['rmse']):
        dmae = res['mae'] - ref_mae_ea
        drmse = res['rmse'] - ref_rmse_ea
        print(f"  {name:<30} {res['r2']:>8.4f} {res.get('r2_adj', res['r2']):>8.4f} "
              f"{res['mae']:>9.3f} {res['rmse']:>9.3f} {dmae:>+8.3f} {drmse:>+8.3f}")

    # ===================================================================
    # FIGURES
    # ===================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1 : Parity plots des meilleurs modèles ---
    for target_name, results_dict, ref_name in [
        ('IP', ip_results, 'Koopmans (HOMO)'),
        ('EA', ea_results, 'Koopmans (LUMO)'),
    ]:
        # Find best model (lowest RMSE, excluding Koopmans)
        best_name = min(
            [k for k in results_dict if k != ref_name],
            key=lambda k: results_dict[k]['rmse']
        )
        best = results_dict[best_name]
        ref = results_dict[ref_name]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        for ax, res, title in [
            (axes[0], ref, f'{ref_name}'),
            (axes[1], best, f'{best_name}'),
        ]:
            y_true = res['y']
            y_pred = res['y_pred']

            ax.scatter(y_true, y_pred, s=50, alpha=0.7, c='steelblue',
                      edgecolors='white', linewidths=0.5, zorder=3)

            vmin = min(y_true.min(), y_pred.min()) - 5
            vmax = max(y_true.max(), y_pred.max()) + 5
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, alpha=0.5)

            unit = 'kJ/mol'
            ax.set_xlabel(f'{target_name} DFT ({unit})', fontsize=11)
            ax.set_ylabel(f'{target_name} prédit ({unit})', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')

            textstr = (f"$R^2 = {res['r2']:.4f}$\n"
                       f"MAE = {res['mae']:.2f} {unit}\n"
                       f"RMSE = {res['rmse']:.2f} {unit}")
            if 'best_param' in res:
                p = res['best_param']
                if isinstance(p, tuple):
                    textstr += f"\nparams = ({p[0]:.3f}, {p[1]:.3f})"
                elif isinstance(p, (int, np.integer)):
                    textstr += f"\nN = {p}"
                else:
                    textstr += f"\nparam = {p:.4f}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        outfile = OUTPUT_DIR / f"{target_name}_orbital_weighting_parity.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  Figure: {outfile}")

    # --- Figure 2 : RMSE vs paramètre pour chaque distribution ---
    for target_name, results_dict, ref_name in [
        ('IP', ip_results, 'Koopmans (HOMO)'),
        ('EA', ea_results, 'Koopmans (LUMO)'),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        dist_1p_names = ['Fermi-Dirac', 'Exponentielle', 'Gaussienne', 'Lorentzienne']
        ref_rmse = results_dict[ref_name]['rmse']

        for ax, dname in zip(axes.flat, dist_1p_names):
            res = results_dict[dname]
            if 'param_grid' in res:
                ax.plot(res['param_grid'], res['rmse_grid'], 'b-', linewidth=1.5)
                ax.axhline(ref_rmse, color='r', linestyle='--', alpha=0.7,
                          label=f'{ref_name}: {ref_rmse:.3f}')
                ax.axvline(res['best_param'], color='g', linestyle=':', alpha=0.7,
                          label=f'Optimal: {res["best_param"]:.3f}')
                ax.scatter([res['best_param']], [res['rmse']], c='green',
                          s=80, zorder=5, marker='*')

                ax.set_xlabel('Paramètre', fontsize=10)
                ax.set_ylabel('RMSE (kJ/mol)', fontsize=10)
                ax.set_title(f'{dname}  (RMSE={res["rmse"]:.3f})', fontsize=11)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'{target_name} — RMSE vs paramètre de distribution',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        outfile = OUTPUT_DIR / f"{target_name}_orbital_weighting_rmse_vs_param.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Figure: {outfile}")

    # --- Figure 3 : Top-N scan ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, target_name, results_dict, ref_name in [
        (axes[0], 'IP', ip_results, 'Koopmans (HOMO)'),
        (axes[1], 'EA', ea_results, 'Koopmans (LUMO)'),
    ]:
        res_topn = results_dict['Top-N']
        if 'results_by_n' in res_topn:
            ns = sorted(res_topn['results_by_n'].keys())
            rmses = [res_topn['results_by_n'][n]['rmse'] for n in ns]
            maes = [res_topn['results_by_n'][n]['mae'] for n in ns]
            r2s = [res_topn['results_by_n'][n]['r2'] for n in ns]

            ax2 = ax.twinx()
            l1, = ax.plot(ns, rmses, 'b-o', markersize=4, label='RMSE')
            l2, = ax.plot(ns, maes, 'r-s', markersize=4, label='MAE')
            l3, = ax2.plot(ns, r2s, 'g-^', markersize=4, label='R²', alpha=0.7)

            best_n = res_topn['best_param']
            ax.axvline(best_n, color='orange', linestyle='--', alpha=0.7,
                      label=f'Best N={best_n}')

            ax.set_xlabel('N (nombre d\'orbitales)', fontsize=10)
            ax.set_ylabel('Erreur (kJ/mol)', fontsize=10, color='b')
            ax2.set_ylabel('R²', fontsize=10, color='g')
            ax.set_title(f'{target_name} — Top-N scan', fontsize=12, fontweight='bold')

            lines = [l1, l2, l3]
            labels = ['RMSE', 'MAE', 'R²']
            ax.legend(lines, labels, fontsize=8, loc='center right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = OUTPUT_DIR / "orbital_weighting_topN_scan.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {outfile}")

    # --- Figure 4 : Bar chart comparatif ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, target_name, results_dict, ref_name in [
        (axes[0], 'IP', ip_results, 'Koopmans (HOMO)'),
        (axes[1], 'EA', ea_results, 'Koopmans (LUMO)'),
    ]:
        sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['rmse'])
        names = [n for n, _ in sorted_models]
        rmses = [r['rmse'] for _, r in sorted_models]
        maes = [r['mae'] for _, r in sorted_models]

        x_pos = np.arange(len(names))
        width = 0.35
        ax.barh(x_pos - width/2, rmses, width, label='RMSE', color='steelblue', alpha=0.8)
        ax.barh(x_pos + width/2, maes, width, label='MAE', color='coral', alpha=0.8)

        ref_rmse = results_dict[ref_name]['rmse']
        ax.axvline(ref_rmse, color='red', linestyle='--', alpha=0.5,
                  label=f'Koopmans RMSE')

        ax.set_yticks(x_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Erreur (kJ/mol)', fontsize=10)
        ax.set_title(f'{target_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    plt.suptitle('Comparaison des modèles de pondération orbitale',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    outfile = OUTPUT_DIR / "orbital_weighting_comparison_bar.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {outfile}")

    print("\nDone.")


if __name__ == '__main__':
    main()
