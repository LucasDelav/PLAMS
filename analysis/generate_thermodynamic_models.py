#!/usr/bin/env amspython
"""
Modèles prédictifs pour ΔΔU, TΔS et ΔG° combiné.

Analyse :
  1. ZPE_neutre vs ΔΔU (ox/red) — corrélation linéaire ou modèle constant
  2. S_vib_neutre vs TΔS (ox/red) — corrélation linéaire ou modèle constant
  3. Combinaison de tous les modèles → ΔG° DFT vs ΔG° prédit

Génère :
  - ZPE_vs_ddU.png          : 2 panels (red/ox)
  - Svib_vs_TdS.png         : 2 panels (red/ox)
  - dG_parity.png           : 2 panels (red/ox), parity plot
  - model_predictions.csv   : toutes les prédictions par molécule
  - model_summary.txt       : statistiques des modèles

Usage:
  amspython generate_thermodynamic_models.py \\
      --batch-dir 1594174/dir_workdir \\
      --summary-csv rapport_full_decomposition/molecules_summary_full_decomposition.csv \\
      --lvc-file lvc_report_lvc2/all_methods_comparison.txt \\
      --output-dir thermodynamic_models
"""

import sys
import csv
import math
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize_scalar

# ============================================================================
# Constants
# ============================================================================
R_JK = 8.314462618          # J/(mol·K)
R_KJK = R_JK / 1000.0      # kJ/(mol·K)
T = 298.15                  # K
h = 6.62607015e-34          # J·s
c_cm = 2.99792458e10        # cm/s
k_B = 1.380649e-23          # J/K
# u_k = h * c * omega_k / (k_B * T) ; omega_k in cm^-1
U_FACTOR = h * c_cm / (k_B * T)  # ≈ 4.82625e-3 per cm^-1

EH_TO_KJ_LOCAL = 2625.499638  # Hartree → kJ/mol


# ============================================================================
# Data I/O
# ============================================================================

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


def parse_lvc_comparison(filepath):
    """
    Parse all_methods_comparison.txt pour extraire LVC, LVC2, Réel et RMSD.
    Retourne dict {mol_name: {reel_red, lvc_red, lvc2_red, rmsd_red,
                               reel_ox, lvc_ox, lvc2_ox, rmsd_ox}}
    """
    data = {}
    section = None  # 'red' or 'ox'

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if 'DUCTION' in stripped and 'M + e' in stripped:
                section = 'red'
                continue
            elif 'OXYDATION' in stripped and 'M →' in stripped:
                section = 'ox'
                continue
            elif stripped.startswith('STATISTIQUES'):
                break

            if section is None:
                continue
            if stripped.startswith('---') or stripped.startswith('==='):
                continue
            if stripped.startswith('Molécule') or stripped.startswith('(kJ/mol)'):
                continue
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 14:
                continue

            mol = parts[0]
            try:
                reel = float(parts[1])
                lvc = float(parts[2])
                lvc2 = float(parts[3])
                rmsd = float(parts[-1])
            except ValueError:
                continue

            if math.isnan(reel) or math.isnan(lvc):
                continue

            if mol not in data:
                data[mol] = {}

            data[mol][f'reel_{section}'] = reel
            data[mol][f'lvc_{section}'] = lvc
            data[mol][f'lvc2_{section}'] = lvc2
            data[mol][f'rmsd_{section}'] = rmsd

    return data


def read_conformer_weights(workdir):
    """Lit les poids de Boltzmann depuis conformers_summary.txt."""
    summary = Path(workdir) / "conformers" / "results" / "conformers_summary.txt"
    if not summary.exists():
        return None
    weights = {}
    in_table = False
    with open(summary, 'r') as f:
        for line in f:
            stripped = line.strip()
            if 'conf_' in stripped and 'weight' in stripped.lower():
                in_table = True
                continue
            if in_table:
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        conf_num = int(parts[0])
                        weight = float(parts[2])
                        weights[conf_num] = weight
                    except (ValueError, IndexError):
                        if stripped.startswith('---') or stripped.startswith('==='):
                            continue
                        in_table = False
    return weights if weights else None


# ============================================================================
# S_vib computation from RKF
# ============================================================================

def compute_svib_from_frequencies(frequencies):
    """
    Calcule l'entropie vibrationnelle RRHO (J/(mol·K)) à partir des fréquences (cm⁻¹).

    S_vib/R = Σ_k [ u_k/(e^u_k - 1) - ln(1 - e^{-u_k}) ]
    où u_k = ℏω_k / (k_BT)
    """
    freqs = np.asarray(frequencies, dtype=float)
    freqs = freqs[freqs > 10.0]  # Ignorer les modes quasi-nuls (< 10 cm⁻¹)

    s_total = 0.0
    for omega in freqs:
        u = U_FACTOR * omega
        if u > 500:
            continue  # Contribution négligeable
        eu = math.exp(u)
        s_k = u / (eu - 1.0) - math.log(1.0 - math.exp(-u))
        s_total += s_k

    return R_JK * s_total  # J/(mol·K)


def extract_svib_for_molecules(batch_dir, molecule_names):
    """
    Pour chaque molécule, lit les fréquences des RKF neutres et calcule S_vib
    (moyenne Boltzmann sur conformères).

    Retourne dict {mol_name: TS_vib_kJ} (= T × S_vib en kJ/mol)
    """
    from scm.plams import KFReader

    # Import local pour find_neutral_rkf_files
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qmanalysis.frequency_extraction import find_neutral_rkf_files

    batch_path = Path(batch_dir)
    results = {}

    for mol_name in molecule_names:
        workdir = batch_path / f"{mol_name}_workdir"
        if not workdir.exists():
            print(f"  [SKIP] {mol_name}: workdir not found")
            continue

        rkf_files = find_neutral_rkf_files(workdir, mol_name)
        if not rkf_files:
            print(f"  [SKIP] {mol_name}: no neutral RKF files")
            continue

        # Boltzmann weights
        weights = read_conformer_weights(workdir)

        svib_values = {}
        for conf_num, rkf_path in sorted(rkf_files.items()):
            try:
                kf = KFReader(str(rkf_path))
                freqs = np.array(kf.read("Vibrations", "Frequencies[cm-1]"))
                svib = compute_svib_from_frequencies(freqs)
                svib_values[conf_num] = svib
            except Exception as e:
                print(f"  [WARN] {mol_name} conf_{conf_num}: {e}")
                continue

        if not svib_values:
            continue

        # Boltzmann averaging
        if weights and all(c in weights for c in svib_values):
            total_w = sum(weights[c] for c in svib_values)
            avg_svib = sum(weights[c] * svib_values[c] for c in svib_values) / total_w
        else:
            avg_svib = np.mean(list(svib_values.values()))

        results[mol_name] = T * avg_svib / 1000.0  # → kJ/mol (TS_vib)

    return results


# ============================================================================
# Morse-LVC: per-mode anharmonic correction
# ============================================================================

def compute_huang_rhys_per_mode(frequencies, force_constants, F_projections):
    """
    Calcule le facteur de Huang-Rhys S_k et l'énergie harmonique par mode.

    S_k = F_k² / (2 * ℏω_k * k_k)  [sans dimension]
    E_harm_k = S_k * ℏω_k           [Hartree]

    Mais dans les unités RKF (Hartree/Bohr² pour k, Hartree/Bohr pour F):
    E_harm_k = F_k² / (2 * k_k)     [Hartree]
    et ℏω_k = h*c*ν_k / N_A en kJ/mol = ν_k * 1.196266e-2 kJ/mol (par mode)

    En fait, S_k = E_harm_k / (ℏω_k) où ℏω_k en Hartree:
    ℏω_k [Ha] = k_k * ... Non, utilisons la relation directe:
    k_k = (2π * ν_k * c)² * μ_k  mais dans les NoWeightNormalMode,
    F_k²/(2*k_k) = ½ω²ΔQ² = S_k * ℏω_k

    Conversion: ℏω_k [Hartree] = ν_k [cm⁻¹] * 4.556335e-6
    (1 cm⁻¹ = 4.556335e-6 Hartree)

    Returns:
        List of dicts: [{mode, freq, F_k, k_k, E_harm_Ha, E_harm_kJ, S_k, hw_Ha}]
    """
    CM_TO_HARTREE = 4.556335e-6  # 1 cm⁻¹ in Hartree
    results = []
    for i in range(len(frequencies)):
        freq = frequencies[i]
        k_k = force_constants[i]
        F_k = F_projections[i]

        if freq < 50.0 or k_k <= 0:
            continue

        E_harm_Ha = F_k**2 / (2.0 * k_k)
        hw_Ha = freq * CM_TO_HARTREE  # ℏω in Hartree
        S_k = E_harm_Ha / hw_Ha if hw_Ha > 0 else 0.0

        results.append({
            'mode': i + 1,
            'freq': freq,
            'F_k': F_k,
            'k_k': k_k,
            'E_harm_Ha': E_harm_Ha,
            'E_harm_kJ': E_harm_Ha * EH_TO_KJ_LOCAL,
            'S_k': S_k,
            'hw_Ha': hw_Ha,
        })
    return results


def morse_edef_per_mode(mode_data, D_prime):
    """
    Calcule l'énergie de déformation Morse pour chaque mode.

    Pour un potentiel de Morse: V(Q) = D(1 - exp(-a·Q))²
    En unités de quanta vibrationnels: D' = D / (ℏω)

    E_Morse_k = D'·ℏω_k · (1 - exp(-√(S_k / D')))²

    Pour S_k << D': E_Morse ≈ S_k·ℏω_k (limite harmonique)
    Pour S_k >> D': E_Morse → D'·ℏω_k (saturation)

    Args:
        mode_data: liste de dicts depuis compute_huang_rhys_per_mode
        D_prime: profondeur du puits en quanta vibrationnels (paramètre à calibrer)

    Returns:
        float: E_def_Morse total en kJ/mol (négatif)
    """
    E_total_Ha = 0.0
    for m in mode_data:
        S_k = m['S_k']
        hw = m['hw_Ha']

        if D_prime <= 0 or S_k <= 0:
            E_total_Ha += m['E_harm_Ha']  # fallback harmonique
            continue

        x = math.sqrt(S_k / D_prime)
        E_morse_Ha = D_prime * hw * (1.0 - math.exp(-x))**2
        E_total_Ha += E_morse_Ha

    return -E_total_Ha * EH_TO_KJ_LOCAL  # négatif (relaxation)


def extract_morse_lvc_for_molecules(batch_dir, molecule_names):
    """
    Pour chaque molécule, calcule E_def par mode (LVC et LVC2) et retourne les
    données per-mode nécessaires au calcul Morse.

    Calcule les per-mode data avec:
    - LVC: modes neutres + gradients ioniques
    - LVC2: modes ioniques (du SP) + gradients ioniques

    Retourne dict {mol_name: {'red_modes': [mode_data], 'red_modes_lvc2': [mode_data],
                               'ox_modes': ..., 'ox_modes_lvc2': ...,
                               'E_lvc_red': float, 'E_lvc2_red': float, ...}}
    """
    from scm.plams import KFReader
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qmanalysis.frequency_extraction import find_neutral_rkf_files
    from qmanalysis.lvc_analysis import find_sp_rkf_files

    batch_path = Path(batch_dir)
    results = {}

    for mol_name in molecule_names:
        workdir = batch_path / f"{mol_name}_workdir"
        if not workdir.exists():
            continue

        neutral_rkfs = find_neutral_rkf_files(workdir, mol_name)
        red_sp_rkfs = find_sp_rkf_files(workdir, mol_name, 'reduced')
        ox_sp_rkfs = find_sp_rkf_files(workdir, mol_name, 'oxidized')

        if not neutral_rkfs:
            continue

        # Boltzmann weights
        weights = read_conformer_weights(workdir)

        # Per-conformer mode data
        conf_data = {}  # conf_num -> {red_modes, red_modes_lvc2, ox_modes, ...}

        for conf_num in sorted(neutral_rkfs.keys()):
            n_rkf = neutral_rkfs.get(conf_num)
            r_rkf = red_sp_rkfs.get(conf_num)
            o_rkf = ox_sp_rkfs.get(conf_num)

            if not n_rkf:
                continue

            cd = {}

            try:
                kf_n = KFReader(str(n_rkf))
                freqs_n = np.array(kf_n.read("Vibrations", "Frequencies[cm-1]"), dtype=float)
                fc_n = np.array(kf_n.read("Vibrations", "ForceConstants"), dtype=float)
                n_modes = len(freqs_n)

                modes_n = []
                for j in range(1, n_modes + 1):
                    mode = np.array(kf_n.read("Vibrations", f"NoWeightNormalMode({j})"), dtype=float)
                    modes_n.append(mode)
            except Exception:
                continue

            for state, sp_rkf in [('red', r_rkf), ('ox', o_rkf)]:
                if not sp_rkf:
                    continue
                try:
                    kf_sp = KFReader(str(sp_rkf))
                    grad = np.array(kf_sp.read("AMSResults", "Gradients"), dtype=float)

                    # LVC: neutral modes + ionic gradient
                    F_proj = [np.dot(grad, modes_n[j]) for j in range(n_modes)]
                    cd[f'{state}_modes'] = compute_huang_rhys_per_mode(freqs_n, fc_n, F_proj)

                    # LVC2: ionic modes + ionic gradient (from SP RKF)
                    try:
                        freqs_ion = np.array(kf_sp.read("Vibrations", "Frequencies[cm-1]"), dtype=float)
                        fc_ion = np.array(kf_sp.read("Vibrations", "ForceConstants"), dtype=float)
                        n_modes_ion = len(freqs_ion)
                        modes_ion = []
                        for j in range(1, n_modes_ion + 1):
                            m_ion = np.array(kf_sp.read("Vibrations", f"NoWeightNormalMode({j})"), dtype=float)
                            modes_ion.append(m_ion)

                        F_proj_ion = [np.dot(grad, modes_ion[j]) for j in range(n_modes_ion)]
                        cd[f'{state}_modes_lvc2'] = compute_huang_rhys_per_mode(
                            freqs_ion, fc_ion, F_proj_ion)
                    except Exception:
                        pass  # No ionic Hessian available

                except Exception:
                    pass

            if cd:
                conf_data[conf_num] = cd

        if not conf_data:
            continue

        # Pick dominant conformer for mode data (used by Morse correction)
        if weights:
            best_conf = max(conf_data.keys(), key=lambda c: weights.get(c, 0.0))
        else:
            best_conf = min(conf_data.keys())

        cd = conf_data[best_conf]
        mol_result = {
            'best_conf': best_conf,
        }
        for key in ['red_modes', 'red_modes_lvc2', 'ox_modes', 'ox_modes_lvc2']:
            mol_result[key] = cd.get(key, [])

        # Compute S_sum from dominant conformer (consistent with LVC modes)
        for proc in ['red', 'ox']:
            for suffix, label in [('', 'lvc'), ('_lvc2', 'lvc2')]:
                modes_key = f'{proc}_modes{suffix}'
                if mol_result.get(modes_key):
                    mol_result[f'S_sum_{proc}{suffix}'] = sum(
                        m['S_k'] for m in mol_result[modes_key])

                # Harmonic Edef from best conformer (for Morse calibration)
                if mol_result.get(modes_key):
                    modes = mol_result[modes_key]
                    mol_result[f'E_{label}_{proc}'] = -sum(m['E_harm_kJ'] for m in modes)

        results[mol_name] = mol_result

    return results


def calibrate_morse_Dprime(morse_data, lvc_data, real_edef, exclude=None, process='red'):
    """
    Calibre D' en minimisant la MAE de E_def_Morse vs E_def_réel pour la réduction.

    Args:
        morse_data: dict {mol: {red_modes, ox_modes, ...}}
        lvc_data: dict from parse_lvc_comparison
        real_edef: dict {mol: Edef_real_kJ} (négatif)
        exclude: set of molecules to exclude
        process: 'red' or 'ox'

    Returns:
        (D_prime_opt, results_dict)
    """
    key = f'{process}_modes'

    # Build calibration dataset
    cal_mols = []
    cal_real = []
    cal_modes = []

    for mol, md in morse_data.items():
        if exclude and mol in exclude:
            continue
        if mol not in real_edef or math.isnan(real_edef[mol]):
            continue
        if not md.get(key):
            continue

        cal_mols.append(mol)
        cal_real.append(real_edef[mol])
        cal_modes.append(md[key])

    if len(cal_mols) < 3:
        return None, {}

    def mae_for_D(D_prime):
        errs = []
        for i in range(len(cal_mols)):
            e_morse = morse_edef_per_mode(cal_modes[i], D_prime)
            errs.append(abs(e_morse - cal_real[i]))
        return np.mean(errs)

    # Search D' in range [5, 200] quanta
    result = minimize_scalar(mae_for_D, bounds=(3, 200), method='bounded')
    D_opt = result.x

    # Compute per-molecule results with optimal D'
    per_mol = {}
    for i, mol in enumerate(cal_mols):
        e_morse = morse_edef_per_mode(cal_modes[i], D_opt)
        e_lvc = -sum(m['E_harm_kJ'] for m in cal_modes[i])
        per_mol[mol] = {
            'E_morse': e_morse,
            'E_lvc': e_lvc,
            'E_real': cal_real[i],
            'err_morse': e_morse - cal_real[i],
            'err_lvc': e_lvc - cal_real[i],
        }

    return D_opt, per_mol


# ============================================================================
# Analysis functions
# ============================================================================

def analyze_correlation(x, y, names, x_label, y_label):
    """
    Teste la corrélation linéaire et compare au modèle constant.
    Retourne un dict avec tous les résultats.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Linear regression
    slope, intercept, r, p, se = linregress(x, y)
    r2 = r ** 2
    y_pred_lin = slope * x + intercept
    mae_lin = np.mean(np.abs(y - y_pred_lin))
    rmse_lin = np.sqrt(np.mean((y - y_pred_lin) ** 2))

    # Constant model (mean)
    c = np.mean(y)
    mae_const = np.mean(np.abs(y - c))
    rmse_const = np.sqrt(np.mean((y - c) ** 2))

    # Choose model
    use_linear = mae_lin < mae_const * 0.8 and p < 0.1

    return {
        'x': x, 'y': y, 'names': names,
        'x_label': x_label, 'y_label': y_label,
        'slope': slope, 'intercept': intercept, 'r': r, 'r2': r2, 'p': p, 'se': se,
        'mae_lin': mae_lin, 'rmse_lin': rmse_lin,
        'constant': c, 'mae_const': mae_const, 'rmse_const': rmse_const,
        'use_linear': use_linear,
        'y_pred': y_pred_lin if use_linear else np.full_like(y, c),
        'model_name': 'linear' if use_linear else 'constant',
    }


def predict_component(result, x_new=None):
    """Prédit y à partir du modèle choisi."""
    if result['use_linear']:
        if x_new is not None:
            return result['slope'] * np.asarray(x_new) + result['intercept']
        return result['y_pred']
    return np.full(len(x_new) if x_new is not None else len(result['y']), result['constant'])


# ============================================================================
# Plotting
# ============================================================================

def plot_correlation_2panels(result_red, result_ox, title_red, title_ox, filepath):
    """2-panel scatter plot (reduction | oxidation) avec regression et modèle constant."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title in [(ax1, result_red, title_red), (ax2, result_ox, title_ox)]:
        x, y = res['x'], res['y']

        ax.scatter(x, y, c='steelblue', s=60, zorder=5, edgecolors='navy', linewidths=0.5)

        # Labels
        for i, name in enumerate(res['names']):
            ax.annotate(name, (x[i], y[i]), fontsize=6, alpha=0.7,
                        xytext=(4, 4), textcoords='offset points')

        # Regression line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = res['slope'] * x_fit + res['intercept']
        ax.plot(x_fit, y_fit, 'r-', alpha=0.7, linewidth=1.5,
                label=f'Lin: $R^2={res["r2"]:.3f}$, MAE={res["mae_lin"]:.2f}')

        # Constant model
        ax.axhline(res['constant'], color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'Const: $\\bar{{y}}={res["constant"]:.2f}$, MAE={res["mae_const"]:.2f}')

        # Chosen model highlight
        chosen = 'Linear' if res['use_linear'] else 'Constant'
        ax.set_title(f'{title}\nModèle retenu : {chosen} ($p = {res["p"]:.3f}$)', fontsize=11)
        ax.set_xlabel(res['x_label'], fontsize=10)
        ax.set_ylabel(res['y_label'], fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_dG_parity(data, filepath, exclude=None):
    """Parity plot ΔG° DFT vs ΔG° prédit, 2 panels (red | ox)."""
    if exclude:
        data = [d for d in data if d['molecule'] not in exclude]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    panel_configs = [
        (ax1, 'ox', '(a) Oxydation', '#2980b9'),
        (ax2, 'red', '(b) Réduction', '#c0392b'),
    ]

    for ax, process, label, color in panel_configs:
        dg_dft = np.array([d[f'dG_{process}_DFT'] for d in data])
        dg_pred = np.array([d[f'dG_{process}_pred'] for d in data])
        names = [d['molecule'] for d in data]
        mask = ~(np.isnan(dg_dft) | np.isnan(dg_pred))

        dg_dft_v = dg_dft[mask]
        dg_pred_v = dg_pred[mask]
        names_v = [n for n, m in zip(names, mask) if m]

        if len(dg_dft_v) < 3:
            ax.text(0.5, 0.5, f'{label}\nInsuffisant (n={len(dg_dft_v)})',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            continue

        slope, intercept, r, p, se = linregress(dg_dft_v, dg_pred_v)
        r2 = r ** 2
        mae = np.mean(np.abs(dg_dft_v - dg_pred_v))
        rmse = np.sqrt(np.mean((dg_dft_v - dg_pred_v) ** 2))

        # y=x line
        lims = [min(dg_dft_v.min(), dg_pred_v.min()),
                max(dg_dft_v.max(), dg_pred_v.max())]
        margin = (lims[1] - lims[0]) * 0.1
        ax.plot([lims[0] - margin, lims[1] + margin],
                [lims[0] - margin, lims[1] + margin],
                'k--', alpha=0.5, linewidth=1, label='y = x')

        # Error band ±10 kJ/mol
        ax.fill_between([lims[0] - margin, lims[1] + margin],
                        [lims[0] - margin - 10, lims[1] + margin - 10],
                        [lims[0] - margin + 10, lims[1] + margin + 10],
                        alpha=0.08, color='grey', label='$\\pm$10 kJ/mol')

        # Scatter with molecule colors
        ax.scatter(dg_dft_v, dg_pred_v, s=80, color=color, marker='o',
                   edgecolors='white', linewidths=0.5, zorder=5)

        for i, name in enumerate(names_v):
            ax.annotate(name, (dg_dft_v[i], dg_pred_v[i]), fontsize=6, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points', color=color)

        # Regression line
        x_fit = np.linspace(lims[0] - margin, lims[1] + margin, 100)
        ax.plot(x_fit, slope * x_fit + intercept, '-', color=color,
                alpha=0.7, linewidth=1.5,
                label=f'$R^2$={r2:.3f}, MAE={mae:.1f} kJ/mol\ny={slope:.2f}x{intercept:+.1f}')

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel('$\\Delta G^\\circ_{DFT}$ (kJ/mol)', fontsize=12)
        ax.set_ylabel('$\\Delta G^\\circ_{pr\\acute{e}dit}$ (kJ/mol)', fontsize=12)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xlim(lims[0] - margin, lims[1] + margin)
        ax.set_ylim(lims[0] - margin, lims[1] + margin)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Modèles prédictifs ΔΔU, TΔS, ΔG°")
    parser.add_argument('--batch-dir', required=True, help='Répertoire batch des workdirs')
    parser.add_argument('--summary-csv', required=True,
                        help='CSV molecules_summary_full_decomposition.csv')
    parser.add_argument('--lvc-file', required=True,
                        help='all_methods_comparison.txt (LVC/LVC2)')
    parser.add_argument('--output-dir',
                        default=str(Path(__file__).resolve().parent.parent / 'results' / 'thermodynamic_models'),
                        help='Répertoire de sortie')
    parser.add_argument('--no-plots', action='store_true', help='Pas de graphiques')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Step 1: Load data
    # ========================================================================
    print("=" * 70)
    print("MODÈLES PRÉDICTIFS THERMODYNAMIQUES")
    print("=" * 70)

    print("\n[1] Chargement des données...")
    summary = read_csv(args.summary_csv)
    lvc_data = parse_lvc_comparison(args.lvc_file)

    # Filter to molecules with valid ddU and zpe data
    valid_mols = []
    for row in summary:
        mol = row['molecule_name']
        has_thermo = (not math.isnan(row.get('ddU_red_kJ', float('nan')))
                      and not math.isnan(row.get('zpe_kj', float('nan'))))
        has_lvc = mol in lvc_data
        if has_thermo and has_lvc:
            valid_mols.append(row)

    mol_names = [m['molecule_name'] for m in valid_mols]
    n = len(valid_mols)
    print(f"  {n} molécules avec données complètes: {', '.join(mol_names)}")

    # ========================================================================
    # Step 2: Extract S_vib from RKF
    # ========================================================================
    print("\n[2] Calcul de S_vib (entropie vibrationnelle RRHO)...")
    ts_vib = extract_svib_for_molecules(args.batch_dir, mol_names)
    print(f"  S_vib obtenu pour {len(ts_vib)}/{n} molécules")
    for mol, val in sorted(ts_vib.items()):
        svib_jk = val / T * 1000.0  # back to J/(mol·K)
        print(f"    {mol:12s}: S_vib = {svib_jk:.1f} J/(mol·K), TS_vib = {val:.1f} kJ/mol")

    # ========================================================================
    # Step 3: Analyze ΔΔU vs ZPE
    # ========================================================================
    print("\n[3] Analyse ΔΔU vs ZPE...")
    zpe = np.array([m['zpe_kj'] for m in valid_mols])
    ddU_red = np.array([m['ddU_red_kJ'] for m in valid_mols])
    ddU_ox = np.array([m['ddU_ox_kJ'] for m in valid_mols])

    res_ddU_red = analyze_correlation(zpe, ddU_red, mol_names,
                                      'ZPE neutre (kJ/mol)', '$\\Delta\\Delta U_{\\rm red}$ (kJ/mol)')
    res_ddU_ox = analyze_correlation(zpe, ddU_ox, mol_names,
                                     'ZPE neutre (kJ/mol)', '$\\Delta\\Delta U_{\\rm ox}$ (kJ/mol)')

    print(f"  Réduction: R²={res_ddU_red['r2']:.3f}, p={res_ddU_red['p']:.4f}, "
          f"MAE_lin={res_ddU_red['mae_lin']:.2f}, MAE_const={res_ddU_red['mae_const']:.2f} "
          f"→ {res_ddU_red['model_name']}")
    print(f"  Oxydation: R²={res_ddU_ox['r2']:.3f}, p={res_ddU_ox['p']:.4f}, "
          f"MAE_lin={res_ddU_ox['mae_lin']:.2f}, MAE_const={res_ddU_ox['mae_const']:.2f} "
          f"→ {res_ddU_ox['model_name']}")

    # ========================================================================
    # Step 4: Analyze TΔS vs S_vib
    # ========================================================================
    print("\n[4] Analyse TΔS vs TS_vib...")
    # Filter to molecules with S_vib
    mols_svib = [m for m in valid_mols if m['molecule_name'] in ts_vib]
    names_svib = [m['molecule_name'] for m in mols_svib]
    ts_vib_arr = np.array([ts_vib[m['molecule_name']] for m in mols_svib])
    TdS_red = np.array([m['TdS_red_kJ'] for m in mols_svib])
    TdS_ox = np.array([m['TdS_ox_kJ'] for m in mols_svib])

    res_TdS_red = analyze_correlation(ts_vib_arr, TdS_red, names_svib,
                                       '$TS_{\\rm vib}$ neutre (kJ/mol)',
                                       '$T\\Delta S_{\\rm red}$ (kJ/mol)')
    res_TdS_ox = analyze_correlation(ts_vib_arr, TdS_ox, names_svib,
                                      '$TS_{\\rm vib}$ neutre (kJ/mol)',
                                      '$T\\Delta S_{\\rm ox}$ (kJ/mol)')

    print(f"  Réduction: R²={res_TdS_red['r2']:.3f}, p={res_TdS_red['p']:.4f}, "
          f"MAE_lin={res_TdS_red['mae_lin']:.2f}, MAE_const={res_TdS_red['mae_const']:.2f} "
          f"→ {res_TdS_red['model_name']}")
    print(f"  Oxydation: R²={res_TdS_ox['r2']:.3f}, p={res_TdS_ox['p']:.4f}, "
          f"MAE_lin={res_TdS_ox['mae_lin']:.2f}, MAE_const={res_TdS_ox['mae_const']:.2f} "
          f"→ {res_TdS_ox['model_name']}")

    # ========================================================================
    # Step 5: Morse-LVC correction
    # ========================================================================
    print("\n[5] Correction Morse par mode (anharmonicité)...")
    morse_data = extract_morse_lvc_for_molecules(args.batch_dir, mol_names)
    print(f"  Données per-mode obtenues pour {len(morse_data)}/{n} molécules")

    # Show S_sum per molecule (LVC and LVC2)
    has_lvc2 = any(morse_data[m].get('red_modes_lvc2') for m in morse_data)
    print(f"    {'Molécule':12s} {'S_LVC_red':>9s} {'S_LVC2_red':>10s} {'nModes':>6s}")
    for mol in sorted(morse_data.keys()):
        md = morse_data[mol]
        s_red = md.get('S_sum_red', float('nan'))
        s_red2 = md.get('S_sum_red_lvc2', float('nan'))
        n_red = len(md.get('red_modes', []))
        n_red2 = len(md.get('red_modes_lvc2', []))
        print(f"    {mol:12s}: {s_red:9.2f}   {s_red2:10.2f}  {n_red:3d}/{n_red2:3d}")

    # Build real Edef dicts
    real_edef_red = {m['molecule_name']: m['Edef_red_kJ'] for m in valid_mols}
    real_edef_ox = {m['molecule_name']: m['Edef_ox_kJ'] for m in valid_mols}
    # Calibrate D' for reduction
    D_red, morse_results_red = calibrate_morse_Dprime(
        morse_data, lvc_data, real_edef_red,
        exclude=set(), process='red')
    if D_red is not None:
        mae_morse_red = np.mean([abs(v['err_morse']) for v in morse_results_red.values()])
        mae_lvc_red = np.mean([abs(v['err_lvc']) for v in morse_results_red.values()])
        print(f"\n  Réduction: D' optimal = {D_red:.1f} quanta")
        print(f"    MAE Morse = {mae_morse_red:.2f} kJ/mol")
        print(f"    MAE LVC   = {mae_lvc_red:.2f} kJ/mol")

        # Per-molecule detail
        print(f"\n    {'Molécule':12s} {'E_réel':>8s} {'E_LVC':>8s} {'E_Morse':>8s} "
              f"{'Δ_LVC':>8s} {'Δ_Morse':>8s} {'S_sum':>8s}")
        print("    " + "-" * 62)
        for mol in sorted(morse_results_red.keys()):
            v = morse_results_red[mol]
            s_sum = morse_data[mol].get('S_sum_red', float('nan'))
            print(f"    {mol:12s} {v['E_real']:8.2f} {v['E_lvc']:8.2f} {v['E_morse']:8.2f} "
                  f"{v['err_lvc']:8.2f} {v['err_morse']:8.2f} {s_sum:8.2f}")

    # Calibrate D' for oxidation
    D_ox, morse_results_ox = calibrate_morse_Dprime(
        morse_data, lvc_data, real_edef_ox,
        exclude=set(), process='ox')
    if D_ox is not None:
        mae_morse_ox = np.mean([abs(v['err_morse']) for v in morse_results_ox.values()])
        mae_lvc_ox = np.mean([abs(v['err_lvc']) for v in morse_results_ox.values()])
        print(f"\n  Oxydation: D' optimal = {D_ox:.1f} quanta")
        print(f"    MAE Morse = {mae_morse_ox:.2f} kJ/mol")
        print(f"    MAE LVC   = {mae_lvc_ox:.2f} kJ/mol")

    # Calibrate Morse-LVC2 for reduction (uses ionic Hessian modes)
    D_red2, morse2_results_red = calibrate_morse_Dprime(
        {m: {'red_modes': d.get('red_modes_lvc2', [])} for m, d in morse_data.items()},
        lvc_data, real_edef_red,
        exclude=set(), process='red')
    if D_red2 is not None:
        mae_morse2_red = np.mean([abs(v['err_morse']) for v in morse2_results_red.values()])
        mae_lvc2_red = np.mean([abs(v['err_lvc']) for v in morse2_results_red.values()])
        print(f"\n  Réduction Morse-LVC2: D' optimal = {D_red2:.1f} quanta")
        print(f"    MAE Morse-LVC2 = {mae_morse2_red:.2f} kJ/mol")
        print(f"    MAE LVC2       = {mae_lvc2_red:.2f} kJ/mol")

        print(f"\n    {'Molécule':12s} {'E_réel':>8s} {'E_LVC2':>8s} {'E_M-LVC2':>8s} "
              f"{'Δ_LVC2':>8s} {'Δ_M-LVC2':>8s} {'S_sum_L2':>8s}")
        print("    " + "-" * 62)
        for mol in sorted(morse2_results_red.keys()):
            v = morse2_results_red[mol]
            s2 = morse_data[mol].get('S_sum_red_lvc2', float('nan'))
            print(f"    {mol:12s} {v['E_real']:8.2f} {v['E_lvc']:8.2f} {v['E_morse']:8.2f} "
                  f"{v['err_lvc']:8.2f} {v['err_morse']:8.2f} {s2:8.2f}")

    # Scan D' values for LVC and LVC2
    print("\n  --- Scan D' pour la réduction ---")
    print(f"    {'D_prime':>8s} {'MAE_M-LVC':>10s} {'MAE_M-LVC2':>11s}")
    for D_test in [5, 10, 15, 20, 30, 50, 100, 200]:
        errs_lvc = []
        errs_lvc2 = []
        for mol, md in morse_data.items():
            if mol not in real_edef_red:
                continue
            if math.isnan(real_edef_red[mol]):
                continue
            if md.get('red_modes'):
                e_m = morse_edef_per_mode(md['red_modes'], D_test)
                errs_lvc.append(abs(e_m - real_edef_red[mol]))
            if md.get('red_modes_lvc2'):
                e_m2 = morse_edef_per_mode(md['red_modes_lvc2'], D_test)
                errs_lvc2.append(abs(e_m2 - real_edef_red[mol]))
        lvc_str = f"{np.mean(errs_lvc):10.2f}" if errs_lvc else "       N/A"
        lvc2_str = f"{np.mean(errs_lvc2):11.2f}" if errs_lvc2 else "        N/A"
        print(f"    {D_test:8d} {lvc_str} {lvc2_str}")

    # ========================================================================
    # Step 6: Combined ΔG° prediction
    # ========================================================================
    print("\n[6] Prédiction combinée de ΔG°...")

    # Koopmans: HOMO/LUMO in CSV are in Hartree (column says _eV but values ~-0.24 Ha)
    # Convert to eV then apply Koopmans: IP = -ε_HOMO, EA = -ε_LUMO
    EH_TO_EV = 27.211386245988
    EV_TO_KJ = 96.485

    homo_ha = np.array([m['HOMO_eV'] for m in valid_mols])  # actually Hartree
    lumo_ha = np.array([m['LUMO_eV'] for m in valid_mols])  # actually Hartree
    homo_eV = homo_ha * EH_TO_EV
    lumo_eV = lumo_ha * EH_TO_EV
    ip_dft = np.array([m['IP_kJ'] for m in valid_mols])
    ea_dft = np.array([m['EA_kJ'] for m in valid_mols])

    # Direct Koopmans: IP_pred = -ε_HOMO * EV_TO_KJ, EA_pred = -ε_LUMO * EV_TO_KJ
    ip_koopmans_direct = -homo_eV * EV_TO_KJ
    ea_koopmans_direct = -lumo_eV * EV_TO_KJ

    # Linear fit (all 15 molecules)
    sl_ip, int_ip, r_ip, _, _ = linregress(homo_eV, ip_dft)
    sl_ea, int_ea, r_ea, _, _ = linregress(lumo_eV, ea_dft)
    print(f"  Koopmans IP: slope={sl_ip:.2f}, intercept={int_ip:.2f}, "
          f"R²={r_ip**2:.4f} (n={len(homo_eV)})")
    print(f"  Koopmans EA: slope={sl_ea:.2f}, intercept={int_ea:.2f}, R²={r_ea**2:.4f}")
    print(f"  Direct IP MAE: "
          f"{np.mean(np.abs(ip_dft - ip_koopmans_direct)):.2f} kJ/mol")
    print(f"  Direct EA MAE: {np.mean(np.abs(ea_dft - ea_koopmans_direct)):.2f} kJ/mol")

    # Use linear fit for predictions (absorbs systematic offset)
    ip_pred = sl_ip * homo_eV + int_ip
    ea_pred = sl_ea * lumo_eV + int_ea

    dG_red_dft = np.array([m['dG_reduction_kJ'] for m in valid_mols])
    dG_ox_dft = np.array([m['dG_oxidation_kJ'] for m in valid_mols])
    edef_red_dft = np.array([m['Edef_red_kJ'] for m in valid_mols])
    edef_ox_dft = np.array([m['Edef_ox_kJ'] for m in valid_mols])

    # Prepare predictions per molecule
    predictions = []
    for i, mol in enumerate(valid_mols):
        mol_name = mol['molecule_name']
        lvc = lvc_data.get(mol_name, {})

        d = {
            'molecule': mol_name,
            # DFT reference
            'dG_red_DFT': dG_red_dft[i],
            'dG_ox_DFT': dG_ox_dft[i],
            'EA_DFT': ea_dft[i],
            'IP_DFT': ip_dft[i],
            'Edef_red_DFT': edef_red_dft[i],
            'Edef_ox_DFT': edef_ox_dft[i],
            'ddU_red_DFT': ddU_red[i],
            'ddU_ox_DFT': ddU_ox[i],
            'TdS_red_DFT': mol['TdS_red_kJ'],
            'TdS_ox_DFT': mol['TdS_ox_kJ'],
            'ZPE_kJ': mol['zpe_kj'],
            'TS_vib_kJ': ts_vib.get(mol_name, float('nan')),
            'RMSD_red': lvc.get('rmsd_red', float('nan')),
            'RMSD_ox': lvc.get('rmsd_ox', float('nan')),
            # Predictions
            'EA_pred': ea_pred[i],
            'IP_pred': ip_pred[i],
            'Edef_ox_pred': lvc.get('lvc2_ox', float('nan')),
        }

        # Edef predictions using several methods:
        # 1. LVC (harmonic, neutral modes)
        # 2. LVC2 (harmonic, ionic modes at neutral geometry)
        # 3. Morse-LVC (per-mode anharmonic, calibrated D')
        edef_lvc_red = lvc.get('lvc_red', float('nan'))
        edef_lvc2_red = lvc.get('lvc2_red', float('nan'))
        rmsd_red = lvc.get('rmsd_red', float('nan'))

        # Method 1: plain LVC
        d['Edef_red_LVC'] = edef_lvc_red

        # Method 2: LVC2
        d['Edef_red_LVC2'] = edef_lvc2_red

        # Method 3: α×LVC (legacy)
        RMSD_THRESH_ANHARM = 0.15  # Å
        ALPHA_ANHARM = 1.45
        if not math.isnan(rmsd_red) and rmsd_red > RMSD_THRESH_ANHARM:
            d['Edef_red_alpha'] = ALPHA_ANHARM * edef_lvc_red
        else:
            d['Edef_red_alpha'] = edef_lvc_red

        # Method 5: Morse-LVC (neutral modes)
        md = morse_data.get(mol_name)
        if md and md.get('red_modes') and D_red is not None:
            d['Edef_red_Morse'] = morse_edef_per_mode(md['red_modes'], D_red)
        else:
            d['Edef_red_Morse'] = edef_lvc_red  # fallback

        # Method 6: Morse-LVC2 (ionic modes from SP)
        if md and md.get('red_modes_lvc2') and D_red2 is not None:
            d['Edef_red_MorseLVC2'] = morse_edef_per_mode(md['red_modes_lvc2'], D_red2)
        else:
            d['Edef_red_MorseLVC2'] = lvc.get('lvc2_red', edef_lvc_red)

        # Use LVC2 for combined prediction
        d['Edef_red_pred'] = d['Edef_red_LVC2']

        # ΔΔU predictions
        d['ddU_red_pred'] = res_ddU_red['slope'] * zpe[i] + res_ddU_red['intercept'] \
            if res_ddU_red['use_linear'] else res_ddU_red['constant']
        d['ddU_ox_pred'] = res_ddU_ox['slope'] * zpe[i] + res_ddU_ox['intercept'] \
            if res_ddU_ox['use_linear'] else res_ddU_ox['constant']

        # TΔS predictions
        if mol_name in ts_vib:
            tsv = ts_vib[mol_name]
            d['TdS_red_pred'] = res_TdS_red['slope'] * tsv + res_TdS_red['intercept'] \
                if res_TdS_red['use_linear'] else res_TdS_red['constant']
            d['TdS_ox_pred'] = res_TdS_ox['slope'] * tsv + res_TdS_ox['intercept'] \
                if res_TdS_ox['use_linear'] else res_TdS_ox['constant']
        else:
            d['TdS_red_pred'] = res_TdS_red['constant']
            d['TdS_ox_pred'] = res_TdS_ox['constant']

        # Combined ΔG: ΔG_red = -EA + Edef_red + ΔΔU_red + TΔS_red
        #               ΔG_ox =  IP + Edef_ox  + ΔΔU_ox  + TΔS_ox
        # Convention analysis.py: EA = Ee(neutre) - Ee(anion_sp) > 0 (positif)
        # La régression Koopmans reproduit EA positif, donc signe - devant EA_pred.
        d['dG_ox_pred'] = d['IP_pred'] + d['Edef_ox_pred'] + d['ddU_ox_pred'] + d['TdS_ox_pred']
        d['dG_red_pred'] = -d['EA_pred'] + d['Edef_red_pred'] + d['ddU_red_pred'] + d['TdS_red_pred']

        # Errors
        d['dG_ox_error'] = d['dG_ox_pred'] - d['dG_ox_DFT']
        d['dG_red_error'] = d['dG_red_pred'] - d['dG_red_DFT']

        predictions.append(d)

    # Print summary (all molecules)
    print(f"\n  --- Oxydation (n={len(predictions)}) ---")
    dg_ox_errs = [d['dG_ox_error'] for d in predictions if not math.isnan(d['dG_ox_error'])]
    if dg_ox_errs:
        print(f"  MAE = {np.mean(np.abs(dg_ox_errs)):.2f} kJ/mol, "
              f"RMSE = {np.sqrt(np.mean(np.array(dg_ox_errs)**2)):.2f} kJ/mol, n={len(dg_ox_errs)}")

    print(f"\n  --- Réduction (n={len(predictions)}) ---")
    dg_red_errs = [d['dG_red_error'] for d in predictions if not math.isnan(d['dG_red_error'])]
    if dg_red_errs:
        print(f"  MAE = {np.mean(np.abs(dg_red_errs)):.2f} kJ/mol, "
              f"RMSE = {np.sqrt(np.mean(np.array(dg_red_errs)**2)):.2f} kJ/mol, n={len(dg_red_errs)}")

    # Compare all Edef methods for ΔG° reduction
    print(f"\n  --- Comparaison des méthodes Edef pour ΔG° réduction (n={len(predictions)}) ---")
    for method_key, method_name in [('Edef_red_LVC', 'LVC'),
                                      ('Edef_red_LVC2', 'LVC2'),
                                      ('Edef_red_alpha', 'α×LVC'),
                                      ('Edef_red_Morse', 'Morse-LVC'),
                                      ('Edef_red_MorseLVC2', 'Morse-LVC2')]:
        errs = []
        for d in predictions:
            # Recompute dG_red with this Edef method
            edef = d[method_key]
            dg_pred = -d['EA_pred'] + edef + d['ddU_red_pred'] + d['TdS_red_pred']
            dg_err = dg_pred - d['dG_red_DFT']
            if not math.isnan(dg_err):
                errs.append(dg_err)
        if errs:
            errs_arr = np.array(errs)
            print(f"    {method_name:12s}: MAE = {np.mean(np.abs(errs_arr)):6.2f} kJ/mol, "
                  f"RMSE = {np.sqrt(np.mean(errs_arr**2)):6.2f} kJ/mol")

    # Per-molecule detail with Edef comparison
    print(f"\n  {'Molécule':12s} {'Err_ox':>7s} {'Err_red':>7s} "
          f"{'Edef_DFT':>8s} {'LVC2':>8s} {'αLVC':>8s}")
    print("  " + "-" * 60)
    for d in predictions:
        print(f"  {d['molecule']:12s} {d['dG_ox_error']:7.1f} {d['dG_red_error']:7.1f} "
              f"{d['Edef_red_DFT']:8.2f} {d['Edef_red_LVC2']:8.2f} "
              f"{d['Edef_red_alpha']:8.2f}")

    # ========================================================================
    # Step 7: Plots
    # ========================================================================
    if not args.no_plots:
        print("\n[7] Génération des graphiques...")
        plot_correlation_2panels(res_ddU_ox, res_ddU_red,
                                'Oxydation', 'Réduction',
                                output_dir / 'ZPE_vs_ddU.png')
        plot_correlation_2panels(res_TdS_ox, res_TdS_red,
                                'Oxydation', 'Réduction',
                                output_dir / 'Svib_vs_TdS.png')
        plot_dG_parity(predictions, output_dir / 'dG_parity.png', exclude=set())

    # ========================================================================
    # Step 8: Export CSV
    # ========================================================================
    print("\n[8] Export des résultats...")
    csv_path = output_dir / 'model_predictions.csv'
    if predictions:
        fields = list(predictions[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(predictions)
        print(f"  Saved: {csv_path}")

    # Summary text
    summary_path = output_dir / 'model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RÉSUMÉ DES MODÈLES PRÉDICTIFS\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- ΔΔU vs ZPE ---\n")
        for label, res in [('Réduction', res_ddU_red), ('Oxydation', res_ddU_ox)]:
            f.write(f"  {label}: modèle={res['model_name']}, R²={res['r2']:.4f}, p={res['p']:.4f}\n")
            f.write(f"    MAE_lin={res['mae_lin']:.3f}, MAE_const={res['mae_const']:.3f}\n")
            if res['use_linear']:
                f.write(f"    y = {res['slope']:.6f} * x + {res['intercept']:.4f}\n")
            else:
                f.write(f"    y = {res['constant']:.4f} (constant)\n")

        f.write("\n--- TΔS vs TS_vib ---\n")
        for label, res in [('Réduction', res_TdS_red), ('Oxydation', res_TdS_ox)]:
            f.write(f"  {label}: modèle={res['model_name']}, R²={res['r2']:.4f}, p={res['p']:.4f}\n")
            f.write(f"    MAE_lin={res['mae_lin']:.3f}, MAE_const={res['mae_const']:.3f}\n")
            if res['use_linear']:
                f.write(f"    y = {res['slope']:.6f} * x + {res['intercept']:.4f}\n")
            else:
                f.write(f"    y = {res['constant']:.4f} (constant)\n")

        f.write("\n--- Morse-LVC (correction anharmonique par mode) ---\n")
        if D_red is not None:
            f.write(f"  Réduction: D' = {D_red:.1f} quanta\n")
            f.write(f"    MAE Edef Morse = {mae_morse_red:.2f} kJ/mol\n")
            f.write(f"    MAE Edef LVC   = {mae_lvc_red:.2f} kJ/mol\n")
        if D_ox is not None:
            f.write(f"  Oxydation: D' = {D_ox:.1f} quanta\n")
            f.write(f"    MAE Edef Morse = {mae_morse_ox:.2f} kJ/mol\n")
            f.write(f"    MAE Edef LVC   = {mae_lvc_ox:.2f} kJ/mol\n")

        f.write("\n--- Koopmans (fit linéaire ε_HOMO/LUMO en eV vs IP/EA en kJ/mol) ---\n")
        f.write(f"  IP: slope={sl_ip:.4f}, intercept={int_ip:.4f}, R²={r_ip**2:.4f} (n={len(homo_eV)})\n")
        f.write(f"  EA: slope={sl_ea:.4f}, intercept={int_ea:.4f}, R²={r_ea**2:.4f}\n")

        f.write(f"\n--- ΔG° combiné (n={len(predictions)}) ---\n")
        for proc, errs_list in [('Oxydation', dg_ox_errs), ('Réduction', dg_red_errs)]:
            if errs_list:
                errs_arr = np.array(errs_list)
                f.write(f"  {proc}: MAE={np.mean(np.abs(errs_arr)):.2f} kJ/mol, "
                        f"RMSE={np.sqrt(np.mean(errs_arr**2)):.2f} kJ/mol, n={len(errs_arr)}\n")

    print(f"  Saved: {summary_path}")
    print("\nTerminé.")


if __name__ == '__main__':
    main()
