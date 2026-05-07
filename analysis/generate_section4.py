#!/usr/bin/env python3
"""
Section 4 — Généralisation multi-solvant des modèles prédictifs.

Charge les données conformères, recalcule les propriétés dérivées depuis
les données brutes (TZ2P cohérent), puis teste des modèles à coefficients
ε-dépendants pour prédire IP, EA, E_relax, TΔS, ΔΔU, et ΔG.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Import rapport style ──────────────────────────────────────────────
_rapport_dir = Path(__file__).resolve().parent.parent / "rapport"
sys.path.insert(0, str(_rapport_dir))
from style import (
    apply_style, setup_grid,
    C_RED_DARK, C_BLUE_DARK, C_GREEN, C_GRAY,
    FIGSIZE_DOUBLE_H, SCATTER_KW, STATS_BOX,
)
apply_style()

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_PDG_DIR = PROJECT_ROOT / "dataset_predict_dg"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "section4"

EV_TO_KJ = 96.4853
EH_TO_KJ = 2625.499638

SOLVENT_EPSILON = {
    "Water":           78.4,
    "DMSO":            47.24,
    "Acetonitrile":    35.69,
    "Methanol":        32.66,
    "Acetone":         20.7,
    "Dichloromethane":  8.93,
    "Tetrahydrofuran":  7.58,
    "Chloroform":       4.81,
    "Toluene":          2.38,
    "Hexane":           1.89,
}

SOLVENT_COLORS = {
    "Water":           "#1f77b4",
    "DMSO":            "#ff7f0e",
    "Acetonitrile":    "#2ca02c",
    "Methanol":        "#d62728",
    "Acetone":         "#9467bd",
    "Dichloromethane": "#8c564b",
    "Tetrahydrofuran": "#e377c2",
    "Chloroform":      "#7f7f7f",
    "Toluene":         "#bcbd22",
    "Hexane":          "#17becf",
}


# =====================================================================
#  DATA LOADING
# =====================================================================

def _parse_json(path):
    """Parse JSON with Python-style None/True/False/nan."""
    content = path.read_text()
    content = re.sub(r'\bNone\b', 'null', content)
    content = re.sub(r'\bTrue\b', 'true', content)
    content = re.sub(r'\bFalse\b', 'false', content)
    content = re.sub(r'\bnan\b', 'null', content)
    return json.loads(content)


def _get_atoms_from_xyz(xyz_string):
    """Extract atom symbols from XYZ block."""
    lines = xyz_string.strip().split('\n')
    atoms = []
    for line in lines[2:]:  # skip count + comment
        parts = line.split()
        if parts:
            atoms.append(parts[0])
    return atoms


def _safe_get(d, *keys):
    """Nested dict access returning None on missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _load_lvc2_lookup():
    """Load LVC2 E_relax from predict-dg dataset.

    Returns dict keyed by (solvent, molecule, conformer) → (LVC2_red, LVC2_ox)
    in kJ/mol.
    """
    lookup = {}
    if not DATASET_PDG_DIR.exists():
        print("  ⚠ dataset_predict_dg/ not found — LVC2 unavailable")
        return lookup

    for solvent_dir in sorted(DATASET_PDG_DIR.iterdir()):
        if not solvent_dir.is_dir():
            continue
        solvent = solvent_dir.name

        for mol_dir in sorted(solvent_dir.iterdir()):
            if not mol_dir.is_dir():
                continue
            mol_name = mol_dir.name

            for conf_file in sorted(mol_dir.glob(f"{mol_name}_conf*_prediction.json")):
                try:
                    data = _parse_json(conf_file)
                except Exception:
                    continue

                meta = data.get('metadata', {})
                conf_num = meta.get('conformer', 0)
                derived = data.get('derived', {})

                lvc2_red = _safe_get(derived, 'E_relax_LVC2_red', 'kj')
                lvc2_ox = _safe_get(derived, 'E_relax_LVC2_ox', 'kj')

                if lvc2_red is not None and lvc2_ox is not None:
                    lookup[(solvent, mol_name, conf_num)] = (lvc2_red, lvc2_ox)

    print(f"  LVC2 chargé: {len(lookup)} conformères depuis dataset_predict_dg/")
    return lookup


def load_conformer_dataset():
    """Load all conformer-level data, recomputing derived quantities.

    Returns list of dicts, one per conformer, with:
      - metadata: molecule, conformer, solvent, epsilon, weight, n_atoms
      - targets: IP, EA, Edef_ox, Edef_red, ddU_ox, ddU_red, TdS_ox, TdS_red,
                 dG_ox, dG_red  (all kJ/mol, TZ2P-consistent)
      - features: HOMO, LUMO, gap, eta, mu, omega (kJ/mol or eV as noted),
                  hirshfeld_sumO, dipole_norm
    """
    records = []
    errors = []
    skipped_neg_ea = []
    skipped_dea = []

    for solvent_dir in sorted(DATASET_DIR.iterdir()):
        if not solvent_dir.is_dir():
            continue
        solvent = solvent_dir.name
        if solvent not in SOLVENT_EPSILON:
            continue
        eps = SOLVENT_EPSILON[solvent]

        for mol_dir in sorted(solvent_dir.iterdir()):
            if not mol_dir.is_dir():
                continue
            mol_name = mol_dir.name

            # Exclude molecules with dissociative electron attachment
            # (C-Br/C-I bond breaking confirmed by Δd > 0.3 Å)
            # and L-DOPA (extreme conformational reorganization upon oxidation)
            DEA_MOLECULES = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}
            EXCLUDED = DEA_MOLECULES | {'L-DOPA'}
            if mol_name in EXCLUDED:
                skipped_dea.append(f"{solvent}/{mol_name}")
                continue

            # Find conformer JSONs
            conf_files = sorted(mol_dir.glob(f"{mol_name}_conf*_DFT.json"))
            if not conf_files:
                continue

            for conf_file in conf_files:
                try:
                    data = _parse_json(conf_file)
                except Exception as e:
                    errors.append(f"{solvent}/{mol_name}/{conf_file.name}: {e}")
                    continue

                states = data.get('states', {})
                meta = data.get('metadata', {})
                conf_num = meta.get('conformer', 0)
                weight = meta.get('weight', 1.0)
                n_atoms = meta.get('n_atoms', 0)

                # --- Extract raw electronic energies (Hartree) ---
                Ee = {}
                required_states = [
                    'neutral_dzp', 'neutral_tz2p',
                    'oxidized_sp', 'oxidized_opt', 'oxidized_tz2p',
                    'reduced_sp', 'reduced_opt', 'reduced_tz2p',
                ]
                missing = False
                for st in required_states:
                    ee_val = _safe_get(states, st, 'thermo', 'Ee')
                    if ee_val is None:
                        missing = True
                        break
                    Ee[st] = ee_val

                if missing:
                    continue

                # --- Recompute derived quantities (TZ2P-consistent) ---
                # IP/EA: vertical at neutral geometry, TZ2P basis
                IP = (Ee['oxidized_sp'] - Ee['neutral_tz2p']) * EH_TO_KJ
                EA = (Ee['neutral_tz2p'] - Ee['reduced_sp']) * EH_TO_KJ

                # Skip negative EA
                if EA < 0:
                    skipped_neg_ea.append(f"{solvent}/{mol_name}/conf{conf_num}: EA={EA:.1f}")
                    continue

                # E_relax: geometry relaxation, TZ2P basis
                Edef_ox = (Ee['oxidized_tz2p'] - Ee['oxidized_sp']) * EH_TO_KJ
                Edef_red = (Ee['reduced_tz2p'] - Ee['reduced_sp']) * EH_TO_KJ

                # ΔΔU: thermal correction difference (DZP)
                U_neutral = _safe_get(states, 'neutral_dzp', 'thermo', 'U')
                U_red_opt = _safe_get(states, 'reduced_opt', 'thermo', 'U')
                U_ox_opt = _safe_get(states, 'oxidized_opt', 'thermo', 'U')
                if None in (U_neutral, U_red_opt, U_ox_opt):
                    continue
                ddU_red = (U_red_opt - U_neutral) * EH_TO_KJ
                ddU_ox = (U_ox_opt - U_neutral) * EH_TO_KJ

                # TΔS (DZP)
                TS_neutral = _safe_get(states, 'neutral_dzp', 'thermo', 'TS')
                TS_red_opt = _safe_get(states, 'reduced_opt', 'thermo', 'TS')
                TS_ox_opt = _safe_get(states, 'oxidized_opt', 'thermo', 'TS')
                if None in (TS_neutral, TS_red_opt, TS_ox_opt):
                    continue
                TdS_red = (TS_red_opt - TS_neutral) * EH_TO_KJ
                TdS_ox = (TS_ox_opt - TS_neutral) * EH_TO_KJ

                # ΔG from composite scheme (verified correct in JSON)
                dG_ox_json = _safe_get(data, 'derived', 'delta_G_ox', 'kj')
                dG_red_json = _safe_get(data, 'derived', 'delta_G_red', 'kj')
                if dG_ox_json is None or dG_red_json is None:
                    continue

                # Verify decomposition consistency
                residual_red = dG_red_json - (-EA + Edef_red + ddU_red - TdS_red)
                residual_ox = dG_ox_json - (IP + Edef_ox + ddU_ox - TdS_ox)

                # --- Extract features ---
                orb = _safe_get(states, 'neutral_dzp', 'orbitals') or {}
                cdft = _safe_get(states, 'neutral_dzp', 'cdft') or {}

                HOMO = orb.get('HOMO')  # eV
                LUMO = orb.get('LUMO')  # eV
                gap = orb.get('gap')    # eV
                eta = cdft.get('eta')   # eV
                mu = cdft.get('mu')     # eV
                omega = cdft.get('omega')  # eV

                if None in (HOMO, LUMO, gap, eta, mu, omega):
                    continue

                # Koopmans predictions in kJ/mol
                IP_koop = -HOMO * EV_TO_KJ
                EA_koop = -LUMO * EV_TO_KJ

                # Hirshfeld sum on O
                hirsh = _safe_get(states, 'neutral_dzp', 'charges', 'hirshfeld')
                xyz_str = _safe_get(states, 'neutral_dzp', 'xyz')
                hirshfeld_sumO = None
                if hirsh is not None and xyz_str is not None:
                    atoms = _get_atoms_from_xyz(xyz_str)
                    if len(atoms) == len(hirsh):
                        hirshfeld_sumO = sum(
                            h for a, h in zip(atoms, hirsh) if a == 'O'
                        )

                # Dipole norm
                dipole = _safe_get(states, 'neutral_dzp', 'charges', 'dipole')
                dipole_norm = None
                if dipole is not None and len(dipole) == 3:
                    dipole_norm = np.sqrt(sum(d**2 for d in dipole))

                records.append({
                    # Metadata
                    'molecule': mol_name,
                    'conformer': conf_num,
                    'solvent': solvent,
                    'epsilon': eps,
                    'weight': weight,
                    'n_atoms': n_atoms,
                    # Targets (kJ/mol)
                    'IP': IP,
                    'EA': EA,
                    'Edef_ox': Edef_ox,
                    'Edef_red': Edef_red,
                    'ddU_ox': ddU_ox,
                    'ddU_red': ddU_red,
                    'TdS_ox': TdS_ox,
                    'TdS_red': TdS_red,
                    'dG_ox': dG_ox_json,
                    'dG_red': dG_red_json,
                    'residual_ox': residual_ox,
                    'residual_red': residual_red,
                    # Features
                    'HOMO': HOMO,         # eV
                    'LUMO': LUMO,         # eV
                    'gap': gap,           # eV
                    'eta': eta,           # eV
                    'mu': mu,             # eV
                    'omega': omega,       # eV
                    'IP_koop': IP_koop,   # kJ/mol
                    'EA_koop': EA_koop,   # kJ/mol
                    'hirshfeld_sumO': hirshfeld_sumO,
                    'dipole_norm': dipole_norm,
                })

    n_mol = len(set((r['molecule'], r['solvent']) for r in records))
    n_solv = len(set(r['solvent'] for r in records))
    print(f"  Chargé: {len(records)} conformères "
          f"({n_mol} mol×solvant, {n_solv} solvants)")
    if skipped_dea:
        n_unique_dea = len(set(s.split('/')[1] for s in skipped_dea))
        print(f"  Exclus (DEA Br/I): {len(skipped_dea)} entrées "
              f"({n_unique_dea} molécules)")
    if skipped_neg_ea:
        print(f"  Exclus (EA<0): {len(skipped_neg_ea)}")
        for s in skipped_neg_ea[:10]:
            print(f"    {s}")
        if len(skipped_neg_ea) > 10:
            print(f"    ... et {len(skipped_neg_ea)-10} autres")
    if errors:
        print(f"  Erreurs: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")

    # Validate decomposition
    resid = [max(abs(r['residual_ox']), abs(r['residual_red'])) for r in records]
    print(f"  Résiduel décomposition max: {max(resid):.2f} kJ/mol, "
          f"moyen: {np.mean(resid):.2f} kJ/mol")

    # --- Merge LVC2 from predict-dg dataset ---
    lvc2_lookup = _load_lvc2_lookup()
    n_matched = 0
    for r in records:
        key = (r['solvent'], r['molecule'], r['conformer'])
        lvc2 = lvc2_lookup.get(key)
        if lvc2 is not None:
            r['LVC2_red'] = lvc2[0]
            r['LVC2_ox'] = lvc2[1]
            n_matched += 1
        else:
            r['LVC2_red'] = None
            r['LVC2_ox'] = None

    print(f"  LVC2 apparié: {n_matched}/{len(records)} conformères")

    return records


# =====================================================================
#  FUNCTION FAMILIES
# =====================================================================

FAMILIES = {
    'constant':  {'n_params': 1, 'label': 'Constante'},
    'pade_11':   {'n_params': 3, 'label': 'Padé [1/1]'},
    'pade_12':   {'n_params': 4, 'label': 'Padé [1/2]'},
    'pade_21':   {'n_params': 4, 'label': 'Padé [2/1]'},
    'pade_22':   {'n_params': 5, 'label': 'Padé [2/2]'},
    'log':       {'n_params': 2, 'label': 'Logarithme'},
    'exp':       {'n_params': 3, 'label': 'Exponentielle'},
    'sigmoid':   {'n_params': 3, 'label': 'Sigmoïde'},
    'tanh':      {'n_params': 4, 'label': 'Tanh'},
    'power':     {'n_params': 3, 'label': 'Puissance'},
}


def _eval_family(family, eps, params):
    """Evaluate a coefficient function at given epsilon values."""
    eps = np.asarray(eps, dtype=float)
    if family == 'constant':
        return np.full_like(eps, params[0])
    elif family == 'pade_11':
        a0, a1, b1 = params
        return (a0 + a1 * eps) / (1.0 + b1 * eps)
    elif family == 'pade_12':
        a0, a1, b1, b2 = params
        return (a0 + a1 * eps) / (1.0 + b1 * eps + b2 * eps**2)
    elif family == 'pade_21':
        a0, a1, a2, b1 = params
        return (a0 + a1 * eps + a2 * eps**2) / (1.0 + b1 * eps)
    elif family == 'pade_22':
        a0, a1, a2, b1, b2 = params
        return (a0 + a1 * eps + a2 * eps**2) / (1.0 + b1 * eps + b2 * eps**2)
    elif family == 'log':
        a0, a1 = params
        return a0 + a1 * np.log(eps)
    elif family == 'exp':
        a0, a1, a2 = params
        return a0 + a1 * np.exp(-a2 * eps)
    elif family == 'sigmoid':
        a0, a1, a2 = params
        return a0 / (1.0 + np.exp(-a1 * (eps - a2)))
    elif family == 'tanh':
        a0, a1, a2, a3 = params
        return a0 + a1 * np.tanh(a2 * (eps - a3))
    elif family == 'power':
        a0, a1, a2 = params
        return a0 + a1 * np.power(np.maximum(eps, 1e-10), a2)
    else:
        raise ValueError(f"Unknown family: {family}")


def _make_model(n_features, family):
    """Build a model function Y = Σ α_i(ε)·f_i + γ(ε).

    X layout: X[0] = epsilon, X[1:] = features (shape n_features × n_points)
    """
    npp = FAMILIES[family]['n_params']
    n_coeffs = n_features + 1  # + intercept
    total_params = n_coeffs * npp

    def model(X, *params):
        eps = X[0]
        result = np.zeros_like(eps, dtype=float)
        for i in range(n_features):
            offset = i * npp
            coeff = _eval_family(family, eps, params[offset:offset+npp])
            result += coeff * X[1 + i]
        # Intercept
        offset = n_features * npp
        result += _eval_family(family, eps, params[offset:offset+npp])
        return result

    return model, total_params


def _initial_guess(n_features, family):
    """Generate initial parameters and bounds for curve_fit."""
    npp = FAMILIES[family]['n_params']
    n_coeffs = n_features + 1

    p0 = []
    lo = []
    hi = []

    for i in range(n_coeffs):
        if family == 'constant':
            p0 += [1.0 if i < n_features else 0.0]
            lo += [-np.inf]
            hi += [np.inf]
        elif family == 'pade_11':
            p0 += [1.0, 0.01, 0.01]
            lo += [-np.inf, -np.inf, 0.0]  # b1 >= 0
            hi += [np.inf, np.inf, np.inf]
        elif family == 'pade_12':
            p0 += [1.0, 0.01, 0.01, 0.0001]
            lo += [-np.inf, -np.inf, 0.0, 0.0]
            hi += [np.inf, np.inf, np.inf, np.inf]
        elif family == 'pade_21':
            p0 += [1.0, 0.01, 0.0001, 0.01]
            lo += [-np.inf, -np.inf, -np.inf, 0.0]
            hi += [np.inf, np.inf, np.inf, np.inf]
        elif family == 'pade_22':
            p0 += [1.0, 0.01, 0.0001, 0.01, 0.0001]
            lo += [-np.inf, -np.inf, -np.inf, 0.0, 0.0]
            hi += [np.inf, np.inf, np.inf, np.inf, np.inf]
        elif family == 'log':
            p0 += [1.0, 0.1]
            lo += [-np.inf, -np.inf]
            hi += [np.inf, np.inf]
        elif family == 'exp':
            p0 += [1.0, 0.1, 0.05]
            lo += [-np.inf, -np.inf, 0.0]
            hi += [np.inf, np.inf, 10.0]
        elif family == 'sigmoid':
            p0 += [1.0, 0.1, 20.0]
            lo += [-np.inf, -np.inf, -np.inf]
            hi += [np.inf, np.inf, np.inf]
        elif family == 'tanh':
            p0 += [1.0, 0.1, 0.1, 20.0]
            lo += [-np.inf, -np.inf, -np.inf, -np.inf]
            hi += [np.inf, np.inf, np.inf, np.inf]
        elif family == 'power':
            p0 += [0.0, 1.0, -0.5]
            lo += [-np.inf, -np.inf, -5.0]
            hi += [np.inf, np.inf, 5.0]

    return np.array(p0), (np.array(lo), np.array(hi))


# =====================================================================
#  FITTING FRAMEWORK
# =====================================================================

def compute_metrics(y_true, y_pred, n_params=0):
    """R², R²_adj, MAE, RMSE, AIC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1) \
        if n > n_params + 1 else r2
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # AIC (assuming Gaussian errors)
    aic = n * np.log(ss_res / n + 1e-30) + 2 * n_params
    return {
        'R2': r2, 'R2_adj': r2_adj, 'MAE': mae, 'RMSE': rmse,
        'n': n, 'n_params': n_params, 'AIC': aic,
    }


def fit_model(epsilon, features, y_target, family):
    """Fit Y = Σ α_i(ε)·f_i + γ(ε) for a given function family.

    Args:
        epsilon: array (n_points,) of dielectric constants
        features: array (n_features, n_points)
        y_target: array (n_points,) of target values
        family: string key into FAMILIES

    Returns:
        dict with 'params', 'metrics', 'y_pred', 'family', 'success'
    """
    epsilon = np.asarray(epsilon, dtype=float)
    features = np.atleast_2d(features).astype(float)
    y_target = np.asarray(y_target, dtype=float)
    n_features = features.shape[0]

    model_func, total_params = _make_model(n_features, family)
    p0, bounds = _initial_guess(n_features, family)

    X = np.vstack([epsilon[np.newaxis, :], features])

    maxfev = 20000  # enough for convergence, avoids hanging

    for attempt, p0_trial in enumerate([p0, p0 * 0.1, p0 * 10]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, pcov = curve_fit(
                    model_func, X, y_target, p0=p0_trial,
                    bounds=bounds, maxfev=maxfev,
                    method='trf',
                )
            y_pred = model_func(X, *popt)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                continue
            metrics = compute_metrics(y_target, y_pred, n_params=total_params)
            return {
                'params': popt,
                'metrics': metrics,
                'y_pred': y_pred,
                'family': family,
                'n_features': n_features,
                'success': True,
            }
        except Exception:
            continue

    return {
        'params': None,
        'metrics': {'R2_adj': -999, 'MAE': 999, 'RMSE': 999,
                    'n': len(y_target), 'n_params': total_params, 'AIC': 999},
        'y_pred': None,
        'family': family,
        'n_features': n_features,
        'success': False,
    }


def fit_all_families(epsilon, features, y_target, families=None):
    """Fit all function families and return sorted results."""
    if families is None:
        families = list(FAMILIES.keys())

    results = {}
    for fam in families:
        res = fit_model(epsilon, features, y_target, fam)
        results[fam] = res
        status = "OK" if res['success'] else "FAIL"
        r2a = res['metrics']['R2_adj']
        mae = res['metrics']['MAE']
        npar = res['metrics']['n_params']
        print(f"    {FAMILIES[fam]['label']:18s}: R²_adj={r2a:7.4f}  "
              f"MAE={mae:6.1f}  n_params={npar:2d}  [{status}]")

    return results


# =====================================================================
#  PLOTTING
# =====================================================================

RC_SMALL = {
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
}


def _scatter_by_solvent(ax, x, y, solvents, show_legend=False):
    """Scatter coloré par solvant."""
    kw = dict(SCATTER_KW, s=20, linewidths=0.3)
    for solvent in sorted(SOLVENT_COLORS, key=lambda s: SOLVENT_EPSILON.get(s, 0),
                          reverse=True):
        mask = np.array([s == solvent for s in solvents])
        if not mask.any():
            continue
        label = f"{solvent} (ε={SOLVENT_EPSILON[solvent]:.0f})"
        ax.scatter(x[mask], y[mask], color=SOLVENT_COLORS[solvent],
                   label=label, **kw)
    if show_legend:
        ax.legend(fontsize=5, loc='lower right', framealpha=0.85,
                  edgecolor='gray', handletextpad=0.3, labelspacing=0.2,
                  markerscale=0.7)


def _add_parity(ax, vals):
    """Parity line y=x adapted to data range."""
    vmin, vmax = np.min(vals) - 10, np.max(vals) + 10
    ax.plot([vmin, vmax], [vmin, vmax], '--', color=C_GRAY, lw=0.8,
            alpha=0.5, zorder=0)


def _stats_box(ax, metrics, pos='upper left'):
    """Stats box: R²_adj, MAE, n."""
    textstr = (f"$R^2_{{\\mathrm{{adj}}}} = {metrics['R2_adj']:.3f}$\n"
               f"MAE = {metrics['MAE']:.1f} kJ/mol\n"
               f"$n = {metrics['n']}$")
    if pos == 'upper left':
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, va='top',
                fontsize=7, bbox=STATS_BOX, family='serif')
    else:
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, va='bottom',
                ha='right', fontsize=7, bbox=STATS_BOX, family='serif')


def plot_progression(y_true, results_sequence, solvents, prop_label,
                     output_path, unit='kJ/mol'):
    """Multi-panel parity plot showing progression of models.

    results_sequence: list of (title, result_dict)
    """
    n_panels = len(results_sequence)
    with plt.rc_context(RC_SMALL):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5))
        if n_panels == 1:
            axes = [axes]

        for i, (title, res) in enumerate(results_sequence):
            ax = axes[i]
            y_pred = res['y_pred']
            metrics = res['metrics']

            _scatter_by_solvent(ax, y_true, y_pred, solvents,
                                show_legend=(i == n_panels - 1))
            all_vals = np.concatenate([y_true, y_pred])
            _add_parity(ax, all_vals)
            _stats_box(ax, metrics)

            ax.set_xlabel(f"{prop_label} DFT ({unit})")
            if i == 0:
                ax.set_ylabel(f"{prop_label} prédit ({unit})")
            ax.set_title(title)
            setup_grid(ax)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"  → {output_path.name}")


def plot_coefficient_functions(results, feature_names, prop_label, output_path):
    """Plot α(ε), β(ε), γ(ε) as functions of ε."""
    family = results['family']
    params = results['params']
    npp = FAMILIES[family]['n_params']
    n_features = results['n_features']
    n_coeffs = n_features + 1
    coeff_names = [f"α_{name}" for name in feature_names] + ["γ (intercept)"]

    eps_curve = np.linspace(1.0, 100.0, 500)
    solvents_sorted = sorted(SOLVENT_EPSILON.items(), key=lambda x: x[1])

    n_panels = n_coeffs
    with plt.rc_context(RC_SMALL):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.0))
        if n_panels == 1:
            axes = [axes]

        for i, (ax, cname) in enumerate(zip(axes, coeff_names)):
            offset = i * npp
            p_coeff = params[offset:offset+npp]
            y_curve = _eval_family(family, eps_curve, p_coeff)
            ax.plot(eps_curve, y_curve, '-', color=C_GRAY, lw=2, zorder=2)

            for j, (solvent, eps_val) in enumerate(solvents_sorted):
                y_val = _eval_family(family, eps_val, p_coeff)
                ax.scatter(eps_val, y_val, s=35, color=SOLVENT_COLORS[solvent],
                           edgecolors='black', linewidths=0.5, zorder=4)
                y_off = 8 if j % 2 == 0 else -12
                ax.annotate(solvent, (eps_val, y_val),
                            xytext=(0, y_off), textcoords='offset points',
                            fontsize=5, ha='center')

            ax.set_xlabel("$\\varepsilon$")
            ax.set_ylabel(cname)
            ax.set_title(f"{cname} — {prop_label}")
            setup_grid(ax)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"  → {output_path.name}")


def print_recap_table(all_results, prop_label):
    """Print a summary table of all fits for a property."""
    print(f"\n  {'─'*70}")
    print(f"  RÉCAP {prop_label}")
    print(f"  {'─'*70}")
    print(f"  {'Famille':<18s} {'R²_adj':>8s} {'MAE':>8s} {'RMSE':>8s} "
          f"{'n_par':>6s} {'AIC':>10s}")
    print(f"  {'─'*70}")
    for fam_key, res in sorted(all_results.items(),
                                key=lambda x: -x[1]['metrics']['R2_adj']):
        m = res['metrics']
        status = "" if res['success'] else " [FAIL]"
        print(f"  {FAMILIES[fam_key]['label']:<18s} {m['R2_adj']:8.4f} "
              f"{m['MAE']:8.1f} {m['RMSE']:8.1f} {m['n_params']:6d} "
              f"{m['AIC']:10.1f}{status}")
    print(f"  {'─'*70}")


def generate_latex_table(all_results, prop_label, caption, label):
    """Generate exhaustive LaTeX table with ALL 10 families."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Famille & $R^2_{\mathrm{adj}}$ & MAE & RMSE & $n_{\mathrm{param}}$ & AIC \\")
    lines.append(r" & & (kJ/mol) & (kJ/mol) & & \\")
    lines.append(r"\midrule")

    for fam_key, res in sorted(all_results.items(),
                                key=lambda x: -x[1]['metrics']['R2_adj']):
        m = res['metrics']
        if not res['success']:
            lines.append(
                f"{FAMILIES[fam_key]['label']} & --- & --- & --- & "
                f"{m['n_params']} & --- \\\\"
            )
        else:
            lines.append(
                f"{FAMILIES[fam_key]['label']} & {m['R2_adj']:.4f} & "
                f"{m['MAE']:.1f} & {m['RMSE']:.1f} & {m['n_params']} & "
                f"{m['AIC']:.0f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return '\n'.join(lines)


# =====================================================================
#  SECTION 4.1: IP / EA
# =====================================================================

def run_ip_ea_analysis(records):
    """Test all function families for IP and EA prediction."""
    print("\n" + "=" * 70)
    print("SECTION 4.1 : PRÉDICTION DE IP ET EA")
    print("=" * 70)

    epsilon = np.array([r['epsilon'] for r in records])
    solvents = [r['solvent'] for r in records]

    all_results = {}

    for prop, target_key, feat_keys in [
        ('IP', 'IP', ['IP_koop', 'EA_koop']),
        ('EA', 'EA', ['IP_koop', 'EA_koop']),
    ]:
        print(f"\n  ── {prop} ──")
        y_target = np.array([r[target_key] for r in records])
        features = np.array([[r[k] for r in records] for k in feat_keys])

        feat_names = [k.replace('_koop', '') for k in feat_keys]

        print(f"  Target: {prop} (TZ2P), Features: {feat_keys}")
        print(f"  y range: [{y_target.min():.0f}, {y_target.max():.0f}] kJ/mol")

        results = fit_all_families(epsilon, features, y_target)
        all_results[prop] = results

        print_recap_table(results, prop)

        # Progression plot: constant → Padé [1/1] → best
        best_key = max(
            (k for k, v in results.items() if v['success']),
            key=lambda k: results[k]['metrics']['R2_adj']
        )
        sequence = [
            ('Constante', results['constant']),
            ('Padé [1/1]', results['pade_11']),
        ]
        if best_key not in ('constant', 'pade_11'):
            sequence.append((FAMILIES[best_key]['label'], results[best_key]))

        plot_progression(
            y_target, sequence, solvents, prop,
            OUTPUT_DIR / f"s4_{prop}_progression.png",
        )

        # Coefficient functions for best model
        if results[best_key]['success']:
            plot_coefficient_functions(
                results[best_key], feat_names, prop,
                OUTPUT_DIR / f"s4_{prop}_coefficients_{best_key}.png",
            )

        # LaTeX table
        latex = generate_latex_table(
            results, prop,
            f"Comparaison des familles de fonctions pour la prédiction de {prop}.",
            f"tab:s4_{prop.lower()}_families",
        )
        (OUTPUT_DIR / f"s4_{prop}_table.tex").write_text(latex)
        print(f"  → s4_{prop}_table.tex")

    return all_results


# =====================================================================
#  SECTION 4.2: E_relax
# =====================================================================

def run_erelax_analysis(records):
    """Test all function families for E_relax prediction.

    Uses LVC2 (ionic hessian at neutral geometry) as primary predictor
    when available, falling back to CDFT descriptors otherwise.
    """
    print("\n" + "=" * 70)
    print("SECTION 4.2 : PRÉDICTION DE E_relax")
    print("=" * 70)

    # Check LVC2 availability
    has_lvc2 = all(r['LVC2_red'] is not None and r['LVC2_ox'] is not None
                   for r in records)
    n_lvc2 = sum(1 for r in records
                 if r['LVC2_red'] is not None and r['LVC2_ox'] is not None)
    print(f"  LVC2 disponible: {n_lvc2}/{len(records)} conformères")

    if has_lvc2:
        return _run_erelax_with_lvc2(records)
    elif n_lvc2 > 0:
        records_lvc2 = [r for r in records
                        if r['LVC2_red'] is not None and r['LVC2_ox'] is not None]
        print(f"  → Utilisation du sous-ensemble avec LVC2 ({n_lvc2} conformères)")
        return _run_erelax_with_lvc2(records_lvc2)
    else:
        print("  → Pas de LVC2, fallback sur descripteurs CDFT")
        return _run_erelax_cdft_fallback(records)


def _run_erelax_with_lvc2(records):
    """E_relax prediction using LVC2 with ε-dependent calibration.

    Filters pathological LVC2 values (imaginary frequencies → divergent
    force constants) then tests LVC2 alone and combined with CDFT,
    comparing against CDFT-only baseline.
    """
    epsilon = np.array([r['epsilon'] for r in records])
    solvents = [r['solvent'] for r in records]

    all_results = {}

    lvc2_key = {'ox': 'LVC2_ox', 'red': 'LVC2_red'}

    feature_pool = {
        'eta_kj': ('η', np.array([r['eta'] * EV_TO_KJ for r in records])),
        'omega_kj': ('ω', np.array([r['omega'] * EV_TO_KJ for r in records])),
        'mu_kj': ('μ', np.array([r['mu'] * EV_TO_KJ for r in records])),
        'IP_koop': ('-ε_HOMO', np.array([r['IP_koop'] for r in records])),
        'EA_koop': ('-ε_LUMO', np.array([r['EA_koop'] for r in records])),
    }

    for direction, target_key in [('ox', 'Edef_ox'), ('red', 'Edef_red')]:
        prop = f"E_relax_{direction}"
        print(f"\n  ── {prop} ──")
        y_target = np.array([r[target_key] for r in records])
        lvc2_vals = np.array([r[lvc2_key[direction]] for r in records])
        print(f"  y range: [{y_target.min():.1f}, {y_target.max():.1f}] kJ/mol")
        print(f"  LVC2 range: [{lvc2_vals.min():.1f}, {lvc2_vals.max():.1f}] kJ/mol")

        # --- Test all feature combos with best Padé family ---
        combos_spec = [
            ('LVC2', [lvc2_vals], ['LVC2']),
            ('LVC2+η', [lvc2_vals, feature_pool['eta_kj'][1]], ['LVC2', 'η']),
            ('LVC2+LUMO', [lvc2_vals, feature_pool['EA_koop'][1]], ['LVC2', '-ε_LUMO']),
            ('η+ω', [feature_pool['eta_kj'][1], feature_pool['omega_kj'][1]], ['η', 'ω']),
            ('Koopmans', [feature_pool['IP_koop'][1], feature_pool['EA_koop'][1]],
             ['-ε_HOMO', '-ε_LUMO']),
        ]

        print(f"\n  Comparaison des combos (meilleure famille):")
        combo_results = {}
        for combo_name, feat_list, feat_names_list in combos_spec:
            feat_arr = np.array(feat_list)
            best_r2 = -999
            best_res = None
            best_fam = None
            for fam in FAMILIES:
                res = fit_model(epsilon, feat_arr, y_target, fam)
                if res['success'] and res['metrics']['R2_adj'] > best_r2:
                    best_r2 = res['metrics']['R2_adj']
                    best_res = res
                    best_fam = fam
            if best_res:
                mae = best_res['metrics']['MAE']
                print(f"    {combo_name:25s}: R²_adj={best_r2:.4f}  "
                      f"MAE={mae:.1f}  ({FAMILIES[best_fam]['label']})")
                combo_results[combo_name] = {
                    'features': feat_arr,
                    'feat_names': feat_names_list,
                    'best_family': best_fam,
                    'best_r2': best_r2,
                    'best_mae': mae,
                }

        # Select the best combo
        best_combo_name = max(combo_results,
                              key=lambda k: combo_results[k]['best_r2'])
        best = combo_results[best_combo_name]
        features = best['features']
        feat_names = best['feat_names']
        print(f"  → Meilleure combo: {best_combo_name} ({feat_names})")

        # Full family sweep on best combo
        print(f"\n  Familles de fonctions pour {best_combo_name}:")
        results = fit_all_families(epsilon, features, y_target)
        all_results[prop] = results

        print_recap_table(results, prop)

        best_key = max(
            (k for k, v in results.items() if v['success']),
            key=lambda k: results[k]['metrics']['R2_adj']
        )
        sequence = [
            ('Constante', results['constant']),
            ('Padé [1/1]', results['pade_11']),
        ]
        if best_key not in ('constant', 'pade_11'):
            sequence.append((FAMILIES[best_key]['label'], results[best_key]))

        plot_progression(
            y_target, sequence, solvents, prop,
            OUTPUT_DIR / f"s4_{prop}_progression.png",
        )

        if results[best_key]['success']:
            plot_coefficient_functions(
                results[best_key], feat_names, prop,
                OUTPUT_DIR / f"s4_{prop}_coefficients_{best_key}.png",
            )

        latex = generate_latex_table(
            results, prop,
            f"Comparaison des familles pour $E_{{\\mathrm{{relax}}}}$ ({direction}).",
            f"tab:s4_erelax_{direction}_families",
        )
        (OUTPUT_DIR / f"s4_{prop}_table.tex").write_text(latex)

    return all_results


def _run_erelax_cdft_fallback(records):
    """E_relax prediction from CDFT descriptors (no LVC2 available)."""
    epsilon = np.array([r['epsilon'] for r in records])
    solvents = [r['solvent'] for r in records]

    all_results = {}

    feature_pool = {
        'eta_kj': ('η', np.array([r['eta'] * EV_TO_KJ for r in records])),
        'omega_kj': ('ω', np.array([r['omega'] * EV_TO_KJ for r in records])),
        'mu_kj': ('μ', np.array([r['mu'] * EV_TO_KJ for r in records])),
        'IP_koop': ('-ε_HOMO', np.array([r['IP_koop'] for r in records])),
        'EA_koop': ('-ε_LUMO', np.array([r['EA_koop'] for r in records])),
    }

    feature_combos = [
        (['IP_koop', 'EA_koop'], "Koopmans"),
        (['eta_kj', 'omega_kj'], "η+ω"),
        (['IP_koop', 'omega_kj'], "HOMO+ω"),
        (['eta_kj', 'mu_kj'], "η+μ"),
    ]

    for direction, target_key in [('ox', 'Edef_ox'), ('red', 'Edef_red')]:
        prop = f"E_relax_{direction}"
        print(f"\n  ── {prop} ──")
        y_target = np.array([r[target_key] for r in records])
        print(f"  y range: [{y_target.min():.1f}, {y_target.max():.1f}] kJ/mol")

        print(f"\n  Feature selection (constant model):")
        best_combo = None
        best_r2 = -999
        for feat_keys, combo_name in feature_combos:
            feat_arr = np.array([feature_pool[k][1] for k in feat_keys])
            res = fit_model(epsilon, feat_arr, y_target, 'constant')
            r2a = res['metrics']['R2_adj']
            mae = res['metrics']['MAE']
            print(f"    {combo_name:15s}: R²_adj={r2a:.4f}  MAE={mae:.1f}")
            if r2a > best_r2:
                best_r2 = r2a
                best_combo = (feat_keys, combo_name)

        feat_keys, combo_name = best_combo
        feat_names = [feature_pool[k][0] for k in feat_keys]
        features = np.array([feature_pool[k][1] for k in feat_keys])
        print(f"  → Meilleure combo: {combo_name} ({feat_names})")

        results = fit_all_families(epsilon, features, y_target)
        all_results[prop] = results

        print_recap_table(results, prop)

        best_key = max(
            (k for k, v in results.items() if v['success']),
            key=lambda k: results[k]['metrics']['R2_adj']
        )
        sequence = [
            ('Constante', results['constant']),
            ('Padé [1/1]', results['pade_11']),
        ]
        if best_key not in ('constant', 'pade_11'):
            sequence.append((FAMILIES[best_key]['label'], results[best_key]))

        plot_progression(
            y_target, sequence, solvents, prop,
            OUTPUT_DIR / f"s4_{prop}_progression.png",
        )

        latex = generate_latex_table(
            results, prop,
            f"Comparaison des familles pour $E_{{\\mathrm{{relax}}}}$ ({direction}).",
            f"tab:s4_erelax_{direction}_families",
        )
        (OUTPUT_DIR / f"s4_{prop}_table.tex").write_text(latex)

    return all_results


# =====================================================================
#  SECTION 4.3: TΔS
# =====================================================================

def run_tds_analysis(records):
    """Test all function families for TΔS prediction."""
    print("\n" + "=" * 70)
    print("SECTION 4.3 : PRÉDICTION DE TΔS")
    print("=" * 70)

    epsilon = np.array([r['epsilon'] for r in records])
    solvents = [r['solvent'] for r in records]

    all_results = {}

    for direction, target_key in [('ox', 'TdS_ox'), ('red', 'TdS_red')]:
        prop = f"TdS_{direction}"
        print(f"\n  ── {prop} ──")
        y_target = np.array([r[target_key] for r in records])
        print(f"  y range: [{y_target.min():.1f}, {y_target.max():.1f}] kJ/mol")
        print(f"  std: {y_target.std():.2f} kJ/mol")

        # Features: μ and η (inspired by Section 3)
        features = np.array([
            [r['mu'] * EV_TO_KJ for r in records],
            [r['eta'] * EV_TO_KJ for r in records],
        ])
        feat_names = ['μ', 'η']

        print(f"  Features: {feat_names}")
        results = fit_all_families(epsilon, features, y_target)
        all_results[prop] = results

        print_recap_table(results, prop)

        best_key = max(
            (k for k, v in results.items() if v['success']),
            key=lambda k: results[k]['metrics']['R2_adj']
        )
        sequence = [
            ('Constante', results['constant']),
            ('Padé [1/1]', results['pade_11']),
        ]
        if best_key not in ('constant', 'pade_11'):
            sequence.append((FAMILIES[best_key]['label'], results[best_key]))

        plot_progression(
            y_target, sequence, solvents, prop,
            OUTPUT_DIR / f"s4_{prop}_progression.png",
        )

        latex = generate_latex_table(
            results, prop,
            f"Comparaison des familles pour $T\\Delta S$ ({direction}).",
            f"tab:s4_tds_{direction}_families",
        )
        (OUTPUT_DIR / f"s4_{prop}_table.tex").write_text(latex)

    return all_results


# =====================================================================
#  SECTION 4.4: ΔΔU
# =====================================================================

def run_ddu_analysis(records):
    """Test all function families for ΔΔU prediction."""
    print("\n" + "=" * 70)
    print("SECTION 4.4 : PRÉDICTION DE ΔΔU")
    print("=" * 70)

    epsilon = np.array([r['epsilon'] for r in records])
    solvents = [r['solvent'] for r in records]

    all_results = {}

    for direction, target_key in [('ox', 'ddU_ox'), ('red', 'ddU_red')]:
        prop = f"ddU_{direction}"
        print(f"\n  ── {prop} ──")
        y_target = np.array([r[target_key] for r in records])
        print(f"  y range: [{y_target.min():.1f}, {y_target.max():.1f}] kJ/mol")
        print(f"  std: {y_target.std():.2f} kJ/mol")

        # Features: η (inspired by Section 3: ΔΔU_red ~ η)
        features = np.array([
            [r['eta'] * EV_TO_KJ for r in records],
            [r['mu'] * EV_TO_KJ for r in records],
        ])
        feat_names = ['η', 'μ']

        print(f"  Features: {feat_names}")
        results = fit_all_families(epsilon, features, y_target)
        all_results[prop] = results

        print_recap_table(results, prop)

        best_key = max(
            (k for k, v in results.items() if v['success']),
            key=lambda k: results[k]['metrics']['R2_adj']
        )
        sequence = [
            ('Constante', results['constant']),
            ('Padé [1/1]', results['pade_11']),
        ]
        if best_key not in ('constant', 'pade_11'):
            sequence.append((FAMILIES[best_key]['label'], results[best_key]))

        plot_progression(
            y_target, sequence, solvents, prop,
            OUTPUT_DIR / f"s4_{prop}_progression.png",
        )

        latex = generate_latex_table(
            results, prop,
            f"Comparaison des familles pour $\\Delta\\Delta U$ ({direction}).",
            f"tab:s4_ddu_{direction}_families",
        )
        (OUTPUT_DIR / f"s4_{prop}_table.tex").write_text(latex)

    return all_results


# =====================================================================
#  SECTION 4.5: COMBINED ΔG
# =====================================================================

def run_dg_combined(records, ip_ea_results, erelax_results,
                    tds_results, ddu_results):
    """Combine best models to predict ΔG."""
    print("\n" + "=" * 70)
    print("SECTION 4.5 : PRÉDICTION COMBINÉE DE ΔG")
    print("=" * 70)

    epsilon = np.array([r['epsilon'] for r in records])
    solvents = [r['solvent'] for r in records]

    def _get_best_pred(results_dict, prop_key):
        """Get predictions from best model for a property."""
        results = results_dict[prop_key]
        best_key = max(
            (k for k, v in results.items() if v['success']),
            key=lambda k: results[k]['metrics']['R2_adj']
        )
        return results[best_key]['y_pred'], FAMILIES[best_key]['label']

    # Get best predictions for each component
    IP_pred, IP_model = _get_best_pred(ip_ea_results, 'IP')
    EA_pred, EA_model = _get_best_pred(ip_ea_results, 'EA')
    Edef_ox_pred, Edef_ox_model = _get_best_pred(erelax_results, 'E_relax_ox')
    Edef_red_pred, Edef_red_model = _get_best_pred(erelax_results, 'E_relax_red')
    TdS_ox_pred, TdS_ox_model = _get_best_pred(tds_results, 'TdS_ox')
    TdS_red_pred, TdS_red_model = _get_best_pred(tds_results, 'TdS_red')
    ddU_ox_pred, ddU_ox_model = _get_best_pred(ddu_results, 'ddU_ox')
    ddU_red_pred, ddU_red_model = _get_best_pred(ddu_results, 'ddU_red')

    print(f"\n  Modèles sélectionnés:")
    print(f"    IP:        {IP_model}")
    print(f"    EA:        {EA_model}")
    print(f"    Edef_ox:   {Edef_ox_model}")
    print(f"    Edef_red:  {Edef_red_model}")
    print(f"    TdS_ox:    {TdS_ox_model}")
    print(f"    TdS_red:   {TdS_red_model}")
    print(f"    ddU_ox:    {ddU_ox_model}")
    print(f"    ddU_red:   {ddU_red_model}")

    # Combine: ΔG_red = -EA + Edef_red + ΔΔU_red - TΔS_red
    #          ΔG_ox  = IP + Edef_ox + ΔΔU_ox - TΔS_ox
    dG_red_pred = -EA_pred + Edef_red_pred + ddU_red_pred - TdS_red_pred
    dG_ox_pred = IP_pred + Edef_ox_pred + ddU_ox_pred - TdS_ox_pred

    dG_red_true = np.array([r['dG_red'] for r in records])
    dG_ox_true = np.array([r['dG_ox'] for r in records])

    m_red = compute_metrics(dG_red_true, dG_red_pred)
    m_ox = compute_metrics(dG_ox_true, dG_ox_pred)

    print(f"\n  ΔG_red: R²_adj={m_red['R2_adj']:.4f}, MAE={m_red['MAE']:.1f} kJ/mol")
    print(f"  ΔG_ox:  R²_adj={m_ox['R2_adj']:.4f}, MAE={m_ox['MAE']:.1f} kJ/mol")

    # Error budget: MAE contribution from each component
    IP_true = np.array([r['IP'] for r in records])
    EA_true = np.array([r['EA'] for r in records])
    Edef_ox_true = np.array([r['Edef_ox'] for r in records])
    Edef_red_true = np.array([r['Edef_red'] for r in records])
    TdS_ox_true = np.array([r['TdS_ox'] for r in records])
    TdS_red_true = np.array([r['TdS_red'] for r in records])
    ddU_ox_true = np.array([r['ddU_ox'] for r in records])
    ddU_red_true = np.array([r['ddU_red'] for r in records])

    print(f"\n  Error budget (MAE par composante):")
    for name, true, pred in [
        ('IP', IP_true, IP_pred), ('EA', EA_true, EA_pred),
        ('Edef_ox', Edef_ox_true, Edef_ox_pred),
        ('Edef_red', Edef_red_true, Edef_red_pred),
        ('ddU_ox', ddU_ox_true, ddU_ox_pred),
        ('ddU_red', ddU_red_true, ddU_red_pred),
        ('TdS_ox', TdS_ox_true, TdS_ox_pred),
        ('TdS_red', TdS_red_true, TdS_red_pred),
    ]:
        mae = np.mean(np.abs(true - pred))
        print(f"    {name:12s}: MAE = {mae:.1f} kJ/mol")

    # MAE by solvent
    print(f"\n  MAE par solvant:")
    for solvent in sorted(SOLVENT_EPSILON, key=lambda s: SOLVENT_EPSILON[s]):
        mask = np.array([r['solvent'] == solvent for r in records])
        if not mask.any():
            continue
        mae_red = np.mean(np.abs(dG_red_true[mask] - dG_red_pred[mask]))
        mae_ox = np.mean(np.abs(dG_ox_true[mask] - dG_ox_pred[mask]))
        print(f"    {solvent:20s} (ε={SOLVENT_EPSILON[solvent]:5.1f}): "
              f"red={mae_red:5.1f}  ox={mae_ox:5.1f} kJ/mol")

    # Parity plots
    for direction, y_true, y_pred, metrics in [
        ('red', dG_red_true, dG_red_pred, m_red),
        ('ox', dG_ox_true, dG_ox_pred, m_ox),
    ]:
        with plt.rc_context(RC_SMALL):
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))
            _scatter_by_solvent(ax, y_true, y_pred, solvents, show_legend=True)
            all_vals = np.concatenate([y_true, y_pred])
            _add_parity(ax, all_vals)
            _stats_box(ax, metrics)
            ax.set_xlabel(f"$\\Delta G_{{\\mathrm{{{direction}}}}}$ DFT (kJ/mol)")
            ax.set_ylabel(f"$\\Delta G_{{\\mathrm{{{direction}}}}}$ prédit (kJ/mol)")
            ax.set_title(f"$\\Delta G_{{\\mathrm{{{direction}}}}}$ — modèle combiné")
            setup_grid(ax)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"s4_dG_{direction}_combined.png",
                        dpi=300, bbox_inches='tight')
            plt.close()
        print(f"  → s4_dG_{direction}_combined.png")

    # Error budget bar plot
    with plt.rc_context(RC_SMALL):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, direction, components in [
            (axes[0], 'Oxydation', [
                ('IP', IP_true, IP_pred),
                ('$E_{relax}$', Edef_ox_true, Edef_ox_pred),
                ('$\\Delta\\Delta U$', ddU_ox_true, ddU_ox_pred),
                ('$T\\Delta S$', TdS_ox_true, TdS_ox_pred),
            ]),
            (axes[1], 'Réduction', [
                ('EA', EA_true, EA_pred),
                ('$E_{relax}$', Edef_red_true, Edef_red_pred),
                ('$\\Delta\\Delta U$', ddU_red_true, ddU_red_pred),
                ('$T\\Delta S$', TdS_red_true, TdS_red_pred),
            ]),
        ]:
            names = [c[0] for c in components]
            maes = [np.mean(np.abs(c[1] - c[2])) for c in components]
            colors = [C_RED_DARK, C_BLUE_DARK, C_GREEN, C_GRAY]
            ax.barh(names, maes, color=colors[:len(names)], edgecolor='white')
            ax.set_xlabel('MAE (kJ/mol)')
            ax.set_title(direction)
            setup_grid(ax, limit_ticks=False)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "s4_error_budget.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
    print(f"  → s4_error_budget.png")

    # ── Outlier analysis ─────────────────────────────────────────────
    print(f"\n  ── Analyse des outliers ──")

    # Per-conformer errors
    err_ox = np.abs(dG_ox_true - dG_ox_pred)
    err_red = np.abs(dG_red_true - dG_red_pred)

    # Component-level errors
    err_IP = np.abs(IP_true - IP_pred)
    err_EA = np.abs(EA_true - EA_pred)
    err_Edef_ox = np.abs(Edef_ox_true - Edef_ox_pred)
    err_Edef_red = np.abs(Edef_red_true - Edef_red_pred)
    err_ddU_ox = np.abs(ddU_ox_true - ddU_ox_pred)
    err_ddU_red = np.abs(ddU_red_true - ddU_red_pred)
    err_TdS_ox = np.abs(TdS_ox_true - TdS_ox_pred)
    err_TdS_red = np.abs(TdS_red_true - TdS_red_pred)

    # Outlier threshold: MAE + 2*std of error
    for direction, err_dg, comp_errs in [
        ('ox', err_ox, [
            ('IP', err_IP), ('E_relax', err_Edef_ox),
            ('ΔΔU', err_ddU_ox), ('TΔS', err_TdS_ox),
        ]),
        ('red', err_red, [
            ('EA', err_EA), ('E_relax', err_Edef_red),
            ('ΔΔU', err_ddU_red), ('TΔS', err_TdS_red),
        ]),
    ]:
        threshold = np.mean(err_dg) + 2 * np.std(err_dg)
        outlier_mask = err_dg > threshold
        n_outliers = outlier_mask.sum()
        print(f"\n  ΔG_{direction}: seuil = {threshold:.1f} kJ/mol, "
              f"{n_outliers} outliers ({n_outliers/len(records)*100:.1f}%)")

        if n_outliers > 0:
            # Sort outliers by error
            outlier_indices = np.where(outlier_mask)[0]
            outlier_indices = outlier_indices[np.argsort(-err_dg[outlier_indices])]

            outlier_data = []
            for idx in outlier_indices[:20]:
                r = records[idx]
                # Find dominant error component
                comp_vals = {name: errs[idx] for name, errs in comp_errs}
                worst_comp = max(comp_vals, key=comp_vals.get)
                outlier_data.append({
                    'molecule': r['molecule'],
                    'solvent': r['solvent'],
                    'conf': r['conformer'],
                    'err_dG': err_dg[idx],
                    'worst_comp': worst_comp,
                    'worst_err': comp_vals[worst_comp],
                    'n_atoms': r['n_atoms'],
                    'epsilon': r['epsilon'],
                })
                print(f"    {r['molecule']:25s} ({r['solvent']:15s} conf{r['conformer']}): "
                      f"|ΔG err|={err_dg[idx]:5.1f}  "
                      f"worst={worst_comp}({comp_vals[worst_comp]:.1f})")

            # Aggregate: which molecules appear most as outliers?
            from collections import Counter
            mol_counts = Counter(d['molecule'] for d in outlier_data)
            print(f"\n    Molécules les plus fréquentes parmi les outliers:")
            for mol, count in mol_counts.most_common(10):
                solvents_out = [d['solvent'] for d in outlier_data if d['molecule'] == mol]
                print(f"      {mol}: {count}× (solvants: {', '.join(sorted(set(solvents_out)))})")

            # Aggregate: which component dominates
            comp_counts = Counter(d['worst_comp'] for d in outlier_data)
            print(f"    Composante dominante des outliers:")
            for comp, count in comp_counts.most_common():
                print(f"      {comp}: {count}/{len(outlier_data)}")

    # Save outlier summary to file
    with open(OUTPUT_DIR / "outlier_analysis.txt", 'w') as f:
        for direction, err_dg, comp_errs in [
            ('ox', err_ox, [
                ('IP', err_IP), ('E_relax', err_Edef_ox),
                ('ΔΔU', err_ddU_ox), ('TΔS', err_TdS_ox),
            ]),
            ('red', err_red, [
                ('EA', err_EA), ('E_relax', err_Edef_red),
                ('ΔΔU', err_ddU_red), ('TΔS', err_TdS_red),
            ]),
        ]:
            threshold = np.mean(err_dg) + 2 * np.std(err_dg)
            outlier_mask = err_dg > threshold
            f.write(f"=== ΔG_{direction} (seuil={threshold:.1f} kJ/mol) ===\n")
            outlier_indices = np.where(outlier_mask)[0]
            outlier_indices = outlier_indices[np.argsort(-err_dg[outlier_indices])]
            for idx in outlier_indices:
                r = records[idx]
                comp_vals = {name: errs[idx] for name, errs in comp_errs}
                worst = max(comp_vals, key=comp_vals.get)
                f.write(f"{r['molecule']:25s} {r['solvent']:15s} conf{r['conformer']:2d} "
                        f"|err|={err_dg[idx]:6.1f} worst={worst}({comp_vals[worst]:.1f}) "
                        f"n_atoms={r['n_atoms']} eps={r['epsilon']:.1f}\n")
            f.write("\n")
    print(f"  → outlier_analysis.txt")

    return {
        'dG_red': {'metrics': m_red, 'y_pred': dG_red_pred},
        'dG_ox': {'metrics': m_ox, 'y_pred': dG_ox_pred},
    }


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("SECTION 4 — GÉNÉRALISATION MULTI-SOLVANT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    all_records = load_conformer_dataset()
    if len(all_records) < 50:
        print("Pas assez de données.")
        sys.exit(1)

    # Filter to records with LVC2 and remove pathological values
    LVC2_THRESHOLD = 150.0  # kJ/mol
    records_lvc2 = [r for r in all_records
                    if r['LVC2_red'] is not None and r['LVC2_ox'] is not None
                    and abs(r['LVC2_red']) <= LVC2_THRESHOLD
                    and abs(r['LVC2_ox']) <= LVC2_THRESHOLD]
    n_pathological = sum(
        1 for r in all_records
        if r['LVC2_red'] is not None and r['LVC2_ox'] is not None
        and (abs(r['LVC2_red']) > LVC2_THRESHOLD
             or abs(r['LVC2_ox']) > LVC2_THRESHOLD))
    if n_pathological:
        print(f"\n  LVC2 pathologiques filtrés (|LVC2| > {LVC2_THRESHOLD}): "
              f"{n_pathological}")
    if len(records_lvc2) > 50:
        records = records_lvc2
        print(f"  → Utilisation des {len(records)} conformères avec LVC2 propre")
    else:
        records = all_records
        print(f"\n  → LVC2 insuffisant ({len(records_lvc2)}), "
              f"utilisation de tous les {len(records)} conformères")

    n_mol = len(set((r['molecule'], r['solvent']) for r in records))
    n_solv = len(set(r['solvent'] for r in records))
    print(f"  Dataset final: {len(records)} conformères, "
          f"{n_mol} mol×solvant, {n_solv} solvants")

    # Run all analyses
    ip_ea_results = run_ip_ea_analysis(records)
    erelax_results = run_erelax_analysis(records)
    tds_results = run_tds_analysis(records)
    ddu_results = run_ddu_analysis(records)
    dg_results = run_dg_combined(records, ip_ea_results, erelax_results,
                                  tds_results, ddu_results)

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print(f"Figures et tables dans {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
