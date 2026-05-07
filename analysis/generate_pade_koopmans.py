#!/usr/bin/env python3
"""
Correction Padé de l'approximation de Koopmans en fonction de epsilon(solvant).

Trois niveaux de correction comparés :
  (a) Brut : Koopmans direct (IP = -HOMO, EA = -LUMO)
  (b) Régression linéaire globale : y = a·x + b
  (c) Padé : y = α(ε)·x + β(ε) avec α, β = approximants Padé [1/1] en ε

Produit (par propriété IP et EA) :
  - {prop}_koopmans_correction.png : 3 parity plots (brut, linéaire, Padé)
  - pade_coefficients_{prop}.png   : α(ε) et β(ε) en fonction de ε
"""

import ast
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# 15 catéchols stables du rapport (sans DEA)
CATECHOLS_15 = [
    'HMC', 'F-HMC', 'Cl-HMC',
    'DA', 'F-DA', 'Cl-DA', 'L-DA',
    'CA', 'Al-CA', 'HO-CA', 'H2C-CA',
    'DHI', 'O-DHI', 'S-DHI', 'P-DHI',
]

# ── Import rapport style ──────────────────────────────────────────────
_rapport_dir = Path(__file__).resolve().parent.parent / "rapport"
sys.path.insert(0, str(_rapport_dir))
from style import (
    apply_style, setup_grid,
    C_GRAY, FIGSIZE_DOUBLE_H, SCATTER_KW, STATS_BOX,
)
apply_style()

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

EV_TO_KJ = 96.4853  # 1 eV = 96.4853 kJ/mol

# Constantes diélectriques (298.15 K)
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

# Couleurs par solvant — palette distincte 10 couleurs
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


# ── Chargement des données ────────────────────────────────────────────

def load_dataset():
    """Charge IP/EA DFT et Koopmans depuis les summary_DFT.json (kJ/mol)."""
    records = []

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
            summary = mol_dir / f"{mol_name}_summary_DFT.json"
            if not summary.exists():
                continue

            try:
                with open(summary) as f:
                    data = ast.literal_eval(f.read())

                derived = data.get("derived", {})
                ip_dft_kj = derived.get("IP_vert_DZP", {}).get("kj")
                ea_dft_kj = derived.get("EA_vert_DZP", {}).get("kj")

                orbitals = data["states"]["neutral_dzp"]["orbitals"]
                ip_koopmans = -orbitals["HOMO"] * EV_TO_KJ
                ea_koopmans = -orbitals["LUMO"] * EV_TO_KJ

                if ip_dft_kj is not None and ea_dft_kj is not None:
                    records.append({
                        "molecule": mol_name,
                        "solvent": solvent,
                        "epsilon": eps,
                        "IP_DFT": ip_dft_kj,
                        "EA_DFT": ea_dft_kj,
                        "IP_Koopmans": ip_koopmans,
                        "EA_Koopmans": ea_koopmans,
                    })
            except Exception as e:
                print(f"  ! Erreur {solvent}/{mol_name}: {e}")

    print(f"  {len(records)} points chargés "
          f"({len(set(r['molecule'] for r in records))} molécules, "
          f"{len(set(r['solvent'] for r in records))} solvants)")
    return records


# ── Modèles ───────────────────────────────────────────────────────────

def pade_11(eps, a0, a1, b1):
    """Approximant de Padé [1/1] : (a0 + a1·ε) / (1 + b1·ε)."""
    return (a0 + a1 * eps) / (1.0 + b1 * eps)


def model_pade(X, a0, a1, b1, c0, c1, d1):
    """y = α(ε) · x_koopmans + β(ε)."""
    eps, x_koop = X
    return pade_11(eps, a0, a1, b1) * x_koop + pade_11(eps, c0, c1, d1)


def fit_pade(epsilons, x_koopmans, y_dft):
    """Ajuste les 6 paramètres Padé (b1, d1 > 0 pour éviter les pôles)."""
    X = np.array([epsilons, x_koopmans])
    p0 = [1.0, 0.01, 0.01, 0.0, 0.01, 0.01]
    bounds_lo = [-np.inf, -np.inf, 0.0, -np.inf, -np.inf, 0.0]
    bounds_hi = [ np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf]
    popt, _ = curve_fit(model_pade, X, y_dft, p0=p0,
                        bounds=(bounds_lo, bounds_hi), maxfev=50000)
    return popt


def fit_linear(x_koopmans, y_dft):
    """Régression linéaire globale y = a·x + b."""
    slope, intercept, r, _, _ = linregress(x_koopmans, y_dft)
    return slope, intercept


# ── Métriques ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, n_params=0):
    """R², R² ajusté, MAE, RMSE."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    n = len(y_true)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1) if n > n_params + 1 else r2
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"R2": r2, "R2_adj": r2_adj, "MAE": mae, "RMSE": rmse, "n": n}


# ── Figure : 3 parity plots ──────────────────────────────────────────

def _scatter_by_solvent(ax, x, y, solvents, show_legend=False):
    """Scatter coloré par solvant."""
    kw = dict(SCATTER_KW, s=25, linewidths=0.3)
    for solvent in sorted(SOLVENT_COLORS, key=lambda s: SOLVENT_EPSILON.get(s, 0),
                          reverse=True):
        mask = np.array([s == solvent for s in solvents])
        if not mask.any():
            continue
        label = f"{solvent} ($\\varepsilon$={SOLVENT_EPSILON[solvent]:.0f})"
        ax.scatter(x[mask], y[mask], color=SOLVENT_COLORS[solvent],
                   label=label, **kw)
    if show_legend:
        ax.legend(fontsize=5.5, loc='lower right', framealpha=0.85,
                  edgecolor='gray', handletextpad=0.3, labelspacing=0.2,
                  markerscale=0.7)


def _add_parity(ax, vals):
    """Ligne de parité y=x adaptée aux données."""
    vmin, vmax = np.min(vals) - 15, np.max(vals) + 15
    ax.plot([vmin, vmax], [vmin, vmax], '--', color=C_GRAY, lw=0.8,
            alpha=0.5, zorder=0)


def _stats_box(ax, metrics, pos='upper left'):
    """Boîte de stats style rapport."""
    textstr = (f"$R^2_{{\\mathrm{{adj}}}} = {metrics['R2_adj']:.3f}$\n"
               f"MAE = {metrics['MAE']:.1f} kJ/mol\n"
               f"$n = {metrics['n']}$")
    if pos == 'upper left':
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, va='top',
                fontsize=8, bbox=STATS_BOX, family='serif')
    else:
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, va='bottom',
                ha='right', fontsize=8, bbox=STATS_BOX, family='serif')


def plot_triple_parity(records, slope_lin, intercept_lin, popt, prop, output_path):
    """3 panneaux : brut, régression linéaire, Padé."""
    eps_arr = np.array([r["epsilon"] for r in records])
    y_dft = np.array([r[f"{prop}_DFT"] for r in records])
    x_koop = np.array([r[f"{prop}_Koopmans"] for r in records])
    solvents = [r["solvent"] for r in records]

    y_lin = slope_lin * x_koop + intercept_lin
    y_pade = model_pade(np.array([eps_arr, x_koop]), *popt)

    m_raw = compute_metrics(y_dft, x_koop)
    m_lin = compute_metrics(y_dft, y_lin, n_params=2)
    m_pade = compute_metrics(y_dft, y_pade, n_params=6)

    # Tailles réduites pour 3 panneaux
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 10, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8}):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

        panels = [
            (axes[0], x_koop, "Koopmans brut", m_raw),
            (axes[1], y_lin,  "Régression linéaire", m_lin),
            (axes[2], y_pade, "Correction Padé", m_pade),
        ]

        for i, (ax, y_pred, title, metrics) in enumerate(panels):
            _scatter_by_solvent(ax, y_dft, y_pred, solvents,
                                show_legend=(i == 2))
            all_vals = np.concatenate([y_dft, y_pred])
            _add_parity(ax, all_vals)
            _stats_box(ax, metrics)

            ax.set_xlabel(f"{prop} DFT (kJ/mol)")
            if i == 0:
                ax.set_ylabel(f"{prop} prédit (kJ/mol)")
            ax.set_title(title)
            setup_grid(ax)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    print(f"  {output_path.name}")


# ── Figure : fonctions Padé ──────────────────────────────────────────

def plot_pade_functions(popt, prop, output_path):
    """Trace α(ε) et β(ε) en fonction de ε, avec les solvants comme points."""
    a0, a1, b1, c0, c1, d1 = popt

    eps_curve = np.linspace(1.0, 100.0, 500)
    alpha_curve = pade_11(eps_curve, a0, a1, b1)
    beta_curve = pade_11(eps_curve, c0, c1, d1)

    solvents_sorted = sorted(SOLVENT_EPSILON.items(), key=lambda x: x[1])

    with plt.rc_context({'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 10, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8}):
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_H)

        # Formules Padé pour la légende
        def _fmt(val):
            """Format compact : supprime les zéros inutiles."""
            if abs(val) < 0.01:
                return f"{val:.1e}"
            return f"{val:.3f}"

        alpha_label = (f"$\\frac{{{_fmt(a0)} {a1:+.3f}\\,\\varepsilon}}"
                       f"{{1 + {_fmt(b1)}\\,\\varepsilon}}$")
        beta_label = (f"$\\frac{{{_fmt(c0)} {c1:+.3f}\\,\\varepsilon}}"
                      f"{{1 + {_fmt(d1)}\\,\\varepsilon}}$")

        for ax, y_curve, y_func, title, ylabel, formula in [
            (axes[0], alpha_curve,
             lambda e: pade_11(e, a0, a1, b1),
             f"$\\alpha(\\varepsilon)$ — {prop}",
             "$\\alpha(\\varepsilon)$",
             alpha_label),
            (axes[1], beta_curve,
             lambda e: pade_11(e, c0, c1, d1),
             f"$\\beta(\\varepsilon)$ — {prop}",
             "$\\beta(\\varepsilon)$ (kJ/mol)",
             beta_label),
        ]:
            ax.plot(eps_curve, y_curve, '-', color=C_GRAY, linewidth=2,
                    zorder=2, label=formula)

            for i, (solvent, eps) in enumerate(solvents_sorted):
                y_val = y_func(eps)
                ax.scatter(eps, y_val, s=45, color=SOLVENT_COLORS[solvent],
                           edgecolors='black', linewidths=0.5, zorder=4)
                y_offset = 10 if i % 2 == 0 else -14
                ax.annotate(solvent, (eps, y_val),
                            xytext=(0, y_offset), textcoords='offset points',
                            fontsize=6.5, ha='center',
                            arrowprops=dict(arrowstyle='-', color=C_GRAY,
                                            lw=0.5, shrinkA=0, shrinkB=3))

            # Asymptote
            if ax == axes[0] and abs(b1) > 1e-10:
                lim_val = a1 / b1
            elif ax == axes[1] and abs(d1) > 1e-10:
                lim_val = c1 / d1
            else:
                lim_val = np.nan
            if np.isfinite(lim_val):
                ax.axhline(y=lim_val, color=C_GRAY, linestyle=':', linewidth=1,
                            alpha=0.5)

            ax.legend(fontsize=8, loc='best', framealpha=0.85, edgecolor='gray')

            ax.set_xlabel("$\\varepsilon$ (constante diélectrique)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            setup_grid(ax)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    print(f"  {output_path.name}")


# ── Figure : parity single solvent ────────────────────────────────────

def plot_single_solvent_parity(records, slope_lin, intercept_lin, popt_global,
                                prop, output_path):
    """3 panneaux pour un solvant unique : brut, linéaire local, Padé global."""
    eps_arr = np.array([r["epsilon"] for r in records])
    y_dft = np.array([r[f"{prop}_DFT"] for r in records])
    x_koop = np.array([r[f"{prop}_Koopmans"] for r in records])
    mol_names = [r["molecule"] for r in records]
    solvent = records[0]["solvent"]

    y_lin = slope_lin * x_koop + intercept_lin
    y_pade = model_pade(np.array([eps_arr, x_koop]), *popt_global)

    m_raw = compute_metrics(y_dft, x_koop)
    m_lin = compute_metrics(y_dft, y_lin, n_params=2)
    m_pade = compute_metrics(y_dft, y_pade, n_params=6)

    color_map = {}
    unique_mols = sorted(set(mol_names))
    cmap = plt.cm.get_cmap('tab20', max(len(unique_mols), 20))
    for i, m in enumerate(unique_mols):
        color_map[m] = cmap(i)

    with plt.rc_context({'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 10, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8}):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

        panels = [
            (axes[0], x_koop, "Koopmans brut", m_raw),
            (axes[1], y_lin,  "Régression linéaire (locale)", m_lin),
            (axes[2], y_pade, "Padé (global)", m_pade),
        ]

        for i, (ax, y_pred, title, metrics) in enumerate(panels):
            colors = [color_map[m] for m in mol_names]
            ax.scatter(y_dft, y_pred, c=colors, s=30, edgecolors='white',
                       linewidths=0.3, alpha=0.85, zorder=3)

            all_vals = np.concatenate([y_dft, y_pred])
            _add_parity(ax, all_vals)
            _stats_box(ax, metrics)

            ax.set_xlabel(f"{prop} DFT (kJ/mol)")
            if i == 0:
                ax.set_ylabel(f"{prop} prédit (kJ/mol)")
            ax.set_title(title)
            setup_grid(ax)

        fig.suptitle(f"{prop} — {solvent} "
                     f"($\\varepsilon$ = {SOLVENT_EPSILON[solvent]:.1f}, "
                     f"n = {len(records)})",
                     fontsize=11, y=1.02)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    print(f"  {output_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CORRECTION PADÉ — KOOPMANS vs DFT EN FONCTION DE ε(SOLVANT)")
    print("=" * 70)

    records = load_dataset()
    if len(records) < 10:
        print("Pas assez de données.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eps_arr = np.array([r["epsilon"] for r in records])

    for prop in ["IP", "EA"]:
        print(f"\n{'─'*40}")
        print(f"  {prop}")
        print(f"{'─'*40}")

        y_dft = np.array([r[f"{prop}_DFT"] for r in records])
        x_koop = np.array([r[f"{prop}_Koopmans"] for r in records])

        # (a) Brut
        m_raw = compute_metrics(y_dft, x_koop)
        print(f"  Brut      : R²_adj = {m_raw['R2_adj']:.4f}, "
              f"MAE = {m_raw['MAE']:.1f} kJ/mol")

        # (b) Régression linéaire globale
        slope_lin, intercept_lin = fit_linear(x_koop, y_dft)
        y_lin = slope_lin * x_koop + intercept_lin
        m_lin = compute_metrics(y_dft, y_lin, n_params=2)
        print(f"  Linéaire  : R²_adj = {m_lin['R2_adj']:.4f}, "
              f"MAE = {m_lin['MAE']:.1f} kJ/mol  "
              f"(y = {slope_lin:.3f}·x {intercept_lin:+.1f})")

        # (c) Padé
        popt = fit_pade(eps_arr, x_koop, y_dft)
        a0, a1, b1, c0, c1, d1 = popt
        y_pade = model_pade(np.array([eps_arr, x_koop]), *popt)
        m_pade = compute_metrics(y_dft, y_pade, n_params=6)
        print(f"  Padé      : R²_adj = {m_pade['R2_adj']:.4f}, "
              f"MAE = {m_pade['MAE']:.1f} kJ/mol")

        print(f"\n  α(ε) = ({a0:.4f} + {a1:.4f}·ε) / (1 + {b1:.4f}·ε)")
        print(f"  β(ε) = ({c0:.1f} + {c1:.1f}·ε) / (1 + {d1:.4f}·ε)  [kJ/mol]")

        if abs(b1) > 1e-10:
            print(f"  α(ε→∞) = {a1/b1:.4f},  α(ε=1) = {pade_11(1, a0, a1, b1):.4f}")
        if abs(d1) > 1e-10:
            print(f"  β(ε→∞) = {c1/d1:.1f} kJ/mol,  β(ε=1) = {pade_11(1, c0, c1, d1):.1f} kJ/mol")

        # Figures globales
        plot_triple_parity(records, slope_lin, intercept_lin, popt, prop,
                           OUTPUT_DIR / f"{prop}_koopmans_correction.png")
        plot_pade_functions(popt, prop,
                            OUTPUT_DIR / f"pade_coefficients_{prop}.png")

        # ── Acetonitrile seul ─────────────────────────────────────────
        for subset_name, mol_filter in [("all", None), ("catechols15", CATECHOLS_15)]:
            if mol_filter is not None:
                acn = [r for r in records
                       if r["solvent"] == "Acetonitrile" and r["molecule"] in mol_filter]
            else:
                acn = [r for r in records if r["solvent"] == "Acetonitrile"]

            if len(acn) < 3:
                continue

            y_dft_acn = np.array([r[f"{prop}_DFT"] for r in acn])
            x_koop_acn = np.array([r[f"{prop}_Koopmans"] for r in acn])
            eps_acn = np.array([r["epsilon"] for r in acn])

            # Régression linéaire locale
            slope_acn, intercept_acn = fit_linear(x_koop_acn, y_dft_acn)
            y_lin_acn = slope_acn * x_koop_acn + intercept_acn

            # Padé global évalué à ε(ACN)
            y_pade_acn = model_pade(np.array([eps_acn, x_koop_acn]), *popt)

            m_raw_acn = compute_metrics(y_dft_acn, x_koop_acn)
            m_lin_acn = compute_metrics(y_dft_acn, y_lin_acn, n_params=2)
            m_pade_acn = compute_metrics(y_dft_acn, y_pade_acn, n_params=6)

            tag = f"Acetonitrile {subset_name}" if mol_filter else "Acetonitrile"
            print(f"\n  {tag} ({len(acn)} mol.) :")
            print(f"    Brut      : MAE = {m_raw_acn['MAE']:.1f} kJ/mol")
            print(f"    Lin local : R²_adj = {m_lin_acn['R2_adj']:.4f}, "
                  f"MAE = {m_lin_acn['MAE']:.1f} kJ/mol  "
                  f"(y = {slope_acn:.3f}·x {intercept_acn:+.1f})")
            print(f"    Padé glob : R²_adj = {m_pade_acn['R2_adj']:.4f}, "
                  f"MAE = {m_pade_acn['MAE']:.1f} kJ/mol")

            suffix = f"_acetonitrile_{subset_name}" if mol_filter else "_acetonitrile"
            plot_single_solvent_parity(
                acn, slope_acn, intercept_acn, popt, prop,
                OUTPUT_DIR / f"{prop}_koopmans{suffix}.png")

    print(f"\nDone. Figures dans {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
