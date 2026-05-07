#!/usr/bin/env python3
"""Step 3 — multisolvent fit of the pair-homogeneous forms retained at Step 1.

For each pair (ΔΔU and TΔS) and each candidate form (the 15 subsets, with
or without Fukui), refit on the full 10-solvent dataset with an additive
dielectric correction g(ε) ∈ {const, Born, Onsager, Padé[2/1]}, K-fold CV
groupée par molécule.

Form (red and ox share monomial structure, independent coefficients):

    y_state = (β_state · d) × (1 + a_state · F_state) + g(ε; φ)

with the dielectric coefficients φ shared between red and ox (same solvent
correction). This is consistent with the rapport convention: μ and η
already absorb most of the COSMO modulation, so the residual ε term is
small and globally shared.

Outputs:
    results/step3_pair_<pool>.csv — best (form, family) ranking
    results/step3_pair_<pool>_final.json — selected joint model
    results/step3_pair_<pool>_parity_red.pdf
    results/step3_pair_<pool>_parity_ox.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))
from step1_pair_homogeneous import POOLS  # noqa: E402

DATA_DIR = REPO / "analysis" / "formula_search" / "data"
RESULTS = REPO / "analysis" / "formula_search" / "results"

EH_TO_KJ = 2625.499638
N_FOLDS = 5
EPSILON_R = {
    "Hexane": 1.88, "Toluene": 2.38, "Chloroform": 4.81,
    "Tetrahydrofuran": 7.43, "Dichloromethane": 8.93,
    "Acetone": 20.49, "Methanol": 32.61, "Acetonitrile": 35.69,
    "DMSO": 46.83, "Water": 78.39,
}


def g_const(eps, *_): return np.zeros_like(eps)
def g_born(eps, alpha): return alpha * (eps - 1.0) / (2.0 * eps + 1.0)
def g_onsager(eps, alpha): return alpha * (eps - 1.0) / (eps + 2.0)
def g_pade21(eps, a0, a1, a2, b1):
    return (a0 + a1 * eps + a2 * eps ** 2) / (1.0 + b1 * eps)

FAMILIES = {
    "const":   ("g(ε)=0", g_const, 0),
    "born":    ("a·(ε−1)/(2ε+1)", g_born, 1),
    "onsager": ("a·(ε−1)/(ε+2)", g_onsager, 1),
    "pade21":  ("(a0+a1ε+a2ε²)/(1+b1ε)", g_pade21, 4),
}


def load_full_data():
    merged = pd.read_parquet(DATA_DIR / "merged_dataset.parquet")
    delta = pd.read_parquet(DATA_DIR / "ddu_delta_descriptors.parquet")
    freq = pd.read_parquet(DATA_DIR / "freq_descriptors_full.parquet")

    df = merged.copy()
    for c in ("ddU_red", "ddU_ox", "TdS_red", "TdS_ox"):
        df[c] = df[c] * EH_TO_KJ
    df["eps_r"] = df["solvent"].map(EPSILON_R)

    df = df.merge(delta[["solvent", "molecule", "conformer",
                          "deta_red", "deta_ox", "f2_max", "f2_min"]]
                   .rename(columns={"f2_max": "f2max_corr", "f2_min": "f2min_corr"}),
                  on=["solvent", "molecule", "conformer"], how="inner")
    df = df.merge(freq[["solvent", "molecule", "conformer",
                         "dRr_A_dU_total", "dRr_A_dTS_total",
                         "dRr_A_dTSq_total", "dRr_A_dUq_total"]],
                  on=["solvent", "molecule", "conformer"], how="inner")
    return df


def build_design_pair(df, descriptors, descriptor_names_state):
    """Return D_red, D_ox, F_red, F_ox according to the descriptor names.

    descriptors: tuple of indices into descriptor_names_state['<state>_names'].
    """
    return None  # placeholder


def fit_with_solvent(D_red, D_ox, F_red, F_ox, eps,
                      y_red, y_ox, family, with_fukui):
    """Joint Nelder-Mead fit over (β_red, β_ox, a_red, a_ox, φ) where φ are
    the dielectric coefficients shared between the two states."""
    n_theta = D_red.shape[1]
    n_a = 2 if with_fukui else 0
    n_phi = FAMILIES[family][2]
    g_fn = FAMILIES[family][1]
    n_total = 2 * n_theta + n_a + n_phi

    def unpack(p):
        beta_r = p[:n_theta]
        beta_o = p[n_theta:2 * n_theta]
        idx = 2 * n_theta
        if with_fukui:
            a_r, a_o = p[idx], p[idx + 1]
            idx += 2
        else:
            a_r = a_o = 0.0
        phi = p[idx:idx + n_phi]
        return beta_r, beta_o, a_r, a_o, phi

    def predict(p):
        beta_r, beta_o, a_r, a_o, phi = unpack(p)
        return ((D_red @ beta_r) * (1.0 + a_r * F_red) + g_fn(eps, *phi),
                (D_ox @ beta_o) * (1.0 + a_o * F_ox) + g_fn(eps, *phi))

    def residual(p):
        pr, po = predict(p)
        return float(np.sum((y_red - pr) ** 2) + np.sum((y_ox - po) ** 2))

    p0 = np.zeros(n_total)
    # warm start: lstsq on each state without Fukui or solvent
    p0[:n_theta] = np.linalg.lstsq(D_red, y_red, rcond=None)[0]
    p0[n_theta:2 * n_theta] = np.linalg.lstsq(D_ox, y_ox, rcond=None)[0]
    res = minimize(residual, p0, method="Nelder-Mead",
                    options={"maxiter": 50000, "xatol": 1e-7, "fatol": 1e-7})
    return res.x


def cv_score_pair(df, D_red, D_ox, F_red, F_ox, y_red, y_ox, family, with_fukui):
    eps = df["eps_r"].to_numpy()
    groups = df["molecule"].to_numpy()
    n = D_red.shape[0]
    pred_r = np.full(n, np.nan)
    pred_o = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D_red, y_red, groups):
        p = fit_with_solvent(D_red[tr], D_ox[tr], F_red[tr], F_ox[tr], eps[tr],
                              y_red[tr], y_ox[tr], family, with_fukui)
        n_theta = D_red.shape[1]
        beta_r = p[:n_theta]
        beta_o = p[n_theta:2 * n_theta]
        idx = 2 * n_theta
        if with_fukui:
            a_r, a_o = p[idx], p[idx + 1]
            idx += 2
        else:
            a_r = a_o = 0.0
        phi = p[idx:idx + FAMILIES[family][2]]
        g_fn = FAMILIES[family][1]
        pred_r[te] = (D_red[te] @ beta_r) * (1.0 + a_r * F_red[te]) + g_fn(eps[te], *phi)
        pred_o[te] = (D_ox[te] @ beta_o) * (1.0 + a_o * F_ox[te]) + g_fn(eps[te], *phi)
    # full fit for parameters
    params_full = fit_with_solvent(D_red, D_ox, F_red, F_ox, eps,
                                     y_red, y_ox, family, with_fukui)
    return pred_r, pred_o, params_full


def metrics_one(y, yhat):
    mask = np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def parity_plot(y, pred, df, target_label, fam, mae, r2, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = plt.get_cmap("tab10")
    solvents = sorted(df.solvent.unique())
    for i, s in enumerate(solvents):
        m = df.solvent == s
        ax.scatter(y[m], pred[m], s=18, color=cmap(i), label=s,
                   alpha=0.7, edgecolors="white", linewidths=0.4)
    finite = np.isfinite(pred)
    lo = float(min(y.min(), pred[finite].min()))
    hi = float(max(y.max(), pred[finite].max()))
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{target_label} DFT (kJ/mol)")
    ax.set_ylabel(f"{target_label} predicted (kJ/mol)")
    ax.set_title(f"{target_label} — {fam} — MAE={mae:.3f} R²={r2:.3f}")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pool(pool_name: str, df: pd.DataFrame, k_select: int | None,
              with_fukui: bool, descriptor_subset: tuple[int, ...] | None):
    pool = POOLS[pool_name]
    name_red, name_ox = pool["targets"]
    descriptor_names = pool["descriptors"]
    print(f"\n=== {pool_name} | {name_red}/{name_ox}")

    # Build per-state design columns for ALL 4 descriptors, then subset.
    state_cols = {}
    for state, suffix in (("red", "_red"), ("ox", "_ox")):
        cols = []
        for d in descriptor_names:
            if d == "deta_paired":
                cols.append(df[f"deta{suffix}"].to_numpy())
            else:
                cols.append(df[d].to_numpy())
        state_cols[state] = np.column_stack(cols)
    F_red = df[pool["fukui_red"]].to_numpy()
    F_ox = df[pool["fukui_ox"]].to_numpy()
    y_red = df[name_red].to_numpy()
    y_ox = df[name_ox].to_numpy()

    # Decide subset: explicit input, or pick from elbow JSON of step 1
    if descriptor_subset is None:
        elbow_json = RESULTS / f"step1_pair_elbow_{pool_name}.json"
        if elbow_json.exists():
            elbow = json.loads(elbow_json.read_text())
            label = "with_fukui_True" if with_fukui else "with_fukui_False"
            sel = elbow[label]["elbow_5pct" if k_select is None else "absolute_best"]
            descs = sel["descriptors"].split(",")
            descriptor_subset = tuple(descriptor_names.index(d) for d in descs)
            print(f"  using elbow form: {sel['descriptors']} (k={sel['k']})")
        else:
            descriptor_subset = tuple(range(len(descriptor_names)))

    sub_names = [descriptor_names[i] for i in descriptor_subset]
    print(f"  descriptors: {sub_names}, with_fukui={with_fukui}")
    D_red = state_cols["red"][:, list(descriptor_subset)]
    D_ox = state_cols["ox"][:, list(descriptor_subset)]

    rows = []
    best_pred_r = None
    best_pred_o = None
    best_params = None
    best_fam = None
    best_mae_total = np.inf
    for fam_name, (fam_label, _, n_phi) in FAMILIES.items():
        try:
            pred_r, pred_o, params = cv_score_pair(df, D_red, D_ox,
                                                    F_red, F_ox,
                                                    y_red, y_ox,
                                                    fam_name, with_fukui)
        except Exception as e:
            print(f"  ! {fam_name} failed: {e}")
            continue
        m_r = metrics_one(y_red, pred_r)
        m_o = metrics_one(y_ox, pred_o)
        mae_total = m_r["mae"] + m_o["mae"]
        n_theta = D_red.shape[1]
        n_total = 2 * n_theta + (2 if with_fukui else 0) + n_phi
        rows.append({
            "family": fam_name,
            "n_total": n_total,
            "mae_red": m_r["mae"], "mae_ox": m_o["mae"], "mae_total": mae_total,
            "rmse_red": m_r["rmse"], "rmse_ox": m_o["rmse"],
            "r2_red": m_r["r2"], "r2_ox": m_o["r2"],
        })
        print(f"  {fam_name:10s}  k_total={n_total:2d}  "
              f"MAE_red={m_r['mae']:.4f} MAE_ox={m_o['mae']:.4f}  "
              f"total={mae_total:.4f}")
        if mae_total < best_mae_total:
            best_mae_total = mae_total
            best_pred_r = pred_r
            best_pred_o = pred_o
            best_params = params
            best_fam = fam_name

    df_res = pd.DataFrame(rows).sort_values("mae_total").reset_index(drop=True)
    out_csv = RESULTS / f"step3_pair_{pool_name}.csv"
    df_res.to_csv(out_csv, index=False, float_format="%.6g")

    # Save final
    n_theta = D_red.shape[1]
    n_phi = FAMILIES[best_fam][2]
    final = {
        "pool": pool_name,
        "targets": list(pool["targets"]),
        "descriptors": sub_names,
        "with_fukui": with_fukui,
        "fukui_red": pool["fukui_red"],
        "fukui_ox": pool["fukui_ox"],
        "best_family": best_fam,
        "n_total": int(2 * n_theta + (2 if with_fukui else 0) + n_phi),
        "beta_red": list(map(float, best_params[:n_theta])),
        "beta_ox": list(map(float, best_params[n_theta:2 * n_theta])),
        "a_red": float(best_params[2 * n_theta]) if with_fukui else 0.0,
        "a_ox": float(best_params[2 * n_theta + 1]) if with_fukui else 0.0,
        "phi": list(map(float, best_params[2 * n_theta + (2 if with_fukui else 0):
                                              2 * n_theta + (2 if with_fukui else 0) + n_phi])),
        "mae_red": float(df_res.iloc[0]["mae_red"]),
        "mae_ox": float(df_res.iloc[0]["mae_ox"]),
        "mae_total": float(df_res.iloc[0]["mae_total"]),
        "r2_red": float(df_res.iloc[0]["r2_red"]),
        "r2_ox": float(df_res.iloc[0]["r2_ox"]),
        "epsilon_r": EPSILON_R,
    }
    with open(RESULTS / f"step3_pair_{pool_name}_final.json", "w") as f:
        json.dump(final, f, indent=2)

    parity_plot(y_red, best_pred_r, df, name_red, best_fam,
                mae=df_res.iloc[0]["mae_red"], r2=df_res.iloc[0]["r2_red"],
                out_path=RESULTS / f"step3_pair_{pool_name}_parity_{name_red}.pdf")
    parity_plot(y_ox, best_pred_o, df, name_ox, best_fam,
                mae=df_res.iloc[0]["mae_ox"], r2=df_res.iloc[0]["r2_ox"],
                out_path=RESULTS / f"step3_pair_{pool_name}_parity_{name_ox}.pdf")
    print(f"  -> {out_csv.name}, _final.json, parity_*.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None)
    ap.add_argument("--no-fukui", action="store_true",
                    help="skip the Fukui multiplicative correction")
    ap.add_argument("--k-best", action="store_true",
                    help="use the absolute best k=4 form instead of the elbow")
    args = ap.parse_args()

    df = load_full_data()
    print(f"Multi-solvent dataset: {df.shape}, {df.solvent.nunique()} solvants, "
          f"{df.molecule.nunique()} molecules")

    pools_to_run = [args.pool] if args.pool else list(POOLS)
    for p in pools_to_run:
        run_pool(p, df,
                  k_select=4 if args.k_best else None,
                  with_fukui=not args.no_fukui,
                  descriptor_subset=None)


if __name__ == "__main__":
    main()
