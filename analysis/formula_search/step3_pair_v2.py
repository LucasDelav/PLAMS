#!/usr/bin/env python3
"""Step 3 v2 — multi-solvent fit using the corrected paired-descriptor scheme.

Each descriptor maps to a state-specific column (red_col, ox_col), so the
red model gets dRr_*/dwr_* and the ox model gets dRo_*/dwo_*. This is the
fix for the bug where dRr_A_dU_total was used for both states.

Form (homogeneous structure red/ox, independent coefficients):
    y_state = (Σ β_state,i · d_state,i) × (1 + a_state · F_state) + α · g(ε)

with α shared between red and ox.

Pools tested (each defined as baseline + one X_paired):
  ΔΔU pair: best candidates from step1_candidate_scan_ddU.csv
  TΔS pair: same / baseline if no winner

Outputs:
  results/step3_v2_<pool_name>.csv
  results/step3_v2_<pool_name>_final.json
  results/step3_v2_<pool_name>_parity_red.pdf
  results/step3_v2_<pool_name>_parity_ox.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Paired:
    name: str
    red_col: str
    ox_col: str
    def values(self, df, state):
        return df[self.red_col if state == "red" else self.ox_col].to_numpy(dtype=float)


def base_descriptors() -> list[Paired]:
    return [
        Paired("mu", "mu", "mu"),
        Paired("eta", "eta", "eta"),
        Paired("deta_paired", "deta_red", "deta_ox"),
    ]


# Pools to test. Each pool = baseline + one X_paired.
POOLS = {
    "baseline_3":     None,                                                                    # μ,η,Δη_paired
    "ddU_dRr_dU":     Paired("dRr_dU_total", "dRr_A_dU_total", "dRo_A_dU_total"),               # standard RRHO ΣΔU
    "ddU_dRr_dUq":    Paired("dRr_dUq_total", "dRr_A_dUq_total", "dRo_A_dUq_total"),            # qRRHO ΣΔU
    "ddU_dRr_dU_0_1500": Paired("dRr_dU_0_1500", "dRr_A_dU_0_1500", "dRo_A_dU_0_1500"),         # low-mid window
    "ddU_dRr_dUq_0_1500": Paired("dRr_dUq_0_1500", "dRr_A_dUq_0_1500", "dRo_A_dUq_0_1500"),
    "ddU_dw_dsoft":   Paired("dw_dsoft", "dwr_A_dsoft", "dwo_A_dsoft"),                          # # softening modes
    "ddU_dw_drms":    Paired("dw_drms", "dwr_A_drms", "dwo_A_drms"),                             # RMS Δω
    "TdS_dw_dmean":   Paired("dw_dmean", "dwr_A_dmean", "dwo_A_dmean"),
    "TdS_dRr_dTSq":   Paired("dRr_dTSq_total", "dRr_A_dTSq_total", "dRo_A_dTSq_total"),
    "TdS_dRr_dTSq_0_500": Paired("dRr_dTSq_0_500", "dRr_A_dTSq_0_500", "dRo_A_dTSq_0_500"),
}


def load_full_data() -> pd.DataFrame:
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
    df = df.merge(freq, on=["solvent", "molecule", "conformer"], how="inner")
    return df


def fit_with_solvent(D_red, D_ox, F_red, F_ox, eps,
                      y_red, y_ox, family, with_fukui):
    n_theta = D_red.shape[1]
    n_a = 2 if with_fukui else 0
    n_phi = FAMILIES[family][2]
    g_fn = FAMILIES[family][1]
    n_total = 2 * n_theta + n_a + n_phi

    def predict(p):
        beta_r = p[:n_theta]
        beta_o = p[n_theta:2 * n_theta]
        idx = 2 * n_theta
        if with_fukui:
            a_r, a_o = p[idx], p[idx + 1]
            idx += 2
        else:
            a_r = a_o = 0.0
        phi = p[idx:idx + n_phi]
        return ((D_red @ beta_r) * (1.0 + a_r * F_red) + g_fn(eps, *phi),
                (D_ox @ beta_o) * (1.0 + a_o * F_ox) + g_fn(eps, *phi))

    def residual(p):
        pr, po = predict(p)
        return float(np.sum((y_red - pr) ** 2) + np.sum((y_ox - po) ** 2))

    p0 = np.zeros(n_total)
    p0[:n_theta] = np.linalg.lstsq(D_red, y_red, rcond=None)[0]
    p0[n_theta:2 * n_theta] = np.linalg.lstsq(D_ox, y_ox, rcond=None)[0]
    res = minimize(residual, p0, method="Nelder-Mead",
                    options={"maxiter": 50000, "xatol": 1e-7, "fatol": 1e-7})
    return res.x


def cv_score(df, descriptors, fukui_red, fukui_ox, y_red, y_ox,
              family, with_fukui):
    eps = df["eps_r"].to_numpy()
    groups = df["molecule"].to_numpy()
    F_red = df[fukui_red].to_numpy()
    F_ox = df[fukui_ox].to_numpy()
    D_red = np.column_stack([d.values(df, "red") for d in descriptors])
    D_ox = np.column_stack([d.values(df, "ox") for d in descriptors])

    n = D_red.shape[0]
    pred_r = np.full(n, np.nan)
    pred_o = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D_red, y_red, groups):
        p = fit_with_solvent(D_red[tr], D_ox[tr], F_red[tr], F_ox[tr], eps[tr],
                              y_red[tr], y_ox[tr], family, with_fukui)
        n_theta = D_red.shape[1]
        beta_r = p[:n_theta]; beta_o = p[n_theta:2 * n_theta]
        idx = 2 * n_theta
        if with_fukui:
            a_r, a_o = p[idx], p[idx + 1]; idx += 2
        else:
            a_r = a_o = 0.0
        phi = p[idx:idx + FAMILIES[family][2]]
        g_fn = FAMILIES[family][1]
        pred_r[te] = (D_red[te] @ beta_r) * (1.0 + a_r * F_red[te]) + g_fn(eps[te], *phi)
        pred_o[te] = (D_ox[te] @ beta_o) * (1.0 + a_o * F_ox[te]) + g_fn(eps[te], *phi)
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


def parity_plot(y, pred, df, target_label, fam, mae, r2, out_path, descriptors_label):
    fig, ax = plt.subplots(figsize=(7, 7))
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
    ax.set_title(f"{target_label} — {descriptors_label} — {fam}\nMAE={mae:.3f} R²={r2:.3f}")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pool(name: str, X_or_none, df: pd.DataFrame, targets, fukui_red, fukui_ox,
              with_fukui=True):
    desc_list = list(base_descriptors())
    if X_or_none is not None:
        desc_list.append(X_or_none)
    desc_label = "+".join(d.name for d in desc_list)
    print(f"\n=== {name} | {targets[0]}/{targets[1]}")
    print(f"  descriptors: {desc_label} | with_fukui={with_fukui}")

    y_red = df[targets[0]].to_numpy()
    y_ox = df[targets[1]].to_numpy()

    rows = []
    best = (np.inf, None)
    best_params = None
    best_preds = None
    for fam_name in FAMILIES:
        try:
            pred_r, pred_o, params = cv_score(df, desc_list, fukui_red, fukui_ox,
                                                y_red, y_ox, fam_name, with_fukui)
        except Exception as e:
            print(f"  ! {fam_name} failed: {e}")
            continue
        m_r = metrics_one(y_red, pred_r)
        m_o = metrics_one(y_ox, pred_o)
        total = m_r["mae"] + m_o["mae"]
        n_phi = FAMILIES[fam_name][2]
        n_theta = len(desc_list)
        n_total = 2 * n_theta + (2 if with_fukui else 0) + n_phi
        rows.append({
            "family": fam_name, "n_total": n_total,
            "mae_red": m_r["mae"], "mae_ox": m_o["mae"], "mae_total": total,
            "rmse_red": m_r["rmse"], "rmse_ox": m_o["rmse"],
            "r2_red": m_r["r2"], "r2_ox": m_o["r2"],
        })
        print(f"  {fam_name:10s}  k_total={n_total:2d}  MAE_red={m_r['mae']:.4f}  "
              f"MAE_ox={m_o['mae']:.4f}  total={total:.4f}")
        if total < best[0]:
            best = (total, fam_name)
            best_params = params
            best_preds = (pred_r, pred_o)

    df_res = pd.DataFrame(rows).sort_values("mae_total").reset_index(drop=True)
    out = RESULTS / f"step3_v2_{name}.csv"
    df_res.to_csv(out, index=False, float_format="%.6g")

    n_theta = len(desc_list)
    n_phi = FAMILIES[best[1]][2]
    p = best_params
    final = {
        "pool_name": name,
        "targets": list(targets),
        "descriptors": [d.name for d in desc_list],
        "red_columns": [d.red_col for d in desc_list],
        "ox_columns": [d.ox_col for d in desc_list],
        "with_fukui": with_fukui,
        "best_family": best[1],
        "n_total": int(2 * n_theta + (2 if with_fukui else 0) + n_phi),
        "beta_red": list(map(float, p[:n_theta])),
        "beta_ox":  list(map(float, p[n_theta:2 * n_theta])),
        "a_red": float(p[2 * n_theta]) if with_fukui else 0.0,
        "a_ox":  float(p[2 * n_theta + 1]) if with_fukui else 0.0,
        "phi": list(map(float, p[2 * n_theta + (2 if with_fukui else 0):
                                    2 * n_theta + (2 if with_fukui else 0) + n_phi])),
        "mae_red": float(df_res.iloc[0]["mae_red"]),
        "mae_ox":  float(df_res.iloc[0]["mae_ox"]),
        "mae_total": float(df_res.iloc[0]["mae_total"]),
        "r2_red": float(df_res.iloc[0]["r2_red"]),
        "r2_ox":  float(df_res.iloc[0]["r2_ox"]),
    }
    with open(RESULTS / f"step3_v2_{name}_final.json", "w") as f:
        json.dump(final, f, indent=2)

    pred_r, pred_o = best_preds
    parity_plot(y_red, pred_r, df, targets[0], best[1],
                df_res.iloc[0]["mae_red"], df_res.iloc[0]["r2_red"],
                RESULTS / f"step3_v2_{name}_parity_{targets[0]}.pdf",
                desc_label)
    parity_plot(y_ox, pred_o, df, targets[1], best[1],
                df_res.iloc[0]["mae_ox"], df_res.iloc[0]["r2_ox"],
                RESULTS / f"step3_v2_{name}_parity_{targets[1]}.pdf",
                desc_label)
    print(f"  -> {out.name}, _final.json, parity_*.pdf")
    return df_res, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None)
    ap.add_argument("--no-fukui", action="store_true")
    args = ap.parse_args()

    df = load_full_data()
    print(f"Multi-solvent dataset: {df.shape}, {df.solvent.nunique()} solvants, "
          f"{df.molecule.nunique()} molecules")

    pools_to_run = [args.pool] if args.pool else list(POOLS)
    summary = {}

    for name in pools_to_run:
        X = POOLS[name]
        if name == "baseline_3":
            targets = ("ddU_red", "ddU_ox")
            fukui_red, fukui_ox = "f2max_corr", "f2min_corr"
            df_res, final = run_pool(f"baseline_ddU", None, df, targets,
                                       fukui_red, fukui_ox,
                                       with_fukui=not args.no_fukui)
            summary["baseline_ddU"] = final
            df_res, final = run_pool(f"baseline_TdS", None, df,
                                       ("TdS_red", "TdS_ox"),
                                       fukui_red, fukui_ox,
                                       with_fukui=not args.no_fukui)
            summary["baseline_TdS"] = final
            continue
        if name.startswith("ddU"):
            targets = ("ddU_red", "ddU_ox")
        else:
            targets = ("TdS_red", "TdS_ox")
        df_res, final = run_pool(name, X, df, targets,
                                   "f2max_corr", "f2min_corr",
                                   with_fukui=not args.no_fukui)
        summary[name] = final

    # Final summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY (multi-solvent)")
    print(f"{'=' * 90}")
    print(f"{'pool':40s} {'k':>3s} {'fam':>10s} {'MAE_red':>9s} {'MAE_ox':>9s} {'total':>9s}")
    for nm, f in sorted(summary.items(), key=lambda kv: kv[1]["mae_total"]):
        print(f"{nm:40s} {f['n_total']:3d} {f['best_family']:>10s} "
              f"{f['mae_red']:9.4f} {f['mae_ox']:9.4f} {f['mae_total']:9.4f}")


if __name__ == "__main__":
    main()
