#!/usr/bin/env python3
"""Step 3 — multisolvent generalisation via dielectric correction.

For each target, load the Step-2 final form (base × Fukui correction),
extend the dataset to all 10 solvents, and refit jointly with a dielectric
additive term g(ε):

    y = base_form(X; θ) × (1 + a · F) + g(ε; φ)

where g ∈ {const(=0), Born, Onsager, Padé[2/1]}. The CV procedure uses
5-fold GroupKFold by molecule (no leakage of conformers across folds).

Outputs:
    results/step3_solvent_<target>.csv — comparison table
    results/step3_final_form_<target>.json — selected model with all coeffs
    results/step3_parity_<target>.pdf — parity plot coloured by solvent
"""
from __future__ import annotations

import argparse
import json
import sys
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
from step2_fukui_correction import parse_label_to_design, metrics  # noqa: E402

DATA_DIR = REPO / "analysis" / "formula_search" / "data"
RESULTS = REPO / "analysis" / "formula_search" / "results"

EH_TO_KJ = 2625.499638
N_FOLDS = 5
TARGETS = ["ddU_red", "ddU_ox", "TdS_red", "TdS_ox"]

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
    "const":     ("g(ε)=0", g_const, 0),
    "born":      ("a·(ε−1)/(2ε+1)", g_born, 1),
    "onsager":   ("a·(ε−1)/(ε+2)", g_onsager, 1),
    "pade21":    ("(a0+a1ε+a2ε²)/(1+b1ε)", g_pade21, 4),
}


def load_full_data() -> pd.DataFrame:
    merged = pd.read_parquet(DATA_DIR / "merged_dataset.parquet")
    delta = pd.read_parquet(DATA_DIR / "ddu_delta_descriptors.parquet")
    freq = pd.read_parquet(DATA_DIR / "freq_descriptors_full.parquet")

    df = merged.copy()
    for c in TARGETS:
        df[c] = df[c] * EH_TO_KJ
    df["eps_r"] = df["solvent"].map(EPSILON_R)

    df = df.merge(delta[["solvent", "molecule", "conformer", "deta_red", "deta_ox",
                          "f2_min"]],
                  on=["solvent", "molecule", "conformer"], how="inner")
    df = df.merge(freq, on=["solvent", "molecule", "conformer"], how="inner")

    # Fukui paired columns (red → f²_max, ox → f²_min).
    df["f2_paired_red"] = df["f2_max"]
    df["f2_paired_ox"] = df["f2_min"]
    df["f2_paired_red_noO"] = df["f2_max_O"]
    df["f2_paired_ox_noO"] = -df["f2_max_O"]   # proxy: no f2_min_noO available
    df["fplus_max_noO"] = df["fplus_max_O"]
    df["fminus_max_noO"] = df["fminus_max_O"]
    df["f2_max_noO_col"] = df["f2_max_O"]
    return df


def build_predictor(D, F, eps, theta, a, family, phi):
    base = (D @ theta) * (1.0 + a * F)
    g_fn = FAMILIES[family][1]
    return base + g_fn(eps, *phi)


def fit_full_model(D, F, eps, y, family, n_phi):
    """Joint fit by alternating: θ via weighted lstsq (with current a, φ),
    a via 1-D optim, φ via Levenberg-Marquardt-like (scipy.minimize)."""
    n_theta = D.shape[1]
    g_fn = FAMILIES[family][1]

    def unpack(params):
        return params[:n_theta], params[n_theta], params[n_theta + 1:]

    def predict(params):
        theta, a, phi = unpack(params)
        return (D @ theta) * (1.0 + a * F) + g_fn(eps, *phi)

    def residual_sq(params):
        return float(np.sum((y - predict(params)) ** 2))

    p0 = np.zeros(n_theta + 1 + n_phi)
    # warm start θ via lstsq (no correction)
    p0[:n_theta] = np.linalg.lstsq(D, y, rcond=None)[0]
    res = minimize(residual_sq, p0, method="Nelder-Mead",
                    options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7})
    return res.x


def cv_score(df: pd.DataFrame, y: np.ndarray,
             D_full: np.ndarray, F_full: np.ndarray, eps_full: np.ndarray,
             family: str) -> tuple[np.ndarray, np.ndarray]:
    n_phi = FAMILIES[family][2]
    groups = df["molecule"].to_numpy()
    n = D_full.shape[0]
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D_full, y, groups):
        try:
            params = fit_full_model(D_full[tr], F_full[tr], eps_full[tr],
                                     y[tr], family, n_phi)
            n_theta = D_full.shape[1]
            theta, a, phi = params[:n_theta], params[n_theta], params[n_theta + 1:]
            pred[te] = build_predictor(D_full[te], F_full[te], eps_full[te],
                                        theta, a, family, phi)
        except Exception:
            pass
    # full fit
    params = fit_full_model(D_full, F_full, eps_full, y, family, n_phi)
    return pred, params


def fukui_column(target: str, fukui: str, df: pd.DataFrame) -> np.ndarray | None:
    if fukui == "none":
        return np.zeros(len(df))
    if fukui == "f2_paired":
        return df[f"f2_paired_red" if target.endswith("_red") else "f2_paired_ox"].to_numpy()
    if fukui == "f2_paired_noO":
        return df[f"f2_paired_red_noO" if target.endswith("_red") else "f2_paired_ox_noO"].to_numpy()
    if fukui == "f2_max_noO":
        return df["f2_max_noO_col"].to_numpy()
    if fukui in df.columns:
        return df[fukui].to_numpy()
    return None


def parity_plot(df, y, pred, target, family, mae, r2, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    palette = plt.get_cmap("tab10")
    solvents = sorted(df.solvent.unique())
    for i, s in enumerate(solvents):
        m = df.solvent == s
        ax.scatter(y[m], pred[m], s=18, color=palette(i), label=s, alpha=0.7,
                   edgecolors="white", linewidths=0.4)
    lo, hi = min(y.min(), pred[np.isfinite(pred)].min()), max(y.max(), pred[np.isfinite(pred)].max())
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{target} DFT (kJ/mol)")
    ax.set_ylabel(f"{target} predicted (kJ/mol)")
    ax.set_title(f"{target} — {family} — MAE={mae:.3f}, R²={r2:.3f}")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_target(target: str, df: pd.DataFrame, top_n: int):
    final_path = RESULTS / f"step2_final_form_{target}.json"
    if not final_path.exists():
        raise FileNotFoundError(f"Run Step 2 first ({final_path})")
    base = json.loads(final_path.read_text())
    print(f"\n=== {target} | base form: {base['label']!r} | Fukui: {base['fukui']}")

    descriptors = sorted(set(d.strip() for d in base["descriptors"].split(",")))
    descriptors += ["mu", "eta", "deta_red" if target.endswith("_red") else "deta_ox"]
    descriptors = [d for d in dict.fromkeys(descriptors) if d in df.columns]

    X = df[descriptors].to_numpy()
    D_full = parse_label_to_design(base["label"], descriptors, X)
    F_full = fukui_column(target, base["fukui"], df)
    if F_full is None:
        raise RuntimeError(f"Fukui column missing: {base['fukui']}")
    y = df[target].to_numpy()
    eps = df["eps_r"].to_numpy()

    rows = []
    for fam_name, (fam_label, _, n_phi) in FAMILIES.items():
        pred, params = cv_score(df, y, D_full, F_full, eps, fam_name)
        mts = metrics(y, pred)
        n_theta = D_full.shape[1]
        rows.append({
            "family": fam_name,
            "family_label": fam_label,
            "n_params_total": n_theta + 1 + n_phi,
            "mae_cv": mts["mae"],
            "rmse_cv": mts["rmse"],
            "r2_cv": mts["r2"],
            "theta": list(map(float, params[:n_theta])),
            "a_corr": float(params[n_theta]),
            "phi": list(map(float, params[n_theta + 1:])),
        })
        print(f"  {fam_name:10s}  k={n_theta+1+n_phi:2d}  MAE={mts['mae']:.4f}  R²={mts['r2']:.4f}")

    df_res = pd.DataFrame(rows).sort_values("mae_cv").reset_index(drop=True)
    out_csv = RESULTS / f"step3_solvent_{target}.csv"
    df_res[["family", "family_label", "n_params_total", "mae_cv", "rmse_cv", "r2_cv"]] \
        .to_csv(out_csv, index=False, float_format="%.6g")

    best = df_res.iloc[0]
    final = {
        "target": target,
        "base_label": base["label"],
        "base_descriptors": base["descriptors"],
        "fukui": base["fukui"],
        "solvent_family": best["family"],
        "n_params_total": int(best["n_params_total"]),
        "mae_cv": float(best["mae_cv"]),
        "rmse_cv": float(best["rmse_cv"]),
        "r2_cv": float(best["r2_cv"]),
        "theta": best["theta"],
        "a_corr": best["a_corr"],
        "phi": best["phi"],
        "epsilon_r": EPSILON_R,
    }
    with open(RESULTS / f"step3_final_form_{target}.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"  -> {out_csv.name}")
    print(f"  -> step3_final_form_{target}.json")

    # Parity plot for the best family — recompute predictions
    fam_name = best["family"]
    pred, params = cv_score(df, y, D_full, F_full, eps, fam_name)
    parity_plot(df, y, pred, target, fam_name,
                mae=best["mae_cv"], r2=best["r2_cv"],
                out_path=RESULTS / f"step3_parity_{target}.pdf")
    print(f"  -> step3_parity_{target}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--top-n", type=int, default=4)
    args = ap.parse_args()
    df = load_full_data()
    print(f"Multi-solvent dataset: {df.shape}, {df.solvent.nunique()} solvants, "
          f"{df.molecule.nunique()} molecules")
    targets = [args.target] if args.target else TARGETS
    for t in targets:
        run_target(t, df, args.top_n)


if __name__ == "__main__":
    main()
