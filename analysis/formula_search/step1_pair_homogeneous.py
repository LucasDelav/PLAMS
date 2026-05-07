#!/usr/bin/env python3
"""Step 1 — homogeneous pair search on physically-motivated descriptor pool.

Constraint: red and ox share the SAME monomial structure (descriptor subset),
with independent coefficients. The 4 candidate descriptors are physically
motivated:

    For ΔΔU pair (ddU_red, ddU_ox):
        μ              -- electrochemical potential
        η              -- chemical hardness
        Δη_paired      -- η_ion - η_neutral (vertical SP)
        dRr_dU_total   -- direct RRHO ΔU prediction from frequency shifts
                          (paired by energy order; identical for both pairings)

    For TΔS pair (TdS_red, TdS_ox):
        μ
        η
        Δη_paired
        dRr_dTSq_total -- direct qRRHO TΔS prediction (Grimme, ω0=100 cm⁻¹)

Form (multiplicative Fukui, all subsets):
    y_state = (Σ_i β_i · d_i) × (1 + a · F_state)

with F_red = f²_max, F_ox = f²_min (rapport convention).

The search enumerates all C(4,k) subsets for k = 1, 2, 3, 4 (15 forms per
pool). Each form is fit separately on red and ox with shared structure.
Score = MAE_red + MAE_ox via 5-fold GroupKFold by molecule.

Output:
    results/step1_pair_<pool>_<fukui_on/off>.csv — full ranking
    results/step1_pair_elbow_<pool>.json — elbow analysis (selected form)
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "formula_search"))
from step2_fukui_correction import _fit_one, metrics  # noqa: E402

DATA_DIR = REPO / "analysis" / "formula_search" / "data"
RESULTS = REPO / "analysis" / "formula_search" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EH_TO_KJ = 2625.499638
SOLVENT = "Acetonitrile"
N_FOLDS = 5


def load_data():
    merged = pd.read_parquet(DATA_DIR / "merged_dataset.parquet")
    delta = pd.read_parquet(DATA_DIR / "ddu_delta_descriptors.parquet")
    freq = pd.read_parquet(DATA_DIR / "freq_descriptors_full.parquet")

    df = merged[merged.solvent == SOLVENT].copy()
    for c in ("ddU_red", "ddU_ox", "TdS_red", "TdS_ox"):
        df[c] = df[c] * EH_TO_KJ

    delta_ace = delta[delta.solvent == SOLVENT][[
        "solvent", "molecule", "conformer",
        "deta_red", "deta_ox", "f2_max", "f2_min",
    ]].rename(columns={"f2_max": "f2max_corr", "f2_min": "f2min_corr"})
    df = df.merge(delta_ace, on=["solvent", "molecule", "conformer"], how="inner")

    freq_ace = freq[freq.solvent == SOLVENT][[
        "solvent", "molecule", "conformer",
        "dRr_A_dU_total", "dRr_A_dTS_total",
        "dRr_A_dUq_total", "dRr_A_dTSq_total",
    ]]
    df = df.merge(freq_ace, on=["solvent", "molecule", "conformer"], how="inner")
    return df


def cv_fit_pair_form(D_red, D_ox, F_red, F_ox, y_red, y_ox, groups, with_fukui):
    """Fit the homogeneous form on red and ox with 5-fold CV by molecule.

    Coefficients are independent per state; the form (= columns of D) is
    shared. Returns out-of-fold predictions for both states.
    """
    n = D_red.shape[0]
    pred_red = np.full(n, np.nan)
    pred_ox = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(D_red, y_red, groups):
        if with_fukui:
            theta_r, a_r = _fit_one(D_red[tr], F_red[tr], y_red[tr])
            theta_o, a_o = _fit_one(D_ox[tr], F_ox[tr], y_ox[tr])
            pred_red[te] = (D_red[te] @ theta_r) * (1.0 + a_r * F_red[te])
            pred_ox[te] = (D_ox[te] @ theta_o) * (1.0 + a_o * F_ox[te])
        else:
            theta_r = np.linalg.lstsq(D_red[tr], y_red[tr], rcond=None)[0]
            theta_o = np.linalg.lstsq(D_ox[tr], y_ox[tr], rcond=None)[0]
            pred_red[te] = D_red[te] @ theta_r
            pred_ox[te] = D_ox[te] @ theta_o
    return pred_red, pred_ox


def search_pair(target_pair: tuple[str, str], pool_def: dict, df: pd.DataFrame):
    """Enumerate all 15 subsets of the 4-descriptor pool, fit homogeneous form
    on the pair, and rank by MAE_red + MAE_ox."""
    name_red, name_ox = target_pair
    descriptor_names = pool_def["descriptors"]   # list[str], length 4
    fukui_red_col = pool_def["fukui_red"]
    fukui_ox_col = pool_def["fukui_ox"]

    # Build design columns for each descriptor & state.
    # For paired descriptors (Δη_paired), columns differ between red/ox.
    state_data = {}
    for state, suffix in (("red", "_red"), ("ox", "_ox")):
        cols = []
        for d in descriptor_names:
            if d == "deta_paired":
                cols.append(df[f"deta{suffix}"].to_numpy())
            else:
                cols.append(df[d].to_numpy())
        state_data[state] = np.column_stack(cols)
    F_red = df[fukui_red_col].to_numpy()
    F_ox = df[fukui_ox_col].to_numpy()
    y_red = df[name_red].to_numpy()
    y_ox = df[name_ox].to_numpy()
    groups = df["molecule"].to_numpy()

    rows = []
    for k in (1, 2, 3, 4):
        for subset in combinations(range(4), k):
            D_red = state_data["red"][:, subset]
            D_ox = state_data["ox"][:, subset]
            sub_names = [descriptor_names[i] for i in subset]
            for with_fukui in (False, True):
                pred_red, pred_ox = cv_fit_pair_form(
                    D_red, D_ox, F_red, F_ox, y_red, y_ox, groups, with_fukui)
                m_red = metrics(y_red, pred_red)
                m_ox = metrics(y_ox, pred_ox)
                n_params = k * 2 + (2 if with_fukui else 0)  # red + ox + (a_red, a_ox)
                rows.append({
                    "k": k,
                    "descriptors": ",".join(sub_names),
                    "with_fukui": with_fukui,
                    "n_params": n_params,
                    "mae_red": m_red["mae"],
                    "mae_ox": m_ox["mae"],
                    "mae_total": m_red["mae"] + m_ox["mae"],
                    "rmse_red": m_red["rmse"],
                    "rmse_ox": m_ox["rmse"],
                    "r2_red": m_red["r2"],
                    "r2_ox": m_ox["r2"],
                })
    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values("mae_total").reset_index(drop=True)
    return df_res


def elbow_select(df_res: pd.DataFrame) -> dict:
    """Find the parsimony elbow: lowest k whose mae_total is within 5% of the
    global minimum, separately for with_fukui = {True, False}."""
    out = {}
    for fk in (True, False):
        sub = df_res[df_res.with_fukui == fk]
        if len(sub) == 0:
            continue
        best_mae = sub.mae_total.min()
        threshold = best_mae * 1.05
        # The most parsimonious form within 5% of the best (lowest k, then lowest MAE)
        eligible = sub[sub.mae_total <= threshold].sort_values(["k", "mae_total"])
        elbow = eligible.iloc[0].to_dict()
        elbow_strict = sub.iloc[0].to_dict()  # absolute best
        out[f"with_fukui_{fk}"] = {
            "elbow_5pct": elbow,
            "absolute_best": elbow_strict,
        }
    return out


# ---------------------------------------------------------------------------
# Pool definitions
# ---------------------------------------------------------------------------
POOLS = {
    "ddU_dRr_total": {
        "descriptors": ["mu", "eta", "deta_paired", "dRr_A_dU_total"],
        "fukui_red": "f2max_corr",
        "fukui_ox": "f2min_corr",
        "targets": ("ddU_red", "ddU_ox"),
    },
    "TdS_dRr_total_qrrho": {
        "descriptors": ["mu", "eta", "deta_paired", "dRr_A_dTSq_total"],
        "fukui_red": "f2max_corr",
        "fukui_ox": "f2min_corr",
        "targets": ("TdS_red", "TdS_ox"),
    },
    "TdS_dRr_total_rrho": {
        "descriptors": ["mu", "eta", "deta_paired", "dRr_A_dTS_total"],
        "fukui_red": "f2max_corr",
        "fukui_ox": "f2min_corr",
        "targets": ("TdS_red", "TdS_ox"),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None,
                    help=f"restrict to one pool: {list(POOLS)}")
    args = ap.parse_args()

    df = load_data()
    print(f"Acetonitrile dataset: {df.shape}, {df.molecule.nunique()} molecules")

    pools_to_run = [args.pool] if args.pool else list(POOLS)
    summary_rows = []

    for pname in pools_to_run:
        pool = POOLS[pname]
        targets = pool["targets"]
        print(f"\n{'=' * 90}")
        print(f"  POOL: {pname}")
        print(f"  Targets: {targets}")
        print(f"  Descriptors: {pool['descriptors']}")
        print(f"  Fukui (red/ox): {pool['fukui_red']} / {pool['fukui_ox']}")
        print(f"{'=' * 90}")

        df_res = search_pair(targets, pool, df)
        elbow = elbow_select(df_res)

        # Persist
        out_csv = RESULTS / f"step1_pair_{pname}.csv"
        df_res.to_csv(out_csv, index=False, float_format="%.6g")
        out_json = RESULTS / f"step1_pair_elbow_{pname}.json"
        with open(out_json, "w") as f:
            json.dump(elbow, f, indent=2)

        # Pretty-print
        print("\nAll 30 forms (with_fukui=True / =False), top 12 by mae_total:")
        cols = ["k", "with_fukui", "descriptors", "n_params",
                "mae_red", "mae_ox", "mae_total", "r2_red", "r2_ox"]
        print(df_res[cols].head(12).to_string(index=False))

        for label in ("with_fukui_True", "with_fukui_False"):
            if label not in elbow:
                continue
            print(f"\n  >>> {label.replace('with_fukui_', 'Fukui=')} elbow @ 5% of best:")
            e = elbow[label]["elbow_5pct"]
            ab = elbow[label]["absolute_best"]
            print(f"      Elbow:      k={e['k']} | {e['descriptors']:50s} | "
                  f"MAE_red={e['mae_red']:.3f} | MAE_ox={e['mae_ox']:.3f} | "
                  f"total={e['mae_total']:.3f}")
            print(f"      Best:       k={ab['k']} | {ab['descriptors']:50s} | "
                  f"MAE_red={ab['mae_red']:.3f} | MAE_ox={ab['mae_ox']:.3f} | "
                  f"total={ab['mae_total']:.3f}")

        summary_rows.append({
            "pool": pname,
            "targets": str(targets),
            "best_with_fukui": elbow.get("with_fukui_True", {}).get("absolute_best"),
            "elbow_with_fukui": elbow.get("with_fukui_True", {}).get("elbow_5pct"),
        })

    print(f"\n{'=' * 90}")
    print("All pools done.")
    print(f"{'=' * 90}")
    for r in summary_rows:
        print(f"\n  {r['pool']}:")
        for tag, key in (("best", "best_with_fukui"), ("elbow", "elbow_with_fukui")):
            d = r[key]
            if d is None:
                continue
            print(f"    {tag}: k={d['k']} | {d['descriptors']:50s} | "
                  f"total={d['mae_total']:.3f}")


if __name__ == "__main__":
    main()
