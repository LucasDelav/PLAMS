#!/usr/bin/env python3
"""
Brute-force v2: extended descriptor engineering + multivariate regression.

New ideas:
  - Individual sorted mode frequencies (ν₁, ν₂, ..., ν_last)
  - Force constants (ν²)
  - Non-linear transforms: log, 1/ν, √ν
  - Spectral power moments Σνⁿ
  - IR power moments Σ I×νⁿ
  - Cumulative fractions
  - Mode gaps (ν₂−ν₁, ν₃−ν₁, ...)
  - 2-variable and 3-variable linear regressions (all combinations of top descriptors)
  - Quadratic fits for best single descriptors
"""

import json, itertools, pathlib, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_JSON = _PROJECT_ROOT / "data" / "raw_freq_data.json"
CSV_PATH = _PROJECT_ROOT / "results" / "full_decomposition" / "molecules_summary_full_decomposition.csv"
OUT_DIR  = _PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(exist_ok=True)

R_KJ = 8.314462618e-3
T = 298.15
kT_cm = T * 0.695039  # kT in cm⁻¹


# ═══════════════════════════════════════════════════════════════════════════
#  Extended descriptor computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_extended_descriptors(freqs, ir):
    freqs = np.sort(np.array(freqs, dtype=float))
    ir_arr = np.array(ir, dtype=float) if ir is not None else None
    n = len(freqs)
    d = {}

    # ── 1. Individual mode frequencies (first 10 lowest, last 5 highest) ──
    for i in range(min(10, n)):
        d[f"mode_{i+1}"] = freqs[i]
        d[f"mode_{i+1}_sq"] = freqs[i]**2                # force constant proxy
        d[f"mode_{i+1}_inv"] = 1.0 / freqs[i] if freqs[i] > 0 else np.nan
        d[f"mode_{i+1}_log"] = np.log(freqs[i]) if freqs[i] > 0 else np.nan
        d[f"mode_{i+1}_sqrt"] = np.sqrt(freqs[i]) if freqs[i] > 0 else np.nan
    for j in range(min(5, n)):
        idx = n - 1 - j
        d[f"mode_last_{j+1}"] = freqs[idx]
        d[f"mode_last_{j+1}_sq"] = freqs[idx]**2
        d[f"mode_last_{j+1}_inv"] = 1.0 / freqs[idx] if freqs[idx] > 0 else np.nan

    # ── 2. Mode gaps ──
    if n >= 2:
        d["gap_1_2"] = freqs[1] - freqs[0]
        d["gap_2_3"] = freqs[2] - freqs[1] if n >= 3 else np.nan
        d["gap_3_4"] = freqs[3] - freqs[2] if n >= 4 else np.nan
        d["gap_1_3"] = freqs[2] - freqs[0] if n >= 3 else np.nan
        d["gap_1_5"] = freqs[4] - freqs[0] if n >= 5 else np.nan
        # Gap between last two modes
        d["gap_last_1_2"] = freqs[-1] - freqs[-2]
        # Largest gap between consecutive modes
        gaps = np.diff(freqs)
        d["max_gap"] = np.max(gaps)
        d["max_gap_position"] = np.argmax(gaps) / (n - 1)  # normalized position
        d["mean_gap"] = np.mean(gaps)
        d["std_gap"] = np.std(gaps)
        d["cv_gap"] = d["std_gap"] / d["mean_gap"] if d["mean_gap"] > 0 else np.nan

    # ── 3. Spectral power moments Σνⁿ (normalized by n) ──
    pos = freqs[freqs > 0]
    if len(pos) > 0:
        for power in [-2, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4]:
            key = f"power_moment_{str(power).replace('.','p').replace('-','m')}"
            d[key] = np.mean(pos**power)

    # ── 4. Non-linear transforms of full spectrum ──
    if len(pos) > 0:
        d["log_freq_mean"] = np.mean(np.log(pos))
        d["log_freq_std"]  = np.std(np.log(pos))
        d["inv_freq_mean"] = np.mean(1.0 / pos)
        d["inv_freq_std"]  = np.std(1.0 / pos)
        d["sqrt_freq_mean"]= np.mean(np.sqrt(pos))
        d["sqrt_freq_std"] = np.std(np.sqrt(pos))
        d["sq_freq_mean"]  = np.mean(pos**2)          # mean force constant proxy
        d["sq_freq_std"]   = np.std(pos**2)

    # ── 5. Cumulative fractions ──
    for threshold in [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500]:
        d[f"cum_frac_{threshold}"] = np.sum(freqs < threshold) / n

    # ── 6. IR-weighted moments ──
    if ir_arr is not None and len(ir_arr) == n:
        ir_tot = np.sum(ir_arr)
        if ir_tot > 0:
            w = ir_arr / ir_tot
            d["ir_weighted_mean"]  = np.sum(w * freqs)
            d["ir_weighted_std"]   = np.sqrt(np.sum(w * (freqs - d["ir_weighted_mean"])**2))
            d["ir_weighted_skew"]  = np.sum(w * ((freqs - d["ir_weighted_mean"]) / (d["ir_weighted_std"] + 1e-30))**3)
            d["ir_weighted_kurt"]  = np.sum(w * ((freqs - d["ir_weighted_mean"]) / (d["ir_weighted_std"] + 1e-30))**4)
            # IR power moments: Σ I×νⁿ
            for power in [-1, 0.5, 1, 2, 3]:
                key = f"ir_power_moment_{str(power).replace('.','p').replace('-','m')}"
                pp = pos[ir_arr[:len(pos)] > 0] if len(pos) <= len(ir_arr) else pos
                ir_p = ir_arr[:len(pos)]
                mask_p = pos > 0
                if np.any(mask_p):
                    d[key] = np.sum(ir_p[mask_p] * pos[mask_p]**power)

        # IR-weighted statistics in sub-ranges
        for name, lo, hi in [("low", 0, 500), ("high", 2000, 5000), ("mid", 500, 2000)]:
            mask = (freqs >= lo) & (freqs < hi)
            ir_sub = ir_arr[mask]
            f_sub  = freqs[mask]
            ir_s   = np.sum(ir_sub)
            if ir_s > 0 and len(f_sub) > 0:
                w_sub = ir_sub / ir_s
                d[f"ir_wt_{name}_mean"] = np.sum(w_sub * f_sub)
                d[f"ir_wt_{name}_std"]  = np.sqrt(np.sum(w_sub * (f_sub - d[f"ir_wt_{name}_mean"])**2))

    # ── 7. Thermal / physical descriptors ──
    if len(pos) > 0:
        x = pos / kT_cm
        x_clip = np.clip(x, 0, 500)
        n_be = 1.0 / (np.exp(x_clip) - 1.0 + 1e-30)
        # Thermal energy per mode
        e_th = pos * (n_be + 0.5)
        d["thermal_energy_total"] = np.sum(e_th) * 4.5563e-6 * 2625.499638  # kJ/mol
        d["thermal_energy_mean"]  = np.mean(e_th) * 4.5563e-6 * 2625.499638
        d["thermal_energy_std"]   = np.std(e_th) * 4.5563e-6 * 2625.499638
        # Vibrational entropy per mode
        s_vib = x_clip / (np.exp(x_clip) - 1 + 1e-30) - np.log(1 - np.exp(-x_clip) + 1e-30)
        d["vib_entropy_total"]    = R_KJ * np.sum(s_vib)
        d["vib_entropy_mean"]     = R_KJ * np.mean(s_vib)
        d["vib_entropy_std"]      = R_KJ * np.std(s_vib)
        # Heat capacity contribution
        cv_vib = x_clip**2 * np.exp(x_clip) / (np.exp(x_clip) - 1 + 1e-30)**2
        d["heat_cap_total"]       = R_KJ * np.sum(cv_vib)
        d["heat_cap_mean"]        = R_KJ * np.mean(cv_vib)
        d["heat_cap_std"]         = R_KJ * np.std(cv_vib)
        d["heat_cap_max"]         = R_KJ * np.max(cv_vib)
        # Debye temperature estimate (from highest freq)
        d["debye_temp"] = 1.4388 * freqs[-1]  # θ_D = hcν/kB

        # Thermal descriptors for low modes only (<500)
        mask_low = pos < 500
        if np.any(mask_low):
            pos_low = pos[mask_low]
            x_low = pos_low / kT_cm
            x_low_clip = np.clip(x_low, 0, 500)
            cv_low = x_low_clip**2 * np.exp(x_low_clip) / (np.exp(x_low_clip) - 1 + 1e-30)**2
            d["heat_cap_low_total"] = R_KJ * np.sum(cv_low)
            d["heat_cap_low_mean"]  = R_KJ * np.mean(cv_low)
            d["heat_cap_low_max"]   = R_KJ * np.max(cv_low)

        # Thermal descriptors for high modes only (>2000)
        mask_high = pos >= 2000
        if np.any(mask_high):
            pos_high = pos[mask_high]
            x_high = pos_high / kT_cm
            x_high_clip = np.clip(x_high, 0, 500)
            cv_high = x_high_clip**2 * np.exp(x_high_clip) / (np.exp(x_high_clip) - 1 + 1e-30)**2
            d["heat_cap_high_total"] = R_KJ * np.sum(cv_high)
            d["heat_cap_high_mean"]  = R_KJ * np.mean(cv_high)

    # ── 8. Spectral shape descriptors ──
    if n >= 4:
        # Spectral centroid, bandwidth, rolloff
        d["spectral_centroid"] = np.sum(freqs * np.arange(1, n+1)) / np.sum(np.arange(1, n+1))
        # Spectral flatness (geometric mean / arithmetic mean)
        if len(pos) > 0:
            d["spectral_flatness"] = np.exp(np.mean(np.log(pos))) / np.mean(pos)
        # Spectral rolloff: frequency below which 85% of energy lies
        cum_energy = np.cumsum(freqs**2)
        total_energy = cum_energy[-1]
        for pct in [0.50, 0.75, 0.85, 0.95]:
            idx = np.searchsorted(cum_energy, pct * total_energy)
            d[f"spectral_rolloff_{int(pct*100)}"] = freqs[min(idx, n-1)]

    # ── 9. Ratio descriptors ──
    if n >= 5:
        d["ratio_mode1_mode_last"] = freqs[0] / freqs[-1] if freqs[-1] > 0 else np.nan
        d["ratio_mode1_median"]    = freqs[0] / np.median(freqs) if np.median(freqs) > 0 else np.nan
        d["ratio_q25_q75"]         = np.percentile(freqs, 25) / np.percentile(freqs, 75)
        # Sum low / sum high
        s_low  = np.sum(freqs[freqs < 500])
        s_high = np.sum(freqs[freqs >= 2000])
        d["ratio_sum_low_high"]    = s_low / s_high if s_high > 0 else np.nan

    return d


def boltzmann_average(conf_descs, weights):
    common = set(conf_descs.keys()) & set(weights.keys())
    if not common:
        common = set(conf_descs.keys())
        w = {c: 1.0 / len(common) for c in common}
    else:
        total = sum(weights[c] for c in common)
        w = {c: weights[c] / total for c in common}

    all_keys = set()
    for c in common:
        all_keys.update(conf_descs[c].keys())

    avg = {}
    for key in all_keys:
        vals, ws = [], []
        for c in common:
            v = conf_descs[c].get(key, np.nan)
            if v is not None and not np.isnan(v):
                vals.append(v)
                ws.append(w[c])
        if vals:
            ws_arr = np.array(ws); ws_arr /= ws_arr.sum()
            avg[key] = float(np.sum(np.array(vals) * ws_arr))
        else:
            avg[key] = np.nan
    return avg


# ═══════════════════════════════════════════════════════════════════════════
#  Load data and compute descriptors
# ═══════════════════════════════════════════════════════════════════════════
print("Loading raw data...")
raw = json.loads(RAW_JSON.read_text())
csv = pd.read_csv(CSV_PATH)

rows = []
for mol_name in csv["molecule_name"]:
    if mol_name not in raw:
        continue
    mol = raw[mol_name]
    weights = {int(k): v for k, v in mol["boltzmann_weights"].items()}
    conf_descs = {}
    for cn_str, cdata in mol["conformers"].items():
        conf_descs[int(cn_str)] = compute_extended_descriptors(cdata["frequencies"], cdata["ir_intensities"])
    avg = boltzmann_average(conf_descs, weights)
    avg["molecule_name"] = mol_name
    rows.append(avg)

df_desc = pd.DataFrame(rows)
targets_cols = ["molecule_name", "ddU_red_kJ", "ddU_ox_kJ", "TdS_red_kJ", "TdS_ox_kJ"]
df = df_desc.merge(csv[targets_cols], on="molecule_name", how="inner")
desc_cols = [c for c in df_desc.columns if c != "molecule_name"]
print(f"Molecules: {len(df)}, Descriptors: {len(desc_cols)}")


# ═══════════════════════════════════════════════════════════════════════════
#  Single-variable correlations
# ═══════════════════════════════════════════════════════════════════════════

def run_correlations(df, desc_cols, target_col):
    results = []
    for dc in desc_cols:
        x = df[dc].values.astype(float)
        y = df[target_col].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 8:  # require at least 8 points
            continue
        x2, y2 = x[mask], y[mask]
        if np.std(x2) < 1e-12 or np.std(y2) < 1e-12:
            continue
        sl, ic, r, p, se = sp_stats.linregress(x2, y2)
        yp = sl * x2 + ic
        mae = np.mean(np.abs(y2 - yp))
        results.append({"descriptor": dc, "R2": r**2, "r": r, "p": p,
                         "slope": sl, "intercept": ic, "MAE": mae, "n": int(mask.sum())})
    results.sort(key=lambda x: -x["R2"])
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Quadratic (polynomial degree 2) fits for top descriptors
# ═══════════════════════════════════════════════════════════════════════════

def run_quadratic(df, desc_cols_top, target_col):
    results = []
    for dc in desc_cols_top:
        x = df[dc].values.astype(float)
        y = df[target_col].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 8:
            continue
        x2, y2 = x[mask], y[mask]
        if np.std(x2) < 1e-12:
            continue
        # Fit y = a*x² + b*x + c
        coeffs = np.polyfit(x2, y2, 2)
        yp = np.polyval(coeffs, x2)
        ss_res = np.sum((y2 - yp)**2)
        ss_tot = np.sum((y2 - np.mean(y2))**2)
        r2_adj = 1 - (ss_res / ss_tot) * (len(y2) - 1) / (len(y2) - 3)  # adjusted R²
        r2 = 1 - ss_res / ss_tot
        mae = np.mean(np.abs(y2 - yp))
        results.append({"descriptor": f"{dc} [quad]", "R2": r2, "R2_adj": r2_adj,
                         "MAE": mae, "n": int(mask.sum()), "coeffs": coeffs,
                         "base_desc": dc})
    results.sort(key=lambda x: -x["R2_adj"])
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Multivariate regression (2-var and 3-var)
# ═══════════════════════════════════════════════════════════════════════════

def run_multivariate(df, top_descs, target_col, max_vars=3):
    """Try all combinations of 2 and 3 descriptors from top_descs."""
    results = []
    y_full = df[target_col].values.astype(float)

    for nv in [2, 3]:
        if nv > max_vars:
            break
        for combo in itertools.combinations(top_descs[:20], nv):  # top 20 combos
            cols = list(combo)
            X = df[cols].values.astype(float)
            mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y_full))
            if mask.sum() < 8 + nv:
                continue
            X2 = X[mask]
            y2 = y_full[mask]
            # Check collinearity
            try:
                corr_matrix = np.corrcoef(X2.T)
                if np.any(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]) > 0.95):
                    continue  # skip highly collinear pairs
            except Exception:
                continue

            model = LinearRegression().fit(X2, y2)
            yp = model.predict(X2)
            r2 = r2_score(y2, yp)
            n_obs = len(y2)
            r2_adj = 1 - (1 - r2) * (n_obs - 1) / (n_obs - nv - 1)
            mae = mean_absolute_error(y2, yp)

            results.append({
                "descriptors": " + ".join(cols),
                "n_vars": nv,
                "R2": r2,
                "R2_adj": r2_adj,
                "MAE": mae,
                "n": n_obs,
                "coefs": list(model.coef_),
                "intercept": model.intercept_,
                "cols": cols,
            })

    results.sort(key=lambda x: -x["R2_adj"])
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Run everything
# ═══════════════════════════════════════════════════════════════════════════

tasks = [
    ("ddU_red_kJ",  "ΔΔU_red"),
    ("ddU_ox_kJ",   "ΔΔU_ox"),
    ("TdS_red_kJ",  "TΔS_red"),
    ("TdS_ox_kJ",   "TΔS_ox"),
]

best_results = {}

for target_col, label in tasks:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # 1. Single-variable linear
    corrs = run_correlations(df, desc_cols, target_col)
    print(f"\n  --- Top 15 single-variable (linear) ---")
    for i, r in enumerate(corrs[:15]):
        sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
        print(f"  {i+1:2d}. {r['descriptor']:45s}  R²={r['R2']:.3f}  p={r['p']:.1e}  n={r['n']} {sig}")

    # 2. Quadratic fits on top 30 descriptors
    top30 = [r["descriptor"] for r in corrs[:30]]
    quad = run_quadratic(df, top30, target_col)
    print(f"\n  --- Top 10 quadratic fits (R²_adj) ---")
    for i, r in enumerate(quad[:10]):
        print(f"  {i+1:2d}. {r['descriptor']:45s}  R²={r['R2']:.3f}  R²_adj={r['R2_adj']:.3f}  MAE={r['MAE']:.2f}  n={r['n']}")

    # 3. Multivariate
    top_descs_for_mv = [r["descriptor"] for r in corrs[:25] if r["n"] >= 13]
    mv = run_multivariate(df, top_descs_for_mv, target_col, max_vars=3)
    print(f"\n  --- Top 10 multivariate (R²_adj) ---")
    for i, r in enumerate(mv[:10]):
        print(f"  {i+1:2d}. [{r['n_vars']}v] {r['descriptors'][:70]:70s}  R²={r['R2']:.3f}  R²_adj={r['R2_adj']:.3f}  MAE={r['MAE']:.2f}  n={r['n']}")

    best_results[label] = {
        "single": corrs,
        "quadratic": quad,
        "multivariate": mv,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Plot best results for each target: 4-panel figure
#  (best linear, best quadratic, best 2-var, best 3-var)
# ═══════════════════════════════════════════════════════════════════════════

def plot_best_4panel(df, best, target_col, label, filename):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # Panel 1: Best single linear
    ax = axes[0, 0]
    r = best["single"][0]
    dc = r["descriptor"]
    x = df[dc].values.astype(float)
    y = df[target_col].values.astype(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x2, y2 = x[mask], y[mask]
    yp = r["slope"] * x2 + r["intercept"]
    ax.scatter(yp, y2, s=70, alpha=0.7, color='steelblue', edgecolors='white', linewidths=0.5, zorder=3)
    for mn, xp_v, yr_v in zip(df["molecule_name"].values[mask], yp, y2):
        ax.annotate(mn, (xp_v, yr_v), xytext=(3,3), textcoords='offset points', fontsize=6, alpha=0.7)
    vv = np.concatenate([y2, yp]); m = (vv.max()-vv.min())*0.12
    ax.plot([vv.min()-m, vv.max()+m], [vv.min()-m, vv.max()+m], 'r--', lw=1.5, alpha=0.8)
    mae_lin = np.mean(np.abs(y2 - yp))
    ax.set_title(f"Linéaire: {dc}\nMAE $= {mae_lin:.1f}$ kJ/mol, $R^2={r['R2']:.3f}$", fontsize=9, fontweight='bold')
    ax.set_xlabel(f'{label} prédit (kJ/mol)', fontsize=8); ax.set_ylabel(f'{label} réel (kJ/mol)', fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)

    # Panel 2: Best quadratic
    ax = axes[0, 1]
    if best["quadratic"]:
        r = best["quadratic"][0]
        dc = r["base_desc"]
        x = df[dc].values.astype(float)
        y = df[target_col].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x2, y2 = x[mask], y[mask]
        yp = np.polyval(r["coeffs"], x2)
        ax.scatter(yp, y2, s=70, alpha=0.7, color='darkorange', edgecolors='white', linewidths=0.5, zorder=3)
        for mn, xp_v, yr_v in zip(df["molecule_name"].values[mask], yp, y2):
            ax.annotate(mn, (xp_v, yr_v), xytext=(3,3), textcoords='offset points', fontsize=6, alpha=0.7)
        vv = np.concatenate([y2, yp]); m = (vv.max()-vv.min())*0.12
        ax.plot([vv.min()-m, vv.max()+m], [vv.min()-m, vv.max()+m], 'r--', lw=1.5, alpha=0.8)
        mae_quad = np.mean(np.abs(y2 - yp))
        ax.set_title(f"Quadratique: {dc}\nMAE $= {mae_quad:.1f}$ kJ/mol, $R^2={r['R2']:.3f}$", fontsize=9, fontweight='bold')
    ax.set_xlabel(f'{label} prédit (kJ/mol)', fontsize=8); ax.set_ylabel(f'{label} réel (kJ/mol)', fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)

    # Panel 3: Best 2-variable
    ax = axes[1, 0]
    mv2 = [r for r in best["multivariate"] if r["n_vars"] == 2]
    if mv2:
        r = mv2[0]
        cols = r["cols"]
        X = df[cols].values.astype(float)
        y = df[target_col].values.astype(float)
        mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        X2, y2 = X[mask], y[mask]
        yp = X2 @ np.array(r["coefs"]) + r["intercept"]
        ax.scatter(yp, y2, s=70, alpha=0.7, color='forestgreen', edgecolors='white', linewidths=0.5, zorder=3)
        for mn, xp_v, yr_v in zip(df["molecule_name"].values[mask], yp, y2):
            ax.annotate(mn, (xp_v, yr_v), xytext=(3,3), textcoords='offset points', fontsize=6, alpha=0.7)
        vv = np.concatenate([y2, yp]); m = (vv.max()-vv.min())*0.12
        ax.plot([vv.min()-m, vv.max()+m], [vv.min()-m, vv.max()+m], 'r--', lw=1.5, alpha=0.8)
        desc_short = "\n+ ".join(cols)
        mae_2v = np.mean(np.abs(y2 - yp))
        ax.set_title(f"2-var: {desc_short}\nMAE $= {mae_2v:.1f}$ kJ/mol, $R^2={r['R2']:.3f}$", fontsize=8, fontweight='bold')
    ax.set_xlabel(f'{label} prédit (kJ/mol)', fontsize=8); ax.set_ylabel(f'{label} réel (kJ/mol)', fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)

    # Panel 4: Best 3-variable
    ax = axes[1, 1]
    mv3 = [r for r in best["multivariate"] if r["n_vars"] == 3]
    if mv3:
        r = mv3[0]
        cols = r["cols"]
        X = df[cols].values.astype(float)
        y = df[target_col].values.astype(float)
        mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        X2, y2 = X[mask], y[mask]
        yp = X2 @ np.array(r["coefs"]) + r["intercept"]
        ax.scatter(yp, y2, s=70, alpha=0.7, color='darkviolet', edgecolors='white', linewidths=0.5, zorder=3)
        for mn, xp_v, yr_v in zip(df["molecule_name"].values[mask], yp, y2):
            ax.annotate(mn, (xp_v, yr_v), xytext=(3,3), textcoords='offset points', fontsize=6, alpha=0.7)
        vv = np.concatenate([y2, yp]); m = (vv.max()-vv.min())*0.12
        ax.plot([vv.min()-m, vv.max()+m], [vv.min()-m, vv.max()+m], 'r--', lw=1.5, alpha=0.8)
        desc_short = "\n+ ".join(cols)
        mae_3v = np.mean(np.abs(y2 - yp))
        ax.set_title(f"3-var: {desc_short}\nMAE $= {mae_3v:.1f}$ kJ/mol, $R^2={r['R2']:.3f}$", fontsize=8, fontweight='bold')
    ax.set_xlabel(f'{label} prédit (kJ/mol)', fontsize=8); ax.set_ylabel(f'{label} réel (kJ/mol)', fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Meilleurs modèles pour {label}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {filename}")


for target_col, label in tasks:
    plot_best_4panel(df, best_results[label], target_col, label,
                     OUT_DIR / f"bruteforce_v2_best_{label}.png")

print("\nDone.")
