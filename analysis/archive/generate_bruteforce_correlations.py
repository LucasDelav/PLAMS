#!/usr/bin/env python3
"""
Brute-force correlation search:
  - High-frequency descriptors (>2000 cm⁻¹) → ΔΔU (red/ox)
  - Low-frequency descriptors  (<500  cm⁻¹) → TΔS (red/ox)

Reads raw frequency data from raw_freq_data.json (extracted by extract_raw_freq_data.py)
and targets from the CSV.  Computes ~100 descriptors per range, tests all correlations,
and produces ranked tables + best pred-vs-real plots.
"""

import json
import pathlib
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_JSON  = _PROJECT_ROOT / "data" / "raw_freq_data.json"
CSV_PATH  = _PROJECT_ROOT / "results" / "full_decomposition" / "molecules_summary_full_decomposition.csv"
OUT_DIR   = _PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Frequency ranges ────────────────────────────────────────────────────
RANGES_HIGH = {
    "high":       (2000, 5000),
    "high_2000_2500": (2000, 2500),
    "high_2500_3000": (2500, 3000),
    "high_3000_3500": (3000, 3500),
    "high_3500_4000": (3500, 4000),
}
RANGES_LOW = {
    "low":        (0, 500),
    "low_0_100":  (0, 100),
    "low_100_200":(100, 200),
    "low_200_300":(200, 300),
    "low_300_400":(300, 400),
    "low_400_500":(400, 500),
    "low_0_250":  (0, 250),
    "low_250_500":(250, 500),
}

R_KJ = 8.314462618e-3  # kJ/(mol·K)
T = 298.15


# ═══════════════════════════════════════════════════════════════════════════
#  Descriptor computation
# ═══════════════════════════════════════════════════════════════════════════

def safe_gmean(a):
    a = a[a > 0]
    if len(a) == 0:
        return np.nan
    return np.exp(np.mean(np.log(a)))

def safe_hmean(a):
    a = a[a > 0]
    if len(a) == 0:
        return np.nan
    return len(a) / np.sum(1.0 / a)

def spectral_entropy(intensities):
    """Shannon entropy of intensity distribution."""
    p = intensities / np.sum(intensities) if np.sum(intensities) > 0 else intensities
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return -np.sum(p * np.log(p))

def compute_range_descriptors(freqs, ir, lo, hi, prefix):
    """Compute a battery of descriptors for frequencies in [lo, hi)."""
    mask = (freqs >= lo) & (freqs < hi)
    f = freqs[mask]
    n = len(f)
    d = {}
    d[f"{prefix}_n"] = n

    if n == 0:
        # Return NaN for everything
        for k in ["mean", "std", "var", "cv", "min", "max", "range",
                   "median", "q10", "q25", "q75", "q90", "iqr",
                   "skew", "kurtosis", "gmean", "hmean", "sum",
                   "rms", "moment2", "moment3", "moment4"]:
            d[f"{prefix}_freq_{k}"] = np.nan
        for k in ["ir_sum", "ir_mean", "ir_std", "ir_max", "ir_min",
                   "ir_median", "ir_range", "ir_cv",
                   "ir_weighted_freq", "ir_energy_sum", "ir_entropy",
                   "ir_top1", "ir_top2", "ir_top3"]:
            d[f"{prefix}_{k}"] = np.nan
        return d

    # ── Frequency statistics ──
    d[f"{prefix}_freq_mean"] = np.mean(f)
    d[f"{prefix}_freq_std"]  = np.std(f, ddof=1) if n > 1 else 0.0
    d[f"{prefix}_freq_var"]  = np.var(f, ddof=1) if n > 1 else 0.0
    d[f"{prefix}_freq_cv"]   = d[f"{prefix}_freq_std"] / d[f"{prefix}_freq_mean"] if d[f"{prefix}_freq_mean"] != 0 else np.nan
    d[f"{prefix}_freq_min"]  = np.min(f)
    d[f"{prefix}_freq_max"]  = np.max(f)
    d[f"{prefix}_freq_range"]= np.ptp(f)
    d[f"{prefix}_freq_median"]= np.median(f)
    d[f"{prefix}_freq_q10"]  = np.percentile(f, 10) if n >= 2 else f[0]
    d[f"{prefix}_freq_q25"]  = np.percentile(f, 25) if n >= 2 else f[0]
    d[f"{prefix}_freq_q75"]  = np.percentile(f, 75) if n >= 2 else f[0]
    d[f"{prefix}_freq_q90"]  = np.percentile(f, 90) if n >= 2 else f[0]
    d[f"{prefix}_freq_iqr"]  = d[f"{prefix}_freq_q75"] - d[f"{prefix}_freq_q25"]
    d[f"{prefix}_freq_skew"] = sp_stats.skew(f) if n >= 3 else 0.0
    d[f"{prefix}_freq_kurtosis"] = sp_stats.kurtosis(f) if n >= 4 else 0.0
    d[f"{prefix}_freq_gmean"]= safe_gmean(f)
    d[f"{prefix}_freq_hmean"]= safe_hmean(f)
    d[f"{prefix}_freq_sum"]  = np.sum(f)
    d[f"{prefix}_freq_rms"]  = np.sqrt(np.mean(f**2))
    # Central moments
    mu = np.mean(f)
    d[f"{prefix}_freq_moment2"] = np.mean((f - mu)**2)
    d[f"{prefix}_freq_moment3"] = np.mean((f - mu)**3)
    d[f"{prefix}_freq_moment4"] = np.mean((f - mu)**4)

    # ── IR statistics ──
    if ir is not None:
        ir_sub = ir[mask]
        d[f"{prefix}_ir_sum"]    = np.sum(ir_sub)
        d[f"{prefix}_ir_mean"]   = np.mean(ir_sub)
        d[f"{prefix}_ir_std"]    = np.std(ir_sub, ddof=1) if n > 1 else 0.0
        d[f"{prefix}_ir_max"]    = np.max(ir_sub)
        d[f"{prefix}_ir_min"]    = np.min(ir_sub)
        d[f"{prefix}_ir_median"] = np.median(ir_sub)
        d[f"{prefix}_ir_range"]  = np.ptp(ir_sub)
        d[f"{prefix}_ir_cv"]     = d[f"{prefix}_ir_std"] / d[f"{prefix}_ir_mean"] if d[f"{prefix}_ir_mean"] > 0 else np.nan
        # IR-weighted frequency
        ir_tot = np.sum(ir_sub)
        d[f"{prefix}_ir_weighted_freq"] = np.sum(f * ir_sub) / ir_tot if ir_tot > 0 else np.nan
        d[f"{prefix}_ir_energy_sum"]    = np.sum(f * ir_sub)
        d[f"{prefix}_ir_entropy"]       = spectral_entropy(ir_sub)
        sorted_ir = np.sort(ir_sub)[::-1]
        d[f"{prefix}_ir_top1"] = sorted_ir[0] if len(sorted_ir) >= 1 else np.nan
        d[f"{prefix}_ir_top2"] = sorted_ir[1] if len(sorted_ir) >= 2 else np.nan
        d[f"{prefix}_ir_top3"] = sorted_ir[2] if len(sorted_ir) >= 3 else np.nan
    else:
        for k in ["ir_sum", "ir_mean", "ir_std", "ir_max", "ir_min",
                   "ir_median", "ir_range", "ir_cv",
                   "ir_weighted_freq", "ir_energy_sum", "ir_entropy",
                   "ir_top1", "ir_top2", "ir_top3"]:
            d[f"{prefix}_{k}"] = np.nan

    return d


def compute_cross_range_descriptors(freqs, ir, n_total):
    """Descriptors that combine information across ranges."""
    d = {}
    mask_low  = freqs < 500
    mask_high = freqs >= 2000
    mask_mid  = (freqs >= 500) & (freqs < 2000)

    n_low  = np.sum(mask_low)
    n_high = np.sum(mask_high)
    n_mid  = np.sum(mask_mid)

    # Fractions
    d["frac_low"]  = n_low  / n_total if n_total > 0 else 0
    d["frac_high"] = n_high / n_total if n_total > 0 else 0
    d["frac_mid"]  = n_mid  / n_total if n_total > 0 else 0

    # Ratio high/low
    d["ratio_high_low"] = n_high / n_low if n_low > 0 else np.nan
    d["ratio_low_mid"]  = n_low  / n_mid if n_mid > 0 else np.nan

    # Mean frequency ratios
    f_low  = freqs[mask_low]
    f_high = freqs[mask_high]
    mean_low  = np.mean(f_low)  if len(f_low) > 0  else np.nan
    mean_high = np.mean(f_high) if len(f_high) > 0 else np.nan
    d["mean_ratio_high_low"] = mean_high / mean_low if (mean_low and mean_low > 0) else np.nan

    # Frequency gap between highest-low and lowest-high
    if len(f_low) > 0 and len(f_high) > 0:
        d["gap_low_high"] = np.min(f_high) - np.max(f_low)
    else:
        d["gap_low_high"] = np.nan

    # IR distribution across ranges
    if ir is not None:
        ir_tot = np.sum(ir) if np.sum(ir) > 0 else 1
        d["ir_frac_low"]  = np.sum(ir[mask_low])  / ir_tot
        d["ir_frac_high"] = np.sum(ir[mask_high]) / ir_tot
        d["ir_frac_mid"]  = np.sum(ir[mask_mid])  / ir_tot
        d["ir_ratio_high_low"] = d["ir_frac_high"] / d["ir_frac_low"] if d["ir_frac_low"] > 0 else np.nan
    else:
        d["ir_frac_low"] = d["ir_frac_high"] = d["ir_frac_mid"] = d["ir_ratio_high_low"] = np.nan

    # ZPE contribution by range (sum of freq / 2)
    CM_TO_KJ = 4.5563e-6 * 2625.499638  # cm⁻¹ → kJ/mol (via Hartree)
    d["zpe_low_kj"]  = np.sum(f_low)  * 0.5 * CM_TO_KJ if len(f_low) > 0 else 0
    d["zpe_high_kj"] = np.sum(f_high) * 0.5 * CM_TO_KJ if len(f_high) > 0 else 0
    d["zpe_frac_high"] = d["zpe_high_kj"] / (d["zpe_low_kj"] + d["zpe_high_kj"]) if (d["zpe_low_kj"] + d["zpe_high_kj"]) > 0 else np.nan

    # Thermal population descriptors (how much each mode is thermally populated at 298K)
    kT_cm = T * 0.695039  # kT in cm⁻¹
    # For low-freq modes: Bose-Einstein population
    if len(f_low) > 0:
        x_low = f_low / kT_cm
        # Avoid overflow
        x_low = np.clip(x_low, 0, 500)
        n_be_low = 1.0 / (np.exp(x_low) - 1.0 + 1e-30)
        d["thermal_pop_low_sum"]  = np.sum(n_be_low)
        d["thermal_pop_low_mean"] = np.mean(n_be_low)
        d["thermal_pop_low_max"]  = np.max(n_be_low)
        # Thermal energy contribution: E_vib = hcν × (n + 1/2)
        d["thermal_energy_low"]   = np.sum(f_low * (n_be_low + 0.5)) * CM_TO_KJ
        # Vibrational entropy contribution: S = kB × [x/(exp(x)-1) - ln(1-exp(-x))]
        s_low = x_low / (np.exp(x_low) - 1 + 1e-30) - np.log(1 - np.exp(-x_low) + 1e-30)
        d["vib_entropy_low"] = R_KJ * np.sum(s_low)
    else:
        d["thermal_pop_low_sum"] = d["thermal_pop_low_mean"] = d["thermal_pop_low_max"] = 0
        d["thermal_energy_low"] = d["vib_entropy_low"] = 0

    if len(f_high) > 0:
        x_high = f_high / kT_cm
        x_high = np.clip(x_high, 0, 500)
        n_be_high = 1.0 / (np.exp(x_high) - 1.0 + 1e-30)
        d["thermal_pop_high_sum"]  = np.sum(n_be_high)
        d["thermal_pop_high_mean"] = np.mean(n_be_high)
        # Vibrational entropy high
        s_high = x_high / (np.exp(x_high) - 1 + 1e-30) - np.log(1 - np.exp(-x_high) + 1e-30)
        d["vib_entropy_high"] = R_KJ * np.sum(s_high)
        d["thermal_energy_high"] = np.sum(f_high * (n_be_high + 0.5)) * CM_TO_KJ
    else:
        d["thermal_pop_high_sum"] = d["thermal_pop_high_mean"] = 0
        d["vib_entropy_high"] = d["thermal_energy_high"] = 0

    # Ratio of thermal entropies
    d["vib_entropy_ratio_low_high"] = d["vib_entropy_low"] / d["vib_entropy_high"] if d["vib_entropy_high"] > 0 else np.nan
    d["vib_entropy_total"] = d["vib_entropy_low"] + d["vib_entropy_high"]

    return d


def compute_all_descriptors(freqs, ir):
    """Compute all descriptors for one conformer's spectrum."""
    freqs = np.array(freqs, dtype=float)
    ir = np.array(ir, dtype=float) if ir is not None else None
    n_total = len(freqs)

    d = {"n_modes_total": n_total}

    # High-frequency range descriptors (for ΔΔU)
    for name, (lo, hi) in RANGES_HIGH.items():
        d.update(compute_range_descriptors(freqs, ir, lo, hi, name))

    # Low-frequency range descriptors (for TΔS)
    for name, (lo, hi) in RANGES_LOW.items():
        d.update(compute_range_descriptors(freqs, ir, lo, hi, name))

    # Cross-range descriptors
    d.update(compute_cross_range_descriptors(freqs, ir, n_total))

    return d


def boltzmann_average(conformer_descriptors, weights):
    """
    Compute Boltzmann-weighted average of descriptors.
    conformer_descriptors: dict {conf_num: {desc_name: value}}
    weights: dict {conf_num: weight}
    """
    # Normalize weights to the conformers we have
    common = set(conformer_descriptors.keys()) & set(weights.keys())
    if not common:
        # No matching weights: equal weights
        common = set(conformer_descriptors.keys())
        w = {c: 1.0 / len(common) for c in common}
    else:
        total = sum(weights[c] for c in common)
        w = {c: weights[c] / total for c in common}

    # Average
    all_keys = set()
    for c in common:
        all_keys.update(conformer_descriptors[c].keys())

    avg = {}
    for key in all_keys:
        vals = []
        ws = []
        for c in common:
            v = conformer_descriptors[c].get(key, np.nan)
            if not np.isnan(v):
                vals.append(v)
                ws.append(w[c])
        if vals:
            ws_arr = np.array(ws)
            ws_arr /= ws_arr.sum()
            avg[key] = float(np.sum(np.array(vals) * ws_arr))
        else:
            avg[key] = np.nan

    return avg


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

print("Loading raw data...")
raw = json.loads(RAW_JSON.read_text())
csv = pd.read_csv(CSV_PATH)

# Compute descriptors per molecule (Boltzmann-averaged)
rows = []
for mol_name in csv["molecule_name"]:
    if mol_name not in raw:
        print(f"  SKIP {mol_name}: not in raw data")
        continue

    mol = raw[mol_name]
    weights = {int(k): v for k, v in mol["boltzmann_weights"].items()}

    conf_descs = {}
    for cn_str, cdata in mol["conformers"].items():
        cn = int(cn_str)
        conf_descs[cn] = compute_all_descriptors(cdata["frequencies"], cdata["ir_intensities"])

    avg = boltzmann_average(conf_descs, weights)
    avg["molecule_name"] = mol_name
    rows.append(avg)

df_desc = pd.DataFrame(rows)

# Merge with targets
targets_cols = ["molecule_name", "ddU_red_kJ", "ddU_ox_kJ", "TdS_red_kJ", "TdS_ox_kJ"]
df = df_desc.merge(csv[targets_cols], on="molecule_name", how="inner")
print(f"Molecules with complete data: {len(df)}")

# Get all descriptor columns
desc_cols = [c for c in df_desc.columns if c != "molecule_name"]
print(f"Total descriptors computed: {len(desc_cols)}")


# ═══════════════════════════════════════════════════════════════════════════
#  Correlation analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_correlations(df, desc_cols, target_col, target_label):
    """Test all descriptors against one target. Return sorted results."""
    results = []
    for dc in desc_cols:
        x = df[dc].values
        y = df[target_col].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            continue
        x2, y2 = x[mask], y[mask]
        # Skip if no variance
        if np.std(x2) < 1e-12 or np.std(y2) < 1e-12:
            continue
        slope, intercept, r, p, se = sp_stats.linregress(x2, y2)
        y_pred = slope * x2 + intercept
        mae = np.mean(np.abs(y2 - y_pred))
        results.append({
            "descriptor": dc,
            "R2": r**2,
            "r": r,
            "p_value": p,
            "slope": slope,
            "intercept": intercept,
            "MAE": mae,
            "n": int(mask.sum()),
        })
    results.sort(key=lambda x: -x["R2"])
    return results


# Define the 4 correlation tasks
tasks = [
    ("ddU_red_kJ",  r"$\Delta\Delta U_{\mathrm{red}}$", "high-freq", "ΔΔU_red"),
    ("ddU_ox_kJ",   r"$\Delta\Delta U_{\mathrm{ox}}$",  "high-freq", "ΔΔU_ox"),
    ("TdS_red_kJ",  r"$T\Delta S_{\mathrm{red}}$",      "low-freq",  "TΔS_red"),
    ("TdS_ox_kJ",   r"$T\Delta S_{\mathrm{ox}}$",       "low-freq",  "TΔS_ox"),
]

all_results = {}
for target_col, target_label, freq_focus, short_label in tasks:
    # For each target, try ALL descriptors (brute force)
    corrs = run_correlations(df, desc_cols, target_col, target_label)
    all_results[short_label] = corrs

    print(f"\n{'='*70}")
    print(f"  TOP 20 descriptors for {short_label}  (all {len(desc_cols)} tested)")
    print(f"{'='*70}")
    for i, r in enumerate(corrs[:20]):
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        print(f"  {i+1:2d}. {r['descriptor']:45s}  R²={r['R2']:.3f}  p={r['p_value']:.2e}  MAE={r['MAE']:.2f} {sig}")


# ═══════════════════════════════════════════════════════════════════════════
#  Save full correlation table
# ═══════════════════════════════════════════════════════════════════════════

all_corr_rows = []
for short_label, corrs in all_results.items():
    for r in corrs:
        row = dict(r)
        row["target"] = short_label
        all_corr_rows.append(row)
corr_df = pd.DataFrame(all_corr_rows)
corr_csv = OUT_DIR / "bruteforce_correlations.csv"
corr_df.to_csv(corr_csv, index=False)
print(f"\nFull correlation table: {corr_csv}")


# ═══════════════════════════════════════════════════════════════════════════
#  Plots: top 6 pred-vs-real for each target (2×2 figure layout)
# ═══════════════════════════════════════════════════════════════════════════

def plot_pred_vs_real_grid(df, corrs, target_col, target_label, short_label, n=6, filename=None):
    """Plot top-n pred-vs-real panels."""
    top = corrs[:n]
    if not top:
        return

    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, res in enumerate(top):
        row_i, col_i = idx // n_cols, idx % n_cols
        ax = axes[row_i, col_i]

        dc = res["descriptor"]
        x = df[dc].values
        y = df[target_col].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x2, y2 = x[mask], y[mask]
        y_pred = res["slope"] * x2 + res["intercept"]
        mol_names = df["molecule_name"].values[mask]

        ax.scatter(y_pred, y2, s=70, alpha=0.7,
                   color='steelblue', edgecolors='white', linewidths=0.5, zorder=3)
        for mn, xp, yr in zip(mol_names, y_pred, y2):
            ax.annotate(mn, (xp, yr), xytext=(3, 3),
                        textcoords='offset points', fontsize=6, alpha=0.7)

        all_vals = np.concatenate([y2, y_pred])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.12
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'r--', linewidth=1.5, alpha=0.8)

        sign = "+" if res["intercept"] >= 0 else "\u2212"
        ax.set_title(f"{dc}\n$R^2={res['R2']:.3f}$, $p={res['p_value']:.1e}$",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel(f'{short_label} prédit (kJ/mol)', fontsize=8)
        ax.set_ylabel(f'{short_label} réel (kJ/mol)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, zorder=0)

    # Hide unused panels
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(f'Top {n} corrélations brute-force → {target_label}',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    if filename is None:
        filename = OUT_DIR / f"bruteforce_top{n}_{short_label}.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {filename}")


for target_col, target_label, _, short_label in tasks:
    plot_pred_vs_real_grid(df, all_results[short_label],
                           target_col, target_label, short_label, n=6)


# ═══════════════════════════════════════════════════════════════════════════
#  Summary heatmap: top descriptors across all 4 targets
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_heatmap(all_results, filename=None):
    """Heatmap of R² for top descriptors across targets."""
    # Collect top-10 unique descriptors per target
    top_descs = set()
    for short_label, corrs in all_results.items():
        for r in corrs[:10]:
            top_descs.add(r["descriptor"])
    top_descs = sorted(top_descs)

    target_labels = list(all_results.keys())
    matrix = np.full((len(top_descs), len(target_labels)), np.nan)
    for j, tl in enumerate(target_labels):
        lookup = {r["descriptor"]: r["R2"] for r in all_results[tl]}
        for i, desc in enumerate(top_descs):
            matrix[i, j] = lookup.get(desc, np.nan)

    fig, ax = plt.subplots(figsize=(8, max(6, len(top_descs) * 0.35)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(len(target_labels)))
    ax.set_xticklabels(target_labels, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(top_descs)))
    ax.set_yticklabels(top_descs, fontsize=7)

    for i in range(len(top_descs)):
        for j in range(len(target_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='$R^2$', shrink=0.8)
    ax.set_title("Brute-force : $R^2$ des meilleurs descripteurs", fontsize=11, fontweight='bold')
    plt.tight_layout()
    if filename is None:
        filename = OUT_DIR / "bruteforce_heatmap.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap: {filename}")

plot_summary_heatmap(all_results)

# Save descriptors CSV for reference
desc_csv = OUT_DIR / "bruteforce_descriptors.csv"
df.to_csv(desc_csv, index=False)
print(f"  Descriptors: {desc_csv}")

print("\nDone.")
