#!/usr/bin/env python3
"""
Exhaustive search for Fukui/CDFT-based corrections to LVC2 E_relax predictions.
Baseline to beat:
  - LVC2 brut calibre: R²=0.744 MAE=1.3 (ox), R²=0.843 MAE=1.7 (red)
  - LVC2 + LUMO corr calibre: R²=0.711 MAE=1.5 (ox), R²=0.894 MAE=1.4 (red)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA LOADING
# ============================================================

BASE = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5"

# LVC summary
lvc = pd.read_csv(f"{BASE}/rapport/data/lvc_summary.csv")
lvc = lvc.rename(columns={"molecule_name": "molecule"})

# Descriptors per molecule (for LUMO_O_frac baseline)
desc_mol = pd.read_csv(f"{BASE}/rapport/data/descriptors_per_molecule.csv")

# predict_dg conformers - only load needed columns
# First get all column names
all_cols = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", nrows=0).columns.tolist()

# Select columns we need
keep_cols = ["molecule", "conformer", "weight"]
# Summed Fukui
for prefix in ["fplus_sum", "fminus_sum", "f0_sum", "dual_sum"]:
    for elem in ["O", "C", "H"]:
        col = f"{prefix}{elem}_n"
        if col in all_cols:
            keep_cols.append(col)

# Global CDFT
for col in ["mu_n", "eta_n", "omega_n", "HOMO_n", "LUMO_n", "gap_n"]:
    if col in all_cols:
        keep_cols.append(col)

# Per-atom columns for min/max computations
for prefix in ["fplus_", "fminus_", "dual_", "f0_"]:
    for col in all_cols:
        if col.startswith(prefix) and col.endswith("_n") and "sum" not in col:
            keep_cols.append(col)

keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate preserving order

pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", usecols=keep_cols)

# Exclude DEA molecules
dea = {"Br-DA", "Br-HMC", "I-DA", "I-HMC"}
lvc = lvc[~lvc["molecule"].isin(dea)].copy()
pdg = pdg[~pdg["molecule"].isin(dea)].copy()

# ============================================================
# 2. BOLTZMANN-WEIGHTED AVERAGES
# ============================================================

# Identify per-atom columns by element
per_atom_O = [c for c in pdg.columns if "_O" in c and c.endswith("_n") and "sum" not in c]
per_atom_C = [c for c in pdg.columns if "_C" in c and c.endswith("_n") and "sum" not in c]
per_atom_H = [c for c in pdg.columns if "_H" in c and c.endswith("_n") and "sum" not in c]

# Separate per-atom fplus/fminus/dual/f0 for O
fplus_O_atoms = [c for c in per_atom_O if c.startswith("fplus_")]
fminus_O_atoms = [c for c in per_atom_O if c.startswith("fminus_")]
dual_O_atoms = [c for c in per_atom_O if c.startswith("dual_")]
dual_C_atoms = [c for c in per_atom_C if c.startswith("dual_")]
dual_H_atoms = [c for c in per_atom_H if c.startswith("dual_")]
dual_all_atoms = dual_O_atoms + dual_C_atoms + dual_H_atoms
fplus_all_atoms = fplus_O_atoms + [c for c in per_atom_C if c.startswith("fplus_")] + [c for c in per_atom_H if c.startswith("fplus_")]

# Numeric columns for Boltzmann averaging
numeric_cols = [c for c in pdg.columns if c not in ["molecule", "conformer", "weight"]]


def boltzmann_avg(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}
    for col in numeric_cols:
        vals = group[col].values
        if np.all(np.isfinite(vals)):
            result[col] = np.sum(w * vals)
        else:
            result[col] = np.nan
    return pd.Series(result)


bw = pdg.groupby("molecule").apply(boltzmann_avg).reset_index()

# Compute derived descriptors per conformer, then Boltzmann average
# Actually, for min/max we need per-conformer then average
# Let's compute min/max of per-atom values per conformer, then Boltzmann-average those

def compute_derived(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}

    # fplus_maxO: max fplus among O atoms per conformer, then Boltzmann avg
    for prefix, atoms, label in [
        ("fplus_", fplus_O_atoms, "fplus_maxO"),
        ("fminus_", fminus_O_atoms, "fminus_maxO"),
    ]:
        if atoms:
            per_conf_max = group[atoms].max(axis=1).values
            result[label] = np.sum(w * per_conf_max)

    # dual_min, dual_max over ALL atoms
    if dual_all_atoms:
        per_conf_min = group[dual_all_atoms].min(axis=1).values
        per_conf_max = group[dual_all_atoms].max(axis=1).values
        result["dual_min"] = np.sum(w * per_conf_min)
        result["dual_max"] = np.sum(w * per_conf_max)

    # dual_minO, dual_maxO
    if dual_O_atoms:
        per_conf_min = group[dual_O_atoms].min(axis=1).values
        per_conf_max = group[dual_O_atoms].max(axis=1).values
        result["dual_minO"] = np.sum(w * per_conf_min)
        result["dual_maxO"] = np.sum(w * per_conf_max)

    # fplus_maxC
    fplus_C_atoms_list = [c for c in per_atom_C if c.startswith("fplus_")]
    if fplus_C_atoms_list:
        per_conf_max = group[fplus_C_atoms_list].max(axis=1).values
        result["fplus_maxC"] = np.sum(w * per_conf_max)

    # Spread: max - min of fplus_O
    if fplus_O_atoms:
        per_conf_spread = group[fplus_O_atoms].max(axis=1).values - group[fplus_O_atoms].min(axis=1).values
        result["fplus_spreadO"] = np.sum(w * per_conf_spread)

    return pd.Series(result)


derived = pdg.groupby("molecule").apply(compute_derived).reset_index()

# Merge
data = lvc.merge(bw, on="molecule", how="inner")
data = data.merge(derived, on="molecule", how="left")

# Also merge LUMO_O_frac from descriptors_per_molecule
desc_mol_sub = desc_mol[["molecule", "LUMO_O_frac"]].copy()
data = data.merge(desc_mol_sub, on="molecule", how="left")

print(f"Molecules in dataset: {len(data)}")
print(f"Molecules: {sorted(data['molecule'].tolist())}")
print()

# ============================================================
# 3. RATIO DESCRIPTORS
# ============================================================

eps = 1e-10
data["fplus_ratio_OC"] = data["fplus_sumO_n"] / (data["fplus_sumC_n"] + eps)
data["dual_ratio_OC"] = data["dual_sumO_n"] / (data["dual_sumC_n"] + eps)
data["fplus_frac_O"] = data["fplus_sumO_n"] / (data["fplus_sumO_n"] + data["fplus_sumC_n"] + eps)
data["fminus_ratio_OC"] = data["fminus_sumO_n"] / (data["fminus_sumC_n"] + eps)
data["fminus_frac_O"] = data["fminus_sumO_n"] / (data["fminus_sumO_n"] + data["fminus_sumC_n"] + eps)

# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def evaluate_correction(E_corr, E_real, label=""):
    """Calibrate E_corr -> E_real via linear regression, return metrics."""
    mask = np.isfinite(E_corr) & np.isfinite(E_real)
    if mask.sum() < 4:
        return None
    X = E_corr[mask].values.reshape(-1, 1)
    y = E_real[mask].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return {"label": label, "R2": r2, "MAE": mae, "n": int(mask.sum()),
            "slope": reg.coef_[0], "intercept": reg.intercept_}


# ============================================================
# 5. DESCRIPTOR LIST
# ============================================================

# Basic descriptors available as columns
basic_descriptors = {
    "fplus_sumO": "fplus_sumO_n",
    "fminus_sumO": "fminus_sumO_n",
    "dual_sumO": "dual_sumO_n",
    "f0_sumO": "f0_sumO_n",
    "fplus_sumC": "fplus_sumC_n",
    "fminus_sumC": "fminus_sumC_n",
    "dual_sumC": "dual_sumC_n",
    "fplus_sumH": "fplus_sumH_n",
    "fminus_sumH": "fminus_sumH_n",
    "dual_sumH": "dual_sumH_n",
    "mu": "mu_n",
    "eta": "eta_n",
    "omega": "omega_n",
    "HOMO": "HOMO_n",
    "LUMO": "LUMO_n",
    "gap": "gap_n",
    "fplus_maxO": "fplus_maxO",
    "fminus_maxO": "fminus_maxO",
    "dual_min": "dual_min",
    "dual_max": "dual_max",
    "dual_minO": "dual_minO",
    "dual_maxO": "dual_maxO",
    "fplus_maxC": "fplus_maxC",
    "fplus_spreadO": "fplus_spreadO",
    "LUMO_O_frac": "LUMO_O_frac",
    "fplus_ratio_OC": "fplus_ratio_OC",
    "dual_ratio_OC": "dual_ratio_OC",
    "fplus_frac_O": "fplus_frac_O",
    "fminus_ratio_OC": "fminus_ratio_OC",
    "fminus_frac_O": "fminus_frac_O",
}

# Check which exist
available = {}
for name, col in basic_descriptors.items():
    if col in data.columns and data[col].notna().sum() > 5:
        available[name] = data[col].values
    else:
        print(f"  WARNING: {name} ({col}) not available or too many NaNs")

print(f"\nAvailable descriptors: {len(available)}")

# ============================================================
# 6. RUN ALL CORRECTIONS
# ============================================================

results_red = []
results_ox = []

lvc2_red = data["lvc2_red"].values
lvc2_ox = data["lvc2_ox"].values
real_red = data["real_red"].values
real_ox = data["real_ox"].values

# --- Baseline: LVC2 brut calibre ---
r = evaluate_correction(pd.Series(lvc2_red), pd.Series(real_red), "LVC2_brut")
if r: results_red.append({**r, "n_params": 0, "category": "baseline"})
r = evaluate_correction(pd.Series(lvc2_ox), pd.Series(real_ox), "LVC2_brut")
if r: results_ox.append({**r, "n_params": 0, "category": "baseline"})

# --- LUMO_O_frac baseline (n=15 originally, but we have n=13 after merge) ---
if "LUMO_O_frac" in available:
    d = available["LUMO_O_frac"]
    corr_red = lvc2_red * (1 + d**2)
    corr_ox = lvc2_ox * (1 + d**2)
    r = evaluate_correction(pd.Series(corr_red), pd.Series(real_red), "LVC2*(1+LUMO_O²) [baseline]")
    if r: results_red.append({**r, "n_params": 0, "category": "baseline"})
    r = evaluate_correction(pd.Series(corr_ox), pd.Series(real_ox), "LVC2*(1+LUMO_O²) [baseline]")
    if r: results_ox.append({**r, "n_params": 0, "category": "baseline"})


def add_result(label, corr_red_vals, corr_ox_vals, n_params, category):
    r = evaluate_correction(pd.Series(corr_red_vals), pd.Series(real_red), label)
    if r:
        results_red.append({**r, "n_params": n_params, "category": category})
    r = evaluate_correction(pd.Series(corr_ox_vals), pd.Series(real_ox), label)
    if r:
        results_ox.append({**r, "n_params": n_params, "category": category})


# ============================================================
# 6a. MULTIPLICATIVE (parameter-free)
# ============================================================

print("\n--- Running multiplicative (parameter-free) corrections ---")

for dname, dvals in available.items():
    # LVC2 * (1 + d)
    add_result(f"LVC2*(1+{dname})", lvc2_red*(1+dvals), lvc2_ox*(1+dvals), 0, "mult_pf")
    # LVC2 * (1 + d²)
    add_result(f"LVC2*(1+{dname}²)", lvc2_red*(1+dvals**2), lvc2_ox*(1+dvals**2), 0, "mult_pf")
    # LVC2 * (1 + |d|)
    add_result(f"LVC2*(1+|{dname}|)", lvc2_red*(1+np.abs(dvals)), lvc2_ox*(1+np.abs(dvals)), 0, "mult_pf")
    # LVC2 * (1 + d³)
    add_result(f"LVC2*(1+{dname}³)", lvc2_red*(1+dvals**3), lvc2_ox*(1+dvals**3), 0, "mult_pf")
    # LVC2 * (1 + sqrt(|d|))
    add_result(f"LVC2*(1+sqrt|{dname}|)", lvc2_red*(1+np.sqrt(np.abs(dvals))), lvc2_ox*(1+np.sqrt(np.abs(dvals))), 0, "mult_pf")

# ============================================================
# 6b. MULTIPLICATIVE with power scan
# ============================================================

print("--- Running multiplicative power scan ---")

power_desc = ["fplus_sumO", "dual_sumO", "fminus_sumO", "LUMO_O_frac",
              "fplus_maxO", "fminus_maxO", "eta", "omega", "gap",
              "fplus_ratio_OC", "fplus_frac_O", "dual_ratio_OC"]
powers = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

for dname in power_desc:
    if dname not in available:
        continue
    dvals = available[dname]
    for p in powers:
        label = f"LVC2*(1+|{dname}|^{p})"
        corr = np.abs(dvals)**p
        add_result(label, lvc2_red*(1+corr), lvc2_ox*(1+corr), 0, "mult_power")

# ============================================================
# 6c. ADDITIVE (1 free parameter, fitted by least-squares)
# ============================================================

print("--- Running additive (1-parameter) corrections ---")

from scipy.optimize import minimize_scalar, minimize


def fit_1param_additive(lvc2, real, dvals, formula="add"):
    """Fit a single parameter 'a' to minimize calibrated MAE."""
    def make_corr(a):
        if formula == "add":
            return lvc2 + a * dvals
        elif formula == "add_sq":
            return lvc2 + a * dvals**2
        elif formula == "mult":
            return lvc2 * (1 + a * dvals)
        elif formula == "mult_sq":
            return lvc2 * (1 + a * dvals**2)

    def objective(a):
        corr = make_corr(a)
        mask = np.isfinite(corr)
        if mask.sum() < 4:
            return 1e10
        X = corr[mask].reshape(-1, 1)
        y = real[mask]
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        return mean_absolute_error(y, y_pred)

    # Coarse grid search + refine
    a_grid = np.linspace(-10, 10, 41)
    maes = np.array([objective(a) for a in a_grid])
    best_idx = np.argmin(maes)
    best_a = a_grid[best_idx]
    # Refine with Nelder-Mead
    res = minimize(objective, best_a, method="Nelder-Mead", options={"xatol": 1e-5, "maxiter": 100})
    if res.fun < maes[best_idx]:
        best_a = res.x[0]
    return best_a


for dname, dvals in available.items():
    for formula, flabel in [("add", "LVC2+a*"), ("add_sq", "LVC2+a*²"),
                             ("mult", "LVC2*(1+a*)"), ("mult_sq", "LVC2*(1+a*²)")]:
        # Fit for reduction
        a_red = fit_1param_additive(lvc2_red, real_red, dvals, formula)
        a_ox = fit_1param_additive(lvc2_ox, real_ox, dvals, formula)

        if formula == "add":
            corr_r = lvc2_red + a_red * dvals
            corr_o = lvc2_ox + a_ox * dvals
        elif formula == "add_sq":
            corr_r = lvc2_red + a_red * dvals**2
            corr_o = lvc2_ox + a_ox * dvals**2
        elif formula == "mult":
            corr_r = lvc2_red * (1 + a_red * dvals)
            corr_o = lvc2_ox * (1 + a_ox * dvals)
        elif formula == "mult_sq":
            corr_r = lvc2_red * (1 + a_red * dvals**2)
            corr_o = lvc2_ox * (1 + a_ox * dvals**2)

        label = f"{flabel}{dname} (a_r={a_red:.3f}, a_o={a_ox:.3f})"

        r = evaluate_correction(pd.Series(corr_r), pd.Series(real_red), label)
        if r:
            results_red.append({**r, "n_params": 1, "category": "additive_1p"})
        r = evaluate_correction(pd.Series(corr_o), pd.Series(real_ox), label)
        if r:
            results_ox.append({**r, "n_params": 1, "category": "additive_1p"})

# ============================================================
# 6d. TWO-DESCRIPTOR MODELS (2 free parameters)
# ============================================================

print("--- Running two-descriptor models ---")

# Specific physically motivated pairs
two_desc_pairs = [
    ("fplus_sumO", "eta"),
    ("fplus_sumO", "omega"),
    ("fplus_sumO", "gap"),
    ("fplus_sumO", "dual_sumC"),
    ("fplus_sumO", "LUMO"),
    ("fplus_sumO", "HOMO"),
    ("fplus_sumO", "mu"),
    ("dual_sumO", "eta"),
    ("dual_sumO", "gap"),
    ("dual_sumO", "omega"),
    ("fminus_sumO", "eta"),
    ("fminus_sumO", "gap"),
    ("fplus_maxO", "eta"),
    ("fplus_maxO", "gap"),
    ("fminus_maxO", "eta"),
    ("LUMO_O_frac", "gap"),
    ("LUMO_O_frac", "eta"),
    ("LUMO_O_frac", "omega"),
    ("LUMO_O_frac", "fplus_sumO"),
    ("LUMO_O_frac", "dual_sumO"),
    ("fplus_ratio_OC", "eta"),
    ("fplus_ratio_OC", "gap"),
    ("fplus_frac_O", "eta"),
    ("fplus_frac_O", "gap"),
    ("dual_ratio_OC", "eta"),
    ("fplus_sumO", "fminus_sumO"),
    ("fplus_sumC", "eta"),
    ("dual_sumC", "eta"),
    ("dual_min", "dual_max"),
    ("fplus_maxO", "fplus_maxC"),
]

# Also add all pairs from a smaller focused set
focused = ["fplus_sumO", "fminus_sumO", "dual_sumO", "eta", "omega", "gap",
           "LUMO", "LUMO_O_frac", "fplus_maxO", "fplus_ratio_OC", "mu"]
for i, d1 in enumerate(focused):
    for d2 in focused[i+1:]:
        if (d1, d2) not in two_desc_pairs and (d2, d1) not in two_desc_pairs:
            two_desc_pairs.append((d1, d2))


def fit_2param(lvc2, real, d1vals, d2vals, formula="mult_add"):
    """Fit 2 parameters a, b."""
    def objective(params):
        if formula == "mult_cross":
            a = params[0]
            corr = lvc2 * (1 + a * d1vals * d2vals)
        else:
            a, b = params
            if formula == "mult_add":
                corr = lvc2 * (1 + a * d1vals + b * d2vals)
            elif formula == "mult_sq":
                corr = lvc2 * (1 + a * d1vals**2 + b * d2vals**2)
            else:
                return 1e10
        mask = np.isfinite(corr)
        if mask.sum() < 5:
            return 1e10
        X = corr[mask].reshape(-1, 1)
        y = real[mask]
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        return mean_absolute_error(y, y_pred)

    if formula == "mult_cross":
        # 1-param: simple grid
        a_grid = np.linspace(-10, 10, 41)
        maes = [objective([a]) for a in a_grid]
        best_a = a_grid[np.argmin(maes)]
        res = minimize(objective, [best_a], method="Nelder-Mead", options={"xatol": 1e-5, "maxiter": 100})
        return (res.x[0], 0)
    else:
        # 2-param: coarse grid + refine
        best = (0, 0)
        best_mae = objective(best)
        a_grid = np.linspace(-5, 5, 21)
        b_grid = np.linspace(-5, 5, 21)
        for a in a_grid:
            for b in b_grid:
                m = objective((a, b))
                if m < best_mae:
                    best = (a, b)
                    best_mae = m
        res = minimize(objective, best, method="Nelder-Mead", options={"xatol": 1e-5, "maxiter": 200})
        if res.fun < best_mae:
            best = tuple(res.x)
        return best


for d1name, d2name in two_desc_pairs:
    if d1name not in available or d2name not in available:
        continue
    d1 = available[d1name]
    d2 = available[d2name]

    for formula, flabel in [("mult_add", "LVC2*(1+a*D1+b*D2)"),
                             ("mult_sq", "LVC2*(1+a*D1²+b*D2²)"),
                             ("mult_cross", "LVC2*(1+a*D1*D2)")]:
        # Fit for reduction
        ab_r = fit_2param(lvc2_red, real_red, d1, d2, formula)
        ab_o = fit_2param(lvc2_ox, real_ox, d1, d2, formula)

        n_p = 2 if formula != "mult_cross" else 1

        if formula == "mult_add":
            corr_r = lvc2_red * (1 + ab_r[0]*d1 + ab_r[1]*d2)
            corr_o = lvc2_ox * (1 + ab_o[0]*d1 + ab_o[1]*d2)
        elif formula == "mult_sq":
            corr_r = lvc2_red * (1 + ab_r[0]*d1**2 + ab_r[1]*d2**2)
            corr_o = lvc2_ox * (1 + ab_o[0]*d1**2 + ab_o[1]*d2**2)
        elif formula == "mult_cross":
            corr_r = lvc2_red * (1 + ab_r[0]*d1*d2)
            corr_o = lvc2_ox * (1 + ab_o[0]*d1*d2)

        label = f"{flabel.replace('D1',d1name).replace('D2',d2name)}"

        r = evaluate_correction(pd.Series(corr_r), pd.Series(real_red), label)
        if r:
            results_red.append({**r, "n_params": n_p, "category": "two_desc"})
        r = evaluate_correction(pd.Series(corr_o), pd.Series(real_ox), label)
        if r:
            results_ox.append({**r, "n_params": n_p, "category": "two_desc"})

# ============================================================
# 7. RESULTS
# ============================================================

def print_results_table(results, process_type, baseline_mae):
    df = pd.DataFrame(results)
    df = df.sort_values("MAE")

    print(f"\n{'='*100}")
    print(f"  {process_type.upper()} - TOP 30 MODELS (sorted by MAE)")
    print(f"  Baseline MAE to beat: {baseline_mae:.1f} kJ/mol")
    print(f"{'='*100}")

    # Show which beat baseline
    beats = df[df["MAE"] < baseline_mae]
    if len(beats) > 0:
        print(f"\n  *** {len(beats)} models BEAT the baseline! ***\n")
        print(f"  {'Rank':<5} {'MAE':>6} {'R2':>6} {'n':>3} {'#p':>3} {'Category':<15} {'Label'}")
        print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*3} {'-'*3} {'-'*15} {'-'*50}")
        for i, (_, row) in enumerate(beats.iterrows()):
            print(f"  {i+1:<5} {row['MAE']:6.3f} {row['R2']:6.3f} {row['n']:3.0f} {row['n_params']:3.0f} {row['category']:<15} {row['label']}")
    else:
        print(f"\n  No models beat the baseline MAE of {baseline_mae:.1f}")

    print(f"\n  --- Top 30 overall ---\n")
    print(f"  {'Rank':<5} {'MAE':>6} {'R2':>6} {'n':>3} {'#p':>3} {'Category':<15} {'Label'}")
    print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*3} {'-'*3} {'-'*15} {'-'*50}")
    for i, (_, row) in enumerate(df.head(30).iterrows()):
        flag = " ***" if row["MAE"] < baseline_mae else ""
        print(f"  {i+1:<5} {row['MAE']:6.3f} {row['R2']:6.3f} {row['n']:3.0f} {row['n_params']:3.0f} {row['category']:<15} {row['label']}{flag}")

    return df


print("\n" + "#"*100)
print("#  EXHAUSTIVE FUKUI/CDFT CORRECTION SEARCH")
print("#  n_molecules = ", len(data))
print("#"*100)

# Baselines:
# Ox: LVC2 brut calibre MAE=1.3, LUMO corr MAE=1.5
# Red: LVC2 brut calibre MAE=1.7, LUMO corr MAE=1.4

df_ox = print_results_table(results_ox, "OXIDATION", baseline_mae=1.3)
df_red = print_results_table(results_red, "REDUCTION", baseline_mae=1.4)

# ============================================================
# 8. FOCUSED ANALYSIS: best parameter-free vs 1-param vs 2-param
# ============================================================

print(f"\n{'='*100}")
print("  SUMMARY BY CATEGORY")
print(f"{'='*100}")

for process, df_proc, baseline in [("OX", df_ox, 1.3), ("RED", df_red, 1.4)]:
    print(f"\n  --- {process} ---")
    for cat in ["baseline", "mult_pf", "mult_power", "additive_1p", "two_desc"]:
        sub = df_proc[df_proc["category"] == cat].sort_values("MAE")
        if len(sub) == 0:
            continue
        best = sub.iloc[0]
        beat = " BEATS" if best["MAE"] < baseline else ""
        print(f"    {cat:<15}: best MAE = {best['MAE']:.3f} (R²={best['R2']:.3f}, #p={best['n_params']:.0f}){beat}")
        print(f"                    {best['label']}")

# ============================================================
# 9. CHECK: does anything beat LVC2 brut for BOTH ox AND red?
# ============================================================

print(f"\n{'='*100}")
print("  MODELS THAT IMPROVE BOTH OX AND RED vs LVC2 BRUT (MAE_ox<1.3 AND MAE_red<1.7)")
print(f"{'='*100}")

# Get all unique labels
all_labels = set(df_ox["label"].values) & set(df_red["label"].values)
joint_results = []
for label in all_labels:
    ox_row = df_ox[df_ox["label"] == label].iloc[0]
    red_row = df_red[df_red["label"] == label].iloc[0]
    joint_results.append({
        "label": label,
        "MAE_ox": ox_row["MAE"],
        "MAE_red": red_row["MAE"],
        "R2_ox": ox_row["R2"],
        "R2_red": red_row["R2"],
        "n_params": ox_row["n_params"],
        "category": ox_row["category"],
    })

df_joint = pd.DataFrame(joint_results)
df_joint["MAE_avg"] = (df_joint["MAE_ox"] + df_joint["MAE_red"]) / 2

# Models that beat both baselines
both_better = df_joint[(df_joint["MAE_ox"] < 1.3) & (df_joint["MAE_red"] < 1.7)]
both_better = both_better.sort_values("MAE_avg")

if len(both_better) > 0:
    print(f"\n  {len(both_better)} models found!\n")
    print(f"  {'Rank':<5} {'MAE_ox':>7} {'MAE_red':>7} {'MAE_avg':>7} {'R2_ox':>6} {'R2_red':>6} {'#p':>3} {'Label'}")
    print(f"  {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*3} {'-'*50}")
    for i, (_, row) in enumerate(both_better.head(20).iterrows()):
        print(f"  {i+1:<5} {row['MAE_ox']:7.3f} {row['MAE_red']:7.3f} {row['MAE_avg']:7.3f} {row['R2_ox']:6.3f} {row['R2_red']:6.3f} {row['n_params']:3.0f} {row['label']}")
else:
    print("\n  No models beat both baselines simultaneously.")

# Top 20 by average MAE regardless
print(f"\n  --- Top 20 by average MAE ---\n")
df_joint_sorted = df_joint.sort_values("MAE_avg").head(20)
print(f"  {'Rank':<5} {'MAE_ox':>7} {'MAE_red':>7} {'MAE_avg':>7} {'R2_ox':>6} {'R2_red':>6} {'#p':>3} {'Cat':<15} {'Label'}")
print(f"  {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*3} {'-'*15} {'-'*50}")
for i, (_, row) in enumerate(df_joint_sorted.iterrows()):
    print(f"  {i+1:<5} {row['MAE_ox']:7.3f} {row['MAE_red']:7.3f} {row['MAE_avg']:7.3f} {row['R2_ox']:6.3f} {row['R2_red']:6.3f} {row['n_params']:3.0f} {row['category']:<15} {row['label']}")

print(f"\n  Total models evaluated: OX={len(df_ox)}, RED={len(df_red)}")
print("  Done.")
