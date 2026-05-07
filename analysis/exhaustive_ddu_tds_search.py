#!/usr/bin/env python3
"""
Exhaustive search for descriptors predicting ΔΔU, TΔS, and ΔΔU - TΔS.

Baselines to beat (constant models, rapport section 3.3-3.4):
  ΔΔU_red : constant = -12.2 kJ/mol, MAE ≈ 1.7
  ΔΔU_ox  : constant = +1.5 kJ/mol,  MAE ≈ 0.3
  TΔS_red : linear(S_vib),            MAE ≈ 1.5  (constant MAE ≈ 1.9)
  TΔS_ox  : constant = -1.1 kJ/mol,  MAE ≈ 0.8

Two tiers of molecules:
  - Tier A (n ≈ 13): molecules with Fukui/CDFT from predict_dg
  - Tier B (n = 15):  rapport molecules with dQ, LUMO_O_frac, S_HR, etc.

Descriptors tested:
  - "Free" from redox data: EA, IP, RMSD → η, μ, ω
  - Thermodynamic neutrals from predict_dg: ZPE_n, TS_vib_n, U_n, etc.
  - Fukui/CDFT from predict_dg: fplus_sumO, fminus_sumO, dual_sumO, η, μ, ω, gap, etc.
  - QTAIM from predict_dg: bader_sumO, DI_sumO, LI_sumO
  - Molecular from descriptors_per_molecule: dQ, LUMO_O_frac, S_HR
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import minimize
from itertools import combinations
import glob
import warnings
warnings.filterwarnings("ignore")

BASE = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5"
DEA = {"Br-DA", "Br-HMC", "I-DA", "I-HMC"}

# ============================================================
# 1. LOAD TARGET DATA: Boltzmann-weighted ΔΔU, TΔS per molecule
# ============================================================

print("=" * 80)
print("  LOADING DATA")
print("=" * 80)

files = glob.glob(f"{BASE}/data/Acetonitrile/*/redox/results/*_redox_conformers.csv")
rows_redox = []
for f in sorted(files):
    mol = f.split("/")[-1].replace("_redox_conformers.csv", "")
    df = pd.read_csv(f)
    w = df["weight"].values
    w = w / w.sum()
    row = {"molecule": mol}
    for col in df.columns:
        if col not in ["conformer", "weight", "bond_breaking_red", "broken_bonds_red",
                        "bond_breaking_ox", "broken_bonds_ox"]:
            row[col] = np.sum(w * df[col].values)
    rows_redox.append(row)

redox = pd.DataFrame(rows_redox)
redox = redox[~redox["molecule"].isin(DEA)].copy()
print(f"  Molecules with redox data (excl DEA): {len(redox)}")

# ============================================================
# 2. LOAD DESCRIPTORS FROM predict_dg (Fukui, CDFT, thermo, QTAIM)
# ============================================================

all_cols_pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", nrows=0).columns.tolist()

# Select useful columns
keep = ["molecule", "conformer", "weight"]
# Neutral thermodynamic
for c in ["ZPE_n", "ZPE_n_kj", "TS_vib_n", "TS_vib_n_kj", "U_n", "U_n_kj",
           "H_n", "H_n_kj", "G_n", "G_n_kj", "Ee_n", "Ee_n_kj",
           "TS_n", "TS_n_kj"]:
    if c in all_cols_pdg:
        keep.append(c)

# Fukui sums
for prefix in ["fplus_sum", "fminus_sum", "f0_sum", "dual_sum"]:
    for elem in ["O", "C", "H", "N"]:
        c = f"{prefix}{elem}_n"
        if c in all_cols_pdg:
            keep.append(c)

# CDFT global
for c in ["mu_n", "eta_n", "omega_n", "HOMO_n", "LUMO_n", "gap_n"]:
    if c in all_cols_pdg:
        keep.append(c)

# QTAIM sums (neutral only)
for prefix in ["bader_sum", "DI_sum", "LI_sum"]:
    for elem in ["O", "C", "H", "N"]:
        c = f"{prefix}{elem}_n"
        if c in all_cols_pdg:
            keep.append(c)

# Per-atom Fukui (for max/min)
for prefix in ["fplus_", "fminus_", "dual_", "f0_"]:
    for c in all_cols_pdg:
        if c.startswith(prefix) and c.endswith("_n") and "sum" not in c:
            keep.append(c)

keep = list(dict.fromkeys(keep))
pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", usecols=keep)
pdg = pdg[~pdg["molecule"].isin(DEA)].copy()
print(f"  Molecules in predict_dg (excl DEA): {pdg['molecule'].nunique()}")

# Boltzmann-average predict_dg descriptors
numeric_pdg = [c for c in pdg.columns if c not in ["molecule", "conformer", "weight"]]


def boltzmann_avg(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}
    for col in numeric_pdg:
        vals = group[col].values
        if np.all(np.isfinite(vals)):
            result[col] = np.sum(w * vals)
        else:
            result[col] = np.nan
    return pd.Series(result)


bw_pdg = pdg.groupby("molecule").apply(boltzmann_avg).reset_index()

# Compute derived descriptors (max/min of per-atom Fukui)
per_atom_O = [c for c in pdg.columns if "_O" in c and c.endswith("_n") and "sum" not in c]
fplus_O_atoms = [c for c in per_atom_O if c.startswith("fplus_")]
fminus_O_atoms = [c for c in per_atom_O if c.startswith("fminus_")]
dual_O_atoms = [c for c in per_atom_O if c.startswith("dual_")]
per_atom_C = [c for c in pdg.columns if "_C" in c and c.endswith("_n") and "sum" not in c]
fplus_C_atoms = [c for c in per_atom_C if c.startswith("fplus_")]
dual_all = dual_O_atoms + [c for c in per_atom_C if c.startswith("dual_")] + \
           [c for c in pdg.columns if "_H" in c and c.endswith("_n") and "sum" not in c and c.startswith("dual_")]


def compute_derived(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}
    for atoms, label in [
        (fplus_O_atoms, "fplus_maxO"),
        (fminus_O_atoms, "fminus_maxO"),
        (fplus_C_atoms, "fplus_maxC"),
    ]:
        if atoms:
            per_conf = group[atoms].max(axis=1).values
            result[label] = np.sum(w * per_conf)
    if dual_O_atoms:
        result["dual_minO"] = np.sum(w * group[dual_O_atoms].min(axis=1).values)
        result["dual_maxO"] = np.sum(w * group[dual_O_atoms].max(axis=1).values)
    if dual_all:
        result["dual_min"] = np.sum(w * group[dual_all].min(axis=1).values)
        result["dual_max"] = np.sum(w * group[dual_all].max(axis=1).values)
    if fplus_O_atoms:
        spread = group[fplus_O_atoms].max(axis=1).values - group[fplus_O_atoms].min(axis=1).values
        result["fplus_spreadO"] = np.sum(w * spread)
    return pd.Series(result)


derived = pdg.groupby("molecule").apply(compute_derived).reset_index()
bw_pdg = bw_pdg.merge(derived, on="molecule", how="left")

# ============================================================
# 3. LOAD descriptors_per_molecule (dQ, LUMO_O_frac, S_HR)
# ============================================================

desc_mol = pd.read_csv(f"{BASE}/rapport/data/descriptors_per_molecule.csv")
desc_mol = desc_mol.rename(columns={"molecule_name": "molecule"} if "molecule_name" in desc_mol.columns else {})
desc_keep = ["molecule"]
for c in ["dQ_red", "dQ_ox", "LUMO_O_frac", "LUMO_C_frac", "S_HR_red", "S_HR_ox",
           "RMSD_red", "RMSD_ox", "LUMO_eV", "gap_eV", "n_imag_red", "n_imag_ox"]:
    if c in desc_mol.columns:
        desc_keep.append(c)
desc_mol = desc_mol[desc_keep].copy()
print(f"  Molecules in descriptors_per_molecule: {len(desc_mol)}")

# ============================================================
# 4. LOAD Politzer descriptors
# ============================================================

politzer = pd.read_csv(f"{BASE}/rapport/data/politzer_per_molecule.csv")
politzer = politzer.rename(columns={"molecule_name": "molecule"} if "molecule_name" in politzer.columns else {})
pol_keep = ["molecule"]
for c in politzer.columns:
    if c != "molecule" and c != "molecule_name" and c != "n_conformers":
        pol_keep.append(c)
politzer = politzer[pol_keep].copy()
print(f"  Molecules in Politzer: {len(politzer)}")

# ============================================================
# 5. MERGE ALL DATA
# ============================================================

data = redox.copy()

# Compute "free" descriptors from redox data
eps = 1e-10
data["eta_redox"] = (data["IP_kJ"] - data["EA_kJ"]) / 2  # hardness
data["mu_redox"] = -(data["IP_kJ"] + data["EA_kJ"]) / 2  # chemical potential
data["omega_redox"] = data["mu_redox"]**2 / (2 * data["eta_redox"] + eps)  # electrophilicity
data["ddU_minus_TdS_red"] = data["ddU_red_kJ"] - data["TdS_red_kJ"]
data["ddU_minus_TdS_ox"] = data["ddU_ox_kJ"] - data["TdS_ox_kJ"]

# Merge predict_dg
data = data.merge(bw_pdg, on="molecule", how="left")

# Merge descriptors_per_molecule
data = data.merge(desc_mol, on="molecule", how="left")

# Merge Politzer
data = data.merge(politzer, on="molecule", how="left")

# Compute ratio descriptors from Fukui
if "fplus_sumO_n" in data.columns and "fplus_sumC_n" in data.columns:
    data["fplus_ratio_OC"] = data["fplus_sumO_n"] / (data["fplus_sumC_n"] + eps)
    data["fplus_frac_O"] = data["fplus_sumO_n"] / (data["fplus_sumO_n"] + data["fplus_sumC_n"] + eps)
if "fminus_sumO_n" in data.columns and "fminus_sumC_n" in data.columns:
    data["fminus_ratio_OC"] = data["fminus_sumO_n"] / (data["fminus_sumC_n"] + eps)
    data["fminus_frac_O"] = data["fminus_sumO_n"] / (data["fminus_sumO_n"] + data["fminus_sumC_n"] + eps)
if "dual_sumO_n" in data.columns and "dual_sumC_n" in data.columns:
    data["dual_ratio_OC"] = data["dual_sumO_n"] / (data["dual_sumC_n"] + eps)

print(f"\n  Final merged dataset: {len(data)} molecules, {len(data.columns)} columns")

# ============================================================
# 6. DEFINE TARGETS AND DESCRIPTORS
# ============================================================

targets = {
    "ddU_red": ("ddU_red_kJ", 1.7, "ΔΔU reduction"),
    "ddU_ox": ("ddU_ox_kJ", 0.3, "ΔΔU oxidation"),
    "TdS_red": ("TdS_red_kJ", 1.5, "TΔS reduction"),
    "TdS_ox": ("TdS_ox_kJ", 0.8, "TΔS oxidation"),
    "ddU_minus_TdS_red": ("ddU_minus_TdS_red", None, "ΔΔU - TΔS reduction"),
    "ddU_minus_TdS_ox": ("ddU_minus_TdS_ox", None, "ΔΔU - TΔS oxidation"),
}

# Build descriptor catalog
descriptors = {}

# --- Tier 0: "Free" from redox data (available for all molecules) ---
tier0 = {
    "EA": "EA_kJ",
    "IP": "IP_kJ",
    "Edef_red": "Edef_red_kJ",
    "Edef_ox": "Edef_ox_kJ",
    "RMSD_red": "RMSD_red_A",
    "RMSD_ox": "RMSD_ox_A",
    "eta_redox": "eta_redox",
    "mu_redox": "mu_redox",
    "omega_redox": "omega_redox",
}

# --- Tier 1: From predict_dg (Fukui, CDFT, thermo, QTAIM) ---
tier1 = {
    "fplus_sumO": "fplus_sumO_n",
    "fminus_sumO": "fminus_sumO_n",
    "dual_sumO": "dual_sumO_n",
    "f0_sumO": "f0_sumO_n",
    "fplus_sumC": "fplus_sumC_n",
    "fminus_sumC": "fminus_sumC_n",
    "dual_sumC": "dual_sumC_n",
    "fplus_sumH": "fplus_sumH_n",
    "eta_cdft": "eta_n",
    "mu_cdft": "mu_n",
    "omega_cdft": "omega_n",
    "HOMO": "HOMO_n",
    "LUMO": "LUMO_n",
    "gap": "gap_n",
    "ZPE": "ZPE_n_kj",
    "TS_vib": "TS_vib_n_kj",
    "U_neutral": "U_n_kj",
    "fplus_maxO": "fplus_maxO",
    "fminus_maxO": "fminus_maxO",
    "fplus_maxC": "fplus_maxC",
    "dual_minO": "dual_minO",
    "dual_maxO": "dual_maxO",
    "dual_min": "dual_min",
    "dual_max": "dual_max",
    "fplus_spreadO": "fplus_spreadO",
    "fplus_ratio_OC": "fplus_ratio_OC",
    "fplus_frac_O": "fplus_frac_O",
    "fminus_ratio_OC": "fminus_ratio_OC",
    "fminus_frac_O": "fminus_frac_O",
    "dual_ratio_OC": "dual_ratio_OC",
    # QTAIM
    "bader_sumO": "bader_sumO_n",
    "bader_sumC": "bader_sumC_n",
    "bader_sumH": "bader_sumH_n",
    "DI_sumO": "DI_sumO_n",
    "DI_sumC": "DI_sumC_n",
    "LI_sumO": "LI_sumO_n",
    "LI_sumC": "LI_sumC_n",
}

# --- Tier 2: From descriptors_per_molecule ---
tier2 = {
    "dQ_red": "dQ_red",
    "dQ_ox": "dQ_ox",
    "LUMO_O_frac": "LUMO_O_frac",
    "LUMO_C_frac": "LUMO_C_frac",
    "S_HR_red": "S_HR_red",
    "S_HR_ox": "S_HR_ox",
    "gap_eV": "gap_eV",
    "LUMO_eV": "LUMO_eV",
}

# --- Tier 3: From Politzer ---
tier3 = {}
for c in politzer.columns:
    if c != "molecule":
        tier3[f"pol_{c}"] = c

# Collect all available descriptors
all_tiers = [("tier0", tier0), ("tier1", tier1), ("tier2", tier2), ("tier3", tier3)]
available = {}
for tier_name, tier_dict in all_tiers:
    for name, col in tier_dict.items():
        if col in data.columns and data[col].notna().sum() >= 5:
            available[name] = {"col": col, "tier": tier_name, "n_valid": int(data[col].notna().sum())}

print(f"\n  Available descriptors: {len(available)}")
for tier_name in ["tier0", "tier1", "tier2", "tier3"]:
    n = sum(1 for v in available.values() if v["tier"] == tier_name)
    print(f"    {tier_name}: {n} descriptors")


# ============================================================
# 7. EVALUATION FUNCTIONS
# ============================================================

def eval_linear(x, y, min_n=5):
    """Fit y = a + b*x via OLS, return metrics."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < min_n:
        return None
    X = x[mask].reshape(-1, 1)
    yv = y[mask]
    reg = LinearRegression().fit(X, yv)
    y_pred = reg.predict(X)
    r2 = r2_score(yv, y_pred)
    mae = mean_absolute_error(yv, y_pred)
    # Adjusted R²
    p = 1
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    return {"R2": r2, "R2_adj": r2_adj, "MAE": mae, "n": n,
            "slope": reg.coef_[0], "intercept": reg.intercept_}


def eval_constant(y, min_n=5):
    """Constant model: predict mean."""
    mask = np.isfinite(y)
    n = mask.sum()
    if n < min_n:
        return None
    yv = y[mask]
    mean_val = np.mean(yv)
    mae = mean_absolute_error(yv, np.full_like(yv, mean_val))
    return {"R2": 0.0, "R2_adj": 0.0, "MAE": mae, "n": n,
            "constant": mean_val}


def eval_2desc_linear(x1, x2, y, min_n=6):
    """Fit y = a + b*x1 + c*x2."""
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    n = mask.sum()
    if n < min_n:
        return None
    X = np.column_stack([x1[mask], x2[mask]])
    yv = y[mask]
    reg = LinearRegression().fit(X, yv)
    y_pred = reg.predict(X)
    r2 = r2_score(yv, y_pred)
    mae = mean_absolute_error(yv, y_pred)
    p = 2
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    return {"R2": r2, "R2_adj": r2_adj, "MAE": mae, "n": n,
            "coef1": reg.coef_[0], "coef2": reg.coef_[1], "intercept": reg.intercept_}


def eval_nonlinear(x, y, transforms, min_n=5):
    """Fit y = a + b*transform(x) for various transforms."""
    results = []
    mask_base = np.isfinite(x) & np.isfinite(y)
    for tname, tfunc in transforms:
        xt = tfunc(x)
        mask = mask_base & np.isfinite(xt)
        n = mask.sum()
        if n < min_n:
            continue
        X = xt[mask].reshape(-1, 1)
        yv = y[mask]
        reg = LinearRegression().fit(X, yv)
        y_pred = reg.predict(X)
        r2 = r2_score(yv, y_pred)
        mae = mean_absolute_error(yv, y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2
        results.append({"transform": tname, "R2": r2, "R2_adj": r2_adj, "MAE": mae, "n": n,
                         "slope": reg.coef_[0], "intercept": reg.intercept_})
    return results


# Standard transforms
TRANSFORMS = [
    ("x", lambda x: x),
    ("x²", lambda x: x**2),
    ("|x|", lambda x: np.abs(x)),
    ("√|x|", lambda x: np.sqrt(np.abs(x))),
    ("1/x", lambda x: np.where(np.abs(x) > 1e-10, 1.0 / x, np.nan)),
    ("log|x|", lambda x: np.where(np.abs(x) > 1e-10, np.log(np.abs(x)), np.nan)),
    ("x³", lambda x: x**3),
]

# ============================================================
# 8. RUN EXHAUSTIVE SEARCH
# ============================================================

print("\n" + "=" * 80)
print("  RUNNING EXHAUSTIVE SEARCH")
print("=" * 80)

all_results = {}

for target_key, (target_col, baseline_mae, target_label) in targets.items():
    print(f"\n--- Target: {target_label} ({target_col}) ---")

    y_full = data[target_col].values
    results = []

    # Baseline: constant model
    cst = eval_constant(y_full)
    if cst:
        results.append({
            "label": f"Constant ({cst['constant']:.2f})",
            "MAE": cst["MAE"], "R2": 0.0, "R2_adj": 0.0,
            "n": cst["n"], "n_params": 0, "category": "baseline",
        })
        print(f"  Constant baseline: MAE = {cst['MAE']:.3f} (n={cst['n']})")

    # ---- Single-descriptor linear ----
    for dname, dinfo in available.items():
        col = dinfo["col"]
        x = data[col].values.astype(float)
        r = eval_linear(x, y_full)
        if r:
            results.append({
                "label": f"linear({dname})",
                "MAE": r["MAE"], "R2": r["R2"], "R2_adj": r["R2_adj"],
                "n": r["n"], "n_params": 1, "category": "linear_1d",
                "slope": r["slope"], "intercept": r["intercept"],
            })

    # ---- Single-descriptor with transforms ----
    for dname, dinfo in available.items():
        col = dinfo["col"]
        x = data[col].values.astype(float)
        nl_results = eval_nonlinear(x, y_full, TRANSFORMS[1:])  # skip "x" (already done)
        for nr in nl_results:
            results.append({
                "label": f"linear({nr['transform']}({dname}))",
                "MAE": nr["MAE"], "R2": nr["R2"], "R2_adj": nr["R2_adj"],
                "n": nr["n"], "n_params": 1, "category": "nonlinear_1d",
            })

    # ---- Two-descriptor linear ----
    # Physically motivated pairs
    focused_desc = [
        "eta_redox", "mu_redox", "omega_redox", "EA", "IP",
        "RMSD_red", "RMSD_ox", "Edef_red", "Edef_ox",
        "fplus_sumO", "fminus_sumO", "dual_sumO",
        "eta_cdft", "omega_cdft", "gap", "HOMO", "LUMO",
        "ZPE", "TS_vib", "fplus_maxO", "fminus_maxO",
        "LUMO_O_frac", "dQ_red", "dQ_ox", "S_HR_red", "S_HR_ox",
        "fplus_ratio_OC", "fplus_frac_O", "bader_sumO", "DI_sumO",
        "fplus_spreadO", "dual_ratio_OC",
    ]
    focused_available = [d for d in focused_desc if d in available]

    two_desc_pairs = set()
    for i, d1 in enumerate(focused_available):
        for d2 in focused_available[i + 1:]:
            two_desc_pairs.add((d1, d2))

    print(f"  Testing {len(two_desc_pairs)} two-descriptor pairs...")

    for d1name, d2name in two_desc_pairs:
        x1 = data[available[d1name]["col"]].values.astype(float)
        x2 = data[available[d2name]["col"]].values.astype(float)
        r = eval_2desc_linear(x1, x2, y_full)
        if r:
            results.append({
                "label": f"linear({d1name}, {d2name})",
                "MAE": r["MAE"], "R2": r["R2"], "R2_adj": r["R2_adj"],
                "n": r["n"], "n_params": 2, "category": "linear_2d",
            })

    # ---- Two descriptors with one transform ----
    # Only for top single descriptors + key partners
    key_desc = ["eta_redox", "fplus_sumO", "gap", "LUMO_O_frac",
                "dQ_red", "dQ_ox", "RMSD_red", "RMSD_ox", "TS_vib", "ZPE",
                "omega_redox", "fplus_maxO", "bader_sumO"]
    key_available = [d for d in key_desc if d in available]
    key_transforms = [("x²", lambda x: x**2), ("|x|", lambda x: np.abs(x)),
                      ("√|x|", lambda x: np.sqrt(np.abs(x)))]

    for d1name in key_available:
        for d2name in key_available:
            if d1name >= d2name:
                continue
            x1 = data[available[d1name]["col"]].values.astype(float)
            x2 = data[available[d2name]["col"]].values.astype(float)
            for tname, tfunc in key_transforms:
                # transform d1, keep d2 linear
                x1t = tfunc(x1)
                r = eval_2desc_linear(x1t, x2, y_full)
                if r:
                    results.append({
                        "label": f"linear({tname}({d1name}), {d2name})",
                        "MAE": r["MAE"], "R2": r["R2"], "R2_adj": r["R2_adj"],
                        "n": r["n"], "n_params": 2, "category": "nonlinear_2d",
                    })

    # ---- Product descriptors (d1 * d2) fitted linearly ----
    for d1name in key_available:
        for d2name in key_available:
            if d1name >= d2name:
                continue
            x1 = data[available[d1name]["col"]].values.astype(float)
            x2 = data[available[d2name]["col"]].values.astype(float)
            xprod = x1 * x2
            r = eval_linear(xprod, y_full)
            if r:
                results.append({
                    "label": f"linear({d1name}*{d2name})",
                    "MAE": r["MAE"], "R2": r["R2"], "R2_adj": r["R2_adj"],
                    "n": r["n"], "n_params": 1, "category": "product_1d",
                })

    all_results[target_key] = pd.DataFrame(results)
    print(f"  Total models: {len(results)}")

# ============================================================
# 9. PRINT RESULTS
# ============================================================


def print_target_results(target_key, target_label, baseline_mae, df_res):
    df = df_res.sort_values("MAE")

    print(f"\n{'=' * 100}")
    print(f"  {target_label.upper()} — TOP 40 MODELS (sorted by MAE)")
    if baseline_mae is not None:
        print(f"  Baseline MAE to beat: {baseline_mae:.1f} kJ/mol")
    print(f"  Total models evaluated: {len(df)}")
    print(f"{'=' * 100}")

    # Show constant baseline first
    cst_rows = df[df["category"] == "baseline"]
    if len(cst_rows) > 0:
        cst = cst_rows.iloc[0]
        print(f"\n  Constant baseline: MAE = {cst['MAE']:.3f} (n={cst['n']:.0f})")

    # Models that beat baseline
    if baseline_mae is not None:
        beats = df[df["MAE"] < baseline_mae]
        if len(beats) > 0:
            print(f"\n  *** {len(beats)} models BEAT the baseline ***")

    print(f"\n  {'Rk':<4} {'MAE':>6} {'R²':>6} {'R²adj':>6} {'n':>3} {'#p':>3} {'Category':<15} {'Label'}")
    print(f"  {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*3} {'-'*3} {'-'*15} {'-'*55}")
    for i, (_, row) in enumerate(df.head(40).iterrows()):
        flag = " ***" if (baseline_mae and row["MAE"] < baseline_mae) else ""
        print(f"  {i+1:<4} {row['MAE']:6.3f} {row['R2']:6.3f} {row['R2_adj']:6.3f} "
              f"{row['n']:3.0f} {row['n_params']:3.0f} {row['category']:<15} {row['label']}{flag}")

    # Summary by category
    print(f"\n  --- Best per category ---")
    for cat in ["baseline", "linear_1d", "nonlinear_1d", "linear_2d", "nonlinear_2d", "product_1d"]:
        sub = df[df["category"] == cat].sort_values("MAE")
        if len(sub) == 0:
            continue
        best = sub.iloc[0]
        beat = " BEATS" if (baseline_mae and best["MAE"] < baseline_mae) else ""
        print(f"    {cat:<15}: MAE={best['MAE']:.3f} R²adj={best['R2_adj']:.3f} n={best['n']:.0f}{beat}")
        print(f"                    {best['label']}")


print("\n" + "#" * 100)
print("#  EXHAUSTIVE ΔΔU / TΔS DESCRIPTOR SEARCH")
print("#  n_molecules total =", len(data))
print("#" * 100)

for target_key, (target_col, baseline_mae, target_label) in targets.items():
    if target_key in all_results:
        print_target_results(target_key, target_label, baseline_mae, all_results[target_key])

# ============================================================
# 10. RAPPORT-15 ONLY ANALYSIS
# ============================================================

rapport15 = {"Al-CA", "CA", "Cl-DA", "Cl-HMC", "DA", "DHI", "F-DA", "F-HMC",
             "H2C-CA", "HMC", "HO-CA", "L-DA", "O-DHI", "P-DHI", "S-DHI"}
data_r15 = data[data["molecule"].isin(rapport15)].copy()

print(f"\n\n{'#' * 100}")
print(f"#  RAPPORT-15 ONLY ANALYSIS (n={len(data_r15)})")
print(f"{'#' * 100}")

for target_key, (target_col, baseline_mae, target_label) in targets.items():
    y_r15 = data_r15[target_col].values
    results_r15 = []

    cst = eval_constant(y_r15)
    if cst:
        results_r15.append({
            "label": f"Constant ({cst['constant']:.2f})",
            "MAE": cst["MAE"], "R2": 0.0, "R2_adj": 0.0,
            "n": cst["n"], "n_params": 0, "category": "baseline",
        })

    # Single descriptor
    for dname, dinfo in available.items():
        col = dinfo["col"]
        x = data_r15[col].values.astype(float)
        r = eval_linear(x, y_r15)
        if r and r["n"] >= 5:
            results_r15.append({
                "label": f"linear({dname})",
                "MAE": r["MAE"], "R2": r["R2"], "R2_adj": r["R2_adj"],
                "n": r["n"], "n_params": 1, "category": "linear_1d",
            })
        # Transforms
        nl = eval_nonlinear(x, y_r15, TRANSFORMS[1:])
        for nr in nl:
            if nr["n"] >= 5:
                results_r15.append({
                    "label": f"linear({nr['transform']}({dname}))",
                    "MAE": nr["MAE"], "R2": nr["R2"], "R2_adj": nr["R2_adj"],
                    "n": nr["n"], "n_params": 1, "category": "nonlinear_1d",
                })

    # Two-descriptor (focused pairs only, n is small)
    for d1name in focused_available:
        for d2name in focused_available:
            if d1name >= d2name:
                continue
            x1 = data_r15[available[d1name]["col"]].values.astype(float)
            x2 = data_r15[available[d2name]["col"]].values.astype(float)
            r = eval_2desc_linear(x1, x2, y_r15, min_n=6)
            if r:
                results_r15.append({
                    "label": f"linear({d1name}, {d2name})",
                    "MAE": r["MAE"], "R2": r["R2"], "R2_adj": r["R2_adj"],
                    "n": r["n"], "n_params": 2, "category": "linear_2d",
                })

    df_r15 = pd.DataFrame(results_r15)
    if len(df_r15) > 0:
        print_target_results(target_key, f"{target_label} [RAPPORT-15]", baseline_mae, df_r15)

# ============================================================
# 11. CROSS-TARGET ANALYSIS: what improves BOTH ddU_red AND TdS_red?
# ============================================================

print(f"\n\n{'=' * 100}")
print("  CROSS-TARGET: MODELS THAT IMPROVE ΔΔU_red AND TΔS_red SIMULTANEOUSLY")
print(f"{'=' * 100}")

# Find descriptors that appear in top-20 for both targets
for dataset_label, df_ddu, df_tds in [
    ("ALL MOLECULES", all_results.get("ddU_red"), all_results.get("TdS_red")),
]:
    if df_ddu is None or df_tds is None:
        continue
    print(f"\n  --- {dataset_label} ---")

    # Get 1D descriptors, extract base descriptor name
    def extract_desc(label):
        """Extract descriptor name from label like 'linear(eta_redox)' or 'linear(x²(eta_redox))'."""
        if "(" in label:
            inner = label.split("(", 1)[1].rstrip(")")
            # Remove transform
            if "(" in inner:
                inner = inner.split("(", 1)[1].rstrip(")")
            return inner
        return label

    top_ddu = df_ddu.sort_values("MAE").head(30)
    top_tds = df_tds.sort_values("MAE").head(30)

    ddu_descs = set(extract_desc(l) for l in top_ddu["label"])
    tds_descs = set(extract_desc(l) for l in top_tds["label"])
    common = ddu_descs & tds_descs
    if common:
        print(f"  Descriptors in top-30 for BOTH: {common}")
    else:
        print("  No common descriptors in top-30 for both targets.")

print("\n  Done.")
