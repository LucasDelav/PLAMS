#!/usr/bin/env python3
"""
Exhaustive search for descriptors predicting ΔΔU, TΔS, and ΔΔU - TΔS.
V2: Only neutral-state descriptors (no RMSD, Edef, dQ, S_HR).
    Searches for symmetric red/ox models (same or mirror descriptors).

Baselines to beat (constant models, rapport section 3.3-3.4):
  ΔΔU_red : constant = -12.7 kJ/mol, MAE ≈ 1.7
  ΔΔU_ox  : constant = +0.5 kJ/mol,  MAE ≈ 0.4
  TΔS_red : linear(S_vib),            MAE ≈ 1.5  (constant MAE ≈ 1.8)
  TΔS_ox  : constant = -2.6 kJ/mol,  MAE ≈ 0.9

FORBIDDEN descriptors (require ionic geometry optimization):
  RMSD_red/ox, Edef_red/ox, dQ_red/ox, S_HR_red/ox
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from itertools import combinations
import glob
import warnings
warnings.filterwarnings("ignore")

BASE = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5"
DEA = {"Br-DA", "Br-HMC", "I-DA", "I-HMC"}
RAPPORT15 = {"Al-CA", "CA", "Cl-DA", "Cl-HMC", "DA", "DHI", "F-DA", "F-HMC",
             "H2C-CA", "HMC", "HO-CA", "L-DA", "O-DHI", "P-DHI", "S-DHI"}

# ============================================================
# 1. LOAD TARGET DATA
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

# ============================================================
# 2. LOAD predict_dg DESCRIPTORS (neutral-state only)
# ============================================================

all_cols_pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", nrows=0).columns.tolist()

keep = ["molecule", "conformer", "weight"]
# Neutral thermodynamic
for c in ["ZPE_n_kj", "TS_vib_n_kj", "U_n_kj", "H_n_kj", "TS_n_kj"]:
    if c in all_cols_pdg:
        keep.append(c)
# Vertical EA/IP (from SP on neutral geom)
for c in ["EA_vert_kj", "IP_vert_kj"]:
    if c in all_cols_pdg:
        keep.append(c)
# Fukui sums (neutral)
for prefix in ["fplus_sum", "fminus_sum", "f0_sum", "dual_sum"]:
    for elem in ["O", "C", "H", "N"]:
        c = f"{prefix}{elem}_n"
        if c in all_cols_pdg:
            keep.append(c)
# CDFT global (neutral)
for c in ["mu_n", "eta_n", "omega_n", "HOMO_n", "LUMO_n", "gap_n"]:
    if c in all_cols_pdg:
        keep.append(c)
# QTAIM sums (neutral)
for prefix in ["bader_sum", "DI_sum", "LI_sum"]:
    for elem in ["O", "C", "H", "N"]:
        c = f"{prefix}{elem}_n"
        if c in all_cols_pdg:
            keep.append(c)
# Per-atom Fukui (neutral, for max/min/spread)
for prefix in ["fplus_", "fminus_", "dual_", "f0_"]:
    for c in all_cols_pdg:
        if c.startswith(prefix) and c.endswith("_n") and "sum" not in c:
            keep.append(c)

keep = list(dict.fromkeys(keep))
pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", usecols=keep)
pdg = pdg[~pdg["molecule"].isin(DEA)].copy()

# Boltzmann average
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

# Compute derived per-atom descriptors
per_atom_O = [c for c in pdg.columns if "_O" in c and c.endswith("_n") and "sum" not in c]
fplus_O_atoms = [c for c in per_atom_O if c.startswith("fplus_")]
fminus_O_atoms = [c for c in per_atom_O if c.startswith("fminus_")]
dual_O_atoms = [c for c in per_atom_O if c.startswith("dual_")]
f0_O_atoms = [c for c in per_atom_O if c.startswith("f0_")]
per_atom_C = [c for c in pdg.columns if "_C" in c and c.endswith("_n") and "sum" not in c]
fplus_C_atoms = [c for c in per_atom_C if c.startswith("fplus_")]
fminus_C_atoms = [c for c in per_atom_C if c.startswith("fminus_")]
dual_C_atoms = [c for c in per_atom_C if c.startswith("dual_")]
per_atom_H = [c for c in pdg.columns if "_H" in c and c.endswith("_n") and "sum" not in c]
dual_H_atoms = [c for c in per_atom_H if c.startswith("dual_")]
dual_all = dual_O_atoms + dual_C_atoms + dual_H_atoms
fplus_all = fplus_O_atoms + fplus_C_atoms + [c for c in per_atom_H if c.startswith("fplus_")]
fminus_all = fminus_O_atoms + fminus_C_atoms + [c for c in per_atom_H if c.startswith("fminus_")]


def compute_derived(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}
    for atoms, label in [
        (fplus_O_atoms, "fplus_maxO"), (fminus_O_atoms, "fminus_maxO"),
        (fplus_C_atoms, "fplus_maxC"), (fminus_C_atoms, "fminus_maxC"),
        (f0_O_atoms, "f0_maxO"),
    ]:
        if atoms:
            result[label] = np.sum(w * group[atoms].max(axis=1).values)
    for atoms, label in [
        (fplus_O_atoms, "fplus_minO"), (fminus_O_atoms, "fminus_minO"),
    ]:
        if atoms:
            result[label] = np.sum(w * group[atoms].min(axis=1).values)
    if dual_O_atoms:
        result["dual_minO"] = np.sum(w * group[dual_O_atoms].min(axis=1).values)
        result["dual_maxO"] = np.sum(w * group[dual_O_atoms].max(axis=1).values)
    if dual_all:
        result["dual_min"] = np.sum(w * group[dual_all].min(axis=1).values)
        result["dual_max"] = np.sum(w * group[dual_all].max(axis=1).values)
        result["dual_spread"] = result["dual_max"] - result["dual_min"]
    if fplus_O_atoms:
        spread = group[fplus_O_atoms].max(axis=1).values - group[fplus_O_atoms].min(axis=1).values
        result["fplus_spreadO"] = np.sum(w * spread)
    if fminus_O_atoms:
        spread = group[fminus_O_atoms].max(axis=1).values - group[fminus_O_atoms].min(axis=1).values
        result["fminus_spreadO"] = np.sum(w * spread)
    return pd.Series(result)


derived = pdg.groupby("molecule").apply(compute_derived).reset_index()
bw_pdg = bw_pdg.merge(derived, on="molecule", how="left")

# ============================================================
# 3. LOAD descriptors_per_molecule (ONLY neutral-state ones)
# ============================================================

desc_mol = pd.read_csv(f"{BASE}/rapport/data/descriptors_per_molecule.csv")
# Only keep LUMO_O_frac, LUMO_C_frac, gap_eV, LUMO_eV (all from neutral)
# EXCLUDE: dQ_red/ox, RMSD_red/ox, S_HR_red/ox, Edef, n_imag
desc_keep = ["molecule", "LUMO_O_frac", "LUMO_C_frac", "gap_eV", "LUMO_eV"]
desc_mol = desc_mol[[c for c in desc_keep if c in desc_mol.columns]].copy()

# ============================================================
# 4. LOAD Politzer (neutral-state electrostatic potential)
# ============================================================

politzer = pd.read_csv(f"{BASE}/rapport/data/politzer_per_molecule.csv")
politzer = politzer.rename(columns={"molecule_name": "molecule"})
# IP_Koopmans = -HOMO, EA_Koopmans = -LUMO (from neutral orbitals)
# I_S_min, V_S_min: from neutral electrostatic potential
pol_keep = ["molecule", "IP_Koopmans_kJ", "EA_Koopmans_kJ",
            "I_S_min_eV", "IP_Politzer_kJ", "V_S_min_eV", "V_S_min_kJ"]
politzer = politzer[[c for c in pol_keep if c in politzer.columns]].copy()

# ============================================================
# 5. MERGE
# ============================================================

data = redox[["molecule", "ddU_red_kJ", "ddU_ox_kJ", "TdS_red_kJ", "TdS_ox_kJ",
              "EA_kJ", "IP_kJ"]].copy()

# Compute vertical η, μ, ω from redox EA/IP (these are vertical = from SP on neutral)
eps = 1e-10
data["eta_vert"] = (data["IP_kJ"] - data["EA_kJ"]) / 2
data["mu_vert"] = -(data["IP_kJ"] + data["EA_kJ"]) / 2
data["omega_vert"] = data["mu_vert"]**2 / (2 * data["eta_vert"] + eps)

# Combined targets
data["ddU_minus_TdS_red"] = data["ddU_red_kJ"] - data["TdS_red_kJ"]
data["ddU_minus_TdS_ox"] = data["ddU_ox_kJ"] - data["TdS_ox_kJ"]

data = data.merge(bw_pdg, on="molecule", how="left")
data = data.merge(desc_mol, on="molecule", how="left")
data = data.merge(politzer, on="molecule", how="left")

# Ratio descriptors
if "fplus_sumO_n" in data.columns and "fplus_sumC_n" in data.columns:
    data["fplus_ratio_OC"] = data["fplus_sumO_n"] / (data["fplus_sumC_n"] + eps)
    data["fplus_frac_O"] = data["fplus_sumO_n"] / (data["fplus_sumO_n"] + data["fplus_sumC_n"] + eps)
if "fminus_sumO_n" in data.columns and "fminus_sumC_n" in data.columns:
    data["fminus_ratio_OC"] = data["fminus_sumO_n"] / (data["fminus_sumC_n"] + eps)
    data["fminus_frac_O"] = data["fminus_sumO_n"] / (data["fminus_sumO_n"] + data["fminus_sumC_n"] + eps)
if "dual_sumO_n" in data.columns and "dual_sumC_n" in data.columns:
    data["dual_ratio_OC"] = data["dual_sumO_n"] / (data["dual_sumC_n"] + eps)

print(f"  Molecules total: {len(data)}")
print(f"  Rapport-15 present: {data['molecule'].isin(RAPPORT15).sum()}")

# ============================================================
# 6. DESCRIPTOR CATALOG (neutral-state only!)
# ============================================================

descriptors = {}

# Global CDFT
global_cdft = {
    "eta_vert": "eta_vert",        # hardness from vertical EA/IP
    "mu_vert": "mu_vert",          # chemical potential
    "omega_vert": "omega_vert",    # electrophilicity
    "eta_cdft": "eta_n",           # hardness from ConceptualDFT
    "mu_cdft": "mu_n",
    "omega_cdft": "omega_n",
    "HOMO": "HOMO_n",
    "LUMO": "LUMO_n",
    "gap": "gap_n",
}

# Fukui sums by element
fukui_sums = {
    "fplus_sumO": "fplus_sumO_n",
    "fminus_sumO": "fminus_sumO_n",
    "dual_sumO": "dual_sumO_n",
    "f0_sumO": "f0_sumO_n",
    "fplus_sumC": "fplus_sumC_n",
    "fminus_sumC": "fminus_sumC_n",
    "dual_sumC": "dual_sumC_n",
    "fplus_sumH": "fplus_sumH_n",
    "fminus_sumH": "fminus_sumH_n",
}

# Per-atom Fukui extrema
fukui_extrema = {
    "fplus_maxO": "fplus_maxO",
    "fminus_maxO": "fminus_maxO",
    "fplus_maxC": "fplus_maxC",
    "fminus_maxC": "fminus_maxC",
    "f0_maxO": "f0_maxO",
    "fplus_minO": "fplus_minO",
    "fminus_minO": "fminus_minO",
    "dual_minO": "dual_minO",
    "dual_maxO": "dual_maxO",
    "dual_min": "dual_min",
    "dual_max": "dual_max",
    "dual_spread": "dual_spread",
    "fplus_spreadO": "fplus_spreadO",
    "fminus_spreadO": "fminus_spreadO",
}

# Fukui ratios
fukui_ratios = {
    "fplus_ratio_OC": "fplus_ratio_OC",
    "fplus_frac_O": "fplus_frac_O",
    "fminus_ratio_OC": "fminus_ratio_OC",
    "fminus_frac_O": "fminus_frac_O",
    "dual_ratio_OC": "dual_ratio_OC",
}

# QTAIM
qtaim = {
    "bader_sumO": "bader_sumO_n",
    "bader_sumC": "bader_sumC_n",
    "bader_sumH": "bader_sumH_n",
    "DI_sumO": "DI_sumO_n",
    "DI_sumC": "DI_sumC_n",
    "LI_sumO": "LI_sumO_n",
    "LI_sumC": "LI_sumC_n",
}

# Neutral thermodynamic
thermo_neutral = {
    "ZPE": "ZPE_n_kj",
    "TS_vib": "TS_vib_n_kj",
}

# Orbital composition
orbital = {
    "LUMO_O_frac": "LUMO_O_frac",
    "LUMO_C_frac": "LUMO_C_frac",
}

# Politzer
politzer_desc = {
    "EA_Koopmans": "EA_Koopmans_kJ",
    "IP_Koopmans": "IP_Koopmans_kJ",
    "I_S_min": "I_S_min_eV",
    "V_S_min": "V_S_min_eV",
}

all_desc_groups = [
    ("global_cdft", global_cdft),
    ("fukui_sums", fukui_sums),
    ("fukui_extrema", fukui_extrema),
    ("fukui_ratios", fukui_ratios),
    ("qtaim", qtaim),
    ("thermo_neutral", thermo_neutral),
    ("orbital", orbital),
    ("politzer", politzer_desc),
]

available = {}
for group_name, group_dict in all_desc_groups:
    for name, col in group_dict.items():
        if col in data.columns and data[col].notna().sum() >= 5:
            available[name] = {"col": col, "group": group_name,
                               "n_valid": int(data[col].notna().sum())}

print(f"\n  Available descriptors: {len(available)}")
for gname in ["global_cdft", "fukui_sums", "fukui_extrema", "fukui_ratios",
               "qtaim", "thermo_neutral", "orbital", "politzer"]:
    descs = [k for k, v in available.items() if v["group"] == gname]
    print(f"    {gname}: {len(descs)} — {descs}")


# ============================================================
# 7. EVALUATION FUNCTIONS
# ============================================================

def eval_linear(x, y, min_n=5):
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
    p = 1
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    return {"R2": r2, "R2_adj": r2_adj, "MAE": mae, "n": n,
            "slope": reg.coef_[0], "intercept": reg.intercept_}


def eval_constant(y, min_n=5):
    mask = np.isfinite(y)
    n = mask.sum()
    if n < min_n:
        return None
    yv = y[mask]
    mean_val = np.mean(yv)
    mae = mean_absolute_error(yv, np.full_like(yv, mean_val))
    return {"R2": 0.0, "R2_adj": 0.0, "MAE": mae, "n": n, "constant": mean_val}


def eval_2desc_linear(x1, x2, y, min_n=6):
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


TRANSFORMS = [
    ("x", lambda x: x),
    ("x²", lambda x: x**2),
    ("|x|", lambda x: np.abs(x)),
    ("√|x|", lambda x: np.sqrt(np.abs(x))),
    ("1/x", lambda x: np.where(np.abs(x) > 1e-10, 1.0 / x, np.nan)),
]


# ============================================================
# 8. RUN SEARCH ON RAPPORT-15
# ============================================================

data_r15 = data[data["molecule"].isin(RAPPORT15)].copy()
print(f"\n  Rapport-15 molecules found: {len(data_r15)}")

targets = {
    "ddU_red": ("ddU_red_kJ", 1.7, "ΔΔU reduction"),
    "ddU_ox": ("ddU_ox_kJ", 0.4, "ΔΔU oxidation"),
    "TdS_red": ("TdS_red_kJ", 1.5, "TΔS reduction"),
    "TdS_ox": ("TdS_ox_kJ", 0.9, "TΔS oxidation"),
    "ddU_minus_TdS_red": ("ddU_minus_TdS_red", 3.2, "ΔΔU - TΔS reduction"),
    "ddU_minus_TdS_ox": ("ddU_minus_TdS_ox", 0.7, "ΔΔU - TΔS oxidation"),
}

print("\n" + "=" * 80)
print("  EXHAUSTIVE SEARCH (Rapport-15, neutral-state descriptors only)")
print("=" * 80)

all_results = {}

for target_key, (target_col, baseline_mae, target_label) in targets.items():
    print(f"\n--- {target_label} ---")
    y = data_r15[target_col].values
    results = []

    # Baseline
    cst = eval_constant(y)
    if cst:
        results.append({"label": f"Constant ({cst['constant']:.2f})",
                         "MAE": cst["MAE"], "R2": 0.0, "R2_adj": 0.0,
                         "n": cst["n"], "n_params": 0, "category": "baseline"})

    # Single descriptor + transforms
    for dname, dinfo in available.items():
        x = data_r15[dinfo["col"]].values.astype(float)
        for tname, tfunc in TRANSFORMS:
            xt = tfunc(x)
            r = eval_linear(xt, y)
            if r and r["n"] >= 5:
                lbl = f"{dname}" if tname == "x" else f"{tname}({dname})"
                results.append({"label": lbl, "MAE": r["MAE"], "R2": r["R2"],
                                "R2_adj": r["R2_adj"], "n": r["n"], "n_params": 1,
                                "category": "1D", "slope": r["slope"], "intercept": r["intercept"]})

    # Two-descriptor linear
    desc_names = list(available.keys())
    for i, d1 in enumerate(desc_names):
        for d2 in desc_names[i + 1:]:
            x1 = data_r15[available[d1]["col"]].values.astype(float)
            x2 = data_r15[available[d2]["col"]].values.astype(float)
            r = eval_2desc_linear(x1, x2, y, min_n=6)
            if r:
                results.append({"label": f"({d1}, {d2})", "MAE": r["MAE"],
                                "R2": r["R2"], "R2_adj": r["R2_adj"],
                                "n": r["n"], "n_params": 2, "category": "2D"})

    # Product descriptors
    for i, d1 in enumerate(desc_names):
        for d2 in desc_names[i + 1:]:
            x1 = data_r15[available[d1]["col"]].values.astype(float)
            x2 = data_r15[available[d2]["col"]].values.astype(float)
            xp = x1 * x2
            r = eval_linear(xp, y)
            if r and r["n"] >= 5:
                results.append({"label": f"{d1}*{d2}", "MAE": r["MAE"],
                                "R2": r["R2"], "R2_adj": r["R2_adj"],
                                "n": r["n"], "n_params": 1, "category": "product"})

    df = pd.DataFrame(results).sort_values("MAE")
    all_results[target_key] = df

    # Print top 30
    print(f"\n  Constant baseline: MAE = {cst['MAE']:.3f} (n={cst['n']})")
    n_beat = (df["MAE"] < baseline_mae).sum()
    if n_beat > 0:
        print(f"  *** {n_beat} models beat baseline MAE={baseline_mae} ***")
    print(f"\n  {'Rk':<4} {'MAE':>6} {'R²':>6} {'R²adj':>6} {'n':>3} {'#p':>3} {'Cat':<8} {'Label'}")
    print(f"  {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*3} {'-'*3} {'-'*8} {'-'*55}")
    for i, (_, row) in enumerate(df.head(30).iterrows()):
        flag = " ***" if row["MAE"] < baseline_mae else ""
        print(f"  {i+1:<4} {row['MAE']:6.3f} {row['R2']:6.3f} {row['R2_adj']:6.3f} "
              f"{row['n']:3.0f} {row['n_params']:3.0f} {row['category']:<8} {row['label']}{flag}")

    # Best per category
    print(f"\n  --- Best per category ---")
    for cat in ["baseline", "1D", "2D", "product"]:
        sub = df[df["category"] == cat].sort_values("MAE")
        if len(sub) > 0:
            b = sub.iloc[0]
            beat = " BEATS" if b["MAE"] < baseline_mae else ""
            print(f"    {cat:<8}: MAE={b['MAE']:.3f} R²adj={b['R2_adj']:.3f} n={b['n']:.0f}{beat}  |  {b['label']}")

    print(f"  Total models: {len(df)}")


# ============================================================
# 9. SYMMETRIC RED/OX MODEL SEARCH
# ============================================================

print(f"\n\n{'#' * 100}")
print("#  SYMMETRIC RED/OX MODEL SEARCH")
print("#  Looking for descriptors that work for BOTH reduction and oxidation")
print("#  with symmetric or mirror-image structure (f+ for red ↔ f- for ox)")
print(f"{'#' * 100}")

# Define symmetric descriptor pairs: (red-relevant, ox-relevant)
# For reduction: f+ (nucleophilic) describes where electron is added
# For oxidation: f- (electrophilic) describes where electron is removed
symmetric_pairs = [
    # (name_for_red, col_for_red, name_for_ox, col_for_ox, label)
    # f+ ↔ f-
    ("fplus_sumO", "fplus_sumO_n", "fminus_sumO", "fminus_sumO_n", "f±_sumO"),
    ("fplus_maxO", "fplus_maxO", "fminus_maxO", "fminus_maxO", "f±_maxO"),
    ("fplus_sumC", "fplus_sumC_n", "fminus_sumC", "fminus_sumC_n", "f±_sumC"),
    ("fplus_maxC", "fplus_maxC", "fminus_maxC", "fminus_maxC", "f±_maxC"),
    ("fplus_ratio_OC", "fplus_ratio_OC", "fminus_ratio_OC", "fminus_ratio_OC", "f±_ratio_OC"),
    ("fplus_frac_O", "fplus_frac_O", "fminus_frac_O", "fminus_frac_O", "f±_frac_O"),
    ("fplus_spreadO", "fplus_spreadO", "fminus_spreadO", "fminus_spreadO", "f±_spreadO"),
    ("fplus_minO", "fplus_minO", "fminus_minO", "fminus_minO", "f±_minO"),
]

# Descriptors that are the SAME for both red and ox (global properties)
same_for_both = [
    ("eta_vert", "eta_vert"),
    ("mu_vert", "mu_vert"),
    ("omega_vert", "omega_vert"),
    ("eta_cdft", "eta_n"),
    ("mu_cdft", "mu_n"),
    ("omega_cdft", "omega_n"),
    ("gap", "gap_n"),
    ("HOMO", "HOMO_n"),
    ("LUMO", "LUMO_n"),
    ("dual_sumO", "dual_sumO_n"),
    ("dual_sumC", "dual_sumC_n"),
    ("dual_maxO", "dual_maxO"),
    ("dual_minO", "dual_minO"),
    ("dual_spread", "dual_spread"),
    ("f0_sumO", "f0_sumO_n"),
    ("f0_maxO", "f0_maxO"),
    ("bader_sumO", "bader_sumO_n"),
    ("bader_sumC", "bader_sumC_n"),
    ("bader_sumH", "bader_sumH_n"),
    ("DI_sumO", "DI_sumO_n"),
    ("DI_sumC", "DI_sumC_n"),
    ("LI_sumO", "LI_sumO_n"),
    ("LI_sumC", "LI_sumC_n"),
    ("ZPE", "ZPE_n_kj"),
    ("TS_vib", "TS_vib_n_kj"),
    ("LUMO_O_frac", "LUMO_O_frac"),
    ("LUMO_C_frac", "LUMO_C_frac"),
    ("EA_Koopmans", "EA_Koopmans_kJ"),
    ("IP_Koopmans", "IP_Koopmans_kJ"),
    ("V_S_min", "V_S_min_eV"),
]


def run_symmetric_search(target_red_col, target_ox_col, label, baseline_red, baseline_ox):
    """Search for models with same structure for red and ox."""
    y_red = data_r15[target_red_col].values
    y_ox = data_r15[target_ox_col].values
    results = []

    # --- A) Same descriptor for both ---
    for dname, dcol in same_for_both:
        if dcol not in data_r15.columns or data_r15[dcol].isna().sum() > 10:
            continue
        x = data_r15[dcol].values.astype(float)
        for tname, tfunc in TRANSFORMS:
            xt = tfunc(x)
            r_red = eval_linear(xt, y_red)
            r_ox = eval_linear(xt, y_ox)
            if r_red and r_ox and r_red["n"] >= 5 and r_ox["n"] >= 5:
                lbl = f"{dname}" if tname == "x" else f"{tname}({dname})"
                results.append({
                    "label": f"SAME: {lbl}",
                    "MAE_red": r_red["MAE"], "R2_red": r_red["R2_adj"],
                    "MAE_ox": r_ox["MAE"], "R2_ox": r_ox["R2_adj"],
                    "n_red": r_red["n"], "n_ox": r_ox["n"],
                    "n_params": 1, "category": "same_1D",
                    "slope_red": r_red["slope"], "intercept_red": r_red["intercept"],
                    "slope_ox": r_ox["slope"], "intercept_ox": r_ox["intercept"],
                })

    # --- B) Mirror descriptors: f+ for red, f- for ox ---
    for dname_r, dcol_r, dname_o, dcol_o, pair_label in symmetric_pairs:
        if dcol_r not in data_r15.columns or dcol_o not in data_r15.columns:
            continue
        x_red = data_r15[dcol_r].values.astype(float)
        x_ox = data_r15[dcol_o].values.astype(float)
        for tname, tfunc in TRANSFORMS:
            xt_red = tfunc(x_red)
            xt_ox = tfunc(x_ox)
            r_red = eval_linear(xt_red, y_red)
            r_ox = eval_linear(xt_ox, y_ox)
            if r_red and r_ox and r_red["n"] >= 5 and r_ox["n"] >= 5:
                lbl = f"{pair_label}" if tname == "x" else f"{tname}({pair_label})"
                results.append({
                    "label": f"MIRROR: {lbl}",
                    "MAE_red": r_red["MAE"], "R2_red": r_red["R2_adj"],
                    "MAE_ox": r_ox["MAE"], "R2_ox": r_ox["R2_adj"],
                    "n_red": r_red["n"], "n_ox": r_ox["n"],
                    "n_params": 1, "category": "mirror_1D",
                })

    # --- C) 2D: same global + mirror Fukui ---
    for gname, gcol in same_for_both:
        if gcol not in data_r15.columns or data_r15[gcol].isna().sum() > 10:
            continue
        xg = data_r15[gcol].values.astype(float)

        # Same global + same global
        for gname2, gcol2 in same_for_both:
            if gname2 <= gname:
                continue
            if gcol2 not in data_r15.columns or data_r15[gcol2].isna().sum() > 10:
                continue
            xg2 = data_r15[gcol2].values.astype(float)
            r_red = eval_2desc_linear(xg, xg2, y_red, min_n=6)
            r_ox = eval_2desc_linear(xg, xg2, y_ox, min_n=6)
            if r_red and r_ox:
                results.append({
                    "label": f"SAME 2D: ({gname}, {gname2})",
                    "MAE_red": r_red["MAE"], "R2_red": r_red["R2_adj"],
                    "MAE_ox": r_ox["MAE"], "R2_ox": r_ox["R2_adj"],
                    "n_red": r_red["n"], "n_ox": r_ox["n"],
                    "n_params": 2, "category": "same_2D",
                })

        # Global + mirror Fukui
        for dname_r, dcol_r, dname_o, dcol_o, pair_label in symmetric_pairs:
            if dcol_r not in data_r15.columns or dcol_o not in data_r15.columns:
                continue
            x_red = data_r15[dcol_r].values.astype(float)
            x_ox = data_r15[dcol_o].values.astype(float)
            r_red = eval_2desc_linear(xg, x_red, y_red, min_n=6)
            r_ox = eval_2desc_linear(xg, x_ox, y_ox, min_n=6)
            if r_red and r_ox:
                results.append({
                    "label": f"MIRROR 2D: ({gname}, {pair_label})",
                    "MAE_red": r_red["MAE"], "R2_red": r_red["R2_adj"],
                    "MAE_ox": r_ox["MAE"], "R2_ox": r_ox["R2_adj"],
                    "n_red": r_red["n"], "n_ox": r_ox["n"],
                    "n_params": 2, "category": "mirror_2D",
                })

    # --- D) Product: global * mirror Fukui ---
    for gname, gcol in same_for_both:
        if gcol not in data_r15.columns or data_r15[gcol].isna().sum() > 10:
            continue
        xg = data_r15[gcol].values.astype(float)
        for dname_r, dcol_r, dname_o, dcol_o, pair_label in symmetric_pairs:
            if dcol_r not in data_r15.columns or dcol_o not in data_r15.columns:
                continue
            x_red = data_r15[dcol_r].values.astype(float)
            x_ox = data_r15[dcol_o].values.astype(float)
            r_red = eval_linear(xg * x_red, y_red)
            r_ox = eval_linear(xg * x_ox, y_ox)
            if r_red and r_ox and r_red["n"] >= 5 and r_ox["n"] >= 5:
                results.append({
                    "label": f"MIRROR PROD: {gname}*{pair_label}",
                    "MAE_red": r_red["MAE"], "R2_red": r_red["R2_adj"],
                    "MAE_ox": r_ox["MAE"], "R2_ox": r_ox["R2_adj"],
                    "n_red": r_red["n"], "n_ox": r_ox["n"],
                    "n_params": 1, "category": "mirror_product",
                })

    if not results:
        print(f"\n  No symmetric models found for {label}")
        return None

    df = pd.DataFrame(results)
    df["MAE_avg"] = (df["MAE_red"] + df["MAE_ox"]) / 2
    df = df.sort_values("MAE_avg")

    print(f"\n  {'=' * 110}")
    print(f"  SYMMETRIC MODELS: {label}")
    print(f"  Baselines: MAE_red={baseline_red:.1f}, MAE_ox={baseline_ox:.1f}")
    print(f"  {'=' * 110}")

    # Models that beat BOTH baselines
    both_beat = df[(df["MAE_red"] < baseline_red) & (df["MAE_ox"] < baseline_ox)]
    if len(both_beat) > 0:
        print(f"\n  *** {len(both_beat)} models beat BOTH baselines! ***\n")
        print(f"  {'Rk':<4} {'MAE_r':>6} {'MAE_o':>6} {'avg':>6} {'R²r':>5} {'R²o':>5} {'n_r':>3} {'n_o':>3} {'#p':>3} {'Cat':<14} {'Label'}")
        print(f"  {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*3} {'-'*3} {'-'*3} {'-'*14} {'-'*50}")
        for i, (_, row) in enumerate(both_beat.head(20).iterrows()):
            print(f"  {i+1:<4} {row['MAE_red']:6.3f} {row['MAE_ox']:6.3f} {row['MAE_avg']:6.3f} "
                  f"{row['R2_red']:5.3f} {row['R2_ox']:5.3f} {row['n_red']:3.0f} {row['n_ox']:3.0f} "
                  f"{row['n_params']:3.0f} {row['category']:<14} {row['label']}")

    print(f"\n  --- Top 30 by average MAE ---\n")
    print(f"  {'Rk':<4} {'MAE_r':>6} {'MAE_o':>6} {'avg':>6} {'R²r':>5} {'R²o':>5} {'n_r':>3} {'n_o':>3} {'#p':>3} {'Cat':<14} {'Label'}")
    print(f"  {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*3} {'-'*3} {'-'*3} {'-'*14} {'-'*50}")
    for i, (_, row) in enumerate(df.head(30).iterrows()):
        br = " *" if row["MAE_red"] < baseline_red else ""
        bo = " *" if row["MAE_ox"] < baseline_ox else ""
        print(f"  {i+1:<4} {row['MAE_red']:6.3f}{br:>2} {row['MAE_ox']:6.3f}{bo:>2} {row['MAE_avg']:6.3f} "
              f"{row['R2_red']:5.3f} {row['R2_ox']:5.3f} {row['n_red']:3.0f} {row['n_ox']:3.0f} "
              f"{row['n_params']:3.0f} {row['category']:<14} {row['label']}")

    # Best per category
    print(f"\n  --- Best per category ---")
    for cat in ["same_1D", "mirror_1D", "same_2D", "mirror_2D", "mirror_product"]:
        sub = df[df["category"] == cat].sort_values("MAE_avg")
        if len(sub) > 0:
            b = sub.iloc[0]
            print(f"    {cat:<16}: avg MAE={b['MAE_avg']:.3f}  (red={b['MAE_red']:.3f} ox={b['MAE_ox']:.3f})  |  {b['label']}")

    return df


# Run symmetric searches
print("\n")
df_sym_ddu = run_symmetric_search("ddU_red_kJ", "ddU_ox_kJ", "ΔΔU (red & ox)", 1.7, 0.4)
df_sym_tds = run_symmetric_search("TdS_red_kJ", "TdS_ox_kJ", "TΔS (red & ox)", 1.5, 0.9)
df_sym_combo = run_symmetric_search("ddU_minus_TdS_red", "ddU_minus_TdS_ox",
                                     "ΔΔU - TΔS (red & ox)", 3.2, 0.7)


# ============================================================
# 10. DETAILED ANALYSIS OF BEST SYMMETRIC MODELS
# ============================================================

print(f"\n\n{'#' * 100}")
print("#  DETAILED BEST SYMMETRIC MODELS")
print(f"{'#' * 100}")


def print_model_details(desc_name, desc_col, y_col, y_label, data_sub):
    """Print detailed regression for a single descriptor."""
    x = data_sub[desc_col].values.astype(float)
    y = data_sub[y_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        print(f"  Not enough data for {desc_name} vs {y_label}")
        return
    X = x[mask].reshape(-1, 1)
    yv = y[mask]
    mols = data_sub["molecule"].values[mask]
    reg = LinearRegression().fit(X, yv)
    y_pred = reg.predict(X)
    r2 = r2_score(yv, y_pred)
    mae = mean_absolute_error(yv, y_pred)
    n = mask.sum()
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2

    print(f"\n  {y_label} = {reg.intercept_:.4f} + {reg.coef_[0]:.4f} * {desc_name}")
    print(f"  R² = {r2:.4f}, R²_adj = {r2_adj:.4f}, MAE = {mae:.3f} kJ/mol, n = {n}")
    print(f"  {'Molecule':<10} {'Desc':>10} {'Real':>8} {'Pred':>8} {'Error':>8}")
    for mol, xv, yr, yp in zip(mols, X.flatten(), yv, y_pred):
        print(f"  {mol:<10} {xv:10.4f} {yr:8.3f} {yp:8.3f} {yr-yp:8.3f}")


# Best 1D symmetric: eta for DDU
print("\n--- η (hardness) for ΔΔU ---")
if "eta_vert" in available:
    print_model_details("eta_vert", "eta_vert", "ddU_red_kJ", "ΔΔU_red", data_r15)
    print_model_details("eta_vert", "eta_vert", "ddU_ox_kJ", "ΔΔU_ox", data_r15)

print("\n--- omega (electrophilicity) for ΔΔU ---")
if "omega_vert" in available:
    print_model_details("omega_vert", "omega_vert", "ddU_red_kJ", "ΔΔU_red", data_r15)
    print_model_details("omega_vert", "omega_vert", "ddU_ox_kJ", "ΔΔU_ox", data_r15)

print("\n--- bader_sumO for TΔS ---")
if "bader_sumO" in available:
    print_model_details("bader_sumO", "bader_sumO_n", "TdS_red_kJ", "TΔS_red", data_r15)
    print_model_details("bader_sumO", "bader_sumO_n", "TdS_ox_kJ", "TΔS_ox", data_r15)

# Mirror Fukui for ΔΔU
print("\n--- f+_maxO (red) / f-_maxO (ox) for ΔΔU ---")
if "fplus_maxO" in available and "fminus_maxO" in available:
    print_model_details("fplus_maxO", "fplus_maxO", "ddU_red_kJ", "ΔΔU_red", data_r15)
    print_model_details("fminus_maxO", "fminus_maxO", "ddU_ox_kJ", "ΔΔU_ox", data_r15)

# Mirror Fukui for TΔS
print("\n--- f+_sumO (red) / f-_sumO (ox) for TΔS ---")
if "fplus_sumO" in available and "fminus_sumO" in available:
    print_model_details("fplus_sumO", "fplus_sumO_n", "TdS_red_kJ", "TΔS_red", data_r15)
    print_model_details("fminus_sumO", "fminus_sumO_n", "TdS_ox_kJ", "TΔS_ox", data_r15)

# DI_sumO
print("\n--- DI_sumO for ΔΔU and TΔS ---")
if "DI_sumO" in available:
    for target_col, label in [("ddU_red_kJ", "ΔΔU_red"), ("ddU_ox_kJ", "ΔΔU_ox"),
                                ("TdS_red_kJ", "TΔS_red"), ("TdS_ox_kJ", "TΔS_ox")]:
        print_model_details("DI_sumO", "DI_sumO_n", target_col, label, data_r15)

print("\n  Done.")
