#!/usr/bin/env python3
"""
Validate improved ΔΔU and TΔS models by predicting ΔG° and comparing
with the current constant-model approach from the rapport.

ΔG decomposition:
  ΔG_ox  =  IP + E_relax^ox + ΔΔU_ox + TΔS_ox
  ΔG_red = -EA + E_relax^red + ΔΔU_red + TΔS_red

Components and their models:
  EA/IP     : Koopmans (linear fit on HOMO/LUMO)
  E_relax   : LVC2 (linear calibration)
  ΔΔU, TΔS  : OLD = constant  vs  NEW = CDFT/QTAIM descriptors
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import linregress
import glob
import json
import warnings
warnings.filterwarnings("ignore")

BASE = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5"
DEA = {"Br-DA", "Br-HMC", "I-DA", "I-HMC"}
RAPPORT15 = ["Al-CA", "CA", "Cl-DA", "Cl-HMC", "DA", "DHI", "F-DA", "F-HMC",
             "H2C-CA", "HMC", "HO-CA", "L-DA", "O-DHI", "P-DHI", "S-DHI"]

# ============================================================
# 1. LOAD ALL DATA
# ============================================================

print("=" * 90)
print("  LOADING DATA")
print("=" * 90)

# --- Redox decomposition per molecule (Boltzmann-weighted) ---
files = glob.glob(f"{BASE}/data/Acetonitrile/*/redox/results/*_redox_conformers.csv")
rows = []
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
    rows.append(row)
redox_all = pd.DataFrame(rows)

# --- LVC2 summary ---
lvc = pd.read_csv(f"{BASE}/rapport/data/lvc_summary.csv")
lvc = lvc.rename(columns={"molecule_name": "molecule"})

# --- LVC2 calibration ---
with open(f"{BASE}/rapport/data/lvc2_calibration.json") as f:
    lvc2_calib = json.load(f)

# --- Descriptors per molecule (HOMO/LUMO for Koopmans, LUMO_O_frac) ---
desc_mol = pd.read_csv(f"{BASE}/rapport/data/descriptors_per_molecule.csv")

# --- predict_dg (Fukui, CDFT, QTAIM, thermo from neutral) ---
all_cols = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", nrows=0).columns.tolist()
keep = ["molecule", "conformer", "weight"]
for c in ["ZPE_n_kj", "TS_vib_n_kj", "HOMO_n", "LUMO_n", "EA_vert_kj", "IP_vert_kj",
           "mu_n", "eta_n", "omega_n", "gap_n",
           "fplus_sumO_n", "fminus_sumO_n", "dual_sumO_n", "f0_sumO_n",
           "fplus_sumC_n", "fminus_sumC_n",
           "bader_sumO_n", "bader_sumC_n", "bader_sumH_n",
           "DI_sumO_n", "DI_sumC_n", "LI_sumO_n", "LI_sumC_n"]:
    if c in all_cols:
        keep.append(c)
# Per-atom Fukui for max/min
for prefix in ["fplus_", "fminus_", "f0_"]:
    for c in all_cols:
        if c.startswith(prefix) and c.endswith("_n") and "sum" not in c:
            keep.append(c)
keep = list(dict.fromkeys(keep))
pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv", usecols=keep)
pdg = pdg[~pdg["molecule"].isin(DEA)].copy()

# Boltzmann-average predict_dg
numeric_pdg = [c for c in pdg.columns if c not in ["molecule", "conformer", "weight"]]


def boltzmann_avg(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}
    for col in numeric_pdg:
        vals = group[col].values
        result[col] = np.sum(w * vals) if np.all(np.isfinite(vals)) else np.nan
    return pd.Series(result)


bw_pdg = pdg.groupby("molecule").apply(boltzmann_avg).reset_index()

# Compute per-atom extrema
per_atom_O = [c for c in pdg.columns if "_O" in c and c.endswith("_n") and "sum" not in c]
fplus_O = [c for c in per_atom_O if c.startswith("fplus_")]
fminus_O = [c for c in per_atom_O if c.startswith("fminus_")]
f0_O = [c for c in per_atom_O if c.startswith("f0_")]


def compute_extrema(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}
    for atoms, label in [(fplus_O, "fplus_maxO"), (fminus_O, "fminus_maxO"), (f0_O, "f0_maxO")]:
        if atoms:
            result[label] = np.sum(w * group[atoms].max(axis=1).values)
    for atoms, label in [(fplus_O, "fplus_minO"), (fminus_O, "fminus_minO")]:
        if atoms:
            result[label] = np.sum(w * group[atoms].min(axis=1).values)
    return pd.Series(result)


extrema = pdg.groupby("molecule").apply(compute_extrema).reset_index()
bw_pdg = bw_pdg.merge(extrema, on="molecule", how="left")

# --- Politzer (Koopmans from orbitals) ---
politzer = pd.read_csv(f"{BASE}/rapport/data/politzer_per_molecule.csv")
politzer = politzer.rename(columns={"molecule_name": "molecule"})

# ============================================================
# 2. BUILD MASTER TABLE FOR RAPPORT-15
# ============================================================

data = redox_all[redox_all["molecule"].isin(RAPPORT15)].copy()
data = data.merge(lvc[["molecule", "lvc2_red", "lvc2_ox", "real_red", "real_ox"]], on="molecule", how="left")
data = data.merge(desc_mol[["molecule", "LUMO_O_frac", "LUMO_eV", "gap_eV"]], on="molecule", how="left")
data = data.merge(bw_pdg, on="molecule", how="left")
data = data.merge(politzer[["molecule", "EA_Koopmans_kJ", "IP_Koopmans_kJ"]], on="molecule", how="left")

# Vertical CDFT from redox data
data["eta_vert"] = (data["IP_kJ"] - data["EA_kJ"]) / 2
data["mu_vert"] = -(data["IP_kJ"] + data["EA_kJ"]) / 2
data["omega_vert"] = data["mu_vert"]**2 / (2 * data["eta_vert"])

# Sort by molecule name for consistent ordering
data = data.sort_values("molecule").reset_index(drop=True)

print(f"  Molecules: {len(data)}")
print(f"  With predict_dg descriptors: {data['eta_n'].notna().sum()}")
print(f"  Missing predict_dg: {data[data['eta_n'].isna()]['molecule'].tolist()}")

# ============================================================
# 3. COMPONENT MODELS
# ============================================================

print("\n" + "=" * 90)
print("  COMPONENT MODELS")
print("=" * 90)

# --- 3a. EA/IP Koopmans ---
# Use Ridge M2 from the rapport: |EA| = -0.987*ε_L + 0.014*ε_H + 43.36
# But simpler: M1 Koopmans for EA, M2 for IP
# Actually, reproduce exactly what generate_thermodynamic_models.py does:
# Simple linregress on HOMO_eV -> IP and LUMO_eV -> EA

# HOMO/LUMO are in the desc_mol file - but those are in Hartree labeled as eV!
# Convert to eV
EH_TO_EV = 27.211386245988

# From predict_dg, HOMO_n and LUMO_n are in Hartree
# desc_mol has LUMO_eV which is actually Hartree too (see rapport code comment)
# Let's use the politzer data which already has Koopmans in kJ

# Direct regression: IP_DFT vs IP_Koopmans_kJ and EA_DFT vs EA_Koopmans_kJ
mask_ip = data["IP_Koopmans_kJ"].notna()
if mask_ip.sum() >= 5:
    sl_ip, int_ip, r_ip, _, _ = linregress(data.loc[mask_ip, "IP_Koopmans_kJ"],
                                             data.loc[mask_ip, "IP_kJ"])
    data.loc[mask_ip, "IP_pred"] = sl_ip * data.loc[mask_ip, "IP_Koopmans_kJ"] + int_ip
    ip_mae = mean_absolute_error(data.loc[mask_ip, "IP_kJ"], data.loc[mask_ip, "IP_pred"])
    print(f"\n  IP Koopmans: slope={sl_ip:.4f}, int={int_ip:.2f}, R²={r_ip**2:.4f}, MAE={ip_mae:.2f} kJ/mol (n={mask_ip.sum()})")

mask_ea = data["EA_Koopmans_kJ"].notna()
if mask_ea.sum() >= 5:
    sl_ea, int_ea, r_ea, _, _ = linregress(data.loc[mask_ea, "EA_Koopmans_kJ"],
                                             data.loc[mask_ea, "EA_kJ"])
    data.loc[mask_ea, "EA_pred"] = sl_ea * data.loc[mask_ea, "EA_Koopmans_kJ"] + int_ea
    ea_mae = mean_absolute_error(data.loc[mask_ea, "EA_kJ"], data.loc[mask_ea, "EA_pred"])
    print(f"  EA Koopmans: slope={sl_ea:.4f}, int={int_ea:.2f}, R²={r_ea**2:.4f}, MAE={ea_mae:.2f} kJ/mol (n={mask_ea.sum()})")

# For molecules without Politzer, use HOMO_n/LUMO_n from predict_dg
for idx in data.index:
    if pd.isna(data.loc[idx, "IP_pred"]) and pd.notna(data.loc[idx, "HOMO_n"]):
        homo_eV = data.loc[idx, "HOMO_n"] * EH_TO_EV
        lumo_eV = data.loc[idx, "LUMO_n"] * EH_TO_EV
        # Direct Koopmans with eV->kJ conversion
        ip_koop = -homo_eV * 96.485
        ea_koop = -lumo_eV * 96.485
        data.loc[idx, "IP_pred"] = sl_ip * ip_koop + int_ip if mask_ip.sum() >= 5 else ip_koop
        data.loc[idx, "EA_pred"] = sl_ea * ea_koop + int_ea if mask_ea.sum() >= 5 else ea_koop

# --- 3b. E_relax LVC2 (with calibration) ---
# Calibration: E_real = slope * LVC2 + intercept
cal_red = lvc2_calib["red"]
cal_ox = lvc2_calib["ox"]
data["Erelax_red_pred"] = cal_red["slope"] * data["lvc2_red"] + cal_red["intercept"]
data["Erelax_ox_pred"] = cal_ox["slope"] * data["lvc2_ox"] + cal_ox["intercept"]

erelax_red_mae = mean_absolute_error(data["real_red"].dropna(), data["Erelax_red_pred"].dropna())
erelax_ox_mae = mean_absolute_error(data["real_ox"].dropna(), data["Erelax_ox_pred"].dropna())
print(f"\n  E_relax LVC2 calibrated: MAE_red={erelax_red_mae:.2f}, MAE_ox={erelax_ox_mae:.2f} kJ/mol")

# --- 3c. ΔΔU and TΔS models ---

# Model A: BASELINE (constant)
ddU_red_cst = data["ddU_red_kJ"].mean()
ddU_ox_cst = data["ddU_ox_kJ"].mean()
TdS_red_cst = data["TdS_red_kJ"].mean()
TdS_ox_cst = data["TdS_ox_kJ"].mean()
print(f"\n  Constants: ΔΔU_red={ddU_red_cst:.2f}, ΔΔU_ox={ddU_ox_cst:.2f}, "
      f"TΔS_red={TdS_red_cst:.2f}, TΔS_ox={TdS_ox_cst:.2f}")


def fit_linear(x, y, label=""):
    """Fit y = a*x + b, return (slope, intercept, R², MAE, predictions)."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 5:
        return None
    sl, inter, r, _, _ = linregress(x[mask], y[mask])
    pred = sl * x + inter
    pred[~mask] = np.nan
    mae = mean_absolute_error(y[mask], pred[mask])
    r2 = r**2
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2)
    print(f"  {label}: y = {sl:.4f}*x + {inter:.4f}, R²adj={r2_adj:.3f}, MAE={mae:.3f} (n={n})")
    return {"slope": sl, "intercept": inter, "R2_adj": r2_adj, "MAE": mae, "pred": pred}


def fit_2d_linear(x1, x2, y, label=""):
    """Fit y = a*x1 + b*x2 + c."""
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    n = mask.sum()
    if n < 6:
        return None
    X = np.column_stack([x1[mask], x2[mask]])
    yv = y[mask]
    reg = LinearRegression().fit(X, yv)
    pred_full = np.full_like(y, np.nan, dtype=float)
    pred_full[mask] = reg.predict(X)
    mae = mean_absolute_error(yv, pred_full[mask])
    r2 = r2_score(yv, pred_full[mask])
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - 3)
    print(f"  {label}: y = {reg.coef_[0]:.4f}*x1 + {reg.coef_[1]:.4f}*x2 + {reg.intercept_:.4f}, "
          f"R²adj={r2_adj:.3f}, MAE={mae:.3f} (n={n})")
    return {"coef": reg.coef_, "intercept": reg.intercept_, "R2_adj": r2_adj,
            "MAE": mae, "pred": pred_full}


# ============================================================
# Model B: η_vert for ΔΔU (same for red & ox) + bader_sumO for TΔS
# ============================================================

print("\n--- Model B: η_vert for ΔΔU, bader_sumO + μ_vert for TΔS ---")

eta = data["eta_vert"].values
bader_O = data["bader_sumO_n"].values
mu = data["mu_vert"].values

m_ddU_red_eta = fit_linear(eta, data["ddU_red_kJ"].values, "ΔΔU_red ~ η_vert")
m_ddU_ox_eta = fit_linear(eta, data["ddU_ox_kJ"].values, "ΔΔU_ox ~ η_vert")

m_TdS_red_badermu = fit_2d_linear(bader_O, mu, data["TdS_red_kJ"].values, "TΔS_red ~ (bader_sumO, μ_vert)")
m_TdS_ox_badermu = fit_2d_linear(bader_O, mu, data["TdS_ox_kJ"].values, "TΔS_ox ~ (bader_sumO, μ_vert)")

# ============================================================
# Model C: (DI_sumO, ω_cdft) for ΔΔU + (LI_sumC, f±_minO) mirror for TΔS
# ============================================================

print("\n--- Model C: (DI_sumO, ω_cdft) for ΔΔU, mirror (LI_sumC, f±_minO) for TΔS ---")

DI_O = data["DI_sumO_n"].values
omega_cdft = data["omega_n"].values
LI_C = data["LI_sumC_n"].values
fplus_minO = data["fplus_minO"].values if "fplus_minO" in data.columns else np.full(len(data), np.nan)
fminus_minO = data["fminus_minO"].values if "fminus_minO" in data.columns else np.full(len(data), np.nan)

m_ddU_red_DI_omega = fit_2d_linear(DI_O, omega_cdft, data["ddU_red_kJ"].values, "ΔΔU_red ~ (DI_sumO, ω_cdft)")
m_ddU_ox_DI_omega = fit_2d_linear(DI_O, omega_cdft, data["ddU_ox_kJ"].values, "ΔΔU_ox ~ (DI_sumO, ω_cdft)")

# Mirror: f+_minO for red, f-_minO for ox
m_TdS_red_LI_fplus = fit_2d_linear(LI_C, fplus_minO, data["TdS_red_kJ"].values, "TΔS_red ~ (LI_sumC, f+_minO)")
m_TdS_ox_LI_fminus = fit_2d_linear(LI_C, fminus_minO, data["TdS_ox_kJ"].values, "TΔS_ox ~ (LI_sumC, f-_minO)")

# ============================================================
# Model D: Combined ΔΔU-TΔS as single term: (f0_maxO, ω_vert) SAME for both
# ============================================================

print("\n--- Model D: (f0_maxO, ω_vert) for (ΔΔU - TΔS) combined ---")

f0_maxO = data["f0_maxO"].values if "f0_maxO" in data.columns else np.full(len(data), np.nan)
omega_vert = data["omega_vert"].values

combo_red = data["ddU_red_kJ"].values + data["TdS_red_kJ"].values  # ΔΔU + TΔS (note: TΔS already has sign)
combo_ox = data["ddU_ox_kJ"].values + data["TdS_ox_kJ"].values

m_combo_red = fit_2d_linear(f0_maxO, omega_vert, combo_red, "(ΔΔU+TΔS)_red ~ (f0_maxO, ω_vert)")
m_combo_ox = fit_2d_linear(f0_maxO, omega_vert, combo_ox, "(ΔΔU+TΔS)_ox ~ (f0_maxO, ω_vert)")

# ============================================================
# Model E: η_vert for ΔΔU (symmetric) + constant TΔS (conservative)
# ============================================================

print("\n--- Model E: η_vert for ΔΔU (symmetric), constant TΔS ---")
# This is the most conservative improvement: only replace ΔΔU model
# Already fitted above as m_ddU_red_eta and m_ddU_ox_eta

# ============================================================
# 4. COMPUTE ΔG PREDICTIONS
# ============================================================

print("\n" + "=" * 90)
print("  ΔG° PREDICTIONS (component-wise)")
print("=" * 90)


def compute_dg(data, ddU_red, ddU_ox, TdS_red, TdS_ox, model_name):
    """Compute predicted ΔG and compare with DFT reference."""
    dG_red_pred = -data["EA_pred"].values + data["Erelax_red_pred"].values + ddU_red + TdS_red
    dG_ox_pred = data["IP_pred"].values + data["Erelax_ox_pred"].values + ddU_ox + TdS_ox

    dG_red_real = data["dG_reduction_kJ"].values
    dG_ox_real = data["dG_oxidation_kJ"].values

    # Mask valid
    mask_red = np.isfinite(dG_red_pred) & np.isfinite(dG_red_real) & np.isfinite(data["EA_pred"].values)
    mask_ox = np.isfinite(dG_ox_pred) & np.isfinite(dG_ox_real) & np.isfinite(data["IP_pred"].values)

    results = {"model": model_name}

    if mask_red.sum() >= 5:
        err_red = dG_red_pred[mask_red] - dG_red_real[mask_red]
        mae_red = np.mean(np.abs(err_red))
        # R² of predicted vs real
        r2_red = r2_score(dG_red_real[mask_red], dG_red_pred[mask_red])
        results["MAE_red"] = mae_red
        results["R2_red"] = r2_red
        results["n_red"] = mask_red.sum()
        results["maxerr_red"] = np.max(np.abs(err_red))
        results["pred_red"] = dG_red_pred
        results["err_red"] = np.where(mask_red, dG_red_pred - dG_red_real, np.nan)
    else:
        results["MAE_red"] = np.nan

    if mask_ox.sum() >= 5:
        err_ox = dG_ox_pred[mask_ox] - dG_ox_real[mask_ox]
        mae_ox = np.mean(np.abs(err_ox))
        r2_ox = r2_score(dG_ox_real[mask_ox], dG_ox_pred[mask_ox])
        results["MAE_ox"] = mae_ox
        results["R2_ox"] = r2_ox
        results["n_ox"] = mask_ox.sum()
        results["maxerr_ox"] = np.max(np.abs(err_ox))
        results["pred_ox"] = dG_ox_pred
        results["err_ox"] = np.where(mask_ox, dG_ox_pred - dG_ox_real, np.nan)
    else:
        results["MAE_ox"] = np.nan

    return results


# --- Model A: BASELINE (constant ΔΔU, constant TΔS) ---
resA = compute_dg(data,
                   np.full(len(data), ddU_red_cst),
                   np.full(len(data), ddU_ox_cst),
                   np.full(len(data), TdS_red_cst),
                   np.full(len(data), TdS_ox_cst),
                   "A: Constant ΔΔU + Constant TΔS")

# --- Model B: η for ΔΔU + (bader_sumO, μ) for TΔS ---
ddU_red_B = m_ddU_red_eta["pred"] if m_ddU_red_eta else np.full(len(data), ddU_red_cst)
ddU_ox_B = m_ddU_ox_eta["pred"] if m_ddU_ox_eta else np.full(len(data), ddU_ox_cst)
TdS_red_B = m_TdS_red_badermu["pred"] if m_TdS_red_badermu else np.full(len(data), TdS_red_cst)
TdS_ox_B = m_TdS_ox_badermu["pred"] if m_TdS_ox_badermu else np.full(len(data), TdS_ox_cst)
# Fallback to constant for molecules without descriptors
ddU_red_B = np.where(np.isfinite(ddU_red_B), ddU_red_B, ddU_red_cst)
ddU_ox_B = np.where(np.isfinite(ddU_ox_B), ddU_ox_B, ddU_ox_cst)
TdS_red_B = np.where(np.isfinite(TdS_red_B), TdS_red_B, TdS_red_cst)
TdS_ox_B = np.where(np.isfinite(TdS_ox_B), TdS_ox_B, TdS_ox_cst)
resB = compute_dg(data, ddU_red_B, ddU_ox_B, TdS_red_B, TdS_ox_B,
                   "B: η(ΔΔU) + (bader,μ)(TΔS)")

# --- Model C: (DI_sumO, ω) for ΔΔU + mirror (LI_sumC, f±_minO) for TΔS ---
ddU_red_C = m_ddU_red_DI_omega["pred"] if m_ddU_red_DI_omega else np.full(len(data), ddU_red_cst)
ddU_ox_C = m_ddU_ox_DI_omega["pred"] if m_ddU_ox_DI_omega else np.full(len(data), ddU_ox_cst)
TdS_red_C = m_TdS_red_LI_fplus["pred"] if m_TdS_red_LI_fplus else np.full(len(data), TdS_red_cst)
TdS_ox_C = m_TdS_ox_LI_fminus["pred"] if m_TdS_ox_LI_fminus else np.full(len(data), TdS_ox_cst)
ddU_red_C = np.where(np.isfinite(ddU_red_C), ddU_red_C, ddU_red_cst)
ddU_ox_C = np.where(np.isfinite(ddU_ox_C), ddU_ox_C, ddU_ox_cst)
TdS_red_C = np.where(np.isfinite(TdS_red_C), TdS_red_C, TdS_red_cst)
TdS_ox_C = np.where(np.isfinite(TdS_ox_C), TdS_ox_C, TdS_ox_cst)
resC = compute_dg(data, ddU_red_C, ddU_ox_C, TdS_red_C, TdS_ox_C,
                   "C: (DI,ω)(ΔΔU) + mirror(LI,f±)(TΔS)")

# --- Model D: Combined (ΔΔU+TΔS) via (f0_maxO, ω_vert) ---
combo_red_D = m_combo_red["pred"] if m_combo_red else np.full(len(data), ddU_red_cst + TdS_red_cst)
combo_ox_D = m_combo_ox["pred"] if m_combo_ox else np.full(len(data), ddU_ox_cst + TdS_ox_cst)
combo_red_D = np.where(np.isfinite(combo_red_D), combo_red_D, ddU_red_cst + TdS_red_cst)
combo_ox_D = np.where(np.isfinite(combo_ox_D), combo_ox_D, ddU_ox_cst + TdS_ox_cst)
# For model D, ΔΔU+TΔS is predicted directly, so set TΔS=0 and ΔΔU=combo
resD = compute_dg(data, combo_red_D, combo_ox_D,
                   np.zeros(len(data)), np.zeros(len(data)),
                   "D: (f0_maxO,ω)(ΔΔU+TΔS combined)")

# --- Model E: η for ΔΔU only + constant TΔS (conservative) ---
resE = compute_dg(data, ddU_red_B, ddU_ox_B,
                   np.full(len(data), TdS_red_cst),
                   np.full(len(data), TdS_ox_cst),
                   "E: η(ΔΔU) + Constant TΔS")

# ============================================================
# 5. COMPARISON TABLE
# ============================================================

print(f"\n{'=' * 90}")
print("  ΔG° PREDICTION COMPARISON")
print(f"{'=' * 90}")

all_models = [resA, resE, resB, resC, resD]

print(f"\n  {'Model':<45} {'MAE_ox':>7} {'MAE_red':>7} {'R²_ox':>6} {'R²_red':>6} {'n_ox':>4} {'n_red':>5} {'maxE_ox':>8} {'maxE_red':>8}")
print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*4} {'-'*5} {'-'*8} {'-'*8}")
for res in all_models:
    mae_o = f"{res.get('MAE_ox', np.nan):.2f}" if np.isfinite(res.get('MAE_ox', np.nan)) else "N/A"
    mae_r = f"{res.get('MAE_red', np.nan):.2f}" if np.isfinite(res.get('MAE_red', np.nan)) else "N/A"
    r2_o = f"{res.get('R2_ox', np.nan):.3f}" if np.isfinite(res.get('R2_ox', np.nan)) else "N/A"
    r2_r = f"{res.get('R2_red', np.nan):.3f}" if np.isfinite(res.get('R2_red', np.nan)) else "N/A"
    n_o = f"{res.get('n_ox', 0):.0f}" if res.get('n_ox', 0) > 0 else "N/A"
    n_r = f"{res.get('n_red', 0):.0f}" if res.get('n_red', 0) > 0 else "N/A"
    me_o = f"{res.get('maxerr_ox', np.nan):.2f}" if np.isfinite(res.get('maxerr_ox', np.nan)) else "N/A"
    me_r = f"{res.get('maxerr_red', np.nan):.2f}" if np.isfinite(res.get('maxerr_red', np.nan)) else "N/A"
    print(f"  {res['model']:<45} {mae_o:>7} {mae_r:>7} {r2_o:>6} {r2_r:>6} {n_o:>4} {n_r:>5} {me_o:>8} {me_r:>8}")

# ============================================================
# 6. PER-MOLECULE DETAIL FOR BEST MODEL
# ============================================================

# Find best model by average MAE
best_res = min(all_models, key=lambda r: (r.get("MAE_red", 99) + r.get("MAE_ox", 99)) / 2)
baseline_res = resA

print(f"\n{'=' * 90}")
print(f"  PER-MOLECULE DETAIL: {best_res['model']}")
print(f"{'=' * 90}")

print(f"\n  --- OXIDATION ---")
print(f"  {'Molecule':<10} {'ΔG_real':>8} {'ΔG_base':>8} {'err_b':>7} {'ΔG_new':>8} {'err_new':>7} {'Δ':>6}")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")
for i, mol in enumerate(data["molecule"].values):
    dg_real = data.iloc[i]["dG_oxidation_kJ"]
    if "pred_ox" in best_res and np.isfinite(best_res["pred_ox"][i]):
        dg_new = best_res["pred_ox"][i]
        dg_base = baseline_res["pred_ox"][i] if "pred_ox" in baseline_res else np.nan
        err_new = dg_new - dg_real
        err_base = dg_base - dg_real if np.isfinite(dg_base) else np.nan
        delta = abs(err_new) - abs(err_base) if np.isfinite(err_base) else np.nan
        d_str = f"{delta:+6.2f}" if np.isfinite(delta) else "   N/A"
        print(f"  {mol:<10} {dg_real:8.2f} {dg_base:8.2f} {err_base:7.2f} {dg_new:8.2f} {err_new:7.2f} {d_str}")

print(f"\n  --- REDUCTION ---")
print(f"  {'Molecule':<10} {'ΔG_real':>8} {'ΔG_base':>8} {'err_b':>7} {'ΔG_new':>8} {'err_new':>7} {'Δ':>6}")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")
for i, mol in enumerate(data["molecule"].values):
    dg_real = data.iloc[i]["dG_reduction_kJ"]
    if "pred_red" in best_res and np.isfinite(best_res["pred_red"][i]):
        dg_new = best_res["pred_red"][i]
        dg_base = baseline_res["pred_red"][i] if "pred_red" in baseline_res else np.nan
        err_new = dg_new - dg_real
        err_base = dg_base - dg_real if np.isfinite(dg_base) else np.nan
        delta = abs(err_new) - abs(err_base) if np.isfinite(err_base) else np.nan
        d_str = f"{delta:+6.2f}" if np.isfinite(delta) else "   N/A"
        print(f"  {mol:<10} {dg_real:8.2f} {dg_base:8.2f} {err_base:7.2f} {dg_new:8.2f} {err_new:7.2f} {d_str}")

# ============================================================
# 7. COMPONENT-WISE ERROR BREAKDOWN
# ============================================================

print(f"\n{'=' * 90}")
print("  COMPONENT-WISE ERROR ANALYSIS (per component MAE)")
print(f"{'=' * 90}")

def component_mae(real, pred, label):
    mask = np.isfinite(real) & np.isfinite(pred)
    if mask.sum() < 3:
        return
    mae = mean_absolute_error(real[mask], pred[mask])
    print(f"  {label:<30}: MAE = {mae:.3f} kJ/mol (n={mask.sum()})")


print("\n  --- EA/IP ---")
component_mae(data["EA_kJ"].values, data["EA_pred"].values, "EA (Koopmans)")
component_mae(data["IP_kJ"].values, data["IP_pred"].values, "IP (Koopmans)")

print("\n  --- E_relax (LVC2 calibrated) ---")
component_mae(data["real_red"].values, data["Erelax_red_pred"].values, "E_relax red")
component_mae(data["real_ox"].values, data["Erelax_ox_pred"].values, "E_relax ox")

print("\n  --- ΔΔU (baseline = constant) ---")
component_mae(data["ddU_red_kJ"].values, np.full(len(data), ddU_red_cst), "ΔΔU_red constant")
component_mae(data["ddU_ox_kJ"].values, np.full(len(data), ddU_ox_cst), "ΔΔU_ox constant")

print("\n  --- ΔΔU (new = η_vert) ---")
if m_ddU_red_eta:
    component_mae(data["ddU_red_kJ"].values, ddU_red_B, "ΔΔU_red η_vert")
if m_ddU_ox_eta:
    component_mae(data["ddU_ox_kJ"].values, ddU_ox_B, "ΔΔU_ox η_vert")

print("\n  --- ΔΔU (new = DI_sumO, ω_cdft) ---")
if m_ddU_red_DI_omega:
    component_mae(data["ddU_red_kJ"].values, ddU_red_C, "ΔΔU_red (DI,ω)")
if m_ddU_ox_DI_omega:
    component_mae(data["ddU_ox_kJ"].values, ddU_ox_C, "ΔΔU_ox (DI,ω)")

print("\n  --- TΔS (baseline = constant) ---")
component_mae(data["TdS_red_kJ"].values, np.full(len(data), TdS_red_cst), "TΔS_red constant")
component_mae(data["TdS_ox_kJ"].values, np.full(len(data), TdS_ox_cst), "TΔS_ox constant")

print("\n  --- TΔS (new = bader_sumO, μ_vert) ---")
if m_TdS_red_badermu:
    component_mae(data["TdS_red_kJ"].values, TdS_red_B, "TΔS_red (bader,μ)")
if m_TdS_ox_badermu:
    component_mae(data["TdS_ox_kJ"].values, TdS_ox_B, "TΔS_ox (bader,μ)")

print("\n  --- (ΔΔU+TΔS) combined (new = f0_maxO, ω_vert) ---")
if m_combo_red:
    real_combo_red = data["ddU_red_kJ"].values + data["TdS_red_kJ"].values
    component_mae(real_combo_red, combo_red_D, "(ΔΔU+TΔS)_red (f0,ω)")
if m_combo_ox:
    real_combo_ox = data["ddU_ox_kJ"].values + data["TdS_ox_kJ"].values
    component_mae(real_combo_ox, combo_ox_D, "(ΔΔU+TΔS)_ox (f0,ω)")

print("\n  Done.")
