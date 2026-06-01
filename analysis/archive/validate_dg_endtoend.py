#!/usr/bin/env python3
"""
End-to-end ΔG° prediction using ONLY predict_dg data (neutral + ionic SP).

Data sources (all from a single neutral opt+freq + ionic SP calculation):
  - HOMO_n, LUMO_n          → Koopmans EA/IP
  - E_relax_LVC2_{red,ox}_kj → E_relax (from ionic hessian at neutral geom)
  - EA_vert_kj, IP_vert_kj  → η, μ, ω for ΔΔU model
  - bader_sumO_n             → TΔS model
  - delta_G_{red,ox}_kj     → predict_dg's own ΔG (for comparison)

Reference ΔG: from full composite TZ2P+DZP calculations (redox_conformers.csv)

Models compared:
  A: Constant ΔΔU + Constant TΔS (old baseline)
  B: η-linear ΔΔU + (bader_sumO, μ)-bilinear TΔS (new)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import glob
import warnings
warnings.filterwarnings("ignore")

BASE = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5"
DEA = {"Br-DA", "Br-HMC", "I-DA", "I-HMC"}

# ============================================================
# 1. LOAD predict_dg DATA (Boltzmann-averaged per molecule)
# ============================================================

print("=" * 80)
print("  LOADING DATA")
print("=" * 80)

keep_cols = [
    "molecule", "conformer", "weight",
    "HOMO_n", "LUMO_n",
    "EA_vert_kj", "IP_vert_kj",
    "E_relax_LVC2_red_kj", "E_relax_LVC2_ox_kj",
    "mu_n", "eta_n", "omega_n", "gap_n",
    "bader_sumO_n", "DI_sumO_n", "LI_sumC_n",
    "ZPE_n_kj", "TS_vib_n_kj",
    "fplus_sumO_n", "fminus_sumO_n",
]

pdg = pd.read_csv(f"{BASE}/predict_dg_results/predict_dg_conformers.csv",
                   usecols=keep_cols)
pdg = pdg[~pdg["molecule"].isin(DEA)].copy()

numeric_cols = [c for c in keep_cols if c not in ["molecule", "conformer", "weight"]]


def bw_avg(group):
    w = group["weight"].values
    w = w / w.sum()
    result = {}
    for col in numeric_cols:
        if col in group.columns:
            vals = group[col].values
            result[col] = float(np.sum(w * vals)) if np.all(np.isfinite(vals)) else np.nan
    return pd.Series(result)


data = pdg.groupby("molecule", group_keys=False).apply(bw_avg).reset_index()

# Derived CDFT from vertical EA/IP
data["eta_vert"] = (data["IP_vert_kj"] - data["EA_vert_kj"]) / 2
data["mu_vert"] = -(data["IP_vert_kj"] + data["EA_vert_kj"]) / 2
data["omega_vert"] = data["mu_vert"]**2 / (2 * data["eta_vert"])

print(f"  Molecules from predict_dg: {len(data)}")
print(f"  Molecules: {sorted(data['molecule'].tolist())}")

# ============================================================
# 2. LOAD REFERENCE ΔG FROM FULL COMPOSITE (redox_conformers.csv)
# ============================================================

files = glob.glob(f"{BASE}/data/Acetonitrile/*/redox/results/*_redox_conformers.csv")
rows_ref = []
for f in sorted(files):
    mol = f.split("/")[-1].replace("_redox_conformers.csv", "")
    if mol in DEA:
        continue
    df = pd.read_csv(f)
    w = df["weight"].values
    w = w / w.sum()
    row = {"molecule": mol}
    for col in ["dG_reduction_kJ", "dG_oxidation_kJ", "EA_kJ", "IP_kJ",
                "Edef_red_kJ", "Edef_ox_kJ", "ddU_red_kJ", "ddU_ox_kJ",
                "TdS_red_kJ", "TdS_ox_kJ"]:
        if col in df.columns:
            row[col] = float(np.sum(w * df[col].values))
    rows_ref.append(row)

ref = pd.DataFrame(rows_ref)

# Merge: keep only molecules present in BOTH
data = data.merge(ref, on="molecule", how="inner")
data = data.sort_values("molecule").reset_index(drop=True)
n = len(data)

print(f"  Molecules with both predict_dg AND full redox: {n}")
print(f"  {sorted(data['molecule'].tolist())}")

# ============================================================
# 3. KOOPMANS EA/IP CALIBRATION
# ============================================================

print("\n" + "=" * 80)
print("  COMPONENT MODELS (fitted on these n={} molecules)".format(n))
print("=" * 80)

EH_TO_EV = 27.211386245988
EV_TO_KJ = 96.485

homo_eV = data["HOMO_n"].values * EH_TO_EV
lumo_eV = data["LUMO_n"].values * EH_TO_EV
ip_dft = data["IP_kJ"].values
ea_dft = data["EA_kJ"].values

# IP = f(ε_HOMO)
sl_ip, int_ip, r_ip, p_ip, _ = linregress(homo_eV, ip_dft)
ip_pred = sl_ip * homo_eV + int_ip
mae_ip = mean_absolute_error(ip_dft, ip_pred)
print(f"\n  IP Koopmans:  IP = {sl_ip:.3f}·ε_HOMO + {int_ip:.1f}")
print(f"    R² = {r_ip**2:.4f}, MAE = {mae_ip:.2f} kJ/mol")

# EA = f(ε_LUMO)
sl_ea, int_ea, r_ea, p_ea, _ = linregress(lumo_eV, ea_dft)
ea_pred = sl_ea * lumo_eV + int_ea
mae_ea = mean_absolute_error(ea_dft, ea_pred)
print(f"  EA Koopmans:  EA = {sl_ea:.3f}·ε_LUMO + {int_ea:.1f}")
print(f"    R² = {r_ea**2:.4f}, MAE = {mae_ea:.2f} kJ/mol")

data["IP_pred"] = ip_pred
data["EA_pred"] = ea_pred

# ============================================================
# 4. E_relax LVC2 CALIBRATION
# ============================================================

erelax_lvc2_red = data["E_relax_LVC2_red_kj"].values
erelax_lvc2_ox = data["E_relax_LVC2_ox_kj"].values
edef_red_dft = data["Edef_red_kJ"].values
edef_ox_dft = data["Edef_ox_kJ"].values

sl_er_red, int_er_red, r_er_red, _, _ = linregress(erelax_lvc2_red, edef_red_dft)
erelax_red_pred = sl_er_red * erelax_lvc2_red + int_er_red
mae_er_red = mean_absolute_error(edef_red_dft, erelax_red_pred)

sl_er_ox, int_er_ox, r_er_ox, _, _ = linregress(erelax_lvc2_ox, edef_ox_dft)
erelax_ox_pred = sl_er_ox * erelax_lvc2_ox + int_er_ox
mae_er_ox = mean_absolute_error(edef_ox_dft, erelax_ox_pred)

print(f"\n  E_relax LVC2 (red): y = {sl_er_red:.3f}·x + {int_er_red:.2f}")
print(f"    R² = {r_er_red**2:.4f}, MAE = {mae_er_red:.2f} kJ/mol")
print(f"  E_relax LVC2 (ox):  y = {sl_er_ox:.3f}·x + {int_er_ox:.2f}")
print(f"    R² = {r_er_ox**2:.4f}, MAE = {mae_er_ox:.2f} kJ/mol")

data["Erelax_red_pred"] = erelax_red_pred
data["Erelax_ox_pred"] = erelax_ox_pred

# ============================================================
# 5. ΔΔU MODELS
# ============================================================

ddU_red_real = data["ddU_red_kJ"].values
ddU_ox_real = data["ddU_ox_kJ"].values

# Constant
ddU_red_cst = np.mean(ddU_red_real)
ddU_ox_cst = np.mean(ddU_ox_real)
mae_ddU_red_cst = mean_absolute_error(ddU_red_real, np.full(n, ddU_red_cst))
mae_ddU_ox_cst = mean_absolute_error(ddU_ox_real, np.full(n, ddU_ox_cst))

# η-linear for red
eta = data["eta_vert"].values
sl_ddU, int_ddU, r_ddU, p_ddU, _ = linregress(eta, ddU_red_real)
ddU_red_eta = sl_ddU * eta + int_ddU
mae_ddU_red_eta = mean_absolute_error(ddU_red_real, ddU_red_eta)
r2_ddU_adj = 1 - (1 - r_ddU**2) * (n - 1) / (n - 2)

print(f"\n  ΔΔU_red constant: {ddU_red_cst:.2f}, MAE = {mae_ddU_red_cst:.2f}")
print(f"  ΔΔU_red η-linear: {sl_ddU:.4f}·η + {int_ddU:.2f}")
print(f"    R²_adj = {r2_ddU_adj:.3f}, MAE = {mae_ddU_red_eta:.2f}, p = {p_ddU:.1e}")
print(f"  ΔΔU_ox constant: {ddU_ox_cst:.2f}, MAE = {mae_ddU_ox_cst:.2f}")

# ============================================================
# 6. TΔS MODELS
# ============================================================

TdS_red_real = data["TdS_red_kJ"].values
TdS_ox_real = data["TdS_ox_kJ"].values

# Constant
TdS_red_cst = np.mean(TdS_red_real)
TdS_ox_cst = np.mean(TdS_ox_real)
mae_TdS_red_cst = mean_absolute_error(TdS_red_real, np.full(n, TdS_red_cst))
mae_TdS_ox_cst = mean_absolute_error(TdS_ox_real, np.full(n, TdS_ox_cst))

# Bilinear (bader_sumO, μ)
bader = data["bader_sumO_n"].values
mu = data["mu_vert"].values

X_tds = np.column_stack([bader, mu])
reg_tds_red = LinearRegression().fit(X_tds, TdS_red_real)
TdS_red_bader = reg_tds_red.predict(X_tds)
mae_TdS_red_bader = mean_absolute_error(TdS_red_real, TdS_red_bader)
r2_tds_red = r2_score(TdS_red_real, TdS_red_bader)
r2_tds_red_adj = 1 - (1 - r2_tds_red) * (n - 1) / (n - 3)

reg_tds_ox = LinearRegression().fit(X_tds, TdS_ox_real)
TdS_ox_bader = reg_tds_ox.predict(X_tds)
mae_TdS_ox_bader = mean_absolute_error(TdS_ox_real, TdS_ox_bader)
r2_tds_ox = r2_score(TdS_ox_real, TdS_ox_bader)
r2_tds_ox_adj = 1 - (1 - r2_tds_ox) * (n - 1) / (n - 3)

print(f"\n  TΔS_red constant: {TdS_red_cst:.2f}, MAE = {mae_TdS_red_cst:.2f}")
print(f"  TΔS_red (bader,μ): {reg_tds_red.coef_[0]:.4f}·bader + {reg_tds_red.coef_[1]:.4f}·μ + {reg_tds_red.intercept_:.2f}")
print(f"    R²_adj = {r2_tds_red_adj:.3f}, MAE = {mae_TdS_red_bader:.2f}")
print(f"  TΔS_ox constant: {TdS_ox_cst:.2f}, MAE = {mae_TdS_ox_cst:.2f}")
print(f"  TΔS_ox (bader,μ): {reg_tds_ox.coef_[0]:.4f}·bader + {reg_tds_ox.coef_[1]:.4f}·μ + {reg_tds_ox.intercept_:.2f}")
print(f"    R²_adj = {r2_tds_ox_adj:.3f}, MAE = {mae_TdS_ox_bader:.2f}")

# ============================================================
# 7. COMPUTE ΔG PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("  ΔG° PREDICTIONS")
print("=" * 80)

# ΔG_red = -EA + E_relax_red + ΔΔU_red + TΔS_red
# ΔG_ox  =  IP + E_relax_ox  + ΔΔU_ox  + TΔS_ox

dG_red_real = data["dG_reduction_kJ"].values
dG_ox_real = data["dG_oxidation_kJ"].values

models = {}

# Model A: constant ΔΔU + constant TΔS
dG_red_A = -ea_pred + erelax_red_pred + ddU_red_cst + TdS_red_cst
dG_ox_A = ip_pred + erelax_ox_pred + ddU_ox_cst + TdS_ox_cst
models["A: Cst ΔΔU + Cst TΔS"] = (dG_red_A, dG_ox_A)

# Model B: η(ΔΔU_red) + constant(ΔΔU_ox) + (bader,μ)(TΔS)
dG_red_B = -ea_pred + erelax_red_pred + ddU_red_eta + TdS_red_bader
dG_ox_B = ip_pred + erelax_ox_pred + ddU_ox_cst + TdS_ox_bader
models["B: η(ΔΔU) + (bader,μ)(TΔS)"] = (dG_red_B, dG_ox_B)

# Model C: η(ΔΔU_red) + constant TΔS (conservative)
dG_red_C = -ea_pred + erelax_red_pred + ddU_red_eta + TdS_red_cst
dG_ox_C = ip_pred + erelax_ox_pred + ddU_ox_cst + TdS_ox_cst
models["C: η(ΔΔU) + Cst TΔS"] = (dG_red_C, dG_ox_C)

# Model D: all constant but with new means (sanity check)
# Just use real component means as constants

print(f"\n  {'Model':<38} {'MAE_ox':>7} {'MAE_red':>7} {'R²_ox':>6} {'R²_red':>6} {'maxE_ox':>8} {'maxE_red':>8}")
print(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")

for name, (dG_red, dG_ox) in models.items():
    mae_red = mean_absolute_error(dG_red_real, dG_red)
    mae_ox = mean_absolute_error(dG_ox_real, dG_ox)
    r2_red = r2_score(dG_red_real, dG_red)
    r2_ox = r2_score(dG_ox_real, dG_ox)
    maxe_red = np.max(np.abs(dG_red - dG_red_real))
    maxe_ox = np.max(np.abs(dG_ox - dG_ox_real))
    print(f"  {name:<38} {mae_ox:7.2f} {mae_red:7.2f} {r2_ox:6.3f} {r2_red:6.3f} {maxe_ox:8.2f} {maxe_red:8.2f}")

# ============================================================
# 8. PER-MOLECULE DETAIL (best model)
# ============================================================

print(f"\n{'=' * 80}")
print("  PER-MOLECULE DETAIL — Model B: η(ΔΔU) + (bader,μ)(TΔS)")
print(f"{'=' * 80}")

dG_red_B_arr = models["B: η(ΔΔU) + (bader,μ)(TΔS)"][0]
dG_ox_B_arr = models["B: η(ΔΔU) + (bader,μ)(TΔS)"][1]

print(f"\n  {'Molecule':<10} {'ΔG_ox_DFT':>10} {'ΔG_ox_pred':>11} {'err_ox':>7} {'ΔG_red_DFT':>11} {'ΔG_red_pred':>12} {'err_red':>8}")
print(f"  {'-'*10} {'-'*10} {'-'*11} {'-'*7} {'-'*11} {'-'*12} {'-'*8}")

for i, mol in enumerate(data["molecule"].values):
    err_ox = dG_ox_B_arr[i] - dG_ox_real[i]
    err_red = dG_red_B_arr[i] - dG_red_real[i]
    print(f"  {mol:<10} {dG_ox_real[i]:10.2f} {dG_ox_B_arr[i]:11.2f} {err_ox:+7.2f} "
          f"{dG_red_real[i]:11.2f} {dG_red_B_arr[i]:12.2f} {err_red:+8.2f}")

# ============================================================
# 9. COMPONENT ERROR BUDGET
# ============================================================

print(f"\n{'=' * 80}")
print("  COMPONENT ERROR BUDGET (MAE per component)")
print(f"{'=' * 80}")

components = [
    ("EA (Koopmans)", ea_dft, ea_pred),
    ("IP (Koopmans)", ip_dft, ip_pred),
    ("E_relax red (LVC2 cal.)", edef_red_dft, erelax_red_pred),
    ("E_relax ox (LVC2 cal.)", edef_ox_dft, erelax_ox_pred),
    ("ΔΔU_red (constant)", ddU_red_real, np.full(n, ddU_red_cst)),
    ("ΔΔU_red (η-linear)", ddU_red_real, ddU_red_eta),
    ("ΔΔU_ox (constant)", ddU_ox_real, np.full(n, ddU_ox_cst)),
    ("TΔS_red (constant)", TdS_red_real, np.full(n, TdS_red_cst)),
    ("TΔS_red (bader,μ)", TdS_red_real, TdS_red_bader),
    ("TΔS_ox (constant)", TdS_ox_real, np.full(n, TdS_ox_cst)),
    ("TΔS_ox (bader,μ)", TdS_ox_real, TdS_ox_bader),
]

print(f"\n  {'Component':<28} {'MAE':>6}")
print(f"  {'-'*28} {'-'*6}")
for name, real, pred in components:
    mae = mean_absolute_error(real, pred)
    print(f"  {name:<28} {mae:6.2f}")

# ============================================================
# 10. SUMMARY TABLE FOR RAPPORT (Section 3.5)
# ============================================================

print(f"\n{'=' * 80}")
print("  TABLE FOR RAPPORT SECTION 3.5")
print(f"{'=' * 80}")

print(f"""
  Composante          | Modèle                                | MAE ox  | MAE red
  --------------------|---------------------------------------|---------|--------
  EA / IP             | Koopmans (ε_HOMO/LUMO, fit linéaire)  | {mae_ip:.1f}     | {mae_ea:.1f}
  E_relax             | LVC2 calibré                          | {mae_er_ox:.1f}     | {mae_er_red:.1f}
  ΔΔU                 | Constant (ox) / η linéaire (red)      | {mae_ddU_ox_cst:.1f}     | {mae_ddU_red_eta:.1f} ({mae_ddU_red_cst:.1f})
  TΔS                 | (bader_sumO, μ) bilinéaire            | {mae_TdS_ox_bader:.1f} ({mae_TdS_ox_cst:.1f}) | {mae_TdS_red_bader:.1f} ({mae_TdS_red_cst:.1f})
""")

# Model A (old) and B (new) for ΔG
mae_dG_red_A = mean_absolute_error(dG_red_real, models["A: Cst ΔΔU + Cst TΔS"][0])
mae_dG_ox_A = mean_absolute_error(dG_ox_real, models["A: Cst ΔΔU + Cst TΔS"][1])
mae_dG_red_B = mean_absolute_error(dG_red_real, models["B: η(ΔΔU) + (bader,μ)(TΔS)"][0])
mae_dG_ox_B = mean_absolute_error(dG_ox_real, models["B: η(ΔΔU) + (bader,μ)(TΔS)"][1])
r2_dG_red_B = r2_score(dG_red_real, models["B: η(ΔΔU) + (bader,μ)(TΔS)"][0])
r2_dG_ox_B = r2_score(dG_ox_real, models["B: η(ΔΔU) + (bader,μ)(TΔS)"][1])

print(f"  ΔG° (old: cst)            | MAE ox = {mae_dG_ox_A:.1f}  | MAE red = {mae_dG_red_A:.1f}")
print(f"  ΔG° (new: η + bader,μ)    | MAE ox = {mae_dG_ox_B:.1f}  | MAE red = {mae_dG_red_B:.1f}")
print(f"  ΔG° (new) R² ox = {r2_dG_ox_B:.3f}, R² red = {r2_dG_red_B:.3f}")
print(f"  n = {n}")

print("\n  Done.")
