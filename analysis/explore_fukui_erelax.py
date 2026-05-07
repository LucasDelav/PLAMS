#!/usr/bin/env python3
"""
Explore correlations between Fukui descriptors and LVC2 residual for E_relax (reduction).

Goal: find descriptors that predict (real_red - lvc2_red) better than LUMO oxygen fraction (R²=0.47).
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize

# ── 1. Load LVC summary ──────────────────────────────────────────────────────
lvc = pd.read_csv("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/data/lvc_summary.csv")

# Exclude DEA molecules (no LVC1 data, but we keep LVC2-only analysis)
dea_molecules = {"Br-DA", "Br-HMC", "I-DA", "I-HMC"}
lvc = lvc[~lvc["molecule_name"].isin(dea_molecules)].copy()
lvc["residual"] = lvc["real_red"] - lvc["lvc2_red"]

print("=" * 80)
print("LVC2 RESIDUAL FOR E_RELAX (REDUCTION)")
print("=" * 80)
print(f"\n{'Molecule':<12} {'real_red':>10} {'lvc2_red':>10} {'residual':>10}")
print("-" * 44)
for _, row in lvc.sort_values("molecule_name").iterrows():
    print(f"{row['molecule_name']:<12} {row['real_red']:10.2f} {row['lvc2_red']:10.2f} {row['residual']:10.2f}")
print(f"\nN = {len(lvc)} molecules")
print(f"Mean |residual| = {lvc['residual'].abs().mean():.2f} kJ/mol")

# ── 2. Load predict_dg_conformers and compute Boltzmann-weighted averages ────
df = pd.read_csv("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/predict_dg_results/predict_dg_conformers.csv")

# Identify all fplus_O*_n columns for max-fplus computation
fplus_O_cols = [c for c in df.columns if c.startswith("fplus_O") and c.endswith("_n")]
fplus_O_individual = [c for c in fplus_O_cols if c != "fplus_sumO_n"]

print(f"\nfplus_O individual columns found: {fplus_O_individual}")

# Compute max fplus on any O atom per conformer
if fplus_O_individual:
    df["fplus_maxO_n"] = df[fplus_O_individual].max(axis=1)
else:
    df["fplus_maxO_n"] = np.nan

# Combined descriptor
df["fplus_sumO_x_eta_n"] = df["fplus_sumO_n"] * df["eta_n"]

# Descriptors to Boltzmann-average
descriptors = [
    "fplus_sumO_n", "fminus_sumO_n", "dual_sumO_n", "f0_sumO_n",
    "fplus_maxO_n",
    "fplus_sumC_n", "fminus_sumC_n", "dual_sumC_n",
    "fplus_sumH_n",
    "mu_n", "eta_n", "omega_n",
    "fplus_sumO_x_eta_n",
    "HOMO_n", "LUMO_n", "gap_n",
    "E_relax_LVC2_red_kj",
]

# Boltzmann-weighted average per molecule
def boltzmann_avg(group, cols):
    w = group["weight"].values
    w = w / w.sum()  # renormalize just in case
    result = {}
    for c in cols:
        vals = group[c].values
        mask = ~np.isnan(vals)
        if mask.sum() > 0:
            w_clean = w[mask]
            w_clean = w_clean / w_clean.sum()
            result[c] = np.sum(w_clean * vals[mask])
        else:
            result[c] = np.nan
    return pd.Series(result)

mol_avg = df.groupby("molecule").apply(lambda g: boltzmann_avg(g, descriptors)).reset_index()

print(f"\nMolecules in predict_dg_conformers: {sorted(mol_avg['molecule'].unique())}")
print(f"Molecules in lvc_summary: {sorted(lvc['molecule_name'].unique())}")

# ── 3. Merge ─────────────────────────────────────────────────────────────────
merged = pd.merge(lvc, mol_avg, left_on="molecule_name", right_on="molecule", how="inner")
print(f"\nMerged molecules: {len(merged)}")
if len(merged) < len(lvc):
    missing = set(lvc["molecule_name"]) - set(mol_avg["molecule"])
    print(f"Missing from predict_dg: {missing}")

residual = merged["residual"].values

# ── 4. Correlation analysis ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CORRELATION: FUKUI DESCRIPTORS vs LVC2 RESIDUAL (real_red - lvc2_red)")
print("=" * 80)

test_descriptors = [
    ("fplus_sumO_n", "f+(O) sum — nucleophilic attack susceptibility on O"),
    ("fminus_sumO_n", "f-(O) sum — electrophilic attack susceptibility on O"),
    ("dual_sumO_n", "dual(O) sum — f+(O) - f-(O)"),
    ("f0_sumO_n", "f0(O) sum — radical attack susceptibility on O"),
    ("fplus_maxO_n", "max f+(O_i) — max individual O Fukui"),
    ("fplus_sumC_n", "f+(C) sum"),
    ("fminus_sumC_n", "f-(C) sum"),
    ("dual_sumC_n", "dual(C) sum"),
    ("fplus_sumH_n", "f+(H) sum"),
    ("mu_n", "chemical potential mu"),
    ("eta_n", "hardness eta"),
    ("omega_n", "electrophilicity omega"),
    ("fplus_sumO_x_eta_n", "f+(O) * eta"),
    ("LUMO_n", "LUMO energy"),
    ("gap_n", "HOMO-LUMO gap"),
]

results = []

print(f"\n{'Descriptor':<28} {'R²':>8} {'p-value':>10} {'slope':>10} {'intercept':>10}")
print("-" * 68)

for desc_col, desc_label in test_descriptors:
    x = merged[desc_col].values
    mask = ~np.isnan(x)
    if mask.sum() < 5:
        print(f"{desc_col:<28} {'N/A':>8} (too few values)")
        continue

    slope, intercept, r, p, se = stats.linregress(x[mask], residual[mask])
    r2 = r**2
    results.append((desc_col, desc_label, r2, p, slope, intercept, mask.sum()))

    star = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
    print(f"{desc_col:<28} {r2:8.4f} {p:10.4e} {slope:10.4f} {intercept:10.4f}{star}")

# Sort by R²
results.sort(key=lambda x: -x[2])

print("\n" + "=" * 80)
print("RANKING BY R²")
print("=" * 80)
print(f"\n{'Rank':<6} {'Descriptor':<28} {'R²':>8} {'p-value':>10} {'Description'}")
print("-" * 90)
for i, (col, label, r2, p, slope, intercept, n) in enumerate(results):
    star = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
    print(f"{i+1:<6} {col:<28} {r2:8.4f} {p:10.4e}  {label}{star}")

# ── 5. Parametric correction: E_corr = LVC2 * (1 + a*descriptor) ────────────
print("\n" + "=" * 80)
print("PARAMETRIC CORRECTIONS: E_corr = LVC2 * (1 + a * descriptor)")
print("=" * 80)

real_red = merged["real_red"].values
lvc2_red = merged["lvc2_red"].values

# Baseline: uncorrected LVC2
mae_baseline = np.mean(np.abs(real_red - lvc2_red))
print(f"\nBaseline MAE (uncorrected LVC2): {mae_baseline:.2f} kJ/mol")

best_corrections = []

for desc_col, desc_label in test_descriptors:
    x = merged[desc_col].values
    mask = ~np.isnan(x)
    if mask.sum() < 5:
        continue

    x_clean = x[mask]
    real_clean = real_red[mask]
    lvc2_clean = lvc2_red[mask]

    # Fit a: minimize |real - LVC2*(1 + a*x)|²
    def objective(a):
        pred = lvc2_clean * (1 + a[0] * x_clean)
        return np.sum((real_clean - pred)**2)

    res = optimize.minimize(objective, [0.0], method="Nelder-Mead")
    a_opt = res.x[0]
    pred_corr = lvc2_clean * (1 + a_opt * x_clean)
    mae_corr = np.mean(np.abs(real_clean - pred_corr))

    # R² of corrected vs real
    ss_res = np.sum((real_clean - pred_corr)**2)
    ss_tot = np.sum((real_clean - real_clean.mean())**2)
    r2_corr = 1 - ss_res / ss_tot

    best_corrections.append((desc_col, desc_label, a_opt, mae_corr, r2_corr, mask.sum()))

best_corrections.sort(key=lambda x: x[3])

print(f"\n{'Descriptor':<28} {'a_opt':>10} {'MAE':>8} {'R²_corr':>8}")
print("-" * 56)
for col, label, a, mae, r2, n in best_corrections:
    print(f"{col:<28} {a:10.6f} {mae:8.2f} {r2:8.4f}")

# ── 6. Quadratic correction: E_corr = LVC2 * (1 + descriptor²) ──────────────
print("\n" + "=" * 80)
print("QUADRATIC CORRECTION: E_corr = LVC2 * (1 + descriptor²)")
print("(analogous to LUMO oxygen fraction formula)")
print("=" * 80)

for desc_col, desc_label in test_descriptors:
    x = merged[desc_col].values
    mask = ~np.isnan(x)
    if mask.sum() < 5:
        continue

    x_clean = x[mask]
    real_clean = real_red[mask]
    lvc2_clean = lvc2_red[mask]

    pred_q = lvc2_clean * (1 + x_clean**2)
    mae_q = np.mean(np.abs(real_clean - pred_q))

    # Only print if MAE is reasonable (< 50 kJ/mol)
    if mae_q < 50:
        print(f"{desc_col:<28} MAE = {mae_q:.2f} kJ/mol")

# ── 7. Parametric quadratic: E_corr = LVC2 * (1 + a*descriptor²) ────────────
print("\n" + "=" * 80)
print("PARAMETRIC QUADRATIC: E_corr = LVC2 * (1 + a * descriptor²)")
print("=" * 80)

quad_corrections = []

for desc_col, desc_label in test_descriptors:
    x = merged[desc_col].values
    mask = ~np.isnan(x)
    if mask.sum() < 5:
        continue

    x_clean = x[mask]
    real_clean = real_red[mask]
    lvc2_clean = lvc2_red[mask]

    def objective_q(a):
        pred = lvc2_clean * (1 + a[0] * x_clean**2)
        return np.sum((real_clean - pred)**2)

    res = optimize.minimize(objective_q, [1.0], method="Nelder-Mead")
    a_opt = res.x[0]
    pred_corr = lvc2_clean * (1 + a_opt * x_clean**2)
    mae_corr = np.mean(np.abs(real_clean - pred_corr))

    ss_res = np.sum((real_clean - pred_corr)**2)
    ss_tot = np.sum((real_clean - real_clean.mean())**2)
    r2_corr = 1 - ss_res / ss_tot

    quad_corrections.append((desc_col, desc_label, a_opt, mae_corr, r2_corr))

quad_corrections.sort(key=lambda x: x[3])

print(f"\n{'Descriptor':<28} {'a_opt':>10} {'MAE':>8} {'R²_corr':>8}")
print("-" * 56)
for col, label, a, mae, r2 in quad_corrections:
    print(f"{col:<28} {a:10.6f} {mae:8.2f} {r2:8.4f}")

# ── 8. Additive correction: E_corr = LVC2 + a*descriptor + b ────────────────
print("\n" + "=" * 80)
print("ADDITIVE CORRECTION: E_corr = LVC2 + a*descriptor + b")
print("(2 free parameters)")
print("=" * 80)

add_corrections = []

for desc_col, desc_label in test_descriptors:
    x = merged[desc_col].values
    mask = ~np.isnan(x)
    if mask.sum() < 5:
        continue

    x_clean = x[mask]
    real_clean = real_red[mask]
    lvc2_clean = lvc2_red[mask]

    # Fit: real = lvc2 + a*x + b  =>  residual = a*x + b
    resid_clean = real_clean - lvc2_clean
    slope, intercept, r, p, se = stats.linregress(x_clean, resid_clean)

    pred_corr = lvc2_clean + slope * x_clean + intercept
    mae_corr = np.mean(np.abs(real_clean - pred_corr))

    ss_res = np.sum((real_clean - pred_corr)**2)
    ss_tot = np.sum((real_clean - real_clean.mean())**2)
    r2_corr = 1 - ss_res / ss_tot

    # Adjusted R² (2 free params on top of LVC2)
    n = len(x_clean)
    p_params = 2  # a and b
    r2_adj = 1 - (1 - r2_corr) * (n - 1) / (n - p_params - 1) if n > p_params + 1 else np.nan

    add_corrections.append((desc_col, desc_label, slope, intercept, mae_corr, r2_corr, r2_adj, p))

add_corrections.sort(key=lambda x: x[4])

print(f"\n{'Descriptor':<28} {'a':>10} {'b':>10} {'MAE':>8} {'R²':>8} {'R²_adj':>8} {'p(a)':>10}")
print("-" * 88)
for col, label, a, b, mae, r2, r2a, p in add_corrections:
    star = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
    print(f"{col:<28} {a:10.4f} {b:10.4f} {mae:8.2f} {r2:8.4f} {r2a:8.4f} {p:10.4e}{star}")

# ── 9. Summary comparison ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: BEST CORRECTIONS vs LUMO OXYGEN FRACTION BENCHMARK")
print("=" * 80)
print(f"\n  LUMO O fraction (from rapport): R²(residual) = 0.47, corrected MAE ~ 1.5 kJ/mol")
print(f"  Baseline LVC2 MAE = {mae_baseline:.2f} kJ/mol")

print("\n  Top 5 additive corrections (lowest MAE):")
for i, (col, label, a, b, mae, r2, r2a, p) in enumerate(add_corrections[:5]):
    print(f"    {i+1}. {col:<28} MAE={mae:.2f} kJ/mol  R²_adj={r2a:.4f}  p={p:.4e}")

print("\n  Top 5 residual correlations (highest R²):")
for i, (col, label, r2, p, slope, intercept, n) in enumerate(results[:5]):
    print(f"    {i+1}. {col:<28} R²={r2:.4f}  p={p:.4e}")

# ── 10. Multi-descriptor: fplus_sumO + eta ───────────────────────────────────
print("\n" + "=" * 80)
print("MULTI-DESCRIPTOR: residual = a*fplus_sumO + b*eta + c")
print("=" * 80)

from numpy.linalg import lstsq

x1 = merged["fplus_sumO_n"].values
x2 = merged["eta_n"].values
mask = ~(np.isnan(x1) | np.isnan(x2))

if mask.sum() >= 5:
    X = np.column_stack([x1[mask], x2[mask], np.ones(mask.sum())])
    y = residual[mask]

    coeffs, res_sum, rank, sv = lstsq(X, y, rcond=None)
    a, b, c = coeffs

    pred = X @ coeffs
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    n = mask.sum()
    p_params = 3
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p_params - 1)
    mae = np.mean(np.abs(y - pred))

    # Corrected E_relax
    pred_erelax = lvc2_red[mask] + pred
    mae_erelax = np.mean(np.abs(real_red[mask] - pred_erelax))

    print(f"  a(fplus_sumO) = {a:.4f}")
    print(f"  b(eta)        = {b:.4f}")
    print(f"  c(intercept)  = {c:.4f}")
    print(f"  R²            = {r2:.4f}")
    print(f"  R²_adj        = {r2_adj:.4f}")
    print(f"  MAE(residual) = {mae:.2f} kJ/mol")
    print(f"  MAE(E_relax)  = {mae_erelax:.2f} kJ/mol")

# ── 11. Print per-molecule detail for best descriptor ────────────────────────
print("\n" + "=" * 80)
print("PER-MOLECULE DETAIL: best single Fukui descriptor")
print("=" * 80)

if results:
    best_col = results[0][0]
    best_label = results[0][1]
    print(f"Best descriptor: {best_col} ({best_label})")

    # Get additive correction params for this descriptor
    for col, label, a, b, mae, r2, r2a, p in add_corrections:
        if col == best_col:
            print(f"Additive correction: E_corr = LVC2 + {a:.4f}*{best_col} + {b:.4f}")
            print(f"\n{'Molecule':<12} {'real':>8} {'LVC2':>8} {'corr':>8} {'err_LVC2':>9} {'err_corr':>9} {best_col:>12}")
            print("-" * 72)
            x = merged[best_col].values
            for _, row in merged.sort_values("molecule_name").iterrows():
                real_val = row["real_red"]
                lvc2_val = row["lvc2_red"]
                desc_val = row[best_col]
                corr_val = lvc2_val + a * desc_val + b
                err_lvc2 = real_val - lvc2_val
                err_corr = real_val - corr_val
                print(f"{row['molecule_name']:<12} {real_val:8.2f} {lvc2_val:8.2f} {corr_val:8.2f} {err_lvc2:9.2f} {err_corr:9.2f} {desc_val:12.4f}")
            break
