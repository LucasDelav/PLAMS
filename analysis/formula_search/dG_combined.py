#!/usr/bin/env python3
"""ΔG combined model — assemble the 4 models per the rapport formula:

    ΔG_red = −EA + E_relax_red + ΔΔU_red − TΔS_red
    ΔG_ox  = +IP + E_relax_ox  + ΔΔU_ox  + TΔS_ox

Each component predicted by its respective minimal model:
  EA       : (μ, η, ω) Padé[1/1] g_Born × (1 + a · f²_max_noO)
  IP       : (μ, η, n=2η/I²) Padé[1/1] g_Born × (1 + a · f²_min_all)
  E_relax  : (μ, η, ω, LVC2_paired) const × (1 + a · f±_max_all)
  ΔΔU     : (μ, η) const × (1 + a · f²_paired_all)
  TΔS      : (μ, η) const × (1 + a · f²_paired_all)

DFT target = same linear combination of the DFT components.
Output: parity plot ΔG_red / ΔG_ox on a single PNG + MAE, RMSE, R².
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "analysis" / "formula_search" / "data" / "merged_dataset.parquet"
PRED_DIR = REPO / "dataset_predict_dg"
RES = REPO / "analysis" / "formula_search" / "results"

EH = 2625.499638
EXCLUDED = {"Br-DA","Br-HMC","I-DA","I-HMC","L-DOPA"}
EPSILON_R = {"Hexane":1.88,"Toluene":2.38,"Chloroform":4.81,"Tetrahydrofuran":7.58,
             "Dichloromethane":8.93,"Acetone":20.7,"Methanol":32.6,"Acetonitrile":35.7,
             "DMSO":46.7,"Water":78.4}
SOLVENT_ORDER = sorted(EPSILON_R, key=EPSILON_R.get)


# ---------- Fukui extraction ----------
def parse_xyz_symbols(xyz):
    lines = xyz.strip().split("\n")
    n = int(lines[0])
    return [lines[2+i].split()[0] for i in range(n)]


def load_fukui_features(df):
    """For each conformer, extract:
      f2_max_all, f2_min_all (max/min of dual over all atoms)
      f2_max_noO              (max of dual over non-O atoms)
      fplus_max_all, fminus_max_all
    """
    rows = []
    for _, row in df.iterrows():
        p = PRED_DIR/row.solvent/row.molecule/f"{row.molecule}_conf{int(row.conformer)}_prediction.json"
        out = {"f2_max_all": np.nan, "f2_min_all": np.nan, "f2_max_noO": np.nan,
               "fplus_max_all": np.nan, "fminus_max_all": np.nan}
        try:
            d = json.load(open(p))
            ns = d["states"]["neutral_sp"]
            cdft = ns["cdft"]
            syms = np.asarray(parse_xyz_symbols(ns["xyz"]))
            dual = np.asarray(cdft["dual"], dtype=float)
            fp = np.asarray(cdft["fplus"], dtype=float)
            fm = np.asarray(cdft["fminus"], dtype=float)
            out["f2_max_all"] = float(dual.max())
            out["f2_min_all"] = float(dual.min())
            mask_noO = syms != "O"
            out["f2_max_noO"] = float(dual[mask_noO].max()) if mask_noO.any() else np.nan
            out["fplus_max_all"] = float(fp.max())
            out["fminus_max_all"] = float(fm.max())
        except Exception:
            pass
        rows.append(out)
    return pd.DataFrame(rows, index=df.index)


# ---------- Component predictors ----------
def g_born(eps): return (eps - 1.0) / (2.0 * eps + 1.0)


def pade11_g(g, p):
    """Padé[1/1] in g_Born: (a0+a1*g)/(1+b1*g)."""
    return (p["a0"] + p["a1"]*g) / (1.0 + p["b1"]*g)


def predict_EA(df, ea_coefs, a_corr):
    """EA = [β_μ·μ + β_η·η + β_ω·ω] × (1 + a · f²_max_noO)."""
    g = g_born(df["epsilon"].values)
    mu, eta, omega = df["mu"].values, df["eta"].values, df["omega"].values
    base = pade11_g(g, ea_coefs["mu"])*mu + pade11_g(g, ea_coefs["eta"])*eta + pade11_g(g, ea_coefs["omega"])*omega
    return base * (1.0 + a_corr * df["f2_max_noO"].values)


def predict_IP(df, ip_coefs, a_corr):
    """IP = [β_μ·μ + β_η·η + β_n·n] × (1 + a · f²_min_all), n = 2η/I² with I = η/2 − μ."""
    g = g_born(df["epsilon"].values)
    mu, eta = df["mu"].values, df["eta"].values
    I = eta/2.0 - mu
    n = 2.0*eta / (I*I)
    base = pade11_g(g, ip_coefs["mu"])*mu + pade11_g(g, ip_coefs["eta"])*eta + pade11_g(g, ip_coefs["n"])*n
    return base * (1.0 + a_corr * df["f2_min_all"].values)


def predict_Erelax(df, target, model):
    """Erelax = (θ_μ·μ + θ_η·η + θ_ω·ω + θ_LVC·LVC2_paired) × (1 + a · f±_max)."""
    suff = "red" if "red" in target else "ox"
    theta = model["theta"]
    a_corr = model["a_corr"]
    mu, eta, omega = df["mu"].values, df["eta"].values, df["omega"].values
    lvc = df[f"lvc2_{suff}"].values
    fcol = "fplus_max_all" if suff == "red" else "fminus_max_all"
    base = theta[0]*mu + theta[1]*eta + theta[2]*omega + theta[3]*lvc
    return base * (1.0 + a_corr * df[fcol].values)


def predict_ddU(df, target, model):
    """ΔΔU = (θ_μ·μ + θ_η·η) × (1 + a · f²_paired)."""
    fcol = "f2_max_all" if "red" in target else "f2_min_all"
    base = model["theta_mu"]*df["mu"].values + model["theta_eta"]*df["eta"].values
    return base * (1.0 + model["a_corr"] * df[fcol].values)


def predict_TdS(df, target, model):
    """TΔS = (θ_μ·μ + θ_η·η) × (1 + a · f²_paired)."""
    fcol = "f2_max_all" if "red" in target else "f2_min_all"
    base = model["theta_mu"]*df["mu"].values + model["theta_eta"]*df["eta"].values
    return base * (1.0 + model["a_corr"] * df[fcol].values)


# ---------- Main ----------
def main():
    # Load DFT data
    df = pd.read_parquet(DATA)
    df = df[~df.molecule.isin(EXCLUDED)].dropna(
        subset=["EA", "IP", "Erelax_red", "Erelax_ox", "ddU_red", "ddU_ox",
                "TdS_red", "TdS_ox", "mu", "eta", "lvc2_red", "lvc2_ox"]
    ).reset_index(drop=True)
    df["omega"] = df["mu"]**2 / (2.0*df["eta"])
    df["epsilon"] = df["solvent"].map(EPSILON_R)
    print(f"Dataset: n={len(df)}, n_mol={df.molecule.nunique()}, n_solv={df.solvent.nunique()}")

    # Load Fukui features
    print("Loading Fukui descriptors...")
    fk = load_fukui_features(df)
    df = df.drop(columns=[c for c in fk.columns if c in df.columns], errors="ignore")
    df = pd.concat([df, fk], axis=1).dropna(subset=list(fk.columns)).reset_index(drop=True)
    print(f"  n with Fukui complete = {len(df)}")

    # Load all models
    final_solv = json.load(open(RES/"final_solvent_models.json"))
    ea_pade = final_solv["pade11_g"]["targets"]["EA"]["params_per_descriptor"]
    co = pd.read_csv(RES/"fukui_correction_coeffs.csv")
    ea_acorr = float(co[(co.target=="EA") & (co.fukui=="f2_max_noO") & (co.param=="a_corr")]["value"].iloc[0])
    # Use the corrected EA coefficients (jointly refit with a_corr)
    ea_coefs = {}
    for d in ("mu","eta","omega"):
        sub = co[(co.target=="EA") & (co.fukui=="f2_max_noO")]
        ea_coefs[d] = {p: float(sub[sub.param==f"{p}_{d}"]["value"].iloc[0]) for p in ("a0","a1","b1")}

    ip_model = json.load(open(RES/"final_ip_model.json"))
    ip_coefs = {d: ip_model["params"][d] for d in ("mu","eta","n")}
    ip_acorr = float(ip_model["params"]["a_corr"])

    erel_models = json.load(open(RES/"erelax_v2_best_fukui_models.json"))
    ddU_models  = json.load(open(RES/"ddu_minimal_optuna.json"))
    TdS_models  = json.load(open(RES/"tds_final_optuna.json"))

    # Predict each component (kJ/mol)
    EA_pred = predict_EA(df, ea_coefs, ea_acorr)
    IP_pred = predict_IP(df, ip_coefs, ip_acorr)
    Er_red_pred = predict_Erelax(df, "Erelax_red", erel_models["Erelax_red"])
    Er_ox_pred  = predict_Erelax(df, "Erelax_ox",  erel_models["Erelax_ox"])
    dU_red_pred = predict_ddU(df, "ddU_red", ddU_models["ddU_red"])
    dU_ox_pred  = predict_ddU(df, "ddU_ox",  ddU_models["ddU_ox"])
    TS_red_pred = predict_TdS(df, "TdS_red", TdS_models["TdS_red"])
    TS_ox_pred  = predict_TdS(df, "TdS_ox",  TdS_models["TdS_ox"])

    # DFT targets in kJ/mol
    EA_dft = df["EA"].values * EH
    IP_dft = df["IP"].values * EH
    Er_red_dft = df["Erelax_red"].values * EH
    Er_ox_dft  = df["Erelax_ox"].values * EH
    dU_red_dft = df["ddU_red"].values * EH
    dU_ox_dft  = df["ddU_ox"].values * EH
    TS_red_dft = df["TdS_red"].values * EH
    TS_ox_dft  = df["TdS_ox"].values * EH

    # ΔG via rapport formula
    dG_red_pred = -EA_pred + Er_red_pred + dU_red_pred - TS_red_pred
    dG_ox_pred  = +IP_pred + Er_ox_pred  + dU_ox_pred  + TS_ox_pred
    dG_red_dft = -EA_dft + Er_red_dft + dU_red_dft - TS_red_dft
    dG_ox_dft  = +IP_dft + Er_ox_dft  + dU_ox_dft  + TS_ox_dft

    # Per-component MAE for context
    print("\n=== Per-component MAE (kJ/mol) ===")
    for name, pred, dft in [("EA", EA_pred, EA_dft), ("IP", IP_pred, IP_dft),
                              ("E_relax_red", Er_red_pred, Er_red_dft), ("E_relax_ox", Er_ox_pred, Er_ox_dft),
                              ("ΔΔU_red", dU_red_pred, dU_red_dft), ("ΔΔU_ox", dU_ox_pred, dU_ox_dft),
                              ("TΔS_red", TS_red_pred, TS_red_dft), ("TΔS_ox", TS_ox_pred, TS_ox_dft)]:
        print(f"  {name:<14}  MAE={np.mean(np.abs(pred-dft)):6.3f}  RMSE={np.sqrt(np.mean((pred-dft)**2)):6.3f}")

    # ΔG metrics
    print("\n=== ΔG combined (rapport formula) ===")
    for tag, pred, dft in [("ΔG_red", dG_red_pred, dG_red_dft), ("ΔG_ox", dG_ox_pred, dG_ox_dft)]:
        mae = float(np.mean(np.abs(pred - dft)))
        rmse = float(np.sqrt(np.mean((pred - dft)**2)))
        ss_res = float(np.sum((pred - dft)**2))
        ss_tot = float(np.sum((dft - dft.mean())**2))
        r2 = 1.0 - ss_res/ss_tot
        print(f"  {tag}:  MAE={mae:6.3f}  RMSE={rmse:6.3f}  R²={r2:.3f}")

    # Save results
    out_df = pd.DataFrame({
        "molecule": df.molecule, "solvent": df.solvent, "conformer": df.conformer,
        "dG_red_DFT": dG_red_dft, "dG_red_pred": dG_red_pred,
        "dG_ox_DFT":  dG_ox_dft,  "dG_ox_pred":  dG_ox_pred,
        "EA_pred": EA_pred, "EA_DFT": EA_dft,
        "IP_pred": IP_pred, "IP_DFT": IP_dft,
        "Er_red_pred": Er_red_pred, "Er_red_DFT": Er_red_dft,
        "Er_ox_pred": Er_ox_pred,   "Er_ox_DFT":  Er_ox_dft,
        "dU_red_pred": dU_red_pred, "dU_red_DFT": dU_red_dft,
        "dU_ox_pred":  dU_ox_pred,  "dU_ox_DFT":  dU_ox_dft,
        "TS_red_pred": TS_red_pred, "TS_red_DFT": TS_red_dft,
        "TS_ox_pred":  TS_ox_pred,  "TS_ox_DFT":  TS_ox_dft,
    })
    out_df.to_csv(RES/"dG_combined_predictions.csv", index=False)
    print(f"\nSaved -> {RES/'dG_combined_predictions.csv'}")

    # Parity plot 1×2
    palette = plt.get_cmap("tab10").colors
    solv_color = {s: palette[i % 10] for i, s in enumerate(SOLVENT_ORDER)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    targets = [("ΔG_red", dG_red_dft, dG_red_pred, r"$\Delta G_{\rm red}$"),
               ("ΔG_ox",  dG_ox_dft,  dG_ox_pred,  r"$\Delta G_{\rm ox}$")]
    for ax, (tag, y_true, y_pred, title) in zip(axes, targets):
        for solv in SOLVENT_ORDER:
            mask = df.solvent.values == solv
            if mask.sum() == 0: continue
            ax.scatter(y_true[mask], y_pred[mask], c=[solv_color[solv]], s=18, alpha=0.65,
                       edgecolor="none", label=f"{solv} (ε={EPSILON_R[solv]})")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.0, alpha=0.6)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        ss_res = float(np.sum((y_true - y_pred)**2))
        ss_tot = float(np.sum((y_true - y_true.mean())**2))
        r2 = 1.0 - ss_res/ss_tot
        ax.set_title(f"{title}  —  MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}",
                     fontsize=12)
        ax.set_xlabel("DFT (kJ/mol)")
        ax.set_ylabel("Modèle (kJ/mol)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(r"$\Delta G$ combiné via formule du rapport — $\Delta G = \mp$EA/IP + $E_{\rm relax}$ + $\Delta\Delta U \mp T\Delta S$"
                 + f"  —  n={len(df)} conformères, 10 solvants",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(RES/"dG_combined_parity.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(RES/"dG_combined_parity.png", bbox_inches="tight", dpi=180)
    print(f"Saved -> {RES/'dG_combined_parity.pdf'}")
    print(f"Saved -> {RES/'dG_combined_parity.png'}")


if __name__ == "__main__":
    main()
