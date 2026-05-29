#!/usr/bin/env python
"""
RMSD neutre->ion (Kabsch) pour les 154 conformères acétonitrile, vs E_relax DFT.
Géométries lues dans dataset/Acetonitrile/<MOL>/<MOL>_conf<N>_DFT.json
(states.neutral_dzp.xyz / reduced_opt.xyz / oxidized_opt.xyz).
But : interprétabilité — montrer que E_relax n'est PAS qu'une affaire de géométrie.
Style maison (rapport/style.py), points coloriés par molécule (muets), n=154.
"""
import sys, json, glob, os
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, FIGSIZE_DOUBLE_V, C_RED_DARK, C_BLUE_DARK, C_GRAY)
import matplotlib.pyplot as plt

apply_style()
EH = 2625.499638
EXCL = {"Br-DA", "Br-HMC", "I-DA", "I-HMC", "L-DOPA"}
DS = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/dataset/Acetonitrile"
FIG = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"


def parse_xyz(s):
    lines = s.splitlines()
    n = int(lines[0].split()[0])
    el, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        el.append(p[0]); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return el, np.array(xyz)


def kabsch_rmsd(P, Q):
    """RMSD minimal après translation au centroïde + rotation optimale (Kabsch)."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    Pr = Pc @ R.T
    return np.sqrt(np.mean(np.sum((Pr - Qc)**2, axis=1)))


def rmsd_for(js, state):
    el0, x0 = parse_xyz(js["states"]["neutral_dzp"]["xyz"])
    el1, x1 = parse_xyz(js["states"][state]["xyz"])
    if el0 != el1 or len(x0) != len(x1):
        return np.nan
    return kabsch_rmsd(x0, x1)


def collect():
    rows = []
    for f in sorted(glob.glob(f"{DS}/*/*_conf*_DFT.json")):
        base = os.path.basename(f)
        mol = os.path.basename(os.path.dirname(f))
        if mol in EXCL:
            continue
        conf = int(base.split("_conf")[1].split("_")[0])
        try:
            js = json.load(open(f))
            if "states" not in js:
                continue
            rows.append(dict(molecule=mol, conformer=conf,
                             RMSD_red=rmsd_for(js, "reduced_opt"),
                             RMSD_ox=rmsd_for(js, "oxidized_opt")))
        except Exception as e:
            print("  skip", base, e)
    return pd.DataFrame(rows)


def muted(rgba):
    h, s, v = mcolors.rgb_to_hsv(np.array(rgba[:3]))
    return (*mcolors.hsv_to_rgb((h, s * 0.45, min(v, 0.78))), 1.0)


def make(d, xcol, ycol, color, xl, fname):
    x = d[xcol].values; y = d[ycol].values; mols = d.molecule.values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, mols = x[ok], y[ok], mols[ok]
    u = sorted(set(mols))
    pal = {m: muted(c) for m, c in zip(u, plt.cm.gist_rainbow(np.linspace(0, 1, len(u))))}
    cols = [pal[m] for m in mols]
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    ax.scatter(x, y, c=cols, **{k: v for k, v in SCATTER_KW.items() if k != "c"})
    regression_with_band(ax, x, y, color=color)
    r2 = 1 - np.sum((y - np.polyval(np.polyfit(x, y, 1), x))**2) / np.sum((y - y.mean())**2)
    ax.text(0.05, 0.95, f"$R^2 = {r2:.2f}$\n$n = {len(x)}$", transform=ax.transAxes,
            va="top", fontsize=14, family="serif",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                      edgecolor="gray", linewidth=0.6))
    ax.set_xlabel(xl); ax.set_ylabel(r"$E_{\mathrm{relax}}$ DFT (kJ/mol)")
    setup_grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}.png"); fig.savefig(f"{FIG}/{fname}.pdf"); plt.close()
    print(f"  {fname}: R2={r2:.3f}  n={len(x)}")


if __name__ == "__main__":
    rm = collect()
    print(f"RMSD calculés : {len(rm)} conformères")
    m = pd.read_parquet("data/merged_dataset.parquet")
    m = m[(m.solvent == "Acetonitrile") & (~m.molecule.isin(EXCL))].copy()
    m["Erelax_red"] *= EH; m["Erelax_ox"] *= EH
    d = rm.merge(m[["molecule", "conformer", "Erelax_red", "Erelax_ox"]],
                 on=["molecule", "conformer"], how="inner")
    print(f"Fusion E_relax : {len(d)} conformères")
    make(d, "RMSD_ox", "Erelax_ox", C_RED_DARK,
         r"RMSD neutre$\rightarrow$oxydé (\AA)", "erelax_vs_rmsd_ox")
    make(d, "RMSD_red", "Erelax_red", C_BLUE_DARK,
         r"RMSD neutre$\rightarrow$réduit (\AA)", "erelax_vs_rmsd_red")
