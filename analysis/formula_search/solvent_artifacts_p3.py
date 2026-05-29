#!/usr/bin/env python
"""Artefacts multi-solvant pour §3.6/§3.7. Parity plots au STYLE MAISON exact
   (make_panel de parity_plots_p3.py), coloriés par SOLVANT au lieu de molécule.
   Réutilise les prédictions OOF de solvent_screen_p3 (results/..._preds.npz)."""
import sys; sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/analysis/formula_search")
sys.path.insert(0, "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import solvent_screen_p3 as S
from style import (apply_style, regression_with_band, stats_text, setup_grid,
                   SCATTER_KW, FIGSIZE_DOUBLE_V, C_RED_DARK, C_BLUE_DARK, C_GRAY)
import matplotlib.pyplot as plt
apply_style()
EH = S.EH
FIG = "/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/rapport/figures"
EPS = S.EPS
CHOICE = {"EA":"pade11","IP":"pade11","Erelax_red":"const","Erelax_ox":"const",
          "ddU_red":"const","ddU_ox":"const","TdS_red":"const","TdS_ox":"const"}

df = S.load()
P = np.load("results/solvent_screen_p3_preds.npz")
D = {t: df[t].values*EH for t in CHOICE}
pred = {t: P[f"{t}|{CHOICE[t]}"] for t in CHOICE}
dGr = -pred["EA"]+pred["Erelax_red"]+pred["ddU_red"]-pred["TdS_red"]
dGo =  pred["IP"]+pred["Erelax_ox"]+pred["ddU_ox"]-pred["TdS_ox"]
dGrD = -D["EA"]+D["Erelax_red"]+D["ddU_red"]-D["TdS_red"]
dGoD =  D["IP"]+D["Erelax_ox"]+D["ddU_ox"]-D["TdS_ox"]

# couleur par solvant : viridis ordonné par epsilon (violet = apolaire -> jaune = polaire)
solv_order = sorted(df.solvent.unique(), key=lambda s: EPS[s])
solv_color = {s: plt.cm.viridis(i/(len(solv_order)-1)) for i, s in enumerate(solv_order)}
pt_colors = [solv_color[s] for s in df.solvent]

def make_panel(ydft, ypred, color, xlab, ylab, fname, lims=None):
    """Patron identique à parity_plots_p3.make_panel ; points coloriés par solvant."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_V)
    scat = dict(SCATTER_KW); scat.pop("c", None)
    ax.scatter(ydft, ypred, c=pt_colors, **scat)
    regression_with_band(ax, ydft, ypred, color=color)
    if lims is None:
        a = np.concatenate([ydft, ypred]); mg = (a.max()-a.min())*0.12
        lims = [a.min()-mg, a.max()+mg]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=0.8, alpha=0.5, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2 = 1-np.sum((ydft-ypred)**2)/np.sum((ydft-ydft.mean())**2)
    stats_text(ax, r2, np.mean(np.abs(ypred-ydft)), len(ydft))
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); setup_grid(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG}/{fname}.png", dpi=150); plt.savefig(f"{FIG}/{fname}.pdf"); plt.close()
    print(f"  {fname}: MAE={np.mean(np.abs(ypred-ydft)):.2f} R2={r2:.3f}")

print("=== parity multi-solvant (style maison, couleur=solvant) ===")
make_panel(D["EA"], pred["EA"], C_BLUE_DARK, r"$\mathrm{EA}^{\mathrm{DFT}}$ (kJ/mol)",
           r"$\mathrm{EA}$ prédit (kJ/mol)", "parity_ms_ea")
make_panel(D["IP"], pred["IP"], C_RED_DARK, r"$\mathrm{IP}^{\mathrm{DFT}}$ (kJ/mol)",
           r"$\mathrm{IP}$ prédit (kJ/mol)", "parity_ms_ip")
make_panel(dGoD, dGo, C_RED_DARK, r"$\Delta G^\circ_{\mathrm{ox}}$ DFT (kJ/mol)",
           r"$\Delta G^\circ_{\mathrm{ox}}$ prédit (kJ/mol)", "parity_ms_dg_ox")
make_panel(dGrD, dGr, C_BLUE_DARK, r"$\Delta G^\circ_{\mathrm{red}}$ DFT (kJ/mol)",
           r"$\Delta G^\circ_{\mathrm{red}}$ prédit (kJ/mol)", "parity_ms_dg_red")

# légende solvant -> couleur (figurine séparée, réutilisable en sous-figure)
fig, ax = plt.subplots(figsize=(7.0, 0.5))
handles = [plt.Line2D([0],[0], marker='o', ls='', mec='white', mew=0.4,
           mfc=solv_color[s], ms=7, label=f"{s} ($\\varepsilon$={EPS[s]:.1f})") for s in solv_order]
ax.legend(handles=handles, loc='center', ncol=5, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0)
ax.axis('off'); fig.savefig(f"{FIG}/solvent_legend.png", dpi=150); fig.savefig(f"{FIG}/solvent_legend.pdf"); plt.close()
print("  solvent_legend")

# MAE par solvant (barres ox/red) -- inchangé
order = solv_order
df2 = df.assign(eR=np.abs(dGr-dGrD), eO=np.abs(dGo-dGoD))
maeR = [df2[df2.solvent==s].eR.mean() for s in order]
maeO = [df2[df2.solvent==s].eO.mean() for s in order]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
x = np.arange(len(order)); w=0.38
ax.bar(x-w/2, maeO, w, label="Oxydation", color=C_RED_DARK, alpha=0.85)
ax.bar(x+w/2, maeR, w, label="Réduction", color=C_BLUE_DARK, alpha=0.85)
ax.axhspan(4, 8, color=C_GRAY, alpha=0.12, zorder=0); ax.axhline(4, color=C_GRAY, lw=0.7, ls="--", alpha=0.6)
ax.set_ylabel("MAE $\\Delta G^\\circ$ (kJ/mol)"); ax.legend(fontsize=8, frameon=False)
setup_grid(ax)
ax.set_xticks(x); ax.set_xticklabels([f"{s} ($\\varepsilon$={EPS[s]:.1f})" for s in order],
              fontsize=7, rotation=35, ha="right")
fig.subplots_adjust(bottom=0.28); fig.tight_layout()
fig.savefig(f"{FIG}/mae_per_solvent_ms.png", dpi=150); fig.savefig(f"{FIG}/mae_per_solvent_ms.pdf"); plt.close()
print("  mae_per_solvent_ms")

# Tables LaTeX : screen 10 familles IP & EA (inchangé)
scr = pd.read_csv("results/solvent_screen_p3.csv")
FAMLAB = {"const":"Constante","lin_g":"Linéaire-Born","onsager":"Onsager-linéaire",
          "pade11":"Padé[1/1]-Born","pade21":"Padé[2/1]-Born","log":"Logarithme",
          "exp":"Exp.\\ saturante","sigmoid":"Sigmoïde","tanh":"Tanh","power":"Puissance"}
def screen_rows(tgt):
    sub = scr[scr.target==tgt].sort_values("mae")
    rows = []
    for _,r in sub.iterrows():
        mark = "\\,$\\star$" if r.fam=="pade11" else ""
        rows.append(f"{FAMLAB[r.fam]}{mark} & {r.mae:.1f} & {r.rmse:.1f} & {r.r2:.3f} & {int(r.npar)} & {r.aic:.0f} \\\\")
    return "\n".join(rows)
def subtab(tgt, cap, lab):
    return (f"\\begin{{subtable}}[t]{{0.49\\textwidth}}\n\\centering\n\\caption{{{cap}}}\\label{{{lab}}}\n"
            f"\\begin{{tabular}}{{lccccc}}\n\\toprule\nFamille & MAE & RMSE & $R^2$ & $n_{{\\mathrm{{par}}}}$ & AIC \\\\\n"
            f" & {{\\tiny kJ/mol}} & {{\\tiny kJ/mol}} & & & \\\\\n\\midrule\n{screen_rows(tgt)}\n\\bottomrule\n\\end{{tabular}}\n\\end{{subtable}}")
combined = ("\\begin{table}[htbp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n"
            + subtab("IP", "IP vertical (oxydation)", "tab:ms_ip_families") + "\n\\hfill\n"
            + subtab("EA", "EA verticale (réduction)", "tab:ms_ea_families") + "\n"
            "\\caption{Screen des 10 familles de correction de solvant pour l'IP et l'EA verticaux "
            "($n = 1336$), classées par MAE croissante. $\\star$~: famille retenue (Padé[1/1]-Born, "
            "physiquement motivée). MAE et RMSE en kJ/mol.}\n\\label{tab:ms_families}\n\\end{table}")
with open(f"{FIG}/screen_tables.tex","w") as f:
    f.write(combined)
print("  screen_tables.tex (IP+EA côte à côte)")

print("\n=== récap ===")
print(f"EA MAE={np.mean(np.abs(pred['EA']-D['EA'])):.2f}  IP MAE={np.mean(np.abs(pred['IP']-D['IP'])):.2f}")
print(f"ΔG_ox MAE={np.mean(np.abs(dGo-dGoD)):.2f}  ΔG_red MAE={np.mean(np.abs(dGr-dGrD)):.2f}")
print("DONE")
