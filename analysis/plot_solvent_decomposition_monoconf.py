#!/usr/bin/env python3
"""Trace l'influence du solvant (epsilon) sur les 4 termes de la
decomposition de DeltaG_red et DeltaG_ox pour une molecule mono-conformere.

GUAIACOL n'a qu'un seul conformere dans les 10 solvants et ses calculs
DFT sont complets, on lit donc directement le summary_DFT.json sans
moyenne de Boltzmann. CATECHOL est incomplet pour Chloroform, DCM et
Tetrahydrofuran (etats ox/red opt absents).
"""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT_DIR = ROOT / "results" / "figures"
MOLECULE = "GUAIACOL"

# Constantes dielectriques a 298 K (valeurs ADF/COSMO standard)
EPSILON = {
    "Hexane": 1.88,
    "Toluene": 2.38,
    "Chloroform": 4.81,
    "Tetrahydrofuran": 7.58,
    "Dichloromethane": 8.93,
    "Acetone": 20.49,
    "Methanol": 32.66,
    "Acetonitrile": 35.69,
    "DMSO": 46.83,
    "Water": 78.36,
}


def load_summary(solvent: str) -> dict:
    path = DATASET / solvent / MOLECULE / f"{MOLECULE}_summary_DFT.json"
    with path.open() as f:
        return ast.literal_eval(f.read())


def collect() -> tuple[dict[str, dict[str, float]], list[str]]:
    """Retourne ({solvent: {...}} en kJ/mol, liste des solvants ecartes).

    Ecarte automatiquement les solvants dont l'energie electronique du
    neutre est exactement egale a celle d'un autre solvant (bug dataset
    detecte sur Water == Chloroform pour ~43% des molecules).
    """
    raw = {}
    for solvent in EPSILON:
        s = load_summary(solvent)
        d = s["derived"]
        try:
            raw[solvent] = {
                "Ee_neutral": s["states"]["neutral_dzp"]["thermo"]["Ee"],
                "EA": d["EA_vert_DZP"]["kj"],
                "IP": d["IP_vert_DZP"]["kj"],
                "Edef_red": d["Edef_red_DZP"]["kj"],
                "Edef_ox": d["Edef_ox_DZP"]["kj"],
                "ddU_red": d["ddU_red"]["kj"],
                "ddU_ox": d["ddU_ox"]["kj"],
                "TdS_red": d["TdS_red"]["kj"],
                "TdS_ox": d["TdS_ox"]["kj"],
                "dG_red": d["delta_G_red"]["kj"],
                "dG_ox": d["delta_G_ox"]["kj"],
            }
        except KeyError as e:
            print(f"[!] {solvent} : etats incomplets ({e}), ignore")

    dropped = []
    seen = {}
    for solv, vals in raw.items():
        key = round(vals["Ee_neutral"], 8)
        if key in seen:
            print(f"[!] {solv} : Ee_neutral identique a {seen[key]} "
                  "-> donnees suspectes, ignore")
            dropped.append(solv)
        else:
            seen[key] = solv

    out = {s: {k: v for k, v in vals.items() if k != "Ee_neutral"}
           for s, vals in raw.items() if s not in dropped}
    return out, dropped


def plot_decomposition(data: dict, kind: str, out_path: Path) -> None:
    """kind = 'red' ou 'ox'. Genere une figure 2x2."""
    solvents = [s for s in EPSILON if s in data]
    eps = [EPSILON[s] for s in solvents]

    if kind == "red":
        # DG_red = -EA + E_relax + ddU + TdS  (TdS contient deja le signe)
        terms = [
            ("$-\\mathrm{EA}$",            [-data[s]["EA"]       for s in solvents]),
            ("$E_\\mathrm{relax}^{red}$",  [ data[s]["Edef_red"] for s in solvents]),
            ("$\\Delta\\Delta U_{red}$",   [ data[s]["ddU_red"]  for s in solvents]),
            ("$T\\Delta S_{red}$",         [ data[s]["TdS_red"]  for s in solvents]),
        ]
        title = f"Decomposition de $\\Delta G_{{red}}$ vs $\\varepsilon$ — {MOLECULE}"
    else:
        terms = [
            ("$\\mathrm{IP}$",             [ data[s]["IP"]       for s in solvents]),
            ("$E_\\mathrm{relax}^{ox}$",   [ data[s]["Edef_ox"]  for s in solvents]),
            ("$\\Delta\\Delta U_{ox}$",    [ data[s]["ddU_ox"]   for s in solvents]),
            ("$T\\Delta S_{ox}$",          [ data[s]["TdS_ox"]   for s in solvents]),
        ]
        title = f"Decomposition de $\\Delta G_{{ox}}$ vs $\\varepsilon$ — {MOLECULE}"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (label, values) in zip(axes.flat, terms):
        ax.scatter(eps, values, s=70, c="steelblue", edgecolor="black",
                   zorder=3)
        for x, y, name in zip(eps, values, solvents):
            ax.annotate(name, (x, y), xytext=(5, 4),
                        textcoords="offset points", fontsize=7.5,
                        color="dimgray")
        ax.set_xscale("log")
        ax.set_xlabel("$\\varepsilon$ (echelle log)")
        ax.set_ylabel(f"{label}  [kJ/mol]")
        ax.set_title(label, fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        # Ligne de reference a la moyenne pour visualiser l'amplitude
        mean_v = sum(values) / len(values)
        ax.axhline(mean_v, color="firebrick", linestyle="--", linewidth=0.8,
                   alpha=0.6, label=f"moy = {mean_v:.1f}")
        amp = max(values) - min(values)
        ax.legend(title=f"$\\Delta$ = {amp:.1f} kJ/mol",
                  fontsize=8, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, dropped = collect()

    print(f"\n=== {MOLECULE} : decomposition par solvant (kJ/mol) ===")
    header = f"{'solvent':<18}{'eps':>7}  {'-EA':>8}  {'Erlx_r':>8}  {'ddU_r':>8}  {'TdS_r':>8}  {'dG_red':>9} | {'IP':>8}  {'Erlx_o':>8}  {'ddU_o':>8}  {'TdS_o':>8}  {'dG_ox':>9}"
    print(header)
    print("-" * len(header))
    for s in EPSILON:
        if s not in data:
            continue
        d = data[s]
        print(f"{s:<18}{EPSILON[s]:>7.2f}  "
              f"{-d['EA']:>8.1f}  {d['Edef_red']:>8.1f}  {d['ddU_red']:>8.1f}  {d['TdS_red']:>8.2f}  {d['dG_red']:>9.1f} | "
              f"{d['IP']:>8.1f}  {d['Edef_ox']:>8.1f}  {d['ddU_ox']:>8.1f}  {d['TdS_ox']:>8.2f}  {d['dG_ox']:>9.1f}")
    if dropped:
        print(f"\n[!] Solvants ecartes (donnees dupliquees) : {dropped}")

    plot_decomposition(data, "red",
                       OUT_DIR / f"{MOLECULE}_decomposition_red_vs_eps.png")
    plot_decomposition(data, "ox",
                       OUT_DIR / f"{MOLECULE}_decomposition_ox_vs_eps.png")


if __name__ == "__main__":
    main()
