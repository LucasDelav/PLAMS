#!/usr/bin/env python
"""LVC (Hessien NEUTRE) pour les 10 solvants, depuis prediction.json (predict-dg).
   lvc = -1/2 g_ion^T H_neutre^{-1} g_ion, modes |w|>=100 cm-1. Sortie en kJ/mol.
   Vérifié contre data/lvc_acetonitrile_thr100.parquet avant le run complet."""
import json, glob, numpy as np, pandas as pd, re, sys
from pathlib import Path
from vibronic_methods import hessian_3n, mw_eig, lvc_grad, masses_from, parse_xyz
ROOT = Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/dataset_predict_dg")
THR = 100.0
SOLVENTS = ["Acetone","Acetonitrile","Chloroform","Dichloromethane","DMSO",
            "Hexane","Methanol","Tetrahydrofuran","Toluene","Water"]

def lvc_pair(fp):
    j = json.load(open(fp)); st = j["states"]
    el, _ = parse_xyz(st["neutral_sp"]["xyz"]); m3 = np.repeat(masses_from(el), 3)
    Hn = hessian_3n(st["neutral_sp"])
    g_red = np.array(st["reduced_sp"]["gradients"]).flatten()
    g_ox  = np.array(st["oxidized_sp"]["gradients"]).flatten()
    return lvc_grad(g_red, Hn, m3, THR), lvc_grad(g_ox, Hn, m3, THR)

def conf_num(fp):
    mtch = re.search(r"_conf(\d+)_prediction", Path(fp).name)
    return int(mtch.group(1)) if mtch else None
def mol_name(fp):
    return re.sub(r"_conf\d+_prediction\.json$", "", Path(fp).name)
def conf_files(solv):
    return [f for f in sorted(glob.glob(str(ROOT/solv/"*"/"*_conf*_prediction.json")))
            if conf_num(f) is not None]

if __name__ == "__main__":
    # 1) verification acetonitrile
    ref = pd.read_parquet("data/lvc_acetonitrile_thr100.parquet").set_index(["molecule","conformer"])
    errs = []
    for fp in conf_files("Acetonitrile")[:25]:
        mol, c = mol_name(fp), conf_num(fp)
        if (mol, c) not in ref.index: continue
        lr, lo = lvc_pair(fp)
        errs.append(abs(lr - ref.loc[(mol,c),"lvc_red_100"]))
        errs.append(abs(lo - ref.loc[(mol,c),"lvc_ox_100"]))
    print(f"[verif acetonitrile] n={len(errs)//2} max|Δ|={max(errs):.4f} kJ/mol mean={np.mean(errs):.4f}", flush=True)
    if not errs or max(errs) > 0.5:
        print("!! ÉCART TROP GRAND ou vide — méthode incohérente, arrêt.", flush=True); sys.exit(1)
    # 2) run complet 10 solvants
    rows = []
    for solv in SOLVENTS:
        files = conf_files(solv)
        ok = 0
        for fp in files:
            try:
                lr, lo = lvc_pair(fp)
                rows.append({"molecule": mol_name(fp), "solvent": solv, "conformer": conf_num(fp),
                             "lvc_red_100": lr, "lvc_ox_100": lo}); ok += 1
            except Exception as e:
                pass
        print(f"  {solv:16s} {ok}/{len(files)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet("data/lvc_multisolvent_thr100.parquet")
    print(f"SAVED data/lvc_multisolvent_thr100.parquet  ({len(out)} conformères)", flush=True)
