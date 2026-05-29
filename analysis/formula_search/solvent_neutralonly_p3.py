#!/usr/bin/env python
"""Palier le plus économique : 1 SP NEUTRE, aucun calcul ionique.
   Seul descripteur perdu = LVC (gradients ioniques). E_relax retombe sur (η)×(1+a·f²).
   EA/IP/ΔΔU/TΔS inchangés (descripteurs neutres). Recombine ΔG + parity plots."""
import sys; sys.path.insert(0,"/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/analysis/formula_search")
import numpy as np
import solvent_screen_p3 as S
import solvent_artifacts_p3 as A   # fournit make_panel, df, D, pred (EA/IP/ddU/TdS), couleurs, dGoD/dGrD
EH = S.EH; df = A.df
print("\n===== PALIER 1-SP-NEUTRE (sans LVC) =====")
# E_relax sans LVC : (eta) x (1 + a f2), const en eps
Eo = S.fit_family(df, ["eta"], "f2_min", "Erelax_ox", "const", True, S.trials_for(2))
Er = S.fit_family(df, ["eta"], "f2_max", "Erelax_red", "const", True, S.trials_for(2))
print(f"E_relax sans LVC : ox MAE={Eo['mae']:.2f} (vs 3.14 avec LVC) | red MAE={Er['mae']:.2f} (vs 2.58)")
# ΔG neutre-seul = EA/IP/ddU/TdS inchangés + E_relax sans LVC
dGo = A.pred["IP"] + Eo["pred"] + A.pred["ddU_ox"] - A.pred["TdS_ox"]
dGr = -A.pred["EA"] + Er["pred"] + A.pred["ddU_red"] - A.pred["TdS_red"]
def mae(a,b): return float(np.mean(np.abs(a-b)))
def r2(d,p): return 1-np.sum((d-p)**2)/np.sum((d-d.mean())**2)
print(f"\nΔG_ox  neutre-seul : MAE={mae(A.dGoD,dGo):.2f}  R2={r2(A.dGoD,dGo):.3f}   (vs 3.84 au palier 3 SP)")
print(f"ΔG_red neutre-seul : MAE={mae(A.dGrD,dGr):.2f}  R2={r2(A.dGrD,dGr):.3f}   (vs 3.59 au palier 3 SP)")
# parity plots style maison, couleur solvant
A.make_panel(A.dGoD, dGo, A.C_RED_DARK, r"$\Delta G^\circ_{\mathrm{ox}}$ DFT (kJ/mol)",
             r"$\Delta G^\circ_{\mathrm{ox}}$ prédit (kJ/mol)", "parity_ms_dg_ox_neutralonly")
A.make_panel(A.dGrD, dGr, A.C_BLUE_DARK, r"$\Delta G^\circ_{\mathrm{red}}$ DFT (kJ/mol)",
             r"$\Delta G^\circ_{\mathrm{red}}$ prédit (kJ/mol)", "parity_ms_dg_red_neutralonly")
print("DONE")
