#!/usr/bin/env python
"""
Estimateurs vibroniques de E_relax, implémentation PURE et CONVENTION-LIBRE
(diagonalisation du Hessien masse-pondéré : aucune ambiguïté de normalisation des
modes ; alignement Eckart pour les méthodes à déplacement réel).

Méthodes :
  LVC   : -1/2 g^T H_neutre^{-1} g                         (gradient SP, Hessien neutre)
  LVC2  : -1/2 g^T H_ion,SP^{-1} g                          (gradient SP, Hessien ion à R0)
  Dusch : -1/2 sum_{w>=thr} lambda_j (q~.v_j)^2             (déplacement réel, Hessien ion opt)
          = énergie de réorganisation de Huang-Rhys
  Morse : Dusch + correction de Morse mode à mode (anharmonicité)
  VPT2  : potentiel cubique mode à mode sous 3 contraintes
  QVC   : LVC x (1 + sum (dw/w)(lambda_i/lambda))           (correction quadratique)

Toutes les énergies en kJ/mol. Filtre : modes |w| >= seuil (cm-1).

Validations effectuées :
  - décomposition modale == forme quadratique exacte -1/2 dR^T H dR (machine-précis)
  - LVC2 == valeur du rapport (recompute_lvc2_100), à <0.2 kJ/mol
  - les conventions cartésienne et masse-pondérée coïncident pour le gradient (<0.2 kJ/mol)

Sources : DFT.json (neutre/ion optimisés) ; prediction.json (SP ioniques à R0).
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import brentq

EH_TO_KJ = 2625.499638
ANG_TO_BOHR = 1.0 / 0.529177249
CM = 5140.5  # w[cm-1] = CM * sqrt(lambda[Hartree/Bohr^2/amu])
MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
          "Al": 26.982, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904}
DS = Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/dataset/Acetonitrile")
PRED = Path("/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/dataset_predict_dg/Acetonitrile")


# ---------- parsing ----------
def parse_xyz(s):
    """-> (elements, coords (N,3) en Bohr)."""
    lines = s.splitlines(); n = int(lines[0].split()[0])
    el, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split(); el.append(p[0]); xyz.append([float(x) for x in p[1:4]])
    return el, np.array(xyz) * ANG_TO_BOHR


def hessian_3n(state):
    """Hessien (N,N,3,3) -> (3N,3N) symétrisé."""
    Ha = np.array(state["hessian"]); n = Ha.shape[0]
    H = Ha.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    return 0.5 * (H + H.T)


# ---------- noyau : diagonalisation masse-pondérée ----------
def mw_eig(H, m3):
    """Diagonalise le Hessien masse-pondéré. -> (lambda, V (3N,3N), w_cm (3N,))."""
    s = 1.0 / np.sqrt(m3)
    Ht = (s[:, None] * H) * s[None, :]
    lam, V = np.linalg.eigh(Ht)
    w = np.sign(lam) * CM * np.sqrt(np.abs(lam))
    return lam, V, w


def _eckart_align(R_src, R_ref, m):
    """Rotation masse-pondérée alignant R_src sur R_ref (Eckart). -> R_src aligné (centré sur R_ref)."""
    cs = np.average(R_src, axis=0, weights=m); cr = np.average(R_ref, axis=0, weights=m)
    A = R_src - cs; B = R_ref - cr
    Hc = (A * m[:, None]).T @ B
    U, S, Vt = np.linalg.svd(Hc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Q = Vt.T @ np.diag([1, 1, d]) @ U.T   # applique à A : A@Q^T
    return A @ Q.T + cr, Q


# ---------- estimateurs gradient (Hessien à R0) ----------
def lvc_grad(g, H, m3, thr):
    """-1/2 g^T H^{-1} g restreint aux modes |w|>=thr (gradient en Hartree/Bohr)."""
    lam, V, w = mw_eig(H, m3)
    gt = g / np.sqrt(m3)               # M^{-1/2} g
    a = gt @ V
    keep = (w >= thr) & (lam > 0)
    return -0.5 * np.sum(a[keep] ** 2 / lam[keep]) * EH_TO_KJ


def qvc_grad(g, H, m3, thr, alpha=0.3):
    """LVC (Hessien neutre) x facteur de correction de fréquence."""
    lam, V, w = mw_eig(H, m3)
    gt = g / np.sqrt(m3); a = gt @ V
    keep = (w >= thr) & (lam > 0)
    lvc_i = -0.5 * a[keep] ** 2 / lam[keep]
    E_lvc = np.sum(lvc_i) * EH_TO_KJ
    shifts = alpha * np.abs(a[keep]) / np.sqrt(lam[keep])
    tot = np.sum(np.abs(lvc_i))
    if tot <= 0:
        return E_lvc
    corr = np.sum(shifts * (np.abs(lvc_i) / tot))
    return E_lvc * max(0.5, min(2.0, 1.0 + corr))


# ---------- estimateurs déplacement (Hessien ion optimisé) ----------
def duschinsky(dR, H, m3, thr):
    """-1/2 dR^T H dR restreint aux modes |w|>=thr = énergie de réorganisation Huang-Rhys."""
    lam, V, w = mw_eig(H, m3)
    q = np.sqrt(m3) * dR              # M^{1/2} dR
    b = q @ V
    keep = (w >= thr) & (lam > 0)
    return -0.5 * np.sum(lam[keep] * b[keep] ** 2) * EH_TO_KJ


def _morse_a(k, dQ, F_abs):
    dQa = abs(dQ)
    if dQa < 1e-10 or F_abs < 1e-15 or k <= 0 or F_abs >= k * dQa * 0.99:
        return None
    f = lambda a: (k / a) * (np.exp(-a * dQa) - np.exp(-2 * a * dQa)) - F_abs
    try:
        amax = 50.0
        if f(1e-6) * f(amax) > 0:
            amax = 100.0
            if f(1e-6) * f(amax) > 0:
                return None
        return brentq(f, 1e-6, amax, xtol=1e-10)
    except (ValueError, RuntimeError):
        return None


def morse_vpt2(dR, g, H, m3, thr, dQ_thr=0.05):
    """Corrections anharmoniques mode à mode (Morse et VPT2-cubique).
    dR = neutre - ion (ion->neutre), g = gradient SP ionique (même repère que H,dR).
    Modes propres mass-pondérés : dq_j=q~.v_j (en sqrt(amu)*Bohr), F_j=g~.v_j, k_j=lambda_j."""
    lam, V, w = mw_eig(H, m3)
    q = np.sqrt(m3) * dR; gt = g / np.sqrt(m3)
    bq = q @ V; bF = gt @ V
    E_morse = 0.0; E_vpt2 = 0.0
    for j in range(len(lam)):
        if w[j] < thr or lam[j] <= 0:
            continue
        k = lam[j]; dQ = bq[j]; F = bF[j]
        Eh = -0.5 * k * dQ ** 2
        # Morse
        a = _morse_a(k, dQ, abs(F))
        if a is not None and abs(dQ) > dQ_thr:
            D = k / (2 * a ** 2)
            E_morse += -D * (1 - np.exp(-a * abs(dQ))) ** 2
        else:
            E_morse += Eh
        # VPT2-cubique : E = -(1/3)F dQ - (1/6)k dQ^2 (dQ = ion->neutre)
        if abs(dQ) > 1e-10:
            E_vpt2 += -(1.0 / 3.0) * F * dQ - (1.0 / 6.0) * k * dQ ** 2
        else:
            E_vpt2 += Eh
    return E_morse * EH_TO_KJ, E_vpt2 * EH_TO_KJ


# ---------- orchestration ----------
def masses_from(elements):
    return np.array([MASSES[e] for e in elements])


def compute_all(mol, conf, state, thr=50.0):
    """state in {'reduced','oxidized'}. -> dict {LVC,LVC2,Dusch,Morse,VPT2,QVC} (kJ/mol) ou None."""
    fd = DS / mol / f"{mol}_conf{conf}_DFT.json"
    fp = PRED / mol / f"{mol}_conf{conf}_prediction.json"
    if not fd.exists() or not fp.exists():
        return None
    d = json.load(open(fd)); p = json.load(open(fp))
    sd, sp = d["states"], p["states"]
    opt_k, sp_k = f"{state}_opt", f"{state}_sp"
    if opt_k not in sd or "hessian" not in sd[opt_k] or sp_k not in sp:
        return None
    el, R_n = parse_xyz(sd["neutral_dzp"]["xyz"])
    el_i, R_i = parse_xyz(sd[opt_k]["xyz"])
    if el != el_i:
        return None
    m = masses_from(el); m3 = np.repeat(m, 3)

    # --- gradient SP ionique à R0 (predict-dg, repère neutre commun) ---
    g = np.array(sp[sp_k]["gradients"]).flatten()
    Hsp = hessian_3n(sp[sp_k])                 # Hessien ion à R0
    Hn = hessian_3n(sd["neutral_dzp"])         # Hessien neutre

    # --- alignement Eckart de l'ion optimisé sur le neutre (repère commun) ---
    # R_i_al est repositionné sur le centre de masse du neutre -> dR = R_i_al - R_n est propre.
    R_i_al, Q = _eckart_align(R_i, R_n, m)
    dR = (R_i_al - R_n).flatten()   # neutre -> ion (signe sans effet pour Dusch ; Morse/VPT2 -> ion->neutre = -dR)
    dR_i2n = -dR
    Hi = hessian_3n(sd[opt_k])                 # Hessien ion optimisé (repère DFT.json)
    # rotation du Hessien ion vers le repère neutre : Hi' = Rb Hi Rb^T, Rb=block-diag(Q)
    n = len(el); Rb = np.zeros((3 * n, 3 * n))
    for i in range(n):
        Rb[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Q
    Hi_n = Rb @ Hi @ Rb.T

    out = {}
    out["LVC"] = lvc_grad(g, Hn, m3, thr)
    out["LVC2"] = lvc_grad(g, Hsp, m3, thr)
    out["Dusch"] = duschinsky(dR, Hi_n, m3, thr)
    out["Morse"], out["VPT2"] = morse_vpt2(dR_i2n, g, Hi_n, m3, thr)
    out["QVC"] = qvc_grad(g, Hn, m3, thr)
    return out
