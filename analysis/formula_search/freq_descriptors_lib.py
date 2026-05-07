"""Frequency-distribution descriptor library.

Reads per-state frequencies and eigenvectors from predict-dg JSONs and
computes a battery of distribution descriptors aimed at predicting the
vibrational components of the redox free energy:
    - ΔΔU = U_ion(geom_neutral) - U_neutral, dominated by high frequencies
    - TΔS = T·S_ion(geom_neutral) - T·S_neutral, sensitive to low frequencies

Descriptor families (computed for each frequency set ω_neutre, ω_red, ω_ox
and for Δω paired in two ways: by energy order [A] and by eigenvector
overlap [B]):
    A. Window counts (multiple bin schemes)
    B. Moments per low-frequency cutoff (mean, std, skew, kurtosis)
    C. Quantiles & extrema (min, max, p10, p25, p50, p75, p90, IQR)
    D. Partial RRHO sums (Σ U_vib, Σ T·S_vib) per window, both standard
       and Grimme qRRHO (low-freq sigmoidal weight, ω0=100 cm⁻¹)
    E. Power moments (mean of ω^p for p ∈ {-2, -1, 1, 2, 3})
    F. Δω-specific: signed mean, |Δω| mean/max, RMS, mode-softening counts
    G. Paired RRHO differences ΣΔU and ΣΔT·S per window (the two terms
       of interest, "honest" baseline derived directly from frequency shift)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurt
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HC_OVER_KB = 1.4387768775039338  # cm·K (h·c/kB)
R_GAS = 8.314462618               # J/(mol·K)
TEMPERATURE = 298.15              # K

QRRHO_OMEGA0 = 100.0              # cm⁻¹ (Grimme default)
QRRHO_ALPHA = 4

# Avogadro · h · c (cm⁻¹ → J/mol via NA·h·c·ω):
NA_HC_J_PER_MOL = 0.011962658     # J·mol⁻¹·cm  (so ZPE_per_mol_J = 0.5 · NA·h·c · ω_cm⁻¹)

# ---------------------------------------------------------------------------
# Window definitions (cm⁻¹)
# ---------------------------------------------------------------------------
WINDOWS_FIXED = [
    (0, 100), (100, 200), (200, 300), (300, 500),
    (500, 800), (800, 1200), (1200, 1600),
    (1600, 2000), (2000, 3000), (3000, 1e9),
]
WINDOWS_CUM_LOW = [
    (0, 100), (0, 200), (0, 300), (0, 500),
    (0, 1000), (0, 1500), (0, 2000),
]
WINDOWS_CUM_HIGH = [
    (1500, 1e9), (2000, 1e9), (2500, 1e9), (3000, 1e9),
]
# Deduplicate by string tag
def _tag(lo, hi):
    hi_s = "inf" if hi >= 1e8 else str(int(hi))
    return f"{int(lo)}_{hi_s}"

_seen = set()
ALL_WINDOWS = []
for w in WINDOWS_FIXED + WINDOWS_CUM_LOW + WINDOWS_CUM_HIGH:
    t = _tag(*w)
    if t not in _seen:
        _seen.add(t)
        ALL_WINDOWS.append(w)

LOW_FREQ_CUTOFFS = [0, 50, 100, 150, 200]  # for moment families with hard cutoffs
ABS_DELTA_THRESHOLDS = [10, 25, 50, 100, 200]  # for Δω count descriptors

# ---------------------------------------------------------------------------
# Per-mode RRHO contributions (kJ/mol)
# ---------------------------------------------------------------------------
def _x(omega_cm1, T):
    return HC_OVER_KB * np.asarray(omega_cm1, dtype=float) / T


def U_per_mode(omega_cm1, T=TEMPERATURE):
    """U_vib per mode in kJ/mol (ZPE + thermal)."""
    omega = np.asarray(omega_cm1, dtype=float)
    x = _x(omega, T)
    zpe = 0.5 * R_GAS * T * x
    with np.errstate(over="ignore", invalid="ignore"):
        thermal = np.where(x > 1e-8, R_GAS * T * x / np.expm1(x), 0.0)
    return (zpe + thermal) * 1e-3


def TS_per_mode(omega_cm1, T=TEMPERATURE):
    """T·S_vib per mode in kJ/mol."""
    omega = np.asarray(omega_cm1, dtype=float)
    x = _x(omega, T)
    with np.errstate(over="ignore", invalid="ignore"):
        a = np.where(x > 1e-8, x / np.expm1(x), 1.0)
        b = np.where(x > 1e-8, -np.log1p(-np.exp(-x)), 0.0)
        S = R_GAS * (a + b)
    return T * S * 1e-3


def qrrho_weight(omega_cm1, omega0=QRRHO_OMEGA0, alpha=QRRHO_ALPHA):
    omega = np.asarray(omega_cm1, dtype=float)
    return 1.0 / (1.0 + (omega0 / np.maximum(omega, 1e-3)) ** alpha)


def U_per_mode_qrrho(omega_cm1, T=TEMPERATURE, omega0=QRRHO_OMEGA0):
    """qRRHO U: weighted average of vibrator U and free-rotor U=0.5·RT."""
    w = qrrho_weight(omega_cm1, omega0)
    U_vib = U_per_mode(omega_cm1, T)
    U_rot = 0.5 * R_GAS * T * 1e-3  # kJ/mol scalar
    return w * U_vib + (1 - w) * U_rot


def TS_per_mode_qrrho(omega_cm1, T=TEMPERATURE, omega0=QRRHO_OMEGA0):
    """qRRHO TS: weighted average of vibrator TS and free-rotor TS (Grimme)."""
    omega = np.asarray(omega_cm1, dtype=float)
    w = qrrho_weight(omega, omega0)
    TS_vib = TS_per_mode(omega, T)
    h = 6.62607015e-34
    c = 2.99792458e10  # cm/s
    kB = 1.380649e-23
    omega_safe = np.maximum(omega, 1.0)
    I_eff = h / (8.0 * np.pi ** 2 * c * omega_safe)  # kg·m²
    I_avg = 1.0e-44
    I_comb = I_eff * I_avg / (I_eff + I_avg)
    arg = np.maximum((8.0 * np.pi ** 3 * I_comb * kB * T) / h ** 2, 1e-30)
    S_rot = 0.5 * R_GAS * (1.0 + np.log(arg))
    TS_rot = T * S_rot * 1e-3
    return w * TS_vib + (1 - w) * TS_rot


# ---------------------------------------------------------------------------
# Pairing strategies
# ---------------------------------------------------------------------------
def pair_by_energy(om_n, om_i):
    """Pairing A: sort both arrays, take element-wise difference."""
    n = min(len(om_n), len(om_i))
    a = np.sort(np.asarray(om_n, dtype=float))[:n]
    b = np.sort(np.asarray(om_i, dtype=float))[:n]
    return b - a, a, b


def pair_by_overlap(disp_n, om_n, disp_i, om_i):
    """Pairing B: Hungarian assignment maximizing |<v_i | v_n>|.

    disp_n, disp_i: list of (n_atoms, 3) arrays of displacements.
    Returns Δω in the order of ion modes (i.e. for ion mode k, Δω[k] = ω_i[k] - ω_n[matched]).
    """
    Vn = np.array([np.asarray(d, dtype=float).ravel() for d in disp_n])
    Vi = np.array([np.asarray(d, dtype=float).ravel() for d in disp_i])
    Vn = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-30)
    Vi = Vi / (np.linalg.norm(Vi, axis=1, keepdims=True) + 1e-30)
    S = np.abs(Vi @ Vn.T)             # rows: ion modes, cols: neutral modes
    row_ind, col_ind = linear_sum_assignment(-S)
    om_n_arr = np.asarray(om_n, dtype=float)
    om_i_arr = np.asarray(om_i, dtype=float)
    delta = om_i_arr[row_ind] - om_n_arr[col_ind]
    return delta, om_n_arr[col_ind], om_i_arr[row_ind], (row_ind, col_ind)


# ---------------------------------------------------------------------------
# Descriptor builders
# ---------------------------------------------------------------------------
def _safe_pow_mean(arr, p):
    arr = np.asarray(arr, dtype=float)
    if p < 0:
        arr = arr[arr > 1.0]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr ** p))


def descriptors_freqset(omega, prefix):
    """Compute all single-set descriptors for a frequency array.

    Returns a flat dict; keys are prefixed by `prefix` (e.g. 'wn', 'wr', 'wo').
    """
    omega = np.asarray(omega, dtype=float)
    out = {}

    # B. moments with low-frequency cutoff
    for cut in LOW_FREQ_CUTOFFS:
        sub = omega[omega >= cut]
        if sub.size == 0:
            continue
        tag = f"cut{cut}"
        out[f"{prefix}_{tag}_n"] = int(sub.size)
        out[f"{prefix}_{tag}_mean"] = float(sub.mean())
        out[f"{prefix}_{tag}_std"] = float(sub.std())
        if cut == 0 and sub.size > 3:
            out[f"{prefix}_{tag}_skew"] = float(_skew(sub))
            out[f"{prefix}_{tag}_kurt"] = float(_kurt(sub))

    # C. quantiles
    if omega.size > 0:
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            out[f"{prefix}_q{int(q * 100)}"] = float(np.quantile(omega, q))
        out[f"{prefix}_min"] = float(omega.min())
        out[f"{prefix}_max"] = float(omega.max())
        out[f"{prefix}_iqr"] = out[f"{prefix}_q75"] - out[f"{prefix}_q25"]

    # A. window counts
    for lo, hi in ALL_WINDOWS:
        m = (omega >= lo) & (omega < hi)
        out[f"{prefix}_N{_tag(lo, hi)}"] = int(m.sum())

    # D. partial RRHO sums
    U = U_per_mode(omega)
    TS = TS_per_mode(omega)
    Uq = U_per_mode_qrrho(omega)
    TSq = TS_per_mode_qrrho(omega)
    out[f"{prefix}_Usum"] = float(U.sum())
    out[f"{prefix}_TSsum"] = float(TS.sum())
    out[f"{prefix}_Usum_q"] = float(Uq.sum())
    out[f"{prefix}_TSsum_q"] = float(TSq.sum())
    for lo, hi in ALL_WINDOWS:
        m = (omega >= lo) & (omega < hi)
        t = _tag(lo, hi)
        out[f"{prefix}_Usum_{t}"] = float(U[m].sum()) if m.any() else 0.0
        out[f"{prefix}_TSsum_{t}"] = float(TS[m].sum()) if m.any() else 0.0
        out[f"{prefix}_Usum_q_{t}"] = float(Uq[m].sum()) if m.any() else 0.0
        out[f"{prefix}_TSsum_q_{t}"] = float(TSq[m].sum()) if m.any() else 0.0

    # E. power moments
    for p in (-2, -1, 1, 2, 3):
        out[f"{prefix}_pm{p}"] = _safe_pow_mean(omega, p)

    return out


def descriptors_delta(delta, prefix):
    """F. Δω-specific descriptors (signed/unsigned, soft/hard counts)."""
    d = np.asarray(delta, dtype=float)
    out = {}
    if d.size == 0:
        return out
    out[f"{prefix}_dmean"] = float(d.mean())
    out[f"{prefix}_dstd"] = float(d.std())
    out[f"{prefix}_dabs_mean"] = float(np.abs(d).mean())
    out[f"{prefix}_dabs_max"] = float(np.abs(d).max())
    out[f"{prefix}_dmax_signed"] = float(d.max())
    out[f"{prefix}_dmin_signed"] = float(d.min())
    out[f"{prefix}_drms"] = float(np.sqrt((d ** 2).mean()))
    out[f"{prefix}_dsoft"] = int(np.sum(d < -1.0))
    out[f"{prefix}_dhard"] = int(np.sum(d > 1.0))
    for thr in ABS_DELTA_THRESHOLDS:
        out[f"{prefix}_dN_abs_gt_{thr}"] = int(np.sum(np.abs(d) > thr))
    return out


def descriptors_paired_rrho(om_n_paired, om_i_paired, prefix):
    """G. Per-window ΣΔU and ΣΔTS where Δ = ion mode - paired neutral mode.

    Bin assignment uses the ion-mode frequency (state of interest)."""
    on = np.asarray(om_n_paired, dtype=float)
    oi = np.asarray(om_i_paired, dtype=float)
    out = {}
    if on.size == 0 or oi.size == 0:
        return out
    dU = U_per_mode(oi) - U_per_mode(on)
    dTS = TS_per_mode(oi) - TS_per_mode(on)
    dUq = U_per_mode_qrrho(oi) - U_per_mode_qrrho(on)
    dTSq = TS_per_mode_qrrho(oi) - TS_per_mode_qrrho(on)
    out[f"{prefix}_dU_total"] = float(dU.sum())
    out[f"{prefix}_dTS_total"] = float(dTS.sum())
    out[f"{prefix}_dUq_total"] = float(dUq.sum())
    out[f"{prefix}_dTSq_total"] = float(dTSq.sum())
    for lo, hi in ALL_WINDOWS:
        m = (oi >= lo) & (oi < hi)
        t = _tag(lo, hi)
        out[f"{prefix}_dU_{t}"] = float(dU[m].sum()) if m.any() else 0.0
        out[f"{prefix}_dTS_{t}"] = float(dTS[m].sum()) if m.any() else 0.0
        out[f"{prefix}_dUq_{t}"] = float(dUq[m].sum()) if m.any() else 0.0
        out[f"{prefix}_dTSq_{t}"] = float(dTSq[m].sum()) if m.any() else 0.0
    return out
