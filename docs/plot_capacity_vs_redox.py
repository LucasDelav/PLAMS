#!/usr/bin/env python3
"""Plot gravimetric capacity vs couple potential for 18 catechol/quinone redox couples.
Shows discharge voltage, charge voltage, midpoint, and hysteresis bar for each couple.
Comparison with commercial Li-ion cathode materials."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib.patches import FancyBboxPatch

E_LI = 1.24  # Absolute Li/Li+ potential (V)

# 18 couples (excluding L-DA/L-DOPA which has computation issues)
# (label, E_red_quinone, |E_ox_catechol|, M_quinone)
couples = [
    ("HMC/HMQ",       4.263, 5.680, 138.12),
    ("F-HMC/F-HMQ",   4.279, 5.686, 156.11),
    ("Cl-HMC/Cl-HMQ", 4.351, 5.687, 172.57),
    ("Br-HMC/Br-HMQ", 4.348, 5.686, 217.02),
    ("I-HMC/I-HMQ",   4.379, 5.700, 264.02),
    ("DA/DOPA",        4.257, 5.619, 151.16),
    ("F-DA/F-DOPA",    4.285, 5.610, 169.15),
    ("Cl-DA/Cl-DOPA",  4.360, 5.652, 185.61),
    ("Br-DA/Br-DOPA",  4.348, 5.665, 230.06),
    ("I-DA/I-DOPA",    4.373, 5.664, 277.06),
    ("CA/CQ",          4.455, 5.776, 178.14),
    ("Al-CA/Al-CQ",    4.537, 5.798, 162.14),
    ("HO-CA/HO-CQ",   4.242, 5.505, 164.16),
    ("H2C-CA/H2C-CQ", 4.263, 5.319, 160.17),
    ("DHI/IQ",         4.149, 5.216, 147.13),
    ("O-DHI/O-IQ",     4.280, 5.448, 148.12),
    ("S-DHI/S-IQ",     4.365, 5.517, 164.19),
    ("P-DHI/P-IQ",     4.308, 5.517, 164.10),
]

# Compute metrics
data = []
for name, e_red_q, e_ox_c, m_q in couples:
    V_red = e_red_q - E_LI
    V_ox = e_ox_c - E_LI
    V_mid = (V_red + V_ox) / 2
    hyst = V_ox - V_red
    Q = 2 * 26801.0 / m_q
    W_mid = Q * V_mid
    W_discharge = Q * V_red
    data.append((name, V_red, V_ox, V_mid, hyst, Q, W_mid, W_discharge))

# --- Figure ---
fig, ax = plt.subplots(figsize=(14, 9))

Q_range = np.linspace(150, 420, 300)

# Commercial Li-ion reference areas
# LiFePO4: 170 mAh/g, 3.4 V
ax.plot(170, 3.4, 'k^', markersize=14, zorder=10, markeredgewidth=1.5,
        markerfacecolor='white', markeredgecolor='black')
ax.annotate(u'LiFePO\u2084\n170 mAh/g, 3.4 V\n(578 Wh/kg)',
            (170, 3.4), xytext=(178, 3.55), fontsize=8, fontweight='bold',
            ha='left', va='bottom')

# NMC622: 180 mAh/g, 3.7 V
ax.plot(180, 3.7, 'k^', markersize=14, zorder=10, markeredgewidth=1.5,
        markerfacecolor='#CCCCCC', markeredgecolor='black')
ax.annotate('NMC622\n180 mAh/g, 3.7 V\n(666 Wh/kg)',
            (180, 3.7), xytext=(188, 3.85), fontsize=8, fontweight='bold',
            ha='left', va='bottom')

# NMC811: 200 mAh/g, 3.7 V
ax.plot(200, 3.7, 'k^', markersize=14, zorder=10, markeredgewidth=1.5,
        markerfacecolor='#888888', markeredgecolor='black')
ax.annotate('NMC811\n200 mAh/g, 3.7 V\n(740 Wh/kg)',
            (200, 3.7), xytext=(208, 3.58), fontsize=8, fontweight='bold',
            ha='left', va='bottom')

# Iso-W curves
for W_t in [400, 600, 800, 1000, 1200, 1400]:
    V_iso = W_t / Q_range
    mask = (V_iso >= 2.5) & (V_iso <= 4.8)
    ax.plot(Q_range[mask], V_iso[mask], '--', color='grey', alpha=0.2, linewidth=0.8)
    # label at right edge
    q_at = min(W_t / 2.7, 410)
    v_at = W_t / q_at
    if 2.6 < v_at < 4.7:
        ax.text(q_at + 2, v_at, f'{W_t}', fontsize=6.5, color='grey',
                alpha=0.4, style='italic', va='center')

# Color by family
family_colors = {
    'HMC': '#2ECC71',   # green
    'DA': '#3498DB',     # blue
    'CA': '#E74C3C',     # red
    'DHI': '#9B59B6',    # purple
}

def get_family(name):
    if 'HMC' in name or 'HMQ' in name:
        return 'HMC'
    elif 'DA' in name or 'DOPA' in name:
        return 'DA'
    elif 'CA' in name or 'CQ' in name:
        return 'CA'
    else:
        return 'DHI'

family_labels = {
    'HMC': 'HMC/HMQ (hydroxymethyl)',
    'DA': 'DA/DOPA (dopamine)',
    'CA': 'CA/CQ (ac. cafeique)',
    'DHI': 'DHI/IQ (indole)',
}

# Plot each couple as a vertical bar (discharge → charge) with midpoint dot
plotted_families = set()
for name, v_red, v_ox, v_mid, hyst, Q, w_mid, w_dis in data:
    fam = get_family(name)
    color = family_colors[fam]

    # Vertical bar: discharge (bottom) to charge (top)
    ax.plot([Q, Q], [v_red, v_ox], '-', color=color, linewidth=2.5, alpha=0.5, zorder=4)

    # Discharge dot (filled)
    label_dis = family_labels[fam] if fam not in plotted_families else None
    ax.plot(Q, v_red, 'o', color=color, markersize=8, markeredgecolor='black',
            markeredgewidth=0.8, zorder=6, label=label_dis)

    # Charge dot (open)
    ax.plot(Q, v_ox, 'o', color='white', markersize=7, markeredgecolor=color,
            markeredgewidth=1.8, zorder=6)

    # Midpoint (small cross)
    ax.plot(Q, v_mid, '+', color='black', markersize=6, markeredgewidth=1.2, zorder=7)

    plotted_families.add(fam)

    # Labels for key couples
    short = name.split('/')[0]
    if short in ('HMC', 'DA', 'CA', 'Al-CA', 'DHI', 'O-DHI',
                 'I-HMC', 'I-DA', 'Br-HMC', 'Br-DA', 'Cl-HMC'):
        ox, ha = 6, 'left'
        if short in ('DHI', 'Al-CA'):
            ox, ha = -6, 'right'
        ax.annotate(short, (Q, v_mid), xytext=(ox, 0),
                    textcoords='offset points', fontsize=7, ha=ha, va='center',
                    fontweight='bold', color='#444444')

# Legend entries for bar meaning
ax.plot([], [], 'o', color='grey', markersize=8, markeredgecolor='black',
        markeredgewidth=0.8, label=u'V d\u00e9charge (E$_{red}$ quinone)')
ax.plot([], [], 'o', color='white', markersize=7, markeredgecolor='grey',
        markeredgewidth=1.8, label=u'V charge (|E$_{ox}$| cat\u00e9chol)')
ax.plot([], [], '+', color='black', markersize=6, markeredgewidth=1.2,
        label=u'V$_{mid}$ (\u00e9quilibre du couple)')
ax.plot([], [], 'k^', markersize=10, markeredgewidth=1.2,
        markerfacecolor='white', label=u'Li-ion commercial (mat. active)')

ax.set_xlabel(u'Capacit\u00e9 gravim\u00e9trique Q = 2F/M (mAh/g)', fontsize=13,
              fontweight='bold')
ax.set_ylabel('V vs Li/Li+ (V)', fontsize=13, fontweight='bold')
ax.set_title(u'Potentiel de couple cat\u00e9chol/quinone vs capacit\u00e9 massique\n'
             u'18 couples biosourc\u00e9s (Ac\u00e9tonitrile, PBE0/TZ2P)',
             fontsize=14, fontweight='bold')

ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95, ncol=2,
          handletextpad=0.5, columnspacing=1.0)
ax.grid(True, alpha=0.15, linestyle='--')
ax.set_xlim(155, 420)
ax.set_ylim(2.5, 4.8)

# Annotation: hysteresis explanation
ax.annotate('', xy=(392, 3.14), xytext=(392, 4.46),
            arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.5))
ax.text(397, 3.80, u'Hyst\u00e9r\u00e8se\n(surtension\nde cyclage)',
        fontsize=7.5, color='#666666', ha='left', va='center', style='italic')

plt.tight_layout()
plt.savefig('/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/docs/capacity_vs_redox.png',
            dpi=200, bbox_inches='tight')
print("Figure saved: docs/capacity_vs_redox.png")
