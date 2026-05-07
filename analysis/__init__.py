"""
Scripts d'analyse post-calcul pour PlamsV5.

Tous les scripts de génération de rapports, corrélations et visualisations.
"""
from pathlib import Path

# Racine du projet (un niveau au-dessus de analysis/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Répertoires standards
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
BATCH_DIR = DATA_DIR / "Acetonitrile_marais"

# 19 catéchols étudiés
CATECHOLS_19 = [
    'HMC', 'F-HMC', 'Cl-HMC', 'Br-HMC', 'I-HMC',
    'DA', 'F-DA', 'Cl-DA', 'Br-DA', 'I-DA',
    'L-DA', 'CA', 'Al-CA', 'HO-CA', 'H2C-CA',
    'DHI', 'O-DHI', 'S-DHI', 'P-DHI',
]

# Molécules exclues : attachement électronique dissociatif (rupture C-X)
DEA_EXCLUDED = {'Br-DA', 'Br-HMC', 'I-DA', 'I-HMC'}

# 15 catéchols stables (sans DEA)
CATECHOLS_15 = [m for m in CATECHOLS_19 if m not in DEA_EXCLUDED]

# Aliases pour compatibilité
MOLECULES_15 = CATECHOLS_19
MOLECULES_11 = CATECHOLS_15
