"""
Configuration des fonctionnelles DFT pour ADF.

Ce module gère la configuration des différentes fonctionnelles DFT
dans les Settings PLAMS pour les calculs ADF.
"""

from scm.plams.core.settings import Settings

# ============================================================================
# DÉFINITION DES FONCTIONNELLES PAR CATÉGORIE
# ============================================================================

LDA_FUNCTIONALS = [
    "VWN", "XALPHA", "Xonly", "Stoll"
]

GGA_FUNCTIONALS = [
    "PBE", "RPBE", "revPBE", "PBEsol", "BLYP", "BP86", "PW91", "mPW",
    "OLYP", "OPBE", "KT1", "KT2", "BEE", "BJLDA", "BJPBE", "BJGGA",
    "S12G", "LB94", "mPBE", "B3LYPgauss"
]

METAGGA_FUNCTIONALS = [
    "M06L", "TPSS", "revTPSS", "MVS", "SCAN", "revSCAN", "r2SCAN", "tb-mBJ"
]

HYBRID_FUNCTIONALS = [
    "B3LYP", "B3LYP*", "B1LYP", "O3LYP", "X3LYP", "BHandH", "BHandHLYP",
    "B1PW91", "MPW1PW", "MPW1K", "PBE0", "OPBE0", "TPSSh", "M06",
    "M06-2X", "S12H"
]

METAHYBRID_FUNCTIONALS = [
    "M08-HX", "M08-SO", "M11", "TPSSH", "PW6B95", "MPW1B95", "MPWB1K",
    "PWB6K", "M06-HF", "BMK"
]

# Liste complète de toutes les fonctionnelles supportées
ALL_FUNCTIONALS = (
    ['HF'] + LDA_FUNCTIONALS + GGA_FUNCTIONALS +
    METAGGA_FUNCTIONALS + HYBRID_FUNCTIONALS + METAHYBRID_FUNCTIONALS
)


# ============================================================================
# FONCTION DE CONFIGURATION
# ============================================================================

def configure_functional(s, functional):
    """
    Configure la fonctionnelle DFT dans un objet Settings PLAMS.

    Cette fonction détecte automatiquement la catégorie de la fonctionnelle
    et configure les paramètres ADF appropriés.

    Args:
        s (Settings): Objet Settings PLAMS à configurer
        functional (str): Nom de la fonctionnelle (ex: "PBE0", "BLYP", "HF")

    Returns:
        None: Modifie l'objet Settings en place

    Examples:
        >>> s = Settings()
        >>> configure_functional(s, "PBE0")
        >>> configure_functional(s, "BLYP")
        >>> configure_functional(s, "HF")
    """
    # Vérifier si la fonctionnelle est reconnue
    if functional not in ALL_FUNCTIONALS:
        print(f"⚠ Attention : Fonctionnelle '{functional}' non reconnue, "
              f"utilisation de PBE0 par défaut")
        functional = "PBE0"

    # Hartree-Fock
    if functional == 'HF':
        s.input.adf.XC.HF = "Yes"
        return

    # LDA
    if functional in LDA_FUNCTIONALS:
        s.input.adf.XC.LDA = functional
        return

    # GGA
    if functional in GGA_FUNCTIONALS:
        s.input.adf.XC.GGA = functional
        return

    # MetaGGA (nécessite NumericalQuality = Good)
    if functional in METAGGA_FUNCTIONALS:
        s.input.adf.XC.MetaGGA = functional
        s.input.adf.NumericalQuality = "Good"
        return

    # Hybrid
    if functional in HYBRID_FUNCTIONALS:
        s.input.adf.XC.Hybrid = functional
        return

    # MetaHybrid (nécessite NumericalQuality = Good)
    if functional in METAHYBRID_FUNCTIONALS:
        s.input.adf.XC.MetaHybrid = functional
        s.input.adf.NumericalQuality = "Good"
        return
