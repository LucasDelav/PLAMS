# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PlamsV5 est un systeme de workflows de chimie computationnelle construit sur l'Amsterdam Modeling Suite (AMS) et PLAMS (Python Library for Automating Molecular Simulations). Le projet implemente quatre pipelines principaux :

1. **Conformer Generation & Optimization** (qmconf) - Pipeline simplifie en 4 etapes
2. **Redox Potential Calculations** (qmredox) - Schema composite TZ2P+DZP, reutilisation du neutre
3. **QTAIM/IQA Topological Analysis** (qmqtaim) - Analyse topologique et decomposition energetique
4. **Frequency-Edef Correlation Analysis** (qmanalysis) - Correlations frequences vibrationnelles vs Edef

### Changements par rapport a V4

- **Conformeres** : 6 etapes -> 4 etapes (suppression score DFT et filter RMSD)
- **GFN1-xTB** remplace DFTB generique pour l'optimisation semi-empirique
- **DFT opt+freq+solvant** obligatoire (etape 4 des conformeres)
- **Neutre non recalcule** dans le pipeline redox : reutilisation du RKF de l'etape 4
- **Schema composite TZ2P+DZP** : Ee(TZ2P) + corrections thermo(DZP)
- **Seuil Boltzmann** : 0.10 (etait 0.05)
- **neutral_rkf_paths.json** : pont entre conformeres et redox

## Architecture du Projet

### Structure des Repertoires

```
PlamsV5/
├── conformers_gen.py            # Script principal conformeres (4 etapes)
├── redox_calc.py                # Script principal redox (composite TZ2P+DZP)
├── qtaim_calc.py                # Script principal QTAIM
│
├── common/                      # Module partage entre tous les pipelines
│   ├── __init__.py
│   ├── constants.py             # Constantes physiques et conversions
│   ├── frequency.py             # Verification de frequences imaginaires (+solvant)
│   ├── functional.py            # Configuration des fonctionnelles DFT
│   ├── multi_molecule_data.py   # Collecte de donnees multi-molecules
│   ├── utils.py                 # Utilitaires partages (Boltzmann, etc.)
│   └── workspace.py             # Gestion unifiee des workdirs
│
├── qmconf/                      # Pipeline conformeres
├── qmredox/                     # Pipeline redox
├── qmqtaim/                     # Pipeline QTAIM/IQA
├── qmanalysis/                  # Module d'analyse frequence-Edef
├── visualization/               # Module de visualisation
│
├── analysis/                    # Scripts d'analyse post-calcul
├── slurm/                       # Scripts SLURM pour le cluster
└── docs/                        # Documentation technique
```

### Organisation des Workdirs

```
<molecule_name>_workdir/
├── conformers/
│   ├── calculations/
│   │   ├── 01_generate/          # RDKit + UFF
│   │   ├── 02_optimize/          # GFN1-xTB
│   │   ├── boltzmann_filtered/   # XYZ apres filtre Boltzmann
│   │   └── 03_dft_opt/           # PBE0/DZP + freq + solvant
│   │       ├── conf_1/plams_workdir/conf_1_dft_opt/adf.rkf  <- REUTILISE par redox
│   │       └── conf_2/...
│   └── results/
│       ├── conf_1.xyz
│       ├── conf_2.xyz
│       ├── conformers_summary.txt
│       └── neutral_rkf_paths.json   # Pont vers pipeline redox
│
├── redox/
│   ├── calculations/plams_workdir/
│   │   ├── <mol>_conf_1_neutral_tz2p/     # TZ2P SP (charge=0)
│   │   ├── <mol>_conf_1_reduced_sp/       # DZP SP (charge=-1)
│   │   ├── <mol>_conf_1_reduced_opt/      # DZP GeomOpt+freq
│   │   ├── <mol>_conf_1_reduced_tz2p/     # TZ2P SP
│   │   ├── <mol>_conf_1_oxidized_sp/      # DZP SP (charge=+1)
│   │   ├── <mol>_conf_1_oxidized_opt/     # DZP GeomOpt+freq
│   │   └── <mol>_conf_1_oxidized_tz2p/    # TZ2P SP
│   └── results/
│       ├── <mol>_redox_analysis.txt
│       ├── <mol>_redox_summary.csv
│       ├── <mol>_redox_conformers.csv
│       ├── <mol>_orbital_data.csv
│       ├── geometries/
│       └── rkf/                 # Symlinks vers RKF cles
│
└── qtaim/
    ├── calculations/
    └── results/
```

## Environment Setup

```bash
# Source AMS environment first
source ~/ams2025.102/amsbashrc.sh

# Then run the executable scripts directly
./conformers_gen.py [options...]
./redox_calc.py [options...]
./qtaim_calc.py [options...]
```

## Commandes Principales

### 1. Generation de Conformeres (4 etapes)

```bash
./conformers_gen.py --name <molecule_name> --smiles "<SMILES>" \
                    --functional PBE0 --basis DZP \
                    --solvent Acetonitrile \
                    --boltzmann-min-weight 0.10

# Mode analyse seule
./conformers_gen.py --name <molecule_name> --analysis-only \
                    --workdir ./<molecule>_workdir
```

**Pipeline en 4 etapes :**

| Etape | Description | Methode |
|-------|-------------|---------|
| 1 | Generate | RDKit + UFF |
| 2 | Optimize | GFN1-xTB |
| 3 | Boltzmann filter | Seuil 0.10 sur energies GFN1-xTB |
| 4 | DFT opt + freq + solvant | PBE0/DZP + NormalModes + COSMO |

Arguments CLI :
- `--name` : nom de la molecule (requis)
- `--smiles` : SMILES d'entree (requis sauf --analysis-only)
- `--functional` : fonctionnelle DFT (defaut: PBE0)
- `--basis` : base ADF (defaut: DZP)
- `--solvent` : solvant COSMO (defaut: Acetonitrile)
- `--boltzmann-min-weight` : poids minimal Boltzmann (defaut: 0.10)
- `--temperature` : temperature pour Boltzmann (defaut: 298.15 K)
- `--analysis-only` + `--workdir` : re-analyser sans refaire les calculs

**Sorties cles :**
- `results/conf_*.xyz` : geometries finales
- `results/neutral_rkf_paths.json` : chemins vers adf.rkf du neutre (pont vers redox)
- `results/conformers_summary.txt` : poids de Boltzmann

### 2. Calculs Redox (schema composite TZ2P+DZP)

```bash
# Mode calculs complets
./redox_calc.py --input-dir <molecule>_workdir/conformers/results \
                --name <molecule_name> \
                --functional PBE0 --basis DZP --refinement-basis TZ2P \
                --solvent Acetonitrile

# Mode analyse seule
./redox_calc.py --analysis-only --workdir <molecule>_workdir \
                --name <molecule_name> --u-method direct
```

**Pipeline par conformere (7 calculs, le neutre n'est PAS re-optimise) :**

| # | Calcul | Charge | Base | COSMO | Type |
|---|--------|--------|------|-------|------|
| - | neutre | 0 | DZP | CONJ | REUTILISE depuis conformeres |
| 1 | neutral_tz2p | 0 | TZ2P | CONJ | SP sur geom neutre opt |
| 2 | reduced_sp | -1 | TZ2P | CONJ CORR | SP sur geom neutre |
| 3 | reduced_opt | -1 | DZP | CONJ CORR | GeomOpt + freq + solvant |
| 4 | reduced_tz2p | -1 | TZ2P | CONJ CORR | SP sur geom red opt |
| 5 | oxidized_sp | +1 | TZ2P | CONJ CORR | SP sur geom neutre |
| 6 | oxidized_opt | +1 | DZP | CONJ CORR | GeomOpt + freq + solvant |
| 7 | oxidized_tz2p | +1 | TZ2P | CONJ CORR | SP sur geom ox opt |

Arguments CLI :
- `--input-dir` : repertoire conformers/results/ contenant XYZ et neutral_rkf_paths.json
- `--name` : nom de la molecule (requis)
- `--functional` : fonctionnelle DFT (defaut: PBE0)
- `--basis` : base DZP pour opt+SP (defaut: DZP)
- `--refinement-basis` : base de raffinement pour SP composites (defaut: TZ2P)
- `--solvent` : solvant COSMO (defaut: Acetonitrile)
- `--temperature` : temperature pour Boltzmann (defaut: 298.15 K)
- `--u-method` : methode de calcul de U (direct ou h_minus_rt)
- `--analysis-only` + `--workdir` : analyser des calculs existants
- `--predict-dg` : mode criblage rapide (3 SP+NormalModes au lieu de 7 calculs)

### 2b. Mode Predict-DG (criblage rapide)

```bash
# Mode predict-dg : 3 SP+NormalModes avec ConceptualDFT + QTAIM
./redox_calc.py --predict-dg \
                --input-dir <molecule>_workdir/conformers/results \
                --name <molecule_name> \
                --functional PBE0 --basis DZP \
                --solvent Acetonitrile

# Analyse seule en mode predict-dg
./redox_calc.py --predict-dg --analysis-only \
                --workdir <molecule>_workdir --name <molecule_name>
```

**Pipeline par conformere (3 calculs SP+NormalModes sur geom neutre) :**

| # | Calcul | Charge | Spin | Base | COSMO | Extras |
|---|--------|--------|------|------|-------|--------|
| 1 | neutral_sp | 0 | 0 | DZP | CONJ | ConceptualDFT + QTAIM |
| 2 | reduced_sp | -1 | 1 | DZP | CONJ CORR | ConceptualDFT + QTAIM |
| 3 | oxidized_sp | +1 | 1 | DZP | CONJ CORR | ConceptualDFT + QTAIM |

Ce mode genere les donnees pour predire DeltaG via :
- EA/IP verticaux et Koopmans (HOMO/LUMO)
- Hessien ionique pour LVC2 (E_relax, eq. 9 du rapport)
- Descripteurs ConceptualDFT (Fukui, durete) et QTAIM (charges Bader, E_intra, E_inter)

### 3. Analyse QTAIM/IQA

```bash
./qtaim_calc.py --name <molecule_name> \
                --conformers conf_1 conf_2 \
                --charges 0 -1 +1 \
                --functional BLYP --basis TZP \
                --quality Good --integration-accuracy 5.0
```

## Schema Composite TZ2P+DZP

Le pipeline redox V5 utilise un schema d'energie composite :

```python
# Energie composite : Ee(TZ2P) + corrections thermo(DZP)
G_composite = Ee(TZ2P) + [G(DZP) - Ee(DZP)]

# DeltaG de reduction/oxydation
DG_red = G_composite(red) - G_composite(neutral)
DG_ox  = G_composite(ox)  - G_composite(neutral)

# Decomposition thermodynamique (tous les SP en TZ2P)
EA  = Ee_neutral(TZ2P) - Ee_red_sp(TZ2P)          # Bases coherentes TZ2P
IP  = Ee_ox_sp(TZ2P) - Ee_neutral(TZ2P)            # Bases coherentes TZ2P
Edef = Ee_ion_tz2p(TZ2P) - Ee_ion_sp(TZ2P)         # Bases coherentes TZ2P (deformation pure)
DDU  = [U-Ee](ion_opt_DZP) - [U-Ee](neutral_DZP)   # Corrections DZP
TDS  = TS(ion_opt_DZP) - TS(neutral_DZP)            # Corrections DZP

# Coherence
DG_red = -EA + Edef + DDU + TDS + residual
DG_ox  = IP + Edef + DDU + TDS + residual
```

Le residuel devrait etre quasi-nul (difference entre schema composite TZ2P et decomposition TZ2P+DZP thermo).

## Module Commun (common/)

### Constantes Physiques (common/constants.py)

```python
F = 96485.33212           # Constante de Faraday [C/mol]
R = 8.314462618           # Constante des gaz parfaits [J/(mol*K)]
TEMPERATURE = 298.15      # Temperature par defaut [K]
EH_TO_EV = 27.211386245988    # Hartree -> eV
EH_TO_KJ = 2625.499638        # Hartree -> kJ/mol
BOLTZMANN_MIN_WEIGHT_DEFAULT = 0.10  # Seuil Boltzmann (V5)
```

### Configuration des Fonctionnelles (common/functional.py)

`configure_functional(s, functional)` gere automatiquement la configuration :
- **LDA**: `s.input.adf.XC.LDA = <functional>`
- **GGA**: `s.input.adf.XC.GGA = <functional>`
- **MetaGGA**: `s.input.adf.XC.MetaGGA = <functional>` + NumericalQuality Good
- **Hybrid**: `s.input.adf.XC.Hybrid = <functional>`
- **MetaHybrid**: `s.input.adf.XC.MetaHybrid = <functional>` + NumericalQuality Good

### Verification de Frequences (common/frequency.py)

Supporte desormais le parametre `solvent` pour COSMO :

```python
from common.frequency import check_and_fix_frequencies

job, freq_ok = check_and_fix_frequencies(
    job=job, functional="PBE0", basis="DZP",
    job_base_name="name", charge=-1, spin_polarization=1,
    quality="Good", max_attempts=3, solvent="Acetonitrile"
)
```

### Gestion des Workdirs (common/workspace.py)

```python
from common.workspace import setup_workspace
workdir, calc_dir, results_dir = setup_workspace("molecule", subfolder="conformers")
```

### Utilitaires (common/utils.py)

```python
from common.utils import boltzmann_weights, print_header, print_success, print_error
weights, Z = boltzmann_weights(energies_kj, temperature=298.15)
```

## Pipeline Conformeres (qmconf)

### qmconf/pipeline.py - Fonctions principales

- `generate_conformers(mol, calc_dir)` : Generation RDKit+UFF
- `optimize_conformers(gen_job, calc_dir)` : Optimisation GFN1-xTB
- `filter_boltzmann_weights(opt_job, calc_dir, weight_threshold=0.10)` : Filtrage Boltzmann
- `dft_optimize_with_freq(conformers, name, calc_dir, ...)` : DFT opt+freq+solvant
- `print_results(job)` : Affichage des resultats

### dft_optimize_with_freq()

```python
valid_conformers, valid_indices, rkf_paths = dft_optimize_with_freq(
    conformers, molecule_name, calc_dir,
    functional="PBE0", basis="DZP",
    solvent="Acetonitrile", max_attempts=3
)
# rkf_paths = {conf_num: path_to_adf.rkf}  -> pont vers redox
```

## Pipeline Redox (qmredox)

### qmredox/common.py - setup_adf_settings()

```python
s = setup_adf_settings(
    task='GeometryOptimization',  # ou 'SinglePoint'
    charge=0, spin_polarization=0,
    solvent="Acetonitrile", functional="PBE0", basis="DZP",
    normal_modes=None  # Auto: Yes pour GeomOpt, No pour SinglePoint
)
```

### qmredox/pipeline.py - Pipeline 7 calculs

```python
from qmredox.pipeline import run_redox_pipeline

jobs = run_redox_pipeline(
    mol_neutral=mol,              # Molecule PLAMS
    neutral_rkf_path=rkf_path,    # str: chemin vers adf.rkf du neutre
    name="mol_conf_1",
    solvent="Acetonitrile",
    functional="PBE0",
    basis="DZP",
    refinement_basis="TZ2P",
    max_attempts=3
)
# Retourne: {'neutral_rkf': str, 'neutral_tz2p': job,
#            'reduced_sp': job, 'reduced_opt': job, 'reduced_tz2p': job,
#            'oxidized_sp': job, 'oxidized_opt': job, 'oxidized_tz2p': job}
```

### qmredox/analysis.py - Analyse

```python
from qmredox.analysis import analyze_molecule_redox

results = analyze_molecule_redox(
    mol_name="molecule",
    jobs_data=all_jobs,      # {conf_num: jobs_dict}
    workdir=workdir,
    temperature=298.15,
    u_method="direct"
)
```

Extraction d'energies depuis RKF :
- `Ee` : Energie electronique (`"Energy"/"SCF Bond Energy"`)
- `H` : Enthalpie (`"Thermodynamics"/"Enthalpy"`)
- `G` : Energie libre de Gibbs (`"Thermodynamics"/"Gibbs free Energy"`)
- `TS` : Terme entropique (`"Thermodynamics"/"TS"`)
- `U` : Energie interne (methode `direct` ou `h_minus_rt`)

Pour les SinglePoints (sans thermo) : seul Ee est extrait.

### Conventions de signes

```python
EA = Ee(neutral) - Ee(anion_sp)    # > 0 si capture favorable
IP = Ee(cation_sp) - Ee(neutral)   # > 0
DG_red = -EA + Edef + DDU + TDS + residual
DG_ox  = IP + Edef + DDU + TDS + residual
```

## Pont Conformeres -> Redox

Le fichier `neutral_rkf_paths.json` dans `conformers/results/` fait le lien :

```json
{
  "1": "/absolute/path/to/conf_1_dft_opt/adf.rkf",
  "2": "/absolute/path/to/conf_2_dft_opt/adf.rkf"
}
```

Le script `redox_calc.py` lit ce JSON via `load_conformers_from_neutral_data()` et passe chaque `(mol, rkf_path)` au pipeline redox.

## Dependencies

- **Amsterdam Modeling Suite (AMS)** avec moteur ADF
- **PLAMS** (`scm.plams`)
- **RDKit** (via `scm.conformers`)
- **NumPy**, **SciPy**, **Pandas** (optionnel), **Matplotlib**, **Seaborn** (optionnel)

Tous les scripts utilisent le shebang `#!/usr/bin/env amspython`.

## Style de Code

- **Docstrings**: Format Google style
- **Messages**: `print_success()` (+), `print_error()` (x), `print_warning()` (!)
- **Headers**: `print_header()` pour titres, `print_section()` pour sections
- **Imports**: Module `common` en premier, puis modules specifiques au pipeline

## Cluster-Continuum Solvation Model

### Principe

Pour les solvants protiques (Water, Methanol), le COSMO implicite seul est insuffisant.
Le mode `--cluster-continuum` ajoute des molecules de solvant explicites autour du solute
aux sites de solvatation, puis applique le COSMO implicite par-dessus.

### Solvants supportes

- **Water** (H2O)
- **Methanol** (CH3OH)

### Workflow en 3 phases

```
Phase 1 (phase gazeuse) : Etapes 1-2-3 standard (RDKit + xTB + Boltzmann)
Intermediaire           : Charges xTB -> identification sites -> placement solvant
Phase 2 (augmente)      : xTB opt + Boltzmann + DFT+freq+COSMO sur systemes augmentes
```

### Usage

```bash
./conformers_gen.py --name <molecule> --smiles "<SMILES>" \
                    --solvent Water --cluster-continuum
```

### Constantes cles

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `LABILE_H_CHARGE_THRESHOLD` | 0.10 e | Seuil charge Mulliken pour H labiles |
| `H_BOND_DISTANCE` | 1.9 A | Distance de placement solvant |
| Cone angle | 60 deg | Demi-angle pour placement multiple |

### Sites de solvatation

- **Paires libres** : N(1), O(2), F(3), P(1), S(2), Cl(3) molecules de solvant
- **H labiles** : 1 molecule de solvant par H avec charge >= 0.10 e

### Module `common/cluster_continuum.py`

Fonctions principales :
- `needs_cluster_continuum(solvent)` : detecte si le solvant necessite le mode CC
- `extract_xtb_charges(mol, calc_dir)` : charges Mulliken GFN1-xTB
- `identify_solvation_sites(mol, charges)` : paires libres + H labiles
- `build_cluster_continuum_system(solute, charges, solvent)` : orchestrateur
- `get_solvent_template(solvent)` : geometrie standard du solvant

### Structure workdir (cluster-continuum)

```
<mol>_workdir/conformers/
  calculations/
    phase1/           # Etapes 1-2-3 phase gazeuse
      01_generate/
      02_optimize/
    phase2/           # Etapes 2-3-4 avec solvant
      00_charges/     # xTB SP pour charges
      02_optimize_cc/ # xTB opt systemes augmentes
      augmented_xyz/  # XYZ pour verification visuelle
      03_dft_opt/     # DFT+freq+COSMO
  results/            # Identique au standard
```

### Compatibilite

Le format de sortie (XYZ + `neutral_rkf_paths.json`) est identique au mode standard.
Le pipeline redox fonctionne sans modification sur les resultats cluster-continuum.
