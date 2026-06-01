# PLAMS V5 — Workflows automatisés pour la prédiction de propriétés redox par DFT

Système de workflows de chimie computationnelle pour le criblage haut-débit de matériaux redox-actifs organiques (couple catéchol/quinone), construit sur l'Amsterdam Modeling Suite (AMS) et la bibliothèque PLAMS. Le projet automatise la chaîne complète conformères → DFT → analyses, et explore des modèles prédictifs physiquement motivés pour réduire le coût computationnel du criblage.

## Pipelines

Trois pipelines de calcul, chacun pilotable depuis son script de tête :

| Script | Pipeline | Description |
|---|---|---|
| `conformers_gen.py` | **qmconf** | Génération de conformères (RDKit) → optimisation GFN1-xTB → filtre Boltzmann → DFT opt+freq+solvant |
| `redox_calc.py` | **qmredox** | Calculs redox composite TZ2P+DZP avec réutilisation du neutre, bases cohérentes |
| `qtaim_calc.py` | **qmqtaim** | Analyse topologique QTAIM/IQA et décomposition énergétique |

Et un module d'analyse post-calcul, **qmanalysis**, qui rassemble la recherche de modèles prédictifs (descripteurs CDFT, LVC vibronique, Padé--Born) dans `analysis/formula_search/`.

## Architecture du projet

```
PlamsV5/
├── conformers_gen.py         # Pipeline conformères (étape 1 → 4)
├── redox_calc.py             # Pipeline redox (7 calculs + composite TZ2P+DZP)
├── qtaim_calc.py             # Pipeline QTAIM/IQA
├── common/                   # Module partagé (constantes, frequency, workspace, …)
├── qmconf/, qmredox/, qmqtaim/, qmanalysis/   # Logique de chaque pipeline
├── visualization/            # Génération de figures (post-calcul)
├── analysis/
│   ├── formula_search/       # Scripts actifs de recherche de modèles (partie 3 du rapport)
│   └── archive/              # Scripts ponctuels conservés pour traçabilité
├── slurm/                    # Scripts SLURM pour cluster (CRIANN)
├── docs/                     # Notes méthodologiques
└── rapport/                  # Rapport LaTeX (CARMeN), source dans rapport/parts/
```

Le dossier `analysis/archive/` contient les scripts ponctuels (extractions, balayages, figures one-shot) qui ont servi à la rédaction du rapport et qui sont gardés pour la traçabilité. Les scripts d'analyse encore régulièrement utilisés sont dans `analysis/formula_search/`.

## Caractéristiques clés (V5)

- **Schéma composite TZ2P+DZP** pour les énergies redox : énergie électronique en TZ2P, corrections thermodynamiques en DZP, avec résiduel de décomposition systématiquement nul par construction
- **Réutilisation du neutre optimisé** entre pipelines via `neutral_rkf_paths.json` (le neutre n'est pas recalculé dans le pipeline redox)
- **GFN1-xTB** comme méthode semi-empirique pour l'optimisation des conformères (remplace DFTB)
- **Filtre Boltzmann** sur les énergies libres DFT (seuil 0.10 par défaut)
- **Mode `--predict-dg`** : protocole de criblage rapide en 3 SP (NormalModes sur le neutre, gradient seul sur les ions) au lieu de 7 calculs (gain ~5×)
- **Cluster-continuum solvation** pour les solvants protiques (eau, méthanol) via molécules de solvant explicites + COSMO

## Prérequis

- [Amsterdam Modeling Suite](https://www.scm.com) (AMS) 2025 ou supérieur, avec moteur ADF
- Python 3.10+ (utiliser `amspython` du SDK AMS)
- Bibliothèques : PLAMS, RDKit (inclus dans AMS), NumPy, SciPy, Matplotlib (Pandas, Seaborn optionnels)

Tous les scripts utilisent le shebang `#!/usr/bin/env amspython`.

## Installation

```bash
git clone https://github.com/LucasDelav/PLAMS.git
cd PLAMS
source ~/ams2025.102/amsbashrc.sh    # adapter au chemin de votre install AMS
chmod +x conformers_gen.py redox_calc.py qtaim_calc.py
```

## Exemples d'utilisation

### 1. Génération de conformères (4 étapes)

```bash
./conformers_gen.py --name catechol --smiles "OC1=CC=CC=C1O" \
                    --functional PBE0 --basis DZP \
                    --solvent Acetonitrile --boltzmann-min-weight 0.10
```

Sorties : `catechol_workdir/conformers/results/conf_*.xyz`, `conformers_summary.txt`, et `neutral_rkf_paths.json` (pont vers le pipeline redox).

### 2. Calculs redox composite (7 calculs par conformère)

```bash
./redox_calc.py --input-dir catechol_workdir/conformers/results \
                --name catechol \
                --functional PBE0 --basis DZP --refinement-basis TZ2P \
                --solvent Acetonitrile
```

Sorties : `catechol_workdir/redox/results/` avec analyse complète, géométries optimisées (réduit, oxydé), et orbitales frontières.

### 3. Criblage rapide (mode predict-dg, 3 SP par conformère)

```bash
./redox_calc.py --predict-dg \
                --input-dir catechol_workdir/conformers/results \
                --name catechol --solvent Acetonitrile
```

Inclut ConceptualDFT et QTAIM sur chaque SP pour les modèles prédictifs.

### 4. Analyse QTAIM/IQA

```bash
./qtaim_calc.py --name catechol --conformers conf_1 conf_2 \
                --charges 0 -1 +1 \
                --functional BLYP --basis TZP \
                --quality Good --integration-accuracy 5.0
```

## Documentation

- **`ARCHITECTURE.md`** — guide technique détaillé (architecture interne, conventions, paramètres de chaque étape, schéma composite, cluster-continuum, exemples d'usage avancé)
- **`docs/`** — notes méthodologiques et justifications
- **`rapport/`** — rapport rédigé pour le laboratoire CARMeN, source LaTeX organisée en `rapport/parts/`

## Cadre du projet

Ce travail s'inscrit dans une collaboration entre :

- **Laboratoire CARMeN** (UMR CNRS 6064, Université de Rouen Normandie) — développement du criblage computationnel
- **Green Chemistry Centre of Excellence**, Université d'York — synthèse ciblée des candidats prometteurs

L'objectif applicatif est l'identification de molécules redox-actives organiques (catéchols, quinones substitués) pour la matière active d'électrodes positives dans les batteries de prochaine génération, en alternative aux technologies Li-ion.

## Citation

Si vous utilisez ce code dans une publication, merci de citer :

```
Amsterdam Modeling Suite, SCM, Theoretical Chemistry, Vrije Universiteit,
Amsterdam, The Netherlands, https://www.scm.com

PLAMS — Python Library for Automating Molecular Simulation

PLAMS V5, Lucas Delavoye, Laboratoire CARMeN (UMR CNRS 6064), 2026
https://github.com/LucasDelav/PLAMS
```

## Licence

MIT. Voir le fichier `LICENSE`.
