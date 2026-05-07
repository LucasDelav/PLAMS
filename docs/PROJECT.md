# PlamsV4 - Pipeline d'Analyse Structure-Propriétés Redox des Quinones

## Objectif du Projet

Ce projet implémente un **pipeline automatisé de chimie computationnelle** pour l'étude des propriétés redox d'une famille de molécules : les **quinones** et leurs dérivés.

L'objectif scientifique est d'établir des **corrélations structure-propriétés** permettant de comprendre les mécanismes redox au sein de cette famille moléculaire. En extrayant des descripteurs structuraux (charges atomiques, énergies d'interaction) et en les corrélant aux propriétés redox calculées (potentiels de réduction/oxydation), nous cherchons à identifier quels paramètres structurels gouvernent la réactivité électrochimique.

---

## Architecture du Pipeline

Le workflow se décompose en **4 phases** :

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: CONFORMÈRES                              │
│  SMILES → Génération RDKit → Optimisation → Scoring DFT → Filtrage RMSD     │
│         → Sélection Boltzmann → Vérification Fréquences                     │
│                                                                             │
│  Sortie: Ensemble de conformères représentatifs avec poids statistiques     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 2: REDOX                                   │
│  Pour chaque conformère:                                                    │
│    Opt Neutre → SP Réduit → Opt Réduit → SP Oxydé → Opt Oxydé               │
│                                                                             │
│  Calcul: ΔG = EA/IP + E_déformation + ΔΔU + TΔS                             │
│  Sortie: Potentiels redox décomposés, HOMO/LUMO, moyennes Boltzmann         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 3: QTAIM/IQA                                │
│  Analyse topologique de la densité électronique:                            │
│    - Charges de Bader (intégration bassins atomiques)                       │
│    - Énergies IQA (décomposition atome-atome)                               │
│    - Réindexation canonique C1-C6, O7-O8                                    │
│                                                                             │
│  Sortie: Descripteurs structuraux comparables entre molécules               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: CORRÉLATIONS                                │
│  Comparaison inter-molécules:                                               │
│    - Chaque point = 1 molécule (moyennes Boltzmann)                         │
│    - Corrélations: charges Bader / IQA vs potentiels redox                  │
│    - Régressions linéaires avec R²                                          │
│                                                                             │
│  Sortie: Graphiques, heatmaps, identification de tendances                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Le Défi de la Réindexation

### Problème

Chaque molécule possède sa propre numérotation atomique dans AMS/ADF. Ainsi :

- **DHI** : le carbone du cycle peut être l'atome #3
- **O-DHI** : le même carbone (structurellement équivalent) peut être l'atome #7

Cette incohérence rend impossible toute comparaison directe des charges ou énergies entre molécules.

### Solution : Réindexation Canonique

Le module `qmqtaim/topology.py` implémente un algorithme de réindexation basé sur la **topologie QTAIM** :

1. **Identification du squelette commun** : Toutes les quinones partagent un cycle à 6 carbones avec 2 oxygènes
2. **Point critique de cycle** : Le ring critical point (rCP) de QTAIM sert d'ancrage géométrique
3. **Détection des atomes clés** :
   - **C1** : Carbone de branchement (3 liaisons C-C)
   - **C5** : Carbone à distance 2 de C1, lié à un oxygène
4. **Traçage du cycle** : Parcours des liaisons pour ordonner C1→C2→C3→C4→C5→C6
5. **Attribution des oxygènes** : O7 et O8 selon l'ordre d'apparition dans le cycle

### Résultat

```
Indices Canoniques (fixes)     Indices AMS (variables)
         C1          ←───────→    Atome #5 (DHI) / #12 (O-DHI)
         C2          ←───────→    Atome #6 (DHI) / #3 (O-DHI)
         ...
         O7          ←───────→    Atome #9 (DHI) / #15 (O-DHI)
         O8          ←───────→    Atome #10 (DHI) / #8 (O-DHI)
```

Grâce à cette réindexation, nous pouvons affirmer que **C3_bader de DHI** et **C3_bader de O-DHI** représentent le même atome du squelette, rendant les comparaisons légitimes.

---

## Corrélations Structure-Propriétés

### Principe

L'analyse finale consiste à tracer des graphiques où :

- **Axe X** : Descripteur structural (charge de Bader, énergie IQA)
- **Axe Y** : Propriété redox (potentiel de réduction/oxydation)
- **Chaque point** : Une molécule différente (valeur moyennée sur ses conformères)

### Exemple d'Interprétation

```
Graphique: Charge Bader O7 vs Potentiel de Réduction

     E_red (V)
        │
   -0.5 │            • DHI
        │       • O-DHI
   -1.0 │  • X-DHI
        │                  R² = 0.85
   -1.5 │──────────────────────────
        └─────────────────────────── Charge O7 (e)
            -0.5    -0.3    -0.1
```

**Hypothèse dérivée** : "Plus l'oxygène O7 est chargé négativement, plus le potentiel de réduction est négatif (molécule plus difficile à réduire)."

### Outputs

- **Grille 9×6** : Tous les graphiques (8 charges Bader + 8 énergies IQA) × (réduction + oxydation)
- **Graphiques individuels** : Pour analyse détaillée
- **Heatmap de corrélation** : Vue d'ensemble des relations entre propriétés
- **CSV consolidé** : Données brutes pour analyses statistiques externes

---

## Utilisation Scientifique

### Ce que le pipeline permet

1. **Explorer** les relations structure-propriétés sur une famille de molécules
2. **Identifier** les descripteurs structuraux corrélés à la réactivité redox
3. **Formuler des hypothèses** sur les mécanismes (ex: "la délocalisation sur O7 favorise la réduction")
4. **Comparer** quantitativement des molécules de structures différentes grâce à la réindexation

### Ce qui reste hors scope (pour l'instant)

- Validation expérimentale des corrélations identifiées
- Modèles prédictifs (QSPR/machine learning)
- Extension à d'autres familles moléculaires

---

## Réutilisabilité

### Pipeline Conformères (qmconf)

Module **réutilisable** pour tout projet nécessitant :

- Génération de conformères depuis SMILES
- Optimisation géométrique multi-niveaux
- Sélection statistique (Boltzmann)
- Validation par fréquences vibrationnelles

### Pipeline Redox (qmredox)

Applicable à toute molécule organique pour :

- Calcul de potentiels redox en solution
- Décomposition thermodynamique (EA, IP, E_def, entropie)
- Analyse orbitaire (HOMO/LUMO)

### Pipeline QTAIM (qmqtaim)

Nécessite une adaptation du module `topology.py` pour chaque nouvelle famille moléculaire (définition du squelette commun).

---

## Structure du Projet

```
PlamsV4/
├── conformers_gen.py        # Script principal conformères
├── redox_calc.py            # Script principal redox
├── qtaim_calc.py            # Script principal QTAIM
├── visualize_batch.py       # Visualisation inter-molécules
│
├── common/                  # Modules partagés
│   ├── constants.py         # Constantes physiques
│   ├── functional.py        # Configuration DFT
│   ├── frequency.py         # Vérification fréquences
│   ├── workspace.py         # Gestion répertoires
│   ├── utils.py             # Utilitaires (Boltzmann, print)
│   └── multi_molecule_data.py  # Consolidation inter-molécules
│
├── qmconf/                  # Pipeline conformères
│   ├── ARCHITECTURE.md      # Documentation technique
│   ├── pipeline.py          # 6 étapes du workflow
│   └── analysis.py          # Mode analysis-only
│
├── qmredox/                 # Pipeline redox
│   ├── ARCHITECTURE.md      # Documentation technique
│   ├── pipeline.py          # 5 calculs par conformère
│   └── analysis.py          # Extraction et moyennage
│
├── qmqtaim/                 # Pipeline QTAIM/IQA
│   ├── ARCHITECTURE.md      # Documentation technique
│   ├── pipeline.py          # Orchestration calculs
│   ├── analysis.py          # Extraction Bader/IQA
│   └── topology.py          # Réindexation canonique
│
├── visualization/           # Graphiques
│   └── batch_correlations.py  # Corrélations inter-molécules
│
├── main.sl                  # Script SLURM workflow complet
├── qtaim.sl                 # Script SLURM QTAIM seul
└── PROJECT.md               # Ce fichier
```

---

## Conclusion

PlamsV4 est un outil de recherche conçu pour transformer des calculs DFT bruts en insights chimiques exploitables. En automatisant la chaîne complète — de la génération de conformères à l'analyse de corrélations — il permet de se concentrer sur l'interprétation scientifique plutôt que sur la gestion technique des calculs.

Les hypothèses générées par ce pipeline constituent le point de départ d'une démarche scientifique : elles devront être confrontées à l'expérience et affinées par des études complémentaires.
