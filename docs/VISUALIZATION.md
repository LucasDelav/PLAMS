# Système de Visualisation

## Vue d'ensemble

Le module `visualization/` génère des graphiques de corrélation pour comparer les propriétés moléculaires. Deux types d'analyses sont disponibles :

1. **Batch (inter-molécules)** - Compare N molécules entre elles (charges Bader, IQA vs potentiels redox)
2. **Fréquence-Edef** - Corrèle les descripteurs vibrationnels avec les énergies de déformation

Chaque point représente **une molécule** (moyennes de Boltzmann), pas un conformère.

---

## Scripts et Modules

| Fichier | Description |
|---------|-------------|
| `visualize_batch.py` | CLI pour visualisation batch inter-molécules |
| `visualization/batch_correlations.py` | Génération des graphiques batch |
| `visualization/frequency_correlations.py` | Graphiques fréquence-Edef |
| `common/multi_molecule_data.py` | Consolidation des données multi-molécules |

---

## Usage

### Visualisation Batch

```bash
# Auto-découverte des workdirs
./visualize_batch.py --batch-dir 1594174/dir_workdir --output-dir batch_viz

# Workdirs explicites
./visualize_batch.py --workdirs DHI_workdir O-DHI_workdir --output-dir results
```

### Analyse Fréquence-Edef

```bash
./frequency_analysis.py --batch-dir workdirs/ --output-dir freq_analysis
```

---

## Données Requises

Chaque `{molecule}_workdir/` doit contenir :

```
molecule_workdir/
├── conformers/results/conformers_summary.txt    # Poids de Boltzmann
├── redox/results/molecule_redox_summary.csv     # Potentiels redox (déjà moyennés)
└── qtaim/results/qtaim_complete_data.json       # Charges Bader + énergies IQA
```

---

## Outputs Générés

### Batch Visualization

```
batch_visualization/
├── all_correlations_grid.png      # Grille 9×6 (réduction | oxydation)
├── batch_correlation_heatmap.png  # Matrice de corrélation
├── molecules_summary.csv          # Données consolidées
└── single_graphs/                 # ~50 graphiques individuels
    ├── Bader_1_vs_reduction.png
    ├── C1_IQA_vs_oxidation.png
    └── ...
```

### Frequency Analysis

```
frequency_analysis/
├── frequency_correlations.csv             # Corrélations Edef triées par R²
├── frequency_descriptors.csv              # Descripteurs par molécule
├── focused_correlations_ddU.csv           # ΔΔU vs descripteurs haute fréquence (stats complètes)
├── focused_correlations_TdS.csv           # TΔS vs descripteurs basse fréquence (stats complètes)
├── thermodynamic_targets.csv              # ΔΔU/TΔS/Edef par molécule
├── top_correlations_reduction.png         # Grille scatter Edef réduction
├── top_correlations_oxidation.png         # Grille scatter Edef oxydation
├── top_correlations_ddU_reduction.png     # Grille scatter ΔΔU réduction
├── top_correlations_ddU_oxidation.png     # Grille scatter ΔΔU oxydation
├── top_correlations_TdS_reduction.png     # Grille scatter TΔS réduction
├── top_correlations_TdS_oxidation.png     # Grille scatter TΔS oxydation
└── single_graphs/                         # Plots individuels détaillés
    ├── <desc>_vs_ddU_red.png
    ├── <desc>_vs_ddU_ox.png
    ├── <desc>_vs_TdS_red.png
    └── <desc>_vs_TdS_ox.png
```

---

## Interprétation

### Qualité des Corrélations

| R² | Interprétation |
|----|----------------|
| > 0.9 | Excellente, forte prédictibilité |
| 0.7-0.9 | Bonne, tendance claire |
| 0.5-0.7 | Modérée, facteurs multiples |
| < 0.5 | Faible, autres facteurs dominants |

### Signification Chimique

- **Charges Bader négatives** → atome riche en électrons
- **LUMO basse** → réduction favorable
- **Gap HOMO-LUMO large** → molécule stable

---

## Dépendances

```bash
pip install matplotlib pandas scipy seaborn
```

**Note** : `visualize_batch.py` utilise `python3` standard (pas `amspython`) pour éviter les conflits d'imports PLAMS.

---

## Troubleshooting

| Problème | Solution |
|----------|----------|
| "Colonnes manquantes" | Vérifier que tous les calculs (conformers, redox, QTAIM) sont terminés |
| "Pas assez de données" | Normal si <2 molécules pour une propriété (warning, pas erreur) |
| Plots vides | Vérifier les CSV/JSON dans `*/results/` |
