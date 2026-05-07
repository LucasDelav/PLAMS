# Module qmqtaim - Architecture et Référence

## Vue d'Ensemble

Pipeline d'analyse QTAIM/IQA avec réindexation topologique canonique.

**Workflow:** DFT Single-Point → QTAIM → Critical Points → Reindex C1-C6/O7-O8 → Bader Charges → IQA Energies

---

## Script Principal: qtaim_calc.py (217 lignes)

### Fonctions

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `parse_arguments()` | 21-124 | Parse CLI, modes calcul ou analysis-only |
| `main()` | 127-217 | Init PLAMS, QTAIMPipeline, export results |

### Arguments CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--name` | requis | Nom de la molécule |
| `--conformers` | tous | Liste conformères spécifiques |
| `--charges` | 0 -1 +1 | États de charge à calculer |
| `--functional` | BLYP | Fonctionnelle DFT |
| `--basis` | TZP | Base ADF |
| `--quality` | Good | Qualité numérique (Basic/Normal/Good/VeryGood/Excellent) |
| `--integration-accuracy` | 5.0 | Précision intégration QTAIM |
| `--no-iqa` | False | Désactiver IQA |
| `--nproc` | 1 | Processus parallèles |
| `--workdir` | ./{name}_workdir | Chemin workdir |
| `--analysis-only` | - | Mode ré-analyse sans calculs |

---

## qmqtaim/common.py (106 lignes)

### Classes

| Classe | Lignes | Description |
|--------|--------|-------------|
| `CalculationError` | 15-17 | Exception personnalisée |

### Fonctions

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `get_conformer_files(workdir)` | 20-46 | Glob `conformers/results/conf_*.xyz` |
| `create_adf_settings(func, basis, charge, quality, accuracy, iqa)` | 49-106 | Configure ADF avec QTAIM/IQA |

### Configuration create_adf_settings [49-106]

```python
s.input.ams.Task = 'SinglePoint'
s.input.ams.System.Charge = charge
s.input.adf.Basis.Type = basis
s.input.adf.Basis.Core = 'None'
s.input.adf.Relativity.Level = 'None'    # CRITIQUE: ZORA incompatible avec QTAIM
s.input.adf.NumericalQuality = quality
configure_functional(s, functional)

# QTAIM
s.input.adf.QTAIM.Enabled = 'Yes'
s.input.adf.QTAIM.AnalysisLevel = 'Full'
s.input.adf.QTAIM.Source = 'Yes'

# IQA (optionnel)
if run_iqa:
    s.input.adf.IQA.Enabled = 'Yes'

# Spin pour états chargés
if charge != 0:
    s.input.adf.SpinPolarization = abs(charge)
    s.input.adf.Unrestricted = 'Yes'
```

---

## qmqtaim/pipeline.py (350 lignes)

### Classe QTAIMPipeline

| Méthode | Lignes | Description |
|---------|--------|-------------|
| `__init__(workdir, func, basis, quality, accuracy, iqa, nproc, restart, mol_name)` | 30-73 | Init analyseurs + répertoires |
| `get_conformers(conformer_ids)` | 75-94 | Liste fichiers conformères (filtre optionnel) |
| `load_existing_calculations()` | 96-166 | Charge RKF existants pour analysis-only |
| `run_calculation(mol, conf_id, charge)` | 168-239 | Exécute 1 calcul QTAIM/IQA |
| `run_pipeline(conformer_ids, charges, analysis_only)` | 241-350 | Orchestration principale |

### Nommage des Jobs [183-195]

```
{molecule_name}_{conformer_id}_{charge_label}
où charge_label = "neutral" (0), "reduced" (-1), "oxidized" (+1)
```

### Workflow run_pipeline [241-350]

```python
# Mode analysis-only:
if analysis_only:
    rkf_paths = load_existing_calculations()
    for conf_id, charge:
        analyzer.analyze_rkf(rkf_paths[conf_id][charge], mol, topology_analyzer)

# Mode calculs:
else:
    for conf_file in get_conformers():
        for charge in charges:
            rkf_path = run_calculation(mol, conf_id, charge)
            analyzer.analyze_rkf(rkf_path, mol, topology_analyzer)
```

---

## qmqtaim/analysis.py (725 lignes)

### Classe QTAIMAnalyzer

### Extraction

| Méthode | Lignes | Description |
|---------|--------|-------------|
| `analyze_rkf(rkf_path, mol, topology_analyzer, conf_id, charge, mol_name)` | 28-144 | **Fonction principale**: CP → reindex → Bader → IQA |
| `_apply_reindex(data_dict, reindex_map)` | 146-169 | Applique mapping AMS→canonique |
| `_extract_bader_charges(kf)` | 171-182 | Lit `Properties/"Bader atomic charges"` |
| `_extract_iqa_energies(kf)` | 184-213 | Lit `Properties/"IQA atom-atom total"` |
| `_extract_critical_points(kf)` | 215-280 | Extrait ring CP (code=4) avec coordonnées et densités |

### Clés RKF Extraites

| Donnée | Section RKF | Format |
|--------|-------------|--------|
| Bader charges | `Properties/"Bader atomic charges"` | Array 1D, index 0-based |
| IQA energies | `Properties/"IQA atom-atom total"` | Array N×N aplati |
| CP codes | `Properties/"CP code number for (Rank,Signatu"` | Array, 4=ring CP |
| CP coords | `Properties/"CP coordinates"` | [x1...xN, y1...yN, z1...zN] |
| CP densities | `Properties/"CP density at"` | Array densités |

### Extraction IQA [184-213]

```python
# Format RKF: matrice N×N aplatie [a11,a12,...,a1N,a21,...,aNN]
iqa_total = kf.read("Properties", "IQA atom-atom total")
n_atoms = int(len(iqa_total) ** 0.5)

# Énergie atome i = somme ligne i
for i in range(n_atoms):
    atom_energy = sum(iqa_total[i*n_atoms:(i+1)*n_atoms])
    iqa_atom_energies[i+1] = atom_energy  # Index 1-based
```

### Export

| Méthode | Lignes | Description |
|---------|--------|-------------|
| `export_results(results, output_dir, workdir)` | 282-315 | Dispatcher export |
| `_export_bader_charges_by_charge(results, dir, charge)` | 319-355 | CSV: C1-C6, O7-O8 |
| `_export_iqa_energies(results, dir, charge)` | 357-393 | CSV: C1_iqa...O8_iqa |
| `_read_boltzmann_weights(workdir)` | 395-426 | Lit conformers_summary.txt |
| `_export_boltzmann_analysis(results, dir, charge, workdir)` | 428-529 | Moyennes pondérées ± std |
| `_export_text_summary(results, dir, workdir)` | 531-682 | Rapport texte complet |
| `_export_json(results, dir)` | 684-725 | JSON complet |

### Colonnes CSV Export

**bader_charges_{charge}.csv:**
```
conformer_id, C1, C2, C3, C4, C5, C6, O7, O8
```

**iqa_energies_{charge}.csv:**
```
conformer_id, C1_iqa, C2_iqa, C3_iqa, C4_iqa, C5_iqa, C6_iqa, O7_iqa, O8_iqa
```

---

## qmqtaim/topology.py (721 lignes)

### Classe TopologyAnalyzer

Algorithme de réindexation canonique pour cohérence entre conformères/états de charge.

### Méthodes Principales

| Méthode | Lignes | Description |
|---------|--------|-------------|
| `__init__(logger, verbose)` | 17-41 | Init attributs (ring_atoms, oxygen_atoms, maps) |
| `find_ring_critical_point(cps, ring_size, mol, expected_oxygens)` | 187-360 | Sélectionne meilleur ring CP |
| `find_ring_and_oxygen_atoms(mol, cp_coords, c_thresh, o_thresh)` | 362-422 | Trouve 6C + 2O près du CP |
| `get_bonded_atoms(mol, atom_idx, allowed_set)` | 424-451 | Voisins liés |
| `find_c1_carbon(mol, ring_atoms)` | 453-481 | C avec 3 liaisons C (2 ring + 1 branche) |
| `find_c5_carbon(mol, c1_idx, ring_atoms, oxygen_atoms)` | 483-526 | C à distance 2 de C1, lié à O |
| `trace_ring_path(mol, c1_idx, c5_idx, ring_atoms)` | 528-622 | Trace [C1,C2,C3,C4,C5,C6] |
| `assign_oxygen_indices(mol, ring_path, oxygen_atoms)` | 624-662 | Assigne O7, O8 |
| `reindex_atoms(mol, cp_coords)` | 664-713 | **Orchestration complète** |
| `get_canonical_index(original_idx)` | 715-717 | Lookup AMS→canonique |
| `get_original_index(canonical_idx)` | 719-721 | Lookup canonique→AMS |

### Méthodes Diagnostic

| Méthode | Lignes | Description |
|---------|--------|-------------|
| `_print_molecular_structure(mol)` | 43-53 | Affiche tous atomes |
| `_print_connectivity_graph(mol)` | 55-68 | Affiche liaisons |
| `_print_ring_connectivity(mol, ring_atoms)` | 70-95 | Matrice adjacence cycle |

### Algorithme de Sélection Ring CP [187-360]

```
SI 1 seul rCP → sélectionner

SI plusieurs rCPs:
    Pour chaque combinaison (C_threshold, O_threshold):
        # Exploration par distance Manhattan
        # C_threshold: 3.0 ± 0.1*n Å
        # O_threshold: 3.8 ± 0.1*n Å

        Pour chaque rCP:
            Compter carbones dans C_threshold
            Compter oxygènes dans O_threshold

            SI exactement 6C + 2O → candidat valide

        SI 1 candidat valide → RETOURNER
        SI 2+ candidats → ERREUR (ambiguïté)
        SI 0 candidats → continuer exploration

FALLBACK (mol=None): sélectionner rCP avec densité max
```

### Algorithme de Réindexation [664-713]

```
ENTRÉE: Molécule PLAMS, coordonnées ring CP
SORTIE: {canonique → original}

Étape 1: find_ring_and_oxygen_atoms()
         → ring_atoms (6 carbones), oxygen_atoms (2 oxygènes)

Étape 2: find_c1_carbon(ring_atoms)
         → c1_idx (carbone avec 3 liaisons C)

Étape 3: find_c5_carbon(c1_idx, ring_atoms, oxygen_atoms)
         → c5_idx (distance 2 de C1, lié à O)

Étape 4: trace_ring_path(c1_idx, c5_idx, ring_atoms)
         → [C1, C2, C3, C4, C5, C6] ordonnés

Étape 5: assign_oxygen_indices(ring_path, oxygen_atoms)
         → (O7_idx, O8_idx) par ordre dans le cycle

Étape 6: Construire maps
         reverse_map = {1: c1, 2: c2, ..., 7: o7, 8: o8}
```

### Système d'Indices Canoniques

```
Canonique → AMS (variable selon conformère)
   C1      →  idx_branching_carbon
   C2      →  idx_neighbor_of_c1
   C3      →  idx_bonded_to_o7
   C4      →  idx_bonded_to_o8
   C5      →  idx_bonded_to_o_at_dist2
   C6      →  idx_other_c5_neighbor
   O7      →  idx_first_oxygen_in_path
   O8      →  idx_second_oxygen_in_path
```

---

## Structures de Données

### Résultats analyze_rkf

```python
results = {
    'critical_points': [
        {'type': 4, 'index': 15, 'coords': [x,y,z], 'density': 0.087}
    ],
    'reindexed': True,
    'reindex_map': {5: 1, 6: 2, 7: 3, 8: 4, 3: 5, 4: 6, 9: 7, 10: 8},
    'bader_charges': {1: -0.325, 2: -0.187, ..., 7: -0.456, 8: -0.398},
    'iqa_atom_energies': {1: -58.34, 2: -58.12, ..., 7: -74.23, 8: -74.18}
}
```

### Résultats Pipeline Complet

```python
{
    'conf_1': {
        0: {...},   # neutral
        -1: {...},  # reduced
        +1: {...}   # oxidized
    },
    'conf_2': {...}
}
```

---

## Dépendances

```
qtaim_calc.py
├── common/workspace.py (get_pipeline_dirs)
├── common/functional.py (configure_functional)
├── common/utils.py (print_*)
└── qmqtaim/
    ├── common.py (get_conformer_files, create_adf_settings)
    ├── pipeline.py (QTAIMPipeline)
    ├── analysis.py (QTAIMAnalyzer)
    └── topology.py (TopologyAnalyzer)
```

---

## Fichiers de Sortie

```
{molecule}_workdir/
└── qtaim/
    ├── calculations/
    │   └── plams_workdir/
    │       ├── {mol}_conf_{N}_neutral/
    │       ├── {mol}_conf_{N}_reduced/
    │       └── {mol}_conf_{N}_oxidized/
    └── results/
        ├── bader_charges_neutral.csv
        ├── bader_charges_charge-1.csv
        ├── bader_charges_charge+1.csv
        ├── iqa_energies_neutral.csv
        ├── iqa_energies_charge-1.csv
        ├── iqa_energies_charge+1.csv
        ├── boltzmann_analysis_neutral.csv
        ├── qtaim_summary.txt
        └── qtaim_complete_data.json
```

---

## Diagnostic Rapide

| Problème | Où chercher |
|----------|-------------|
| Pas de ring CP trouvé | `analysis.py:215-280` (_extract_critical_points) |
| Réindexation échoue | `topology.py:187-360` (find_ring_critical_point), activer verbose=True |
| Mauvais nombre C/O | `topology.py:362-422` (find_ring_and_oxygen_atoms), ajuster thresholds |
| Erreur C1 detection | `topology.py:453-481` (find_c1_carbon), vérifier liaisons mol.guess_bonds() |
| Erreur trace cycle | `topology.py:528-622` (trace_ring_path), diagnostic _print_ring_connectivity |
| Bader charges manquantes | `analysis.py:171-182`, vérifier QTAIM.Enabled dans settings |
| IQA manquant | `analysis.py:184-213`, vérifier IQA.Enabled |
| JSON export | `analysis.py:684-725` (_export_json) |
| Poids Boltzmann | `analysis.py:395-426` (_read_boltzmann_weights) |
