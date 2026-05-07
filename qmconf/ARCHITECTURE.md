# Module qmconf - Architecture et Référence

## Vue d'Ensemble

Pipeline de génération et optimisation de conformères moléculaires.

**Workflow:** SMILES → Generate (RDKit) → Optimize → Score (DFT) → Filter (RMSD) → Boltzmann → Freq Check → XYZ

---

## Script Principal: conformers_gen.py (473 lignes)

### Fonctions

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `parse_arguments()` | 34-190 | Parse CLI, validation, groupe exclusif --freq/--no-freq |
| `main()` | 193-472 | Orchestration: 2 modes (analysis-only ou calculs complets) |

### Arguments CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--name` | requis | Nom de la molécule |
| `--smiles` | requis* | SMILES d'entrée (*sauf analysis-only) |
| `--functional` | PBE0 | Fonctionnelle DFT scoring |
| `--basis` | DZP | Base ADF (SZ/DZ/DZP/TZP/TZ2P) |
| `--dispersion` | GrimmeBJ | Correction dispersion |
| `--e-window-opt` | 10.0 | Fenêtre énergie optimisation (kcal/mol) |
| `--e-window-score` | 6.0 | Fenêtre énergie scoring (kcal/mol) |
| `--e-window-filter` | 3.0 | Fenêtre énergie filtrage (kcal/mol) |
| `--rmsd-threshold` | 1.0 | Seuil RMSD équivalence (Å) |
| `--boltzmann-min-weight` | 0.05 | Poids Boltzmann minimum |
| `--freq` / `--no-freq` | --freq | Vérification fréquences (ACTIVÉ par défaut) |
| `--freq-functional` | =functional | Fonctionnelle pour fréquences |
| `--freq-basis` | DZ | Base pour fréquences |
| `--analysis-only` | - | Mode ré-analyse sans calculs |
| `--workdir` | - | Workdir (requis avec --analysis-only) |
| `--subfolder` | boltzmann_filtered | Source pour analysis-only |
| `--temperature` | 298.15 | Température Boltzmann (K) |

---

## qmconf/common.py (20 lignes)

Module de ré-export des utilitaires communs.

### Exports

```python
from common.constants import TEMPERATURE, R, UNIT, KCAL_TO_KJ
from common.functional import configure_functional, ALL_FUNCTIONALS
from common.utils import DefaultJobManager, boltzmann_weights
```

### Constantes Clés

| Constante | Valeur | Usage |
|-----------|--------|-------|
| `TEMPERATURE` | 298.15 K | Calcul Boltzmann |
| `R` | 8.314462618 J/(mol·K) | Constante gaz parfaits |
| `UNIT` | "kJ/mol" | Unité énergie interne |
| `KCAL_TO_KJ` | 4.184 | Conversion kcal→kJ |

---

## qmconf/pipeline.py (405 lignes)

### Fonctions

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `setup_workspace(name)` | 32-46 | Crée `{name}_workdir/conformers/{calculations,results}` |
| `_safe_relative_energies(job, unit)` | 49-75 | Extraction énergies relatives avec gestion erreurs |
| `print_results(job, T, unit)` | 78-99 | Affiche énergies et poids Boltzmann |
| `generate_conformers(mol, calc_dir)` | 102-132 | **Étape 1**: RDKit + UFF, 1000 conformères |
| `optimize_conformers(job, calc_dir, e_window)` | 135-165 | **Étape 2**: Optimisation géométrie |
| `_apply_dispersion(s, dispersion)` | 168-191 | Configure XC.DISPERSION |
| `score_conformers(job, calc_dir, func, basis, e_window, disp)` | 194-231 | **Étape 3**: Single-point DFT |
| `filter_conformers(job, calc_dir, e_window, rmsd)` | 234-266 | **Étape 4**: Filtrage RMSD + énergie |
| `filter_boltzmann_weights(job, calc_dir, threshold)` | 269-300 | **Étape 5**: Sélection statistique |
| `verify_frequencies(job, name, results_dir, calc_dir, func, basis)` | 303-404 | **Étape 6**: Vérif/correction fréquences imaginaires |

### Workflow Détaillé

```
Étape 1: generate_conformers() [102-132]
├── Settings: Task="Generate", Generator.Method="RDKit"
├── ForceField: UFF
├── Génère: 1000 conformères initiaux
└── Output: 01_generate/

Étape 2: optimize_conformers() [135-165]
├── Settings: Task="Optimize", InputMaxEnergy=e_window
├── Optimise géométries avec fenêtre énergie
└── Output: 02_optimize/

Étape 3: score_conformers() [194-231]
├── Settings: Task="Score", configure_functional()
├── _apply_dispersion(): Grimme3/GrimmeBJ/TS/D4
├── DFT single-point pour énergies précises
└── Output: 03_score/

Étape 4: filter_conformers() [234-266]
├── Settings: Task="Filter", Equivalence.CREST.RMSDThreshold
├── Supprime doublons par RMSD
└── Output: 04_filter/

Étape 5: filter_boltzmann_weights() [269-300]
├── Calcule: β = 1/(R*T), w_i = exp(-β*ΔE_i)/Z
├── Garde conformères avec w ≥ threshold
└── Output: boltzmann_filtered/

Étape 6: verify_frequencies() [303-404]
├── Pour chaque conformère sélectionné:
│   ├── run_frequency_calculation() avec optimize=True
│   ├── check_and_fix_frequencies(max_attempts=3)
│   └── Perturbations: ±0.2, ±0.4, ±0.6 Å
├── Seuil: 0.0 cm⁻¹ (AUCUNE fréquence négative)
└── Output: 05_freq/conf_{N}/
```

### Schémas de Dispersion [168-191]

| Paramètre | Configuration ADF |
|-----------|-------------------|
| `"None"` | Pas de dispersion |
| `"Grimme3"` | `XC.DISPERSION = "GRIMME3"` |
| `"GrimmeBJ"` | `XC.DISPERSION = "GRIMME3 BJDAMP"` |
| `"TS"` | `XC.DISPERSION = "TS"` |
| `"D4"` | `XC.DISPERSION = "D4"` |

---

## qmconf/analysis.py (287 lignes)

Mode analysis-only pour ré-analyser des conformères existants.

### Fonctions

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `extract_conf_number(filename)` | 19-43 | Extrait numéro conformère du nom fichier |
| `load_conformers_from_workdir(workdir, subfolder)` | 46-142 | Charge XYZ + RKF + énergies existants |
| `analyze_existing_conformers(workdir, T, subfolder)` | 145-212 | Recalcule poids Boltzmann à nouvelle T |
| `export_conformers_summary(results, output_file, mol_name)` | 215-257 | Export résumé texte formaté |
| `export_conformers_csv(results, output_file)` | 260-286 | Export CSV pour analyse |

### Structure de Données (load_conformers_from_workdir)

```python
conformers = {
    conf_num: {
        'xyz': Path,      # Fichier XYZ
        'rkf': Path,      # Fichier RKF (si trouvé)
        'energy': float   # Énergie en Hartree
    }
}
```

### Recherche RKF [85-119]

Priorité de recherche:
1. `{subfolder}/plams_workdir/`
2. `05_freq/conf_{N}/plams_workdir/`
3. `04_filter/plams_workdir/`
4. `03_score/plams_workdir/`

---

## Dépendances

```
conformers_gen.py
├── common/constants.py (TEMPERATURE, R, UNIT)
├── common/functional.py (configure_functional)
├── common/workspace.py (setup_workspace)
├── common/utils.py (boltzmann_weights, print_*)
├── common/frequency.py (run_frequency_calculation, check_and_fix_frequencies)
└── qmconf/
    ├── common.py (ré-exports)
    ├── pipeline.py (6 étapes workflow)
    └── analysis.py (mode analysis-only)
```

---

## Fichiers de Sortie

```
{molecule}_workdir/
└── conformers/
    ├── calculations/
    │   ├── 01_generate/plams_workdir/
    │   ├── 02_optimize/plams_workdir/
    │   ├── 03_score/plams_workdir/
    │   ├── 04_filter/plams_workdir/
    │   ├── 05_freq/conf_{N}/plams_workdir/
    │   ├── boltzmann_filtered/conf_*.xyz
    │   └── boltzmann_filter_summary.txt
    └── results/
        ├── conf_*.xyz (conformères validés)
        ├── conformers_summary.txt
        └── conformers_data_*.csv
```

---

## Diagnostic Rapide

| Problème | Où chercher |
|----------|-------------|
| Erreur génération SMILES | `conformers_gen.py:311-317` (from_smiles) |
| Peu de conformères après filtrage | `boltzmann_filter_summary.txt`, ajuster --boltzmann-min-weight |
| Fréquences imaginaires | `05_freq/frequency_check_summary.txt` |
| Erreur dispersion | `pipeline.py:168-191` (_apply_dispersion) |
| Calcul Boltzmann | `common/utils.py:boltzmann_weights()` |
