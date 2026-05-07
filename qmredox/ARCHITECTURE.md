# Module qmredox - Architecture et Référence

## Vue d'Ensemble

Pipeline de calcul de potentiels redox par DFT avec décomposition énergétique.

**Workflow par conformère:** Opt Neutral → SP Reduced → Opt Reduced → SP Oxidized → Opt Oxidized

---

## Script Principal: redox_calc.py (389 lignes)

### Fonctions

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `parse_arguments()` | 30-148 | Parse CLI, 2 modes: calculs ou analysis-only |
| `load_conformers_from_dir(input_dir)` | 151-189 | Charge XYZ, extrait numéros conformères |
| `load_existing_jobs(workdir, name)` | 192-270 | Récupère jobs PLAMS existants pour ré-analyse |
| `main()` | 273-388 | Orchestration: boucle conformères + analyse finale |

### Arguments CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--name` | requis | Nom de la molécule |
| `--input-dir` | requis* | Répertoire XYZ conformères (*sauf analysis-only) |
| `--functional` | PBE0 | Fonctionnelle DFT |
| `--basis` | DZP | Base ADF (SZ/DZ/DZP/TZP/TZ2P/QZ4P) |
| `--solvent` | Acetonitrile | Solvant COSMO |
| `--temperature` | 298.15 | Température Boltzmann (K) |
| `--u-method` | direct | Méthode calcul U: direct ou h_minus_rt |
| `--max-freq-attempts` | 3 | Tentatives correction fréquences |
| `--analysis-only` | - | Mode ré-analyse sans calculs |
| `--workdir` | - | Workdir (requis avec --analysis-only) |

---

## qmredox/common.py (179 lignes)

### Fonctions

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `setup_workspace(name)` | 49-64 | Crée `{name}_workdir/redox/{calculations,results}` |
| `setup_adf_settings(task, charge, func, basis, solvent)` | 67-123 | Configure Settings ADF avec COSMO |
| `kabsch_rmsd(mol1, mol2)` | 128-179 | RMSD après alignement Kabsch |

### Configuration setup_adf_settings [67-125]

```python
# Paramètres clés
s.input.ams.Task = task  # "GeometryOptimization" ou "SinglePoint"
s.input.ams.Properties.NormalModes = "Yes"
s.input.adf.Basis.Type = basis
s.input.adf.Basis.Core = "None"
s.input.adf.Relativity.Level = "None"  # Désactivé pour QTAIM
configure_functional(s, functional)

# Spin pour états chargés
if charge != 0:
    s.input.ams.System.Charge = charge
    s.input.adf.SpinPolarization = abs(charge)
    s.input.adf.Unrestricted = "Yes"

# Solvatation COSMO (via _apply_solvation)
s.input.adf.Solvation.Surf = "Delley"
s.input.adf.Solvation.Solv = f"name={solvent}"
s.input.adf.Solvation.Charged = "method=CONJ"   # ou "method=CONJ CORR" si charge != 0
s.input.adf.Solvation['C-Mat'] = "POT"
s.input.adf.Solvation.SCF = "VAR ALL"
```

### Constantes Ré-exportées

| Constante | Valeur | Usage |
|-----------|--------|-------|
| `R` | 8.314462618 J/(mol·K) | Gaz parfaits |
| `F` | 96485.33212 C/mol | Faraday |
| `TEMPERATURE` | 298.15 K | Défaut |
| `EH_TO_KJ` | 2625.499638 | Hartree → kJ/mol |
| `EH_TO_EV` | 27.211386 | Hartree → eV |

---

## qmredox/pipeline.py (416 lignes)

### Fonctions - 5 Étapes du Workflow

| Fonction | Lignes | Charge | Spin | Description |
|----------|--------|--------|------|-------------|
| `optimize_neutral(mol, name, solvent, func, basis, cosmors, max_att)` | 13-90 | 0 | 0 | Optimise neutre + vérif freq |
| `sp_reduced(neutral_job, name, solvent, func, basis, cosmors)` | 93-138 | -1 | 1 | SP sur géom neutre, état réduit |
| `optimize_reduced(neutral_job, name, solvent, func, basis, cosmors, max_att)` | 141-216 | -1 | 1 | Optimise réduit depuis neutre |
| `sp_oxidized(neutral_job, name, solvent, func, basis, cosmors)` | 219-264 | +1 | 1 | SP sur géom neutre, état oxydé |
| `optimize_oxidized(neutral_job, name, solvent, func, basis, cosmors, max_att)` | 267-342 | +1 | 1 | Optimise oxydé depuis neutre |
| `run_redox_pipeline(mol, name, solvent, func, basis, cosmors, max_att)` | 345-416 | - | - | Orchestration 5 étapes |

### Workflow Détaillé

```
Étape 1: optimize_neutral() [13-90]
├── Charge: 0, Spin: 0
├── Optimise géométrie neutre
├── Vérifie/corrige fréquences (check_and_fix_frequencies)
└── Base pour toutes les autres étapes

Étape 2: sp_reduced() [93-138]
├── Charge: -1, Spin: 1 (doublet)
├── Single-point sur géométrie NEUTRE
├── Calcule EA vertical (affinité électronique)
└── Pas d'optimisation géométrique

Étape 3: optimize_reduced() [141-216]
├── Charge: -1, Spin: 1
├── Optimise depuis géométrie neutre
├── Vérifie/corrige fréquences
└── Calcule E_def (énergie déformation)

Étape 4: sp_oxidized() [219-264]
├── Charge: +1, Spin: 1 (doublet)
├── Single-point sur géométrie NEUTRE
├── Calcule IP vertical (potentiel ionisation)
└── Pas d'optimisation géométrique

Étape 5: optimize_oxidized() [267-342]
├── Charge: +1, Spin: 1
├── Optimise depuis géométrie neutre
├── Vérifie/corrige fréquences
└── Calcule E_def (énergie déformation)
```

### Structure Retournée par run_redox_pipeline

```python
{
    'neutral': AMSJob,      # Optimisation neutre
    'reduced_sp': AMSJob,   # Single-point réduit
    'reduced_opt': AMSJob,  # Optimisation réduit
    'oxidized_sp': AMSJob,  # Single-point oxydé
    'oxidized_opt': AMSJob  # Optimisation oxydé
}
```

---

## qmredox/analysis.py (985 lignes)

### Extraction d'Énergies

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `extract_energies_from_rkf(rkf_path, u_method)` | 15-74 | Extrait Ee, H, G, TS, U, HOMO, LUMO |
| `extract_all_energies(jobs_data, u_method)` | 77-110 | Extrait pour tous conformères/états |

### Clés RKF Extraites [15-74]

| Clé | Section RKF | Description |
|-----|-------------|-------------|
| `Ee` | `"Energy"/"SCF Bond Energy"` | Énergie électronique |
| `H` | `"Thermodynamics"/"Enthalpy"` | Enthalpie |
| `G` | `"Thermodynamics"/"Gibbs free Energy"` | Énergie libre Gibbs |
| `TS` | `"Thermodynamics"/"TS"` | Terme entropique |
| `U` | `"Thermodynamics"/"Internal Energy total"` | Énergie interne (méthode direct) |
| `HOMO` | `"Properties"/"HOMO"` | Orbital HOMO |
| `LUMO` | `"Properties"/"LUMO"` | Orbital LUMO |

### Poids de Boltzmann

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `read_conformer_weights_from_summary(workdir)` | 166-218 | Lit poids depuis conformers_summary.txt |
| `calculate_boltzmann_weights(energies, T)` | 221-265 | Recalcule depuis énergies Gibbs (fallback) |

**IMPORTANT:** Utilise poids du pipeline conformères si disponibles (plus fiables).

### Calcul Paramètres Redox

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `calculate_redox_parameters(energies)` | 268-379 | Calcule ΔG, EA, IP, E_def, ΔΔU, TΔS par conformère |
| `calculate_boltzmann_averaged_potentials(params, weights)` | 382-450 | Moyenne pondérée → Volts |

### Formules Redox [268-379]

**Réduction (M + e⁻ → M⁻):**
```
ΔG_red = G(M⁻_opt) - G(M_neutral)
EA = Ee(M⁻_sp) - Ee(M_neutral)           # Affinité électronique verticale
E_def = Ee(M⁻_opt) - Ee(M⁻_sp)           # Énergie déformation
ΔΔU = [U(M⁻_opt) - Ee(M⁻_opt)] - [U(M_neutral) - Ee(M_neutral)]
TΔS = TS(M⁻_opt) - TS(M_neutral)

Vérification: ΔG = EA + E_def + ΔΔU + TΔS
```

**Oxydation (M → M⁺ + e⁻):**
```
ΔG_ox = G(M⁺_opt) - G(M_neutral)
IP = Ee(M⁺_sp) - Ee(M_neutral)           # Potentiel ionisation vertical
E_def = Ee(M⁺_opt) - Ee(M⁺_sp)
ΔΔU = [U(M⁺_opt) - Ee(M⁺_opt)] - [U(M_neutral) - Ee(M_neutral)]
TΔS = TS(M⁺_opt) - TS(M_neutral)

Vérification: ΔG = IP + E_def + ΔΔU + TΔS
```

### Analyse Structurale

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `compare_conformers_rmsd(jobs_data)` | 453-508 | RMSD neutre vs réduit/oxydé |
| `extract_orbital_energies(jobs_data)` | 511-570 | HOMO/LUMO moyennés Boltzmann |

**Seuil RMSD:** 0.3 Å indique changement conformationnel significatif.

### Export

| Fonction | Lignes | Description |
|----------|--------|-------------|
| `export_results_to_file(...)` | 573-723 | Rapport texte complet |
| `export_redox_summary_csv(...)` | 725-771 | CSV 1 ligne: potentiels moyennés |
| `export_redox_conformers_csv(...)` | 774-823 | CSV 1 ligne/conformère |
| `export_orbital_data_csv(...)` | 826-868 | CSV HOMO/LUMO par conformère |
| `analyze_molecule_redox(...)` | 871-985 | **Orchestration principale** |

### Fonction Principale analyze_molecule_redox [871-985]

```python
# Workflow:
1. extract_all_energies(jobs_data, u_method)           # 890-894
2. read_conformer_weights_from_summary(workdir)        # 896-906
   └── OU calculate_boltzmann_weights() si fichier absent
3. calculate_redox_parameters(energies)                # 912-914
4. calculate_boltzmann_averaged_potentials(params, w)  # 916-918
5. compare_conformers_rmsd(jobs_data)                  # 920-922
6. extract_orbital_energies(jobs_data)                 # 924-931
7. Affichage + Export                                  # 933-983
```

---

## Dépendances

```
redox_calc.py
├── common/constants.py (R, F, TEMPERATURE, conversions)
├── common/functional.py (configure_functional, ALL_FUNCTIONALS)
├── common/frequency.py (check_and_fix_frequencies)
├── common/utils.py (print_*, boltzmann_weights)
└── qmredox/
    ├── common.py (setup_workspace, setup_adf_settings, kabsch_rmsd)
    ├── pipeline.py (5 étapes + run_redox_pipeline)
    └── analysis.py (extraction, calculs, export)
```

---

## Fichiers de Sortie

```
{molecule}_workdir/
└── redox/
    ├── calculations/
    │   └── plams_workdir/
    │       ├── {mol}_conf_{N}_neutral/
    │       ├── {mol}_conf_{N}_reduced_sp/
    │       ├── {mol}_conf_{N}_reduced_opt/
    │       ├── {mol}_conf_{N}_oxidized_sp/
    │       └── {mol}_conf_{N}_oxidized_opt/
    └── results/
        ├── {mol}_redox_analysis.txt
        ├── {mol}_redox_summary.csv      ← 1 ligne: potentiels moyennés
        ├── {mol}_redox_conformers.csv   ← 1 ligne/conformère
        └── {mol}_orbital_data.csv
```

---

## Diagnostic Rapide

| Problème | Où chercher |
|----------|-------------|
| Erreur chargement conformères | `redox_calc.py:151-189` (load_conformers_from_dir) |
| Poids Boltzmann incorrects | `analysis.py:166-218` (read_conformer_weights) |
| Incohérence ΔG | `analysis.py:268-379` (vérification décomposition) |
| Fréquences imaginaires | `pipeline.py` (check_and_fix_frequencies dans chaque opt) |
| Extraction énergies | `analysis.py:15-74` (extract_energies_from_rkf) |
| Conversion V | `analysis.py:382-450` (÷ F = 96485.33212) |
| RMSD élevé | `analysis.py:453-508` (compare_conformers_rmsd) |
