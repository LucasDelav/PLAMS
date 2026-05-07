# AORESPONSE — Calcul de polarizabilité statique pour TΔS

**Date** : 2026-05-02
**Objectif** : extraire la polarizabilité moléculaire α_iso(ω→0) pour chaque conformère afin d'enrichir le modèle prédictif de TΔS au-delà du plafond actuel (MAE 1.39/1.86 kJ/mol).

## Contexte

Toutes les hypothèses physiques basées sur les SP-verticaux actuels (CDFT, QTAIM, RRHO, qRRHO Grimme, Marcus inner-sphere, Δspectrum) plafonnent à corr_intra ~0.30 et MAE ~1.77 (TΔS_ox). La polarizabilité est la seule donnée additionnelle accessible **sans relaxation ionique**, qui pourrait capter la **distorsion électronique post-ionisation** modulant les modes vibrationnels mous via Sum-Over-States.

## Modification du pipeline qmredox

### Pour chaque SP vertical (neutral_sp, reduced_sp, oxidized_sp)

Ajouter le block `AORESPONSE` dans la définition `Settings` du SP. Dans `qmredox/common.py` (ou wherever `setup_adf_settings` est défini), ajouter pour les SP :

```python
# Polarizability via TDDFT-Sternheimer (static limit)
s.input.adf.AORESPONSE.STATIC = ""   # bloc "STATIC" sans valeur = polarizabilité statique
# Optionnel : freq dépendante (à activer pour α(ω) au-delà de DC)
# s.input.adf.AORESPONSE.FREQUENCY = "0.0 0.0428 Hartree"   # ex. DC + 1064 nm
```

**Recommandation basis set** : la polarizabilité bénéficie de fonctions diffuses. Notre TZ2P actuel est OK, mais TZ2P+ ou ATZP donneraient α plus précis. À évaluer sur quelques conformères de test.

**Coût** :
- ~5–10× un SP standard (résolution Sternheimer pour 3 directions x, y, z)
- Géom neutre déjà optimisée → on relance juste les 3 SP par conformère
- Total estimé : 1336 conformères × 3 SP × ~5 min = **~330 h CPU**, soit ~3 jours sur 5 nœuds CRIANN parallèles

## Extraction depuis le RKF

Une fois calculé, α est stocké dans la section `Properties` du fichier `adf.rkf` :

```python
from scm.plams import KFFile
import numpy as np

kf = KFFile("path/to/adf.rkf")

# Tenseur de polarizabilité statique (3×3, en a.u.)
alpha_tensor = np.asarray(kf.read("Properties", "Polarizability tensor")).reshape(3, 3)

# Quantités dérivées
alpha_iso = float(np.trace(alpha_tensor) / 3.0)         # isotrope (scalar)
alpha_aniso = float(np.sqrt(0.5 * ((alpha_tensor[0,0]-alpha_tensor[1,1])**2
                                    + (alpha_tensor[1,1]-alpha_tensor[2,2])**2
                                    + (alpha_tensor[2,2]-alpha_tensor[0,0])**2
                                    + 6*(alpha_tensor[0,1]**2 + alpha_tensor[0,2]**2 + alpha_tensor[1,2]**2))))
alpha_eigenvalues = sorted(np.linalg.eigvalsh(alpha_tensor))
```

**Clé exacte du RKF** à confirmer : ADF utilise tantôt `"Polarizability tensor"`, tantôt `"polarizability"`, tantôt `"alpha"` selon la version. Faire `kf.sections()` et `kf.read("Properties", "")` pour lister.

## Modification de extract_predict_dg.py

Ajouter dans la fonction d'extraction des propriétés du SP :

```python
def extract_polarizability(rkf_path):
    """Extract static polarizability tensor from RKF."""
    out = {'alpha_iso': None, 'alpha_aniso': None, 'alpha_eigs': None}
    try:
        kf = KFReader(rkf_path)
        for key_name in ("Polarizability tensor", "polarizability", "alpha"):
            try:
                val = kf.read("Properties", key_name)
                if val is not None:
                    arr = np.asarray(val).reshape(3, 3)
                    out['alpha_iso'] = float(np.trace(arr) / 3.0)
                    out['alpha_aniso'] = float(...)  # voir formule ci-dessus
                    out['alpha_eigs'] = sorted(np.linalg.eigvalsh(arr).tolist())
                    break
            except Exception:
                pass
    except Exception:
        pass
    return out
```

Et l'incorporer dans la sortie JSON pour chaque état :

```python
state_dict['polarizability'] = extract_polarizability(rkf_path)
```

## Hypothèses à tester une fois α disponible

### H_α1 — Δα_iso comme correction multiplicative
```
y_TΔS = (θ_μ·μ + θ_η·η) × (1 + a·f²) × (1 + b·Δα_iso_paired)
```
avec `Δα_iso_paired = α_iso(ion_sp) − α_iso(neutre)`. Justification : un ion plus polarizable → modes vibrationnels affaiblis → ΔS plus grand.

### H_α2 — α_iso(neutre)/V_mol comme proxy de la mollesse électronique
```
y_TΔS = ... + λ · (α_iso_neutre / N_atoms)
```
La polarizabilité par atome capture la "mollesse électronique" intrinsèque qui devrait corréler avec la fragilité vibrationnelle.

### H_α3 — Anisotropie Δ(α_aniso) comme proxy de redistribution directionnelle
```
y_TΔS = ... + λ · Δα_aniso_paired
```
Justification : la redistribution électronique post-ionisation crée une asymétrie dans la PES → modes mous redistribués → ΔS modifié.

### H_α4 — Combinaison avec Marcus outer-sphere
La polarizabilité du solvant (intégrée dans COSMO) + α_solute permet de calculer le **terme outer-sphere de Marcus** pour la composante entropique :
```
TΔS_outer ∝ T·(∂λ_outer/∂T)
```
Plus complexe à implémenter mais physiquement défendable.

## Estimation du gain attendu

D'après la littérature (openCOSMO-RS atomic polarizability), α_atomique améliore les prédictions thermodynamiques de ~10–30 % sur des datasets type COSMO. Pour TΔS spécifiquement, je m'attends à un gain MAE de **0.1–0.3 kJ/mol** (5–15 %) si l'hypothèse physique est correcte. Ça nous amènerait à TΔS_ox MAE ~1.6 kJ/mol et briserait peut-être le plafond intra-capture vers 20–25 %.

Si le gain est < 5 %, la conclusion finale sera que **TΔS atteint sa limite intrinsèque** sans relaxation ionique.

## Procédure de relance

1. **Test sur 5–10 conformères** : valider le keyword AORESPONSE et l'extraction RKF avant la relance massive
2. **Relance batchée** : utiliser le même workflow SLURM CRIANN avec input modifié
3. **Synchronisation nextcloud** : nouveaux RKF + extraction predict-dg
4. **Test** : lancer `analysis/formula_search/test_polarizability_tds.py` (à créer) avec hypothèses H_α1 à H_α3

## Fichiers à modifier

- `qmredox/common.py` : ajouter `AORESPONSE STATIC` dans `setup_adf_settings(task='SinglePoint')`
- `extract_predict_dg.py` : ajouter `extract_polarizability()` et l'inclure dans le JSON
- `analysis/formula_search/test_polarizability_tds.py` : nouveau script de test des hypothèses H_α

## Références

- [ADF AORESPONSE documentation](https://www.scm.com/doc/ADF/Input/Polarizabilities.html)
- [openCOSMO-RS atomic polarizability](https://arxiv.org/html/2502.08520) — précédent d'utilisation pour thermodynamique
- [Vibrational effects on polarizability](https://www.nature.com/articles/s41598-025-88066-5) — couplage α ↔ vibrations
