# PLAN — Recherche de modèle ΔG avec sélection de descripteurs

**Statut** : en attente d'exécution (rédigé 2026-04-29, à reprendre dans une conversation propre).

**Contexte** : ce travail fait suite à une exploration en 4 phases (build dataset, cost estimate, exhaustive search, solvent correction) déjà menée et dont les résultats sont dans `analysis/formula_search/results/`. Le présent plan **remplace** l'approche précédente par une démarche en 3 étapes basée sur la sélection de descripteurs avant la recherche de forme.

---

## Pré-requis — déjà en place

- `analysis/formula_search/data/merged_dataset.parquet` : 1602 lignes (73 mols × 10 solvants × ~2.3 conformères), 25 colonnes. Bug Water≡Chloroform déjà filtré (80 lignes Water dropées).
- 14 descripteurs candidats : `mu, eta, gamma, fplus_max, fplus_max_O, fminus_max, fminus_max_O, f2_max, f2_max_O, qbader_max, qbader_max_O, lr_diag_max, lvc2_red, lvc2_ox`.
- 8 cibles (toutes en Hartree) : `EA, IP, Erelax_red, Erelax_ox, ddU_red, ddU_ox, TdS_red, TdS_ox`.
- 4 paires (forme partagée, coefficients red/ox distincts) : (EA, IP), (Erelax_red, Erelax_ox), (ddU_red, ddU_ox), (TdS_red, TdS_ox).
- venv local : `analysis/formula_search/.venv/bin/python` (numpy, pandas, scipy, sklearn, optuna, pyarrow, matplotlib, seaborn).
- venv marais : `~/data/PlamsV5/analysis/formula_search/.venv/bin/python` (mêmes paquets, sans optuna ni matplotlib — réinstaller si besoin).
- Modules existants à réutiliser : `forms.py` (PolynomialForm, PadeForm, enumerate_*), `search.py` (CV K-fold groupée + ProcessPool), `solvent_correction.py` (familles Born/Onsager/log/Padé).

## À trancher au démarrage (poser ces 3 questions en début d'exécution)

1. **Pool initial** : ajoute-t-on un descripteur `lvc2_paired` (= lvc2_red quand on prédit un target red, lvc2_ox quand on prédit un target ox) ? Sinon on garde le pool actuel (lvc2_red et lvc2_ox séparés, communs aux deux targets de chaque paire). Recommandation : ajouter `lvc2_paired` car physiquement plus défendable.
2. **Étape 1 — agrégation des métriques de corrélation** : comment combiner Pearson + Spearman + dCor + MI + score recherche polynomiale pour décider du sous-pool retenu par paire ? Options : (a) ranking moyen, (b) seuil sur chaque métrique avec vote majoritaire, (c) score composite normalisé, (d) validation manuelle sur tableau récapitulatif. Recommandation : (d) avec un tableau pré-calculé.
3. **Étape 2 — structure du modèle solvant** : (a) ε_r entre dans le pool comme descripteur de plus et on relance l'exhaustif polynômes/Padé, ou (b) structure additive imposée `f(descripteurs) + g(ε_r)` avec g ∈ {Born, Onsager, log, Padé(ε)}. Recommandation : (a), plus général.

---

## Étape 1 — Sélection de descripteurs + forme générale (Acetonitrile uniquement)

### But
Pour chaque paire, identifier le sous-ensemble de descripteurs physiquement pertinents et la forme générale du modèle prédictif. **Tout sur Acetonitrile uniquement** (~169 lignes).

### Méthodes de corrélation à tester
Pour chaque descripteur × chaque cible (14 × 8 = 112 combinaisons par paire) :

- **Pearson** (linéaire) — `scipy.stats.pearsonr`
- **Spearman** (monotone) — `scipy.stats.spearmanr`
- **Distance correlation** (non-linéaire) — `dcor.distance_correlation` (à installer si absent : `dcor`)
- **Mutual information** — `sklearn.feature_selection.mutual_info_regression`
- **Score recherche polynomiale** : pour chaque descripteur seul, MAE 5-fold CV du meilleur polynôme de degré ≤ 3 (1, x, x², x³ et toutes leurs combinaisons de monômes). C'est la "corrélation polynomiale" demandée.

### Procédure
1. Calculer les 5 métriques par (descripteur, cible) → tableau 14 × 8 × 5 métriques.
2. Pour chaque paire, agréger les scores red/ox en une métrique paire (ex : max ou moyenne).
3. Ranger les descripteurs par paire selon chaque métrique.
4. Décision du sous-pool : selon la stratégie tranchée (cf. question 2 ci-dessus). Sortie attendue : 3 à 5 descripteurs retenus par paire.
5. Sur chaque sous-pool retenu, relancer la recherche exhaustive polynômes + Padé (cf. `forms.py` / `search.py` mais limité au sous-pool) avec K-fold CV groupée par molécule (5 folds). Top 10 par paire.
6. Choisir **une formule retenue par paire** parmi le top 10 (par MAE_total + parcimonie BIC + cohérence physique). Cette formule fixe les **monômes** (et donc les descripteurs) qui seront repris à l'étape 2.

### Validation
- Tous les fits font une **K-fold groupée par molécule** (5 folds, `GroupKFold(groups=molecule)`) — pas de fuite entre conformères de la même molécule.
- Métriques rapportées : MAE_red_cv, MAE_ox_cv, MAE_total, R²_red_cv, R²_ox_cv, BIC_total.

### Livrables
- `analysis/formula_search/results/step1_correlations.csv` : tableau 14 × 8 × 5 métriques.
- `analysis/formula_search/results/step1_selected_descriptors.json` : sous-pool retenu par paire.
- `analysis/formula_search/results/step1_top10_<pair>.csv` × 4 : recherche exhaustive sur sous-pool.
- `analysis/formula_search/results/step1_final_form_<pair>.json` × 4 : forme retenue avec coefficients (Acetonitrile fit).

### Coût estimé
- Métriques de corrélation : < 1 min (local).
- Recherche exhaustive sur sous-pool ~5 descripteurs : ~5 min sur 20 cores marais.

---

## Étape 2 — Correction solvant sur le sous-pool fixé

### But
Étendre la formule de l'étape 1 aux 10 solvants. Le pool de descripteurs est **figé** par paire au sous-pool retenu à l'étape 1 (plus aucun test sur les descripteurs hors-pool).

### Procédure
- Dataset complet : 1602 lignes (10 solvants).
- Ajouter `eps_r` au DataFrame via la table de constantes diélectriques (déjà dans `solvent_correction.py`, dict `EPSILON_R`).
- Selon la décision (question 3) :
  - **(a) ε_r comme descripteur** : pool = sous-pool étape 1 ∪ {ε_r}. Relancer la recherche exhaustive polynômes/Padé sur ce pool agrandi (cf. `forms.py`/`search.py`). Note : le pool augmenté reste petit (4-6 descripteurs) → recherche tractable.
  - **(b) structure additive** : reprendre la forme retenue à l'étape 1, garder ses monômes, ajouter terme `g(ε_r)` ∈ {Born, Onsager, log, Padé(ε)}. Fit conjoint des coefficients θ et φ via curve_fit non-linéaire (préférer Levenberg-Marquardt avec bornes pour éviter divergence numérique observée à la phase 4 précédente).
- 5-fold CV groupée par molécule sur les 1602 lignes.
- Sortie : forme finale par paire avec coefficients θ_red, θ_ox et paramètres solvant φ.

### Validation
- MAE pooled (toutes lignes confondues) ET MAE moyennée par solvant — vérifier qu'aucun solvant n'est aberrant.
- Comparer à la baseline : forme étape 1 appliquée tel quel sur multi-solvant sans correction.

### Livrables
- `analysis/formula_search/results/step2_top10_<pair>.csv` × 4 : recherche multi-solvant.
- `analysis/formula_search/results/step2_final_form_<pair>.json` × 4 : forme finale + coeffs + correction.
- `analysis/formula_search/results/step2_per_solvent_mae.csv` : MAE par (paire, solvant) pour la forme finale.

### Coût estimé
- ~10 min sur 20 cores marais (pool ~6 descripteurs, 1602 points × 5 folds × ~50000 formes).

---

## Étape 3 — Parity plots

### But
Visualiser la qualité prédictive du modèle final sur les 8 termes + ΔG dérivés.

### Plots à produire
8 plots individuels (un par terme) :
- EA, IP, Erelax_red, Erelax_ox, ddU_red, ddU_ox, TdS_red, TdS_ox.

2 plots dérivés (combinaison via formules du rapport) :
- ΔG_red = −EA + Erelax_red + ddU_red − TdS_red
- ΔG_ox  = +IP + Erelax_ox + ddU_ox + TdS_ox

(Vérifier les conventions de signe dans `qmredox/analysis.py` avant la combinaison.)

### Spécifications graphiques
- Format : grille 4×3 (8 + 2 + 2 vides) ou 5×2 (10 plots), choix esthétique.
- Abscisse : valeur DFT en **kJ/mol** (conversion × 2625.499638).
- Ordonnée : prédiction modèle en **kJ/mol**.
- Couleur : par solvant (palette qualitative à 10 valeurs).
- Diagonale y = x en pointillés noirs.
- Annotations : MAE et R² globaux dans chaque sous-plot.
- Tous les 1602 points affichés.

### Livrables
- `analysis/formula_search/results/step3_parity_plots.pdf` : grille des 10 parity plots.
- `analysis/formula_search/results/step3_parity_<term>.pdf` × 10 : versions individuelles haute résolution (optionnel).

### Coût estimé
- < 1 min (local).

---

## Ordre d'exécution attendu

1. Trancher les 3 questions de cadrage avec l'utilisateur.
2. Étape 1 → trancher la sélection de descripteurs avec l'utilisateur (point 4 de la procédure).
3. Étape 2 → confirmer la forme finale avec l'utilisateur avant de figer.
4. Étape 3 → générer les plots.

## Fichiers à créer

- `analysis/formula_search/step1_correlations.py`
- `analysis/formula_search/step2_solvent_search.py` (peut réutiliser une grande partie de `search.py` et `solvent_correction.py`)
- `analysis/formula_search/step3_parity_plots.py`

## Fichiers existants à NE PAS détruire

- `build_dataset.py`, `forms.py`, `search.py`, `solvent_correction.py`, `cost_estimate.py`, `inspect_results.py`, `export_models.py` : restent comme bibliothèque réutilisable.
- `data/merged_dataset.parquet` : ne pas reconstruire sauf si on ajoute lvc2_paired (alors mettre à jour `build_dataset.py`).
- `results/*` : conserver comme baseline d'ancien plan, ajouter les nouveaux livrables avec préfixe `step1_`, `step2_`, `step3_`.
