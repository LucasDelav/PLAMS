# Plan de corrections du rapport — 2026-04-27

Plan d'execution pour les corrections du rapport `rapport_workflow_plamsv5.tex`
identifiees lors de l'audit exhaustif du 2026-04-27.

## Phase A — Verifications prealables (TERMINEE 2026-04-27)

- [x] A1. Inspecter structure JSON `dataset_predict_dg/` :
  - **Disponibles 100%** (1710/1710 conformeres, 10 solvants) :
    - CDFT globaux : `states.neutral_sp.cdft.{mu,eta,omega}`
    - Fukui agreges : `aggregated.neutral_sp.{fplus,fminus,f0}.O`
    - Fukui atomiques : `states.neutral_sp.cdft.{fplus,fminus,dual}` (vecteurs)
    - Bader sumO : `aggregated.neutral_sp.bader_charges.O`
    - LVC2 ox/red : `derived.E_relax_LVC2_{red,ox}`
    - Hessien neutre : `states.neutral_sp.hessian` (matrice complete)
    - Modes vibrationnels ioniques : `states.reduced_sp.vibrations.modes`
- [x] A2. **Descripteur MANQUANT** : `f_O^LUMO` (composition de la LUMO sur les
  oxygenes via analyse SFO).
  - `sfo.lumo_pop = null` partout, autant dans `dataset/` que dans
    `dataset_predict_dg/` (1722 + 1710 conformeres tous null)
  - Seul endroit ou f_O^LUMO existe : `rapport/data/descriptors_per_molecule.csv`
    pour 19 molecules **Acetonitrile uniquement**
  - Les RKF locaux dans `data/{solvant}/{mol}_workdir/redox/results/rkf/` sont
    des liens symboliques cassés (vrais fichiers sur cluster CRIANN)
  - **Decision** : f_O^LUMO sera recupere plus tard (calcul SFO post-traitement
    DENSF/AMS sur les RKF du cluster). En attendant :
    - Sec. 3.2.2 (Acetonitrile, F10/F11/F14) reste inchangee (CSV existant)
    - Sec. 4.2 (multi-solvant) utilise Fukui (f+_sumO, f±_max,O) au lieu de
      f_O^LUMO. Justification : r=0.98 documentee entre f+_sumO et f_O^LUMO
      sur Acetonitrile (rapport F14, p.19).

## Phase B — Regenerations existantes (corrections valeurs n)

- [ ] B1. Copier `results/figures/section4/*.png` -> `rapport/figures/section4/`
  (regen deja faite, n=1402 au lieu de 1394).
- [ ] B2. Modifier `n=1394` -> `n=1402` dans `rapport/generate_section5_figures.py`
  (ligne 119) puis relancer pour mettre a jour s5_mae_comparison.png
  et s5_parity_dG.png.
- [ ] B3. Tableau 4 du rapport (.tex) : corriger HO-CA EA `153{,}6` -> `136{,}3`
  (effet de la suppression d'un conformere depuis la generation initiale).

## Phase C — Nouvelles analyses E_relax multi-solvant

- [ ] C1. Adapter `rapport/generate_section32_figures.py` :
  - Charger les donnees depuis `dataset/` + `dataset_predict_dg/` multi-solvant
    (au lieu de Acetonitrile uniquement)
  - Reproduire F8 (LVC1 vs LVC2 parity) sur n=1402 au lieu de n=15
  - Reproduire F9 (LVC2 vs DFT bar charts), agreges et/ou par solvant
  - Reproduire F10 (residu LVC2 vs f_O^LUMO) avec couleur par epsilon
  - Reproduire F11 (LVC2 brut vs corrige) multi-solvant
- [ ] C2. Adapter `rapport/generate_section32_fukui_figures.py` :
  - Reproduire F12-F13 (Fukui vs residu LVC2) multi-solvant
  - Reproduire F15-F16 (modele f±_max,O sans/avec parametre)
- [ ] C3. Etendre tableau T9 (performances E_relax) sur n=1402.
- [ ] C4. **Nouvelle analyse** : qualite de prediction LVC2 par solvant
  (montrer si l'eau est un cas limite, si les solvants apolaires degradent
  la qualite, etc).
- [ ] C5. **Nouvelle analyse** : distribution des residus LVC2 par famille
  moleculaire x solvant (catechols vs quinones vs lignanes, etc).

## Phase D — Texte LaTeX

- [ ] D1. Appliquer les ~25 corrections de comptes :
  - `1394` -> `1402` (20 occurrences ligne 1113, 1122, 1138, 1145, 1152, 1176,
    1212, 1219, 1226, 1250, 1286, 1310, 1343, 1367, 1403, 1410, 1424)
  - `1694` -> `1682` (4 occurrences ligne 1491, 1502, 1531, 1538)
- [ ] D2. Ligne 1122 : corriger description DEA :
  - **Faux** : « 5 molecules presentant un attachement electronique dissociatif
    (Δd > 0.3 Å) -- Br-DA, Br-HMC, I-DA, I-HMC, L-DOPA »
  - **Correct** : « 4 molecules DEA halogenees (Br-DA, Br-HMC, I-DA, I-HMC)
    detectees par rupture de liaison C-X (Δd > 1.0 Å) plus L-DOPA exclue
    pour reorganisation conformationnelle extreme en oxydation »
- [ ] D3. Ligne 1122 : corriger filtre EA<0 : `112 conformeres` -> `111 conformeres`.
- [ ] D4. Ligne 1491 : supprimer mention « deduplication des alias historiques
  (CAFFEIC_ACID, CAFFEYL_ALCOHOL et DOPAMINE) » — ces entrees n'existent plus
  dans le dataset actuel, le filtre est inactif.
- [ ] D5. **Ajouter paragraphe explicatif des filtres** dans sec. 4 (jeu de
  donnees) :
  > « A partir des 1722 conformeres bruts (73 mol x 10 solv), la chaine de
  > filtrage retire successivement : 159 conformeres des 5 molecules DEA
  > exclues, 24 conformeres avec etats DFT manquants, 111 conformeres dans
  > l'hexane (epsilon trop faible -> EA<0), 12 conformeres sans LVC2
  > disponible, et 14 conformeres avec LVC2 pathologique (|LVC2| > 150 kJ/mol,
  > frequences imaginaires dans le hessien ionique), laissant 1402 conformeres
  > pour les analyses Pade. »
- [ ] D6. **Distinguer le compte ML (1682) du compte sec. 4 (1402)** : preciser
  que le ML n'applique pas le filtre EA<0.

## Phase E — Compilation et verification

- [ ] E1. Recompiler `pdflatex rapport_workflow_plamsv5.tex` deux fois (refs
  croisees).
- [ ] E2. Verifier visuellement le PDF :
  - Tous les `n = 1402` apparaissent correctement
  - Le nouveau paragraphe explicatif est lisible
  - Les figures section 4 et 5 ont bien l'annotation `n` mise a jour
  - Le tableau 4 a la valeur HO-CA EA corrigee

## Etat d'avancement

Audit initial : termine 2026-04-27 (cf. tasks #1-#6 dans la session).
Phases A, B, C, D : terminees 2026-04-27.
Phase E : tex valide (pas de ref orpheline), recompilation pdflatex restante
  cote utilisateur (LaTeX non installe sur la machine de developpement).

## Recapitulatif des changements appliques au .tex

- Tableau 4 (sec. 3) regenere avec valeurs Boltzmann actuelles (15 catechols)
- 20 occurrences `n=1394` -> `n=1402` dans sec. 4
- 4 occurrences `n=1694` -> `n=1682` dans sec. 5
- Paragraphe « Jeu de donnees » sec. 4 reecrit avec chaine de filtrage
  detaillee (1722 brut -> 1402 final)
- Suppression mention obsolete deduplication alias (CAFFEIC_ACID etc.)
  en sec. 5
- Distinction explicite ML (n=1682) vs Pade (n=1402) ajoutee
- Nouvelle sous-section 4.2.1 « Analyse multi-solvant des Fukui et qualite
  par solvant » : 5 figures et 1 tableau sur n=1511
- Note explicite que f_O^LUMO manquant en multi-solvant, justification de
  la substitution par f+_sumO (r=0.98)

## Figures regenerees

- Toutes les figures section 4 (s4_*.png) avec n=1402
- Section 5 ML : s5_mae_comparison.png, s5_parity_dG.png avec n=1682
- Section 4.2 multi-solvant (NEW) : 5 nouvelles figures dans figures/section42/

## Reste a faire (cote utilisateur)

1. Recompiler le PDF : `cd rapport && pdflatex rapport_workflow_plamsv5.tex`
   (deux passes pour les references croisees)
2. Verifier visuellement : chaine de filtrage en sec. 4, nouvelle
   sous-section 4.2.1, tableau 4 mis a jour
3. (Plus tard) Calcul SFO pour recuperer f_O^LUMO multi-solvant et
   etoffer la sec. 4.2.1 avec F10/F11/F14 etendues

## Valeurs de reference confirmees (ne pas retoucher)

Toutes les MAE/R² des tableaux T6, T7, T8, T12-T19, T21-T22 ont ete verifies
exacts. Seul le compte de conformeres et la description DEA en sec. 4
necessitent une correction. Voir audit complet dans la session du 2026-04-27.
