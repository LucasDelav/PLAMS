# Rapport des molécules manquantes par solvant

**Molécules attendues** : 73 (depuis `input.txt`)

**Solvants** : 10

## Résumé

| Solvant | Présentes | Workdir sans dataset | Manquantes | Total attendu |
|---------|-----------|---------------------|------------|---------------|
| Acetone | 38 | 0 | 35 | 73 |
| Acetonitrile | 67 | 1 | 5 | 73 |
| Chloroform | 38 | 0 | 35 | 73 |
| Dichloromethane | 38 | 0 | 35 | 73 |
| DMSO | 38 | 0 | 35 | 73 |
| Hexane | 71 | 0 | 2 | 73 |
| Methanol | 38 | 0 | 35 | 73 |
| Tetrahydrofuran | 38 | 0 | 35 | 73 |
| Toluene | 71 | 0 | 2 | 73 |
| Water | 61 | 1 | 11 | 73 |

Les 35 molécules manquantes dans 6 solvants (Acetone, Chloroform, Dichloromethane, DMSO, Methanol, Tetrahydrofuran) sont les **lignine-derived et phénylpropanoïdes** (lignes 50-85 de `input.txt`). Seuls Acetonitrile, Hexane, Toluene et Water disposent de ces calculs.

---

## Acetone

**38/73** molécules extraites dans le dataset.

### Molécules manquantes (35)

Aucun `*_workdir/` trouvé dans `data/Acetone/` :

- 3,4-DIHYDROXYACETOPHENONE
- 3-METHOXYCATECHOL
- 4-ALLYLCATECHOL
- 4-ETHYLCATECHOL
- 4-METHYLCATECHOL
- 4-PROPYLCATECHOL
- ACETOSYRINGONE
- ACETOVANILLONE
- CAFFEALDEHYDE
- CATECHOL
- CONIFERALDEHYDE
- CONIFERYL_ALCOHOL
- DIHYDROCAFFEIC_ACID
- DOPAC
- EPINEPHRINE
- ETHYL_PROTOCATECHUATE
- EUGENOL
- FERULIC_ACID
- GUAIACOL
- HYDROXYTYROSOL
- ISOEUGENOL
- METHYL_PROTOCATECHUATE
- NDGA
- NOREPINEPHRINE
- PROTOCATECHUALDEHYDE
- PROTOCATECHUIC_ACID
- SESAMOL
- SINAPALDEHYDE
- SINAPIC_ACID
- SINAPYL_ALCOHOL
- SYRINGALDEHYDE
- SYRINGIC_ACID
- SYRINGOL
- VANILLIC_ACID
- VANILLIN

---

## Acetonitrile

**67/73** molécules extraites dans le dataset.

### Molécules manquantes (5)

Aucun `*_workdir/` trouvé dans `data/Acetonitrile/` :

- 3,4-DIHYDROXYACETOPHENONE
- EUGENOL
- ISOEUGENOL
- NDGA
- SESAMOL

### Workdir présent mais extraction échouée (1)

Le `*_workdir/` existe mais aucun JSON n'a été produit :

- SINAPALDEHYDE

---

## Chloroform

**38/73** molécules extraites dans le dataset.

### Molécules manquantes (35)

Aucun `*_workdir/` trouvé dans `data/Chloroform/` :

- 3,4-DIHYDROXYACETOPHENONE
- 3-METHOXYCATECHOL
- 4-ALLYLCATECHOL
- 4-ETHYLCATECHOL
- 4-METHYLCATECHOL
- 4-PROPYLCATECHOL
- ACETOSYRINGONE
- ACETOVANILLONE
- CAFFEALDEHYDE
- CATECHOL
- CONIFERALDEHYDE
- CONIFERYL_ALCOHOL
- DIHYDROCAFFEIC_ACID
- DOPAC
- EPINEPHRINE
- ETHYL_PROTOCATECHUATE
- EUGENOL
- FERULIC_ACID
- GUAIACOL
- HYDROXYTYROSOL
- ISOEUGENOL
- METHYL_PROTOCATECHUATE
- NDGA
- NOREPINEPHRINE
- PROTOCATECHUALDEHYDE
- PROTOCATECHUIC_ACID
- SESAMOL
- SINAPALDEHYDE
- SINAPIC_ACID
- SINAPYL_ALCOHOL
- SYRINGALDEHYDE
- SYRINGIC_ACID
- SYRINGOL
- VANILLIC_ACID
- VANILLIN

---

## Dichloromethane

**38/73** molécules extraites dans le dataset.

### Molécules manquantes (35)

Aucun `*_workdir/` trouvé dans `data/Dichloromethane/` :

- 3,4-DIHYDROXYACETOPHENONE
- 3-METHOXYCATECHOL
- 4-ALLYLCATECHOL
- 4-ETHYLCATECHOL
- 4-METHYLCATECHOL
- 4-PROPYLCATECHOL
- ACETOSYRINGONE
- ACETOVANILLONE
- CAFFEALDEHYDE
- CATECHOL
- CONIFERALDEHYDE
- CONIFERYL_ALCOHOL
- DIHYDROCAFFEIC_ACID
- DOPAC
- EPINEPHRINE
- ETHYL_PROTOCATECHUATE
- EUGENOL
- FERULIC_ACID
- GUAIACOL
- HYDROXYTYROSOL
- ISOEUGENOL
- METHYL_PROTOCATECHUATE
- NDGA
- NOREPINEPHRINE
- PROTOCATECHUALDEHYDE
- PROTOCATECHUIC_ACID
- SESAMOL
- SINAPALDEHYDE
- SINAPIC_ACID
- SINAPYL_ALCOHOL
- SYRINGALDEHYDE
- SYRINGIC_ACID
- SYRINGOL
- VANILLIC_ACID
- VANILLIN

---

## DMSO

**38/73** molécules extraites dans le dataset.

### Molécules manquantes (35)

Aucun `*_workdir/` trouvé dans `data/DMSO/` :

- 3,4-DIHYDROXYACETOPHENONE
- 3-METHOXYCATECHOL
- 4-ALLYLCATECHOL
- 4-ETHYLCATECHOL
- 4-METHYLCATECHOL
- 4-PROPYLCATECHOL
- ACETOSYRINGONE
- ACETOVANILLONE
- CAFFEALDEHYDE
- CATECHOL
- CONIFERALDEHYDE
- CONIFERYL_ALCOHOL
- DIHYDROCAFFEIC_ACID
- DOPAC
- EPINEPHRINE
- ETHYL_PROTOCATECHUATE
- EUGENOL
- FERULIC_ACID
- GUAIACOL
- HYDROXYTYROSOL
- ISOEUGENOL
- METHYL_PROTOCATECHUATE
- NDGA
- NOREPINEPHRINE
- PROTOCATECHUALDEHYDE
- PROTOCATECHUIC_ACID
- SESAMOL
- SINAPALDEHYDE
- SINAPIC_ACID
- SINAPYL_ALCOHOL
- SYRINGALDEHYDE
- SYRINGIC_ACID
- SYRINGOL
- VANILLIC_ACID
- VANILLIN

---

## Hexane

**71/73** molécules extraites dans le dataset.

### Molécules manquantes (2)

Aucun `*_workdir/` trouvé dans `data/Hexane/` :

- 3,4-DIHYDROXYACETOPHENONE
- NDGA

---

## Methanol

**38/73** molécules extraites dans le dataset.

### Molécules manquantes (35)

Aucun `*_workdir/` trouvé dans `data/Methanol/` :

- 3,4-DIHYDROXYACETOPHENONE
- 3-METHOXYCATECHOL
- 4-ALLYLCATECHOL
- 4-ETHYLCATECHOL
- 4-METHYLCATECHOL
- 4-PROPYLCATECHOL
- ACETOSYRINGONE
- ACETOVANILLONE
- CAFFEALDEHYDE
- CATECHOL
- CONIFERALDEHYDE
- CONIFERYL_ALCOHOL
- DIHYDROCAFFEIC_ACID
- DOPAC
- EPINEPHRINE
- ETHYL_PROTOCATECHUATE
- EUGENOL
- FERULIC_ACID
- GUAIACOL
- HYDROXYTYROSOL
- ISOEUGENOL
- METHYL_PROTOCATECHUATE
- NDGA
- NOREPINEPHRINE
- PROTOCATECHUALDEHYDE
- PROTOCATECHUIC_ACID
- SESAMOL
- SINAPALDEHYDE
- SINAPIC_ACID
- SINAPYL_ALCOHOL
- SYRINGALDEHYDE
- SYRINGIC_ACID
- SYRINGOL
- VANILLIC_ACID
- VANILLIN

---

## Tetrahydrofuran

**38/73** molécules extraites dans le dataset.

### Molécules manquantes (35)

Aucun `*_workdir/` trouvé dans `data/Tetrahydrofuran/` :

- 3,4-DIHYDROXYACETOPHENONE
- 3-METHOXYCATECHOL
- 4-ALLYLCATECHOL
- 4-ETHYLCATECHOL
- 4-METHYLCATECHOL
- 4-PROPYLCATECHOL
- ACETOSYRINGONE
- ACETOVANILLONE
- CAFFEALDEHYDE
- CATECHOL
- CONIFERALDEHYDE
- CONIFERYL_ALCOHOL
- DIHYDROCAFFEIC_ACID
- DOPAC
- EPINEPHRINE
- ETHYL_PROTOCATECHUATE
- EUGENOL
- FERULIC_ACID
- GUAIACOL
- HYDROXYTYROSOL
- ISOEUGENOL
- METHYL_PROTOCATECHUATE
- NDGA
- NOREPINEPHRINE
- PROTOCATECHUALDEHYDE
- PROTOCATECHUIC_ACID
- SESAMOL
- SINAPALDEHYDE
- SINAPIC_ACID
- SINAPYL_ALCOHOL
- SYRINGALDEHYDE
- SYRINGIC_ACID
- SYRINGOL
- VANILLIC_ACID
- VANILLIN

---

## Toluene

**71/73** molécules extraites dans le dataset.

### Molécules manquantes (2)

Aucun `*_workdir/` trouvé dans `data/Toluene/` :

- 3,4-DIHYDROXYACETOPHENONE
- NDGA

---

## Water

**61/73** molécules extraites dans le dataset.

### Molécules manquantes (11)

Aucun `*_workdir/` trouvé dans `data/Water/` :

- 3,4-DIHYDROXYACETOPHENONE
- Al-CQ
- Br-DOPA
- Cl-DOPA
- H2C-CQ
- HO-CQ
- I-HMC
- IQ
- O-IQ
- P-IQ
- S-IQ

### Workdir présent mais extraction échouée (1)

Le `*_workdir/` existe mais aucun JSON n'a été produit :

- NDGA

---

## Molécules manquantes globalement (dans tous les solvants)

- **3,4-DIHYDROXYACETOPHENONE** : manquante partout (10/10 solvants)
- **NDGA** : manquante dans 9/10 solvants (présente uniquement dans Acetonitrile)
