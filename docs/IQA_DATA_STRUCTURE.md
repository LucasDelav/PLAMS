# Structure des données IQA dans les fichiers RKF

Ce document explique comment les énergies IQA (Interacting Quantum Atoms) sont stockées dans les fichiers RKF d'ADF et comment extraire les énergies additives EaddIQA.

## 1. Formule des énergies additives IQA

L'énergie additive IQA d'un atome A est définie comme :

```
EaddIQA(A) = Eintra(A) + ½ × Σ_{B≠A} Einter(A,B)
```

Où :
- **Eintra(A)** : énergie intra-atomique (cinétique + potentiel interne)
- **Einter(A,B)** : énergie d'interaction entre les atomes A et B

La somme des EaddIQA de tous les atomes donne l'énergie totale du système.

## 2. Le tableau `"IQA atom-atom total"`

### Localisation dans le RKF

```python
from scm.plams import KFFile
kf = KFFile("adf.rkf")
atom_atom = kf.read("Properties", "IQA atom-atom total")
```

### Contenu

Ce tableau contient **n(n+1)/2 éléments** pour n atomes, correspondant à la partie triangulaire supérieure d'une matrice symétrique n×n :

- **Éléments diagonaux** : Eintra(A) - énergies intra-atomiques
- **Éléments hors-diagonaux** : Einter(A,B) - énergies d'interaction

### Format de stockage : Triangulaire supérieur, ROW-MAJOR

Les éléments sont stockés **ligne par ligne**, en ne gardant que la partie triangulaire supérieure (i ≤ j).

#### Exemple pour n=3 atomes (H₂O)

Matrice conceptuelle :
```
          j=0        j=1        j=2
i=0  [ Eintra(O)   Einter(O,H2)  Einter(O,H3) ]
i=1  [    -        Eintra(H2)    Einter(H2,H3) ]
i=2  [    -            -         Eintra(H3)   ]
```

Stockage linéaire (6 éléments) :
```
Index 0 → (0,0) = Eintra(O)      = -74.914886
Index 1 → (0,1) = Einter(O,H2)   = -0.491975
Index 2 → (0,2) = Einter(O,H3)   = -0.490646
Index 3 → (1,1) = Eintra(H2)     = -0.303379
Index 4 → (1,2) = Einter(H2,H3)  = +0.122488
Index 5 → (2,2) = Eintra(H3)     = -0.304516
```

#### Formules de conversion

**Position matricielle (i,j) → Index linéaire** (avec i ≤ j) :
```python
idx = i * n - i * (i + 1) // 2 + j
```

**Indices diagonaux** :
```python
def diagonal_indices(n):
    """Retourne les indices des éléments diagonaux"""
    indices = []
    idx = 0
    for i in range(n):
        indices.append(idx)
        idx += (n - i)  # Sauter à la ligne suivante
    return indices
# Pour n=3: [0, 3, 5]
# Pour n=18: [0, 18, 35, 51, 66, 80, 93, 105, 116, 126, 135, 143, 150, 156, 161, 165, 168, 170]
```

## 3. Ordre des atomes : INPUT vs INTERNAL

### Le tableau `"atom order index"`

ADF peut réordonner les atomes en interne (regroupement par type chimique). Le tableau `"atom order index"` de longueur 2n contient :

| Partie | Indices | Signification |
|--------|---------|---------------|
| Première moitié | `[0:n]` | INPUT → INTERNAL |
| Seconde moitié | `[n:2n]` | INTERNAL → INPUT |

### Exemple : H₂O (pas de réordonnancement)

```python
atom_order = [1, 2, 3, 1, 2, 3]  # Mapping identité
```

### Exemple : Br-HMC (réordonnancement par type)

```python
atom_order = [1, 4, 5, 6, 7, 2, 8, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  # INPUT→INTERNAL
              1, 6, 8, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # INTERNAL→INPUT
```

ADF regroupe : O(1,6,8) → INTERNAL(1,2,3), C(2-5,7,9,10) → INTERNAL(4-10), Br(11) → INTERNAL(11), H(12-18) → INTERNAL(12-18)

### Ordre des tableaux IQA

**Les tableaux IQA utilisent l'ordre INPUT** (pas INTERNAL).

Cela signifie que l'élément diagonal d'index i correspond à l'atome i dans le fichier XYZ d'entrée, sans conversion nécessaire.

## 4. Code Python complet

```python
import numpy as np
from scm.plams import KFFile

def array_to_matrix_rowmajor(arr, n):
    """
    Convertit un tableau triangulaire supérieur row-major en matrice pleine symétrique.

    Args:
        arr: Tableau linéaire de n(n+1)/2 éléments
        n: Nombre d'atomes

    Returns:
        Matrice symétrique n×n
    """
    matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = arr[idx]
            matrix[j, i] = arr[idx]  # Symétrique
            idx += 1
    return matrix


def extract_EaddIQA(rkf_path):
    """
    Extrait les énergies additives IQA depuis un fichier RKF.

    Args:
        rkf_path: Chemin vers le fichier adf.rkf

    Returns:
        dict: {
            'n_atoms': int,
            'Eintra': np.array de taille n,
            'Einter': np.array matrice n×n (diagonale = 0),
            'EaddIQA': np.array de taille n,
            'total': float (somme des EaddIQA)
        }
    """
    kf = KFFile(rkf_path)

    # Nombre d'atomes
    n_atoms = int(kf.read("Geometry", "nr of atoms"))

    # Lire le tableau IQA atom-atom total
    atom_atom = np.array(kf.read("Properties", "IQA atom-atom total"))

    # Convertir en matrice pleine
    matrix = array_to_matrix_rowmajor(atom_atom, n_atoms)

    # Extraire Eintra (diagonale) et Einter (hors-diagonale)
    Eintra = np.diag(matrix)
    Einter = matrix.copy()
    np.fill_diagonal(Einter, 0)

    # Calculer EaddIQA pour chaque atome
    EaddIQA = np.zeros(n_atoms)
    for i in range(n_atoms):
        EaddIQA[i] = Eintra[i] + 0.5 * np.sum(Einter[i, :])

    return {
        'n_atoms': n_atoms,
        'Eintra': Eintra,
        'Einter': Einter,
        'EaddIQA': EaddIQA,
        'total': np.sum(EaddIQA)
    }


# Exemple d'utilisation
if __name__ == "__main__":
    result = extract_EaddIQA("adf.rkf")

    print(f"Nombre d'atomes: {result['n_atoms']}")
    print(f"\nÉnergies additives IQA (Hartree):")
    for i, e in enumerate(result['EaddIQA']):
        print(f"  Atome {i+1}: {e:.6f}")
    print(f"\nÉnergie totale: {result['total']:.6f}")
```

## 5. Vérification avec H₂O

### Valeurs de référence (depuis water.out)

| Atome | Eintra | Einter avec autres | EaddIQA |
|-------|--------|-------------------|---------|
| O1 | -74.914886 | O-H2: -0.491975, O-H3: -0.490646 | -75.406196 |
| H2 | -0.303379 | H2-O: -0.491975, H2-H3: +0.122488 | -0.488122 |
| H3 | -0.304516 | H3-O: -0.490646, H3-H2: +0.122488 | -0.488595 |
| **Total** | | | **-76.382913** |

### Calcul détaillé

```
EaddIQA(O)  = -74.914886 + ½×(-0.491975 + (-0.490646)) = -75.406196 ✓
EaddIQA(H2) = -0.303379  + ½×(-0.491975 + 0.122488)    = -0.488122  ✓
EaddIQA(H3) = -0.304516  + ½×(-0.490646 + 0.122488)    = -0.488595  ✓
```

## 6. Énergies typiques par type d'atome

Pour identifier le type d'un atome à partir de son Eintra :

| Type | Eintra approximatif (Ha) |
|------|-------------------------|
| H | -0.3 à -0.5 |
| C | -37 à -38 |
| N | -54 à -55 |
| O | -74 à -75 |
| Br | -2570 à -2575 |

## 7. Résumé

| Propriété | Valeur |
|-----------|--------|
| Section RKF | `"Properties"` |
| Variable | `"IQA atom-atom total"` |
| Format | Triangulaire supérieur, row-major |
| Taille | n(n+1)/2 éléments |
| Ordre des atomes | **INPUT** (ordre du XYZ d'entrée) |
| Diagonale | Eintra(A) |
| Hors-diagonale | Einter(A,B) |
| Formule | EaddIQA(A) = Eintra(A) + ½ × Σ_B Einter(A,B) |
