# PLAMS - Outils pour l'analyse conformationnelle et les calculs redox

Ce repository contient des scripts Python utilisant la bibliothèque PLAMS (Python Library for Automating Molecular Simulation) pour effectuer des calculs de chimie computationnelle avancés avec Amsterdam Modeling Suite (AMS).

## Fonctionnalités

- **conformers_gen.py** : Génère des conformères à partir d'une structure SMILES via un workflow en quatre étapes (génération, optimisation, scoring et filtrage)
- **redox_calc.py** : Effectue des calculs redox sur les conformères générés (optimisation neutre, calcul du point simple réduit, optimisation réduite)

## Prérequis

- [Amsterdam Modeling Suite (AMS)](https://www.scm.com/amsterdam-modeling-suite/) 2021 ou plus récent
- Python 3.7 ou plus récent
- Module PLAMS (inclus dans l'installation d'AMS)
- Module RDKit (pour la génération des conformères, inclus dans l'installation d'AMS)

## Installation et configuration

1. Installez Amsterdam Modeling Suite en suivant les instructions sur [le site web de SCM](https://www.scm.com/support/downloads/)

2. Clonez ce repository :
   ```bash
   git clone https://github.com/LucasDelav/PLAMS.git
   cd PLAMS
   ```

3. Rendez les scripts exécutables :
   ```bash
   chmod +x conformers_gen.py redox_calc.py
   ```

4. Assurez-vous que les variables d'environnement d'AMS sont correctement configurées :
   ```bash
   source $AMSDIR/amsbashrc.sh  # Pour Linux/MacOS
   # ou
   %AMSDIR%\amscmd.bat  # Pour Windows
   ```

## Utilisation des scripts

### Générateur de conformères (conformers_gen.py)

Ce script génère un ensemble de conformères pour une molécule donnée à partir de sa représentation SMILES, puis effectue une série d'optimisations et de filtrage pour obtenir les conformères les plus stables.

**Workflow du script** :
1. **Génération** : Utilise RDKit pour générer 1000 conformères initiaux
2. **Optimisation** : Optimise la géométrie des conformères avec DFTB3
3. **Scoring** : Calcule les énergies précises avec ADF (fonctionnelle PBE0, base TZP)
4. **Filtrage** : Filtre les conformères redondants par RMSD

**Exemple d'utilisation** :
```bash
$AMSBIN/amspython conformers_gen.py "C1(=O)C(=O)C=CC(=C1)C[C@@H](C(=O)O)N" "L-DOPA"
```

**Paramètres** :
- Premier argument : Chaîne SMILES de la molécule
- Deuxième argument : Nom de la molécule (utilisé pour les fichiers de sortie)

**Sortie** :
- Fichiers XYZ des conformères générés dans un dossier `[nom]_conformers`
- Informations sur les énergies relatives et les poids de Boltzmann

### Calculateur redox (redox_calc.py)

Ce script prend les conformères générés (fichiers XYZ) et réalise trois étapes de calcul pour étudier leurs propriétés à l'état réduit.

**Workflow du script** :
1. **Optimisation neutre** : Optimisation de géométrie de la molécule neutre
2. **Calcul simple point réduit** : Calcul d'un point simple sur la molécule réduite (charge -1)
3. **Optimisation réduite** : Optimisation de géométrie de la molécule réduite

**Exemple d'utilisation** :
```bash
$AMSBIN/amspython redox_calc.py L-DOPA_conformers --prefix L-DOPA --solvent Acetonitrile
```

**Paramètres** :
- `input_dir` : Dossier contenant les fichiers XYZ des conformères
- `--prefix` : Préfixe pour les fichiers de sortie (facultatif)
- `--solvent` : Solvant à utiliser pour les calculs (par défaut: Acetonitrile)
- `--skip-confirm` : Ne pas demander de confirmation avant de démarrer les calculs (facultatif)

**Sortie** :
- Structures optimisées (neutres et réduites) dans le dossier `redox_structures`
- Tableau récapitulatif des énergies et des énergies de réduction
- Identification des conformères les plus stables à l'état neutre et réduit

## Détails des calculs

### Paramètres computationnels utilisés

#### Génération des conformères
- Méthode initiale : RDKit avec 1000 conformères initiaux
- Optimisation géométrique : DFTB3 avec le paramétrage 3ob-3-1
- Calculs d'énergie finaux : ADF avec fonctionnelle PBE0, base TZP et dispersion Grimme3
- Critère de filtrage RMSD : 0.5 Å

#### Calculs redox
- Méthode DFT : ADF avec fonctionnelle PBE0, base TZP
- Modèle de solvatation : COSMO
- État réduit : Charge -1, spin polarisé 1.0

## Estimations de temps de calcul

Les temps de calcul typiques observés pour une molécule organique de taille moyenne (comme la L-DOPA) sont :

- **Génération de conformères** : ~2-4 heures pour un workflow complet
- **Calculs redox par conformère** :
  - Optimisation neutre : ~32 minutes (intervalle typique: 21-40 min)
  - Simple point réduit : ~43 minutes (intervalle typique: 27-52 min)
  - Optimisation réduite : ~50 minutes (intervalle typique: 30-60 min)

Le script `redox_calc.py` inclut une estimation automatique du temps total basée sur le nombre de conformères à traiter.

## Conseils d'utilisation

1. **Choix de la structure SMILES** :
   - Assurez-vous que la structure SMILES est correcte, notamment pour la stéréochimie
   - Pour les molécules complexes, préparez le SMILES avec un logiciel comme ChemDraw

2. **Ressources computationnelles** :
   - Les calculs peuvent être intensifs, surtout pour les grandes molécules
   - Considérez l'utilisation d'un cluster de calcul pour les molécules avec plus de 40 atomes

3. **Analyse des résultats** :
   - Le conformère avec l'énergie la plus basse n'est pas toujours celui avec les meilleures propriétés redox
   - Examinez les structures optimisées pour comprendre les changements structurels lors de la réduction

## Dépannage

- **Erreur "ImportError: No module named scm.plams"** :
  - Assurez-vous que les variables d'environnement d'AMS sont correctement configurées
  - Exécutez les scripts avec `$AMSBIN/amspython` et non `python`

- **Échec d'un calcul d'optimisation** :
  - Vérifiez la structure initiale pour détecter d'éventuelles anomalies
  - Essayez d'augmenter le nombre maximal d'itérations dans les paramètres d'optimisation

- **Temps de calcul excessif** :
  - Réduisez le nombre de conformères en ajustant les seuils d'énergie
  - Utilisez une méthode moins coûteuse pour les calculs préliminaires

## Licence

Ce projet est disponible sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Contributions

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Citation

Si vous utilisez ces scripts dans votre travail de recherche, veuillez citer :

```
AMS 2023.1, SCM, Theoretical Chemistry, Vrije Universiteit, Amsterdam, The Netherlands, https://www.scm.com
PLAMS - Python Library for Automating Molecular Simulation
https://github.com/LucasDelav/PLAMS
```
