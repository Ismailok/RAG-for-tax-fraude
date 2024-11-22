# Advanced-RAG-for-Tax-Fraud-Project

Ce projet est une démonstration d'un assistant virtuel utilisant le modèle **Retrieval-Augmented Generation (RAG)** pour répondre à des questions sur le thème de la fraude fiscale. L'objectif est de simuler un système qui peut fournir des réponses précises à partir d'un ensemble de documents juridiques disponibles en ligne.

## Structure du Projet

### Arborescence des fichiers

- **`data/`** : Contient les ensembles de données nécessaires au projet.
  - `cleaned_data.csv` : Données nettoyées.
  - `data.csv` : Données brutes issues des documents juridiques.
  - `ground-truth-retrieval.csv` : Fichier contenant des exemples pour évaluer la précision.
  - `rag-eval-gpt-4o-mini.csv` : Résultats d'évaluations générés pour le modèle.
- **`src/`** : Contient les scripts pour l'ingestion des données et l'évaluation.
  - `db_prep.py` : Script pour préparer et structurer les bases de données.
  - `ingest.py` : Script pour ingérer les données dans un index de recherche.
  - `minsearch.py` : Fournit les fonctionnalités pour récupérer les informations à partir des données.
  - `rag.ipynb` : Contient l'implémentation principale du modèle RAG.
  - `rag_evaluation_data_generation.ipynb` : Permet de générer des données d'évaluation pour mesurer les performances.
- **`requirements.txt`** : Liste des dépendances.
- **`.envrc`** : Configuration des variables d'environnement pour le projet.
- **`Pipfile` et `Pipfile.lock`** : Gestion des dépendances et de l'environnement Python.
- **`README.md`** : Le Readme.

---

## Objectif et Description

L'objectif principal est de démontrer comment un assistant virtuel peut utiliser des techniques modernes de RAG pour fournir des réponses pertinentes à partir de documents juridiques concernant la fraude fiscale.

---

## Dataset

Les données utilisées sont issues de documents juridiques disponibles publiquement.

### Colonnes des fichiers :
- `id` : Identifiant unique pour chaque document ou section.
- `content` : Texte brut du document.
- `metadata` : Informations additionnelles, telles que la source ou la date.

### Préparation des données :
- Les données brutes ont été nettoyées, normaliser les formats et structurer le contenu dans un format tabulaire (`cleaned_data.csv`).

---

## Installation et Configuration

### Prérequis

- Python 3.10+
- Conda (ou un environnement virtuel équivalent)

### Étapes d'installation

1. **Cloner ce dépôt :**
   ```bash
   git clone https://github.com/username/tax-fraud-rag-project.git
   cd tax-fraud-rag-project
   ```


2. **Créer un environnement Conda :**
   ```bash
    conda create --name tax-fraud-env python=3.10+
    conda activate tax-fraud-env
    ```

3. **Installer les dépendances via requirements.txt :**
    ```bash
    pip install -r requirements.txt
    ```
    
    Ou, si vous préférez utiliser pipenv :
    
     ```bash
    pipenv install
    pipenv shell
    ```

## Dépendances

Les principales bibliothèques nécessaires sont :

- `tqdm` : Suivi de la progression des tâches.
- `openai` : Intégration avec GPT-4 pour la génération de réponses.
- `scikit-learn` : Pour l'analyse et l'évaluation des résultats.
- `pandas` : Manipulation et traitement des données tabulaires.

---

## Reproduction des Résultats


### Etape 1 : Préparer les données
1. Exécuter le script pour préparer la base de données :
   ```bash
   python src/db_prep.py
  ```

2. Indexer les données dans le moteur de recherche :
   ```bash
   python src/ingest.py
  ```

### Etape 2 : Exécuter le modèle RAG
    ```bash
    jupyter notebook src/rag.ipynb
    ```
Suivre les étapes du notebook rag.ipynb pour exécuter le modèle RAG et tester l'assistant virtuel.

ou lancer directement avec

    ```bash
    python src/rag.py
    ```

## Étape 3 : Évaluer les performances
1. Lancer le notebook d'évaluation des données générées :
    ```bash
    jupyter notebook src/rag_evaluation_data_generation.ipynb
    ```

2. Générer les métriques d'évaluation :
- **Précision** : Mesure si les passages retournés par l'index sont pertinents.
- **Score BLEU** : Compare les réponses générées à une vérité terrain définie dans ground-truth-retrieval.csv.


3. Les résultats d'évaluation sont enregistrés dans le fichier `rag-eval-gpt-4o-mini.csv`.


---
## Évaluation du Modèle RAG
Le modèle suit l'approche Retrieval-Augmented Generation (RAG) :

1. **Recherche** : Un index est utilisé pour retrouver les passages pertinents dans les données.
2. **Génération** : Un modèle GPT-4 utilise ces passages pour fournir des réponses contextuelles et précises.


### Métriques utilisées :
- **Précision** : Évalue la pertinence des passages extraits par rapport à la requête.
- **Score BLEU** : Mesure la similarité des réponses générées avec les réponses idéales (ground truth).
