# AFD Studio : Analyse Factorielle des Données & Clustering

Application interactive développée avec **Streamlit** pour la réalisation d'un pipeline complet d'Analyse Factorielle des Données (AFD) et de classification non supervisée.

---

## 📋 Table des matières
1. [Fonctionnalités](#-fonctionnalités)
2. [Structure du projet](#-structure-du-projet)
3. [Prérequis & Installation](#-prérequis--installation)
4. [Lancement de l'application](#-lancement-de-lapplication)
5. [Méthodologie statistique](#-méthodologie-statistique)
   - [Prétraitement & Normalisation](#1-prétraitement--normalisation)
   - [Analyse en Composantes Principales (ACP)](#2-analyse-en-composantes-principales-acp-normée)
   - [Classification non supervisée (K-Means & CAH)](#3-classification-non-supervisée)
6. [Documentation du code (`src/`)](#-documentation-du-code-src)

---

## 🚀 Fonctionnalités

- **Exploration & Nettoyage** :
  - Détection automatique des dimensions et des valeurs manquantes.
  - Encodage des variables qualitatives (ex: `GENDER` M $\rightarrow$ 1, F $\rightarrow$ 2).
  - Imputation des données manquantes par la moyenne de la variable.
  - Test et standardisation centrée-réduite ($Z$-score : $\mu = 0, \sigma = 1$).
- **Matrice de Corrélation** :
  - Heatmap des coefficients de corrélation linéaire de Pearson avec échelle de couleurs divergente.
  - Tableau interactif des corrélations.
- **Analyse en Composantes Principales (ACP)** :
  - Éboulis des valeurs propres (Scree plot) avec affichage du seuil de Kaiser ($\lambda = 1$) et du pourcentage d'inertie expliquée.
  - Cercle des corrélations vectoriel avec axes paramétrables.
  - Tableau des saturations factorielles (loadings des variables) et inerties cumulées.
- **Clustering Non Supervisé** :
  - **K-Means** : Partitionnement sur le premier plan factoriel (CP1 / CP2) avec affichage des observations et des centroïdes.
  - **CAH (Classification Ascendante Hiérarchique)** : Arbre hiérarchique (méthode de Ward) avec dendrogramme et ligne de coupure visuelle.
- **Exportation** :
  - Téléchargement du jeu de données enrichi des labels de cluster en un clic (CSV).

---

## 📂 Structure du projet

```text
AFD/
├── .venv/                      # Environnement virtuel Python
├── cancer_des_poumons.csv      # Jeu de données d'exemple
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation complète
├── app.py                      # Point d'entrée de l'application Streamlit
└── src/                        # Modules Python modulaires et documentés
    ├── __init__.py             # Initialisation du package src
    ├── preprocessor.py         # Fonctions de nettoyage, encodage et normalisation
    ├── pca_analysis.py         # Moteur de calcul statistique de l'ACP
    ├── clustering.py           # Algorithmes de partitionnement K-Means et CAH
    └── plots.py                # Visualisations scientifiques Matplotlib & Seaborn
```

---

## 🛠️ Prérequis & Installation

### 1. Prérequis
- **Python 3.10+** (recommandé : Python 3.12).

### 2. Création et activation de l'environnement virtuel

Sous Windows (PowerShell) :
```powershell
# Créer le venv
python -m venv .venv

# Activer le venv
.\.venv\Scripts\Activate.ps1
```

Sous macOS / Linux :
```bash
# Créer le venv
python3 -m venv .venv

# Activer le venv
source .venv/bin/activate
```

### 3. Installation des dépendances

Une fois l'environnement activé :
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Lancement de l'application

Lancez le serveur Streamlit local avec :
```powershell
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur par défaut à l'adresse :
```
http://localhost:8501
```

---

## 🔬 Méthodologie statistique

### 1. Prétraitement & Normalisation
Pour que les variables ayant de grandes échelles (ex: `AGE`) n'écrasent pas les variables binaires ou ordonnées (ex: `SMOKING`, `ANXIETY`), une normalisation standard centrée-réduite est appliquée :

$$z = \frac{x - \mu}{\sigma}$$

Chaque variable transformée possède une moyenne $\mu = 0$ et une variance $\sigma^2 = 1$.

### 2. Analyse en Composantes Principales (ACP normée)
L'ACP projette les données sur un repère orthogonal où chaque axe maximise la variance projetée (inertie) :
- **Valeurs propres ($\lambda_k$)** : Variance expliquée par l'axe $k$. Selon le critère de Kaiser, on conserve généralement les axes dont $\lambda_k > 1$.
- **Pourcentage d'inertie** :
  $$\% \text{Inertie}_k = \frac{\lambda_k}{\sum_{j} \lambda_j} \times 100$$
- **Saturations factorielles (Loadings)** : Corrélation entre la variable initiale $j$ et la composante $k$ :
  $$r(X_j, CP_k) = v_{jk} \sqrt{\lambda_k}$$

### 3. Classification non supervisée
- **K-Means** : Minimise la variance intra-classe (inertie intra) en assignant chaque observation au centroïde le plus proche dans l'espace factoriel.
- **Classification Ascendante Hiérarchique (CAH)** : Agrégation itérative selon le critère de Ward (minimisation de l'augmentation de l'inertie intra-classe).

---

## 📖 Documentation du code (`src/`)

Chaque fichier du dossier `src/` contient des docstrings au format standard et le typage statique :

- [`src/preprocessor.py`](src/preprocessor.py) :
  - `preprocess_data(df)` : Nettoie, convertit le genre et impute les NaN par la moyenne.
  - `check_and_normalize(df)` : Détecte l'état centré-réduit et applique `StandardScaler`.
  - `get_dataset_summary(df)` : Calcule les métriques globales et les décomptes de valeurs manquantes.
- [`src/pca_analysis.py`](src/pca_analysis.py) :
  - `perform_pca(df, n_components)` : Entraîne `sklearn.decomposition.PCA` et retourne une dataclass `PCAResult`.
- [`src/clustering.py`](src/clustering.py) :
  - `perform_kmeans(data, n_clusters)` : Exécute K-Means et retourne `KMeansResult`.
  - `perform_hierarchical(data, n_clusters, method)` : Calcule la matrice de liaison (`linkage`) et retourne `CAHResult`.
- [`src/plots.py`](src/plots.py) :
  - `plot_correlation_matrix(df)` : Heatmap Seaborn.
  - `plot_scree(eigenvalues, inertia_percentage)` : Diagramme d'éboulis avec droite de Kaiser.
  - `plot_correlation_circle(pca_components, eigenvalues, feature_names)` : Cercle trigonométrique unitaire factoriel.
  - `plot_kmeans_clusters(components, labels, centroids)` : Nuage de points sur CP1 / CP2 avec centroïdes marqués.
  - `plot_dendrogram(linkage_matrix, max_d)` : Arbre hiérarchique avec ligne de coupure horizontale.
