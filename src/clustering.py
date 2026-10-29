"""
Module de classification non supervisée (Clustering).

Ce module implémente deux approches complémentaires de partitionnement :
1. Les K-Moyennes (K-Means) appliquées sur les coordonnées factorielles de l'ACP.
2. La Classification Ascendante Hiérarchique (CAH) avec calcul de dendrogramme.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage


@dataclass
class KMeansResult:
    """Résultat de l'algorithme K-Means."""
    model: KMeans
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_clusters: int


@dataclass
class CAHResult:
    """Résultat de la Classification Ascendante Hiérarchique."""
    model: AgglomerativeClustering
    linkage_matrix: np.ndarray
    labels: np.ndarray
    n_clusters: int
    method: str
    metric: str


def perform_kmeans(
    data: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 0
) -> KMeansResult:
    """
    Exécute l'algorithme K-Means sur les données (ou sur les composantes de l'ACP).

    Paramètres:
    -----------
    data : np.ndarray
        Matrice de données (généralement les coordonnées des 2 premières composantes principales).
    n_clusters : int, optionnel (défaut=2)
        Nombre de clusters souhaité.
    random_state : int, optionnel (défaut=0)
        Graine pour la reproductibilité de l'initialisation.

    Retourne:
    ---------
    KMeansResult
        Objet contenant le modèle, les étiquettes de cluster, les centroïdes et l'inertie intra-classe.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)

    return KMeansResult(
        model=kmeans,
        labels=labels,
        centroids=kmeans.cluster_centers_,
        inertia=float(kmeans.inertia_),
        n_clusters=n_clusters
    )


def perform_hierarchical(
    data: np.ndarray | pd.DataFrame,
    n_clusters: int = 2,
    method: str = "ward",
    metric: str = "euclidean"
) -> CAHResult:
    """
    Exécute une Classification Ascendante Hiérarchique (CAH).

    Paramètres:
    -----------
    data : np.ndarray ou pd.DataFrame
        Données numériques d'apprentissage.
    n_clusters : int, optionnel (défaut=2)
        Nombre de classes à extraire du dendrogramme.
    method : str, optionnel (défaut="ward")
        Critère d'agrégation ('ward', 'complete', 'average', 'single').
    metric : str, optionnel (défaut="euclidean")
        Métrique de distance. Pour 'ward', seule la distance euclidienne est admise.

    Retourne:
    ---------
    CAHResult
        Objet contenant la matrice des liens (linkage_matrix), le modèle et les labels.
    """
    # Matrice des liens pour le tracé du dendrogramme
    linkage_matrix = linkage(data, method=method, metric=metric)

    # Modèle Scikit-Learn pour la prédiction des partitions
    hca = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=method
    )
    labels = hca.fit_predict(data)

    return CAHResult(
        model=hca,
        linkage_matrix=linkage_matrix,
        labels=labels,
        n_clusters=n_clusters,
        method=method,
        metric=metric
    )
