"""
Module d'Analyse en Composantes Principales (ACP / PCA).

Ce module implémente les calculs statistiques fondamentaux de l'ACP normée :
valeurs propres, pourcentages d'inertie simple et cumulée, saturations factorielles
(loadings) et projection des observations dans l'espace factoriel.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    """
    Structure de données contenant l'ensemble des résultats d'une ACP.
    """
    model: PCA
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inertia_percentage: np.ndarray
    cumulative_inertia: np.ndarray
    inertia_table: pd.DataFrame
    loadings_table: pd.DataFrame
    components_coordinates: np.ndarray
    components_df: pd.DataFrame


def perform_pca(df: pd.DataFrame, n_components: Optional[int] = None) -> PCAResult:
    """
    Exécute une Analyse en Composantes Principales (ACP normée) sur un DataFrame.

    Paramètres:
    -----------
    df : pd.DataFrame
        Données numériques préalablement standardisées (centrées-réduites).
    n_components : int, optionnel (défaut=None)
        Nombre de composantes à retenir. Si None, toutes les composantes sont calculées.

    Retourne:
    ---------
    PCAResult
        Objet contenant le modèle ajusté, les valeurs propres, les tables d'inertie
        et les coordonnées projetées des individus.
    """
    if n_components is None:
        n_components = min(df.shape[0], df.shape[1])

    pca = PCA(n_components=n_components)
    components_coordinates = pca.fit_transform(df)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    inertia_percentage = pca.explained_variance_ratio_ * 100.0
    cumulative_inertia = np.cumsum(inertia_percentage)

    # Noms des composantes
    cp_names = [f"CP{i + 1}" for i in range(len(eigenvalues))]

    # Tableau récapitulatif des inerties
    inertia_table = pd.DataFrame(
        {
            "Valeur Propre (λ)": np.round(eigenvalues, 4),
            "% Inertie": np.round(inertia_percentage, 2),
            "% Inertie Cumulée": np.round(cumulative_inertia, 2),
        },
        index=cp_names
    )

    # Calcul des saturations / chargements factoriels (loadings = vecteur propre * sqrt(valeur propre))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_table = pd.DataFrame(
        np.round(loadings, 4),
        index=df.columns,
        columns=cp_names
    )

    # DataFrame des composantes pour chaque individu
    components_df = pd.DataFrame(
        components_coordinates,
        index=df.index,
        columns=cp_names
    )

    return PCAResult(
        model=pca,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        inertia_percentage=inertia_percentage,
        cumulative_inertia=cumulative_inertia,
        inertia_table=inertia_table,
        loadings_table=loadings_table,
        components_coordinates=components_coordinates,
        components_df=components_df
    )
