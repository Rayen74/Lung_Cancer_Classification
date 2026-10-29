"""
Module de prétraitement et de normalisation des données.

Ce module fournit des fonctions pour nettoyer, encoder et standardiser
les jeux de données avant l'application de méthodes d'analyse factorielle (ACP)
et de classification automatique (K-Means, CAH).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite un DataFrame en encodant les variables catégorielles courantes
    et en imputant les valeurs manquantes par la moyenne de chaque colonne.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame brut chargé depuis le fichier de données.

    Retourne:
    ---------
    pd.DataFrame
        Une nouvelle copie du DataFrame nettoyé et prêt pour l'analyse.
    """
    df_clean = df.copy()

    # Encodage de la variable 'GENDER' si elle existe
    if 'GENDER' in df_clean.columns:
        df_clean['GENDER'] = df_clean['GENDER'].astype(str).str.strip().str.upper()
        df_clean['GENDER'] = df_clean['GENDER'].replace({'M': 1, 'F': 2})
        # Forcer la conversion en numérique au cas où des valeurs inattendues subsisteraient
        df_clean['GENDER'] = pd.to_numeric(df_clean['GENDER'], errors='coerce')

    # Encodage des autres colonnes textuelles / binaires éventuelles
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        # Tenter une conversion numérique
        converted = pd.to_numeric(df_clean[col], errors='coerce')
        if converted.notna().sum() > 0:
            df_clean[col] = converted
        else:
            # Facteurs catégoriels : encodage par codes entiers (1-indexés)
            df_clean[col] = pd.factorize(df_clean[col])[0] + 1

    # Remplacement des valeurs manquantes par la moyenne de chaque variable
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())

    return df_clean


def check_and_normalize(df: pd.DataFrame, tolerance: float = 1e-4) -> Tuple[pd.DataFrame, bool, str]:
    """
    Vérifie si le DataFrame est déjà centré-réduit (moyenne ~ 0 et écart-type ~ 1).
    Si ce n'est pas le cas, applique une standardisation (Z-score).

    Paramètres:
    -----------
    df : pd.DataFrame
        Le jeu de données numérique à vérifier.
    tolerance : float, optionnel (défaut=1e-4)
        Marge d'erreur tolérée pour la moyenne (0) et l'écart-type (1).

    Retourne:
    ---------
    Tuple[pd.DataFrame, bool, str]:
        - DataFrame (normalisé si nécessaire).
        - Booléen indiquant si la base était déjà normalisée.
        - Message informatif détaillant l'état de la normalisation.
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe()

    means = stats.loc['mean']
    stds = stats.loc['std']

    is_already_normalized = bool(
        (means.abs() < tolerance).all() and ((stds - 1.0).abs() < tolerance).all()
    )

    if is_already_normalized:
        message = "La base de données est déjà normalisée (centrée-réduite)."
        return df.copy(), True, message

    # Effectuer la normalisation via StandardScaler
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(scaled_matrix, columns=numeric_df.columns, index=df.index)

    message = "La base de données a été normalisée avec succès (centrée-réduite : moyenne = 0, écart-type = 1)."
    return normalized_df, False, message


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Génère un résumé descriptif du jeu de données (dimensions, types, valeurs manquantes).

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame à analyser.

    Retourne:
    ---------
    Dict[str, Any]
        Dictionnaire contenant les métriques clés.
    """
    missing_by_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "total_missing": total_missing,
        "missing_by_column": missing_by_col,
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(exclude=[np.number]).columns)
    }
