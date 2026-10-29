"""
Module de visualisation graphique scientifique avec support des thèmes Clair et Sombre.

Fournit des figures Matplotlib et Seaborn adaptées pour Streamlit :
- Matrice de corrélation (Heatmap)
- Éboulis des valeurs propres (Scree plot) avec seuil de Kaiser
- Cercle des corrélations factorielles
- Projection 2D des individus et centroïdes K-Means
- Dendrogramme de la Classification Ascendante Hiérarchique (CAH)
"""

from typing import List, Optional, Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def _get_theme_colors(dark_mode: bool = False) -> Dict[str, Any]:
    """Retourne la palette de couleurs en fonction du mode choisi."""
    if dark_mode:
        return {
            "fig_bg": "#111827",       # Surface de carte sombre (slate-900)
            "ax_bg": "#1f2937",        # Surface intérieure des axes (slate-800)
            "text": "#f8fafc",         # Texte principal clair (slate-50)
            "text_muted": "#94a3b8",   # Texte secondaire (slate-400)
            "grid": "#374151",         # Grille discrète (slate-700)
            "spine": "#4b5563",        # Bordures des axes (slate-600)
            "bar": "#3b82f6",          # Barres valeurs propres (bleu vif)
            "line": "#f43f5e",         # Ligne d'inertie (rose/rouge vif)
            "kaiser": "#10b981",       # Seuil de Kaiser (émeraude)
            "circle": "#38bdf8",       # Cercle de corrélation (cyan)
            "arrow": "#60a5fa",        # Flèches vectorielles
            "arrow_text": "#ffffff",   # Libellés des variables
            "palette": "bright",
        }
    else:
        return {
            "fig_bg": "#ffffff",       # Surface de carte blanche
            "ax_bg": "#f8fafc",        # Surface intérieure très douce (slate-50)
            "text": "#0f172a",         # Texte principal foncé (slate-900)
            "text_muted": "#475569",   # Texte secondaire (slate-600)
            "grid": "#e2e8f0",         # Grille claire (slate-200)
            "spine": "#cbd5e1",        # Bordures des axes (slate-300)
            "bar": "#2563eb",          # Barres valeurs propres (bleu royal)
            "line": "#e11d48",         # Ligne d'inertie (rouge rubis)
            "kaiser": "#059669",       # Seuil de Kaiser (vert forêt)
            "circle": "#0284c7",       # Cercle de corrélation (bleu ciel)
            "arrow": "#1d4ed8",        # Flèches vectorielles
            "arrow_text": "#0f172a",   # Libellés des variables
            "palette": "tab10",
        }


def _apply_theme(fig: plt.Figure, ax: plt.Axes, dark_mode: bool = False) -> None:
    """Applique les styles de thème à la figure et aux axes."""
    c = _get_theme_colors(dark_mode)
    fig.patch.set_facecolor(c["fig_bg"])
    ax.set_facecolor(c["ax_bg"])

    ax.tick_params(colors=c["text_muted"], which="both", labelsize=9)
    ax.xaxis.label.set_color(c["text"])
    ax.yaxis.label.set_color(c["text"])
    ax.title.set_color(c["text"])

    for spine in ax.spines.values():
        spine.set_color(c["spine"])
        spine.set_linewidth(0.8)


def plot_correlation_matrix(df: pd.DataFrame, dark_mode: bool = False, *args: Any, **kwargs: Any) -> plt.Figure:
    """
    Génère la matrice de corrélation sous forme de heatmap annotée.
    """
    c = _get_theme_colors(dark_mode)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=c["fig_bg"])
    ax.set_facecolor(c["ax_bg"])

    cmap = "vlag" if not dark_mode else sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor=c["fig_bg"],
        annot_kws={"size": 8.5},  # Seaborn ajuste automatiquement le contraste noir/blanc des nombres
        cbar_kws={"shrink": 0.82, "label": "Coefficient de Pearson (r)"},
        ax=ax
    )

    ax.set_title("Matrice des Corrélations Linéaires de Pearson", fontsize=12.5, fontweight="bold", pad=14, color=c["text"])
    _apply_theme(fig, ax, dark_mode)

    # Style de la colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=c["text_muted"])
    cbar.set_label("Coefficient de Pearson (r)", color=c["text"], fontsize=9.5, fontweight="normal")
    for t in cbar.ax.get_yticklabels():
        t.set_color(c["text_muted"])
        t.set_fontsize(8.5)

    plt.tight_layout()
    return fig


def plot_scree(
    eigenvalues: np.ndarray,
    inertia_percentage: np.ndarray,
    dark_mode: bool = False,
    *args: Any,
    **kwargs: Any
) -> plt.Figure:
    """
    Génère le diagramme d'éboulis des valeurs propres (Scree plot).
    """
    c = _get_theme_colors(dark_mode)
    n_cp = len(eigenvalues)
    x = range(1, n_cp + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5.2), facecolor=c["fig_bg"])
    _apply_theme(fig, ax1, dark_mode)

    # Barres pour les valeurs propres
    ax1.bar(x, eigenvalues, alpha=0.75, color=c["bar"], width=0.58, label="Valeur propre (λ)")
    ax1.set_xlabel("Composantes Principales", fontsize=10.5, fontweight="bold", color=c["text"], labelpad=8)
    ax1.set_ylabel("Valeur propre (λ)", color=c["bar"], fontsize=10.5, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=c["bar"])
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"CP{i}" for i in x], color=c["text_muted"], fontsize=8.5)

    # Ligne horizontale du critère de Kaiser (λ = 1)
    ax1.axhline(1.0, color=c["kaiser"], linestyle="--", linewidth=1.5, label="Seuil de Kaiser (λ = 1)")

    # Deuxième axe pour le pourcentage d'inertie
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    ax2.plot(x, inertia_percentage, color=c["line"], marker="o", markersize=5.5, linewidth=2.0, label="% Inertie")
    ax2.set_ylabel("% Inertie expliquée", color=c["line"], fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=c["line"])

    for spine in ax2.spines.values():
        spine.set_color(c["spine"])

    ax1.set_title("Éboulis des Valeurs Propres & % d'Inertie", fontsize=12.5, fontweight="bold", pad=14, color=c["text"])
    ax1.grid(True, linestyle=":", alpha=0.35, color=c["grid"])

    # Légende unifiée
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(h1 + h2, l1 + l2, loc="upper right", facecolor=c["ax_bg"], edgecolor=c["spine"], fontsize=9)
    for text in leg.get_texts():
        text.set_color(c["text"])

    plt.tight_layout()
    return fig


def plot_correlation_circle(
    pca_components: np.ndarray,
    eigenvalues: np.ndarray,
    feature_names: List[str],
    pc_x: int = 0,
    pc_y: int = 1,
    dark_mode: bool = False,
    *args: Any,
    **kwargs: Any
) -> plt.Figure:
    """
    Trace le cercle des corrélations factorielles des variables.
    """
    c = _get_theme_colors(dark_mode)
    fig, ax = plt.subplots(figsize=(7.8, 7.8), facecolor=c["fig_bg"])
    _apply_theme(fig, ax, dark_mode)

    # Axes orthogonaux
    ax.axhline(0, color=c["spine"], linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(0, color=c["spine"], linestyle="--", linewidth=0.9, alpha=0.7)

    # Cercle unitaire
    circle = plt.Circle((0, 0), 1, color=c["circle"], fill=False, linewidth=1.8, linestyle="-")
    ax.add_artist(circle)

    scale_x = np.sqrt(eigenvalues[pc_x])
    scale_y = np.sqrt(eigenvalues[pc_y])

    for i, name in enumerate(feature_names):
        x = pca_components[pc_x, i] * scale_x
        y = pca_components[pc_y, i] * scale_y

        ax.arrow(
            0, 0, x, y,
            head_width=0.035,
            head_length=0.045,
            color=c["arrow"],
            length_includes_head=True,
            alpha=0.9
        )
        offset_x = 0.035 if x >= 0 else -0.075
        offset_y = 0.035 if y >= 0 else -0.075
        ax.text(x + offset_x, y + offset_y, name, fontsize=8.5, color=c["arrow_text"], fontweight="normal")

    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Composante Principale {pc_x + 1}", fontsize=10.5, fontweight="bold", color=c["text"], labelpad=6)
    ax.set_ylabel(f"Composante Principale {pc_y + 1}", fontsize=10.5, fontweight="bold", color=c["text"], labelpad=6)
    ax.set_title(f"Cercle des Corrélations (CP{pc_x + 1} vs CP{pc_y + 1})", fontsize=12.5, fontweight="bold", pad=14, color=c["text"])
    ax.grid(True, linestyle=":", alpha=0.35, color=c["grid"])

    plt.tight_layout()
    return fig


def plot_kmeans_clusters(
    components: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    pc_x: int = 0,
    pc_y: int = 1,
    dark_mode: bool = False,
    *args: Any,
    **kwargs: Any
) -> plt.Figure:
    """
    Trace la projection des observations et des centroïdes sur le plan factoriel ACP.
    """
    c = _get_theme_colors(dark_mode)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), facecolor=c["fig_bg"])
    _apply_theme(fig, ax, dark_mode)

    unique_labels = np.unique(labels)
    palette = sns.color_palette("bright" if dark_mode else "tab10", len(unique_labels))

    # Points des observations
    ax.scatter(
        components[:, pc_x],
        components[:, pc_y],
        c=[palette[label] for label in labels],
        alpha=0.82,
        edgecolor=c["fig_bg"],
        linewidth=0.6,
        s=62,
        label="Observations"
    )

    # Tracé des centroïdes
    ax.scatter(
        centroids[:, pc_x],
        centroids[:, pc_y],
        c="#f43f5e",
        marker="X",
        s=240,
        edgecolor="white" if dark_mode else "black",
        linewidth=1.8,
        label="Centroïdes"
    )

    ax.axhline(0, color=c["spine"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color=c["spine"], linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(f"Composante {pc_x + 1}", fontsize=10.5, fontweight="bold", color=c["text"], labelpad=6)
    ax.set_ylabel(f"Composante {pc_y + 1}", fontsize=10.5, fontweight="bold", color=c["text"], labelpad=6)
    ax.set_title(f"Partitionnement K-Means sur le plan factoriel (CP{pc_x + 1} / CP{pc_y + 1})", fontsize=12.5, fontweight="bold", pad=14, color=c["text"])
    ax.grid(True, linestyle=":", alpha=0.35, color=c["grid"])

    leg = ax.legend(loc="best", facecolor=c["ax_bg"], edgecolor=c["spine"], fontsize=9)
    for text in leg.get_texts():
        text.set_color(c["text"])

    plt.tight_layout()
    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    max_d: Optional[float] = None,
    p: int = 30,
    dark_mode: bool = False,
    *args: Any,
    **kwargs: Any
) -> plt.Figure:
    """
    Génère le dendrogramme de la Classification Ascendante Hiérarchique (CAH).
    """
    c = _get_theme_colors(dark_mode)
    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor=c["fig_bg"])
    _apply_theme(fig, ax, dark_mode)

    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=p,
        leaf_rotation=90.0,
        leaf_font_size=8.5,
        show_contracted=True,
        ax=ax
    )

    if max_d is not None:
        ax.axhline(y=max_d, c="#f43f5e", linestyle="--", linewidth=1.8, label=f"Ligne de coupure (d={max_d:.2f})")
        leg = ax.legend(loc="upper right", facecolor=c["ax_bg"], edgecolor=c["spine"], fontsize=9)
        for text in leg.get_texts():
            text.set_color(c["text"])

    ax.set_title("Dendrogramme de la CAH (Méthode de Ward)", fontsize=12.5, fontweight="bold", pad=14, color=c["text"])
    ax.set_xlabel("Observations (ou grappes condensées)", fontsize=10.5, fontweight="bold", color=c["text"], labelpad=6)
    ax.set_ylabel("Distances d'agrégation", fontsize=10.5, fontweight="bold", color=c["text"], labelpad=6)
    ax.grid(True, linestyle=":", alpha=0.35, color=c["grid"])

    plt.tight_layout()
    return fig
