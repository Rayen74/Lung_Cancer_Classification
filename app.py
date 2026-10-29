"""
Application Principale Streamlit - AFD Studio (Analyse Factorielle des Données & Clustering).

Interface moderne, réactive et ergonomique sans icônes superflues,
avec support complet et lisible des thèmes Sombre et Clair.
"""

import os
import io
import base64
from typing import Optional, Dict, Any, Callable
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessor import preprocess_data, check_and_normalize, get_dataset_summary
from src.pca_analysis import perform_pca, PCAResult
from src.clustering import perform_kmeans, perform_hierarchical
from src.plots import (
    plot_correlation_matrix,
    plot_scree,
    plot_correlation_circle,
    plot_kmeans_clusters,
    plot_dendrogram,
)

# ---------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="AFD Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_DATASET_PATH = "cancer_des_poumons.csv"


# ---------------------------------------------------------
# Initialisation de l'état de session (Session State)
# ---------------------------------------------------------
def init_session_state() -> None:
    """Initialise de façon idempotente les variables persistantes."""
    defaults = {
        "theme_mode": "dark",  # "dark" ou "light"
        "df_raw": None,
        "df_preprocessed": None,
        "df_normalized": None,
        "norm_message": None,
        "pca_res": None,
        "kmeans_res": None,
        "cah_res": None,
        "active_k": 2,
        "active_cah_k": 2,
        "active_cah_cutoff": 0.0,
        "pca_axis_x": 0,
        "pca_axis_y": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------
# Design Tokens & Styles CSS (Light & Dark Mode)
# ---------------------------------------------------------
THEME_TOKENS: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg_app": "#0b0f19",
        "bg_sidebar": "#070a12",
        "bg_card": "#111827",
        "bg_card_elevated": "#1e293b",
        "border_color": "#1f2937",
        "border_highlight": "#374151",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#94a3b8",
        "accent_primary": "#3b82f6",
        "accent_hover": "#2563eb",
        "accent_soft": "rgba(59, 130, 246, 0.15)",
        "badge_bg": "#1e293b",
        "badge_border": "#334155",
        "input_bg": "#111827",
        "popover_bg": "#111827",
        "card_shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.4)",
    },
    "light": {
        "bg_app": "#f8fafc",
        "bg_sidebar": "#f1f5f9",
        "bg_card": "#ffffff",
        "bg_card_elevated": "#f8fafc",
        "border_color": "#cbd5e1",
        "border_highlight": "#94a3b8",
        "text_primary": "#0f172a",
        "text_secondary": "#334155",
        "text_muted": "#475569",
        "accent_primary": "#2563eb",
        "accent_hover": "#1d4ed8",
        "accent_soft": "rgba(37, 99, 235, 0.1)",
        "badge_bg": "#e2e8f0",
        "badge_border": "#cbd5e1",
        "input_bg": "#ffffff",
        "popover_bg": "#ffffff",
        "card_shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
    }
}


def apply_theme_css(theme: str) -> None:
    """Injecte un design system complet garantissant un contraste parfait et la lisibilité des listes déroulantes."""
    tokens = THEME_TOKENS.get(theme, THEME_TOKENS["dark"])
    css = f"""
    <style>
    :root, body {{
        --afd-bg-app: {tokens['bg_app']};
        --afd-bg-sidebar: {tokens['bg_sidebar']};
        --afd-bg-card: {tokens['bg_card']};
        --afd-bg-card-elevated: {tokens['bg_card_elevated']};
        --afd-border: {tokens['border_color']};
        --afd-border-highlight: {tokens['border_highlight']};
        --afd-text-primary: {tokens['text_primary']};
        --afd-text-secondary: {tokens['text_secondary']};
        --afd-text-muted: {tokens['text_muted']};
        --afd-accent: {tokens['accent_primary']};
        --afd-accent-hover: {tokens['accent_hover']};
        --afd-accent-soft: {tokens['accent_soft']};
        --afd-badge-bg: {tokens['badge_bg']};
        --afd-badge-border: {tokens['badge_border']};
        --afd-input-bg: {tokens['input_bg']};
        --afd-popover-bg: {tokens['popover_bg']};
        --afd-shadow: {tokens['card_shadow']};
    }}

    /* Fond de page et texte global */
    .stApp {{
        background-color: var(--afd-bg-app) !important;
        color: var(--afd-text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Panneau latéral (Sidebar) */
    [data-testid="stSidebar"] {{
        background-color: var(--afd-bg-sidebar) !important;
        border-right: 1px solid var(--afd-border) !important;
    }}

    /* Titres et paragraphes */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--afd-text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}

    p, span, label, div {{
        color: var(--afd-text-primary);
    }}

    label[data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] > div,
    [data-testid="stWidgetLabel"] p {{
        color: var(--afd-text-primary) !important;
        font-weight: 500 !important;
    }}

    [data-testid="stMarkdownContainer"] p {{
        color: var(--afd-text-primary) !important;
    }}

    /* Selectboxes, Inputs et boîtes de saisie */
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {{
        background-color: var(--afd-input-bg) !important;
        border: 1px solid var(--afd-border) !important;
        color: var(--afd-text-primary) !important;
        border-radius: 6px !important;
    }}

    div[data-baseweb="select"] * {{
        color: var(--afd-text-primary) !important;
    }}

    /* Popover du menu déroulant (RÈGLE CRUCIALE POUR LA LISIBILITÉ) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    ul[role="listbox"],
    ul[data-baseweb="menu"] {{
        background-color: var(--afd-popover-bg) !important;
        border: 1px solid var(--afd-border) !important;
        color: var(--afd-text-primary) !important;
        border-radius: 6px !important;
    }}

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li[role="option"],
    li[role="option"],
    div[role="option"] {{
        background-color: var(--afd-popover-bg) !important;
        color: var(--afd-text-primary) !important;
        font-size: 0.925rem !important;
        padding: 8px 12px !important;
        cursor: pointer !important;
    }}

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[role="option"]:hover,
    li[role="option"]:hover,
    div[role="option"]:hover {{
        background-color: var(--afd-accent) !important;
        color: #ffffff !important;
    }}

    div[data-baseweb="popover"] li[aria-selected="true"],
    li[role="option"][aria-selected="true"] {{
        background-color: var(--afd-accent) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    /* Cartes */
    .afd-card {{
        background-color: var(--afd-bg-card);
        border: 1px solid var(--afd-border);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--afd-shadow);
    }}

    /* Tuiles de métriques */
    .afd-metric-card {{
        background-color: var(--afd-bg-card);
        border: 1px solid var(--afd-border);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: var(--afd-shadow);
    }}
    .afd-metric-val {{
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--afd-accent);
        line-height: 1.2;
    }}
    .afd-metric-lbl {{
        font-size: 0.825rem;
        font-weight: 500;
        color: var(--afd-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.25rem;
    }}

    /* Badges de statut */
    .afd-badge {{
        display: inline-flex;
        align-items: center;
        background-color: var(--afd-badge-bg);
        border: 1px solid var(--afd-badge-border);
        color: var(--afd-text-primary);
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
    }}

    /* Onglets de navigation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        border-bottom: 1px solid var(--afd-border);
        padding-bottom: 0.25rem;
        margin-bottom: 1.25rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: var(--afd-text-muted) !important;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.925rem;
        transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--afd-text-primary) !important;
        background-color: var(--afd-accent-soft);
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--afd-bg-card) !important;
        color: var(--afd-accent) !important;
        border-color: var(--afd-border) !important;
        font-weight: 600 !important;
        box-shadow: var(--afd-shadow);
    }}

    /* Boutons standards */
    .stButton > button {{
        background-color: var(--afd-accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.55rem 1.35rem !important;
        font-weight: 500 !important;
        font-size: 0.925rem !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        background-color: var(--afd-accent-hover) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
    }}

    /* Expanders */
    details[data-testid="stExpander"] {{
        background-color: var(--afd-bg-card) !important;
        border: 1px solid var(--afd-border) !important;
        border-radius: 8px !important;
        color: var(--afd-text-primary) !important;
    }}
    details[data-testid="stExpander"] summary {{
        color: var(--afd-text-primary) !important;
    }}

    /* Alertes */
    div[data-testid="stAlert"] {{
        background-color: var(--afd-bg-card) !important;
        border: 1px solid var(--afd-border) !important;
        color: var(--afd-text-primary) !important;
    }}

    /* Conteneur graphique */
    .afd-plot-container {{
        background-color: var(--afd-bg-card);
        border: 1px solid var(--afd-border);
        border-radius: 8px;
        padding: 0.75rem;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: var(--afd-shadow);
        margin: 0.5rem 0;
    }}
    .afd-plot-container img {{
        max-width: 100%;
        height: auto;
        border-radius: 6px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------
# Moteur de rendu graphique sécurisé (Sans erreur 404)
# ---------------------------------------------------------
def render_plot(fig: plt.Figure) -> None:
    """
    Rend une figure Matplotlib de façon sécurisée sous forme de Base64 inline.
    Élimine toute dépendance envers le gestionnaire de fichiers médias de Streamlit,
    garantissant l'absence totale d'erreurs 404.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=135,
        facecolor=fig.get_facecolor(),
        edgecolor="none"
    )
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    html = f"""
    <div class="afd-plot-container">
        <img src="data:image/png;base64,{b64}" alt="Graphique scientifique" />
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def safe_plot_call(plot_func: Callable[..., plt.Figure], *args: Any, is_dark: bool = False, **kwargs: Any) -> plt.Figure:
    """
    Exécute une fonction graphique de façon rétrocompatible.
    Transmet dark_mode si la fonction l'accepte, ou s'y replie automatiquement en cas d'incompatibilité.
    """
    try:
        return plot_func(*args, dark_mode=is_dark, **kwargs)
    except TypeError:
        return plot_func(*args, **kwargs)


# ---------------------------------------------------------
# Barre latérale (Sidebar) - Configuration & Thème
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Préférences")

    # Commutateur de Thème Clair / Sombre (Sans icônes)
    theme_options = ["Mode Sombre", "Mode Clair"]
    current_idx = 0 if st.session_state.theme_mode == "dark" else 1

    selected_theme = st.radio(
        "Thème visuel de l'interface :",
        options=theme_options,
        index=current_idx,
        key="theme_mode_radio"
    )
    is_dark = (selected_theme == "Mode Sombre")
    st.session_state.theme_mode = "dark" if is_dark else "light"

    st.markdown("---")
    st.markdown("### Source des données")

    data_source = st.radio(
        "Origine du fichier :",
        options=["Dataset local (Cancer des poumons)", "Téléverser un fichier CSV"],
        index=0,
        key="data_source_radio"
    )

    uploaded_file = None
    if data_source == "Téléverser un fichier CSV":
        uploaded_file = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")
    else:
        if os.path.exists(DEFAULT_DATASET_PATH):
            st.markdown(
                f'<div class="afd-badge">Fichier local détecté : <code>{DEFAULT_DATASET_PATH}</code></div>',
                unsafe_allow_html=True
            )
        else:
            st.error(f"Fichier `{DEFAULT_DATASET_PATH}` introuvable.")

    st.markdown("---")
    st.markdown("### Statut du pipeline")

    s_loaded = "[ X ]" if st.session_state.df_raw is not None else "[   ]"
    s_prep = "[ X ]" if st.session_state.df_normalized is not None else "[   ]"
    s_pca = "[ X ]" if st.session_state.pca_res is not None else "[   ]"
    s_clust = "[ X ]" if st.session_state.kmeans_res is not None else "[   ]"

    pipeline_status_html = f"""
    <div style="font-size: 0.85rem; line-height: 1.8; color: var(--afd-text-secondary);">
        <div><code>{s_loaded}</code> Données chargées</div>
        <div><code>{s_prep}</code> Prétraitement et Normalisation</div>
        <div><code>{s_pca}</code> Analyse en Composantes (ACP)</div>
        <div><code>{s_clust}</code> Classification (K-Means et CAH)</div>
    </div>
    """
    st.markdown(pipeline_status_html, unsafe_allow_html=True)


# Appliquer le thème CSS dynamiquement
apply_theme_css(st.session_state.theme_mode)

# ---------------------------------------------------------
# Chargement du jeu de données
# ---------------------------------------------------------
df_loaded: Optional[pd.DataFrame] = None

if data_source == "Téléverser un fichier CSV" and uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
    except Exception as err:
        st.error(f"Erreur lors de la lecture du fichier : {err}")
elif data_source == "Dataset local (Cancer des poumons)" and os.path.exists(DEFAULT_DATASET_PATH):
    try:
        df_loaded = pd.read_csv(DEFAULT_DATASET_PATH)
    except Exception as err:
        st.error(f"Erreur de lecture du dataset local : {err}")

# Synchronisation dans session_state si nouveau fichier
if df_loaded is not None:
    if st.session_state.df_raw is None or not df_loaded.equals(st.session_state.df_raw):
        st.session_state.df_raw = df_loaded
        st.session_state.df_preprocessed = None
        st.session_state.df_normalized = None
        st.session_state.pca_res = None
        st.session_state.kmeans_res = None
        st.session_state.cah_res = None

# ---------------------------------------------------------
# En-tête principal de l'application
# ---------------------------------------------------------
col_h_left, col_h_right = st.columns([4, 1.2])
with col_h_left:
    st.markdown(
        """
        <div style="margin-bottom: 0.5rem;">
            <h1 style="margin: 0; font-size: 2.1rem;">
                AFD Studio
            </h1>
            <p style="margin: 0.35rem 0 0 0; color: var(--afd-text-muted); font-size: 0.95rem;">
                Analyse Factorielle des Données et Classification Non Supervisée
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_h_right:
    status_label = "Données Normalisées" if st.session_state.df_normalized is not None else "En attente"
    st.markdown(
        f"""
        <div style="text-align: right; margin-top: 0.5rem;">
            <div class="afd-badge">
                Statut : {status_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.session_state.df_raw is None:
    st.info("Veuillez sélectionner ou charger un fichier de données dans le panneau latéral pour commencer.")
    st.stop()


# ---------------------------------------------------------
# Navigation par Onglets (Tabs)
# ---------------------------------------------------------
tab_prep, tab_corr, tab_pca, tab_clust = st.tabs([
    "1. Données et Prétraitement",
    "2. Matrice de Corrélation",
    "3. Analyse Factorielle (ACP)",
    "4. Classification et Clustering"
])


# =========================================================
# Onglet 1: Données et Prétraitement
# =========================================================
with tab_prep:
    summary_raw = get_dataset_summary(st.session_state.df_raw)

    # Cartes de métriques
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="afd-metric-card">
                <div class="afd-metric-val">{summary_raw['n_rows']}</div>
                <div class="afd-metric-lbl">Observations (Lignes)</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="afd-metric-card">
                <div class="afd-metric-val">{summary_raw['n_cols']}</div>
                <div class="afd-metric-lbl">Variables (Colonnes)</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""<div class="afd-metric-card">
                <div class="afd-metric-val">{len(summary_raw['numeric_columns'])}</div>
                <div class="afd-metric-lbl">Variables Numériques</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c4:
        val_color = "var(--afd-accent)" if summary_raw['total_missing'] == 0 else "#f59e0b"
        st.markdown(
            f"""<div class="afd-metric-card">
                <div class="afd-metric-val" style="color: {val_color};">{summary_raw['total_missing']}</div>
                <div class="afd-metric-lbl">Valeurs Manquantes</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Section données brutes
    st.markdown("### Aperçu des données brutes")
    st.dataframe(st.session_state.df_raw.head(8), width="stretch")

    # Signalement des valeurs manquantes si présentes
    missing_series = pd.Series(summary_raw["missing_by_column"])
    missing_cols = missing_series[missing_series > 0]
    if not missing_cols.empty:
        with st.expander(f"Détail des variables contenant des valeurs manquantes ({len(missing_cols)})"):
            st.dataframe(pd.DataFrame({"Nombre de NaN": missing_cols}), width="stretch")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Bouton d'action de prétraitement
    col_act, col_info = st.columns([1.5, 3])
    with col_act:
        btn_process = st.button("Prétraiter et Normaliser les données", width="stretch")
        if btn_process:
            with st.spinner("Encodage, imputation et normalisation centrée-réduite..."):
                # Encodage & imputation
                df_prep = preprocess_data(st.session_state.df_raw)
                st.session_state.df_preprocessed = df_prep

                # Normalisation Z-score
                df_norm, was_norm, norm_msg = check_and_normalize(df_prep)
                st.session_state.df_normalized = df_norm
                st.session_state.norm_message = norm_msg

                # ACP automatique
                st.session_state.pca_res = perform_pca(df_norm)
                st.rerun()

    with col_info:
        if st.session_state.df_normalized is not None:
            st.markdown(
                f"""
                <div class="afd-card" style="padding: 0.75rem 1rem; margin: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 0.9rem; color: var(--afd-text-secondary);">{st.session_state.norm_message}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Affichage des données après normalisation
    if st.session_state.df_normalized is not None:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Données Normalisées (Centrées-Réduites)")
        st.dataframe(st.session_state.df_normalized.head(8), width="stretch")

        with st.expander("Contrôle statistique après normalisation (Moyenne ≈ 0, Écart-Type ≈ 1)"):
            desc = st.session_state.df_normalized.describe().T[["mean", "std", "min", "max"]]
            desc["mean"] = desc["mean"].apply(lambda v: f"{v:.2e}")
            desc["std"] = desc["std"].apply(lambda v: f"{v:.4f}")
            st.dataframe(desc, width="stretch")


# =========================================================
# Onglet 2: Matrice de Corrélation
# =========================================================
with tab_corr:
    if st.session_state.df_normalized is None:
        st.warning("Veuillez d'abord exécuter le prétraitement dans le premier onglet.")
    else:
        st.markdown("### Matrice des Corrélations Linéaires")
        st.markdown(
            "<p style='color: var(--afd-text-muted); font-size: 0.925rem;'>"
            "Coefficients de Pearson mesurant l'intensité de la liaison linéaire entre chaque couple de variables."
            "</p>",
            unsafe_allow_html=True
        )

        fig_corr = safe_plot_call(plot_correlation_matrix, st.session_state.df_normalized, is_dark=is_dark)
        render_plot(fig_corr)

        with st.expander("Afficher le tableau complet des coefficients de Pearson"):
            st.dataframe(st.session_state.df_normalized.corr().round(3), width="stretch")


# =========================================================
# Onglet 3: Analyse en Composantes Principales (ACP)
# =========================================================
with tab_pca:
    if st.session_state.df_normalized is None or st.session_state.pca_res is None:
        st.warning("Veuillez d'abord exécuter le prétraitement dans le premier onglet.")
    else:
        pca_res: PCAResult = st.session_state.pca_res

        # Métriques de variance
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""<div class="afd-metric-card">
                    <div class="afd-metric-val">{pca_res.inertia_percentage[0]:.2f}%</div>
                    <div class="afd-metric-lbl">Inertie Composante 1 (CP1)</div>
                </div>""",
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""<div class="afd-metric-card">
                    <div class="afd-metric-val">{pca_res.inertia_percentage[1]:.2f}%</div>
                    <div class="afd-metric-lbl">Inertie Composante 2 (CP2)</div>
                </div>""",
                unsafe_allow_html=True
            )
        with m3:
            total_inertia_12 = pca_res.inertia_percentage[0] + pca_res.inertia_percentage[1]
            st.markdown(
                f"""<div class="afd-metric-card">
                    <div class="afd-metric-val" style="color: #10b981;">{total_inertia_12:.2f}%</div>
                    <div class="afd-metric-lbl">Inertie Cumulée (Plan 1-2)</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        # Graphiques côte à côte
        col_scree, col_circle = st.columns(2)

        with col_scree:
            st.markdown("#### 1. Éboulis des Valeurs Propres")
            st.markdown(
                "<p style='color: var(--afd-text-muted); font-size: 0.85rem; margin-bottom: 0.5rem;'>"
                "Décroissance de l'inertie factorielle et repère du critère de Kaiser (λ = 1)."
                "</p>",
                unsafe_allow_html=True
            )
            fig_scree = safe_plot_call(plot_scree, pca_res.eigenvalues, pca_res.inertia_percentage, is_dark=is_dark)
            render_plot(fig_scree)

        with col_circle:
            st.markdown("#### 2. Cercle des Corrélations")
            max_cp = len(pca_res.eigenvalues)
            col_ax1, col_ax2 = st.columns(2)
            with col_ax1:
                axis_x = st.selectbox("Axe horizontal :", options=range(1, max_cp + 1), index=st.session_state.pca_axis_x) - 1
                st.session_state.pca_axis_x = axis_x
            with col_ax2:
                axis_y = st.selectbox("Axe vertical :", options=range(1, max_cp + 1), index=st.session_state.pca_axis_y) - 1
                st.session_state.pca_axis_y = axis_y

            fig_circle = safe_plot_call(
                plot_correlation_circle,
                pca_res.eigenvectors,
                pca_res.eigenvalues,
                list(st.session_state.df_normalized.columns),
                pc_x=axis_x,
                pc_y=axis_y,
                is_dark=is_dark
            )
            render_plot(fig_circle)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Tableaux factoriels
        col_tab1, col_tab2 = st.columns(2)
        with col_tab1:
            st.markdown("#### Tableau des Valeurs Propres et Inertie")
            st.dataframe(pca_res.inertia_table, width="stretch")

        with col_tab2:
            st.markdown("#### Chargements Factoriels (Saturations)")
            st.dataframe(pca_res.loadings_table, width="stretch")


# =========================================================
# Onglet 4: Classification (Clustering)
# =========================================================
with tab_clust:
    if st.session_state.df_normalized is None or st.session_state.pca_res is None:
        st.warning("Veuillez d'abord exécuter le prétraitement et l'ACP.")
    else:
        pca_coords = st.session_state.pca_res.components_coordinates

        col_km, col_cah = st.columns(2)

        # -----------------------------------------------------
        # Bloc K-Means
        # -----------------------------------------------------
        with col_km:
            st.markdown("### A. Partitionnement K-Moyennes (K-Means)")
            st.markdown(
                "<p style='color: var(--afd-text-muted); font-size: 0.85rem;'>"
                "Classification itérative appliquée sur les axes factoriels CP1 et CP2."
                "</p>",
                unsafe_allow_html=True
            )

            k_val = st.slider(
                "Nombre de clusters (K) :",
                min_value=2,
                max_value=8,
                value=st.session_state.active_k,
                key="slider_k_means"
            )
            st.session_state.active_k = k_val

            km_res = perform_kmeans(pca_coords[:, :2], n_clusters=k_val)
            st.session_state.kmeans_res = km_res

            fig_km = safe_plot_call(
                plot_kmeans_clusters,
                pca_coords,
                km_res.labels,
                km_res.centroids,
                pc_x=0,
                pc_y=1,
                is_dark=is_dark
            )
            render_plot(fig_km)

            # Effectifs K-Means
            counts_km = pd.Series(km_res.labels).value_counts().sort_index()
            st.markdown("**Distribution des observations (K-Means) :**")
            st.dataframe(
                pd.DataFrame({
                    "Cluster": [f"Cluster {i}" for i in counts_km.index],
                    "Observations": counts_km.values,
                    "Proportion (%)": np.round((counts_km.values / len(km_res.labels)) * 100, 1)
                }),
                width="stretch"
            )

        # -----------------------------------------------------
        # Bloc CAH
        # -----------------------------------------------------
        with col_cah:
            st.markdown("### B. Classification Ascendante Hiérarchique (CAH)")
            st.markdown(
                "<p style='color: var(--afd-text-muted); font-size: 0.85rem;'>"
                "Agrégation selon la méthode de Ward et dendrogramme d'inertie."
                "</p>",
                unsafe_allow_html=True
            )

            col_ck, col_cut = st.columns(2)
            with col_ck:
                cah_k_val = st.slider(
                    "Nombre de classes :",
                    min_value=2,
                    max_value=8,
                    value=st.session_state.active_cah_k,
                    key="slider_cah_k"
                )
                st.session_state.active_cah_k = cah_k_val
            with col_cut:
                cutoff_input = st.number_input(
                    "Hauteur de coupe visuelle :",
                    min_value=0.0,
                    value=float(st.session_state.active_cah_cutoff),
                    step=1.0,
                    key="input_cah_cutoff"
                )
                st.session_state.active_cah_cutoff = cutoff_input

            cutoff_val = cutoff_input if cutoff_input > 0 else None
            cah_res = perform_hierarchical(st.session_state.df_normalized, n_clusters=cah_k_val, method="ward")
            st.session_state.cah_res = cah_res

            fig_dendro = safe_plot_call(plot_dendrogram, cah_res.linkage_matrix, max_d=cutoff_val, is_dark=is_dark)
            render_plot(fig_dendro)

            # Effectifs CAH
            counts_cah = pd.Series(cah_res.labels).value_counts().sort_index()
            st.markdown("**Distribution des observations (CAH) :**")
            st.dataframe(
                pd.DataFrame({
                    "Classe CAH": [f"Classe {i}" for i in counts_cah.index],
                    "Observations": counts_cah.values,
                    "Proportion (%)": np.round((counts_cah.values / len(cah_res.labels)) * 100, 1)
                }),
                width="stretch"
            )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # Section d'exportation
        # -----------------------------------------------------
        st.markdown("### Exportation des Résultats")
        export_df = st.session_state.df_raw.copy()
        export_df["Cluster_KMeans"] = km_res.labels
        export_df["Classe_CAH"] = cah_res.labels

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

        col_exp_left, col_exp_right = st.columns([3, 1.2])
        with col_exp_left:
            st.markdown(
                "<p style='color: var(--afd-text-muted); font-size: 0.9rem; margin-top: 0.4rem;'>"
                "Téléchargez la base de données brute enrichie avec les assignations des classes K-Means et CAH."
                "</p>",
                unsafe_allow_html=True
            )
        with col_exp_right:
            st.download_button(
                label="Télécharger le fichier CSV",
                data=csv_bytes,
                file_name="dataset_analyse_clusters.csv",
                mime="text/csv",
                width="stretch"
            )
