"""
Application Streamlit pour la détection de fraude bancaire.
Interface utilisateur interactive permettant la prédiction de transactions frauduleuses.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ajouter le répertoire racine au path pour les imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.xgboost_model import FraudDetectionModel
from src.utils.config import FEATURE_COLUMNS, MODEL_PATH, TEST_DATA_PATH


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Système Expert de Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown(
    """
    <style>
    .fraud-alert {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_fraud_model(model_path: str):
    """
    Charge le modèle de détection de fraude depuis un fichier pickle.

    Args:
        model_path: Chemin vers le fichier pickle du modèle.

    Returns:
        Instance du modèle FraudDetectionModel chargée.
    """
    try:
        model = FraudDetectionModel.load(model_path)
        st.sidebar.success("✅ Modèle chargé avec succès")
        return model
    except FileNotFoundError as e:
        st.error(f"❌ Fichier modèle introuvable : {e}")
        raise
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        raise


@st.cache_data
def load_test_data(data_path: str):
    """
    Charge les données de test depuis un fichier parquet.

    Args:
        data_path: Chemin vers le fichier parquet des données de test.

    Returns:
        DataFrame pandas contenant les données de test.
    """
    try:
        df = pd.read_parquet(data_path)
        return df
    except FileNotFoundError as e:
        st.error(f"❌ Fichier de données introuvable : {e}")
        raise
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {e}")
        raise


def predict_transaction(model, features: pd.DataFrame) -> tuple:
    """
    Effectue une prédiction sur une transaction.

    Args:
        model: Instance du modèle de détection de fraude.
        features: DataFrame contenant les features de la transaction.

    Returns:
        Tuple contenant la probabilité de fraude et la classe prédite.
    """
    try:
        result = model.predict(features)
        proba = result.iloc[0]["probability"]
        prediction = int(result.iloc[0]["prediction"])
        return proba, prediction
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        raise


def display_prediction_result(probability: float, prediction: int):
    """
    Affiche le résultat de la prédiction avec un formatage visuel.

    Args:
        probability: Probabilité de fraude (entre 0 et 1).
        prediction: Classe prédite (0 = normal, 1 = fraude).
    """
    percentage = probability * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Probabilité de fraude",
            value=f"{percentage:.2f}%",
            delta=f"{'Élevée' if percentage > 50 else 'Faible'}",
        )

    with col2:
        if prediction == 1:
            st.markdown(
                '<div class="fraud-alert" style="background-color: #ffebee; color: #c62828;">'
                "⚠️ FRAUDE DÉTECTÉE"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="fraud-alert" style="background-color: #e8f5e9; color: #2e7d32;">'
                "✅ TRANSACTION NORMALE"
                "</div>",
                unsafe_allow_html=True,
            )

    # Barre de progression
    st.progress(min(percentage / 100, 1.0))
    st.caption(f"Seuil de détection : 50%")


def create_input_form(feature_names: list) -> dict:
    """
    Crée le formulaire de saisie des features dans la barre latérale.

    Args:
        feature_names: Liste des noms des features.

    Returns:
        Dictionnaire contenant les valeurs saisies.
    """
    st.sidebar.header("📊 Caractéristiques de la transaction")

    # Organiser les features en accordéons
    input_values = {}

    # Time et Amount dans un groupe séparé
    st.sidebar.subheader("Informations générales")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        input_values["Time"] = st.number_input(
            "Time",
            value=0.0,
            step=1.0,
            format="%.0f",
            key="Time",
            help="Délai en secondes depuis la première transaction",
        )
    with col2:
        input_values["Amount"] = st.number_input(
            "Amount",
            value=0.0,
            step=0.01,
            format="%.2f",
            key="Amount",
            help="Montant de la transaction",
        )

    # Features V1 à V28 dans des accordéons pour économiser l'espace
    st.sidebar.subheader("Variables PCA (V1-V28)")

    # Répartir les features V en groupes de 7
    v_features = [f for f in feature_names if f.startswith("V")]
    num_groups = 4
    v_per_group = len(v_features) // num_groups

    for i in range(num_groups):
        start_idx = i * v_per_group
        end_idx = start_idx + v_per_group
        group_features = v_features[start_idx:end_idx]

        with st.sidebar.expander(
            f"V{start_idx+1} à V{end_idx}", expanded=(i == 0)
        ):
            cols = st.columns(4)
            for j, feat in enumerate(group_features):
                col_idx = j % 4
                with cols[col_idx]:
                    input_values[feat] = st.number_input(
                        feat,
                        value=0.0,
                        step=0.001,
                        format="%.3f",
                        key=feat,
                    )

    return input_values


def main():
    """
    Fonction principale de l'application Streamlit.
    """
    # Titre et description
    st.title("🛡️ Système Expert de Détection de Fraude")
    st.markdown(
        """
        Interface interactive de détection de fraude bancaire.
        Renseignez les caractéristiques d'une transaction ou chargez un exemple
        pour obtenir une prédiction en temps réel.
        """
    )

    # Barre latérale - Chargement des données
    st.sidebar.header("📦 Chargement des ressources")

    # Chemins des fichiers
    model_path = Path(MODEL_PATH)
    test_data_path = Path(TEST_DATA_PATH)

    # Chargement du modèle
    model = None
    if model_path.exists():
        model = load_fraud_model(str(model_path))
    else:
        st.error(f"❌ Modèle non trouvé à : {model_path}")
        st.stop()

    # Chargement des données de test
    test_data = None
    if test_data_path.exists():
        test_data = load_test_data(str(test_data_path))
        if test_data is not None:
            st.sidebar.success(
                f"✅ Données chargées : {len(test_data)} transactions"
            )
    else:
        st.sidebar.warning(
            f"⚠️ Données de test non trouvées à : {test_data_path}"
        )

    # Features disponibles
    feature_names = FEATURE_COLUMNS

    # Bouton pour charger un exemple aléatoire
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Exemple aléatoire")

    if st.sidebar.button(
        "Charger un exemple aléatoire",
        type="primary",
        use_container_width=True,
    ):
        if test_data is not None:
            random_idx = np.random.randint(0, len(test_data))
            random_transaction = test_data.iloc[random_idx]

            # Mettre à jour les valeurs dans la session
            for feat in feature_names:
                if feat in random_transaction.index:
                    st.session_state[feat] = float(random_transaction[feat])

            st.sidebar.success(
                f"✅ Transaction n°{random_idx} chargée (réel: "
                f"{test_data.iloc[random_idx].name})"
            )
            st.rerun()
        else:
            st.sidebar.error(
                "❌ Données de test non disponibles"
            )

    # Création du formulaire de saisie
    input_values = create_input_form(feature_names)

    # Zone de prédiction
    st.markdown("---")
    st.header("🔍 Résultat de la prédiction")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        predict_button = st.button(
            "🔮 Prédire", type="primary", use_container_width=True
        )

    if predict_button:
        if model is not None:
            # Créer le DataFrame d'entrée
            input_df = pd.DataFrame(
                [[input_values[feat] for feat in feature_names]],
                columns=feature_names,
            )

            # Afficher les valeurs saisies
            with st.expander("📋 Valeurs saisies", expanded=False):
                st.dataframe(input_df, use_container_width=True)

            # Effectuer la prédiction
            probability, prediction = predict_transaction(model, input_df)

            # Afficher le résultat
            display_prediction_result(probability, prediction)

            # Informations supplémentaires
            with st.expander("📊 Détails de la prédiction", expanded=False):
                st.write(f"Probabilité brute : {probability:.6f}")
                st.write(f"Seuil de décision : 0.5 (50%)")
                st.write(
                    f"Décision : {'Fraude' if prediction == 1 else 'Normale'}"
                )
        else:
            st.error("❌ Modèle non chargé. Impossible de prédire.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>
        Système Expert de Détection de Fraude | Modèle XGBoost |
        Les prédictions sont fournies à titre indicatif
        </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()