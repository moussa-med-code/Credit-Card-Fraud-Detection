"""
Interface Gradio pour le système expert de détection de fraude bancaire.
Permet de tester le modèle avec des entrées manuelles ou des exemples aléatoires.
"""

import logging
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

# Ajout du dossier racine au path pour les imports locaux
sys.path.append(str(Path(__file__).parents[2]))

from src.models.xgboost_model import FraudDetectionModel
from src.utils.config import FEATURE_COLUMNS, MODEL_PATH, TEST_DATA_PATH

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Chargement des ressources ---

try:
    logger.info(f"Chargement du modèle depuis {MODEL_PATH}...")
    model = FraudDetectionModel.load(MODEL_PATH)
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {e}")
    model = None

try:
    logger.info(f"Chargement des données de test depuis {TEST_DATA_PATH}...")
    test_X = pd.read_parquet(TEST_DATA_PATH)
    logger.info(f"Données de test chargées ({len(test_X)} lignes).")
except Exception as e:
    logger.error(f"Erreur lors du chargement des données de test : {e}")
    test_X = None


# --- Fonctions de l'interface ---

def predict_fraud(*args):
    """
    Récupère les entrées, effectue la prédiction et formate le résultat.
    """
    if model is None:
        return "❌ Erreur : Modèle non chargé", "N/A"

    try:
        # Création du DataFrame d'entrée à partir des arguments
        # L'ordre des args doit correspondre exactement à FEATURE_COLUMNS
        input_data = dict(zip(FEATURE_COLUMNS, args))
        df = pd.DataFrame([input_data])

        # Prédiction via le modèle
        result = model.predict(df)
        probability = result.iloc[0]["probability"]
        prediction = result.iloc[0]["prediction"]

        # Formatage de la probabilité
        prob_percent = f"{probability * 100:.2f}%"

        # Détermination du message de décision et de la couleur
        if prediction == 1:
            decision = "⚠️ FRAUDE DÉTECTÉE"
            color = "#ef4444"  # Rouge moderne
        else:
            decision = "✅ TRANSACTION NORMALE"
            color = "#22c55e"  # Vert moderne

        # Création du composant HTML pour l'affichage de la décision
        decision_html = f"""
        <div style="
            text-align: center; 
            padding: 20px; 
            border-radius: 10px; 
            background-color: {color}22; 
            color: {color}; 
            font-size: 24px; 
            font-weight: bold; 
            border: 2px solid {color};
            margin-top: 20px;
        ">
            {decision}
        </div>
        """

        return decision_html, prob_percent

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return f"❌ Erreur : {str(e)}", "N/A"


def load_random_example():
    """
    Pioche une ligne aléatoire dans le jeu de test et renvoie les valeurs des features.
    """
    if test_X is None or test_X.empty:
        logger.warning("Données de test non disponibles pour l'exemple aléatoire.")
        return [0.0] * len(FEATURE_COLUMNS)

    try:
        sample = test_X.sample(1).iloc[0]
        # Conversion explicite en float Python pour éviter les types numpy/pandas
        return [float(sample[col]) for col in FEATURE_COLUMNS]
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'exemple : {e}")
        return [0.0] * len(FEATURE_COLUMNS)


# --- Construction de l'interface Gradio ---

def create_app():
    """
    Définit le layout et les composants de l'application Gradio.
    """
    with gr.Blocks(title="Système Expert de Détection de Fraude") as demo:
        gr.Markdown(
            """
            # 🛡️ Système Expert de Détection de Fraude
            
            Ce système utilise un modèle **XGBoost** de pointe pour analyser les transactions bancaires. 
            Il identifie les comportements suspects en temps réel en se basant sur 30 caractéristiques techniques.
            
            ---
            """
        )

        with gr.Row():
            # Colonne de gauche : Entrées
            with gr.Column(scale=3):
                gr.Markdown("### 📝 Paramètres de la Transaction")
                
                with gr.Group():
                    with gr.Row():
                        time_input = gr.Slider(
                            minimum=-2.5, maximum=2.5, step=0.01, 
                            label="🕒 Time (Standardisé)", value=0.0
                        )
                        amount_input = gr.Slider(
                            minimum=-2.0, maximum=5.0, step=0.01, 
                            label="💰 Amount (Log-Robust Scaled)", value=0.0
                        )

                with gr.Tabs():
                    with gr.TabItem("Features V1 - V10"):
                        v_inputs_1_10 = [
                            gr.Number(label=f"V{i}", value=0.0) 
                            for i in range(1, 11)
                        ]
                    with gr.TabItem("Features V11 - V20"):
                        v_inputs_11_20 = [
                            gr.Number(label=f"V{i}", value=0.0) 
                            for i in range(11, 21)
                        ]
                    with gr.TabItem("Features V21 - V28"):
                        v_inputs_21_28 = [
                            gr.Number(label=f"V{i}", value=0.0) 
                            for i in range(21, 29)
                        ]

                # Liste ordonnée de tous les composants d'entrée pour correspondre à FEATURE_COLUMNS
                all_inputs = [time_input] + v_inputs_1_10 + v_inputs_11_20 + v_inputs_21_28 + [amount_input]

                with gr.Row():
                    btn_random = gr.Button("🎲 Charger un exemple aléatoire", variant="secondary")
                    btn_predict = gr.Button("🔍 Analyser la transaction", variant="primary")

            # Colonne de droite : Résultats
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Résultats de l'Analyse")
                
                with gr.Group():
                    output_prob = gr.Textbox(
                        label="Probabilité de fraude estimée",
                        placeholder="Cliquez sur Analyser pour voir le résultat",
                        interactive=False
                    )
                    output_decision = gr.HTML()

                gr.Markdown(
                    """
                    > **Note :** Les variables V1 à V28 sont issues d'une transformation PCA 
                    > pour protéger la confidentialité des données bancaires. 
                    > Les valeurs sont ici attendues sous forme standardisée.
                    """
                )

        # -- Interactivité --
        
        # Action du bouton de prédiction
        btn_predict.click(
            fn=predict_fraud,
            inputs=all_inputs,
            outputs=[output_decision, output_prob]
        )

        # Action du bouton de chargement aléatoire
        btn_random.click(
            fn=load_random_example,
            outputs=all_inputs
        )

        gr.Markdown(
            """
            ---
            *Développé dans le cadre du projet de détection de fraude - Modèle XGBoost*
            """
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    # Thème déplacé dans launch() pour Gradio 6.0+
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
    )
