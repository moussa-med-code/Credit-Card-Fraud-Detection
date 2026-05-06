import os
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ajout du dossier racine au path pour l'import des modules locaux
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.xgboost_model import FraudDetectionModel

app = FastAPI(
    title="API de Détection de Fraude Bancaire",
    description="Service d'inférence pour la détection de transactions frauduleuses utilisant XGBoost.",
    version="1.0.0"
)

# Modèle de données pour une transaction
class Transaction(BaseModel):
    features: List[float] = Field(
        ..., 
        description="Liste des caractéristiques de la transaction (V1-V28, Time, Amount)",
        example=[0.0] * 30
    )

# Variable globale pour stocker le modèle
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = Path("saved_models/fraud_detection_xgboost.pkl")
    
    if not model_path.exists():
        # Tentative de chargement du modèle s'il existe
        print(f"ATTENTION : Le modèle {model_path} est introuvable. L'API ne pourra pas prédire.")
        return

    try:
        model = FraudDetectionModel.load(model_path)
        print("✅ Modèle chargé avec succès pour l'API")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "message": "Bienvenue sur l'API de détection de fraude"
    }

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Le modèle n'est pas chargé sur le serveur."
        )
    
    try:
        # Conversion de la liste en DataFrame avec les noms de colonnes originaux
        if model.feature_names and len(transaction.features) == len(model.feature_names):
            df = pd.DataFrame([transaction.features], columns=model.feature_names)
        else:
            # Fallback si les noms de colonnes ne sont pas disponibles
            df = pd.DataFrame([transaction.features])
            
        result = model.predict(df)
        
        return {
            "is_fraud": bool(result["prediction"].iloc[0]),
            "probability": float(result["probability"].iloc[0]),
            "threshold_used": float(result["threshold"].iloc[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'inférence : {str(e)}")

@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        data = [t.features for t in transactions]
        df = pd.DataFrame(data, columns=model.feature_names if model.feature_names else None)
        
        results = model.predict(df)
        
        return {
            "predictions": [
                {
                    "is_fraud": bool(row["prediction"]),
                    "probability": float(row["probability"])
                }
                for _, row in results.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
