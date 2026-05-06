import joblib
from pathlib import Path

# 1. Charger le fichier
data = joblib.load(Path("saved_models/fraud_detection_xgboost.pkl"))

# 2. Explorer son contenu
print("Type:", type(data))  # C'est un dictionnaire
print("Clés disponibles:", data.keys())

# 3. Analyser les métriques
print("Scores CV:", data["cv_scores"])
print("Métriques test:", data["test_metrics"])
print("Seuil optimal:", data["best_threshold"])
print("Features utilisées:", data["feature_names"])

# 4. Examiner le modèle XGBoost
model = data["model"]
print("Nombre d'arbres:", model.n_estimators)
print("Importance des features:", model.feature_importances_)