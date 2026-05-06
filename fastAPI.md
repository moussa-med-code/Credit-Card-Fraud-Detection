# Documentation de l'API FastAPI

Ce projet utilise **FastAPI** pour exposer le modèle de détection de fraude sous forme de service web RESTful. Cette interface permet d'intégrer le modèle de Machine Learning dans n'importe quelle application (Web, Mobile, ou système bancaire tiers).

## 1. Travail Réalisé
L'implémentation dans `src/api/fastapi_app.py` comprend :
*   **Initialisation au démarrage :** Le modèle XGBoost est chargé une seule fois lors du lancement du serveur (`@app.on_event("startup")`) pour garantir des prédictions ultra-rapides.
*   **Validation des données :** Utilisation de **Pydantic** pour s'assurer que les données envoyées à l'API respectent le format attendu (liste de caractéristiques numériques).
*   **Points de terminaison (Endpoints) :**
    *   `GET /` : Vérification de l'état du serveur et du modèle.
    *   `POST /predict` : Prédiction pour une seule transaction.
    *   `POST /predict/batch` : Prédiction simultanée pour une liste de transactions (optimisé pour les flux de données).
*   **Gestion des erreurs :** Retourne des codes d'erreur HTTP explicites (ex: 503 si le modèle est manquant, 400 si les données sont mal formées).

## 2. Comment utiliser l'API

### A. Lancer le serveur
Pour démarrer l'API localement, utilisez `uvicorn` depuis la racine du projet :
```bash
uvicorn src.api.fastapi_app:app --reload
```
*Le flag `--reload` permet de redémarrer automatiquement le serveur à chaque modification du code.*

### B. Documentation Interactive (Swagger)
Une fois le serveur lancé, FastAPI génère automatiquement une documentation interactive :
*   **URL :** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Vous pouvez tester les prédictions directement depuis cette page sans écrire une seule ligne de code client.

### C. Exemple de requête avec `curl`
Pour tester la détection de fraude en ligne de commande :
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [0.0, 1.1, -0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
}'
```
