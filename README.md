# 🛡️ Détection de Fraude par Carte de Crédit (XGBoost)

Ce projet est une solution complète (End-to-End) de détection de fraude bancaire. Il utilise l'algorithme **XGBoost** pour traiter le déséquilibre extrême des données et propose une architecture modulaire prête pour la production.

## 🚀 Fonctionnalités
- **Pipeline de données optimisé** : Utilisation du format Parquet pour la rapidité.
- **Gestion du déséquilibre** : Calcul automatique du `scale_pos_weight` pour les classes minoritaires.
- **Modèle performant** : XGBoost avec Early Stopping et Cross-Validation stratifiée.
- **Multi-interface** : FastAPI (Inférence), Dashboard HTML (Vue.js), Streamlit et Gradio.

## 📁 Structure du Projet
```text
├── data/               # Données brutes et traitées (Parquet)
├── notebooks/          # Exploration et analyse (EDA)
├── src/
│   ├── api/            # FastAPI, Streamlit, Gradio
│   ├── models/         # Logique du modèle et hyperparamètres
│   └── preprocessing/  # Transformation des données
├── dashboard.html      # Interface utilisateur moderne (HTML/Vue.js)
├── Dockerfile          # Configuration du conteneur
└── requirements.txt    # Dépendances Python
```

## 🛠️ Installation et Utilisation

### Option 1 : Installation Locale (Développement)
Idéal pour explorer le code ou réentraîner le modèle.

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/votre-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Créer un environnement virtuel :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l'API :**
   ```bash
   uvicorn src.api.fastapi_app:app --reload
   ```

5. **Accéder au Dashboard :**
   Ouvrez simplement le fichier `dashboard.html` dans votre navigateur ou lancez Streamlit :
   ```bash
   streamlit run src/api/streamlit_app.py
   ```


---

### Option 2 : Installation via Docker (Déploiement)
La méthode la plus rapide et fiable pour tester l'application.

1. **Construire l'image :**
   ```bash
   docker build -t fraud-detection .
   ```

2. **Lancer le conteneur :**
   ```bash
   docker run -p 8000:8000 fraud-detection
   ```