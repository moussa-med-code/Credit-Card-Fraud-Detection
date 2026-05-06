FROM python:3.9-slim

# Installation des dépendances système nécessaires pour XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Définition du chemin de recherche Python pour les imports locaux
ENV PYTHONPATH=/app

# Exposition du port utilisé par FastAPI
EXPOSE 8000

CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
