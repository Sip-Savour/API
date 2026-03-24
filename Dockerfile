FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des librairies système requises par l'IA (libgomp1)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*


# Dépendances publiques (FastAPI, Pandas, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copie de tout le projet
COPY app ./app
COPY automl ./automl
COPY data ./data
COPY generated_files ./generated_files

# Installation officielle de votre package custom 'automl'
RUN pip install -e ./automl/

# Définition du chemin Python
ENV PYTHONPATH=/app

EXPOSE 8000

# On se place dans le dossier contenant api.py pour le lancement
WORKDIR /app/app

# Lancement de l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]