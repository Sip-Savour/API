from fastapi import FastAPI
from database import init_db
import sys
import os
from routers import predict, auth

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI(
    title="Sommelier IA API", 
    description="API de recommandation de vin (AutoML + KNN) & Gestion Utilisateurs",
    version="1.0"
)

# Initialisation DB
@app.on_event("startup")
def startup_event():
    print("🚀 Démarrage de l'API...")
    init_db()

@app.get("/")
def home():
    return {"status": "online", "message": "API opérationnelle."}

# --- ENREGISTREMENT DES ROUTERS ---
app.include_router(predict.router)
app.include_router(auth.router)