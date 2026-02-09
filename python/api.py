from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json
from predict import fast_predict

# ================= CONFIGURATION =================
app = FastAPI(title="Sommelier IA API", description="API de recommandation de vin pour Android")

BASE_DIR = "../" 

GENERATED_DIR = BASE_DIR + "generated_files/pkl/"
DATA_DIR      = BASE_DIR + "data/"

MODEL_CLASSIF = BASE_DIR + "automl/results/best_model.pkl"
MODEL_KNN     = GENERATED_DIR + "model_knn.pkl"
VECT_KNN      = GENERATED_DIR + "vectorizer_knn.pkl"
METADATA      = GENERATED_DIR + "wines_metadata.pkl"
GROUPS_FILE   = GENERATED_DIR + "keyword_groups.pkl"
COLUMNS_FILE  = GENERATED_DIR + "keywords_list.pkl"
COLORS_FILE   = DATA_DIR      + "wine_colors.json"

ai_resources = {}

# ================= MODÈLES DE DONNÉES (JSON) =================
# Ce que l'Android envoie
class WineRequest(BaseModel):
    features: str        # Ex: "body_full tannins pepper"
    color: str = None    # Ex: "red","white","rose" ou null

# Ce que l'Android reçoit
class BottleInfo(BaseModel):
    title: str
    description: str
    price: float
    variety: str

class WineResponse(BaseModel):
    cepage: str
    bottle: BottleInfo

# ================= CHARGEMENT AU DÉMARRAGE =================
@app.on_event("startup")
def load_resources():
    print("Chargement des cerveaux de l'IA...")
    try:
        #Chargement Classification 
        if os.path.exists(MODEL_CLASSIF):
            ai_resources['clf'] = joblib.load(MODEL_CLASSIF)
        else:
            ai_resources['clf'] = None
            print("⚠️ Modèle Classification absent (Mode Recommandation seule).")

        # Chargement KNN 
        ai_resources['knn'] = joblib.load(MODEL_KNN)
        ai_resources['vect'] = joblib.load(VECT_KNN)
        
        # Données & Métadonnées
        if os.path.exists(METADATA):
            ai_resources['meta'] = pd.read_pickle(METADATA)
        else:
            # Fallback CSV
            ai_resources['meta'] = pd.read_csv(DATA_DIR + "wines_db_full.csv", on_bad_lines='skip', low_memory=False)

        # Synonymes & Couleurs
        ai_resources['groups'] = joblib.load(GROUPS_FILE)
        ai_resources['cols'] = joblib.load(COLUMNS_FILE)
        
        with open(COLORS_FILE, "r", encoding="utf-8") as f:
            ai_resources['colors'] = json.load(f)

        print("✅ API Prête à servir !")
        
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE au démarrage : {e}")

# ================= LOGIQUE MÉTIER (Interne) =================
def text_to_vector(user_text):
    """Traduit le texte en vecteur 0/1 selon les synonymes"""
    columns = ai_resources['cols']
    groups = ai_resources['groups']
    
    vector = np.zeros((1, len(columns)), dtype=int)
    user_text = user_text.lower()
    
    for i, col_name in enumerate(columns):
        synonyms = groups.get(col_name, [])
        for word in synonyms:
            if word in user_text:
                vector[0, i] = 1
                break
    return vector

# ================= ENDPOINTS (Routes) =================

@app.get("/")
def home():
    return {"status": "online", "message": "Le Sommelier est réveillé. Utilisez /predict"}

@app.post("/predict", response_model=WineResponse)
def predict_wine(req: WineRequest):
    """
    Reçoit : { "features": "beef pepper", "color": "red" }
    Retourne : Le meilleur vin trouvé.
    """
    if 'knn' not in ai_resources:
        raise HTTPException(status_code=500, detail="L'IA n'est pas chargée.")

    description = req.features
    color_constraint = req.color
    bouteille= fast_predict(description, color_constraint)

        # Construction de la réponse propre
    return {
        "bottle": {
            "title": str(bouteille['title']),
            "description": str(bouteille['description']), 
            "variety": str(bouteille['variety'])
        }
    }

