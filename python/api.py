from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json

# ================= CONFIGURATION =================
app = FastAPI(title="Sommelier IA API", description="API de recommandation de vin pour Android")

# Chemins (Identiques à votre script de test)
BASE_DIR = "../" 

GENERATED_DIR = BASE_DIR + "generated_files/pkl/"
DATA_DIR      = BASE_DIR + "data/"

# Mise à jour des chemins avec BASE_DIR
MODEL_CLASSIF = BASE_DIR + "automl/results/best_model.pkl"
MODEL_KNN     = GENERATED_DIR + "model_knn.pkl"
VECT_KNN      = GENERATED_DIR + "vectorizer_knn.pkl"
METADATA      = GENERATED_DIR + "wines_metadata.pkl"
GROUPS_FILE   = GENERATED_DIR + "keyword_groups.pkl"
COLUMNS_FILE  = GENERATED_DIR + "keywords_list.pkl"
COLORS_FILE   = DATA_DIR      + "wine_colors.json"

# Variables Globales (Chargées au démarrage)
ai_resources = {}

# ================= MODÈLES DE DONNÉES (JSON) =================
# Ce que l'Android envoie
class WineRequest(BaseModel):
    features: str        # Ex: "body_full tannins pepper"
    color: str = None    # Ex: "red" ou null

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
    print("⏳ Chargement des cerveaux de l'IA...")
    try:
        # 1. Chargement Classification (Optionnel si erreur dimensions)
        if os.path.exists(MODEL_CLASSIF):
            ai_resources['clf'] = joblib.load(MODEL_CLASSIF)
        else:
            ai_resources['clf'] = None
            print("⚠️ Modèle Classification absent (Mode Recommandation seule).")

        # 2. Chargement KNN (Requis)
        ai_resources['knn'] = joblib.load(MODEL_KNN)
        ai_resources['vect'] = joblib.load(VECT_KNN)
        
        # 3. Données & Métadonnées
        if os.path.exists(METADATA):
            ai_resources['meta'] = pd.read_pickle(METADATA)
        else:
            # Fallback CSV
            ai_resources['meta'] = pd.read_csv(DATA_DIR + "wines_db_full.csv", on_bad_lines='skip', low_memory=False)

        # 4. Synonymes & Couleurs
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
    
    # --- 1. CLASSIFICATION (Tentative de deviner le cépage) ---
    cepage_estime = "Inconnu"
    if ai_resources['clf']:
        try:
            vec = text_to_vector(description)
            cepage_estime = ai_resources['clf'].predict(vec)[0]
        except:
            pass # On ignore les erreurs de dimension ici, le KNN est prioritaire

    # --- 2. RECOMMANDATION (KNN) ---
    try:
        # Vectorisation TF-IDF
        vec_knn = ai_resources['vect'].transform([description])
        # Recherche des 50 plus proches
        distances, indices = ai_resources['knn'].kneighbors(vec_knn, n_neighbors=50)
        
        df = ai_resources['meta']
        variety_map = ai_resources['colors']
        
        best_bottle = None
        
        # Filtrage par couleur
        for i in indices[0]:
            candidate = df.iloc[i]
            variety = candidate['variety']
            wine_color = variety_map.get(variety, "unknown")
            
            if color_constraint:
                if wine_color != "unknown" and wine_color != color_constraint:
                    continue # On saute si la couleur ne matche pas
            
            best_bottle = candidate
            break
        
        # Fallback (Si aucun vin de la bonne couleur n'est trouvé dans le top 50)
        if best_bottle is None:
            best_bottle = df.iloc[indices[0][0]] # On prend le plus proche mathématiquement
            cepage_estime += " (Couleur non garantie)"

        # Construction de la réponse propre
        return {
            "cepage": str(cepage_estime),
            "bottle": {
                "title": str(best_bottle['title']),
                "description": str(best_bottle['description'])[:200] + "...", # On coupe un peu
                "price": float(best_bottle['price']) if pd.notnull(best_bottle['price']) else 0.0,
                "variety": str(best_bottle['variety'])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur algorithmique : {str(e)}")
