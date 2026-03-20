import joblib
import pandas as pd
import json
import time
from pathlib import Path
from automl import predict as automl_predict
# =============================================================================
# 1. CONFIGURATION DES CHEMINS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "generated_files" / "pkl"
DATA_DIR = BASE_DIR / "data"

MODEL_KNN     = GENERATED_DIR / "model_knn.pkl"
VECT_KNN      = GENERATED_DIR / "vectorizer_knn.pkl"
METADATA      = GENERATED_DIR / "wines_metadata.pkl"
COLORS_FILE   = DATA_DIR / "wine_colors.json"

# =============================================================================
# 2. CHARGEMENT GLOBAL AU DÉMARRAGE
# =============================================================================
try:
    print("⏳ Initialisation du moteur KNN (Multi-recommandations)...")
    knn_model = joblib.load(MODEL_KNN)
    knn_vect  = joblib.load(VECT_KNN)
    
    if METADATA.exists():
        df_meta = pd.read_pickle(METADATA)
    else:
        df_meta = pd.read_csv(DATA_DIR / "wines_db_full.csv", low_memory=False)
        
    with open(COLORS_FILE, "r", encoding="utf-8") as f:
        variety_map = json.load(f)
        
    SYSTEM_READY = True
    print(f"✅ Système prêt : {len(df_meta)} vins chargés en RAM.")
except Exception as e:
    print(f"❌ Erreur au chargement du moteur : {e}")
    SYSTEM_READY = False

# =============================================================================
# 3. FONCTION DE PRÉDICTION (SUPPORT TOP_N)
# =============================================================================
def fast_predict(data_path: str, color_constraint: str = None, top_n: int = 5):
    """
    Recherche les 'top_n' vins qui matchent le mieux la description (saveurs),
    tout en respectant STRICTEMENT la couleur demandée.
    """
    if not SYSTEM_READY:
        return {"error": "Modèle non chargé sur le serveur"}

    description = data_path
    
    # Sécurité pour éviter l'Erreur 500
    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else 0.0
        except:
            return 0.0

    try:
        # 1. On appelle le module automl
        # (Assurez-vous que l'import en haut de fichier est `from automl import predictor`)
        distances, indices = automl_predict(description, color_constraint, top_n)
        
        # 2. Sécurité : si le modèle renvoie vide
        if len(indices) == 0:
            return {"error": "Échec de la prédiction KNN."}

        recommendations = []
        
        # 3. Parcours des résultats du plus proche au plus éloigné
        for i in indices[0]:
            if len(recommendations) >= top_n:
                break # On a trouvé nos 5 pépites
                
            candidat = df_meta.iloc[i]
            variete = candidat.get('variety', 'unknown')
            couleur_vin = variety_map.get(variete, "unknown")
            
            # FILTRE DE COULEUR ABSOLU
            if color_constraint and color_constraint != "null" and color_constraint != "":
                if couleur_vin != color_constraint:
                    continue # Mauvaise couleur, on ignore
            
            # On ajoute le vin (uniquement les 4 champs utiles)
            recommendations.append({
                "title": str(candidat.get("title", "Nom inconnu")),
                "description": str(candidat.get("description", "")),
                "variety": str(candidat.get("variety", "unknown")),
                "color": str(variety_map.get(candidat.get('variety', ''), "unknown"))
            })
            
        # 4. LE FALLBACK (Corrigé et bien indenté)
        # Si la liste est vide mais qu'on a des indices, on renvoie les meilleurs vins trouvés
        if not recommendations and len(indices[0]) > 0:
            for i in indices[0][:top_n]:
                candidat = df_meta.iloc[i]
                recommendations.append({
                    "title": str(candidat.get("title", "Nom inconnu")),
                    "description": str(candidat.get("description", "")),
                    "variety": str(candidat.get("variety", "unknown")),
                    "color": str(variety_map.get(candidat.get('variety', ''), "unknown"))
                })

        return recommendations

    except Exception as e:
        return {"error": f"Erreur critique de l'IA : {str(e)}"}