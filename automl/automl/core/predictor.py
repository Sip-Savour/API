"""Module de prédiction sur de nouvelles données."""
import pickle
import joblib
from automl._config import RESULTS_DIR,BASE_DIR
from automl.utils.io import load_test_data
from automl.utils.logging import log
import pandas as pd
import json
import time
from pathlib import Path


GENERATED_DIR = BASE_DIR / "generated_files" / "pkl"
DATA_DIR = BASE_DIR / "data"

MODEL_KNN     = GENERATED_DIR / "model_knn.pkl"
VECT_KNN      = GENERATED_DIR / "vectorizer_knn.pkl"
METADATA      = GENERATED_DIR / "wines_metadata.pkl"
COLORS_FILE   = DATA_DIR / "wine_colors.json"

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

def predict(data_path: str,color_constraint : str,top_n : int = 5):
    """
    Charge le meilleur modèle entraîné et prédit sur un nouveau dataset.
    
    Args:
        data_path: Chemin vers le fichier .data (sans extension ou avec)
        
    Returns:
        list: Les prédictions
    """

    log("predict", f"Démarrage prédiction sur : {data_path}")
    vec_knn = knn_vect.transform([data_path])
        
        # 2. LA MAGIE EST ICI : 
        # Si une couleur est exigée, on demande au KNN de nous donner les 15 000 vins
        # les plus proches aromatiquement (ça reste ultra-rapide).
    search_k = 15000 if (color_constraint and color_constraint != "null") else max(top_n * 2, 50) 
    return knn_model.kneighbors(vec_knn, n_neighbors=search_k)