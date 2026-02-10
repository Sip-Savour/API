import time
import os
import sys
import pandas as pd
import numpy as np
import json
import joblib

# On remonte d'un niveau pour trouver le package automl
sys.path.append("..")

try:
    import automl
    print("Package 'automl' importé.")
except ImportError:
    print("ERREUR : Package 'automl' introuvable.")

# ================= CONFIGURATION =================
BASE_DIR = "../" 
GENERATED_DIR = BASE_DIR + "generated_files/pkl/"
DATA_DIR      = BASE_DIR + "data/"

MODEL_KNN     = GENERATED_DIR + "model_knn.pkl"
VECT_KNN      = GENERATED_DIR + "vectorizer_knn.pkl"
METADATA      = GENERATED_DIR + "wines_metadata.pkl"
GROUPS_FILE   = GENERATED_DIR + "keyword_groups.pkl"
COLUMNS_FILE  = GENERATED_DIR + "keywords_list.pkl"
COLORS_FILE   = DATA_DIR      + "wine_colors.json"

# Chargement unique au démarrage (Global)
knn_model = None
knn_vect = None
df_meta = None
KEYWORD_GROUPS = {}
ORDERED_COLUMNS = []
variety_map = {}

try:
    if os.path.exists(MODEL_KNN):
        knn_model = joblib.load(MODEL_KNN)
        knn_vect  = joblib.load(VECT_KNN)
    
    if os.path.exists(METADATA):
        df_meta = pd.read_pickle(METADATA)
    else:
        df_meta = pd.read_csv(DATA_DIR + "wines_db_full.csv", on_bad_lines='skip', low_memory=False)

    if os.path.exists(GROUPS_FILE):
        KEYWORD_GROUPS = joblib.load(GROUPS_FILE)
        ORDERED_COLUMNS = joblib.load(COLUMNS_FILE)

    if os.path.exists(COLORS_FILE):
        with open(COLORS_FILE, "r", encoding="utf-8") as f:
            variety_map = json.load(f)

    print("Moteur de prédiction chargé.")

except Exception as e:
    print(f"Erreur chargement ressources : {e}")


# ================= OUTILS =================
def text_to_dataframe(user_text):
    vector = np.zeros((1, len(ORDERED_COLUMNS)), dtype=int)
    if user_text:
        user_text = user_text.lower()
        for i, col_name in enumerate(ORDERED_COLUMNS):
            synonyms = KEYWORD_GROUPS.get(col_name, [])
            for word in synonyms:
                if word in user_text:
                    vector[0, i] = 1
                    break 
    return pd.DataFrame(vector, columns=ORDERED_COLUMNS)

# ================= PRÉDICTION =================
def fast_predict(description, color_constraint=None, top_n=5):
    start = time.time()
    
    # --- 1. AUTOML ---
    cepage_decision = "Inconnu"
    base_filename = "temp_query"      
    real_filename = "temp_query.data" 
    
    try:
        input_data = text_to_dataframe(description)
        input_data.to_csv(real_filename, index=False, header=False, sep=" ")
        
        prediction = automl.predict(base_filename)
        
        if isinstance(prediction, (list, np.ndarray)):
            cepage_decision = prediction[0] if len(prediction) > 0 else "Inconnu"
        else:
            cepage_decision = prediction
            
    except Exception as e:
        print(f"Erreur AutoML: {e}")
        cepage_decision = "Erreur"
    finally:
        if os.path.exists(real_filename):
            try: os.remove(real_filename)
            except: pass

    # --- 2. KNN (Récupération de 5 bouteilles) ---
    best_bottles = [] 
    seen_titles = set() 
    
    if knn_model:
        try:
            vec_knn = knn_vect.transform([description])
            distances, indices = knn_model.kneighbors(vec_knn, n_neighbors=100)
            
            if cepage_decision not in ["Inconnu", "Erreur"]:
                for i in indices[0]:
                    if len(best_bottles) >= top_n: break
                    
                    candidat = df_meta.iloc[i]
                    if candidat['title'] in seen_titles: continue
                    
                    col = variety_map.get(candidat['variety'], "unknown")
                    if color_constraint and col != "unknown" and col != color_constraint:
                        continue
                    if candidat['variety'] != cepage_decision:
                        continue
                    
                    best_bottles.append(candidat)
                    seen_titles.add(candidat['title'])
            
            if len(best_bottles) < top_n:
                for i in indices[0]:
                    if len(best_bottles) >= top_n: break

                    candidat = df_meta.iloc[i]
                    if candidat['title'] in seen_titles: continue

                    col = variety_map.get(candidat['variety'], "unknown")
                    if color_constraint and col != "unknown" and col != color_constraint:
                        continue
                    
                    best_bottles.append(candidat)
                    seen_titles.add(candidat['title'])

        except Exception as e:
            print(f"Erreur KNN: {e}")

    return best_bottles