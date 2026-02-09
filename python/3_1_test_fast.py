import time
import os
import sys
import pandas as pd
import numpy as np
import json
import joblib

sys.path.append("..")

# 1. IMPORT PACKAGE
try:
    import automl
    print("✅ Package 'automl' importé.")
except ImportError:
    print("❌ ERREUR : Package 'automl' introuvable.")
    exit(1)

# =============================================================================
# 1. INITIALISATION
# =============================================================================
print("⏳ Initialisation du Pipeline...")
t_init = time.time()

BASE_DIR = ""
DATA_DIR      = BASE_DIR + "data/"
GENERATED_DIR = BASE_DIR + "generated_files/pkl/"

MODEL_KNN     = GENERATED_DIR + "model_knn.pkl"
VECT_KNN      = GENERATED_DIR + "vectorizer_knn.pkl"
METADATA      = GENERATED_DIR + "wines_metadata.pkl"
COLORS_FILE   = DATA_DIR      + "wine_colors.json"
GROUPS_FILE   = GENERATED_DIR + "keyword_groups.pkl"
COLUMNS_FILE  = GENERATED_DIR + "keywords_list.pkl"

try:
    if os.path.exists(MODEL_KNN):
        knn_model = joblib.load(MODEL_KNN)
        knn_vect  = joblib.load(VECT_KNN)
    else:
        knn_model = None

    if os.path.exists(METADATA):
        df_meta = pd.read_pickle(METADATA)
    else:
        df_meta = pd.read_csv(DATA_DIR + "wines_db_full.csv", on_bad_lines='skip', low_memory=False)

    if os.path.exists(GROUPS_FILE):
        KEYWORD_GROUPS = joblib.load(GROUPS_FILE)
        ORDERED_COLUMNS = joblib.load(COLUMNS_FILE)
    else:
        print("❌ ERREUR Config : Lancez 1_prepare.py")
        exit(1)

    variety_map = {}
    if os.path.exists(COLORS_FILE):
        with open(COLORS_FILE, "r", encoding="utf-8") as f:
            variety_map = json.load(f)

except Exception as e:
    print(f"❌ Erreur Init : {e}")
    exit(1)

print(f"🚀 Système prêt en {time.time() - t_init:.2f} s.\n")

# =============================================================================
# 2. OUTILS
# =============================================================================
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

# =============================================================================
# 3. PRÉDICTION (DOUBLE PASSE : STRICT -> FALLBACK)
# =============================================================================
def fast_predict(description, color_constraint=None):
    start = time.time()
    
    # --- A. DÉCISION AUTOML ---
    cepage_decision = "Inconnu"
    base_filename = "temp_query"      
    real_filename = "temp_query.data" 
    
    try:
        input_data = text_to_dataframe(description)
        # Format .DATA (Espaces + Pas de header)
        input_data.to_csv(real_filename, index=False, header=False, sep=" ")
        
        prediction = automl.predict(base_filename)
        
        if isinstance(prediction, (list, np.ndarray)):
            cepage_decision = prediction[0] if len(prediction) > 0 else "Inconnu"
        else:
            cepage_decision = prediction
            
    except Exception as e:
        cepage_decision = f"Erreur: {e}"
    finally:
        if os.path.exists(real_filename):
            try: os.remove(real_filename)
            except: pass

    # --- B. RECHERCHE KNN ---
    best_bottle = None
    strategy_used = "Aucune"
    
    if knn_model:
        try:
            vec_knn = knn_vect.transform([description])
            distances, indices = knn_model.kneighbors(vec_knn, n_neighbors=100)
            
            # ---------------------------------------------------------
            # PASSE 1 : MODE STRICT (On respecte l'AutoML)
            # ---------------------------------------------------------
            for i in indices[0]:
                candidat = df_meta.iloc[i]
                
                # Vérif Couleur
                variete = candidat['variety']
                couleur_reelle = variety_map.get(variete, "unknown")
                if color_constraint and couleur_reelle != "unknown" and couleur_reelle != color_constraint:
                    continue
                
                # Vérif AutoML (C'est la condition stricte)
                if candidat['variety'] == cepage_decision:
                    best_bottle = candidat
                    strategy_used = "✅ Accord AutoML"
                    break
            
            # ---------------------------------------------------------
            # PASSE 2 : MODE FALLBACK (Si Passe 1 a échoué)
            # On ignore l'AutoML, on prend juste le meilleur match texte
            # ---------------------------------------------------------
            if best_bottle is None:
                for i in indices[0]:
                    candidat = df_meta.iloc[i]
                    
                    # On garde quand même le filtre couleur (c'est important pour l'utilisateur)
                    variete = candidat['variety']
                    couleur_reelle = variety_map.get(variete, "unknown")
                    if color_constraint and couleur_reelle != "unknown" and couleur_reelle != color_constraint:
                        continue
                    
                    best_bottle = candidat
                    strategy_used = "⚠️ Fallback (AutoML ignoré)"
                    break

        except Exception as e:
            strategy_used = f"Erreur KNN: {e}"

    duration = time.time() - start
    return cepage_decision, best_bottle, duration, strategy_used

# =============================================================================
# 4. TESTS
# =============================================================================
tests = [
    ("Poulet (White)", "chicken white citrus butter", "white"),
    ("Steak (Red)", "steak grilled pepper smoke", "red"),
    ("Dessert (Piege)", "cake marzipan honey", "white"), # Cas où l'AutoML peut se tromper
]

print("=== DÉBUT DES TESTS AVEC FALLBACK ===")
print(f"{'TEST':<15} | {'DÉCISION IA':<20} | {'STRATÉGIE':<25} | {'RÉSULTAT'}")
print("-" * 100)

for nom, desc, color in tests:
    decision, bouteille, duree, strategie = fast_predict(desc, color)

    if bouteille is not None:
        res = f"{bouteille['title'][:30]}..."
    else:
        res = "Aucun résultat"

    print(f"{nom:<15} | {str(decision)[:20]:<20} | {strategie:<25} | {res}")

print("-" * 100)
