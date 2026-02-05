import joblib
import pandas as pd
import numpy as np
import json
import time
import os

# =============================================================================
# 1. INITIALISATION (On charge tout une seule fois)
# =============================================================================
print("⏳ Initialisation du système (Chargement en RAM)...")
t_load_start = time.time()

# Chemins des fichiers
GENERATED_DIR = "../generated_files/pkl/"
DATA_DIR      = "../data/"
MODEL_CLASSIF = GENERATED_DIR + "../automl/results/best_model.pkl"  
MODEL_KNN     = GENERATED_DIR + "model_knn.pkl"
VECT_KNN      = GENERATED_DIR + "vectorizer_knn.pkl"
KEYWORDS      = GENERATED_DIR + "keywords_list.pkl"
METADATA      = GENERATED_DIR + "wines_metadata.pkl"
COLORS        = DATA_DIR      + "wine_colors.json"

# Chargement
try:
    # 1. Le modèle de classification (Cerveau 1)
    if os.path.exists(MODEL_CLASSIF):
        clf_model = joblib.load(MODEL_CLASSIF)
        print("   ✅ Modèle Classification chargé.")
    else:
        print(f"   ⚠️ '{MODEL_CLASSIF}' introuvable. La classification sera simulée.")
        clf_model = None

    # 2. Le système de recommandation (Cerveau 2)
    knn_model = joblib.load(MODEL_KNN)
    knn_vect  = joblib.load(VECT_KNN)
    df_meta   = pd.read_pickle(METADATA)
    keywords  = joblib.load(KEYWORDS)
    print("   ✅ Système de Recommandation chargé.")

    # 3. La configuration couleur
    variety_map = {}
    if os.path.exists(COLORS):
        with open(COLORS, "r", encoding="utf-8") as f:
            variety_map = json.load(f)
        print("   ✅ Filtres Couleurs chargés.")
    
except Exception as e:
    print(f"\n❌ ERREUR FATALE : Impossible de charger les ressources.\n   Détail : {e}")
    exit(1)

print(f"🚀 Prêt en {time.time() - t_load_start:.2f} secondes.\n")


# =============================================================================
# 2. FONCTION DE PRÉDICTION ULTRA-RAPIDE (0 I/O Disque)
# =============================================================================
def fast_predict(description, color_constraint=None):
    start = time.time()
    
    # --- A. Classification (En mémoire) ---
    cepage_estime = "Non Disponible"
    if clf_model:
        # Transformation One-Hot manuelle (sans passer par fichier .data)
        vec = np.zeros((1, len(keywords)), dtype=int)
        desc_lower = description.lower()
        for i, word in enumerate(keywords):
            if word in desc_lower:
                vec[0, i] = 1
        
        # Prédiction directe via Sklearn
        try:
            cepage_estime = clf_model.predict(vec)[0]
        except:
            pass

    # --- B. Recommandation KNN (En mémoire) ---
    vec_knn = knn_vect.transform([description])
    distances, indices = knn_model.kneighbors(vec_knn, n_neighbors=50)
    
    best_bottle = None
    
    for i in indices[0]:
        candidat = df_meta.iloc[i]
        variete = candidat['variety']
        couleur_reelle = variety_map.get(variete, "unknown")
        
        # Filtre Couleur
        if color_constraint:
            if couleur_reelle == "unknown" or couleur_reelle != color_constraint:
                continue
        
        best_bottle = candidat
        break
    
    # Fallback si trop strict
    if best_bottle is None:
        best_bottle = df_meta.iloc[indices[0][0]]
        status = "Fallback (Pas de match exact couleur)"
    else:
        status = "Optimal"

    duration = time.time() - start
    return cepage_estime, best_bottle, duration

# =============================================================================
# 3. BOUCLE DE TEST RAPIDE
# =============================================================================
tests = [
    ("Poulet Crème", "white butter creamy oak vanilla", None),
    ("Risotto", "red light cherry earth berry", None),
    ("Steak Poivre", "red spicy pepper dark structure", "red"),
    ("Steak Piège (Force Blanc)", "red spicy pepper structure", "white"), # Le fameux test du Pinot Gris
    ("Salade", "white crisp apple pear fresh", "white"),
]

print("=== DÉBUT DES TESTS RAPIDES ===")
print(f"{'TEST':<25} | {'TEMPS':<10} | {'CLASSIF':<20} | {'RECO BOUTEILLE'}")
print("-" * 100)

total_time = 0

for nom, desc, color in tests:
    cepage, bouteille, duree = fast_predict(desc, color)
    total_time += duree
    
    # Affichage compact
    nom_bouteille = bouteille['title'][:30] + "..." if bouteille is not None else "Aucune"
    print(f"{nom:<25} | {duree:.4f}s    | {str(cepage):<20} | {nom_bouteille}")

avg = total_time / len(tests)
print("-" * 100)
print(f"⚡ Moyenne par prédiction : {avg:.4f} secondes")
print(f"🚀 Vitesse estimée : {1/avg:.0f} requêtes / seconde")
