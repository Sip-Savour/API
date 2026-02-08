import joblib
import pandas as pd
import numpy as np
import json
import time
import os
import re

# ================= INITIALISATION =================

print("⏳ Initialisation du système (Chargement en RAM)...")
t_load_start = time.time()

# ================= CONFIGURATION =================

DATA_DIR      = "data/"
GENERATED_DIR = "generated_files/pkl/"

# Modèles et Données
MODEL_CLASSIF = "automl/results/best_model.pkl" 
MODEL_KNN     = GENERATED_DIR + "model_knn.pkl"
VECT_KNN      = GENERATED_DIR + "vectorizer_knn.pkl"
METADATA      = GENERATED_DIR + "wines_metadata.pkl"
COLORS_FILE   = DATA_DIR      + "wine_colors.json"

GROUPS_FILE   = GENERATED_DIR + "keyword_groups.pkl"  
COLUMNS_FILE  = GENERATED_DIR + "keywords_list.pkl"   

try:
    # Chargement des IA
    if os.path.exists(MODEL_CLASSIF):
        clf_model = joblib.load(MODEL_CLASSIF)
        print("   Modèle Classification (AutoML) chargé.")
    else:
        print(f"   Modèle Classification manquant : {MODEL_CLASSIF}")
        clf_model = None

    if os.path.exists(MODEL_KNN):
        knn_model = joblib.load(MODEL_KNN)
        knn_vect  = joblib.load(VECT_KNN)
        print("   Moteur de Recommandation (KNN) chargé.")
    else:
        print(f"   ERREUR : Modèle KNN introuvable ({MODEL_KNN})")
        exit(1)
    
    # Chargement des Données
    if os.path.exists(METADATA):
        df_meta = pd.read_pickle(METADATA)
        print(f"   Métadonnées chargées ({len(df_meta)} vins).")
    else:
        # Fallback sur le CSV si le pickle n'existe pas
        csv_path = DATA_DIR + "wines_db_full.csv"
        print(f"   Pickle métadonnées absent, lecture CSV ({csv_path})...")
        df_meta = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
    
    # Chargement de la "Carte Mentale" (Mapping Synonymes -> Features)
    if os.path.exists(GROUPS_FILE) and os.path.exists(COLUMNS_FILE):
        KEYWORD_GROUPS = joblib.load(GROUPS_FILE)
        ORDERED_COLUMNS = joblib.load(COLUMNS_FILE)
        print(f"   Dictionnaire de synonymes chargé : {len(KEYWORD_GROUPS)} méta-catégories.")
    else:
        print("   ERREUR CRITIQUE : Fichiers de configuration manquants (keyword_groups.pkl).")
        exit(1)

    # Chargement Couleurs
    variety_map = {}
    if os.path.exists(COLORS_FILE):
        with open(COLORS_FILE, "r", encoding="utf-8") as f:
            variety_map = json.load(f)
    
except Exception as e:
    print(f"\n ERREUR FATALE LORS DU CHARGEMENT.\n   Détail : {e}")
    exit(1)

print(f"Système prêt en {time.time() - t_load_start:.2f} secondes.\n")



def text_to_vector(user_text):
    """
    Transforme l'entrée utilisateur ("I want cherry") 
    en vecteur compréhensible pour le modèle([1, 0, 0...]) 
    en utilisant les groupes de synonymes.
    """
    if not user_text:
        return np.zeros((1, len(ORDERED_COLUMNS)), dtype=int)

    user_text = user_text.lower()
    
    # On crée un vecteur de zéros de la taille du nombre de colonnes d'entraînement
    vector = np.zeros((1, len(ORDERED_COLUMNS)), dtype=int)
    
    # Pour chaque colonne connue de l'IA (ex: "red_fruit", "oak"...)
    for i, col_name in enumerate(ORDERED_COLUMNS):
        synonyms = KEYWORD_GROUPS.get(col_name, [])
        
        # On regarde si L'UN des synonymes est dans le texte
        for word in synonyms:
            # Recherche simple (plus robuste que regex complexe pour des tests rapides)
            if word in user_text:
                vector[0, i] = 1
                break # Une seule occurrence suffit pour activer la feature
                
    return vector

# ================= PREDICTION =================
def fast_predict(description, color_constraint=None):
    start = time.time()
    
    # --- Classification (Type de vin probable) ---
    cepage_estime = "Non Disponible"
    if clf_model:
        try:
            # Traduction (User -> Vecteur IA via Synonymes)
            vec_automl = text_to_vector(description)
            
            # Prédiction
            pred = clf_model.predict(vec_automl)
            cepage_estime = pred[0]
        except Exception as e:
            cepage_estime = f"Erreur Classif"
            # print(e) # Décommenter pour debug

    # --- Recommandation KNN (Recherche de bouteille) ---
    # Le KNN utilise le vectorizer TF-IDF entraîné sur le texte brut
    try:
        vec_knn = knn_vect.transform([description])
        distances, indices = knn_model.kneighbors(vec_knn, n_neighbors=50)
        
        best_bottle = None
        status = "Aucun résultat"
        
        for i in indices[0]:
            candidat = df_meta.iloc[i]
            variete = candidat['variety']
            
            # --- FILTRE COULEUR ---
            couleur_reelle = variety_map.get(variete, "unknown")
            
            if color_constraint:
                # Si on connait la couleur et qu'elle ne matche pas -> Skip
                if couleur_reelle != "unknown" and couleur_reelle != color_constraint:
                    continue
            
            best_bottle = candidat
            status = "Optimal"
            break
        
        # Fallback (si le filtre couleur a tout éliminé)
        if best_bottle is None and len(indices[0]) > 0:
            best_bottle = df_meta.iloc[indices[0][0]]
            status = "Fallback (Couleur ignorée)"
            
    except Exception as e:
        best_bottle = None
        status = f"Erreur KNN: {e}"

    duration = time.time() - start
    return cepage_estime, best_bottle, duration, status


# ================= TEST =================
tests = [
    # Test 1 : Vocabulaire simple 
    ("Poulet Classique", "chicken white citrus butter", "white"),
    
    # Test 2 : Utilisation des Synonymes 
    ("Barbecue Expert", "steak grilled ash cigar pepper", "red"),
    
    # Test 3 : Fruits précis 
    ("Bordeaux Style", "beef cassis cedar structured", "red"),
    
    # Test 4 : Dessert 
    ("Dessert Noix", "cake marzipan honey sweet", "white"),
]

print("=== DÉBUT DES TESTS ===")
print(f"{'TEST':<20} | {'TEMPS':<8} | {'CLASSIF (IA)':<25} | {'RECO (KNN)'}")
print("-" * 110)

total_time = 0

for nom, desc, color in tests:
    cepage, bouteille, duree, st = fast_predict(desc, color)
    total_time += duree
    
    if bouteille is not None:
        nom_bouteille = str(bouteille['title'])[:40] + "..."
    else:
        nom_bouteille = "Aucune suggestion"
        
    print(f"{nom:<20} | {duree:.4f}s  | {str(cepage):<25} | {nom_bouteille}")

avg = total_time / len(tests)
print("-" * 110)
print(f"⚡ Moyenne : {avg:.4f} s/req")