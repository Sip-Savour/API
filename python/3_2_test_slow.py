import automl
import joblib
import numpy as np
import os
import pandas as pd
import json
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# 1. CONFIGURATION : CHARGEMENT DU JSON DE COULEURS
# =============================================================================
DATA_DIR = "data/"
GENERATED_DIR = "generated_files/pkl/"
JSON_FILE = DATA_DIR+"wine_colors.json"
VARIETY_COLOR_MAP = {}

if os.path.exists(JSON_FILE):
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            VARIETY_COLOR_MAP = json.load(f)
        print(f"✅ Configuration chargée : {len(VARIETY_COLOR_MAP)} variétés de vin connues.")
    except Exception as e:
        print(f"⚠️ Erreur de lecture du JSON : {e}")
else:
    print(f"⚠️ Attention : '{JSON_FILE}' introuvable. Le filtrage couleur sera inactif.")


# =============================================================================
# 2. FONCTION DE TEST
# =============================================================================
def tester_prediction_avec_filtre(nom_test, mots_cles, couleur_imposee=None):
    print(f"\n--- Test : {nom_test} ---")
    print(f"   Demande : '{mots_cles}'")
    if couleur_imposee:
        print(f"   🔒 CONTRAINTE : Doit être '{couleur_imposee}'")

    # A. Chargement des outils
    try:
        keywords = joblib.load(GENERATED_DIR+"keywords_list.pkl")
        knn = joblib.load(GENERATED_DIR+"model_knn.pkl")
        tfidf = joblib.load(GENERATED_DIR+"vectorizer_knn.pkl")
        df_meta = pd.read_pickle(GENERATED_DIR+"wines_metadata.pkl")
    except FileNotFoundError:
        print("❌ Fichiers modèles manquants. Lancez '1_prepare.py' et '3_train_recommender.py'.")
        return

    # B. Étape AutoML (Classification)
    vecteur = np.zeros((1, len(keywords)), dtype=int)
    for i, word in enumerate(keywords):
        if word in mots_cles.lower():
            vecteur[0, i] = 1
            
    temp_file = "temp_test"
    np.savetxt(f"{temp_file}.data", vecteur, fmt='%d')
    
    try:
        preds = automl.predict(temp_file)
        cepage_predit = preds[0]
        print(f"   🍷 Prédiction IA : {cepage_predit}")
    except:
        cepage_predit = None
    finally:
        if os.path.exists(f"{temp_file}.data"): os.remove(f"{temp_file}.data")

    # C. Étape Recommandation (KNN + FILTRE COULEUR)
    # On vectorise la demande pour le KNN
    vec_knn = tfidf.transform([mots_cles])
    
    # On cherche large (les 50 plus proches voisins) pour avoir du choix après filtrage
    distances, indices = knn.kneighbors(vec_knn, n_neighbors=50)
    
    best_bottle = None
    
    for i in indices[0]:
        candidat = df_meta.iloc[i]
        variete = candidat['variety']
        
        # --- UTILISATION DU JSON ---
        couleur_reelle = VARIETY_COLOR_MAP.get(variete, "unknown")
        
        if couleur_imposee:
            # Si on connait la couleur et qu'elle ne matche pas -> On passe au suivant
            if couleur_reelle != "unknown" and couleur_reelle != couleur_imposee:
                continue
        
        # Si on passe le filtre, on garde le candidat
        best_bottle = candidat
        break
    
    # Résultat
    if best_bottle is not None:
        c = VARIETY_COLOR_MAP.get(best_bottle['variety'], '?')
        print(f"   🏆 RECOMMANDATION : {best_bottle['title']}")
        print(f"      Cépage : {best_bottle['variety']} (Couleur: {c})")
        print(f"      Note : {best_bottle['points']}/100")
        print(f"      Prix : {best_bottle['price']}$")
    else:
        print("   ❌ Aucun vin trouvé respectant la couleur demandée.")

# =============================================================================
# 3. EXÉCUTION DES TESTS
# =============================================================================
if __name__ == "__main__":
    # 1. Tests classiques
    tester_prediction_avec_filtre("Poulet Sauce Crème", "white butter creamy oak vanilla")
    print("-> Attendu : Chardonnay")

    tester_prediction_avec_filtre("Risotto aux Champignons", "red light cherry earth berry")
    print("-> Attendu : Pinot Noir")

    tester_prediction_avec_filtre("Steak au Poivre", "red spicy pepper dark structure")
    print("-> Attendu : Syrah ou Shiraz")

    tester_prediction_avec_filtre("Salade d'été", "white crisp apple pear fresh")
    print("-> Attendu : Sauvignon Blanc ou Pinot Grigio")

    tester_prediction_avec_filtre("Tarte aux Fruits", "sweet honey fruit apricot")
    print("-> Attendu : Riesling ou Moscato")

    # 2. Tests de Contraintes (Force Rouge ou Force Blanc)
    print("\n=== TESTS DE ROBUSTESSE (FILTRE COULEUR) ===")
    
    # Cas normal : Steak -> on force Rouge
    tester_prediction_avec_filtre("Steak au Poivre (Force Red)", "red spicy pepper structure", couleur_imposee="red")
    
    # Cas Piège : Steak -> on force Blanc (Doit donner un vin blanc, pas rouge !)
    tester_prediction_avec_filtre("Steak au Poivre (Force White)", "red spicy pepper structure", couleur_imposee="white")
