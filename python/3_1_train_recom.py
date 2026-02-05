import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Fichier généré par 1_prepare.py
DATA_DIR = "data/"
GENERATED_DIR = "generated_files/pkl/"
INPUT_DB = DATA_DIR + "wines_db_full.csv"

def train_knn():
    print("⏳ Chargement de la base de vins pour le recommandeur...")
    
    if not os.path.exists(INPUT_DB):
        print(f"❌ Erreur : {INPUT_DB} introuvable. Avez-vous lancé 1_prepare.py ?")
        return

    # On charge la base complète
    # on_bad_lines='skip' permet d'éviter les erreurs de parsing CSV
    df = pd.read_csv(INPUT_DB, on_bad_lines='skip', low_memory=False)
    
    # --- CORRECTION DU BUG 'TITLE' ---
    # Si la colonne 'title' n'existe pas, on la crée nous-mêmes
    if 'title' not in df.columns:
        print("⚠️ Colonne 'title' absente. Création automatique (Winery + Variety)...")
        # On remplace les trous par du vide pour éviter d'écrire "nan"
        df['winery'] = df['winery'].fillna("Domaine Inconnu")
        df['variety'] = df['variety'].fillna("")
        
        # On crée le titre : "Château Margaux Cabernet Sauvignon"
        df['title'] = df['winery'].astype(str) + " " + df['variety'].astype(str)
    # ---------------------------------

    # Nettoyage de sécurité
    # Maintenant que 'title' existe forcément, cette ligne ne plantera plus
    df = df.dropna(subset=['description', 'variety', 'title'])
    
    # On garde uniquement les colonnes utiles
    cols_to_keep = ['title', 'variety', 'description', 'points', 'price']
    # On vérifie que 'points' et 'price' existent aussi, sinon on les crée vides
    for col in ['points', 'price']:
        if col not in df.columns:
            df[col] = 0

    df_meta = df[cols_to_keep]
    
    print(f"   > Entraînement sur {len(df_meta)} bouteilles.")

    # 1. VECTORISATION
    print("   > Vectorisation des descriptions...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(df_meta['description'])
    
    # 2. MODÈLE KNN
    print("   > Entraînement du modèle KNN...")
    knn = NearestNeighbors(n_neighbors=20, metric='cosine', n_jobs=-1)
    knn.fit(matrix)
    
    # 3. SAUVEGARDE
    print("💾 Sauvegarde des fichiers...")
    joblib.dump(knn, GENERATED_DIR+"model_knn.pkl")
    joblib.dump(tfidf, GENERATED_DIR+"vectorizer_knn.pkl")
    df_meta.reset_index(drop=True).to_pickle(GENERATED_DIR+"wines_metadata.pkl")
    
    print("✅ SUCCÈS ! Recommandeur prêt.")
    print("   - model_knn.pkl")
    print("   - vectorizer_knn.pkl")
    print("   - wines_metadata.pkl")

if __name__ == "__main__":
    train_knn()
