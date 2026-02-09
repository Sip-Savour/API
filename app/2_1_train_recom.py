import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

DATA_DIR = "data/"
GENERATED_DIR = "generated_files/pkl/"
INPUT_DB = DATA_DIR + "wines_db_full.csv"

def train_knn():
    print("Chargement de la base de vins pour les recommandations...")
    
    if not os.path.exists(INPUT_DB):
        print(f"Erreur : {INPUT_DB} introuvable.")
        return

    df = pd.read_csv(INPUT_DB, on_bad_lines='skip', low_memory=False)
    
    if 'title' not in df.columns:
        print("Colonne 'title' absente. Création automatique (Winery + Variety)...")
        df['winery'] = df['winery'].fillna("Domaine Inconnu")
        df['variety'] = df['variety'].fillna("")
        
        df['title'] = df['winery'].astype(str) + " " + df['variety'].astype(str)

    # Nettoyage de sécurité
    df = df.dropna(subset=['description', 'variety', 'title'])
    
    # On garde uniquement les colonnes utiles
    cols_to_keep = ['title', 'variety', 'description']

    df_meta = df[cols_to_keep]
    
    print(f"   > Entraînement sur {len(df_meta)} bouteilles.")

    # VECTORISATION
    print("   > Vectorisation des descriptions...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(df_meta['description'])
    
    # MODÈLE KNN
    print("   > Entraînement du modèle KNN...")
    knn = NearestNeighbors(n_neighbors=20, metric='cosine', n_jobs=-1)
    knn.fit(matrix)
    
    # SAUVEGARDE
    print("Sauvegarde des fichiers...")
    joblib.dump(knn, GENERATED_DIR+"model_knn.pkl")
    joblib.dump(tfidf, GENERATED_DIR+"vectorizer_knn.pkl")
    df_meta.reset_index(drop=True).to_pickle(GENERATED_DIR+"wines_metadata.pkl")
    
    print("Fichier crées avec succès:")
    print("   - model_knn.pkl")
    print("   - vectorizer_knn.pkl")
    print("   - wines_metadata.pkl")

if __name__ == "__main__":
    train_knn()
