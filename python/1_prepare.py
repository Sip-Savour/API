import pandas as pd
import numpy as np
import re
import os
import joblib

# ================= CONFIGURATION =================
INPUT_CSV = "winemag-data_first150k.csv"
BASENAME  = "wine_train"
SAMPLE_SIZE = 5000  # On peut en mettre plus car c'est très rapide à calculer

# C'est ici qu'on définit vos "Colonnes"
# Chaque mot deviendra une feature (0 ou 1)
KEYWORDS_COLUMNS = sorted([
    "tannin", "dry", "sweet", "acid", "crisp", "structure", 
    "full-bodied", "light", "rich", "creamy", "butter", "fruit", "berry", 
    "cherry", "citrus", "lemon", "apple", "pear", "oak", "wood", "vanilla", 
    "spicy", "pepper", "mineral", "honey", "earth", "red", "white", "black",
    "chocolate", "plum", "tobacco", "leather", "smoke"
])

def main():
    print(f"--- 1. Chargement & Nettoyage ---")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fichier {INPUT_CSV} manquant.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Suppression des vides
    df = df.dropna(subset=['description', 'variety'])
    
    # Filtre Top 10 Cépages (pour simplifier la classification)
   # top_10 = df['variety'].value_counts().nlargest(10).index
   # df = df[df['variety'].isin(top_10)]
    
    # Sampling
    #if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    #    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    print(f"   > Dataset de travail : {len(df)} vins.")

    # --- 2. CRÉATION DES COLONNES (MANUAL ONE-HOT) ---
    print("--- 2. Génération des colonnes 'Mots-clés' ---")
    
    # On met tout en minuscule pour la recherche
    descriptions = df['description'].str.lower()
    
    # Matrice qui va contenir nos 0 et 1
    # On initialise tout à 0
    X_matrix = np.zeros((len(df), len(KEYWORDS_COLUMNS)), dtype=int)
    
    # On remplit colonne par colonne
    for i, word in enumerate(KEYWORDS_COLUMNS):
        # On regarde si le mot est présent dans chaque description
        # contains est plus rapide que apply
        # astype(int) transforme True en 1 et False en 0
        presence = descriptions.str.contains(word, regex=False).astype(int).values
        X_matrix[:, i] = presence
        
        # Petit log pour voir ce qui se passe
        count = np.sum(presence)
        if count > 0:
            print(f"   - Colonne '{word}' : trouvée dans {count} vins")

    print(f"   > Matrice générée : {X_matrix.shape} (Vins x Mots-clés)")
    
    # Sauvegarde de la liste des colonnes pour pouvoir faire pareil lors du test
    joblib.dump(KEYWORDS_COLUMNS, "keywords_list.pkl")

    # --- 3. SAUVEGARDE FORMAT S1 ---
    print(f"--- 3. Écriture des fichiers {BASENAME} ---")
    
    # Fichier .data : Que des 0 et des 1 séparés par des espaces
    # fmt='%d' veut dire "entier" (pas de virgule)
    np.savetxt(f"{BASENAME}.data", X_matrix, fmt='%d')
    
    # Fichier .solution : Les Labels
    df['variety'].to_csv(f"{BASENAME}.solution", index=False, header=False)
    
    # Sauvegarde DB complète pour l'app
    df.to_csv("wines_db_full.csv", index=False)

    print("✅ SUCCÈS ! Données binaires prêtes.")

if __name__ == "__main__":
    main()
