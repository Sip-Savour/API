import pandas as pd
import numpy as np
import re
import os
import joblib

# ================= CONFIGURATION =================
DATA_DIR = "data/"
GENERATED_ML_DIR = "generated_files/automl/"
GENERATED_PKL_DIR = "generated_files/pkl/"
INPUT_CSV = DATA_DIR + "winemag-data_first150k.csv"
OUTPUT_CSV = DATA_DIR + "wines_db_full.csv"
BASENAME  = "wine_train"

KEYWORDS_COLUMNS = [
    # --- VOS ANCIENS MOTS (Gardez-les) ---
    "red", "white", "rose", "dry", "sweet",
    "acid", "sugar", "fruit", "berry", "cherry",
    "lemon", "apple", "pear", "citrus",
    "oak", "wood", "vanilla", "butter", "cream",
    "pepper", "spicy", "structure", "light",
    "earth", "mineral", "honey", "chocolate", "tobacco", "smoke",

    # Structure & Corps
    "acidity", "tannins", "bodied", "smooth", "tannic", "dense",
    "richness", "silky", "round", "heavy", "crisp", "firm",
    
    # Style & Caractère
    "ripe", "fresh", "soft", "dark", "green", "balanced",
    "fruity", "clean", "elegant", "complex", "jammy", "lush",
    "savory", "pure", "refined", "bright",
    
    # Arômes de Fruits & Plantes
    "blackberry", "raspberry", "peach", "plum", "currant",
    "lime", "melon", "pineapple", "apricot", "orange",
    "grapefruit", "strawberry", "cassis", "tropical",
    "herbal", "herb", "floral", "mint", "grass",
    
    # Arômes d'Élevage & Épices
    "spice", "toast", "toasty", "cedar", "coffee",
    "mocha", "cinnamon", "licorice", "cola", "cocoa",
    "leather", "stone", "slate", "smoky"
]

def main():
    print(f"--- 1. Chargement & Nettoyage ---")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fichier {INPUT_CSV} manquant.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Suppression des vides
    df = df.dropna(subset=['description', 'variety'])
    
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
    joblib.dump(KEYWORDS_COLUMNS, GENERATED_PKL_DIR + "keywords_list.pkl")

    # --- 3. SAUVEGARDE FORMAT S1 ---
    print(f"--- 3. Écriture des fichiers {GENERATED_DIR + BASENAME} ---")
    
    # Fichier .data : Que des 0 et des 1 séparés par des espaces
    # fmt='%d' veut dire "entier" (pas de virgule)
    np.savetxt(f"{GENERATED_DIR + BASENAME}.data", X_matrix, fmt='%d')
    
    # Fichier .solution : Les Labels
    df['variety'].to_csv(f"{GENERATED_DIR + BASENAME}.solution", index=False, header=False)
    
    # Sauvegarde DB complète pour l'app
    df.to_csv(OUTPUT_CSV, index=False)

    print("✅ SUCCÈS ! Données binaires prêtes.")

if __name__ == "__main__":
    main()
