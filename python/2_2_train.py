import automl  
import os
import numpy as np
import time

# ================= CONFIGURATION =================
GENERATED_DIR = "generated_files/automl/"
DATASET_NAME  = GENERATED_DIR + "wine_train" 
DATA_FILE     = DATASET_NAME + ".data"

def train_with_automl():
    print(f"--- Lancement de l'entraînement AutoML (Cours) ---")

    # Vérification de sécurité
    if not os.path.exists(DATA_FILE):
        print(f"zRREUR : Le fichier '{DATA_FILE}' est introuvable.")
        return

    # Vérification des dimensions
    try:
        # On lit juste la première ligne pour voir combien il y a de colonnes
        with open(DATA_FILE, 'r') as f:
            first_line = f.readline()
            n_cols = len(first_line.strip().split())
            
        print(f"   > Données détectées : {n_cols} colonnes (Features).")
        
        if n_cols > 100:
            print("   ATTENTION : Nombre de colonnes élevé (>100).")
            print("      Le regroupement par synonymes n'a peut-être pas fonctionné.")
        else:
            print("   DIMENSIONS VALIDES : Le regroupement par synonymes est actif.")
            
    except Exception as e:
        print(f"   Impossible de vérifier les dimensions : {e}")

    # Lancement de l'AutoML
    print(f"   > Démarrage de automl.fit() sur '{DATASET_NAME}'...")
    start_time = time.time()
    
    try:
        automl.fit(DATASET_NAME, mode="stable")
        
        duration = time.time() - start_time
        print(f"Entraînement terminé en {duration:.1f} secondes.")
        print("Le nouveau modèle 'best_model.pkl' a écrasé l'ancien.")
        
    except Exception as e:
        print(f"CRASH AUTOML : {e}")

if __name__ == "__main__":
    train_with_automl()