import automl
import os

# Nom de base
DATASET_NAME = "wine_train" 

# CETTE PROTECTION EST OBLIGATOIRE POUR LE MULTIPROCESSING
if __name__ == "__main__":
    if os.path.exists(f"{DATASET_NAME}.data"):
        print(f"--- Lancement de l'entraînement sur '{DATASET_NAME}' ---")
        
        # On lance l'AutoML
        try:
            automl.fit(DATASET_NAME, mode="stable")
            print("✅ Entraînement terminé.")
        except Exception as e:
            print(f"❌ Erreur critique : {e}")
    else:
        print(f"❌ ERREUR : Lancez d'abord 1_prepare.py")
