import pandas as pd
import numpy as np
from database import SessionLocal, Wine, init_db
import os

# ================= CONFIGURATION =================
CSV_FILE = "data/wines_db_full.csv"

def migrate():
    print("Démarrage de la migration CSV -> SQL...")
    
    # Création des tables vides
    init_db()
    
    # Lecture du CSV
    if not os.path.exists(CSV_FILE):
        print(f"Erreur : Fichier CSV introuvable ici : {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Remplacer les NaN (vide) par None pour SQL
    df = df.replace({np.nan: None})
    
    print(f"   > Chargement de {len(df)} vins...")

    # Insertion en base
    db = SessionLocal()
    
    # Vérif anti-doublon (si on lance le script 2 fois)
    if db.query(Wine).count() > 0:
        print("La base contient déjà des données. Migration annulée.")
        db.close()
        return

    batch = []
    count = 0
    
    for index, row in df.iterrows():
        # Création de l'objet Vin
        wine = Wine(
            title=row['title'],
            description=row['description'],
            variety=row['variety'],
            price=row['price'],
            winery=row.get('winery', None),
            country=row.get('country', None),
            province=row.get('province', None),
            points=row.get('points', None)
        )
        batch.append(wine)
        
        if len(batch) >= 1000:
            db.add_all(batch)
            db.commit()
            batch = []
            count += 1000
            print(f"   > {count} vins insérés...", end='\r')
            
    if batch:
        db.add_all(batch)
        db.commit()
        
    db.close()
    print(f"\nMigration Terminée ! Base de données prête.")

if __name__ == "__main__":
    migrate()
