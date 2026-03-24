import sys
import os
from pathlib import Path

# =====================================================================
# ASTUCE POUR LE CHEMIN PYTHON (Imports)
# =====================================================================
# __file__ correspond à app/config/5_migration.py
current_dir = Path(__file__).resolve().parent

# parent_dir correspond à app/
parent_dir = current_dir.parent

# On ajoute le dossier "app/" au système pour qu'il trouve database.py
sys.path.append(str(parent_dir))

import pandas as pd
import numpy as np
from database import SessionLocal, Wine, init_db

# =====================================================================
# CONFIGURATION DES FICHIERS
# =====================================================================
# project_root correspond au dossier racine de votre API (au-dessus de app)
project_root = parent_dir.parent

# On pointe vers le dossier data situé à la racine du projet
CSV_FILE = project_root / "data" / "wines_db_full.csv"


def migrate():
    print("Démarrage de la migration CSV -> SQL...")

    # Création des tables vides (se connecte à sommelier.db)
    init_db()

    # Lecture du CSV
    if not CSV_FILE.exists():
        print(f"❌ Erreur : Fichier CSV introuvable au chemin absolu : {CSV_FILE}")
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
        # Création de l'objet Vin avec UNIQUEMENT les champs conservés dans database.py
        wine = Wine(
            title=row['title'],
            description=row['description'],
            variety=row['variety']
        )
        batch.append(wine)

        # Insertion par lots de 1000 pour ne pas saturer la RAM
        if len(batch) >= 1000:
            db.add_all(batch)
            db.commit()
            batch = []
            count += 1000
            print(f"   > {count} vins insérés...", end='\r')

    # Insertion des derniers éléments restants
    if batch:
        db.add_all(batch)
        db.commit()

    db.close()
    print(f"\n✅ Migration Terminée ! Base de données prête avec les champs nettoyés.")


if __name__ == "__main__":
    migrate()