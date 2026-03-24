import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Date
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ================= CONFIGURATION =================
# On calcule le chemin ABSOLU du projet pour trouver sommelier.db à tous les coups
# __file__ correspond à app/database.py (ou similaire), on remonte au parent pour la racine
BASE_DIR = Path(__file__).resolve().parent.parent
db_path = BASE_DIR / "sommelier.db"

SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ================= TABLES =================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    date_naissance = Column(Date, nullable=False)

    favorites = relationship("Favorite", back_populates="user")


class Wine(Base):
    __tablename__ = "wines"
    id = Column(Integer, primary_key=True, index=True)

    # Infos de base
    title = Column(String, index=True)
    description = Column(Text)
    variety = Column(String, index=True)
    color = Column(String)  # <--- NOUVELLE COLONNE COULEUR

    favorited_by = relationship("Favorite", back_populates="wine")


class Favorite(Base):
    __tablename__ = "favorites"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    wine_id = Column(Integer, ForeignKey("wines.id"))

    user = relationship("User", back_populates="favorites")
    wine = relationship("Wine", back_populates="favorited_by")


# ================= INITIALISATION & MIGRATION =================

def init_db():
    # 1. Création des tables si elles n'existent pas
    Base.metadata.create_all(bind=engine)
    print(f"Base de données initialisée à l'emplacement : {db_path}")

    # 2. Peuplement automatique de la table Wine
    db = SessionLocal()
    try:
        # On vérifie si la table des vins est vide
        if db.query(Wine).count() == 0:
            print("⏳ Table 'wines' vide. Lancement du peuplement automatique depuis le CSV...")

            # Imports placés ici pour ne pas ralentir l'API au quotidien
            import pandas as pd
            import numpy as np

            # Le chemin vers le fichier CSV
            csv_file = BASE_DIR / "data" / "wines_db_full.csv"

            if not csv_file.exists():
                print(f"❌ Erreur : Fichier CSV introuvable au chemin : {csv_file}")
                return

            # Lecture et nettoyage
            df = pd.read_csv(csv_file)
            df = df.replace({np.nan: None})

            print(f"   > Chargement de {len(df)} vins...")
            batch = []
            count = 0

            for index, row in df.iterrows():
                # On récupère la couleur (si la colonne n'existe pas ou est vide, on met "Inconnue")
                wine_color = row.get('color', 'Inconnue') if 'color' in df.columns else 'Inconnue'

                wine = Wine(
                    title=row['title'],
                    description=row['description'],
                    variety=row['variety'],
                    color=wine_color  # <--- SAUVEGARDE DE LA COULEUR ICI
                )
                batch.append(wine)

                # Insertion par lots de 1000 pour préserver la RAM
                if len(batch) >= 1000:
                    db.add_all(batch)
                    db.commit()
                    batch = []
                    count += 1000
                    print(f"   > {count} vins insérés...", end='\r')

            # Insertion du reliquat
            if batch:
                db.add_all(batch)
                db.commit()

            print("\n✅ Peuplement de la base terminé avec succès !")
        else:
            print("✅ La table 'wines' contient déjà des données. Migration ignorée.")

    except Exception as e:
        print(f"❌ Erreur lors du peuplement automatique : {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_db()