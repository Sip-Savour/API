from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

# Ajout du chemin pour trouver predict.py et database.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import fast_predict
from database import SessionLocal, User, init_db
from passlib.context import CryptContext
from sqlalchemy.exc import IntegrityError

# ================= CONFIGURATION =================
app = FastAPI(
    title="Sommelier IA API", 
    description="API de recommandation de vin (AutoML + KNN) & Gestion Utilisateurs",
    version="1.0"
)

# Sécurité (Hashage mots de passe)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

# Initialisation de la DB au démarrage
@app.on_event("startup")
def startup_event():
    print("🚀 Démarrage de l'API...")
    init_db() # Crée les tables si elles n'existent pas

# ================= MODÈLES DE DONNÉES (Pydantic) =================

# --- Partie VINS ---
class WineRequest(BaseModel):
    features: str        # Ex: "steak pepper"
    color: str = None    # Ex: "red" (Optionnel)

class BottleInfo(BaseModel):
    title: str
    description: str
    variety: str
    

class WineResponse(BaseModel):
    bottle: BottleInfo | None # Peut être null si aucun résultat

# --- Partie UTILISATEURS (Inscription) ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

# ================= ROUTES (Endpoints) =================

@app.get("/")
def home():
    return {"status": "online", "message": "API Sommelier opérationnelle."}

@app.post("/predict", response_model=WineResponse)
def predict_wine(req: WineRequest):
    """
    Reçoit une description et une couleur.
    Retourne le cépage estimé et la meilleure bouteille.
    """
    try:
        # Appel à predict.py
        cepage_estime, bouteille_trouvee = fast_predict(req.features, req.color)

        if bouteille_trouvee is None:
            return WineResponse(cepage=str(cepage_estime), bottle=None)

        # --- CORRECTION DE SÉCURITÉ ICI ---
        # On utilise .get() pour éviter le crash "KeyError: 'price'"
        # Si la colonne n'existe pas, on met une valeur par défaut.
        
        def safe_get(key, default):
            val = bouteille_trouvee.get(key, default)
            return default if pd.isna(val) else val

        info_bouteille = BottleInfo(
            title=str(safe_get('title', "Titre Inconnu")),
            description=str(safe_get('description', "Pas de description")),
            variety=str(safe_get('variety', "Inconnu")),
            
        )

        return WineResponse(
            bottle=info_bouteille
        )

    except Exception as e:
        print(f"❌ Erreur API Predict : {e}")
        # On renvoie l'erreur détaillée pour comprendre
        raise HTTPException(status_code=500, detail=str(e))

# --- Route 2 : Inscription (SQL) ---
@app.post("/signup", response_model=UserResponse)
def create_user(user: UserCreate):
    db = SessionLocal()
    try:
        # 1. Vérif email
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email déjà utilisé.")
        
        # 2. Vérif username
        existing_username = db.query(User).filter(User.username == user.username).first()
        if existing_username:
            raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris.")

        # 3. Création
        new_user = User(
            username=user.username,
            email=user.email,
            password_hash=get_password_hash(user.password)
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return new_user

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur inscription : {str(e)}")
    finally:
        db.close()