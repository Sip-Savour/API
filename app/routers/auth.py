from fastapi import APIRouter, HTTPException, Depends, status
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
import sys
import os
from database import SessionLocal, User, Favorite, Wine
from models import UserCreate, UserResponse, UserLogin, AuthResponse
from datetime import date, timedelta, datetime
from typing import Optional
from jose import jwt, JWTError

router = APIRouter()
SECRET_KEY = "put-it-in-env-variable-in-prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# On utilise la configuration hybride pour éviter les crashs de mot de passe
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


# ==========================================================
# GESTION DES TOKENS & SÉCURITÉ
# ==========================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
        )


def get_password_hash(password):
    return pwd_context.hash(password)


def check_age(birth_date: date):
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age >= 18


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- NOUVEAUTÉ : On récupère VRAIMENT l'utilisateur connecté ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    username: str = payload.get("sub")

    if username is None:
        raise HTTPException(status_code=401, detail="Token invalide")

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Utilisateur non trouvé")

    return user


# ==========================================================
# ROUTES D'AUTHENTIFICATION
# ==========================================================

@router.post("/signup", response_model=AuthResponse, tags=["Auth"])
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        if db.query(User).filter(User.email == user.email).first():
            raise HTTPException(status_code=400, detail="Email déjà utilisé.")

        if db.query(User).filter(User.username == user.username).first():
            raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris.")

        if not check_age(user.date_naissance):
            raise HTTPException(status_code=400, detail="Vous devez avoir au moins 18 ans.")

        new_user = User(
            username=user.username,
            email=user.email,
            password_hash=get_password_hash(user.password),
            date_naissance=user.date_naissance
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        access_token = create_access_token(data={"sub": new_user.username})

        return AuthResponse(
            token=access_token,
            userId=new_user.id,
            username=new_user.username,
            email=new_user.email
        )

    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@router.post("/login", response_model=AuthResponse, tags=["Auth"])
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()

    if not user or not pwd_context.verify(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Identifiants incorrects")

    access_token = create_access_token(data={"sub": user.username})

    return AuthResponse(
        token=access_token,
        userId=user.id,
        username=user.username,
        email=user.email
    )


# ==========================================================
# ROUTES DES FAVORIS (L'utilisateur est détecté automatiquement)
# ==========================================================

# Modèle pour lire la requête envoyée par Android ({"wineId": 123})
class FavoriteRequest(BaseModel):
    wineId: int


@router.post("/favorites", tags=["Favorites"])
def add_favorite(fav: FavoriteRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 1. Vérifier si le vin existe en base
    wine = db.query(Wine).filter(Wine.id == fav.wineId).first()
    if not wine:
        raise HTTPException(status_code=404, detail="Vin introuvable.")

    # 2. Vérifier si le favori existe déjà pour cet utilisateur
    existing = db.query(Favorite).filter(
        Favorite.user_id == current_user.id,
        Favorite.wine_id == fav.wineId
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Ce vin est déjà dans vos favoris.")

    # 3. Ajout
    new_fav = Favorite(user_id=current_user.id, wine_id=fav.wineId)
    db.add(new_fav)
    db.commit()

    return {"message": "Vin ajouté aux favoris avec succès !"}


@router.delete("/favorites/{wine_id}", tags=["Favorites"])
def remove_favorite(wine_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    fav = db.query(Favorite).filter(
        Favorite.user_id == current_user.id,
        Favorite.wine_id == wine_id
    ).first()

    if not fav:
        raise HTTPException(status_code=404, detail="Ce vin n'est pas dans vos favoris.")

    db.delete(fav)
    db.commit()

    return {"message": "Vin retiré des favoris."}


@router.get("/favorites", tags=["Favorites"])
def list_favorites(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_favs = db.query(Favorite).filter(Favorite.user_id == current_user.id).all()

    # On reformate les données pour que ça corresponde exactement au "WineDto" d'Android
    results = []
    for f in user_favs:
        if f.wine:
            results.append({
                "id": f.wine.id,
                "title": f.wine.title,
                "description": f.wine.description or "",
                "variety": f.wine.variety or "Inconnu",
                "color": f.wine.color or "Inconnue"
            })

    return results