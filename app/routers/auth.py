from fastapi import APIRouter, HTTPException, Depends, status
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import sys
import os
from database import SessionLocal, User, Favorite, Wine, engine
from models import UserCreate, UserResponse, UserLogin, Token, FavoriteCreate
from datetime import date

##sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter()

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

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

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token manquant ou invalide",
        )
    return token

@router.post("/signup", response_model=UserResponse, tags=["Auth"])
def create_user(user: UserCreate):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == user.email).first():
            raise HTTPException(status_code=400, detail="Email déjà utilisé.")
        
        if db.query(User).filter(User.username == user.username).first():
            raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris.")

        if not check_age(user.date_naissance):
            raise HTTPException(status_code=400, detail="Vous devez avoir au moins 18 ans pour vous inscrire.")

        new_user = User(
            username=user.username,
            email=user.email,
            password_hash=get_password_hash(user.password),
            date_naissance=user.date_naissance
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return new_user

    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Erreur inscription : {str(e)}")
    finally:
        db.close()

@router.post("/login", response_model=Token)
def login(user_data: UserLogin):
    db = SessionLocal()
    user = db.query(User).filter(User.email == user_data.email).first()
    
    # Vérification du mot de passe haché
    if not user or not pwd_context.verify(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
    
    return {"access_token": f"fake-token-for-{user.id}", "token_type": "bearer"}

@router.get("/test-db", tags=["Test"])
def get_all_users():
    """Affiche tous les utilisateurs pour vérifier l'enregistrement"""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return [
            {
                "id": u.id, 
                "username": u.username, 
                "email": u.email, 
                "date_naissance": u.date_naissance
            } for u in users
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/favorites", tags=["User Profile"])
def add_favorite(fav: FavoriteCreate, current_user_token: str = Depends(get_current_user), db: Session = Depends(get_db)):
    # 1. Retrouver l'ID de l'utilisateur à partir du token (pour l'instant exemple statique)
    # Dans un vrai système JWT, on décoderait le token ici.
    # Exemple statique ici :
    user_id = 1 

    # 2. Vérifier si le favori existe déjà
    existing = db.query(Favorite).filter(
        Favorite.user_id == user_id, 
        Favorite.wine_id == fav.wine_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Ce vin est déjà dans vos favoris.")

    new_fav = Favorite(user_id=user_id, wine_id=fav.wine_id)
    db.add(new_fav)
    db.commit()
    
    return {"message": "Vin ajouté aux favoris avec succès !"}

@router.get("/favorites", tags=["User Profile"])
def list_favorites(current_user_token: str = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = 1 # Exemple. En vrai, tu extrairais l'ID de l'utilisateur du token JWT ici.
    user_favs = db.query(Favorite).filter(Favorite.user_id == user_id).all()
    
    # On renvoie les détails des vins favoris
    return [
        {
            "id": f.wine.id,
            "title": f.wine.title,
            "variety": f.wine.variety
        } for f in user_favs
    ]

##To do : implémenter la logique de gestion des tokens d'authentification (JWT, OAuth2, etc.) pour sécuriser les endpoints et gérer les sessions utilisateur.
## actuellement, le token est un simple placeholder pour démonstration.

## To do 2 : vérifier/finaliser la logique de gestion des favoris (ajout/suppression de vins aux favoris d'un utilisateur) et créer les endpoints correspondants.