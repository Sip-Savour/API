from fastapi import APIRouter, HTTPException
from passlib.context import CryptContext
import sys
import os
from database import SessionLocal, User
from models import UserCreate, UserResponse, UserLogin, Token
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def check_age(birth_date: date):
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age >= 18

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