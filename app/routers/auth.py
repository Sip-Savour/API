from fastapi import APIRouter, HTTPException
from passlib.context import CryptContext
import sys
import os
from database import SessionLocal, User
from models import UserCreate, UserResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

@router.post("/signup", response_model=UserResponse, tags=["Auth"])
def create_user(user: UserCreate):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == user.email).first():
            raise HTTPException(status_code=400, detail="Email déjà utilisé.")
        
        if db.query(User).filter(User.username == user.username).first():
            raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris.")

        new_user = User(
            username=user.username,
            email=user.email,
            password_hash=get_password_hash(user.password)
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