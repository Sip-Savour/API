from datetime import date

from pydantic import BaseModel
from typing import List, Optional 

# --- VINS ---
class WineRequest(BaseModel):
    features: str
    color: str = None

class BottleInfo(BaseModel):
    title: str
    description: str
    variety: str
    color: str 


class WineResponse(BaseModel):
    bottle: List[BottleInfo] | None

# --- UTILISATEURS ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    date_naissance : date

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    date_naissance : date

class UserLogin(BaseModel):
    email: str
    password: str

# --- TOKEN ---

class Token(BaseModel):
    access_token: str
    token_type: str

# --- FAVORIS ---
class FavoriteCreate(BaseModel):
    user_id: int
    wine_id: int
