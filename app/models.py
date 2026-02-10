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

class WineResponse(BaseModel):
    bottle: List[BottleInfo] | None

# --- UTILISATEURS ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str