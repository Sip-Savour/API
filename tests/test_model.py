import pytest
from datetime import date
from pydantic import ValidationError
# Adaptez l'import selon le nom de votre fichier, ex: from schemas import UserCreate, WineRequest
from models import UserCreate, WineRequest, BottleInfo, AuthResponse, FavoriteCreate

def test_user_create_valid():
    """Teste la création d'un utilisateur avec des données valides[cite: 1]."""
    user_data = {
        "username": "johndoe",
        "email": "john@example.com",
        "password": "securepassword123",
        "date_naissance": date(1990, 1, 1)
    }
    user = UserCreate(**user_data)
    assert user.username == "johndoe"
    assert user.email == "john@example.com"

def test_wine_request_optional_color():
    """Teste que la couleur est bien optionnelle dans la requête de vin[cite: 1]."""
    req = WineRequest(features="Fruité et léger")
    assert req.features == "Fruité et léger"
    assert req.color is None

def test_favorite_create_types():
    """Vérifie que les ID sont bien reconnus comme des entiers[cite: 1]."""
    fav = FavoriteCreate(user_id=1, wine_id=42)
    assert fav.user_id == 1
    assert fav.wine_id == 42