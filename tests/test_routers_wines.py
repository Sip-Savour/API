import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# IMPORTS STRICTS (Assurez-vous qu'il n'y a pas de 'app.')
from api import app
from routers.wines import get_db
# On importe EXACTEMENT les mêmes objets que le routeur utilise
from routers.auth import get_current_user, oauth2_scheme
from database import Base, Wine, User

# Configuration de la base de données de test
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Création d'un utilisateur factice qui a les attributs nécessaires pour votre fonction
# (id, etc. pour le random.seed)
mock_user = User(id=1, username="testuser", email="test@test.com")

# Surcharges des dépendances
app.router.on_startup.clear()  # Désactive init_db() de api.py
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = lambda: mock_user
app.dependency_overrides[oauth2_scheme] = lambda: "token-factice"

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    # On ajoute un vin pour être sûr que random.choice ne plante pas
    db.add(Wine(id=1, title="Vin Test", variety="Chardonnay", color="White"))
    db.commit()
    yield
    Base.metadata.drop_all(bind=engine)


def test_get_random_wines():
    response = client.get("/wines/random")
    assert response.status_code == 200


def test_get_weekly_recommendation():
    # On ajoute le Header Authorization pour satisfaire la validation de FastAPI
    # Même si la dépendance est surchargée, FastAPI vérifie parfois la signature
    headers = {"Authorization": "Bearer fake-token"}
    response = client.get("/wines/weekly", headers=headers)

    # DEBUG : Si ça échoue encore, on affiche pourquoi
    if response.status_code == 422:
        print("\nERREUR 422 DÉTAILLÉE :", response.json())

    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert data["id"] == 1