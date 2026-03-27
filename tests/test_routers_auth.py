import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ON UTILISE LES IMPORTS COURTS (SANS app.)
from api import app
from routers.auth import get_db
from database import Base, Wine, User, Favorite

engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.router.on_startup.clear()
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_signup_success():
    response = client.post("/signup", json={
        "username": "tester", "email": "tester@test.com",
        "password": "password123", "date_naissance": "1990-01-01"
    })
    assert response.status_code == 200
    assert "token" in response.json()


def test_login_and_favorites():
    # 1. Setup
    db = TestingSessionLocal()
    db.add(Wine(id=1, title="Vin", variety="V", color="Red"))
    db.commit()

    # 2. Signup
    client.post("/signup", json={
        "username": "user1", "email": "u1@test.com",
        "password": "pwd", "date_naissance": "1990-01-01"
    })

    # 3. Login
    login_res = client.post("/login", json={"email": "u1@test.com", "password": "pwd"})
    token = login_res.json()["token"]

    # 4. Favorite
    res = client.post("/favorites", json={"wineId": 1}, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200