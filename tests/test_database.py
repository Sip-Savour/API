import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, User, Wine, Favorite

# Création d'une BDD en mémoire pour les tests[cite: 4]
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Crée une nouvelle session de base de données pour un test[cite: 4]."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_create_user(db_session):
    """Teste l'insertion d'un utilisateur dans la table users[cite: 4]."""
    new_user = User(
        username="testuser",
        email="test@test.com",
        password_hash="hashed_pw",
        date_naissance=date(1995, 5, 5)
    )
    db_session.add(new_user)
    db_session.commit()

    fetched_user = db_session.query(User).filter(User.username == "testuser").first()
    assert fetched_user is not None
    assert fetched_user.email == "test@test.com"


def test_create_wine_and_favorite(db_session):
    """Teste l'insertion d'un vin et la relation avec les favoris[cite: 4]."""
    user = User(username="favuser", email="fav@test.com", password_hash="pw", date_naissance=date(1990, 1, 1))
    wine = Wine(title="Chateau Test", description="Très bon", variety="Merlot", color="Rouge")

    db_session.add_all([user, wine])
    db_session.commit()

    favorite = Favorite(user_id=user.id, wine_id=wine.id)
    db_session.add(favorite)
    db_session.commit()

    fetched_fav = db_session.query(Favorite).first()
    assert fetched_fav.user_id == user.id
    assert fetched_fav.wine.title == "Chateau Test"