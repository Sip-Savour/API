from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ================= CONFIGURATION =================
SQLALCHEMY_DATABASE_URL = "sqlite:///../sommelier.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ================= TABLES =================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String) 
    
    favorites = relationship("Favorite", back_populates="user")

class Wine(Base):
    __tablename__ = "wines"
    id = Column(Integer, primary_key=True, index=True) 
    
    # Infos de base
    title = Column(String, index=True)
    description = Column(Text)
    variety = Column(String, index=True)
    
    # Infos détaillées
    winery = Column(String, nullable=True)
    province = Column(String, nullable=True)
    country = Column(String, nullable=True)

    favorited_by = relationship("Favorite", back_populates="wine")

class Favorite(Base):
    __tablename__ = "favorites"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    wine_id = Column(Integer, ForeignKey("wines.id"))
    
    user = relationship("User", back_populates="favorites")
    wine = relationship("Wine", back_populates="favorited_by")


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Base de données initialisée (sommelier.db).")

if __name__ == "__main__":
    init_db()
