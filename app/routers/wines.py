from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func
from database import SessionLocal, Wine, User
from models import BottleInfo
import json
from pathlib import Path
import random
import datetime
from typing import Optional
from fastapi import Query

from routers.auth import get_current_user

router = APIRouter()

# ==========================================================
# PRÉPARATION DU MAPPING DES COULEURS
# ==========================================================
# On charge le fichier JSON pour déduire la couleur à partir du cépage (variety)
BASE_DIR = Path(__file__).resolve().parent.parent
COLORS_FILE = BASE_DIR / "data" / "wine_colors.json"

variety_map = {}
if COLORS_FILE.exists():
    with open(COLORS_FILE, "r", encoding="utf-8") as f:
        variety_map = json.load(f)


# ==========================================================
# DÉPENDANCE DE BASE DE DONNÉES
# ==========================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================================
# LA ROUTE : GET /wines/random (Shake to Discover)
# ==========================================================
@router.get("/wines/random", tags=["Recommendations"])
def get_random_wines(db: Session = Depends(get_db)):
    """Retourne 5 vins complètement aléatoires pour la fonctionnalité Shake to Discover."""

    # func.random() mélange la base de données et limit(5) prend les 5 premiers.
    # C'est la méthode la plus robuste pour éviter les erreurs liées aux IDs manquants !
    random_wines = db.query(Wine).order_by(func.random()).limit(5).all()

    if not random_wines:
        raise HTTPException(status_code=404, detail="Aucun vin disponible dans la base.")

    # On formate la réponse sous forme de liste pour Android
    result = []
    for wine in random_wines:
        result.append({
            "id": wine.id,
            "title": wine.title,
            "description": wine.description or "",
            "variety": wine.variety or "Inconnu",
            "color": wine.color
        })

    return result
# ==========================================================
# LA ROUTE : GET /vins/{wine_id}
# ==========================================================
@router.get("/wines/{wine_id}", response_model=BottleInfo, tags=["Vins"])
def get_vin_by_id(wine_id: int, db: Session = Depends(get_db)):
    # 1. On interroge la base de données
    vin = db.query(Wine).filter(Wine.id == wine_id).first()

    # 2. Si le vin n'existe pas, on renvoie une erreur 404
    if not vin:
        raise HTTPException(status_code=404, detail=f"Aucun vin trouvé avec l'ID {wine_id}")

    # 3. On déduit la couleur grâce au cépage
    couleur_vin = variety_map.get(vin.variety, "Inconnue")

    # 4. On renvoie l'objet formaté
    return BottleInfo(
        id=vin.id,
        title=vin.title,
        description=vin.description or "",
        variety=vin.variety or "Inconnu",
        color=couleur_vin
    )


@router.get("/wines/weekly", tags=["Recommendations"])
def get_weekly_recommendation(
        color: Optional[str] = Query(None, description="Couleur préférée (Red, White, Rose)"),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Retourne une recommandation unique par semaine, filtrée par la préférence locale"""

    # 1. Obtenir l'année et le numéro de la semaine actuelle
    year, week, _ = datetime.date.today().isocalendar()

    # 2. Préparer la recherche
    query = db.query(Wine)
    if color:
        # Si Android envoie une préférence, on filtre par cette couleur
        query = query.filter(Wine.color.ilike(f"%{color}%"))

    wines = query.all()

    # Si la couleur demandée ne donne rien (ou base vide), on prend tout
    if not wines:
        wines = db.query(Wine).all()
        if not wines:
            raise HTTPException(status_code=404, detail="Aucun vin disponible dans la base.")

    # 3. La graine de hasard (Seed) indestructible
    # On inclut la couleur dans la graine. Ainsi, si l'utilisateur change sa préférence
    # en cours de semaine, il obtiendra immédiatement un nouveau vin stable !
    random.seed(f"{current_user.id}-{year}-{week}-{color}")
    recommended_wine = random.choice(wines)

    # On réinitialise le hasard du serveur
    random.seed()

    return {
        "id": recommended_wine.id,
        "title": recommended_wine.title,
        "description": recommended_wine.description,
        "variety": recommended_wine.variety,
        "color": recommended_wine.color
    }


