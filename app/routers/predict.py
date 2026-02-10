from fastapi import APIRouter, HTTPException
import sys
import os
from models import WineRequest, WineResponse, BottleInfo
from predict import fast_predict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
router = APIRouter()

@router.post("/predict", response_model=WineResponse, tags=["AI"])
def predict_wine(req: WineRequest):
    try:
        bouteille_trouvee = fast_predict(req.features, req.color)

        if bouteille_trouvee is None:
            return WineResponse(bottle=None)

        info_bouteille = BottleInfo(
            title=str(bouteille_trouvee.get('title', 'Inconnu')),
            description=str(bouteille_trouvee.get('description', '')),
            variety=str(bouteille_trouvee.get('variety', 'Inconnu')),
        )

        return WineResponse(
            bottle=info_bouteille
        )

    except Exception as e:
        print(f"Erreur API : {e}")
        raise HTTPException(status_code=500, detail=str(e))