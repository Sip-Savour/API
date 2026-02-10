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
        bouteilles_trouvees = fast_predict(req.features, req.color, top_n=5)

        results = []
        if bouteilles_trouvees:
            for b in bouteilles_trouvees:
                info = BottleInfo(
                    title=str(b.get('title', 'Inconnu')),
                    description=str(b.get('description', '')),
                    variety=str(b.get('variety', 'Inconnu')),
                    price=float(b.get('price', 0.0)) if b.get('price') == b.get('price') else 0.0
                )
                results.append(info)

        return WineResponse(
            bottles=results 
        )
    except Exception as e:
        print(f"Erreur API : {e}")
        raise HTTPException(status_code=500, detail=str(e))