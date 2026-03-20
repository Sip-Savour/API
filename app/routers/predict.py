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
        # Appel à l'IA
        bouteilles_trouvees = fast_predict(req.features, req.color)

        # ==========================================================
        # 1. LA CORRECTION EST ICI : VÉRIFICATION DE L'ERREUR IA
        # ==========================================================
        # Si le retour est un dictionnaire contenant la clé "error"
        if isinstance(bouteilles_trouvees, dict) and "error" in bouteilles_trouvees:
            # On déclenche l'exception proprement pour qu'Android reçoive le message
            raise ValueError(bouteilles_trouvees["error"])

        # 2. TRAITEMENT NORMAL (Si c'est bien une liste de bouteilles)
        results = []
        if bouteilles_trouvees and isinstance(bouteilles_trouvees, list):
            for b in bouteilles_trouvees:
                info = BottleInfo(
                    title=str(b.get('title', 'Inconnu')),
                    description=str(b.get('description', '')),
                    variety=str(b.get('variety', 'Inconnu')),
                    color=str(b.get('color', 'Inconnue'))
                )
                results.append(info)

        return WineResponse(
            bottle=results 
        )
        
    except ValueError as ve:
        # Erreur "métier" (ex: Aucun vin trouvé) -> On renvoie une erreur 400 (Bad Request / Not Found)
        print(f"Information IA : {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        # Vraie erreur de serveur (Crash) -> On renvoie 500
        print(f"Erreur Serveur API : {e}")
        raise HTTPException(status_code=500, detail=str(e))