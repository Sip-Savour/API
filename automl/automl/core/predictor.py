"""Module de prédiction sur de nouvelles données."""
import pickle
from automl._config import RESULTS_DIR
from automl.utils.io import load_test_data
from automl.utils.logging import log

def predict(data_path: str):
    """
    Charge le meilleur modèle entraîné et prédit sur un nouveau dataset.
    
    Args:
        data_path: Chemin vers le fichier .data (sans extension ou avec)
        
    Returns:
        list: Les prédictions
    """
    model_path = RESULTS_DIR / "best_model.pkl"
    
    if not model_path.exists():
        print("Aucun modèle entraîné trouvé. Veuillez lancer fit() d'abord.")
        return []

    log("predict", f"Démarrage prédiction sur : {data_path}")

    # Chargement du modèle
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
        
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    task_type = pipeline["task_type"]
    
    log("predict", f"   Modèle chargé : {type(model).__name__} ({task_type})")

    #Chargement des données (Features uniquement)
    try:
        X_raw = load_test_data(data_path)
    except Exception as e:
        print(f"Erreur chargement données : {e}")
        return []

    #Preprocessing (Transform uniquement)
    try:
        X_clean = preprocessor.transform(X_raw)
        log("predict", f"   Données nettoyées. Shape: {X_clean.shape}")
    except Exception as e:
        print(f"Erreur preprocessing : {e}")
        return []

    #Prédiction
    try:
        predictions = model.predict(X_clean)
        
        # Formatage de sortie (convertir ndarray en list)
        preds_list = predictions.tolist()
        
        log("predict", f"{len(preds_list)} prédictions générées.")
        return preds_list
        
    except Exception as e:
        print(f"Erreur prédiction : {e}")
        return []