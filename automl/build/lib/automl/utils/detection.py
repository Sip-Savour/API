"""Détection automatique du type de problème ML."""
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from automl.utils.logging import log

def detect_type_model(data) -> str:
    """
    Détecte le type de modèle à appliquer.
    """
    # Extraction y: supporte sklearn Bunch, DataFrame, tuple (X,y), dict
    if hasattr(data, "data") and hasattr(data, "target"):
        y = data.target
    elif isinstance(data, pd.DataFrame):
        if "target" in data.columns:
            y = data["target"]
        else:
            y = data.iloc[:, -1]
    elif isinstance(data, tuple) and len(data) == 2:
        _, y = data
    elif isinstance(data, dict) and "target" in data:
        y = data["target"]
    else:
        raise ValueError(f"Format non supporté: {type(data)}")

    # --- CORRECTIF CRITIQUE POUR PROJET VIN ---
    # On convertit en Pandas Series pour utiliser fillna()
    # On remplace les trous par "Inconnu" et on force le texte
    y = pd.Series(y).fillna("Inconnu").astype(str)
    # ------------------------------------------

    # Normalisation vers array numpy 1D
    y = np.array(y)
    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y.ravel()

    # Première passe via sklearn
    try:
        y_type = type_of_target(y)
    except Exception:
        # Si sklearn plante encore, on force la classification
        y_type = "multiclass"
        
    log("detection", f"Type sklearn détecté: {y_type}")

    # Multi-label
    if y_type in ["multilabel-indicator", "multiclass-multioutput"]:
        return "multi-label-classification"

    # Cible numérique
    # Note: Comme on a forcé .astype(str) plus haut pour le vin, 
    # ce bloc ne s'activera que si 'y' contenait vraiment des chiffres au départ.
    if np.issubdtype(y.dtype, np.number):
        n_unique = len(np.unique(y))
        ratio = n_unique / len(y)
        if n_unique == 2:
            return "binary-classification"
        if n_unique < 20 or ratio < 0.01:
            return "multi-classification"
        return "regression"

    # Cible catégorielle (strings)
    n_unique = len(np.unique(y))
    return "binary-classification" if n_unique == 2 else "multi-classification"
