"""
AutoML - Pipeline automatique de Machine Learning
==================================================
"""

# API publique exposée au niveau du package pour simplifier les imports utilisateur
from automl.core.trainer import fit, fit_multiple
from automl.core.evaluator import eval, show_results
from automl.core.predictor import predict

__version__ = "0.2.0"

# Contrôle les symboles exportés lors de "from automl import *"
__all__ = [
    # Entraînement
    "fit",
    "fit_multiple",
    # Prédiction
    "predict",
    # Évaluation et affichage
    "eval",
    "show_results"
]