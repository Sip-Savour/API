"""Module de classification."""
from automl.classification.stable import run_classification_models
from automl.classification.explo import run_classification_models_explo

# stable: selection des meilleurs modeles,peu de modeles, rapide | explo: grande quantitée de modeles, plus lent
# API publique du module — limite ce qui est exposé lors d'un `from module import *`
__all__ = ["run_classification_models", "run_classification_models_explo"]