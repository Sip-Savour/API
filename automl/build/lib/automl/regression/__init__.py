"""Module de régression."""

# Modèles stables vs expérimentaux
from automl.regression.stable import run_regression_models
from automl.regression.explo import run_regression_models_explo

# stable: selection des meilleurs modeles,peu de modeles, rapide | explo: grande quantitée de modeles, plus lent
# API publique du module — limite ce qui est exposé lors d'un `from module import *`
__all__ = ["run_regression_models", "run_regression_models_explo"]