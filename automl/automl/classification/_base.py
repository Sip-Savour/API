
from automl.utils.logging import log
from automl.utils.optuna import optuna_search

import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

# Modèles sklearn pertinents pour le Texte/Sparse
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    Perceptron, PassiveAggressiveClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    BaggingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

# Boosters externes
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    CatBoostClassifier = None
    HAS_CATBOOST = False

# =============================================================================
# PARAMÈTRES PAR DÉFAUT (Optimisés pour la rapidité)
# =============================================================================

DEFAULT_PARAMS_STABLE = {
    "LogisticRegression": {"C": 1.0, "solver": "liblinear", "n_jobs": 1},
    "RidgeClassifier": {"alpha": 1.0},
    "SGDClassifier": {"alpha": 0.0001, "loss": "hinge", "penalty": "l2", "n_jobs": 1},
    "LinearSVC": {"C": 1.0, "dual": False}, 
    "MultinomialNB": {"alpha": 1.0}, 
    "BernoulliNB": {"alpha": 1.0},
    "RandomForestClassifier": {"n_estimators": 100, "max_depth": 50, "n_jobs": 1},
    "ExtraTreesClassifier": {"n_estimators": 100, "max_depth": 50, "n_jobs": 1},
    "XGBClassifier": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "n_jobs": 1, "verbosity": 0},
    "LGBMClassifier": {"n_estimators": 100, "learning_rate": 0.1, "n_jobs": 1, "verbosity": -1}
}

DEFAULT_PARAMS_EXPLO = {
    "LogisticRegression": {"C": 1.0, "solver": "lbfgs", "n_jobs": 1},
    "RidgeClassifier": {"alpha": 1.0},
    "SGDClassifier": {"alpha": 0.0001, "n_jobs": 1},
    "Perceptron": {"alpha": 0.0001, "n_jobs": 1},
    "PassiveAggressiveClassifier": {"C": 1.0, "n_jobs": 1},
    "MLPClassifier": {"hidden_layer_sizes": (100,), "alpha": 0.0001, "max_iter": 200},
    "RandomForestClassifier": {"n_estimators": 100, "n_jobs": 1},
    "GradientBoostingClassifier": {"n_estimators": 100, "learning_rate": 0.1},
    "AdaBoostClassifier": {"n_estimators": 50, "learning_rate": 1.0},
    "ExtraTreesClassifier": {"n_estimators": 100, "n_jobs": 1},
    "BaggingClassifier": {"n_estimators": 10, "n_jobs": 1},
    "HistGradientBoostingClassifier": {"max_iter": 100},
    "LinearSVC": {"C": 1.0, "dual": "auto"},
    "MultinomialNB": {"alpha": 1.0},
    "ComplementNB": {"alpha": 1.0},
    "BernoulliNB": {"alpha": 1.0},
    "DummyClassifier": {"strategy": "prior"},
    "XGBClassifier": {"n_estimators": 100, "learning_rate": 0.1, "n_jobs": 1, "verbosity": 0},
    "LGBMClassifier": {"n_estimators": 100, "learning_rate": 0.1, "n_jobs": 1, "verbosity": -1},
    "CatBoostClassifier": {"iterations": 100, "learning_rate": 0.1, "thread_count": 1, "verbose": 0}
}


# =============================================================================
# MODÈLES DISPONIBLES
# =============================================================================

def get_models_stable():
    models = {
        "LogisticRegression": LogisticRegression,
        "RidgeClassifier": RidgeClassifier,
        "SGDClassifier": SGDClassifier,
        "LinearSVC": LinearSVC,             
        "MultinomialNB": MultinomialNB,
        "BernoulliNB": BernoulliNB,         
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
    }
    if HAS_XGB: models["XGBClassifier"] = XGBClassifier
    if HAS_LGBM: models["LGBMClassifier"] = LGBMClassifier
    return models


def get_models_explo():
    models = get_models_stable()
    models.update({
        "Perceptron": Perceptron,
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
        "MLPClassifier": MLPClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "BaggingClassifier": BaggingClassifier,
        "ComplementNB": ComplementNB,
        "DummyClassifier": DummyClassifier,
    })
    if HAS_CATBOOST: models["CatBoostClassifier"] = CatBoostClassifier
    return models


# =============================================================================
# CONFIGURATION MULTI-LABEL
# =============================================================================

NATIVE_MULTILABEL = [
    "RandomForestClassifier", "ExtraTreesClassifier", "MLPClassifier"
]

# --- CORRECTIF ---
# Ces listes sont requises par explo.py, même si on ne les utilise plus ici.
# On les définit vides car on a déjà retiré les modèles incompatibles (QDA, LDA).
MULTILABEL_INCOMPATIBLE_STABLE = []
MULTILABEL_INCOMPATIBLE_EXPLO = []
# -----------------


# =============================================================================
# FONCTION D'ÉVALUATION
# =============================================================================

def evaluate_model(args):
    """Évalue un modèle avec gestion intelligente des Scalers pour NLP."""
    (name, models, default_params, X_train, X_test,
     y_train, y_test, multilabel, iterations, log_name) = args

    gc.collect()

    base_params = default_params.get(name, {}).copy()
    model_class = models[name]

    try:
        best_params, best_score = optuna_search(
            model_class=model_class,
            param_grid={},
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            task_type="classification",
            base_params=base_params,
            n_trials=iterations,
            log_name=log_name
        )

        model = model_class(**best_params)

        if multilabel and name not in NATIVE_MULTILABEL:
            model = OneVsRestClassifier(model, n_jobs=1)

        # GESTION SCALERS
        if name in ["MultinomialNB", "ComplementNB", "BernoulliNB"]:
            estimator = model 
        elif any(kw in name for kw in ["Logistic", "Ridge", "SVC", "SGD", "MLP",
                                        "Perceptron", "PassiveAggressive", "LinearSVC"]):
            estimator = Pipeline([
                ("scaler", MaxAbsScaler()), 
                ("model", model)
            ])
        else:
            estimator = model

        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)

        result = {
            "model": name,
            "best_params": best_params,
            "best_score": float(score)
        }

        log(log_name, f" {name} | Score: {score:.4f}")

        del model, estimator
        gc.collect()

        return result

    except Exception as e:
        log(log_name, f" {name}: {str(e)}")
        return {
            "model": name,
            "best_params": {},
            "best_score": -999.0,
            "error": str(e)
        }
