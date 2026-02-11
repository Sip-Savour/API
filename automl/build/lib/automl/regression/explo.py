"""
Régression - Mode Exploratoire
Exploration complète des modèles de régression.
"""
import gc
import warnings
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from automl.utils.logging import log
from automl.regression._base import (
    get_models_explo,
    PARAM_GRIDS_STABLE,  
    evaluate_model
)

LOG_NAME = "regression_explo"

# Modèles à complexité O(n2) ou plus en mémoire/temps → exclus sur gros volumes
HEAVY_MODELS = ["GaussianProcessRegressor", "RadiusNeighborsRegressor", "NuSVR", "SVR", "KNeighborsRegressor"]


def run_regression_models_explo(
    X,
    y,
    n_jobs: int = 2,
    iterations: int = 10,
    sample_ratio: float = 1.0
):
    """
    Exploration complète des modèles de régression.
    Gère automatiquement le sous-échantillonnage et l'exclusion
    des modèles lourds selon la taille du dataset.
    """
    warnings.filterwarnings("ignore")
    
    # 1. Utilisation du getter
    models = get_models_explo()
    
    log(LOG_NAME, f"Régression explo: {len(models)} modèles disponibles")

    # Seuil empirique : au-delà, les modèles O(n2) deviennent impraticables
    if X.shape[0] > 10_000:
        for m in HEAVY_MODELS:
            if m in models:
                del models[m]
                log(LOG_NAME, f" Dataset volumineux: {m} exclu")

    # Sous-échantillonnage optionnel pour accélérer l'exploration initiale
    if 0 < sample_ratio < 1.0:
        try:
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=sample_ratio, random_state=42
            )
            log(LOG_NAME, f"Sous-échantillonnage: {sample_ratio*100:.1f}%")
        except Exception:
            # Échec du split, utiliser le dataset complet
            X_sub, y_sub = X, y
    else:
        X_sub, y_sub = X, y

    # Split Train/Test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )
    except Exception as e:
        log(LOG_NAME, f" Erreur split: {e}")
        return []

    # Préparation Tâches
    tasks = []
    for name, model_class in models.items():
        param_grid = PARAM_GRIDS_STABLE.get(name, {})
        tasks.append((
            model_class,
            param_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            iterations,
            LOG_NAME
        ))

    results = []

    # Adaptation du parallélisme selon la taille des données pour éviter la surcharge mémoire
    effective_n_jobs = 1 if len(X_sub) < 1000 else n_jobs
    if X.shape[0] > 50_000 and effective_n_jobs > 2:
        effective_n_jobs = 2
        log(LOG_NAME, f"Réduction n_jobs à {effective_n_jobs}")

    # Exécution
    try:
        # spawn évite les problèmes de fork avec numpy/sklearn 
        ctx = mp.get_context("spawn")
        with ctx.Pool(effective_n_jobs) as pool:
            for res in pool.imap_unordered(evaluate_model, tasks):
                if res is not None:
                    results.append(res)
            pool.close()
            pool.join()
    except Exception as e:
        log(LOG_NAME, f"Erreur multiprocessing ({e}). Mode séquentiel.")
        for args in tasks:
            res = evaluate_model(args)
            if res is not None:
                results.append(res)
    finally:
        # Libération mémoire 
        gc.collect()

    # Tri des résultats par score décroissant
    results.sort(key=lambda x: x.get("best_score", -np.inf), reverse=True)

    valid_results = [
        r for r in results
        if np.isfinite(r.get("best_score", -np.inf)) 
        and r.get("best_score", -np.inf) > -10.0
    ]

    log(LOG_NAME, f"{len(valid_results)}/{len(models)} modèles valides")

    return valid_results