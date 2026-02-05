"""
Régression - Mode Stable
Ensemble de modèles fiables avec optimisation Optuna.
"""
import gc
import warnings
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from automl.utils.logging import log
from automl.regression._base import (
    get_models_stable,
    PARAM_GRIDS_STABLE,
    evaluate_model
)

LOG_NAME = "regression_stable"


def run_regression_models(
    X, y,
    n_jobs=2,
    iterations=50,
    sample_ratio=0.3
):
    """
    Teste les modèles de régression en mode stable.
    """
    warnings.filterwarnings("ignore")
    

    models = get_models_stable()
    
    log(LOG_NAME, f"Régression stable: {len(models)} modèles")

    # Sous-échantillonnage
    if 0 < sample_ratio < 1.0:
        try:
            X_sub, _, y_sub, _ = train_test_split(X, y, train_size=sample_ratio, random_state=42)
            log(LOG_NAME, f"Sous-échantillonnage: {sample_ratio*100:.0f}% ({X_sub.shape[0]} samples)")
        except:
             X_sub, y_sub = X, y
    else:
        X_sub, y_sub = X, y

    # Split Train/Test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )
    except Exception as e:
        log(LOG_NAME, f"Erreur split: {e}")
        return []

    # Préparation des tâches pour etre évaluées en parallèle
    tasks = []
    for name, model_class in models.items():
        param_grid = PARAM_GRIDS_STABLE.get(name, {})
        tasks.append((
            model_class, 
            param_grid,
            X_train, y_train, X_test, y_test,
            iterations, LOG_NAME
        ))

    results = []

    # Exécution parallèle
    try:
        effective_n_jobs = 1 if len(X_sub) < 1000 else n_jobs
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
        gc.collect()

    # Tri
    results.sort(key=lambda x: x["best_score"], reverse=True)

    if results:
        log(LOG_NAME, f"{len(results)} modèles évalués")
        log(LOG_NAME, f"Meilleur: {results[0]['model']} (R²={results[0]['best_score']:.4f})")
    else:
        log(LOG_NAME, "Aucun modèle valide")
    return results