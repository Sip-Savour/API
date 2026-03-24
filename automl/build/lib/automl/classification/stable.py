"""
Classification - Mode Stable (Optimisé NLP / Gros Volumes)
"""
import gc
import warnings
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from automl.utils.logging import log
# On n'importe plus les listes d'exclusion car _base.py est déjà filtré
from automl.classification._base import (
    get_models_stable, 
    DEFAULT_PARAMS_STABLE,
    evaluate_model
)

LOG_NAME = "classification_stable"

def run_classification_models(
    X, y, 
    n_jobs=1,  # n_jobs=1 par défaut pour éviter les conflits RAM sur gros dataset
    model_type="binary-classification", 
    iterations=50, 
    sample_ratio=0.3
):
    """
    Teste les modèles de classification en mode stable.
    Optimisé pour les datasets larges (150k lignes) via sous-échantillonnage intelligent.
    """
    warnings.filterwarnings("ignore")
    
    # Récupération des modèles (déjà filtrés pour être rapides dans _base.py)
    models = get_models_stable()
    
    multilabel = model_type in ["multi-label-classification", "multi-output"]

    log(LOG_NAME, f"Classification stable: {len(models)} modèles, type={model_type}")

    # --- SOUS-ÉCHANTILLONNAGE POUR L'OPTIMISATION (OPTUNA) ---
    # Pour 150k lignes, on ne veut pas que Optuna teste 50 itérations sur tout le dataset.
    # On optimise les hyperparamètres sur un échantillon (ex: 30% ou max 50k lignes), 
    # puis le meilleur modèle sera réentraîné sur tout le dataset à la fin.
    if 0 < sample_ratio < 1.0:
        # Sécurité : on plafonne à 50 000 lignes pour la recherche d'hyperparamètres
        # pour que ça tourne en un temps raisonnable.
        max_samples = 50000
        n_samples = len(X)
        
        # Si 30% du dataset dépasse 50k, on réduit le ratio
        if int(n_samples * sample_ratio) > max_samples:
            real_ratio = max_samples / n_samples
            log(LOG_NAME, f"Dataset très large ({n_samples}). Limitation de l'échantillon Optuna à {max_samples} lignes.")
        else:
            real_ratio = sample_ratio

        try:
            stratify = y if model_type != "regression" else None
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=real_ratio, random_state=42, stratify=stratify
            )
            log(LOG_NAME, f"Sous-échantillonnage: {real_ratio*100:.1f}% ({len(X_sub)} samples)")
        except:
            X_sub, _, y_sub, _ = train_test_split(X, y, train_size=real_ratio, random_state=42)
            log(LOG_NAME, f"Sous-échantillonnage (sans stratify): {real_ratio*100:.1f}%")
    else:
        X_sub, y_sub = X, y

    # Split Train/Test pour l'évaluation interne
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )
    except Exception as e:
        log(LOG_NAME, f"Erreur split train/test: {e}")
        return []

    model_names = list(models.keys())

    # Préparation des tâches pour le multiprocessing
    tasks = [
        (name, models, DEFAULT_PARAMS_STABLE, X_train, X_test, 
         y_train, y_test, multilabel, iterations, LOG_NAME)
        for name in model_names
    ]

    results = []

    # --- EXÉCUTION (MULTIPROCESSING) ---
    try:
        # On utilise 'spawn' ou 'forkserver' pour éviter les deadlocks avec les libs C (numpy/pandas)
        # Si X_sub est petit, on reste en séquentiel pour éviter l'overhead
        effective_n_jobs = 1 if len(X_sub) < 1000 else n_jobs
        
        if effective_n_jobs > 1:
            # Context 'spawn' est le plus sûr (compatible Windows/Linux/MacOS)
            ctx = mp.get_context("spawn")
            with ctx.Pool(effective_n_jobs) as pool:
                for res in pool.imap_unordered(evaluate_model, tasks):
                    if res:
                        results.append(res)
                        # Petit nettoyage mémoire progressif
                        gc.collect()
                pool.close()
                pool.join()
        else:
            # Mode séquentiel forcé
            for args in tasks:
                res = evaluate_model(args)
                if res:
                    results.append(res)
                gc.collect()

    except Exception as e:
        log(LOG_NAME, f"Erreur ou interruption multiprocessing ({e}). Passage en séquentiel.")
        # Fallback séquentiel en cas de pépin
        results = [] # On recommence
        for args in tasks:
            try:
                res = evaluate_model(args)
                if res:
                    results.append(res)
            except Exception as e_seq:
                log(LOG_NAME, f"Erreur modèle {args[0]}: {e_seq}")
            gc.collect()
    finally:
        gc.collect()

    # Tri des résultats
    results.sort(key=lambda x: x["best_score"], reverse=True)
    valid_results = [r for r in results if r["best_score"] > -1e10]

    if not valid_results:
        log(LOG_NAME, "Aucun modèle valide trouvé")
    else:
        log(LOG_NAME, f"{len(valid_results)} modèles évalués avec succès")

    return valid_results
