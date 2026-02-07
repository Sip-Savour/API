"""Évaluation et affichage des résultats."""
import pickle
import argparse
from pathlib import Path
from automl._config import RESULTS_DIR


def show_results(mode: str = "stable", top_n: int = 5) -> dict:
    """
    Affiche les résultats d'un run AutoML.
    
    Args:
        mode: "stable" ou "explo"
        top_n: Nombre de modèles à afficher
        
    Returns:
        Dict des résultats
    """
    path = RESULTS_DIR / f"results_{mode}.pkl"
    
    if not path.exists():
        print(f"Fichier non trouvé: {path}")
        return {}

    with open(path, "rb") as f:
        results = pickle.load(f)

    for dataset, info in results.items():
        print(f" \n {'='*50}")
        print(f" Dataset: {dataset}")
        # Fallback sur 'type' pour rétrocompatibilité avec anciens formats
        print(f" Type: {info.get('task_type', info.get('type', 'N/A'))}")
        print(f" \n Top {top_n} modèles:")
        
        models = info.get("best_models", [])
        for i, m in enumerate(models[:top_n], 1):
            score = m.get("best_score", 0)
            print(f" {i}. {m['model']:30s} | Score: {score:.4f}")

    return results


def eval(mode: str = "stable"):
    """Alias pour show_results."""
    return show_results(mode)


def main():
    """Point d'entrée CLI: automl-results"""
    parser = argparse.ArgumentParser(description="Afficher les résultats AutoML")
    parser.add_argument("--mode", choices=["stable", "explo"], default="stable")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--dir", default=".", help="Répertoire des résultats")
    args = parser.parse_args()

    # Override du répertoire par défaut pour permettre l'exécution depuis n'importe où
    global RESULTS_DIR
    RESULTS_DIR = Path(args.dir) / "results"
    
    show_results(args.mode, args.top)


if __name__ == "__main__":
    main()