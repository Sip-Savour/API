"""Configuration globale du package."""
import os
from pathlib import Path

# Répertoires par défaut (peuvent être surchargés via variables d'environnement ou config)
BASE_DIR = Path.cwd()
LOG_DIR = BASE_DIR / "automl/logs"
CHECKPOINT_DIR = BASE_DIR / "automl/checkpoints"
RESULTS_DIR = BASE_DIR / "automl/results"

# Création à l'import pour garantir leur existence avant toute opération I/O
for d in [LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Paramètres par défaut
DEFAULT_MODE = "stable"           # "stable" = modèles éprouvés, "experimental" = tous les modèles
DEFAULT_ITERATIONS = 50
DEFAULT_SAMPLE_RATIO = 0.3        # Fraction des données utilisée pour la recherche d'hyperparamètres
DEFAULT_N_JOBS = 2                # Parallélisation conservative pour éviter la surcharge mémoire
AUTOSAVE_INTERVAL = 600           # secondes