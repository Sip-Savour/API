"""Chargement des données."""
import os
import numpy as np
import pandas as pd
from scipy import sparse


"""Chargement des données avec gestion robuste Texte vs Chiffres."""
import os
import numpy as np
import pandas as pd
from scipy import sparse


def load_data(path: str) -> pd.DataFrame:
    """Charge un dataset d'entraînement (.data + .solution) en un seul DataFrame."""
    solution_path = f"{path}.solution"
    data_path = f"{path}.data"
    
    # 1. Chargement de la solution (y)
    # On force le type string pour éviter les bugs si y contient des entiers et des NaN
    try:
        y_df = pd.read_csv(
            solution_path, header=None, engine="python", on_bad_lines="skip"
        )
        # On prend la 1ere colonne, on bouche les trous par "Inconnu", on force le texte
        y = y_df.iloc[:, 0].fillna("Inconnu").astype(str).rename("target")
    except Exception as e:
        print(f"Warning: Problème lecture solution ({e}), création target vide.")
        y = pd.Series(name="target", dtype=str)

    # 2. Chargement des features (X)
    X = _read_features(data_path)
    
    # 3. Alignement de sécurité (Troncature si décalage)
    min_len = min(len(X), len(y))
    if len(X) != len(y):
        print(f"⚠️ Warning: Taille X ({len(X)}) != Taille y ({len(y)}). Troncature à {min_len}.")
        X = X.iloc[:min_len].reset_index(drop=True)
        y = y.iloc[:min_len].reset_index(drop=True)
    
    df = pd.concat([X, y], axis=1)
    return df


def load_test_data(path: str) -> pd.DataFrame:
    """Charge un dataset de test (features uniquement)."""
    data_path = f"{path}" if path.endswith(".data") else f"{path}.data"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
        
    return _read_features(data_path)


def _read_features(data_path: str) -> pd.DataFrame:
    """
    Lecture intelligente des features.
    - Essaie de lire comme des chiffres (Dense S1).
    - Détecte le format Sparse (SVMLight).
    - Si échec, lit comme du TEXTE BRUT (Une seule colonne 'description').
    """
    if not os.path.exists(data_path):
        return pd.DataFrame()

    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        # Lecture des premières lignes pour détection
        sample_lines = [line.strip() for line in f if line.strip()]
    
    if not sample_lines:
        return pd.DataFrame()

    # CAS 1 : Format Sparse (SVMLight "idx:val ...")
    if any(":" in line for line in sample_lines[:20]):
        return _read_sparse(sample_lines)

    # CAS 2 : Format Dense Numérique (S1 Classique) ou Texte
    try:
        # On essaie de charger avec numpy (très strict, ne veut que des chiffres)
        X_arr = np.loadtxt(data_path)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        # Si ça passe, on retourne un DataFrame numérique
        return pd.DataFrame(X_arr)
        
    except ValueError:
        # CAS 3 : FALLBACK TEXTE (C'est votre cas "Projet Vin")
        # numpy a échoué car il y a des mots ("tannin", "red"...).
        # On lit chaque ligne comme une description unique.
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        return pd.DataFrame(lines, columns=['description'])


def _read_sparse(lines):
    """Helper pour lire le format sparse sans dépendre de sklearn."""
    entries = []
    max_col = 0
    for row_idx, line in enumerate(lines):
        for entry in line.split():
            if ":" in entry:
                try:
                    c, v = entry.split(":")
                    c, v = int(c), float(v)
                    entries.append((row_idx, c, v))
                    max_col = max(max_col, c)
                except ValueError: continue
    
    if not entries:
        return pd.DataFrame()
        
    rows, cols, vals = zip(*entries)
    n_rows = len(lines)
    n_cols = max_col + 1
    
    # Construction matrice
    X_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    
    # Conversion DataFrame (Dense ou Sparse selon taille)
    if n_cols <= 5000:
        return pd.DataFrame(X_sparse.toarray())
    else:
        return pd.DataFrame.sparse.from_spmatrix(X_sparse)

import pandas as pd
import numpy as np
import os
from scipy import sparse

def importer(datafile: str, max_dense_cols: int = 5000):
    """
    Importe les données (.data) et la solution (.solution).
    VERSION BLINDÉE CONTRE LES ERREURS NONETYPE
    """
    data_path = f"{datafile}.data"
    solution_path = f"{datafile}.solution"

    if not os.path.exists(data_path) or not os.path.exists(solution_path):
        raise FileNotFoundError(f"Fichier manquant: {data_path} ou {solution_path}")

    # --- 1. CHARGEMENT DE LA SOLUTION (y) ---
    try:
        # On lit le fichier sans entête
        y_df = pd.read_csv(solution_path, header=None, engine="python")
        y = y_df.iloc[:, 0]
        
        # === ÉTAPE CRUCIALE DE DÉSINFECTION ===
        # On remplace les trous (NaN/None) par une chaine vide ou "Inconnu"
        y = y.fillna("Inconnu")
        # On FORCE le type String (texte), pour que scikit-learn ne plante plus
        y = y.astype(str)
        # ======================================
        
    except Exception as e:
        print(f"Erreur lecture solution: {e}")
        raise

    # --- 2. CHARGEMENT DES DONNÉES (X) ---
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        # On lit toutes les lignes non vides
        sample_lines = [line.strip() for line in f if line.strip()]
    
    if not sample_lines:
        return pd.DataFrame(), y

    # Vérification d'alignement (Optionnel mais recommandé)
    # Si X et y n'ont pas la même taille, on coupe au plus court pour éviter un crash plus loin
    min_len = min(len(sample_lines), len(y))
    if len(sample_lines) != len(y):
        print(f"⚠️ ATTENTION : Décalage détecté ! X={len(sample_lines)}, y={len(y)}")
        print(f"   -> On tronque les données à {min_len} lignes.")
        sample_lines = sample_lines[:min_len]
        y = y.iloc[:min_len]

    # A. DÉTECTION SVMLight (Sparse)
    is_svmlight = any(":" in line for line in sample_lines[:50])

    if is_svmlight:
        entries = []
        max_col = 0
        for row_idx, line in enumerate(sample_lines):
            for entry in line.split():
                if ":" in entry:
                    try:
                        c, v = entry.split(":")
                        c, v = int(c), float(v)
                        entries.append((row_idx, c, v))
                        max_col = max(max_col, c)
                    except ValueError: continue
        
        n_rows, n_cols = len(sample_lines), max_col + 1
        if not entries: return pd.DataFrame(), y
        
        rows, cols, vals = zip(*entries)
        X_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        
        if n_cols <= max_dense_cols:
            return pd.DataFrame(X_sparse.toarray()), y
        else:
            return pd.DataFrame.sparse.from_spmatrix(X_sparse), y

    # B. DÉTECTION NUMÉRIQUE OU TEXTE
    try:
        # Tentative S1 : Lire comme des chiffres
        X_arr = np.loadtxt(data_path)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        # Si on a tronqué y, il faut tronquer X aussi si c'est un array numpy
        if len(X_arr) > len(y):
             X_arr = X_arr[:len(y)]
        return pd.DataFrame(X_arr), y

    except ValueError:
        # FALLBACK PROJET VIN : C'est du texte !
        return pd.DataFrame(sample_lines, columns=['description']), y
