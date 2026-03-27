import pytest
from unittest.mock import patch
import pandas as pd

# On importe le vrai module depuis le dossier app
import app.predict as predict_module

# Données fictives pour simuler df_meta
mock_df_meta = pd.DataFrame({
    'title': ['Vin Rouge A', 'Vin Blanc B'],
    'description': ['Fruité', 'Sec'],
    'variety': ['Merlot', 'Chardonnay']
})

mock_variety_map = {
    'Merlot': 'Rouge',
    'Chardonnay': 'Blanc'
}

# CORRECTION ICI : On utilise le vrai chemin d'import 'app.predict' pour le patch
@patch('app.predict.SYSTEM_READY', True)
@patch('app.predict.df_meta', mock_df_meta)
@patch('app.predict.variety_map', mock_variety_map)
@patch('app.predict.automl_predict')
def test_fast_predict_success(mock_automl):
    """Teste une prédiction réussie avec un filtre de couleur valide."""
    # Le mock retourne des distances bidons et les indices [0, 1] correspondant au dataframe
    mock_automl.return_value = ([0.1, 0.5], [[0, 1]])

    # On demande un vin avec la contrainte "Rouge"
    results = predict_module.fast_predict("Un vin fruité", color_constraint="Rouge", top_n=5)

    assert isinstance(results, list)
    assert len(results) == 1  # Seul le 'Vin Rouge A' devrait passer le filtre de couleur
    assert results[0]["title"] == "Vin Rouge A"
    assert results[0]["color"] == "Rouge"
    assert results[0]["id"] == 1  # L'index 0 + 1

# CORRECTION ICI AUSSI : 'app.predict'
@patch('app.predict.SYSTEM_READY', False)
def test_fast_predict_system_not_ready():
    """Teste le comportement de l'API si le modèle n'a pas réussi à charger en RAM."""
    results = predict_module.fast_predict("Un vin fruité")
    assert "error" in results
    assert results["error"] == "Modèle non chargé sur le serveur"