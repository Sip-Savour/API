import pytest
# On tente d'importer votre librairie pour vérifier qu'elle est bien installée
try:
    import automl
    AUTOML_INSTALLED = True
except ImportError:
    AUTOML_INSTALLED = False

def test_basic_math():
    """Vérifie juste que le runner fonctionne."""
    assert 1 + 1 == 2

def test_automl_import():
    """Vérifie que la librairie locale a bien été installée par le CI."""
    assert AUTOML_INSTALLED is True, "La librairie 'automl' n'a pas été trouvée !"
