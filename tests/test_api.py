from fastapi.testclient import TestClient
# Adaptez l'import selon le nom de votre fichier principal, ex: from main import app
from api import app

client = TestClient(app)

def test_home_status():
    """Vérifie que la route racine renvoie le bon statut[cite: 3]."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "API opérationnelle."}