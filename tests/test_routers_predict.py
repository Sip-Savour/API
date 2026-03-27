import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api import app # Import court

client = TestClient(app)

# Patch sans 'app.'
@patch("routers.predict.fast_predict")
def test_predict_success(mock_fast):
    mock_fast.return_value = [{"id": 1, "title": "Wine", "description": "D", "variety": "V", "color": "R"}]
    response = client.post("/predict", json={"features": "fruité", "color": "Rouge"})
    assert response.status_code == 200
    assert len(response.json()["bottle"]) == 1

@patch("routers.predict.fast_predict")
def test_predict_ai_error(mock_fast):
    mock_fast.return_value = {"error": "IA Off"}
    response = client.post("/predict", json={"features": "fruité"})
    assert response.status_code == 400