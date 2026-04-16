import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.schemas import HousePredictionRequest

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "year_built": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1500,
        "location": "TestCity",
        "condition": "Good"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "confidence_interval" in data
