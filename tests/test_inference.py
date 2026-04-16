import pytest
from datetime import datetime
from src.inference import predict_price
from src.schemas import HousePredictionRequest

def test_predict_price_basic():
    request = HousePredictionRequest(
        year_built=2000,
        bedrooms=3,
        bathrooms=2,
        sqft=1500,
        location="TestCity",
        condition="Good"
    )
    response = predict_price(request)
    assert response.predicted_price > 0
    assert len(response.confidence_interval) == 2
    assert response.prediction_time is not None
