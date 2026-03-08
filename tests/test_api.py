from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_trade_endpoint_success():
    payload = {
        "current_price": 0.5,
        "forecasted_demand": 2.5,
        "battery_level": 25.0,
        "account_balance": 100.0,
    }
    response = client.post("/api/v1/trade", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "action" in data
    assert "confidence" in data
    assert data["action"] in [0, 1, 2]
    assert 0.0 <= data["confidence"] <= 1.0


def test_trade_endpoint_invalid_payload():
    payload = {
        "current_price": -0.5,  # Invalid, should be >= 0
        "forecasted_demand": 2.5,
        "battery_level": 25.0,
        "account_balance": 100.0,
    }
    response = client.post("/api/v1/trade", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_trade_endpoint_missing_fields():
    payload = {"current_price": 0.5}
    response = client.post("/api/v1/trade", json=payload)
    assert response.status_code == 422
