from fastapi.testclient import TestClient

from src.telco_churn.api import prediction
from src.telco_churn.api.main import app


def sample_customer_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 1080.0,
    }


def test_predict_returns_business_risk_response():
    client = TestClient(app)

    response = client.post("/predict", json=sample_customer_payload())

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "churn_probability",
        "risk_segment",
        "top_churn_drivers",
        "recommendation",
    }
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_segment"] in {"Low risk", "Moderate risk", "High risk"}
    assert len(data["top_churn_drivers"]) <= 2
    assert data["recommendation"]
    assert "churn_prediction" not in data


def test_predict_uses_risk_based_fallback_when_shap_is_unavailable(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(prediction, "shap", None)

    response = client.post("/predict", json=sample_customer_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["top_churn_drivers"] == []
    assert data["recommendation"] == [
        prediction.FALLBACK_RECOMMENDATIONS[data["risk_segment"]]
    ]
