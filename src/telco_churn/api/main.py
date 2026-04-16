from fastapi import FastAPI

from src.telco_churn.api.prediction import MODEL, predict_customer
from src.telco_churn.api.schemas import CustomerInput, HealthResponse, PredictionResponse

app = FastAPI(title="Telco Churn Prediction API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=MODEL is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    prediction = predict_customer(customer.model_dump())
    return PredictionResponse(**prediction)
