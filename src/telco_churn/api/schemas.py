from pydantic import BaseModel, ConfigDict


class CustomerInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_segment: str
    top_churn_drivers: list[str]
    recommendation: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
