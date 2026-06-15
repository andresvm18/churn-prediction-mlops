from typing import Literal
from pydantic import BaseModel, Field


# Customer data model for telecom churn prediction
class CustomerData(BaseModel):
    # Basic demographics
    gender: Literal["Male", "Female"]  # Customer's gender
    senior_citizen: Literal[0, 1]  # 1 = senior citizen, 0 = not senior
    partner: Literal["Yes", "No"]  # Has a partner or not
    dependents: Literal["Yes", "No"]  # Has dependents or not

    # Time as customer (months)
    tenure_months: int = Field(ge=0, le=72)  # Between 0 and 72 months

    # Phone service
    phone_service: Literal["Yes", "No"]  # Has phone service or not
    multiple_lines: Literal["Yes", "No", "No phone service"]  # Multiple phone lines

    # Internet service
    internet_service: Literal["DSL", "Fiber optic", "No"]  # Type of internet

    # Security/backup services (depend on internet service)
    online_security: Literal[
        "Yes", "No", "No internet service"
    ]  # Online security add-on
    online_backup: Literal["Yes", "No", "No internet service"]  # Online backup add-on
    device_protection: Literal[
        "Yes", "No", "No internet service"
    ]  # Device protection add-on
    tech_support: Literal["Yes", "No", "No internet service"]  # Tech support add-on

    # Streaming services
    streaming_tv: Literal["Yes", "No", "No internet service"]  # TV streaming add-on
    streaming_movies: Literal[
        "Yes", "No", "No internet service"
    ]  # Movies streaming add-on

    # Contract and billing
    contract: Literal["Month-to-month", "One year", "Two year"]  # Contract duration
    paperless_billing: Literal["Yes", "No"]  # Paperless billing enabled

    # Payment method
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    # Monthly and total charges
    monthly_charges: float = Field(ge=0.0, le=200.0)  # Monthly charge amount (USD)
    total_charges: float = Field(ge=0.0)  # Total lifetime charges (USD)

    # Example valid data for reference
    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure_months": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 95.5,
                "total_charges": 1146.0,
            }
        }
    }


class BatchPredictionRow(BaseModel):
    # Single row result in a batch prediction response.
    row_index: int
    status: Literal["ok", "error"]
    prediction: int | None = None
    prediction_label: str | None = None
    churn_probability: float | None = None
    risk_level: str | None = None
    recommendation: str | None = None
    top_factors: list[str] = []
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    # Full response for a batch prediction request.
    total_rows: int
    successful: int
    failed: int
    results: list[BatchPredictionRow]
