import logging

from fastapi import FastAPI, HTTPException

from api.schemas import CustomerData
from api.model_service import predict_customer_churn

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts the probability that a customer will churn "
        "based on their profile."
    ),
    version="2.1.0",
)


# Root endpoint - API info
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


# Health check endpoint - for monitoring
@app.get("/health")
def health():
    return {"status": "healthy"}


# Main prediction endpoint
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    try:
        # Call service layer to make prediction
        result = predict_customer_churn(customer_data)
        return result

    except RuntimeError as exc:
        # Model or encoders not loaded (service unavailable)
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable. Model files may be missing.",
        )

    except ValueError as exc:
        # Unseen category in LabelEncoder or bad numeric conversion
        logger.warning("Invalid input value: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Input contains an unexpected value: {exc}",
        )

    except Exception as exc:
        # Catch-all: log full traceback, return generic 500
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        )
