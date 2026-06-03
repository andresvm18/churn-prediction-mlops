from fastapi import FastAPI

from api.schemas import CustomerData
from api.model_service import predict_customer_churn

# Initialize FastAPI app with title
app = FastAPI(title="Customer Churn Prediction API")


# Health check / welcome endpoint
@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# Simple health check for monitoring
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Main prediction endpoint
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    # Delegate to service function
    return predict_customer_churn(customer_data)