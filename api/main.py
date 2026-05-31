from fastapi import FastAPI

from api.schemas import CustomerData  # Import request data schema
from api.model_service import predict_customer_churn  # Import prediction logic

# Initialize FastAPI application
app = FastAPI(title="Customer Churn Prediction API")


# Health check endpoint - verifies API is running
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


# Prediction endpoint - accepts customer data and returns churn prediction
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    return predict_customer_churn(customer_data)  # Delegate to service function
