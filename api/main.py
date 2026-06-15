import io
import logging

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.model_service import predict_customer_churn
from api.schemas import BatchPredictionResponse, BatchPredictionRow, CustomerData

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
        "Predicts the probability that a customer " "will churn based on their profile."
    ),
    version="2.2.0",
)

# Expected CSV columns in order (for batch predictions)
BATCH_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
]


# Root endpoint - API info
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


# Health check endpoint - for monitoring
@app.get("/health")
def health():
    return {"status": "healthy"}


# Single prediction endpoint
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    try:
        # Call service layer to make prediction
        return predict_customer_churn(customer_data)

    except RuntimeError as exc:
        # Model or encoders not loaded (service unavailable)
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable. Model files may be missing.",
        )
    except ValueError as exc:
        # Unseen category or bad numeric conversion
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


# Batch prediction endpoint (returns JSON)
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    # Read CSV file
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV file: {exc}",
        )

    # Validate required columns exist
    missing_cols = set(BATCH_COLUMNS) - set(df.columns.str.lower())
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {sorted(missing_cols)}",
        )

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Validate file is not empty
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    # Enforce batch size limit
    if len(df) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 1000 rows. Please split your file.",
        )

    # Run predictions row by row
    results = []
    successful = 0
    failed = 0

    for idx, row in df.iterrows():
        try:
            # Convert CSV row to CustomerData model
            customer = CustomerData(
                gender=str(row["gender"]),
                senior_citizen=int(row["senior_citizen"]),
                partner=str(row["partner"]),
                dependents=str(row["dependents"]),
                tenure_months=int(row["tenure_months"]),
                phone_service=str(row["phone_service"]),
                multiple_lines=str(row["multiple_lines"]),
                internet_service=str(row["internet_service"]),
                online_security=str(row["online_security"]),
                online_backup=str(row["online_backup"]),
                device_protection=str(row["device_protection"]),
                tech_support=str(row["tech_support"]),
                streaming_tv=str(row["streaming_tv"]),
                streaming_movies=str(row["streaming_movies"]),
                contract=str(row["contract"]),
                paperless_billing=str(row["paperless_billing"]),
                payment_method=str(row["payment_method"]),
                monthly_charges=float(row["monthly_charges"]),
                total_charges=float(row["total_charges"]),
            )

            # Get prediction for this customer
            prediction = predict_customer_churn(customer)

            # Add successful result
            results.append(
                BatchPredictionRow(
                    row_index=int(idx),
                    status="ok",
                    **prediction,
                )
            )
            successful += 1

        except Exception as exc:
            # Add failed result with error message
            results.append(
                BatchPredictionRow(
                    row_index=int(idx),
                    status="error",
                    error=str(exc),
                )
            )
            failed += 1
            logger.warning("Row %d failed: %s", idx, exc)

    # Return batch response summary
    return BatchPredictionResponse(
        total_rows=len(df),
        successful=successful,
        failed=failed,
        results=results,
    )


# Batch prediction endpoint (returns CSV download)
@app.post("/predict/batch/download")
async def predict_batch_download(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Read CSV file
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    # Validate required columns exist
    missing_cols = set(BATCH_COLUMNS) - set(df.columns.str.lower())
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {sorted(missing_cols)}",
        )

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Validate file is not empty and within size limit
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    if len(df) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000 rows.")

    # Run predictions and build result DataFrame
    predictions = []
    for idx, row in df.iterrows():
        try:
            # Convert CSV row to CustomerData model
            customer = CustomerData(
                gender=str(row["gender"]),
                senior_citizen=int(row["senior_citizen"]),
                partner=str(row["partner"]),
                dependents=str(row["dependents"]),
                tenure_months=int(row["tenure_months"]),
                phone_service=str(row["phone_service"]),
                multiple_lines=str(row["multiple_lines"]),
                internet_service=str(row["internet_service"]),
                online_security=str(row["online_security"]),
                online_backup=str(row["online_backup"]),
                device_protection=str(row["device_protection"]),
                tech_support=str(row["tech_support"]),
                streaming_tv=str(row["streaming_tv"]),
                streaming_movies=str(row["streaming_movies"]),
                contract=str(row["contract"]),
                paperless_billing=str(row["paperless_billing"]),
                payment_method=str(row["payment_method"]),
                monthly_charges=float(row["monthly_charges"]),
                total_charges=float(row["total_charges"]),
            )

            # Get prediction for this customer
            result = predict_customer_churn(customer)

            # Add original data plus prediction results
            predictions.append(
                {
                    **row.to_dict(),
                    "prediction": result["prediction"],
                    "prediction_label": result["prediction_label"],
                    "churn_probability": result["churn_probability"],
                    "risk_level": result["risk_level"],
                    "recommendation": result["recommendation"],
                    "top_factors": " | ".join(
                        result["top_factors"]
                    ),  # Join list as string
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            # Add failed row with error information
            predictions.append(
                {
                    **row.to_dict(),
                    "prediction": "",
                    "prediction_label": "",
                    "churn_probability": "",
                    "risk_level": "",
                    "recommendation": "",
                    "top_factors": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            logger.warning("Row %d failed: %s", idx, exc)

    # Convert results to CSV
    result_df = pd.DataFrame(predictions)
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Return CSV as downloadable file
    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )
