import io
import logging

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.model_service import predict_customer_churn
from api.schemas import BatchPredictionResponse, BatchPredictionRow, CustomerData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a customer will churn based on their profile.",
    version="2.2.0",
)

# Expected columns for batch CSV uploads
BATCH_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure_months",
    "phone_service", "multiple_lines", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support", "streaming_tv",
    "streaming_movies", "contract", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges",
]


# Helpers

def _customer_from_row(row: pd.Series) -> CustomerData:
    return CustomerData(
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


def _validate_csv(df: pd.DataFrame) -> None:
    # Check for missing required columns
    missing_cols = set(BATCH_COLUMNS) - set(df.columns.str.lower())
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {sorted(missing_cols)}",
        )
    # Check if file is empty
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    # Enforce row limit
    if len(df) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 1000 rows. Please split your file.",
        )


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        # Read and decode file contents
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()
        return df
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV file: {exc}")


# Endpoints 

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    try:
        return predict_customer_churn(customer_data)
    except RuntimeError as exc:
        # Model files not found or not loaded
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable. Model files may be missing.",
        )
    except ValueError as exc:
        # Invalid input values (e.g., unexpected category)
        logger.warning("Invalid input value: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Input contains an unexpected value: {exc}",
        )
    except Exception as exc:
        # Catch-all for unexpected errors
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    # Read and validate CSV
    df = await _read_csv(file)
    _validate_csv(df)

    results, successful, failed = [], 0, 0

    # Process each row individually
    for idx, row in df.iterrows():
        try:
            # Convert row to CustomerData and predict
            customer = _customer_from_row(row)
            prediction = predict_customer_churn(customer)
            # Add successful prediction to results
            results.append(
                BatchPredictionRow(row_index=int(idx), status="ok", **prediction)
            )
            successful += 1
        except Exception as exc:
            # Log failed predictions
            results.append(
                BatchPredictionRow(row_index=int(idx), status="error", error=str(exc))
            )
            failed += 1
            logger.warning("Row %d failed: %s", idx, exc)

    # Return summary and results
    return BatchPredictionResponse(
        total_rows=len(df),
        successful=successful,
        failed=failed,
        results=results,
    )


@app.post("/predict/batch/download")
async def predict_batch_download(file: UploadFile = File(...)):
    # Read and validate CSV
    df = await _read_csv(file)
    _validate_csv(df)

    predictions = []

    # Process each row
    for idx, row in df.iterrows():
        try:
            # Convert row to CustomerData and predict
            customer = _customer_from_row(row)
            result = predict_customer_churn(customer)
            # Add original data plus predictions
            predictions.append({
                **row.to_dict(),
                "prediction":        result["prediction"],
                "prediction_label":  result["prediction_label"],
                "churn_probability": result["churn_probability"],
                "risk_level":        result["risk_level"],
                "recommendation":    result["recommendation"],
                "top_factors":       " | ".join(result["top_factors"]),  # Join list to string
                "status":            "ok",
                "error":             "",
            })
        except Exception as exc:
            # Log failed predictions and add empty values
            predictions.append({
                **row.to_dict(),
                "prediction": "", "prediction_label": "", "churn_probability": "",
                "risk_level": "", "recommendation": "", "top_factors": "",
                "status": "error", "error": str(exc),
            })
            logger.warning("Row %d failed: %s", idx, exc)

    # Convert to CSV and return as downloadable file
    result_df = pd.DataFrame(predictions)
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )