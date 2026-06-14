import json
import logging
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configuration constants
from api.config import (
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_REMOVE,
    ENCODER_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    XGBOOST_PARAMS,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# Load and clean raw data
def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_excel(path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Replace spaces with NaN and fill with "Unknown"
    for col in df.columns:
        df[col] = df[col].replace(" ", pd.NA)
    df = df.fillna("Unknown")

    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


# Preprocess data for training
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    # Separate target from features
    target = df[TARGET_COLUMN]
    features = df.drop(columns=COLUMNS_TO_REMOVE)

    # Encode categorical columns
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        enc = LabelEncoder()
        features[col] = enc.fit_transform(features[col].astype(str))
        encoders[col] = enc

    # Convert all to numeric, fill NaN with 0
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    logger.info("Features shape after preprocessing: %s", features.shape)
    return features, target, encoders


# Train the XGBoost model
def train(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple:
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    logger.info("Class distribution before SMOTE:\n%s", y_train.value_counts().to_string())

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    logger.info("Class distribution after SMOTE:\n%s", y_train_bal.value_counts().to_string())

    # Train XGBoost classifier
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_bal, y_train_bal)

    logger.info("Model training complete.")
    return model, X_test, y_test


# Evaluate model performance
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("\n%s", classification_report(y_test, y_pred))

    return {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
    }


# Save trained artifacts
def save_artifacts(model, encoders: dict, metrics: dict) -> None:
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # Save the encoders
    joblib.dump(encoders, ENCODER_PATH)
    logger.info("Encoders saved to %s", ENCODER_PATH)

    # Save training metadata
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "hyperparameters": XGBOOST_PARAMS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "metrics": metrics,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", METADATA_PATH)


# Main execution pipeline
def main() -> None:
    # Load and clean data
    df = load_data(RAW_DATA_PATH)
    
    # Preprocess features and target
    features, target, encoders = preprocess(df)
    
    # Train the model
    model, X_test, y_test = train(features, target)
    
    # Evaluate performance
    metrics = evaluate(model, X_test, y_test)
    
    # Save all artifacts
    save_artifacts(model, encoders, metrics)
    
    logger.info("Pipeline complete.")


# Run the training script
if __name__ == "__main__":
    main()