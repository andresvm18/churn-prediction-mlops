# Model artifact file paths
MODEL_PATH = "src/models/churn_model.pkl"
ENCODER_PATH = "src/models/label_encoders.pkl"
METADATA_PATH = "src/models/metadata.json"

# Raw data file path
RAW_DATA_PATH = "data/raw/telco_churn.xlsx"

# Columns to remove before training (leakage, identifiers, geography)
COLUMNS_TO_REMOVE = [
    "Churn Value",  # Target variable - handled separately
    # Leakage columns (not available at prediction time)
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CLTV",
    # Identifier columns
    "CustomerID",
    # Geographic columns
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    # Constant column (same value for all rows)
    "Count",
]

# Target column name
TARGET_COLUMN = "Churn Value"

# Categorical columns (must match between training and inference)
CATEGORICAL_COLUMNS = [
    "Gender",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

# Risk level thresholds (probability values between 0 and 1)
RISK_THRESHOLD_HIGH = 0.7  # >= 70% = High risk
RISK_THRESHOLD_MEDIUM = 0.4  # >= 40% = Medium risk, below 40% = Low risk

# Baseline churn rate for dashboard comparison (percentage)
BASELINE_CHURN_RATE = 26.0

# XGBoost model hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 200,  # Number of trees
    "max_depth": 6,  # Maximum tree depth
    "learning_rate": 0.05,  # Step size shrinkage
    "subsample": 0.8,  # Fraction of samples per tree
    "colsample_bytree": 0.8,  # Fraction of features per tree
    "random_state": 42,  # Reproducibility seed
    "eval_metric": "logloss",  # Evaluation metric
}
