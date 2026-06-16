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

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.41

# Risk level thresholds (probability values between 0 and 1)
RISK_THRESHOLD_HIGH = 0.7  # >= 70% = High risk
RISK_THRESHOLD_MEDIUM = 0.4  # >= 40% = Medium risk, below 40% = Low risk

# Baseline churn rate for dashboard comparison (percentage)
BASELINE_CHURN_RATE = 26.0

# XGBoost model hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 270,  # Number of boosting rounds (trees)
    "max_depth": 7,  # Maximum tree depth - controls model complexity
    "learning_rate": 0.022554,  # Step size shrinkage - prevents overfitting
    "subsample": 0.515124,  # Fraction of training samples used per tree
    "min_child_weight": 1,  # Minimum sum of instance weights in a child node
    "gamma": 0.128656,  # Minimum loss reduction required for further partition
    "reg_alpha": 0.339024,  # L1 regularization on weights (Lasso)
    "reg_lambda": 1.587913,  # L2 regularization on weights (Ridge)
    "colsample_bytree": 0.8,  # Fraction of features used per tree
    "random_state": 42,  # Reproducibility seed
    "eval_metric": "logloss",  # Evaluation metric for binary classification
}
