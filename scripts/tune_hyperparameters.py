import argparse
import json
import logging
import os
import sys

import joblib
import mlflow
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configuration constants
from api.config import (
    CATEGORICAL_COLUMNS,
    CLASSIFICATION_THRESHOLD,
    COLUMNS_TO_REMOVE,
    ENCODER_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
MLFLOW_EXPERIMENT = "churn-tuning"
OUTPUT_PATH = "scripts/tuning_results/best_params.json"


# Data loading


def load_splits() -> tuple:
    logger.info("Loading data from %s", RAW_DATA_PATH)
    df = pd.read_excel(RAW_DATA_PATH)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Replace spaces with NaN and fill with "Unknown"
    for col in df.columns:
        df[col] = df[col].replace(" ", pd.NA)
    df = df.fillna("Unknown")

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

    # Split data (same random_state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    logger.info(
        "Train: %d rows | Test: %d rows | After SMOTE: %d rows",
        len(X_train),
        len(X_test),
        len(X_train_bal),
    )
    return X_train_bal, X_test, y_train_bal, y_test, encoders


# Objective


def make_objective(X_train, X_test, y_train, y_test):

    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "random_state": 42,
            "eval_metric": "logloss",
        }

        # Train model with current parameters
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions using the classification threshold from config
        probabilities = model.predict_proba(X_test)[:, 1]
        y_pred = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

        # Calculate metrics
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log trial to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("recall_churn", recall)
            mlflow.log_metric("f1_churn", f1)
            mlflow.log_metric("trial_number", trial.number)

        # Optimize F1 score
        return f1

    return objective


# Results


def save_best_params(study: optuna.Study) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Prepare output data
    output = {
        "best_value": round(study.best_value, 4),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }

    # Save to JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Best params saved to %s", OUTPUT_PATH)


def print_config_snippet(params: dict) -> None:
    print("\n" + "=" * 60)
    print("UPDATE api/config.py with these values:")
    print("=" * 60)
    print("XGBOOST_PARAMS = {")
    for k, v in params.items():
        if isinstance(v, float):
            print(f'    "{k}": {round(v, 6)},')
        else:
            print(f'    "{k}": {v},')
    print('    "random_state": 42,')
    print('    "eval_metric": "logloss",')
    print("}")
    print("=" * 60 + "\n")


# Entry point
def main(n_trials: int = 50) -> None:
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Load and prepare data
    logger.info("Loading data...")
    X_train, X_test, y_train, y_test, _ = load_splits()

    # Start tuning
    logger.info("Starting Optuna study with %d trials...", n_trials)

    # Main MLflow run for the entire tuning process
    with mlflow.start_run(run_name=f"optuna-{n_trials}-trials"):
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name="churn-xgboost",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Run optimization
        study.optimize(
            make_objective(X_train, X_test, y_train, y_test),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Log best result to parent MLflow run
        mlflow.log_metric("best_f1_churn", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    # Display results
    logger.info("Best F1: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Save and display results
    save_best_params(study)
    print_config_snippet(study.best_params)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters with Optuna."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    args = parser.parse_args()

    # Run main function
    main(n_trials=args.trials)
