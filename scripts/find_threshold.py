import json
import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configuration constants
from api.config import (
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_REMOVE,
    ENCODER_PATH,
    MODEL_PATH,
    METADATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Directory where threshold analysis outputs will be saved
OUTPUT_DIR = "scripts/threshold_analysis"


def load_test_set() -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

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

    # Load and apply saved encoders
    encoders = joblib.load(ENCODER_PATH)
    for col in CATEGORICAL_COLUMNS:
        features[col] = encoders[col].transform(features[col].astype(str))

    # Convert all to numeric, fill NaN with 0
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Same split as training — same random_state guarantees identical test set
    _, X_test, _, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    logger.info("Test set: %d rows", len(X_test))
    return X_test, y_test


def evaluate_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    # Get probabilities for churn class (index 1)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate precision-recall pairs for all thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)

    # precision_recall_curve returns n+1 values for precisions/recalls
    # and n values for thresholds — align them
    f1_scores = (
        2
        * precisions[:-1]
        * recalls[:-1]
        / (
            precisions[:-1] + recalls[:-1] + 1e-8
        )  # Add small epsilon to avoid division by zero
    )

    # Create DataFrame with results for each threshold
    results = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precisions[:-1],
            "recall": recalls[:-1],
            "f1": f1_scores,
        }
    )

    return results


def find_best_threshold(results: pd.DataFrame) -> dict:
    # Find threshold with best F1 score
    best_f1_idx = results["f1"].idxmax()
    best_f1_row = results.loc[best_f1_idx]

    # Find best precision while maintaining at least 80% recall
    high_recall = results[results["recall"] >= 0.80]
    if not high_recall.empty:
        best_recall_row = high_recall.loc[high_recall["precision"].idxmax()]
    else:
        best_recall_row = best_f1_row
        logger.warning("No threshold achieves recall >= 0.80. Using best F1 threshold.")

    return {
        "best_f1": {
            "threshold": round(float(best_f1_row["threshold"]), 4),
            "precision": round(float(best_f1_row["precision"]), 4),
            "recall": round(float(best_f1_row["recall"]), 4),
            "f1": round(float(best_f1_row["f1"]), 4),
        },
        "best_recall_80": {
            "threshold": round(float(best_recall_row["threshold"]), 4),
            "precision": round(float(best_recall_row["precision"]), 4),
            "recall": round(float(best_recall_row["recall"]), 4),
            "f1": round(float(best_recall_row["f1"]), 4),
        },
    }


def plot_precision_recall_curve(results: pd.DataFrame, best: dict) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Precision, Recall, and F1 vs Threshold
    ax = axes[0]
    ax.plot(
        results["threshold"], results["precision"], label="Precision", color="#2980b9"
    )
    ax.plot(results["threshold"], results["recall"], label="Recall", color="#e74c3c")
    ax.plot(results["threshold"], results["f1"], label="F1", color="#27ae60")

    # Mark best F1 threshold
    ax.axvline(
        best["best_f1"]["threshold"],
        color="#27ae60",
        linestyle="--",
        alpha=0.7,
        label=f'Best F1 threshold = {best["best_f1"]["threshold"]}',
    )

    # Mark threshold for 80% recall
    ax.axvline(
        best["best_recall_80"]["threshold"],
        color="#e74c3c",
        linestyle="--",
        alpha=0.7,
        label=f'Recall≥80% threshold = {best["best_recall_80"]["threshold"]}',
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right plot: Precision-Recall curve
    ax2 = axes[1]
    ax2.plot(results["recall"], results["precision"], color="#8e44ad")

    # Mark best F1 point
    ax2.scatter(
        best["best_f1"]["recall"],
        best["best_f1"]["precision"],
        color="#27ae60",
        zorder=5,
        s=80,
        label=f'Best F1 ({best["best_f1"]["threshold"]})',
    )

    # Mark recall≥80% point
    ax2.scatter(
        best["best_recall_80"]["recall"],
        best["best_recall_80"]["precision"],
        color="#e74c3c",
        zorder=5,
        s=80,
        label=f'Recall≥80% ({best["best_recall_80"]["threshold"]})',
    )

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "threshold_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Plot saved to %s", path)


def print_comparison(model, X_test, y_test, best: dict) -> None:
    # Get probabilities for churn class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Generate predictions using different thresholds
    default_preds = (probabilities >= 0.5).astype(int)
    best_f1_preds = (probabilities >= best["best_f1"]["threshold"]).astype(int)
    best_rec_preds = (probabilities >= best["best_recall_80"]["threshold"]).astype(int)

    # Print comparison reports
    logger.info(
        "\n%s\nDefault threshold (0.5):\n%s",
        "=" * 50,
        classification_report(y_test, default_preds),
    )
    logger.info(
        "\n%s\nBest F1 threshold (%s):\n%s",
        "=" * 50,
        best["best_f1"]["threshold"],
        classification_report(y_test, best_f1_preds),
    )
    logger.info(
        "\n%s\nBest Recall≥80%% threshold (%s):\n%s",
        "=" * 50,
        best["best_recall_80"]["threshold"],
        classification_report(y_test, best_rec_preds),
    )


def save_results(best: dict) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save threshold results as JSON
    output_path = os.path.join(OUTPUT_DIR, "threshold_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    logger.info("Threshold results saved to %s", output_path)

    # Append analysis to existing model metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["threshold_analysis"] = best
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata updated with threshold analysis.")


def main() -> None:
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Load test set
    X_test, y_test = load_test_set()

    # Evaluate metrics across all thresholds
    results = evaluate_thresholds(model, X_test, y_test)

    # Find optimal thresholds
    best = find_best_threshold(results)

    # Log results
    logger.info(
        "Best F1 threshold:       %s → precision=%.3f recall=%.3f f1=%.3f",
        best["best_f1"]["threshold"],
        best["best_f1"]["precision"],
        best["best_f1"]["recall"],
        best["best_f1"]["f1"],
    )

    logger.info(
        "Best Recall≥80%% threshold: %s → precision=%.3f recall=%.3f f1=%.3f",
        best["best_recall_80"]["threshold"],
        best["best_recall_80"]["precision"],
        best["best_recall_80"]["recall"],
        best["best_recall_80"]["f1"],
    )

    # Generate and save visualizations
    plot_precision_recall_curve(results, best)

    # Print comparison reports
    print_comparison(model, X_test, y_test, best)

    # Save results
    save_results(best)

    # Print recommendation for config update
    logger.info(
        "\nRecommendation: set CLASSIFICATION_THRESHOLD = %s in api/config.py"
        " (best F1) or %s (maximize recall at cost of precision).",
        best["best_f1"]["threshold"],
        best["best_recall_80"]["threshold"],
    )


# Run the threshold optimization script
if __name__ == "__main__":
    main()
