import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Get database path from environment variable or use default
DB_PATH = os.getenv("DB_PATH", "predictions.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        # Create table with all prediction fields
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at        TEXT    NOT NULL,
                gender            TEXT,
                senior_citizen    INTEGER,
                partner           TEXT,
                dependents        TEXT,
                tenure_months     INTEGER,
                internet_service  TEXT,
                contract          TEXT,
                monthly_charges   REAL,
                total_charges     REAL,
                prediction        INTEGER,
                prediction_label  TEXT,
                churn_probability REAL,
                risk_level        TEXT,
                recommendation    TEXT,
                top_factors       TEXT
            )
            """
        )
        
        # Check for missing columns and add them (schema migration)
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        
        # Required columns with their SQLite types
        required_cols = {
            "gender": "TEXT", "senior_citizen": "INTEGER",
            "partner": "TEXT", "dependents": "TEXT",
            "tenure_months": "INTEGER", "internet_service": "TEXT",
            "contract": "TEXT", "monthly_charges": "REAL",
            "total_charges": "REAL", "prediction": "INTEGER",
            "prediction_label": "TEXT", "churn_probability": "REAL",
            "risk_level": "TEXT", "recommendation": "TEXT",
            "top_factors": "TEXT",
        }
        
        # Add any missing columns
        for col, col_type in required_cols.items():
            if col not in existing_cols:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {col} {col_type}"
                )
                logger.info("Added missing column '%s' to predictions table", col)
        
        conn.commit()
    logger.info("Database initialized at %s", DB_PATH)


def save_prediction(customer_data, result: dict) -> None:
    try:
        with get_connection() as conn:
            # Insert prediction record
            conn.execute(
                """
                INSERT INTO predictions (
                    created_at, gender, senior_citizen, partner, dependents,
                    tenure_months, internet_service, contract,
                    monthly_charges, total_charges,
                    prediction, prediction_label, churn_probability,
                    risk_level, recommendation, top_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    # Timestamp in UTC
                    datetime.now(timezone.utc).isoformat(),
                    # Customer features
                    customer_data.gender,
                    customer_data.senior_citizen,
                    customer_data.partner,
                    customer_data.dependents,
                    customer_data.tenure_months,
                    customer_data.internet_service,
                    customer_data.contract,
                    customer_data.monthly_charges,
                    customer_data.total_charges,
                    # Prediction results
                    result["prediction"],
                    result["prediction_label"],
                    result["churn_probability"],
                    result["risk_level"],
                    result["recommendation"],
                    # Store list as JSON string
                    json.dumps(result["top_factors"]),
                ),
            )
            conn.commit()
    except Exception as exc:
        # Log error but don't raise because prediction should succeed even if DB fails
        logger.error("Failed to save prediction: %s", exc)


def get_predictions(
    limit: int = 100,
    risk_level: str | None = None,
    prediction: int | None = None,
) -> list[dict]:
    # Build query dynamically based on filters
    query = "SELECT * FROM predictions"
    params = []
    filters = []

    # Apply filters if provided
    if risk_level:
        filters.append("risk_level = ?")
        params.append(risk_level)
    if prediction is not None:
        filters.append("prediction = ?")
        params.append(prediction)

    # Add WHERE clause if filters exist
    if filters:
        query += " WHERE " + " AND ".join(filters)

    # Order by newest first and limit results
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    # Execute query and fetch results
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    # Convert rows to dictionaries
    return [dict(row) for row in rows]


def get_summary_stats() -> dict:
    with get_connection() as conn:
        # Get total number of predictions
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # If no predictions, return empty stats
        if total == 0:
            return {
                "total": 0,
                "churn_count": 0,
                "churn_rate": 0.0,
                "avg_probability": 0.0,
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
            }

        # Count churned customers
        churn_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 1"
        ).fetchone()[0]

        # Calculate average churn probability
        avg_prob = conn.execute(
            "SELECT AVG(churn_probability) FROM predictions"
        ).fetchone()[0]

        # Get risk level distribution
        risk_counts = conn.execute(
            "SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level"
        ).fetchall()

        # Convert to dictionary
        risk_map = {row[0]: row[1] for row in risk_counts}

    # Return formatted statistics
    return {
        "total":           total,
        "churn_count":     churn_count,
        "churn_rate":      round(churn_count / total * 100, 1),
        "avg_probability": round(avg_prob * 100, 1),
        "high_risk":       risk_map.get("High", 0),
        "medium_risk":     risk_map.get("Medium", 0),
        "low_risk":        risk_map.get("Low", 0),
    }


def clear_history() -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM predictions")
        conn.commit()
        return cursor.rowcount