# Customer Churn Prediction

A Machine Learning project focused on predicting customer churn using classification models and data preprocessing techniques. This project is being built step by step following real-world ML and MLOps practices.

---

# Project Goal

The objective of this project is to predict whether a customer is likely to leave a service based on customer behavior, subscription information, and account-related features.

The project uses the IBM Telco Customer Churn dataset and focuses on building a complete ML pipeline from data preprocessing to model deployment.

---

# Current Features

## Machine Learning Pipeline

- Dataset loading with Pandas
- Data cleaning and preprocessing
- Missing value handling
- Feature selection
- Label encoding
- Train/Test split
- SMOTE balancing
- XGBoost classification
- Model evaluation
- Model serialization with Joblib

---

# API Features

The project includes a FastAPI service that exposes a prediction endpoint for customer churn analysis.

Current API capabilities:

- Real-time churn prediction
- Churn probability scoring
- Risk level classification
- Business recommendations
- Interactive Swagger documentation

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| Scikit-learn | Machine Learning utilities |
| XGBoost | Classification model |
| FastAPI | API development |
| Uvicorn | ASGI server |
| Joblib | Model serialization |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |

---

# Project Structure

```text
customer-churn-prediction/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda_analysis.ipynb
│
├── scripts/
│   └── train_model.py
│
├── src/
│   └── models/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

This project uses the IBM Telco Customer Churn dataset.

Target column:

```text
Churn Value
```

- `1` = Customer leaves the company
- `0` = Customer stays

---

# Model Performance

Current XGBoost + SMOTE results:

| Metric | Score |
|---|---|
| Accuracy | 0.73 |
| Churn Recall | 0.73 |
| Churn F1-Score | 0.59 |

The dataset is imbalanced, making churn prediction more realistic and challenging.

---

# Key Concepts Explored

During development, several important Machine Learning concepts were explored:

- Data leakage
- Feature selection
- Class imbalance
- SMOTE oversampling
- Precision vs Recall
- F1-score analysis
- Correlation heatmaps
- Feature importance
- Model comparison
- API deployment

---

# API Usage

## Run the API

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Prediction Endpoint

## POST `/predict`

This endpoint receives customer information and returns a churn prediction.

### Example Request

```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure_months": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 95.5,
  "total_charges": 1100.0
}
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.9137,
  "risk_level": "High",
  "recommendation": "Customer is at high risk of churn. Consider retention actions."
}
```

---

# Response Fields

| Field | Description |
|---|---|
| `prediction` | Numeric prediction (`1` = churn, `0` = no churn) |
| `prediction_label` | Human-readable prediction |
| `churn_probability` | Probability of churn |
| `risk_level` | Low, Medium, or High risk |
| `recommendation` | Suggested business action |

---

# Exploratory Data Analysis

EDA work includes:

- Churn distribution analysis
- Contract type analysis
- Monthly charges analysis
- Correlation heatmaps
- Feature importance analysis

Notebook location:

```text
notebooks/01_eda_analysis.ipynb
```

---

# Setup

## Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run Model Training

```bash
python scripts/train_model.py
```

---

# Future Improvements

Upcoming project goals:

- Docker containerization
- Streamlit dashboard
- MLflow experiment tracking
- CI/CD pipelines
- Cloud deployment
- Monitoring and logging
- Authentication and rate limiting
- Model versioning

---

# Author

Andrés Víquez
