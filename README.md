# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Tests](https://github.com/andresvm18/churn-prediction-mlops/actions/workflows/tests.yml/badge.svg)

## Live Demo

### Dashboard: https://churn-dashboard-z386.onrender.com/
![Dashboard Home](images/dashboard-home.png)

![Prediction](images/dashboard-prediction.png)

### API Docs: https://churn-api-mlbh.onrender.com/docs
![Swagger](images/swagger-api.png)


A complete end-to-end Machine Learning project for predicting customer churn using the IBM Telco Customer Churn dataset.

This project demonstrates the full lifecycle of a Machine Learning solution, including:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training
- Model explainability with SHAP
- FastAPI deployment
- Streamlit dashboard
- Docker containerization
- Automated testing with Pytest
- Continuous Integration with GitHub Actions
- Modular project architecture


## Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave a service.

Using customer demographics, subscription information, account history, and service usage data, the model estimates the probability that a customer will churn and provides business recommendations for retention actions.

## Features

### Machine Learning Pipeline
- Data cleaning and preprocessing
- Missing value handling
- Feature encoding with LabelEncoder
- Feature selection (leakage-free)
- Train/Test split
- SMOTE class balancing
- XGBoost classification
- Model evaluation with classification report
- Model persistence with Joblib
- Training metadata saved to JSON (date, metrics, hyperparameters)

### Explainable AI (XAI)

The project includes SHAP (SHapley Additive Explanations) for model interpretability.

- Global feature importance
- Local prediction explanations
- Top 3 churn risk factors per customer, returned as `"Column: value"` strings
- Business-friendly labels in the dashboard

### FastAPI Backend

REST API providing:

- Strict input validation via Pydantic `Literal` types and `Field` constraints
- Real-time predictions
- Churn probability estimation
- Risk classification (Low / Medium / High)
- Personalized recommendations
- SHAP-based feature explanations
- Structured error responses (422 / 503 / 500)
- Interactive Swagger documentation

### Streamlit Dashboard

Modern customer-facing interface featuring:

- Customer profile forms organized in tabs
- Circular gauge for churn probability
- Risk factor badges with human-readable labels
- Recommendation panel
- Customer snapshot card
- Responsive UI with professional styling

### Dockerized Deployment

The application is fully containerized using Docker Compose.

- FastAPI backend with `/health` healthcheck endpoint
- Streamlit frontend waits for API to be healthy before starting (`condition: service_healthy`)
- Automatic restart policy (`unless-stopped`)

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| ML Framework | Scikit-Learn |
| Model | XGBoost |
| Class Balancing | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Data Processing | Pandas / NumPy |
| API | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Serialization | Joblib |
| Testing | Pytest + pytest-cov |
| CI/CD | GitHub Actions |

## Project Structure

~~~
customer-churn-prediction/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── api/
│   ├── __init__.py
│   ├── config.py          # Single source of truth: paths, columns, thresholds
│   ├── main.py            # FastAPI app and endpoint definitions
│   ├── model_service.py   # Prediction pipeline and SHAP logic
│   └── schemas.py         # Pydantic request/response schemas
│
├── app/
│   ├── streamlit_app.py
│   └── styles.css
│
├── data/
│   ├── raw/
│   │   └── telco_churn.xlsx
│   └── processed/
│
├── images/
│   ├── dashboard-home.png
│   ├── dashboard-prediction.png
│   └── swagger-api.png
│
├── notebooks/
│   ├── 01_eda_analysis.ipynb
│   └── 02_model_explainability.ipynb
│
├── scripts/
│   └── train_model.py     # Structured training pipeline with metadata output
│
├── src/
│   └── models/
│       ├── churn_model.pkl
│       ├── label_encoders.pkl
│       └── metadata.json  # Training date, metrics, hyperparameters
│
├── tests/
│   ├── test_api.py           # Endpoint tests including validation and error cases
│   ├── test_model.py         # Business logic tests
│   └── test_preprocessing.py # Encoding, risk level, and SHAP factor tests
│
├── .dockerignore
├── .flake8
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── pytest.ini
├── README.md
├── requirements-dev.txt
├── requirements-prod.txt
└── requirements.txt
~~~

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

**Target variable:** `Churn Value`

- `1` = Customer churns
- `0` = Customer remains

## Model Performance

Current production model: **XGBoost + SMOTE**

| Metric | Class 0 (No Churn) | Class 1 (Churn) |
|--------|-------------------|-----------------|
| Precision | 0.88 | 0.58 |
| Recall | 0.82 | 0.68 |
| F1 Score | 0.85 | 0.63 |
| **Accuracy** | | **78.5%** |

Business objective: maximize churn detection (recall on class 1) while maintaining acceptable precision. The dataset is naturally imbalanced (~27% churn), which is addressed with SMOTE during training.

## API Endpoints

**Health Check**
~~~
GET /health

Response:
{
  "status": "healthy"
}
~~~

**Predict Churn**
~~~
POST /predict

Example Request:
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
  "total_charges": 1146.0
}

Example Response:
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.9137,
  "risk_level": "High",
  "recommendation": "Customer is at high risk of churn. Consider retention actions.",
  "top_factors": [
    "Contract: Month-to-month",
    "Monthly Charges: 95.5",
    "Tenure Months: 12"
  ]
}
~~~

**Validation errors** return HTTP `422` with a description of the invalid field.
**Service unavailable** (model not loaded) returns HTTP `503`.

## Running Locally

**Clone Repository**
~~~
git clone https://github.com/andresvm18/customer-churn-prediction.git
cd customer-churn-prediction
~~~

**Create Virtual Environment**
~~~
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / Mac:
source venv/bin/activate
~~~

**Install Dependencies**
~~~
# Development (includes training, testing, and linting tools):
pip install -r requirements-dev.txt

# Production only:
pip install -r requirements-prod.txt
~~~

**Train the Model**
~~~
python scripts/train_model.py
~~~

This generates:
- `src/models/churn_model.pkl`
- `src/models/label_encoders.pkl`
- `src/models/metadata.json`

**Run FastAPI**
~~~
uvicorn api.main:app --reload

API available at:  http://127.0.0.1:8000
Swagger UI:        http://127.0.0.1:8000/docs
~~~

**Run Streamlit**
~~~
streamlit run app/streamlit_app.py

Dashboard: http://localhost:8501
~~~

**Docker Deployment**
~~~
# Build and start both services:
docker compose up --build

# Run in background:
docker compose up -d --build

# Stop services:
docker compose down
~~~

The dashboard container waits for the API healthcheck to pass before starting.

## Docker Architecture
~~~
┌─────────────────────┐
│     Streamlit       │
│     Dashboard       │
│     Port 8501       │
└──────────┬──────────┘
           │ (waits for healthy)
           ▼
┌─────────────────────┐
│      FastAPI        │
│   Prediction API    │
│     Port 8000       │
│   GET /health ✓     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  XGBoost + SHAP     │
│    Model Layer      │
└─────────────────────┘
~~~

## MLOps Architecture
~~~
Dataset
   │
   ▼
  EDA (notebooks/)
   │
   ▼
Preprocessing + Encoding
   │
   ▼
 SMOTE
   │
   ▼
XGBoost Training
   │
   ▼
Joblib Artifacts + metadata.json
   │
   ├── FastAPI (api/)
   │
   └── Streamlit (app/)
~~~

## CI/CD Pipeline

GitHub Actions runs on every push and pull request to `main`:

1. Install dependencies (`requirements-dev.txt`)
2. Run all tests with coverage (`pytest --cov=api`)
3. Check code style (`flake8`)
4. Validate formatting (`black --check`)

**Workflow:** Developer → Push → GitHub Actions → Tests + Lint → Pass / Fail

## Testing

The project includes 45 automated tests across 3 files with 78% coverage on the API layer.

| File | What it covers |
|------|---------------|
| `test_api.py` | Endpoints: happy path, 7 validation cases (422), service errors (503 / 500) |
| `test_model.py` | Risk level thresholds, recommendations, DataFrame building |
| `test_preprocessing.py` | Encoding pipeline, unseen categories, SHAP factor format |

Run locally:

~~~
# All tests with coverage:
pytest --cov=api --cov-report=term-missing -v

# Quick run:
pytest
~~~

## Future Improvements

- MLflow experiment tracking and model versioning
- Automated retraining pipeline with data drift detection
- Authentication and authorization (API key / JWT)
- Cloud deployment monitoring and alerting
- A/B testing for model versions

## Author

Andrés Víquez

LinkedIn: https://www.linkedin.com/in/andr%C3%A9s-v%C3%ADquez-marchena-b39490310/

GitHub: https://github.com/andresvm18