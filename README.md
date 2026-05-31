# Customer Churn Prediction

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
- MLOps-oriented project structure 


## Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave a service.

Using customer demographics, subscription information, account history, and service usage data, the model estimates the probability that a customer will churn and provides business recommendations for retention actions.

## Features

### Machine Learning Pipeline
- Data cleaning and preprocessing
- Missing value handling
- Feature encoding
- Feature selection
- Train/Test split
- SMOTE class balancing
- XGBoost classification
- Model evaluation
- Model persistence with Joblib

### Explainable AI (XAI)

The project includes SHAP (SHapley Additive Explanations) for model interpretability.

#### Features:
- Global feature importance
- Local prediction explanations
- Top churn risk factors per customer
- Business-friendly insights

### FastAPI Backend

REST API providing:

- Real-time predictions
- Churn probability estimation
- Risk classification
- Personalized recommendations
- SHAP explanation factorsa
- Interactive Swagger documentation

### Streamlit Dashboard

Modern customer-facing interface featuring:

- Customer profile forms
- Risk visualization
- Churn probability indicators
- Risk factor badges
- Recommendation panel
- Responsive UI
- Professional styling

### Dockerized Deployment

The application is fully containerized using Docker Compose.

Services:

- FastAPI Backend
- Streamlit Frontend

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| ML Framework | Scikit-Learn |
| Model | XGBoost |
| Explainability | SHAP |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| API | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Serialization | Joblib |

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
│   ├── main.py 
│   ├── model_service.py
│   └── schemas.py
│ 
├── app/
│   ├── assets/ 
│   ├── streamlit_app.py 
│   └── styles.css 
│ 
├── data/ 
│   ├── raw/
│   │   └── telco_churn.xlsx
│   └── processed/ 
│ 
├── notebooks/ 
│   ├── 01_eda_analysis.ipynb
│   └── 02_model_explainability.ipynb 
│
├── scripts/ 
│   └── train_model.py 
│ 
├── src/ 
│   ├── data/
│   └── models/ 
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── .dockerignore
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

**Dataset used:** IBM Telco Customer Churn Dataset

**Target variable:** Churn Value

Values:

- 1 = Customer churns
- 0 = Customer remains

## Model Performance

Current production model: **XGBoost + SMOTE**

| Metric | Score |
|----------|------------|
| Accuracy | 0.73 |
| Churn Recall | 0.73 |
| Churn F1 Score | 0.59 |

The dataset is naturally imbalanced, making recall an important metric for identifying customers at risk.

## API Endpoints
**Health Check**
~~~
GET /

Response:
{
  "status": "running"
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
  "total_charges": 1100.0
}

Example Response:

{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.91,
  "risk_level": "High",
  "recommendation": "Customer is at high risk of churn. Consider retention actions.",
  "top_factors": [
    "Month-to-month contract",
    "Fiber optic service",
    "High monthly charges"
  ]
}
~~~

## Running Locally

**Clone Repository**
~~~
git clone https://github.com/yourusername/customer-churn-prediction.git 
cd customer-churn-prediction
~~~

**Create Virtual Environment**
~~~
python -m venv venv

Windows:
  venv\Scripts\activate

Linux / Mac:
  source venv/bin/activate
~~~

**Install Dependencies**
~~~
Development environment:
  pip install -r requirements.txt

Production environment:
  pip install -r requirements-prod.txt
~~~

**Run FastAPI**
~~~
uvicorn api.main:app --reload

API available at: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs
~~~

**Run Streamlit**
~~~
streamlit run app/streamlit_app.py

Dashboard: http://localhost:8501
~~~

**Docker Deployment**
~~~
Build and run both services: docker compose up --build

Run in background: docker compose up -d --build

Stop services: docker compose down
~~~

## Docker Architecture
~~~
┌─────────────────────┐
│     Streamlit       │
│     Dashboard       │
│     Port 8501       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      FastAPI        │
│   Prediction API    │
│     Port 8000       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  XGBoost + SHAP     │
│    Model Layer      │
└─────────────────────┘
~~~

## Future Improvements

Planned enhancements:

- MLflow experiment tracking
- CI/CD pipelines
- Unit testing
- Cloud deployment
- Monitoring and logging
- Authentication
- Model versioning
- Automated retraining
- Feature store integration