# Customer Churn Prediction

A Machine Learning project focused on predicting customer churn using classification models and data preprocessing techniques. This project is being built step by step following real-world ML and MLOps practices.

---

# Project Goal

The objective of this project is to predict whether a customer is likely to leave a service based on customer behavior, subscription information, and account-related features.

The project uses the IBM Telco Customer Churn dataset and focuses on building a complete ML pipeline from data preprocessing to model deployment.

---

# Current Progress

## Completed Features

- Dataset loading with Pandas
- Data cleaning and preprocessing
- Missing value handling
- Categorical feature encoding
- Feature and target separation
- Train/Test split
- Baseline Random Forest model
- Model evaluation
- Model serialization with Joblib
- Label encoder serialization

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| Scikit-learn | Machine Learning |
| Joblib | Model serialization |
| Jupyter | Data analysis and experimentation |

---

# Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── scripts/
│   └── train_model.py
│
├── src/
│   ├── data/
│   └── models/
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Dataset

This project uses the IBM Telco Customer Churn dataset.

Main target column:

```text
Churn Value
```

- `1` = Customer left the service
- `0` = Customer stayed

---

# Machine Learning Pipeline

The current pipeline includes:

```text
Dataset → Cleaning → Encoding → Training → Evaluation → Model Saving
```

---

# Model

Current baseline model:

```text
RandomForestClassifier
```

Configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
```

---

# Current Results

Example metrics from the baseline model:

| Metric | Score |
|---|---|
| Accuracy | 0.76 |
| Churn Recall | 0.41 |
| Churn F1-Score | 0.48 |

The dataset is imbalanced, which makes churn prediction more challenging and realistic.

---

# Lessons Learned

During development, several important ML concepts were explored:

- Data leakage
- Class imbalance
- Precision vs Recall
- Feature preprocessing
- Categorical encoding
- Baseline modeling

One major issue discovered was that removing all missing values accidentally removed most non-churn customers, causing unrealistic 100% accuracy. This helped demonstrate the importance of proper preprocessing in Machine Learning workflows.

---

# Next Steps

Upcoming development phases:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Hyperparameter tuning
- XGBoost and LightGBM models
- FastAPI prediction API
- Streamlit dashboard
- Docker containerization
- MLflow experiment tracking
- CI/CD pipelines
- Monitoring and drift detection

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

# Run Training Script

```bash
python scripts/train_model.py
```

---

# Future Goals

This project aims to evolve into a complete MLOps system including:

- Model serving
- Automated training pipelines
- Experiment tracking
- Monitoring
- Continuous deployment

---

# Author

Andrés Víquez