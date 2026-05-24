import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset
customer_churn_data = pd.read_excel("data/raw/telco_churn.xlsx")

print("\nDataset loaded successfully\n")
print(customer_churn_data.head())

# Clean column names
customer_churn_data.columns = customer_churn_data.columns.str.strip()

# Replace empty values
for column_name in customer_churn_data.columns:
    customer_churn_data[column_name] = customer_churn_data[column_name].replace(" ", pd.NA)

# Fill missing values
customer_churn_data = customer_churn_data.fillna("Unknown")

# Define target column
target_column = "Churn Value"

# Remove leakage, identifier, geographic, and constant columns
columns_to_remove = [
    target_column,

    # Leakage columns
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

    # Constant columns
    "Count"
]

# Separate features and target
feature_data = customer_churn_data.drop(columns=columns_to_remove)
target_data = customer_churn_data[target_column]

# Define categorical columns used by both training and API
categorical_columns = [
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
    "Payment Method"
]

# Encode categorical features
categorical_encoders = {}

for column_name in categorical_columns:
    column_encoder = LabelEncoder()

    feature_data[column_name] = column_encoder.fit_transform(
        feature_data[column_name].astype(str)
    )

    categorical_encoders[column_name] = column_encoder

# Convert all features to numeric
feature_data = feature_data.apply(pd.to_numeric, errors="coerce")
feature_data = feature_data.fillna(0)

# Split dataset
features_train, features_test, target_train, target_test = train_test_split(
    feature_data,
    target_data,
    test_size=0.2,
    random_state=42,
    stratify=target_data
)

# Balance training data using SMOTE
smote_balancer = SMOTE(random_state=42)

features_train_balanced, target_train_balanced = smote_balancer.fit_resample(
    features_train,
    target_train
)

print("\nTraining class distribution before SMOTE:")
print(target_train.value_counts())

print("\nTraining class distribution after SMOTE:")
print(target_train_balanced.value_counts())

# Train XGBoost model
churn_classifier = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

churn_classifier.fit(features_train_balanced, target_train_balanced)

# Generate predictions
target_predictions = churn_classifier.predict(features_test)

# Evaluate model
model_accuracy = accuracy_score(target_test, target_predictions)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(f"\nAccuracy: {model_accuracy:.4f}\n")
print(classification_report(target_test, target_predictions))

# Save model artifacts
joblib.dump(churn_classifier, "src/models/churn_model.pkl")
joblib.dump(categorical_encoders, "src/models/label_encoders.pkl")

print("\nModel saved successfully")
print("Encoders saved successfully")