import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
customer_churn_data = pd.read_excel("data/raw/telco_churn.xlsx")

print("\nDataset loaded successfully\n")
print(customer_churn_data.head())

# Clean column names
customer_churn_data.columns = customer_churn_data.columns.str.strip()  # Remove leading/trailing spaces

# Replace empty values
for column_name in customer_churn_data.columns:
    customer_churn_data[column_name] = customer_churn_data[column_name].replace(" ", pd.NA)  # Convert spaces to NA

# Fill missing values
customer_churn_data = customer_churn_data.fillna("Unknown")

# Define target column
target_column = "Churn Value"

# Remove leakage columns (data not available at prediction time) and "useless data" 
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

feature_data = customer_churn_data.drop(columns=columns_to_remove)  # Input data (X)
target_data = customer_churn_data[target_column]  # Correct answer (y)

# Encode categorical features
categorical_encoders = {}

for column_name in feature_data.columns:
    if feature_data[column_name].dtype == "object":  # If column is categorical/text
        column_encoder = LabelEncoder()  # Initialize the encoder
        # Convert to string, fit encoder, and transform to numbers
        feature_data[column_name] = column_encoder.fit_transform(feature_data[column_name].astype(str))
        categorical_encoders[column_name] = column_encoder  # Save encoder for future use

# Convert all features to numeric
feature_data = feature_data.apply(pd.to_numeric, errors="coerce")  # Convert non-numeric to NaN
feature_data = feature_data.fillna(0)  # Replace NaN with 0

# Split dataset
features_train, features_test, target_train, target_test = train_test_split(
    feature_data,      # Input features (X)
    target_data,       # Target labels (y)
    test_size=0.2,     # 20% for testing, 80% for training
    random_state=42,   # Fixed seed for reproducibility
    stratify=target_data  # Preserve class balance across splits
)


# Balance training data using SMOTE (Synthetic Minority Over-sampling Technique)
smote_balancer = SMOTE(random_state=42)  # Initialize with fixed seed for reproducibility

# Create synthetic samples for the minority class to balance the dataset
features_train_balanced, target_train_balanced = smote_balancer.fit_resample(
    features_train,   # Original imbalanced features (X)
    target_train      # Original imbalanced labels (y)
)

# Train model
churn_classifier = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    random_state=42,       # Fixed seed for reproducibility
    class_weight="balanced"  # Automatically adjust for imbalanced classes
)

churn_classifier.fit(features_train_balanced, target_train_balanced)  # Train the model

# Generate predictions
target_predictions = churn_classifier.predict(features_test)  # Predict on test set

# Evaluate model
model_accuracy = accuracy_score(target_test, target_predictions)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(f"\nAccuracy: {model_accuracy:.4f}\n")

print(classification_report(target_test, target_predictions))  # Precision, recall, f1-score

# Save model artifacts
joblib.dump(churn_classifier, "src/models/churn_model.pkl")  # Save trained model
joblib.dump(categorical_encoders, "src/models/label_encoders.pkl")  # Save encoders

print("\nModel saved successfully")
print("Encoders saved successfully")