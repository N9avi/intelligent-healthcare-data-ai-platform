import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

DATA_PATH = "data/processed/silver_patient_encounters.csv"
MODEL_DIR = "src/ml/artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n=== HEALTHCARE AI/ML - READMISSION PREDICTION ===")

df = pd.read_csv(DATA_PATH)

features = [
    "age",
    "gender",
    "diagnosis",
    "previous_admissions",
    "length_of_stay",
    "num_medications",
    "abnormal_lab_results",
    "insurance_type",
    "total_cost"
]

target = "readmitted_30_days"

X = df[features]
y = df[target]

numeric_features = [
    "age",
    "previous_admissions",
    "length_of_stay",
    "num_medications",
    "abnormal_lab_results",
    "total_cost"
]

categorical_features = [
    "gender",
    "diagnosis",
    "insurance_type"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, predictions), 4),
    "precision": round(precision_score(y_test, predictions), 4),
    "recall": round(recall_score(y_test, predictions), 4),
    "f1_score": round(f1_score(y_test, predictions), 4),
    "roc_auc": round(roc_auc_score(y_test, probabilities), 4)
}

print("\n=== MODEL PERFORMANCE ===")

for metric, value in metrics.items():
    print(f"{metric.upper():12}: {value}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(
    pipeline,
    f"{MODEL_DIR}/readmission_model.joblib"
)

with open(f"{MODEL_DIR}/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print(f"Model: {MODEL_DIR}/readmission_model.joblib")
print(f"Metrics: {MODEL_DIR}/metrics.json")

sample_patient = X_test.iloc[[0]]

risk_probability = pipeline.predict_proba(sample_patient)[0][1]

print("\n=== SAMPLE AI PREDICTION ===")
print(f"Diagnosis: {sample_patient.iloc[0]['diagnosis']}")
print(f"Age: {sample_patient.iloc[0]['age']}")
print(f"Previous Admissions: {sample_patient.iloc[0]['previous_admissions']}")
print(f"30-Day Readmission Risk: {risk_probability * 100:.2f}%")

if risk_probability >= 0.70:
    print("Risk Level: HIGH")
elif risk_probability >= 0.40:
    print("Risk Level: MEDIUM")
else:
    print("Risk Level: LOW")
