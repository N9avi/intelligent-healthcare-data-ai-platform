from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

MODEL_PATH = "src/ml/artifacts/readmission_model.joblib"

app = FastAPI(
    title="Healthcare AI Readmission API",
    description="AI-powered 30-day hospital readmission risk prediction service",
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)


class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=120)
    gender: str
    diagnosis: str
    previous_admissions: int = Field(..., ge=0)
    length_of_stay: int = Field(..., ge=1)
    num_medications: int = Field(..., ge=0)
    abnormal_lab_results: int = Field(..., ge=0)
    insurance_type: str
    total_cost: float = Field(..., ge=0)


@app.get("/")
def home():
    return {
        "service": "Intelligent Healthcare AI Platform",
        "status": "running",
        "model": "30-Day Readmission Risk Prediction"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_readmission(patient: PatientData):

    patient_df = pd.DataFrame([patient.model_dump()])

    probability = float(
        model.predict_proba(patient_df)[0][1]
    )

    prediction = int(probability >= 0.5)

    if probability >= 0.70:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "prediction": prediction,
        "readmission_probability": round(probability, 4),
        "readmission_probability_percent": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendation": (
            "Prioritize discharge follow-up and care coordination."
            if risk_level == "HIGH"
            else "Continue standard monitoring and follow-up."
        )
}
