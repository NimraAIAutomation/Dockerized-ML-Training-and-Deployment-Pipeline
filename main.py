import json
import joblib
import requests
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from api.schemas import PatientData

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")

with open(ARTIFACTS_DIR / "metadata.json") as file:
    metadata = json.load(file)

app = FastAPI(
    title="Breast Cancer Prediction API",
    version=metadata["model_version"]
)

@app.get("/")
def home():
    return {"message": "ML Prediction API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": metadata["model_name"],
        "model_version": metadata["model_version"]
    }

@app.post("/predict")
def predict(patient: PatientData):

    try:
        payload = patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()
        feature_names = getattr(model, "feature_names_in_", None)

        renamed_payload = {
            key.replace("_", " "): value
            for key, value in payload.items()
        }

        if feature_names is not None:
            input_data = pd.DataFrame([renamed_payload], columns=feature_names)
        else:
            input_data = pd.DataFrame([renamed_payload])

        prediction = int(model.predict(input_data)[0])

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        return {
            "prediction": prediction,
            "class": "Malignant" if prediction == 1 else "Benign",
            "malignant_probability": round(probability, 4),
            "model_version": metadata["model_version"]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
