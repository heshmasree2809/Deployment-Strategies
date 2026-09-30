import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI

app = FastAPI(title="Kidney Disease Prediction API")

BASE_DIR = Path(__file__).resolve().parent

if (BASE_DIR / "kidney_disease_model.pkl").exists():
    MODEL_PATH = BASE_DIR / "kidney_disease_model.pkl"
else:
    MODEL_PATH = BASE_DIR.parent / "kidney_disease_model.pkl"

model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {
        "message": "Kidney Disease Prediction API is running"
    }

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)

    return {
        "prediction": prediction[0]
    }
