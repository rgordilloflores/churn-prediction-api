"""
API de inferencia - Despliegue del modelo de predicción de churn.
Modelo: Regresión Logística (umbral = 0.50), consistente con las
secciones 3.3-3.7 del informe.
Sección 3.8 (Despliegue) y Sección 4 (Pruebas de Inferencia).

Cómo ejecutar localmente:
    uvicorn app:app --host 0.0.0.0 --port 8001
"""
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Churn Prediction API - Regresión Logística",
    description="Predice la probabilidad de churn de un cliente de telecomunicaciones.",
    version="1.0.0",
)

model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
NUMERIC_COLUMNS = joblib.load("numeric_columns.pkl")


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def preprocess(customer: Customer) -> pd.DataFrame:
    """Replica exactamente la transformación de la sección 2.3 del informe:
    mapeo binario directo + One-Hot Encoding (drop_first=True) + escalado
    de variables numéricas ajustado sobre el set de entrenamiento."""
    d = customer.dict()
    row = {
        "gender": 1 if d["gender"] == "Male" else 0,
        "SeniorCitizen": d["SeniorCitizen"],
        "Partner": 1 if d["Partner"] == "Yes" else 0,
        "Dependents": 1 if d["Dependents"] == "Yes" else 0,
        "tenure": d["tenure"],
        "PhoneService": 1 if d["PhoneService"] == "Yes" else 0,
        "PaperlessBilling": 1 if d["PaperlessBilling"] == "Yes" else 0,
        "MonthlyCharges": d["MonthlyCharges"],
        "TotalCharges": d["TotalCharges"],
    }
    # Dummies manuales equivalentes a pd.get_dummies(drop_first=True)
    row["MultipleLines_No phone service"] = int(d["MultipleLines"] == "No phone service")
    row["MultipleLines_Yes"] = int(d["MultipleLines"] == "Yes")
    row["InternetService_Fiber optic"] = int(d["InternetService"] == "Fiber optic")
    row["InternetService_No"] = int(d["InternetService"] == "No")
    for base in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        row[f"{base}_No internet service"] = int(d[base] == "No internet service")
        row[f"{base}_Yes"] = int(d[base] == "Yes")
    row["Contract_One year"] = int(d["Contract"] == "One year")
    row["Contract_Two year"] = int(d["Contract"] == "Two year")
    row["PaymentMethod_Credit card (automatic)"] = int(d["PaymentMethod"] == "Credit card (automatic)")
    row["PaymentMethod_Electronic check"] = int(d["PaymentMethod"] == "Electronic check")
    row["PaymentMethod_Mailed check"] = int(d["PaymentMethod"] == "Mailed check")

    df = pd.DataFrame([row])[FEATURE_COLUMNS]
    df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])
    return df


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "modelo": "LogisticRegression"}


@app.post("/predict")
def predict(customer: Customer):
    row = preprocess(customer)
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.50)  # umbral = 0.50, consistente con sección 3.3
    return {
        "churn_prediction": pred,
        "churn_probability": round(proba, 4),
        "umbral_decision": 0.50,
        "risk_level": "Alto" if proba >= 0.5 else ("Medio" if proba >= 0.3 else "Bajo"),
    }


@app.post("/predict_batch")
def predict_batch(customers: list[Customer]):
    rows = pd.concat([preprocess(c) for c in customers], ignore_index=True)
    probas = model.predict_proba(rows)[:, 1]
    return [
        {"index": i, "churn_prediction": int(p >= 0.50), "churn_probability": round(float(p), 4)}
        for i, p in enumerate(probas)
    ]
