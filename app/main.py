"""Churn prediction API.

POST /predict          score one customer (with the top reasons behind the score)
POST /predict/batch    score up to 1,000 customers
GET  /model            model card: version, metrics, why this model was chosen
GET  /monitoring/drift input drift vs training data (PSI)
GET  /monitoring/stats prediction volume, risk mix, latency
GET  /health           liveness + model version
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .monitoring import PredictionLog, drift_report

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.environ.get("MODEL_DIR", ROOT / "model" / "artifacts"))
LOG_PATH = Path(os.environ.get("PREDICTION_LOG", ROOT / "data" / "predictions.db"))

YesNo = Literal["Yes", "No"]
Service = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer. Field names match the source data so the API and training never disagree."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "tenure": 2, "MonthlyCharges": 89.1, "TotalCharges": 178.2, "SeniorCitizen": 0, "gender": "Female",
        "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}})

    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0)
    SeniorCitizen: Literal[0, 1]
    gender: Literal["Female", "Male"]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Service
    OnlineBackup: Service
    DeviceProtection: Service
    TechSupport: Service
    StreamingTV: Service
    StreamingMovies: Service
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


class Prediction(BaseModel):
    churn_probability: float
    risk_band: Literal["high", "medium", "low"]
    top_reasons: list[str]
    model_version: str


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


app = FastAPI(title="Churn Model API", version="1.0.0",
              description="Scores telecom customers for churn risk, explains each score, and monitors drift.")
_model = joblib.load(ART / "model.joblib")
_card = json.loads((ART / "model_card.json").read_text())
_reference = json.loads((ART / "reference.json").read_text())
_log = PredictionLog(LOG_PATH)


def _band(p: float) -> str:
    cuts = _card["risk_bands"]
    return "high" if p >= cuts["high"] else "medium" if p >= cuts["medium"] else "low"


def _reasons(df: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """Per-customer drivers for a linear model: coefficient x feature value, summed per original column."""
    prep, model = _model.named_steps["prep"], _model.named_steps["model"]
    if not hasattr(model, "coef_"):
        return [[] for _ in range(len(df))]
    Z = prep.transform(df)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    contrib = Z * model.coef_[0]
    names = prep.get_feature_names_out()
    out = []
    for row, raw in zip(contrib, df.to_dict("records")):
        per_col: dict[str, float] = {}
        for name, c in zip(names, row):
            col = name.split("__", 1)[1]
            col = next((f for f in raw if col == f or col.startswith(f + "_")), col)
            per_col[col] = per_col.get(col, 0.0) + c
        top = [c for c, v in sorted(per_col.items(), key=lambda kv: -kv[1]) if v > 0][:k]
        out.append([f"{c} = {raw[c]}" for c in top])
    return out


def _score(customers: list[Customer]) -> list[Prediction]:
    t0 = time.perf_counter()
    df = pd.DataFrame([c.model_dump() for c in customers])
    probs = _model.predict_proba(df)[:, 1]
    reasons = _reasons(df)
    latency = (time.perf_counter() - t0) * 1000 / len(customers)
    preds = []
    for c, p, r in zip(customers, probs, reasons):
        pred = Prediction(churn_probability=round(float(p), 4), risk_band=_band(float(p)),
                          top_reasons=r, model_version=_card["model_version"])
        _log.write(pred.model_version, c.model_dump(), pred.churn_probability, pred.risk_band, latency)
        preds.append(pred)
    return preds


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": _card["model_version"], "model": _card["chosen_model"]}


@app.get("/model")
def model_card() -> dict:
    return _card


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch) -> list[Prediction]:
    return _score(batch.customers)


@app.get("/monitoring/drift")
def drift(window: int = 1000) -> dict:
    if not 1 <= window <= 50000:
        raise HTTPException(400, "window must be between 1 and 50000")
    return drift_report(_reference, _log.recent(window))


@app.get("/monitoring/stats")
def stats() -> dict:
    return _log.stats()
