import os
import tempfile

os.environ["PREDICTION_LOG"] = os.path.join(tempfile.mkdtemp(), "test.db")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import Customer, app  # noqa: E402
from app.monitoring import drift_report, psi  # noqa: E402

client = TestClient(app)
EXAMPLE = Customer.model_config["json_schema_extra"]["example"]
LOYAL = {**EXAMPLE, "tenure": 70, "TotalCharges": 4500.0, "MonthlyCharges": 64.0, "Contract": "Two year",
         "InternetService": "DSL", "PaymentMethod": "Credit card (automatic)", "OnlineSecurity": "Yes",
         "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No", "Partner": "Yes", "Dependents": "Yes"}


def test_health_and_model_card():
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/model").json()
    assert card["holdout"]["roc_auc"] > 0.80 and card["chosen_model"] in card["cv_roc_auc"]


def test_predict_returns_probability_band_and_reasons():
    r = client.post("/predict", json=EXAMPLE).json()
    assert 0 <= r["churn_probability"] <= 1 and r["risk_band"] in {"high", "medium", "low"}
    assert r["top_reasons"], "a linear model should explain its score"


def test_model_ranks_risky_customer_above_loyal_customer():
    risky = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
    loyal = client.post("/predict", json=LOYAL).json()["churn_probability"]
    assert risky > 0.5 > loyal


@pytest.mark.parametrize("patch", [
    {"tenure": -1},
    {"Contract": "Lifetime"},
    {"MonthlyCharges": "a lot"},
    {"SeniorCitizen": 2},
])
def test_invalid_input_is_rejected_with_422(patch):
    assert client.post("/predict", json={**EXAMPLE, **patch}).status_code == 422


def test_missing_field_is_rejected():
    bad = {k: v for k, v in EXAMPLE.items() if k != "tenure"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_limits():
    assert len(client.post("/predict/batch", json={"customers": [EXAMPLE] * 5}).json()) == 5
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_psi_math():
    ref = [0.25, 0.25, 0.25, 0.25]
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_drift_report_flags_a_shift_and_not_stable_traffic():
    import json
    from pathlib import Path
    ref = json.loads((Path(__file__).resolve().parent.parent / "model/artifacts/reference.json").read_text())
    stable = [{**LOYAL, "tenure": int(t)} for t in np.linspace(0, 72, 200)]
    assert drift_report(ref, stable[:10])["status"] == "insufficient_data"
    shifted = [{**EXAMPLE, "tenure": 1} for _ in range(200)]
    rep = drift_report(ref, shifted)
    assert rep["status"] == "alert" and rep["features"]["tenure"]["status"] == "alert"


def test_predictions_are_logged():
    before = client.get("/monitoring/stats").json()["predictions"]
    client.post("/predict", json=EXAMPLE)
    assert client.get("/monitoring/stats").json()["predictions"] == before + 1
