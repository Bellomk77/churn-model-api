"""Send realistic traffic to the API, then a drifted batch, and chart the PSI drift scores.

Normal traffic = customers sampled from the dataset.
Drifted traffic = a pricing change + a promo that pulls in new, month-to-month,
fiber customers paying by electronic check (the kind of shift that silently degrades a model).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PREDICTION_LOG", str(ROOT / "data" / "simulation.db"))
Path(os.environ["PREDICTION_LOG"]).unlink(missing_ok=True)

from app.main import app  # noqa: E402
from model.train import FEATURES, load_data  # noqa: E402


def records(df: pd.DataFrame) -> list[dict]:
    out = df[FEATURES].to_dict("records")
    for r in out:
        r["SeniorCitizen"] = int(r["SeniorCitizen"])
        r["tenure"] = int(r["tenure"])
    return out


def drifted(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = df.copy()
    n = len(d)
    new = rng.random(n) < 0.6                      # 60% of traffic is newly acquired customers
    d.loc[new, "tenure"] = rng.integers(0, 12, new.sum())
    d["MonthlyCharges"] = (d["MonthlyCharges"] * 1.15).clip(upper=150).round(2)  # price increase
    d["TotalCharges"] = (d["MonthlyCharges"] * d["tenure"]).round(2)
    d.loc[rng.random(n) < 0.6, "Contract"] = "Month-to-month"
    to_fiber = (rng.random(n) < 0.5) & (d["InternetService"] != "Fiber optic")
    was_none = to_fiber & (d["InternetService"] == "No")
    d.loc[to_fiber, "InternetService"] = "Fiber optic"
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        d.loc[was_none, col] = "No"
    d.loc[rng.random(n) < 0.4, "PaymentMethod"] = "Electronic check"
    return d


def chart(before: dict, after: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = list(after["features"])
    b = [before["features"][f]["psi"] for f in feats]
    a = [after["features"][f]["psi"] for f in feats]
    y = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.barh(y + 0.2, b, 0.4, color="#2a9d8f", label="Normal traffic")
    ax.barh(y - 0.2, a, 0.4, color="#e76f51", label="After promo / price change")
    ax.axvline(0.25, color="#b42318", ls="--", lw=1)
    ax.text(0.26, len(feats) - 0.6, "alert (0.25)", color="#b42318", fontsize=9)
    ax.set_yticks(y, feats)
    ax.invert_yaxis()
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Input drift detected by /monitoring/drift", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X, _ = load_data()
    client = TestClient(app)
    normal = X.sample(800, random_state=1)
    for i in range(0, 800, 200):
        assert client.post("/predict/batch", json={"customers": records(normal.iloc[i:i + 200])}).status_code == 200
    before = client.get("/monitoring/drift", params={"window": 800}).json()
    shifted = drifted(X.sample(800, random_state=2), rng)
    r = client.post("/predict/batch", json={"customers": records(shifted)})
    after = client.get("/monitoring/drift", params={"window": 800}).json()
    mean_p = float(np.mean([p["churn_probability"] for p in r.json()]))
    chart(before, after, ROOT / "docs" / "drift_chart.png")
    print(json.dumps({"before": before["status"], "after": after["status"],
                      "mean_churn_probability_after_shift": round(mean_p, 3),
                      "stats": client.get("/monitoring/stats").json()}, indent=2))
