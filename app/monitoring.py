"""Prediction logging and drift detection.

Every prediction is logged to SQLite. Drift is measured with the Population
Stability Index (PSI), comparing recent inputs to the training distribution:
    PSI < 0.10  stable · 0.10 - 0.25  moderate shift · > 0.25  significant shift (investigate / retrain)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

PSI_WARN, PSI_ALERT = 0.10, 0.25
_EPS = 1e-4


def psi(expected: list[float] | np.ndarray, actual: list[float] | np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


class PredictionLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, model_version TEXT,
                features TEXT, probability REAL, risk_band TEXT, latency_ms REAL)""")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, check_same_thread=False)

    def write(self, model_version: str, features: dict, probability: float, band: str, latency_ms: float) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO predictions (model_version, features, probability, risk_band, latency_ms) VALUES (?,?,?,?,?)",
                      (model_version, json.dumps(features), probability, band, latency_ms))

    def recent(self, limit: int = 1000) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT features FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            n, avg_p, p50 = c.execute("SELECT count(*), avg(probability), avg(latency_ms) FROM predictions").fetchone()
            bands = dict(c.execute("SELECT risk_band, count(*) FROM predictions GROUP BY 1").fetchall())
        return {"predictions": n, "mean_probability": round(avg_p or 0, 4), "mean_latency_ms": round(p50 or 0, 2),
                "risk_bands": bands}


def drift_report(reference: dict, recent: list[dict], min_rows: int = 100) -> dict:
    if len(recent) < min_rows:
        return {"status": "insufficient_data", "rows": len(recent), "min_rows": min_rows, "features": {}}
    features = {}
    for col, ref in reference["numeric"].items():
        vals = np.array([r[col] for r in recent], dtype=float)
        edges = np.array(ref["edges"], dtype=float)
        counts, _ = np.histogram(np.clip(vals, edges[0], edges[-1]), bins=edges)
        features[col] = psi(ref["proportions"], counts / counts.sum())
    for col, ref in reference["categorical"].items():
        cats = list(ref)
        actual = np.array([sum(1 for r in recent if r[col] == c) for c in cats], dtype=float) / len(recent)
        features[col] = psi([ref[c] for c in cats], actual)
    worst = max(features.values())
    status = "alert" if worst > PSI_ALERT else "warn" if worst > PSI_WARN else "stable"
    return {"status": status, "rows": len(recent),
            "features": {k: {"psi": round(v, 4), "status": "alert" if v > PSI_ALERT else "warn" if v > PSI_WARN else "stable"}
                         for k, v in sorted(features.items(), key=lambda kv: -kv[1])}}
