"""Train, compare and package the churn model.

Outputs (in model/artifacts/):
- model.joblib       the fitted sklearn Pipeline (preprocessing + model)
- model_card.json    version, data hash, chosen model, metrics, and why it was chosen
- reference.json     training distributions used for drift monitoring (PSI)
- gains_chart.png    business view: share of churners caught vs share of customers contacted
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "telco_churn.csv"
ART = ROOT / "model" / "artifacts"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
TOP_SHARE = 0.20  # retention team can call the riskiest 20% of customers each month
EXPLAINABILITY_MARGIN = 0.01  # prefer the explainable model if it is within 0.01 AUC of the best


def load_data(path: Path = DATA) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # TotalCharges is blank for brand-new customers (tenure 0): they have been billed nothing yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)
    return df[FEATURES], (df["Churn"] == "Yes").astype(int)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.5),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=0),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1),
    }


def capture_at_top(y_true: np.ndarray, scores: np.ndarray, share: float) -> tuple[float, float]:
    """Recall and precision if we contact only the top `share` of customers by risk."""
    n = int(len(scores) * share)
    top = np.argsort(-scores)[:n]
    hits = y_true[top].sum()
    return hits / y_true.sum(), hits / n


def reference_stats(X: pd.DataFrame) -> dict:
    ref = {"numeric": {}, "categorical": {}}
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        edges = np.unique(np.quantile(X[col], np.linspace(0, 1, 11)))
        counts, _ = np.histogram(X[col], bins=edges)
        ref["numeric"][col] = {"edges": edges.tolist(), "proportions": (counts / counts.sum()).tolist()}
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        ref["categorical"][col] = X[col].value_counts(normalize=True).to_dict()
    return ref


def gains_chart(y: np.ndarray, scores: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-scores)
    cum = np.cumsum(y[order]) / y.sum()
    share = np.arange(1, len(y) + 1) / len(y)
    r20 = cum[int(len(y) * TOP_SHARE) - 1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    ax.plot(share * 100, cum * 100, color="#1f3b73", lw=2.5, label="Model ranking")
    ax.plot([0, 100], [0, 100], color="#8d99ae", ls="--", lw=1.2, label="Random outreach")
    ax.scatter([TOP_SHARE * 100], [r20 * 100], color="#e76f51", zorder=3)
    ax.annotate(f"Call the riskiest 20%\n→ reach {r20:.0%} of churners", xy=(20, r20 * 100), xytext=(32, r20 * 100 - 22),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_xlabel("% of customers contacted (highest risk first)")
    ax.set_ylabel("% of churners reached")
    ax.set_title("Cumulative gains on the hold-out set", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = {}
    for name, est in candidates().items():
        s = cross_val_score(Pipeline([("prep", preprocessor()), ("model", est)]), X_tr, y_tr, cv=cv, scoring="roc_auc")
        cv_scores[name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}

    best = max(cv_scores, key=lambda k: cv_scores[k]["mean"])
    chosen = best
    reason = "highest cross-validated ROC-AUC"
    if best != "logistic_regression" and cv_scores[best]["mean"] - cv_scores["logistic_regression"]["mean"] < EXPLAINABILITY_MARGIN:
        chosen = "logistic_regression"
        reason = (f"within {EXPLAINABILITY_MARGIN} AUC of the best model ({best}), and it gives per-customer "
                  "explanations the retention team can act on")

    pipe = Pipeline([("prep", preprocessor()), ("model", candidates()[chosen])]).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    recall20, precision20 = capture_at_top(y_te.to_numpy(), p, TOP_SHARE)
    high_cut = float(np.quantile(pipe.predict_proba(X_tr)[:, 1], 1 - TOP_SHARE))
    med_cut = float(np.quantile(pipe.predict_proba(X_tr)[:, 1], 0.5))

    card = {
        "model_version": datetime.now(timezone.utc).strftime("%Y.%m.%d"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"source": "IBM Telco Customer Churn sample (Apache-2.0)", "rows": int(len(X)),
                 "churn_rate": round(float(y.mean()), 4),
                 "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()[:16]},
        "features": FEATURES,
        "cv_roc_auc": cv_scores,
        "chosen_model": chosen,
        "choice_reason": reason,
        "holdout": {
            "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
            "pr_auc": round(float(average_precision_score(y_te, p)), 4),
            "brier": round(float(brier_score_loss(y_te, p)), 4),
            "baseline_pr_auc": round(float(y_te.mean()), 4),
            f"recall_at_top_{int(TOP_SHARE * 100)}pct": round(float(recall20), 4),
            f"precision_at_top_{int(TOP_SHARE * 100)}pct": round(float(precision20), 4),
        },
        "risk_bands": {"high": round(high_cut, 4), "medium": round(med_cut, 4)},
    }
    joblib.dump(pipe, ART / "model.joblib")
    (ART / "model_card.json").write_text(json.dumps(card, indent=2))
    (ART / "reference.json").write_text(json.dumps(reference_stats(X_tr), indent=2))
    gains_chart(y_te.to_numpy(), p, ROOT / "docs" / "gains_chart.png")
    return card


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
