from __future__ import annotations

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings


def build_model(settings: Settings) -> Pipeline:
    """A simple, honest baseline. Swap in a stronger estimator as your project grows."""
    if settings.problem_type == "classification":
        estimator = LogisticRegression(max_iter=settings.max_iter)
    else:
        estimator = LinearRegression()
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def evaluate_model(model: Pipeline, x_test, y_test, settings: Settings) -> dict:
    predictions = model.predict(x_test)
    if settings.problem_type == "classification":
        return {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        }
    return {
        "rmse": round(float(root_mean_squared_error(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }
