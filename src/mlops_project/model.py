from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Settings


def build_model(settings: Settings) -> Pipeline:
    """A simple, honest baseline. Swap in a stronger estimator as your project grows."""
    if settings.problem_type == "classification":
        estimator = LogisticRegression(max_iter=settings.max_iter)
    else:
        estimator = LinearRegression()

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


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
