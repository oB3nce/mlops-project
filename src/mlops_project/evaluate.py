from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from .config import load_settings
from .model import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_evaluation_params() -> dict[str, Any]:
    params_path = PROJECT_ROOT / "params.yaml"
    with params_path.open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file) or {}

    evaluation = params.get("evaluate", {})
    positive_label = evaluation.get("positive_label")
    slice_definitions = evaluation.get("slices")
    fairness_slice = evaluation.get("fairness_slice")
    if not isinstance(positive_label, str) or not positive_label:
        raise ValueError("evaluate.positive_label must be a non-empty string.")
    if not isinstance(slice_definitions, list) or len(slice_definitions) < 2:
        raise ValueError("evaluate.slices must define at least two slice dimensions.")

    names: set[str] = set()
    for definition in slice_definitions:
        if not isinstance(definition, dict):
            raise ValueError("Each evaluate.slices item must be a mapping.")
        name = definition.get("name")
        column = definition.get("column")
        groups = definition.get("groups")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Slice names must be non-empty and unique: {name!r}")
        if not isinstance(column, str) or not column:
            raise ValueError(f"Slice {name!r} must define a raw column.")
        if not isinstance(groups, dict) or len(groups) < 2:
            raise ValueError(f"Slice {name!r} must define at least two groups.")
        names.add(name)
    if fairness_slice not in names:
        raise ValueError("evaluate.fairness_slice must name one of evaluate.slices.")

    return {
        "positive_label": positive_label,
        "fairness_slice": fairness_slice,
        "slices": slice_definitions,
    }


def _classification_metrics(
    labels: pd.Series,
    predictions: np.ndarray,
    positive_label: str,
) -> dict[str, float | int | None]:
    positives = labels.eq(positive_label)
    positive_count = int(positives.sum())
    predicted_positive = pd.Series(predictions, index=labels.index).eq(positive_label)
    true_positives = int((positives & predicted_positive).sum())
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(
            float(
                precision_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "selection_rate": round(float(predicted_positive.mean()), 4),
        "tpr": round(true_positives / positive_count, 4) if positive_count else None,
    }


def _regression_metrics(
    labels: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": round(float(root_mean_squared_error(labels, predictions)), 4),
        "mae": round(float(mean_absolute_error(labels, predictions)), 4),
        "r2": round(float(r2_score(labels, predictions)), 4),
    }


def _constant_baseline(
    training_labels: pd.Series,
    test_labels: pd.Series,
    problem_type: str,
    positive_label: str,
) -> dict[str, Any]:
    if problem_type == "classification":
        prediction = training_labels.mode().iloc[0]
        predictions = np.full(len(test_labels), prediction, dtype=object)
        metrics = _classification_metrics(test_labels, predictions, positive_label)
    else:
        prediction = float(training_labels.mean())
        predictions = np.full(len(test_labels), prediction)
        metrics = _regression_metrics(test_labels, predictions)
    return {"prediction": prediction, "metrics": metrics}


def _calculate_slice_metrics(
    raw_test: pd.DataFrame,
    labels: pd.Series,
    predictions: np.ndarray,
    definitions: list[dict[str, Any]],
    problem_type: str,
    positive_label: str,
    fairness_slice: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for definition in definitions:
        slice_name = definition["name"]
        column = definition["column"]
        if column not in raw_test.columns:
            raise KeyError(
                f"Slice {slice_name!r} requires raw column {column!r}, "
                f"which is missing from the raw test data."
            )

        for group_name, group_value in definition["groups"].items():
            mask = raw_test[column].eq(group_value).fillna(False)
            group_labels = labels.loc[mask]
            group_predictions = predictions[mask.to_numpy()]
            n = int(mask.sum())
            row: dict[str, Any] = {
                "slice": slice_name,
                "column": column,
                "group": group_name,
                "value": group_value,
                "n": n,
            }

            if problem_type == "classification":
                positive_count = int(group_labels.eq(positive_label).sum())
                row["positive_count"] = positive_count
                if n:
                    metrics = _classification_metrics(
                        group_labels,
                        group_predictions,
                        positive_label,
                    )
                    row["metrics"] = metrics
                    if slice_name == fairness_slice:
                        fairness_rows.append(row)
                else:
                    row["metrics"] = {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "selection_rate": None,
                        "tpr": None,
                    }
            else:
                row["target_std"] = (
                    round(float(group_labels.std(ddof=0)), 4) if n else None
                )
                row["metrics"] = (
                    _regression_metrics(group_labels, group_predictions) if n else {}
                )
            rows.append(row)

    fairness_gap: dict[str, float | str | None] | None = None
    if problem_type == "classification":
        rates = {
            metric: [
                float(row["metrics"][metric])
                for row in fairness_rows
                if row["metrics"][metric] is not None
            ]
            for metric in ("selection_rate", "tpr")
        }
        fairness_gap = {
            "slice": fairness_slice,
            "selection_rate_gap": (
                round(max(rates["selection_rate"]) - min(rates["selection_rate"]), 4)
                if len(rates["selection_rate"]) >= 2
                else None
            ),
            "tpr_gap": (
                round(max(rates["tpr"]) - min(rates["tpr"]), 4)
                if len(rates["tpr"]) >= 2
                else None
            ),
        }

    return {
        "slice_definitions": definitions,
        "slices": rows,
        "fairness_gap": fairness_gap,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def evaluate() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    settings = load_settings(PROJECT_ROOT)
    evaluation_params = _load_evaluation_params()

    model_path = PROJECT_ROOT / "models" / "model.pkl"
    run_id_path = PROJECT_ROOT / "models" / "run_id.txt"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    raw_test_path = PROJECT_ROOT / "data" / "processed" / "test_raw_slices.csv"
    for required_path in (model_path, run_id_path, test_path, raw_test_path):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required evaluation input does not exist: {required_path}"
            )

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    test_frame = pd.read_csv(test_path)
    raw_test = pd.read_csv(raw_test_path, dtype={"TotalCharges": "string"})
    if settings.target_column not in test_frame.columns:
        raise KeyError(
            f"Target column {settings.target_column!r} not found in {test_path}."
        )
    if settings.target_column not in raw_test.columns:
        raise KeyError(
            f"Target column {settings.target_column!r} not found in {raw_test_path}."
        )

    labels = test_frame.pop(settings.target_column)
    raw_labels = raw_test[settings.target_column]
    if labels.isna().any() or raw_labels.isna().any():
        raise ValueError(f"Test target {settings.target_column!r} contains missing values.")
    if len(test_frame) != len(raw_test) or not labels.reset_index(drop=True).equals(
        raw_labels.reset_index(drop=True)
    ):
        raise ValueError("Raw slice rows and model test rows are not aligned.")
    if test_frame.empty or test_frame.shape[1] == 0:
        raise ValueError("Test data has no feature columns.")

    training_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    if not training_path.is_file():
        raise FileNotFoundError(f"Training data does not exist: {training_path}")
    training_frame = pd.read_csv(training_path)
    if settings.target_column not in training_frame.columns:
        raise KeyError(
            f"Target column {settings.target_column!r} not found in {training_path}."
        )
    training_labels = training_frame[settings.target_column]

    definitions = evaluation_params["slices"]
    for definition in definitions:
        if definition["column"] not in raw_test.columns:
            raise KeyError(
                f"Configured slice column {definition['column']!r} is absent "
                f"from {raw_test_path}."
            )

    predictions = model.predict(test_frame)
    model_metrics = evaluate_model(
        model,
        test_frame,
        labels,
        settings,
    )
    if settings.problem_type == "classification":
        positive_label = evaluation_params["positive_label"]
        if positive_label not in set(labels.unique()):
            raise ValueError(
                f"Positive label {positive_label!r} not found in test target values."
            )
    baseline_metrics = _constant_baseline(
        training_labels,
        labels,
        settings.problem_type,
        evaluation_params["positive_label"],
    )

    aggregate = {
        "model": {name: float(value) for name, value in model_metrics.items()},
        "constant_baseline": baseline_metrics,
    }
    slices = _calculate_slice_metrics(
        raw_test,
        raw_labels,
        predictions,
        definitions,
        settings.problem_type,
        evaluation_params["positive_label"],
        evaluation_params["fairness_slice"],
    )

    _write_json(PROJECT_ROOT / "metrics" / "metrics.json", aggregate)
    _write_json(PROJECT_ROOT / "metrics" / "slices.json", slices)

    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"MLflow run ID is empty: {run_id_path}")
    logged_metrics = {
        f"model_{name}": float(value)
        for name, value in model_metrics.items()
    }
    logged_metrics.update(
        {
            f"baseline_{name}": float(value)
            for name, value in baseline_metrics["metrics"].items()
        }
    )
    if slices["fairness_gap"] is not None:
        for name in ("selection_rate_gap", "tpr_gap"):
            value = slices["fairness_gap"][name]
            if value is not None:
                logged_metrics[name] = float(value)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(logged_metrics)

    print(f"Aggregate metrics: {aggregate}")
    print(f"Slice metrics and fairness gap written for MLflow run {run_id}.")
    return {"metrics": aggregate, "slices": slices}


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
