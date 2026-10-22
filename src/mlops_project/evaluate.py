from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

from .config import load_settings
from .model import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate() -> dict[str, float]:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_settings(PROJECT_ROOT)
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    run_id_path = PROJECT_ROOT / "models" / "run_id.txt"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    for required_path in (model_path, run_id_path, test_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required evaluation input does not exist: {required_path}")

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    test_frame = pd.read_csv(test_path)
    if settings.target_column not in test_frame.columns:
        raise KeyError(
            f"Target column {settings.target_column!r} not found in {test_path}."
        )
    labels = test_frame.pop(settings.target_column)
    if labels.isna().any():
        raise ValueError(f"Test target {settings.target_column!r} contains missing values.")
    if test_frame.empty or test_frame.shape[1] == 0:
        raise ValueError("Test data has no feature columns.")

    metrics = evaluate_model(model, test_frame, labels, settings)
    numeric_metrics = {name: float(value) for name, value in metrics.items()}

    metrics_path = PROJECT_ROOT / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics_path = metrics_path.with_suffix(".tmp")
    temporary_metrics_path.write_text(
        json.dumps(numeric_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metrics_path.replace(metrics_path)

    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"MLflow run ID is empty: {run_id_path}")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(numeric_metrics)

    print(f"Evaluation metrics logged to MLflow run {run_id}: {numeric_metrics}")
    return numeric_metrics


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
