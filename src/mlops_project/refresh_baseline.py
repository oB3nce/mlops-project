from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def refresh_baseline(
    alias: str = "champion",
    model_name: str | None = None,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    selected_model_name = (
        model_name
        or os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
        or "telco-churn-model"
    ).strip()
    selected_alias = alias.strip()
    if not selected_model_name:
        raise ValueError("Registered model name cannot be empty.")
    if not selected_alias:
        raise ValueError("Model alias cannot be empty.")

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(
        selected_model_name,
        selected_alias,
    )
    run_id = (model_version.run_id or "").strip()
    if not run_id:
        raise RuntimeError(
            f"Model {selected_model_name!r} alias {selected_alias!r} "
            "resolved to a version without a run ID."
        )
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise RuntimeError(
            f"Champion run {run_id!r} has status {run.info.status!r}, "
            "expected 'FINISHED'."
        )

    model_metrics = {
        name.removeprefix("model_"): float(value)
        for name, value in run.data.metrics.items()
        if name.startswith("model_")
    }
    if not {"accuracy", "f1"}.issubset(model_metrics):
        raise ValueError(
            f"Champion run {run_id!r} must have model_accuracy and model_f1 "
            "metrics before refreshing the baseline."
        )

    baseline = {
        "source": {
            "model_name": selected_model_name,
            "alias": selected_alias,
            "model_version": str(model_version.version),
            "run_id": run_id,
        },
        "metrics": model_metrics,
    }
    baseline_path = PROJECT_ROOT / "metrics" / "baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = baseline_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(baseline_path)
    print(
        f"Refreshed {baseline_path.relative_to(PROJECT_ROOT)} from "
        f"{selected_model_name}@{selected_alias} "
        f"(run_id={run_id}; model_version={model_version.version})."
    )
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the committed baseline using an MLflow model alias."
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="MLflow model alias to resolve (default: champion).",
    )
    parser.add_argument(
        "--model-name",
        help="Registered model name (defaults to environment or telco-churn-model).",
    )
    args = parser.parse_args()
    refresh_baseline(args.alias, args.model_name)


if __name__ == "__main__":
    main()
