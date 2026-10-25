from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from .gate import require_go_verdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def register_best_model(
    run_id: str | None = None,
    model_name: str | None = None,
    alias: str = "staging",
) -> ModelVersion:
    load_dotenv(PROJECT_ROOT / ".env")

    selected_run_id = run_id
    if selected_run_id is None:
        run_id_path = PROJECT_ROOT / "models" / "run_id.txt"
        if not run_id_path.is_file():
            raise FileNotFoundError(f"MLflow run ID file does not exist: {run_id_path}")
        selected_run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        selected_run_id = selected_run_id.strip()

    if not selected_run_id:
        raise ValueError("Cannot register model: MLflow run ID is empty.")
    require_go_verdict(PROJECT_ROOT, selected_run_id)

    selected_model_name = model_name or os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "telco-churn-model",
    )
    selected_model_name = selected_model_name.strip()
    if not selected_model_name:
        raise ValueError("Registered model name cannot be empty.")
    if not alias.strip():
        raise ValueError("Model alias cannot be empty.")

    client = MlflowClient()
    run = client.get_run(selected_run_id)
    if run.info.status != "FINISHED":
        raise ValueError(
            f"Cannot register model from run {selected_run_id}: "
            f"run status is {run.info.status!r}, expected 'FINISHED'."
        )

    existing_version = next(
        (
            version
            for version in client.search_model_versions(
                f"name='{selected_model_name}'"
            )
            if version.run_id == selected_run_id
        ),
        None,
    )
    if existing_version is None:
        model_uri = f"runs:/{selected_run_id}/model"
        model_version = mlflow.register_model(model_uri, selected_model_name)
    else:
        model_version = existing_version
    registered_run_id = (model_version.run_id or "").strip()
    if not registered_run_id:
        raise RuntimeError(
            f"Model version {model_version.version} was registered without a run_id; "
            f"the {alias!r} alias was not set."
        )
    if registered_run_id != selected_run_id:
        raise RuntimeError(
            f"Registered model run_id {registered_run_id!r} does not match "
            f"requested run_id {selected_run_id!r}; the {alias!r} alias was not set."
        )

    client.set_registered_model_alias(
        selected_model_name,
        alias,
        str(model_version.version),
    )
    print(
        f"Registered version {model_version.version} with "
        f"run_id={registered_run_id}; alias={alias}"
    )
    return model_version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a completed MLflow run's model and assign an alias."
    )
    parser.add_argument(
        "--run-id",
        help="Run ID to register (defaults to models/run_id.txt from the latest DVC run).",
    )
    parser.add_argument(
        "--model-name",
        help="Registered model name (defaults to MLFLOW_REGISTERED_MODEL_NAME or telco-churn-model).",
    )
    parser.add_argument("--alias", default="staging", help="Alias to assign (default: staging).")
    args = parser.parse_args()
    register_best_model(args.run_id, args.model_name, args.alias)


if __name__ == "__main__":
    main()
