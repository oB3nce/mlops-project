from __future__ import annotations

import configparser
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


mlflow.set_tracking_uri("http://localhost:5500")  # Set the MLflow tracking URI to the local MLflow server

def _load_train_params() -> dict[str, Any]:
    params_path = PROJECT_ROOT / "params.yaml"
    with params_path.open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file) or {}

    train_params = params.get("train", {})
    model_type = train_params.get("model_type")
    if model_type != "random_forest":
        raise ValueError(f"Unsupported train.model_type: {model_type!r}")

    n_estimators = int(train_params["n_estimators"])
    max_depth = train_params.get("max_depth")
    max_depth = None if max_depth is None else int(max_depth)
    random_state = int(train_params["random_state"])
    if n_estimators <= 0:
        raise ValueError("train.n_estimators must be greater than 0.")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("train.max_depth must be greater than 0 or null.")

    return {
        "model_type": model_type,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
    }


def _dataset_metadata() -> tuple[str, str]:
    pointer_path = PROJECT_ROOT / "data" / "raw" / "dataset.csv.dvc"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"DVC pointer does not exist: {pointer_path}")

    with pointer_path.open(encoding="utf-8") as pointer_file:
        pointer = yaml.safe_load(pointer_file) or {}
    outputs = pointer.get("outs", [])
    dataset_output = next(
        (output for output in outputs if output.get("path") == "dataset.csv"),
        None,
    )
    if dataset_output is None or not dataset_output.get("md5"):
        raise ValueError(f"No MD5 for dataset.csv found in {pointer_path}")

    dvc_config_path = PROJECT_ROOT / ".dvc" / "config"
    config = configparser.ConfigParser()
    if not config.read(dvc_config_path, encoding="utf-8"):
        raise FileNotFoundError(f"DVC configuration does not exist: {dvc_config_path}")

    remote_name = config.get("core", "remote", fallback=None)
    if not remote_name:
        raise ValueError(f"No default DVC remote configured in {dvc_config_path}")
    expected_section = f'remote "{remote_name}"'
    remote_section = next(
        (
            section
            for section in config.sections()
            if section.strip("'") == expected_section
        ),
        None,
    )
    if remote_section is None:
        raise ValueError(f"Remote {remote_name!r} is not defined in {dvc_config_path}")
    remote_uri = config.get(remote_section, "url", fallback=None)
    if not remote_uri:
        raise ValueError(f"No URL configured for DVC remote {remote_name!r}")

    return str(dataset_output["md5"]), remote_uri


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty commit SHA.")
    return commit


def _build_model(params: dict[str, Any], problem_type: str) -> Pipeline:
    if problem_type == "classification":
        estimator = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
    elif problem_type == "regression":
        estimator = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
    else:
        raise ValueError(f"Unsupported problem type: {problem_type!r}")

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
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
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_settings(PROJECT_ROOT)
    params = _load_train_params()
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Training data does not exist: {train_path}")

    train_frame = pd.read_csv(train_path)
    if settings.target_column not in train_frame.columns:
        raise KeyError(
            f"Target column {settings.target_column!r} not found in {train_path}."
        )
    labels = train_frame.pop(settings.target_column)
    if labels.isna().any():
        raise ValueError(f"Training target {settings.target_column!r} contains missing values.")
    features = train_frame
    if features.empty or features.shape[1] == 0:
        raise ValueError("Training data has no feature columns.")

    md5, remote_uri = _dataset_metadata()
    commit = _git_commit()
    model = _build_model(params, settings.problem_type)

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn"))
    with mlflow.start_run(run_name="random-forest-training") as active_run:
        mlflow.set_tags(
            {
                "git_commit": commit,
                "dvc_md5": md5,
                "dvc_remote_uri": remote_uri,
            }
        )
        mlflow.log_params(params)
        model.fit(features, labels)

        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.pkl"
        with model_path.open("wb") as model_file:
            pickle.dump(model, model_file)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        run_id_path = models_dir / "run_id.txt"
        temporary_run_id_path = run_id_path.with_suffix(".tmp")
        temporary_run_id_path.write_text(active_run.info.run_id, encoding="utf-8")
        temporary_run_id_path.replace(run_id_path)
        run_id = active_run.info.run_id

    print(f"Trained {params['model_type']} model; MLflow run ID: {run_id}")
    return run_id


def main() -> None:
    train()


if __name__ == "__main__":
    main()
