from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.entities import Run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD5_PATTERN = re.compile(r"[0-9a-fA-F]{32}\Z")


def find_runs_for_data(md5: str, experiment_name: str | None = None) -> list[Run]:
    load_dotenv(PROJECT_ROOT / ".env")

    normalized_md5 = md5.strip()
    if not MD5_PATTERN.fullmatch(normalized_md5):
        raise ValueError("DVC md5 must be exactly 32 hexadecimal characters.")
    normalized_md5 = normalized_md5.lower()

    selected_experiment = (
        experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
    ).strip()
    if not selected_experiment:
        raise ValueError("MLflow experiment name cannot be empty.")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(selected_experiment)
    if experiment is None:
        raise LookupError(f"MLflow experiment {selected_experiment!r} does not exist.")

    return list(
        client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.dvc_md5 = '{normalized_md5}'",
            order_by=["start_time DESC"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List MLflow runs trained on a dataset with the given DVC MD5."
    )
    parser.add_argument("md5", help="32-character MD5 hash from the DVC pointer.")
    parser.add_argument(
        "--experiment",
        help="MLflow experiment name (defaults to MLFLOW_EXPERIMENT_NAME or telco-churn).",
    )
    args = parser.parse_args()

    runs = find_runs_for_data(args.md5, args.experiment)
    if not runs:
        print(f"No MLflow runs found for DVC md5 {args.md5.strip().lower()}.")
        return

    for run in runs:
        print(f"run_id: {run.info.run_id}")
        print(f"params: {json.dumps(run.data.params, sort_keys=True)}")
        print(f"metrics: {json.dumps(run.data.metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
