from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {dataset_path}")

    original_bytes = dataset_path.read_bytes()
    try:
        frame = pd.read_csv(dataset_path, dtype={"TotalCharges": "string"})
        if frame.empty or "MonthlyCharges" not in frame.columns:
            raise ValueError("Dataset must contain rows and a MonthlyCharges column.")
        frame.loc[frame.index[0], "MonthlyCharges"] = -1
        frame.to_csv(dataset_path, index=False)
        print("Temporarily changed the first MonthlyCharges value to -1.")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            ["uv", "run", "dvc", "repro"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        dataset_path.write_bytes(original_bytes)
        if dataset_path.read_bytes() != original_bytes:
            raise RuntimeError("Failed to restore the original dataset bytes.")
        print("Restored the original dataset.")

    output = result.stdout + result.stderr
    print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode == 0:
        raise RuntimeError("Expected dvc repro to fail on invalid input, but it succeeded.")
    if "Data validation FAILED" not in output or "validate" not in output:
        raise RuntimeError(
            "dvc repro failed, but not at the expected ingestion validation gate."
        )
    print("Confirmed: dvc repro stopped at the ingestion validation gate.")


if __name__ == "__main__":
    main()
