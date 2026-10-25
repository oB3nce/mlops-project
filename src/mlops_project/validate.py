import json
from pathlib import Path
import sys

import pandas as pd
from pandera.errors import SchemaErrors

from .schemas import IngestionSchema


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_dataset() -> dict[str, object]:
    raw_path = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
    report_path = PROJECT_ROOT / "reports" / "validation_report.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {raw_path}")

    df = pd.read_csv(raw_path, dtype={"TotalCharges": "string"})
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        IngestionSchema.validate(df, lazy=True)
    except SchemaErrors as exc:
        failure_cases = exc.failure_cases.to_dict(orient="records")
        verdict = {
            "contract": "IngestionSchema",
            "n_rows": int(len(df)),
            "status": "fail",
            "error_count": len(failure_cases),
            "errors": [str(e) for e in failure_cases[:10]],
        }
        report_path.write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("Data validation FAILED. Stopping pipeline.", file=sys.stderr)
        raise SystemExit(1) from exc

    verdict = {
        "contract": "IngestionSchema",
        "n_rows": int(len(df)),
        "status": "pass",
        "errors": [],
    }
    report_path.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Data validation passed: {len(df)} rows.")
    return verdict


def main() -> None:
    validate_dataset()


if __name__ == "__main__":
    main()