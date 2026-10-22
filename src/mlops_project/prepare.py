from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .config import load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_params() -> tuple[float, int]:
    params_path = PROJECT_ROOT / "params.yaml"
    with params_path.open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file) or {}

    prepare_params = params.get("prepare", {})
    test_size = float(prepare_params["test_size"])
    random_state = int(prepare_params["random_state"])
    if not 0.0 < test_size < 1.0:
        raise ValueError("prepare.test_size must be between 0 and 1 (exclusive).")
    return test_size, random_state


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.select_dtypes(include="object"):
        values = frame[column].astype("string").str.strip()
        non_empty = values.notna() & values.ne("")
        numeric_values = pd.to_numeric(values.mask(values.eq(""), pd.NA), errors="coerce")
        if non_empty.any() and numeric_values.notna().sum() == non_empty.sum():
            frame[column] = numeric_values
        else:
            frame[column] = values.replace("", pd.NA)
    return frame


def prepare() -> tuple[Path, Path]:
    settings = load_settings(PROJECT_ROOT)
    data_path = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {data_path}")

    frame = _normalize_columns(pd.read_csv(data_path))
    if settings.target_column not in frame.columns:
        raise KeyError(
            f"Target column {settings.target_column!r} not found in {data_path}. "
            f"Available columns: {list(frame.columns)}"
        )

    if frame[settings.target_column].isna().any():
        raise ValueError(f"Target column {settings.target_column!r} contains missing values.")

    features = frame.drop(columns=[settings.target_column])
    identifier_columns = [
        column
        for column in features
        if column.casefold().endswith("id") and features[column].is_unique
    ]
    features = features.drop(columns=identifier_columns)
    prepared = pd.concat([features, frame[settings.target_column]], axis=1)

    test_size, random_state = _load_prepare_params()
    labels = prepared[settings.target_column]
    stratify = labels if settings.problem_type == "classification" else None
    train_frame, test_frame = train_test_split(
        prepared,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    print(
        f"Prepared {len(train_frame)} training and {len(test_frame)} test rows "
        f"from {data_path.name}; excluded identifier columns: {identifier_columns}."
    )
    return train_path, test_path


def main() -> None:
    prepare()


if __name__ == "__main__":
    main()
