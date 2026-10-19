from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Settings


def load_dataframe(settings: Settings) -> pd.DataFrame:
    """Load your dataset CSV (set DATA_PATH in .env)."""
    if not settings.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {settings.data_path}\n"
            "Place your dataset under data/ and set DATA_PATH in .env."
        )
    return pd.read_csv(settings.data_path)


def build_dataset(settings: Settings, frame: pd.DataFrame | None = None):
    """Prepare the data and split it into train/test sets."""
    frame = load_dataframe(settings) if frame is None else frame
    if settings.target_column not in frame.columns:
        raise KeyError(
            f"Target column '{settings.target_column}' not in data "
            f"(columns: {list(frame.columns)}). Set TARGET_COLUMN in .env."
        )

    frame = frame.copy()
    for column in frame.select_dtypes(include="object"):
        values = frame[column].astype("string").str.strip()
        non_empty = values.notna() & values.ne("")
        numeric_values = pd.to_numeric(values.mask(values.eq(""), pd.NA), errors="coerce")
        if non_empty.any() and numeric_values.notna().sum() == non_empty.sum():
            frame[column] = numeric_values

    features = frame.drop(columns=[settings.target_column])
    identifier_columns = [
        column
        for column in features
        if column.casefold().endswith("id") and features[column].is_unique
    ]
    features = features.drop(columns=identifier_columns)
    labels = frame[settings.target_column]
    stratify = labels if settings.problem_type == "classification" else None
    return train_test_split(
        features,
        labels,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=stratify,
    )
