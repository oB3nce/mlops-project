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
    """Split into train/test. Stratifies on the target for classification."""
    frame = load_dataframe(settings) if frame is None else frame
    if settings.target_column not in frame.columns:
        raise KeyError(
            f"Target column '{settings.target_column}' not in data "
            f"(columns: {list(frame.columns)}). Set TARGET_COLUMN in .env."
        )
    features = frame.drop(columns=[settings.target_column])
    labels = frame[settings.target_column]
    stratify = labels if settings.problem_type == "classification" else None
    return train_test_split(
        features,
        labels,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=stratify,
    )
