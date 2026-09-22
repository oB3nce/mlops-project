from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROBLEM_TYPES = ("classification", "regression")


@dataclass(frozen=True)
class Settings:
    """Project configuration. Override via environment variables / `.env`."""

    data_path: Path = Path("data/dataset.csv")
    target_column: str = "target"
    problem_type: str = "classification"
    test_size: float = 0.25
    random_seed: int = 42
    max_iter: int = 1000


def load_settings(project_root: Path | None = None) -> Settings:
    base_path = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(base_path / ".env")

    settings = Settings(
        data_path=base_path / os.getenv("DATA_PATH", "data/dataset.csv"),
        target_column=os.getenv("TARGET_COLUMN", "target"),
        problem_type=os.getenv("PROBLEM_TYPE", "classification").strip().lower(),
        test_size=float(os.getenv("TEST_SIZE", "0.25")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        max_iter=int(os.getenv("MAX_ITER", "1000")),
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if settings.problem_type not in PROBLEM_TYPES:
        raise ValueError(f"PROBLEM_TYPE must be one of {PROBLEM_TYPES}.")
    if not 0.0 < settings.test_size < 1.0:
        raise ValueError("TEST_SIZE must be between 0 and 1 (exclusive).")
    if settings.max_iter <= 0:
        raise ValueError("MAX_ITER must be greater than 0.")
