from __future__ import annotations

import json

from .config import load_settings
from .data import build_dataset
from .model import build_model, evaluate_model


def main() -> None:
    settings = load_settings()
    x_train, x_test, y_train, y_test = build_dataset(settings)

    model = build_model(settings)
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test, settings)

    print(f"Project baseline — {settings.problem_type}")
    print(f"Dataset:   {settings.data_path.name}")
    print(f"Target:    {settings.target_column}")
    print(f"Train/Test: {len(x_train)} / {len(x_test)} rows (seed {settings.random_seed})")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
