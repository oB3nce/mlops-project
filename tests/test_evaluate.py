from __future__ import annotations

import pandas as pd

from mlops_project.evaluate import (
    _calculate_slice_metrics,
    _constant_baseline,
    _load_evaluation_params,
)


def test_slice_parameters_define_stakeholder_relevant_groups():
    params = _load_evaluation_params()

    assert len(params["slices"]) >= 2
    assert {item["name"] for item in params["slices"]} >= {
        "contract",
        "senior_citizen",
    }


def test_majority_class_baseline_uses_training_labels():
    train_labels = pd.Series(["No", "No", "No", "Yes"])
    test_labels = pd.Series(["Yes", "Yes", "No"])

    baseline = _constant_baseline(
        train_labels,
        test_labels,
        "classification",
        "Yes",
    )

    assert baseline["prediction"] == "No"
    assert baseline["metrics"]["accuracy"] == 0.3333


def test_slice_metrics_include_support_and_fairness_gaps():
    raw_test = pd.DataFrame(
        {
            "Contract": ["Monthly", "Monthly", "Annual", "Annual"],
            "SeniorCitizen": [0, 0, 1, 1],
        }
    )
    labels = pd.Series(["Yes", "No", "Yes", "No"])
    predictions = pd.Series(["Yes", "No", "No", "No"]).to_numpy()
    definitions = [
        {
            "name": "contract",
            "column": "Contract",
            "groups": {"monthly": "Monthly", "annual": "Annual"},
        },
        {
            "name": "senior_citizen",
            "column": "SeniorCitizen",
            "groups": {"non_senior": 0, "senior": 1},
        },
    ]

    result = _calculate_slice_metrics(
        raw_test,
        labels,
        predictions,
        definitions,
        "classification",
        "Yes",
        "senior_citizen",
    )

    assert len(result["slices"]) == 4
    assert all(row["n"] == 2 and row["positive_count"] == 1 for row in result["slices"])
    assert result["fairness_gap"] == {
        "slice": "senior_citizen",
        "selection_rate_gap": 0.5,
        "tpr_gap": 1.0,
    }
