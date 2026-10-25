from __future__ import annotations

from mlops_project.gate import evaluate_acceptance


def _acceptance():
    return {
        "criteria_version": "1",
        "absolute_floors": {"accuracy": 0.79, "f1": 0.78},
        "regression": {"primary_metric": "f1", "tolerance": 0.01},
        "slice_rules": [
            {
                "slice": "senior_citizen",
                "metric": "f1",
                "minimum": 0.65,
                "minimum_support": 200,
            }
        ],
    }


def _metrics():
    return {"model": {"accuracy": 0.802, "f1": 0.7939}}


def _slices():
    return {
        "slices": [
            {
                "slice": "senior_citizen",
                "group": "non_senior",
                "n": 1187,
                "metrics": {"f1": 0.8125},
            },
            {
                "slice": "senior_citizen",
                "group": "senior",
                "n": 222,
                "metrics": {"f1": 0.692},
            },
        ]
    }


def test_acceptance_passes_with_baseline_and_supported_slices():
    baseline = {"metrics": {"f1": 0.7939}}

    result = evaluate_acceptance(_metrics(), _slices(), baseline, _acceptance())

    assert result["verdict"] == "GO"
    assert all(check["status"] == "pass" for check in result["checks"])


def test_acceptance_fails_when_floor_or_slice_support_is_not_met():
    metrics = {"model": {"accuracy": 0.75, "f1": 0.70}}
    slices = {
        "slices": [
            {
                "slice": "senior_citizen",
                "group": "senior",
                "n": 50,
                "metrics": {"f1": 0.8},
            }
        ]
    }
    baseline = {"metrics": {"f1": 0.79}}

    result = evaluate_acceptance(metrics, slices, baseline, _acceptance())

    assert result["verdict"] == "NO-GO"
    assert any(
        check["name"] == "slice.senior_citizen.senior.f1"
        and check["status"] == "fail"
        for check in result["checks"]
    )


def test_missing_baseline_skips_regression_without_passing_verdict():
    result = evaluate_acceptance(_metrics(), _slices(), None, _acceptance())

    assert result["verdict"] == "SKIPPED"
    assert any(
        check["name"] == "regression.f1" and check["status"] == "skipped"
        for check in result["checks"]
    )
