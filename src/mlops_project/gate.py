from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_acceptance(
    metrics: dict[str, Any],
    slices: dict[str, Any],
    baseline: dict[str, Any] | None,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate configured model acceptance criteria without external I/O."""
    model_metrics = metrics.get("model")
    if not isinstance(model_metrics, dict):
        raise ValueError("metrics must contain a 'model' mapping.")
    floors = acceptance.get("absolute_floors")
    regression = acceptance.get("regression")
    slice_rules = acceptance.get("slice_rules")
    criteria_version = acceptance.get("criteria_version")
    if not isinstance(criteria_version, str) or not criteria_version.strip():
        raise ValueError("acceptance.criteria_version must be a non-empty string.")
    if not isinstance(floors, dict) or len(floors) < 2:
        raise ValueError("acceptance.absolute_floors must define at least two metrics.")
    if not isinstance(regression, dict):
        raise ValueError("acceptance.regression must be a mapping.")
    if not isinstance(slice_rules, list) or not slice_rules:
        raise ValueError("acceptance.slice_rules must define at least one rule.")

    checks: list[dict[str, Any]] = []
    for metric_name, floor in floors.items():
        if metric_name not in model_metrics:
            raise ValueError(f"Model metrics do not contain {metric_name!r}.")
        observed = float(model_metrics[metric_name])
        threshold = float(floor)
        checks.append(
            {
                "name": f"absolute_floor.{metric_name}",
                "status": "pass" if observed >= threshold else "fail",
                "observed": observed,
                "minimum": threshold,
            }
        )

    primary_metric = regression.get("primary_metric")
    tolerance = regression.get("tolerance")
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ValueError("acceptance.regression.primary_metric must be a metric name.")
    tolerance_value = float(tolerance)
    if tolerance_value < 0:
        raise ValueError("acceptance.regression.tolerance cannot be negative.")
    if primary_metric not in model_metrics:
        raise ValueError(f"Model metrics do not contain {primary_metric!r}.")

    if baseline is None:
        checks.append(
            {
                "name": f"regression.{primary_metric}",
                "status": "skipped",
                "reason": "No champion baseline file is available.",
            }
        )
    else:
        baseline_metrics = baseline.get("metrics")
        if not isinstance(baseline_metrics, dict) or primary_metric not in baseline_metrics:
            raise ValueError(
                f"Baseline metrics do not contain {primary_metric!r}."
            )
        observed = float(model_metrics[primary_metric])
        baseline_value = float(baseline_metrics[primary_metric])
        minimum = baseline_value - tolerance_value
        checks.append(
            {
                "name": f"regression.{primary_metric}",
                "status": "pass" if observed >= minimum else "fail",
                "observed": observed,
                "baseline": baseline_value,
                "tolerance": tolerance_value,
                "minimum": minimum,
            }
        )

    slice_rows = slices.get("slices")
    if not isinstance(slice_rows, list):
        raise ValueError("slices must contain a 'slices' list.")
    for rule in slice_rules:
        if not isinstance(rule, dict):
            raise ValueError("Each acceptance.slice_rules item must be a mapping.")
        slice_name = rule.get("slice")
        metric_name = rule.get("metric")
        if not isinstance(slice_name, str) or not slice_name:
            raise ValueError("Each slice rule must define a slice name.")
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError(f"Slice rule {slice_name!r} must define a metric.")
        minimum = float(rule["minimum"])
        minimum_support = int(rule["minimum_support"])
        matching_rows = [row for row in slice_rows if row.get("slice") == slice_name]
        if not matching_rows:
            checks.append(
                {
                    "name": f"slice.{slice_name}",
                    "status": "fail",
                    "reason": "No evaluated groups found for this slice.",
                }
            )
            continue
        for row in matching_rows:
            group = row.get("group", "<unknown>")
            support = int(row.get("n", 0))
            value = (row.get("metrics") or {}).get(metric_name)
            if support < minimum_support:
                status = "fail"
                reason = "Slice support is below the configured minimum."
            elif value is None:
                status = "fail"
                reason = f"Slice metric {metric_name!r} is unavailable."
            else:
                status = "pass" if float(value) >= minimum else "fail"
                reason = None
            check = {
                "name": f"slice.{slice_name}.{group}.{metric_name}",
                "status": status,
                "n": support,
                "minimum_support": minimum_support,
                "observed": None if value is None else float(value),
                "minimum": minimum,
            }
            if reason is not None:
                check["reason"] = reason
            checks.append(check)

    statuses = {check["status"] for check in checks}
    verdict = "NO-GO" if "fail" in statuses else (
        "SKIPPED" if "skipped" in statuses else "GO"
    )
    return {
        "criteria_version": criteria_version,
        "verdict": verdict,
        "checks": checks,
    }


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required gate input does not exist: {path}")
        return None
    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def require_go_verdict(
    project_root: Path = PROJECT_ROOT,
    expected_run_id: str | None = None,
) -> None:
    verdict_path = project_root / "reports" / "gate.json"
    if not verdict_path.is_file():
        raise FileNotFoundError(
            f"Cannot publish model: acceptance verdict does not exist: {verdict_path}"
        )
    with verdict_path.open(encoding="utf-8") as verdict_file:
        report = json.load(verdict_file)
    if not isinstance(report, dict):
        raise ValueError(f"Acceptance verdict must be a JSON object: {verdict_path}")
    verdict = report.get("verdict")
    if verdict != "GO":
        raise RuntimeError(
            f"Cannot publish model: gate verdict is {verdict!r}; expected 'GO'."
        )
    if expected_run_id is not None and report.get("run_id") != expected_run_id:
        raise RuntimeError(
            "Cannot publish model: gate verdict run ID "
            f"{report.get('run_id')!r} does not match candidate run ID "
            f"{expected_run_id!r}."
        )


def run_gate() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file) or {}
    acceptance = params.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("params.yaml must define an acceptance mapping.")

    metrics = _read_json(PROJECT_ROOT / "metrics" / "metrics.json")
    slices = _read_json(PROJECT_ROOT / "metrics" / "slices.json")
    baseline = _read_json(
        PROJECT_ROOT / "metrics" / "baseline.json",
        required=False,
    )
    if metrics is None or slices is None:
        raise RuntimeError("Required gate metrics unexpectedly failed to load.")
    report = evaluate_acceptance(metrics, slices, baseline, acceptance)
    run_id_path = PROJECT_ROOT / "models" / "run_id.txt"
    if not run_id_path.is_file():
        raise FileNotFoundError(f"Candidate MLflow run ID does not exist: {run_id_path}")
    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"Candidate MLflow run ID is empty: {run_id_path}")
    report["run_id"] = run_id
    _write_report(PROJECT_ROOT / "reports" / "gate.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] == "NO-GO":
        raise SystemExit(1)
    return report


def main() -> None:
    run_gate()


if __name__ == "__main__":
    main()
