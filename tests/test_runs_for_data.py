from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import runs_for_data


def test_find_runs_filters_by_normalized_md5_and_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(runs_for_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runs_for_data, "load_dotenv", Mock())
    experiment = SimpleNamespace(experiment_id="42")
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
    client = Mock()
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = [run]
    monkeypatch.setattr(runs_for_data, "MlflowClient", Mock(return_value=client))

    result = runs_for_data.find_runs_for_data("  0F9DE68E012BD3AED5FA7CDC9FC421AF  ")

    assert result == [run]
    client.get_experiment_by_name.assert_called_once_with("telco-churn")
    client.search_runs.assert_called_once_with(
        experiment_ids=["42"],
        filter_string="tags.dvc_md5 = '0f9de68e012bd3aed5fa7cdc9fc421af'",
        order_by=["start_time DESC"],
    )


@pytest.mark.parametrize("invalid_md5", ["", "abc", "g" * 32, "0" * 31])
def test_find_runs_rejects_invalid_md5(monkeypatch, invalid_md5):
    client = Mock()
    monkeypatch.setattr(runs_for_data, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(runs_for_data, "load_dotenv", Mock())

    with pytest.raises(ValueError, match="32 hexadecimal characters"):
        runs_for_data.find_runs_for_data(invalid_md5)

    client.search_runs.assert_not_called()


def test_find_runs_reports_missing_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(runs_for_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runs_for_data, "load_dotenv", Mock())
    client = Mock()
    client.get_experiment_by_name.return_value = None
    monkeypatch.setattr(runs_for_data, "MlflowClient", Mock(return_value=client))

    with pytest.raises(LookupError, match="does not exist"):
        runs_for_data.find_runs_for_data("0" * 32, "missing-experiment")

    client.search_runs.assert_not_called()
