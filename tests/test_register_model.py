from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlops_project import register_model


def test_register_model_rejects_empty_run_id_before_registry_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(register_model, "PROJECT_ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "run_id.txt").write_text(" \n", encoding="utf-8")
    client = Mock()
    monkeypatch.setattr(register_model, "MlflowClient", Mock(return_value=client))

    with pytest.raises(ValueError, match="run ID is empty"):
        register_model.register_best_model()

    client.get_run.assert_not_called()


def test_register_model_sets_alias_only_for_matching_nonempty_run(monkeypatch, tmp_path):
    monkeypatch.setattr(register_model, "PROJECT_ROOT", tmp_path)
    client = Mock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED")
    )
    client.search_model_versions.return_value = []
    version = SimpleNamespace(version="3", run_id="run-123")
    monkeypatch.setattr(register_model, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(register_model.mlflow, "register_model", Mock(return_value=version))

    result = register_model.register_best_model(" run-123 ", "telco-model")

    assert result is version
    register_model.mlflow.register_model.assert_called_once_with(
        "runs:/run-123/model",
        "telco-model",
    )
    client.set_registered_model_alias.assert_called_once_with(
        "telco-model",
        "staging",
        "3",
    )


def test_register_model_reuses_version_for_same_run(monkeypatch, tmp_path):
    monkeypatch.setattr(register_model, "PROJECT_ROOT", tmp_path)
    client = Mock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED")
    )
    version = SimpleNamespace(version="1", run_id="run-123")
    client.search_model_versions.return_value = [version]
    monkeypatch.setattr(register_model, "MlflowClient", Mock(return_value=client))
    register_model_call = Mock()
    monkeypatch.setattr(register_model.mlflow, "register_model", register_model_call)

    result = register_model.register_best_model("run-123", "telco-model")

    assert result is version
    register_model_call.assert_not_called()
    client.set_registered_model_alias.assert_called_once_with(
        "telco-model",
        "staging",
        "1",
    )


def test_register_model_does_not_set_alias_without_registered_run_id(monkeypatch, tmp_path):
    monkeypatch.setattr(register_model, "PROJECT_ROOT", tmp_path)
    client = Mock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED")
    )
    version = SimpleNamespace(version="3", run_id="")
    client.search_model_versions.return_value = []
    monkeypatch.setattr(register_model, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(register_model.mlflow, "register_model", Mock(return_value=version))

    with pytest.raises(RuntimeError, match="registered without a run_id"):
        register_model.register_best_model("run-123", "telco-model")

    client.set_registered_model_alias.assert_not_called()
