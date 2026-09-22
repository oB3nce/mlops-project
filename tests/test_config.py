import pytest

from mlops_project.config import Settings, _validate


def test_defaults():
    s = Settings()
    assert s.problem_type == "classification"
    assert 0.0 < s.test_size < 1.0


def test_invalid_problem_type_rejected():
    with pytest.raises(ValueError):
        _validate(Settings(problem_type="clustering"))


def test_invalid_test_size_rejected():
    with pytest.raises(ValueError):
        _validate(Settings(test_size=1.5))
