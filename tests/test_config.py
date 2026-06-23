import sys
import os

import yaml
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))


def _write_backends_yml(tmp_path, data: dict) -> str:
    config_file = tmp_path / "backends.yml"
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return str(config_file)


def test_load_backends_sorts_by_priority(tmp_path):
    from config import load_backends

    path = _write_backends_yml(tmp_path, {"backends": [
        {"url": "http://b:11434", "name": "b", "models": ["m2"], "priority": 2},
        {"url": "http://a:11434", "name": "a", "models": ["m1"], "priority": 1},
    ]})
    result = load_backends(path)
    assert result[0].name == "a"
    assert result[1].name == "b"


def test_load_backends_models_list(tmp_path):
    from config import load_backends

    path = _write_backends_yml(tmp_path, {"backends": [
        {"url": "http://a:11434", "name": "a", "models": ["llama3:70b", "gemma4:12b"], "priority": 1},
    ]})
    result = load_backends(path)
    assert "llama3:70b" in result[0].models
    assert "gemma4:12b" in result[0].models


def test_load_backends_creates_independent_locks(tmp_path):
    from config import load_backends

    path = _write_backends_yml(tmp_path, {"backends": [
        {"url": "http://a:11434", "name": "a", "models": ["m1"], "priority": 1},
        {"url": "http://b:11434", "name": "b", "models": ["m2"], "priority": 2},
    ]})
    result = load_backends(path)
    assert result[0].lock is not result[1].lock
    assert not result[0].lock.locked()
    assert not result[1].lock.locked()
