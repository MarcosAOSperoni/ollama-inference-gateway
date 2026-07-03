import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))

import config
from config import Backend
from router import select_backend


def _make_backend(name: str, models: list[str], priority: int) -> Backend:
    return Backend(url=f"http://{name}:11434", name=name, models=models, priority=priority)


def setup_function():
    config.backends = [
        _make_backend("mac", ["llama3:70b", "gemma4:12b", "qwen2.5:7b"], priority=1),
    ]


def test_explicit_model_routes_to_correct_backend():
    model, backend = select_backend("llama3:70b", None)
    assert model == "llama3:70b"
    assert backend.name == "mac"


def test_classify_task_type_resolves_to_small_model():
    model, backend = select_backend(None, "classify")
    assert model == "gemma4:12b"


def test_generate_task_type_resolves_to_default_model():
    model, backend = select_backend(None, "generate")
    assert model == "llama3:70b"


def test_tool_task_type_resolves_to_tool_model():
    model, backend = select_backend(None, "tool")
    assert model == "qwen2.5:7b"


def test_unknown_task_type_resolves_to_default_model():
    model, backend = select_backend(None, "unknown")
    assert model == "llama3:70b"


def test_no_args_resolves_to_default_model():
    model, backend = select_backend(None, None)
    assert model == "llama3:70b"


def test_picks_first_available_backend_by_priority():
    mac = _make_backend("mac", ["llama3:70b"], priority=1)
    gaming = _make_backend("gaming", ["llama3:70b"], priority=2)
    config.backends = [mac, gaming]
    _, backend = select_backend("llama3:70b", None)
    assert backend is mac


async def test_falls_back_to_lower_priority_when_primary_busy():
    mac = _make_backend("mac", ["llama3:70b"], priority=1)
    gaming = _make_backend("gaming", ["llama3:70b"], priority=2)
    config.backends = [mac, gaming]
    await mac.lock.acquire()
    try:
        _, backend = select_backend("llama3:70b", None)
        assert backend is gaming
    finally:
        mac.lock.release()


async def test_returns_primary_when_all_busy_to_queue():
    mac = _make_backend("mac", ["llama3:70b"], priority=1)
    gaming = _make_backend("gaming", ["llama3:70b"], priority=2)
    config.backends = [mac, gaming]
    await mac.lock.acquire()
    await gaming.lock.acquire()
    try:
        _, backend = select_backend("llama3:70b", None)
        assert backend is mac
    finally:
        mac.lock.release()
        gaming.lock.release()


def test_falls_back_to_priority_one_when_no_backend_serves_model():
    config.backends = [_make_backend("mac", ["llama3:70b"], priority=1)]
    model, backend = select_backend("unknown-model:latest", None)
    assert model == "unknown-model:latest"
    assert backend.name == "mac"
