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
    # Serve whatever the settings actually route to, so these stay true when
    # a model is swapped out. Naming models literally here made changing
    # default_model look like a routing regression.
    config.backends = [
        _make_backend("mac", [
            config.settings.default_model,
            config.settings.small_model,
            config.settings.tool_model,
        ], priority=1),
    ]


def test_explicit_model_routes_to_correct_backend():
    model, backend = select_backend("llama3:70b", None)
    assert model == "llama3:70b"
    assert backend.name == "mac"


def test_classify_task_type_resolves_to_small_model():
    model, backend = select_backend(None, "classify")
    assert model == config.settings.small_model


def test_generate_task_type_resolves_to_default_model():
    model, backend = select_backend(None, "generate")
    assert model == config.settings.default_model


def test_tool_task_type_resolves_to_tool_model():
    model, backend = select_backend(None, "tool")
    assert model == config.settings.tool_model


def test_unknown_task_type_resolves_to_default_model():
    model, backend = select_backend(None, "unknown")
    assert model == config.settings.default_model


def test_no_args_resolves_to_default_model():
    model, backend = select_backend(None, None)
    assert model == config.settings.default_model


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


def test_every_routed_model_is_served_by_a_backend():
    """The task_type map and backends.yml must not drift apart.

    _resolve_model returns a model name and select_backend then looks for a
    backend that declares it. If a routed model is missing from backends.yml
    the request fails at dispatch rather than at startup, so the mistake shows
    up as a runtime error on one task type only — exactly the trap when
    default_model was moved to qwen2.5:32b and the model had to be added to
    the mac-studio list in the same change.
    """
    import os
    import yaml
    from config import settings

    here = os.path.dirname(__file__)
    # backends.yml is gitignored environment config; a fresh clone and CI have
    # only the example. Check the real file when it exists, the example
    # otherwise, so this test runs everywhere.
    path = os.path.join(here, "../infra/backends.yml")
    if not os.path.exists(path):
        path = os.path.join(here, "../infra/backends.yml.example")
    with open(path) as f:
        data = yaml.safe_load(f)

    served = {m for b in data["backends"] for m in b["models"]}
    for name, model in (
        ("default_model", settings.default_model),
        ("small_model", settings.small_model),
        ("tool_model", settings.tool_model),
    ):
        assert model in served, f"{name}={model!r} is not served by any backend"
