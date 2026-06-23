import sys
import os
import asyncio

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))

import config
from config import Backend


def _make_test_backends() -> list[Backend]:
    return [
        Backend(
            url="http://localhost:11434",
            name="test-ollama",
            models=["llama3:70b", "gemma4:12b"],
            priority=1,
        )
    ]


@pytest.fixture(autouse=True)
def mock_background_poller():
    async def _noop():
        await asyncio.sleep(0)

    with patch("metrics.poll_models", new=_noop):
        yield


@pytest.fixture(autouse=True)
def setup_test_backends():
    config.backends = _make_test_backends()
    yield
    config.backends = []


@pytest.fixture
async def client():
    from main import app

    with patch("config.load_backends", return_value=_make_test_backends()):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
