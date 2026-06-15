import sys
import os
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))


@pytest.fixture(autouse=True)
def mock_background_poller():
    async def _noop():
        await asyncio.sleep(0)

    with patch("metrics.poll_models", new=_noop):
        yield


@pytest.fixture
async def client():
    from main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
