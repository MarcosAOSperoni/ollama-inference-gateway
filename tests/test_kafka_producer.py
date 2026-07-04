import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))

import pytest
from unittest.mock import AsyncMock
import kafka_producer


@pytest.mark.asyncio
async def test_start_skips_when_no_bootstrap_servers(monkeypatch):
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    kafka_producer._producer = None
    await kafka_producer.start()
    assert kafka_producer._producer is None


@pytest.mark.asyncio
async def test_emit_is_silent_when_producer_is_none():
    kafka_producer._producer = None
    await kafka_producer.emit(
        model="llama3:70b", task_type="generate",
        latency_ms=100.0, status="success", backend="mac-studio",
    )  # must not raise


@pytest.mark.asyncio
async def test_emit_sends_event_with_correct_fields():
    mock_producer = AsyncMock()
    kafka_producer._producer = mock_producer

    await kafka_producer.emit(
        model="llama3:70b", task_type="generate",
        latency_ms=2100.0, status="success", backend="mac-studio",
    )

    mock_producer.send_and_wait.assert_awaited_once()
    topic, payload = mock_producer.send_and_wait.call_args[0]
    assert topic == "inference-events"
    assert payload["model"] == "llama3:70b"
    assert payload["latency_ms"] == 2100.0
    assert payload["status"] == "success"
    assert payload["backend"] == "mac-studio"
    assert "timestamp" in payload


@pytest.mark.asyncio
async def test_emit_does_not_raise_on_kafka_error():
    mock_producer = AsyncMock()
    mock_producer.send_and_wait.side_effect = Exception("broker unavailable")
    kafka_producer._producer = mock_producer
    await kafka_producer.emit(
        model="llama3:70b", task_type="generate",
        latency_ms=100.0, status="success", backend="mac-studio",
    )  # must not raise


@pytest.mark.asyncio
async def test_stop_clears_producer():
    mock_producer = AsyncMock()
    kafka_producer._producer = mock_producer
    await kafka_producer.stop()
    mock_producer.stop.assert_awaited_once()
    assert kafka_producer._producer is None
