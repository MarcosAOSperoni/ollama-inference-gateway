import pytest
from unittest.mock import AsyncMock, patch

FAKE_GENERATE_RESPONSE = {
    "model": "llama3:70b",
    "response": "hello world",
    "eval_count": 20,
    "eval_duration": 2_000_000_000,
    "prompt_eval_count": 5,
    "total_duration": 2_100_000_000,
}

FAKE_CHAT_RESPONSE = {
    "model": "llama3:70b",
    "message": {"role": "assistant", "content": "hi"},
    "eval_count": 10,
    "eval_duration": 1_000_000_000,
    "prompt_eval_count": 8,
    "total_duration": 1_100_000_000,
}


async def test_health_returns_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_metrics_endpoint_returns_prometheus_text(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "ollama_requests_total" in response.text


async def test_generate_proxies_to_ollama(client):
    with patch("main.ollama_client.generate", new=AsyncMock(return_value=FAKE_GENERATE_RESPONSE)):
        response = await client.post(
            "/api/generate",
            json={"prompt": "say hello", "model": "llama3:70b"},
        )
    assert response.status_code == 200
    assert response.json()["eval_count"] == 20


async def test_generate_routes_classify_to_small_model(client):
    with patch("main.ollama_client.generate", new=AsyncMock(return_value=FAKE_GENERATE_RESPONSE)) as mock:
        await client.post(
            "/api/generate",
            json={"prompt": "classify this email", "task_type": "classify"},
        )
        called_payload = mock.call_args[0][0]
        assert called_payload["model"] == "llama3:8b"


async def test_generate_strips_task_type_before_forwarding(client):
    with patch("main.ollama_client.generate", new=AsyncMock(return_value=FAKE_GENERATE_RESPONSE)) as mock:
        await client.post(
            "/api/generate",
            json={"prompt": "hello", "task_type": "classify"},
        )
        called_payload = mock.call_args[0][0]
        assert "task_type" not in called_payload


async def test_chat_proxies_to_ollama(client):
    with patch("main.ollama_client.chat", new=AsyncMock(return_value=FAKE_CHAT_RESPONSE)):
        response = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "model": "llama3:70b"},
        )
    assert response.status_code == 200
    assert response.json()["eval_count"] == 10


async def test_generate_returns_502_on_ollama_failure(client):
    with patch("main.ollama_client.generate", new=AsyncMock(side_effect=Exception("connection refused"))):
        response = await client.post("/api/generate", json={"prompt": "hello"})
    assert response.status_code == 502


async def test_chat_returns_502_on_ollama_failure(client):
    with patch("main.ollama_client.chat", new=AsyncMock(side_effect=Exception("timeout"))):
        response = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 502
