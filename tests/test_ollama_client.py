import pytest
import httpx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))

from pytest_httpx import HTTPXMock
from ollama_client import generate, chat, get_running_models


FAKE_GENERATE_RESPONSE = {
    "model": "llama3:70b",
    "response": "hello world",
    "eval_count": 20,
    "eval_duration": 2_000_000_000,
    "prompt_eval_count": 5,
    "total_duration": 2_100_000_000,
    "load_duration": 50_000_000,
}

FAKE_CHAT_RESPONSE = {
    "model": "llama3:70b",
    "message": {"role": "assistant", "content": "hi"},
    "eval_count": 10,
    "eval_duration": 1_000_000_000,
    "prompt_eval_count": 8,
    "total_duration": 1_100_000_000,
    "load_duration": 50_000_000,
}

FAKE_PS_RESPONSE = {
    "models": [
        {
            "name": "llama3:70b",
            "size": 42_387_234_816,
            "size_vram": 42_387_234_816,
            "digest": "abc123",
            "details": {"family": "llama", "parameter_size": "70B"},
            "expires_at": "2024-06-15T10:00:00Z",
        }
    ]
}


async def test_generate_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        json=FAKE_GENERATE_RESPONSE,
    )
    result = await generate({"model": "llama3:70b", "prompt": "say hello"})
    assert result["eval_count"] == 20
    assert result["response"] == "hello world"


async def test_chat_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/chat",
        json=FAKE_CHAT_RESPONSE,
    )
    result = await chat({
        "model": "llama3:70b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert result["eval_count"] == 10


async def test_get_running_models_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:11434/api/ps",
        json=FAKE_PS_RESPONSE,
    )
    models = await get_running_models()
    assert len(models) == 1
    assert models[0]["name"] == "llama3:70b"


async def test_generate_retries_on_connect_error(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11434/api/generate",
        json=FAKE_GENERATE_RESPONSE,
    )
    result = await generate({"model": "llama3:70b", "prompt": "hi"})
    assert result["eval_count"] == 20


async def test_generate_raises_after_max_retries(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    with pytest.raises(httpx.ConnectError):
        await generate({"model": "llama3:70b", "prompt": "hi"})
