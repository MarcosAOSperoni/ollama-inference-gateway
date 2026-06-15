import asyncio
import httpx
from config import settings


async def _post_with_retry(url: str, payload: dict) -> dict:
    last_exc: Exception | None = None
    payload = {**payload, "stream": False}
    for attempt in range(settings.max_retries):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt < settings.max_retries - 1:
                await asyncio.sleep(2**attempt)
    raise last_exc


async def generate(payload: dict) -> dict:
    return await _post_with_retry(
        f"{settings.ollama_url}/api/generate", payload
    )


async def chat(payload: dict) -> dict:
    return await _post_with_retry(
        f"{settings.ollama_url}/api/chat", payload
    )


async def embeddings(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/embeddings", json=payload
        )
        response.raise_for_status()
        return response.json()


async def get_running_models() -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{settings.ollama_url}/api/ps")
        response.raise_for_status()
        return response.json().get("models", [])
