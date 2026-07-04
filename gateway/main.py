import asyncio
import time
from contextlib import asynccontextmanager

import config
import kafka_producer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import ollama_client
from config import settings
from metrics import (
    EMBEDDING_DURATION,
    EMBEDDING_REQUESTS_TOTAL,
    REQUESTS_TOTAL,
    poll_models,
    record_inference,
)
from router import select_backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.backends = config.load_backends(settings.backends_file)
    task = asyncio.create_task(poll_models())
    await kafka_producer.start()
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await kafka_producer.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    task_type = body.pop("task_type", None)
    model, backend = select_backend(body.get("model"), task_type)
    body["model"] = model

    start = time.time()
    async with backend.lock:
        try:
            response = await ollama_client.generate(body, base_url=backend.url)
        except Exception as exc:
            REQUESTS_TOTAL.labels(model=model, endpoint="generate", status="error").inc()
            asyncio.create_task(kafka_producer.emit(
                model=model, task_type=task_type,
                latency_ms=(time.time() - start) * 1000,
                status="error", backend=backend.name,
            ))
            raise HTTPException(status_code=502, detail=str(exc))

    elapsed = time.time() - start
    record_inference(model, "generate", response, elapsed)
    asyncio.create_task(kafka_producer.emit(
        model=model, task_type=task_type,
        latency_ms=elapsed * 1000,
        status="success", backend=backend.name,
    ))
    return response


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    task_type = body.pop("task_type", None)
    model, backend = select_backend(body.get("model"), task_type)
    body["model"] = model

    start = time.time()
    async with backend.lock:
        try:
            response = await ollama_client.chat(body, base_url=backend.url)
        except Exception as exc:
            REQUESTS_TOTAL.labels(model=model, endpoint="chat", status="error").inc()
            asyncio.create_task(kafka_producer.emit(
                model=model, task_type=task_type,
                latency_ms=(time.time() - start) * 1000,
                status="error", backend=backend.name,
            ))
            raise HTTPException(status_code=502, detail=str(exc))

    elapsed = time.time() - start
    record_inference(model, "chat", response, elapsed)
    asyncio.create_task(kafka_producer.emit(
        model=model, task_type=task_type,
        latency_ms=elapsed * 1000,
        status="success", backend=backend.name,
    ))
    return response


@app.post("/api/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model = body.get("model", "nomic-embed-text")
    if not config.backends:
        raise HTTPException(status_code=503, detail="No backends configured")
    backend = config.backends[0]

    start = time.time()
    async with backend.lock:
        try:
            response = await ollama_client.embeddings(body, base_url=backend.url)
        except Exception as exc:
            EMBEDDING_REQUESTS_TOTAL.labels(model=model, status="error").inc()
            raise HTTPException(status_code=502, detail=str(exc))

    EMBEDDING_DURATION.labels(model=model).observe(time.time() - start)
    EMBEDDING_REQUESTS_TOTAL.labels(model=model, status="success").inc()
    return response
