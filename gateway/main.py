import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import ollama_client
from metrics import REQUESTS_TOTAL, poll_models, record_inference
from router import select_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(poll_models())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


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
    model = select_model(body.get("model"), task_type)
    body["model"] = model

    start = time.time()
    try:
        response = await ollama_client.generate(body)
    except Exception as exc:
        REQUESTS_TOTAL.labels(model=model, endpoint="generate", status="error").inc()
        raise HTTPException(status_code=502, detail=str(exc))

    record_inference(model, "generate", response, time.time() - start)
    return response


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    task_type = body.pop("task_type", None)
    model = select_model(body.get("model"), task_type)
    body["model"] = model

    start = time.time()
    try:
        response = await ollama_client.chat(body)
    except Exception as exc:
        REQUESTS_TOTAL.labels(model=model, endpoint="chat", status="error").inc()
        raise HTTPException(status_code=502, detail=str(exc))

    record_inference(model, "chat", response, time.time() - start)
    return response
