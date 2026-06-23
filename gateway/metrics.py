import asyncio
from prometheus_client import Counter, Histogram, Gauge

import ollama_client
import config
from config import settings


REQUEST_DURATION = Histogram(
    "ollama_request_duration_seconds",
    "Inference request wall-clock duration in seconds",
    ["model", "endpoint"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

TOKENS_GENERATED = Counter(
    "ollama_tokens_generated_total",
    "Total tokens generated across all requests",
    ["model"],
)

PROMPT_TOKENS = Counter(
    "ollama_prompt_tokens_total",
    "Total prompt tokens processed across all requests",
    ["model"],
)

REQUESTS_TOTAL = Counter(
    "ollama_requests_total",
    "Total inference requests",
    ["model", "endpoint", "status"],
)

MODELS_LOADED = Gauge(
    "ollama_models_loaded_count",
    "Number of models currently loaded per backend",
    ["backend"],
)

MODEL_SIZE_BYTES = Gauge(
    "ollama_model_size_bytes",
    "Memory consumed by a loaded model in bytes",
    ["model", "backend"],
)

MODEL_LOADED = Gauge(
    "ollama_model_loaded",
    "Whether a model is currently loaded (1=yes 0=no)",
    ["model", "backend"],
)

INFERENCE_TPS = Histogram(
    "ollama_inference_tps",
    "Tokens per second for each inference request (from Ollama eval_duration)",
    ["model"],
    buckets=[5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100],
)

EMBEDDING_DURATION = Histogram(
    "ollama_embedding_duration_seconds",
    "Embedding request wall-clock duration in seconds",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

EMBEDDING_REQUESTS_TOTAL = Counter(
    "ollama_embedding_requests_total",
    "Total embedding requests",
    ["model", "status"],
)


def calculate_tps(eval_count: int, eval_duration_ns: int) -> float:
    if eval_duration_ns <= 0 or eval_count <= 0:
        return 0.0
    return eval_count / (eval_duration_ns / 1e9)


def record_inference(
    model: str, endpoint: str, response: dict, wall_duration: float
) -> None:
    eval_count = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 0)
    prompt_eval_count = response.get("prompt_eval_count", 0)

    REQUEST_DURATION.labels(model=model, endpoint=endpoint).observe(wall_duration)
    TOKENS_GENERATED.labels(model=model).inc(eval_count)
    PROMPT_TOKENS.labels(model=model).inc(prompt_eval_count)
    REQUESTS_TOTAL.labels(model=model, endpoint=endpoint, status="success").inc()
    tps = calculate_tps(eval_count, eval_duration_ns)
    if tps > 0:
        INFERENCE_TPS.labels(model=model).observe(tps)


def _clean_model_name(name: str) -> str:
    return name.removesuffix(":latest")


async def poll_models() -> None:
    known: dict[str, set[str]] = {}
    while True:
        for backend in config.backends:
            try:
                models = await ollama_client.get_running_models(base_url=backend.url)
                current = {_clean_model_name(m.get("name", "unknown")) for m in models}
                prev = known.get(backend.name, set())
                for name in prev - current:
                    MODEL_SIZE_BYTES.labels(model=name, backend=backend.name).set(0)
                    MODEL_LOADED.labels(model=name, backend=backend.name).set(0)
                MODELS_LOADED.labels(backend=backend.name).set(len(models))
                for m in models:
                    name = _clean_model_name(m.get("name", "unknown"))
                    MODEL_SIZE_BYTES.labels(model=name, backend=backend.name).set(m.get("size", 0))
                    MODEL_LOADED.labels(model=name, backend=backend.name).set(1)
                known[backend.name] = current
            except Exception:
                pass
        await asyncio.sleep(settings.poll_interval_seconds)
