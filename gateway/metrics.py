import asyncio
from prometheus_client import Counter, Histogram, Gauge
import ollama_client
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

TOKENS_PER_SECOND = Gauge(
    "ollama_tokens_per_second",
    "Tokens generated per second for the most recent request",
    ["model"],
)

REQUESTS_TOTAL = Counter(
    "ollama_requests_total",
    "Total inference requests",
    ["model", "endpoint", "status"],
)

MODELS_LOADED = Gauge(
    "ollama_models_loaded_count",
    "Number of models currently loaded in Ollama",
)

MODEL_SIZE_BYTES = Gauge(
    "ollama_model_size_bytes",
    "Memory consumed by a loaded model in bytes",
    ["model"],
)

MODEL_LOADED = Gauge(
    "ollama_model_loaded",
    "Whether a model is currently loaded (1=yes 0=no)",
    ["model"],
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

    tps = calculate_tps(eval_count, eval_duration_ns)

    REQUEST_DURATION.labels(model=model, endpoint=endpoint).observe(wall_duration)
    TOKENS_GENERATED.labels(model=model).inc(eval_count)
    PROMPT_TOKENS.labels(model=model).inc(prompt_eval_count)
    TOKENS_PER_SECOND.labels(model=model).set(tps)
    REQUESTS_TOTAL.labels(model=model, endpoint=endpoint, status="success").inc()


def _clean_model_name(name: str) -> str:
    return name.removesuffix(":latest")


async def poll_models() -> None:
    known_models: set[str] = set()
    while True:
        try:
            models = await ollama_client.get_running_models()
            current = {_clean_model_name(m.get("name", "unknown")) for m in models}
            for name in known_models - current:
                MODEL_SIZE_BYTES.labels(model=name).set(0)
                MODEL_LOADED.labels(model=name).set(0)
            MODELS_LOADED.set(len(models))
            for m in models:
                name = _clean_model_name(m.get("name", "unknown"))
                MODEL_SIZE_BYTES.labels(model=name).set(m.get("size", 0))
                MODEL_LOADED.labels(model=name).set(1)
            known_models = current
        except Exception:
            pass
        await asyncio.sleep(settings.poll_interval_seconds)
