# ollama-inference-gateway

A FastAPI proxy gateway in front of [Ollama](https://ollama.com) that captures per-request inference metrics and exposes them to Prometheus. Pairs with a Grafana dashboard for real-time LLM observability.

## Architecture

```
your app
    ↓
gateway :8080         ← routes by model, captures metrics
    ↓
Ollama :11434         ← LLM inference (llama3:70b, etc.)

Prometheus :9090      ← scrapes gateway /metrics every 15s
    ↑
Grafana :3001         ← dashboards
```

## Metrics

| Metric | Type | Description |
|---|---|---|
| `ollama_tokens_per_second` | Gauge | Tokens generated per second (last request) |
| `ollama_request_duration_seconds` | Histogram | Wall-clock latency per request |
| `ollama_requests_total` | Counter | Total requests by model, endpoint, status |
| `ollama_tokens_generated_total` | Counter | Total output tokens |
| `ollama_prompt_tokens_total` | Counter | Total input tokens |
| `ollama_models_loaded_count` | Gauge | Models currently loaded in Ollama |
| `ollama_model_size_bytes` | Gauge | Memory consumed per loaded model |

## Model Routing

Requests can include a `task_type` field to route to the appropriate model:

| `task_type` | Model |
|---|---|
| `classify` | `small_model` (default: `llama3:8b`) |
| `generate` | `default_model` (default: `llama3:70b`) |
| `summarize` | `default_model` |
| _(unset)_ | `default_model` |

Callers can also set `model` directly to bypass routing.

## Quick Start

### 1. Run the gateway (on your Ollama host)

```bash
cd gateway
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

Verify: `curl http://localhost:8080/health`

### 2. Send a request

```bash
curl http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "explain gradient descent in two sentences", "model": "llama3:70b"}'
```

### 3. Start Prometheus + Grafana

```bash
cd infra
cp .env.example .env
# edit .env and set GATEWAY_HOST to your Ollama host IP
make start
```

Prometheus: `http://your-server:9090`
Grafana: `http://your-server:3001` (login: admin/admin)

Import `infra/grafana/dashboards/ollama.json` into Grafana to get the pre-built dashboard.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `DEFAULT_MODEL` | `llama3:70b` | Model for generation tasks |
| `SMALL_MODEL` | `llama3:8b` | Model for classification tasks |
| `GATEWAY_PORT` | `8080` | Gateway listen port |
| `MAX_RETRIES` | `3` | Retry attempts on connection failure |
| `POLL_INTERVAL_SECONDS` | `15` | Ollama model poll interval |

## Running Tests

```bash
cd gateway
source .venv/bin/activate
cd ..
pytest tests/ -v
```

23 tests, no external dependencies required.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) running locally or on your network
- Docker + Docker Compose (for Prometheus + Grafana)
