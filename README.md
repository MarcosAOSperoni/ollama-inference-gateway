# ollama-inference-gateway

A FastAPI proxy gateway in front of [Ollama](https://ollama.com) that captures per-request inference metrics and exposes them to Prometheus. Pairs with a Grafana dashboard for real-time LLM observability.

## Architecture

```
your app
    ↓
gateway :8080         ← routes by task_type, captures metrics
    ↓
Ollama :11434         ← LLM inference (one or more backends, see backends.yml)

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

| `task_type` | Setting | Current model |
|---|---|---|
| `classify` | `small_model` | `gemma4:12b` |
| `generate` | `default_model` | `qwen2.5:32b` |
| `summarize` | `default_model` | `qwen2.5:32b` |
| `tool` | `tool_model` | `qwen2.5:7b` |
| _(unset)_ | `default_model` | `qwen2.5:32b` |

A model is only reachable if some backend in `backends.yml` lists it — the
router resolves a name and then looks for a backend serving it, so a model
missing from that file fails at dispatch rather than at startup.

### Setting `model` bypasses routing entirely

A request that sets `model` explicitly wins over `task_type`, which is
occasionally useful and much more often a trap. A downstream app pinned
`model` on twelve call sites, which made `default_model` here **inert** for
all of them: changing it, rebuilding, and confirming the gateway routed
`generate → qwen2.5:32b` changed nothing downstream, because every request
still named the old model. Prefer `task_type` and omit `model`.

### Why these models

`qwen2.5:32b` replaced `llama3:70b` as the default on 2026-09-09. Replaying a
real production prompt six times per model, llama3:70b gave the wrong answer
6/6 while ignoring structured context it was handed; qwen2.5:32b was correct
5/6. It is also ~1.9x faster per token (21.2 vs 11.3 tok/s). The deciding
factor was residency: on a 64GB host llama3:70b's 39GB could not coexist with
the small model, so every `task_type` switch evicted and reloaded 39GB —
classify calls averaged 35s wall despite `gemma4:12b` running at 42 tok/s,
almost all of it loading. At 19GB the whole set fits (~31.6GB) and nothing
swaps. See the comment on `default_model` in `gateway/config.py`.

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
  -d '{"prompt": "explain gradient descent in two sentences", "task_type": "generate"}'
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
| `BACKENDS_FILE` | `/app/backends.yml` | Inference backends and the models each serves |
| `DEFAULT_MODEL` | `qwen2.5:32b` | `generate`, `summarize`, and unrouted requests |
| `SMALL_MODEL` | `gemma4:12b` | `classify` requests |
| `TOOL_MODEL` | `qwen2.5:7b` | `tool` requests (native tool-calling) |
| `GATEWAY_PORT` | `8080` | Gateway listen port |
| `MAX_RETRIES` | `3` | Retry attempts on connection failure |
| `POLL_INTERVAL_SECONDS` | `3` | Ollama model poll interval |

Backend URLs are **not** an env var — they live in `backends.yml`, which also
declares which models each backend serves and their failover `priority`. Copy
`infra/backends.yml.example` to `infra/backends.yml` and edit it; the real file
is gitignored. Any model named by `DEFAULT_MODEL`/`SMALL_MODEL`/`TOOL_MODEL`
must appear in it, and a test enforces that.

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
