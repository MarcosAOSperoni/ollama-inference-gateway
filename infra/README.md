# infra

Docker Compose stack for the Ollama inference gateway.

## Services

| Service | Port | Description |
|---|---|---|
| gateway | 8080 | FastAPI proxy in front of Ollama backends |
| kafka | 9092 | KRaft-mode Kafka broker for inference event streaming |
| pushgateway | 9091 | Prometheus Pushgateway for Flink job metrics |
| flink-job | — | Streaming aggregation job (Kafka → Pushgateway) |

## Observability

Prometheus and Grafana are **not** part of this stack. Scraping and dashboards
are managed centrally in
[homelab-telemetry](https://github.com/MarcosAOSperoni/homelab-telemetry).

The homelab-telemetry Prometheus instance scrapes:
- `192.168.0.30:8080` → `job="ollama-gateway"`
- `192.168.0.30:9091` → `job="pushgateway"`

## Usage

```bash
cp backends.yml.example backends.yml   # edit to add your Ollama backends
make start
make stop
```
