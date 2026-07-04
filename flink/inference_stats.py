import json
import os
from typing import Iterable

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
WINDOW_SECONDS = int(os.getenv("WINDOW_SECONDS", "60"))
KAFKA_JAR = "file:///opt/flink-jars/flink-connector-kafka.jar"
KAFKA_CLIENTS_JAR = "file:///opt/flink-jars/kafka-clients.jar"


def _compute_stats(latencies: list[float], error_count: int) -> dict:
    if not latencies:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "error_rate": 0.0}
    sorted_l = sorted(latencies)
    p95_idx = max(0, int(len(sorted_l) * 0.95) - 1)
    return {
        "mean_ms": sum(sorted_l) / len(sorted_l),
        "p95_ms": sorted_l[p95_idx],
        "error_rate": error_count / len(latencies),
    }


def _push_metrics(model: str, count: int, stats: dict) -> None:
    import requests  # noqa: PLC0415 — deferred; only available at runtime inside the container

    instance = model.replace(":", "_").replace("/", "_")
    body = (
        f'kafka_stream_requests_total{{model="{model}"}} {count}\n'
        f'kafka_stream_latency_mean_ms{{model="{model}"}} {stats["mean_ms"]:.1f}\n'
        f'kafka_stream_latency_p95_ms{{model="{model}"}} {stats["p95_ms"]:.1f}\n'
        f'kafka_stream_error_rate{{model="{model}"}} {stats["error_rate"]:.4f}\n'
    )
    try:
        requests.post(
            f"{PUSHGATEWAY_URL}/metrics/job/flink_inference/instance/{instance}",
            data=body,
            headers={"Content-Type": "text/plain"},
            timeout=5,
        )
    except Exception as exc:
        print(f"[warn] pushgateway push failed: {exc}")


def main() -> None:
    # Deferred imports: pyflink is only available inside the container at runtime.
    # Keeping them here allows _compute_stats to be imported and unit-tested without PyFlink.
    from pyflink.common import WatermarkStrategy
    from pyflink.common.serialization import SimpleStringSchema
    from pyflink.common.time import Time
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
    from pyflink.datastream.functions import KeySelector, ProcessWindowFunction, RuntimeContext
    from pyflink.datastream.window import TumblingProcessingTimeWindows

    class _ModelKeySelector(KeySelector):
        def get_key(self, value: str) -> str:
            return json.loads(value)["model"]

    class _InferenceWindowFunction(ProcessWindowFunction):
        def process(self, model: str, context, elements: Iterable[str]):
            parsed = [json.loads(e) for e in elements]
            latencies = [e["latency_ms"] for e in parsed]
            errors = sum(1 for e in parsed if e["status"] == "error")
            stats = _compute_stats(latencies, errors)
            _push_metrics(model, len(parsed), stats)
            yield model

        def open(self, runtime_context: RuntimeContext) -> None:
            pass

        def close(self) -> None:
            pass

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.add_jars(KAFKA_JAR, KAFKA_CLIENTS_JAR)

    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(KAFKA_BOOTSTRAP_SERVERS)
        .set_topics("inference-events")
        .set_group_id("flink-inference-stats")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    (
        env.from_source(source, WatermarkStrategy.no_watermarks(), "kafka-source")
        .key_by(_ModelKeySelector())
        .window(TumblingProcessingTimeWindows.of(Time.seconds(WINDOW_SECONDS)))
        .process(_InferenceWindowFunction())
        .print()
    )

    env.execute("inference-stats")


if __name__ == "__main__":
    main()
