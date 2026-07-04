import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)
TOPIC = "inference-events"

_producer: Optional[AIOKafkaProducer] = None


async def start() -> None:
    global _producer
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    if not bootstrap:
        logger.info("KAFKA_BOOTSTRAP_SERVERS not set — Kafka producer disabled")
        return
    try:
        _producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await _producer.start()
        logger.info("Kafka producer connected to %s", bootstrap)
    except Exception as exc:
        logger.warning("Kafka producer failed to start: %s", exc)
        _producer = None


async def stop() -> None:
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None


async def emit(
    *,
    model: str,
    task_type: Optional[str],
    latency_ms: float,
    status: str,
    backend: str,
) -> None:
    if _producer is None:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "task_type": task_type or "unknown",
        "latency_ms": round(latency_ms, 1),
        "status": status,
        "backend": backend,
    }
    try:
        await _producer.send_and_wait(TOPIC, event)
    except Exception as exc:
        logger.warning("Failed to emit inference event: %s", exc)
