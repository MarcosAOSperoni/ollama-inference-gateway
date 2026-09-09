import asyncio
import yaml
from dataclasses import dataclass, field

from pydantic_settings import BaseSettings


@dataclass
class Backend:
    url: str
    name: str
    models: list[str]
    priority: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Settings(BaseSettings):
    backends_file: str = "/app/backends.yml"
    gateway_port: int = 8080
    # qwen2.5:32b, not llama3:70b. Two reasons, both measured 2026-09-09
    # against the real prompts this gateway serves:
    #
    # 1. Quality. Replaying Pilot's production workout-planner prompt six
    #    times, llama3:70b answered "Legs" 6/6 while ignoring the recency
    #    list it was given (Pull was the stalest group). qwen2.5:32b answered
    #    "Pull" 5/6. It also kept more of the source facts in briefing
    #    summaries and emitted better-formed JSON (llama3 prefixes chatter and
    #    puts literal newlines inside JSON strings).
    # 2. Residency. The Mac Studio has 64GB and llama3:70b occupies 39GB, so
    #    it cannot coexist with the small model — every task_type switch
    #    evicted and reloaded 39GB. That is why classify calls averaged 35s
    #    wall despite running at 42 tok/s: almost all of it was loading. At
    #    19GB, qwen2.5:32b sits alongside gemma4:12b + qwen2.5:7b + nomic
    #    (~31.6GB total) and nothing ever swaps.
    #
    # It is also ~1.9x faster per token (21.2 vs 11.3 tok/s).
    default_model: str = "qwen2.5:32b"
    small_model: str = "gemma4:12b"
    tool_model: str = "qwen2.5:7b"
    poll_interval_seconds: int = 3
    max_retries: int = 3


settings = Settings()
backends: list[Backend] = []


def load_backends(path: str) -> list[Backend]:
    with open(path) as f:
        data = yaml.safe_load(f)
    result = [
        Backend(
            url=b["url"],
            name=b["name"],
            models=b["models"],
            priority=b["priority"],
        )
        for b in data["backends"]
    ]
    return sorted(result, key=lambda b: b.priority)
