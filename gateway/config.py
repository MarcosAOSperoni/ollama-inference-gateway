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
    default_model: str = "llama3:70b"
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
