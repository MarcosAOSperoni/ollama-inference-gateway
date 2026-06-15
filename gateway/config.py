from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    gateway_port: int = 8080
    default_model: str = "llama3:70b"
    small_model: str = "llama3:8b"
    poll_interval_seconds: int = 15
    max_retries: int = 3


settings = Settings()
