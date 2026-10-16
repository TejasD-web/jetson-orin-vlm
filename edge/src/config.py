"""Edge service configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "qwen2.5-vl-3b"
    llama_server_url: str = "http://localhost:8080"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Metrics
    metrics_enabled: bool = True

    class Config:
        env_prefix = "EDGE_"
