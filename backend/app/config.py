from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    court_provider: str = "mock"
    parser_api_key: str | None = None
    parser_api_base_url: str = "https://parser-api.com/parser/ras_arbitr_api"
    parser_api_timeout_seconds: float = 120.0
    parser_api_max_retries: int = 3
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    yandex_api_key: str | None = None
    yandex_folder_id: str | None = None
    yandex_model: str = "yandexgpt-5-lite"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
