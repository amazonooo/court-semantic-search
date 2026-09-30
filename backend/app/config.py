from functools import lru_cache
from pathlib import Path

from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    court_provider: str = "mock"
    parser_api_key: str | None = None
    parser_api_base_url: str = "https://parser-api.com/parser/ras_arbitr_api"
    parser_api_timeout_seconds: float = Field(default=25.0, gt=0, le=120)
    parser_api_max_retries: int = Field(default=1, ge=1, le=3)
    search_timeout_seconds: float = Field(default=120.0, gt=0, le=150)
    retrieval_timeout_seconds: float = Field(default=45.0, gt=0, le=90)
    plan_timeout_seconds: float = Field(default=45.0, gt=0, le=60)
    relevance_timeout_seconds: float = Field(default=35.0, gt=0, le=45)
    search_max_queries: int = Field(default=10, ge=2, le=10)
    search_max_pages_per_query: int = Field(default=2, ge=1, le=5)
    search_max_search_calls: int = Field(default=10, ge=1, le=30)
    search_max_cases: int = Field(default=6, ge=1, le=10)
    search_max_pdf_downloads: int = Field(default=6, ge=1, le=10)
    llm_provider: str = "gigachat"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    gigachat_auth_key: str | None = None
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_plan_model: str = "GigaChat-2"
    gigachat_relevance_model: str = "GigaChat-2-Pro"
    gigachat_ca_bundle: str | None = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
