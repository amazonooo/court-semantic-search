from fastapi import HTTPException

from ..config import get_settings
from .base import QueryPlanner
from .ollama import OllamaQueryPlanner
from .yandex import YandexQueryPlanner


def get_query_planner() -> QueryPlanner:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaQueryPlanner(model=settings.ollama_model, base_url=settings.ollama_base_url)
    if settings.llm_provider == "yandex":
        if not settings.yandex_api_key or not settings.yandex_folder_id:
            raise HTTPException(status_code=503, detail="YANDEX_API_KEY and YANDEX_FOLDER_ID are required")
        return YandexQueryPlanner(
            settings.yandex_api_key,
            settings.yandex_folder_id,
            model=settings.yandex_model,
        )
    raise HTTPException(status_code=503, detail="Unsupported LLM_PROVIDER")
