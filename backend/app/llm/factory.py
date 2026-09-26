from ..config import get_settings
from .base import LlmError, QueryPlanner, RelevanceReranker
from .ollama import OllamaQueryPlanner
from .gigachat import GigaChatClient, GigaChatQueryPlanner, GigaChatRelevanceReranker


class UnavailablePlanner(QueryPlanner):
    def __init__(self, message: str):
        self.message = message

    async def plan(self, description):
        raise LlmError(self.message)


def get_query_planner() -> QueryPlanner:
    settings = get_settings()
    if settings.llm_provider.strip().lower() == "ollama":
        return OllamaQueryPlanner(model=settings.ollama_model, base_url=settings.ollama_base_url)
    if settings.llm_provider.strip().lower() == "gigachat":
        if not settings.gigachat_auth_key:
            return UnavailablePlanner("Укажите GIGACHAT_AUTH_KEY в .env и перезапустите сервер")
        return GigaChatQueryPlanner(
            GigaChatClient(settings.gigachat_auth_key, scope=settings.gigachat_scope,
                           ca_bundle=settings.gigachat_ca_bundle),
            model=settings.gigachat_plan_model,
        )
    return UnavailablePlanner("Unsupported LLM_PROVIDER")


def get_relevance_reranker() -> RelevanceReranker | None:
    settings = get_settings()
    if settings.llm_provider.strip().lower() == "gigachat" and settings.gigachat_auth_key:
        return GigaChatRelevanceReranker(
            GigaChatClient(settings.gigachat_auth_key, scope=settings.gigachat_scope,
                           ca_bundle=settings.gigachat_ca_bundle),
            model=settings.gigachat_relevance_model,
        )
    return None
