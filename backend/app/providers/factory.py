from functools import lru_cache

from ..config import get_settings
from .base import CourtProvider, CourtProviderConfigurationError
from .mock import MockCourtProvider
from .parser_api import ParserApiProvider


@lru_cache
def get_court_provider() -> CourtProvider:
    settings = get_settings()
    provider_name = settings.court_provider.strip().lower()

    if provider_name == "mock":
        return MockCourtProvider()

    if provider_name == "parser_api":
        if not settings.parser_api_key:
            raise CourtProviderConfigurationError(
                "PARSER_API_KEY is required when COURT_PROVIDER=parser_api"
            )
        return ParserApiProvider(
            api_key=settings.parser_api_key,
            base_url=settings.parser_api_base_url,
            timeout_seconds=settings.parser_api_timeout_seconds,
            max_retries=settings.parser_api_max_retries,
        )

    raise CourtProviderConfigurationError(
        f"Unsupported COURT_PROVIDER: {settings.court_provider}"
    )
