from datetime import date
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.llm.base import QueryPlanner
from backend.app.llm.ollama import OllamaQueryPlanner
from backend.app.llm.yandex import YandexQueryPlanner
from backend.app.models import (
    CourtDocument,
    DocumentSearchParams,
    DocumentSearchResult,
    SearchPlan,
    SemanticSearchRequest,
)
from backend.app.providers.base import CourtProvider
from backend.app.services.semantic_search import SemanticSearchService


class FakePlanner(QueryPlanner):
    async def plan(self, description: str) -> SearchPlan:
        return SearchPlan(
            queries=[
                "присоединение заем налоговая выгода",
                "убытки после присоединения проценты по займу",
            ],
            must_have=["заем", "присоединение", "налоговая выгода"],
        )


class FakeProvider(CourtProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.document = CourtDocument(
            document_id="doc-1",
            case_id="case-1",
            case_number="А40-1/2023",
            registration_date=date(2023, 1, 1),
            document_type="Решение",
            file_url="https://example.org/doc-1.pdf",
        )

    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        self.calls.append((params.text or "", params.page))
        return DocumentSearchResult(count=1, pages=1, page=params.page, items=[self.document])

    async def download_pdf(self, file_url: str) -> bytes | None:
        return None


@pytest.mark.asyncio
async def test_ollama_planner_uses_local_json_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:11434/api/chat"
        body = json.loads(request.content)
        assert body["format"]["properties"]["queries"]
        assert body["stream"] is False
        return httpx.Response(200, json={"message": {"content": (
            '{"queries":["присоединение заем","убытки после присоединения"],'
            '"must_have":["заем"],"exclude":[]}'
        )}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plan = await OllamaQueryPlanner(client=client).plan(
            "Компания присоединила заемщика и учитывала проценты по займу"
        )
    assert len(plan.queries) == 2
    assert plan.must_have == ["заем"]


@pytest.mark.asyncio
async def test_yandex_planner_uses_api_key_header_and_folder_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Api-Key test-key"
        assert "test-key" not in str(request.url)
        body = json.loads(request.content)
        assert body["modelUri"] == "gpt://folder-1/yandexgpt-lite"
        assert body["jsonSchema"]["schema"]["properties"]["queries"]
        return httpx.Response(200, json={"result": {"alternatives": [{"message": {"text": (
            '{"queries":["присоединение заем","убытки после присоединения"],'
            '"must_have":["заем"],"exclude":[]}'
        )}}]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plan = await YandexQueryPlanner("test-key", "folder-1", client=client).plan(
            "Компания присоединила заемщика и учитывала проценты по займу"
        )
    assert len(plan.queries) == 2


@pytest.mark.asyncio
async def test_search_runs_each_query_and_deduplicates_with_source_ids() -> None:
    provider = FakeProvider()
    result = await SemanticSearchService(provider, FakePlanner()).search(
        SemanticSearchRequest(description="Компания присоединила заемщика и учла проценты по займу")
    )
    assert len(provider.calls) == 2
    assert result.document_count == 1
    assert result.case_count == 1
    assert result.items[0].matched_queries == result.plan.queries
    assert result.items[0].case.preferred_document.document_id == "doc-1"
    assert result.items[0].case.preferred_document.file_url == "https://example.org/doc-1.pdf"


def test_semantic_search_endpoint() -> None:
    from backend.app.llm.factory import get_query_planner
    from backend.app.main import app
    from backend.app.providers.factory import get_court_provider

    app.dependency_overrides[get_court_provider] = FakeProvider
    app.dependency_overrides[get_query_planner] = FakePlanner
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/cases/semantic-search",
                json={"description": "Компания присоединила заемщика и учла проценты по займу"},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["items"][0]["case"]["case_id"] == "case-1"
