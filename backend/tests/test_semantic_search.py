from datetime import date
import json
from types import SimpleNamespace

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
    EvidenceSearchRequest,
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
async def test_search_keeps_nonmatching_candidate_explicitly_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(case: CourtDocument, provider: CourtProvider, *, role: str) -> SimpleNamespace:
        return SimpleNamespace(document=FakeProvider().document, text="Дело о банкротстве гражданина и реализации имущества")

    monkeypatch.setattr(
        "backend.app.services.semantic_search.extract_case_document_text",
        fake_extract,
    )

    result = await SemanticSearchService(FakeProvider(), FakePlanner()).search_with_evidence(
        EvidenceSearchRequest(
            description="Компания присоединила заемщика и учла проценты по займу",
        )
    )

    assert result.case_count == 1
    assert result.cases_checked == 1
    assert len(result.items) == 1
    assert result.items[0].verification_status != "terms_found"
    assert result.items[0].missing_terms


@pytest.mark.asyncio
async def test_search_reports_missing_terms_without_arbitrary_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(case: CourtDocument, provider: CourtProvider, *, role: str) -> SimpleNamespace:
        return SimpleNamespace(document=FakeProvider().document, text="В документе упомянут только заем, без реорганизации и налоговой выгоды")

    monkeypatch.setattr(
        "backend.app.services.semantic_search.extract_case_document_text",
        fake_extract,
    )

    result = await SemanticSearchService(FakeProvider(), FakePlanner()).search_with_evidence(
        EvidenceSearchRequest(
            description="Компания присоединила заемщика и учла проценты по займу",
        )
    )

    assert result.cases_checked == 1
    assert len(result.items) == 1
    assert result.items[0].verification_status != "terms_found"
    assert result.items[0].missing_terms


@pytest.mark.asyncio
async def test_search_with_evidence_keeps_three_of_four_required_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FourTermPlanner(FakePlanner):
        async def plan(self, description: str) -> SearchPlan:
            return SearchPlan(
                queries=["присоединение компании", "налоговая выгода"],
                must_have=[
                    "присоединение компании",
                    "проценты по займу",
                    "налог на прибыль",
                    "налоговая выгода",
                ],
            )

    async def fake_extract(case: CourtDocument, provider: CourtProvider, *, role: str) -> SimpleNamespace:
        return SimpleNamespace(
            document=FakeProvider().document, text="Присоединение компании, проценты по займу и налог на прибыль"
        )

    monkeypatch.setattr(
        "backend.app.services.semantic_search.extract_case_document_text",
        fake_extract,
    )

    result = await SemanticSearchService(FakeProvider(), FourTermPlanner()).search_with_evidence(
        EvidenceSearchRequest(
            description="Компания присоединила заемщика и учла проценты по займу",
        )
    )

    assert len(result.items) == 1
    assert result.items[0].coverage == 0.75
    assert result.items[0].matched_terms == [
        "присоединение компании", "проценты по займу", "налог на прибыль"]
    assert result.items[0].missing_terms == ["налоговая выгода"]
    assert all(match.highlights for match in result.items[0].evidence)


@pytest.mark.asyncio
async def test_other_topics_never_imply_a_bankruptcy_ban() -> None:
    class BankruptcyMetadataProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.document = self.document.model_copy(
                update={"content_types": ["Банкротство"]}
            )

    provider = BankruptcyMetadataProvider()
    result = await SemanticSearchService(provider, FakePlanner()).search(
        SemanticSearchRequest(
            description="Компания присоединила заемщика и учла проценты по займу",
        )
    )

    assert result.document_count == 1
    assert result.case_count == 1
    assert result.plan.exclude == []


@pytest.mark.asyncio
async def test_explicit_bankruptcy_search_keeps_bankruptcy_documents() -> None:
    class BankruptcyPlanner(FakePlanner):
        async def plan(self, description: str) -> SearchPlan:
            return SearchPlan(
                queries=["дело о банкротстве гражданина", "финансовый управляющий"],
                must_have=["банкротство"],
                exclude=[],
            )

    result = await SemanticSearchService(
        FakeProvider(), BankruptcyPlanner()
    ).search(
        SemanticSearchRequest(description="Найти дела о банкротстве гражданина")
    )

    assert result.document_count == 1
    assert result.case_count == 1
    assert "банкротство" not in result.plan.exclude
    assert "несостоятельность" not in result.plan.exclude


@pytest.mark.asyncio
async def test_search_ranks_cases_by_query_coverage_before_document_date() -> None:
    class RankingPlanner(FakePlanner):
        async def plan(self, description: str) -> SearchPlan:
            return SearchPlan(queries=["первый запрос", "второй запрос"])

    class RankingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.newer = self.document.model_copy(
                update={
                    "document_id": "newer",
                    "case_id": "newer-case",
                    "case_number": "А40-2/2025",
                    "registration_date": date(2025, 1, 1),
                }
            )
            self.older = self.document.model_copy(
                update={
                    "document_id": "older",
                    "case_id": "older-case",
                    "case_number": "А40-1/2024",
                    "registration_date": date(2024, 1, 1),
                }
            )

        async def search_documents(
            self, params: DocumentSearchParams
        ) -> DocumentSearchResult:
            self.calls.append((params.text or "", params.page))
            items = [self.newer, self.older] if params.text == "первый запрос" else [self.older]
            return DocumentSearchResult(
                count=len(items), pages=1, page=params.page, items=items
            )

    result = await SemanticSearchService(RankingProvider(), RankingPlanner()).search(
        SemanticSearchRequest(description="Проверить два поисковых запроса по делам")
    )

    assert [item.case.case_number for item in result.items] == [
        "А40-1/2024",
        "А40-2/2025",
    ]
    assert result.items[0].matched_queries == result.plan.queries


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
        assert request.headers["Content-Type"] == "application/json"
        assert "test-key" not in str(request.url)
        body = json.loads(request.content)
        assert body["modelUri"] == "gpt://folder-1/yandexgpt-5-lite"
        assert body["completionOptions"]["maxTokens"] == "1000"
        assert body["completionOptions"]["reasoningOptions"] == {"mode": "DISABLED"}
        assert body["jsonSchema"]["schema"]["properties"]["queries"]
        assert body["jsonSchema"]["schema"]["required"] == [
            "queries", "must_have", "exclude", "filters"
        ]
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
                json={"description": "Компания присоединила заемщика и учла проценты по займу",
                      "plan": {"queries": ["присоединение заемщика", "проценты по займу"], "must_have": ["заем"]},
                      "plan_approval": {"queries": ["присоединение заемщика", "проценты по займу"], "must_have": ["заем"]}},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["items"][0]["case"]["case_id"] == "case-1"


def test_plan_endpoint_returns_llm_plan_before_provider_search() -> None:
    from backend.app.main import app
    from backend.app.providers.factory import get_court_provider
    from backend.app.llm.factory import get_query_planner

    app.dependency_overrides[get_court_provider] = FakeProvider
    app.dependency_overrides[get_query_planner] = FakePlanner
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/cases/plan",
                json={"description": "Компания присоединила заемщика и учла проценты по займу"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["queries"]
    assert response.json()["exclude"] == []
