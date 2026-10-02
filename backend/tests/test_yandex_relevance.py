import json

import httpx
import pytest

from backend.app.llm.base import LlmError, RelevanceReranker
from backend.app.llm.yandex_relevance import YandexRelevanceReranker
from backend.app.services.evidence import find_evidence
from backend.app.services.relevance import (
    RelevanceCandidate, model_passages,
)
from backend.tests.test_relevance_ranking import (
    SUPPLY_DESCRIPTION, SUPPLY_FALSE, SUPPLY_MATCH, SUPPLY_TERMS,
)


@pytest.mark.asyncio
async def test_yandex_reranker_uses_bounded_grounded_excerpts_and_validates_ids():
    candidates = [
        RelevanceCandidate(key="0", case_number="А40-1/2025", passages={"P1": SUPPLY_MATCH}),
        RelevanceCandidate(key="1", case_number="А40-2/2025", passages={"P1": SUPPLY_FALSE}),
    ]

    def handler(request: httpx.Request):
        assert request.headers["Authorization"] == "Api-Key test-key"
        assert "test-key" not in str(request.url)
        body = json.loads(request.content)
        assert body["modelUri"] == "gpt://folder-1/yandexgpt-5-lite"
        assert body["jsonSchema"]["schema"]["properties"]["items"]
        data = json.loads(body["messages"][1]["text"])
        assert data["situation"] == SUPPLY_DESCRIPTION
        assert len(data["cases"]) == 2
        result = {"items": [
            {"key": "0", "score": 5, "reason": "Иск о неоплате переданного товара", "passage_id": "P1"},
            {"key": "1", "score": 1, "reason": "Другой предмет спора", "passage_id": "P1"},
            {"key": "unknown", "score": 5, "reason": "Выдумано", "passage_id": "P1"},
        ]}
        return httpx.Response(200, json={"result": {"alternatives": [{"message": {
            "text": json.dumps(result, ensure_ascii=False),
        }}]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        judgments = await YandexRelevanceReranker(
            "test-key", "folder-1", client=client,
        ).judge(SUPPLY_DESCRIPTION, candidates)
    assert [(row.key, row.score, row.passage_id) for row in judgments] == [
        ("0", 5, "P1"), ("1", 1, "P1"),
    ]


def test_only_short_pdf_passages_are_selected_for_model():
    text = ("Материалы дела о поставке товара. " + SUPPLY_MATCH) * 1000
    excerpts = model_passages(text, SUPPLY_DESCRIPTION, SUPPLY_TERMS,
                              find_evidence(text, SUPPLY_TERMS))
    assert 1 <= len(excerpts) <= 5
    assert all(len(part) <= 1000 for part in excerpts.values())
    assert sum(map(len, excerpts.values())) < len(text) // 10


@pytest.mark.asyncio
async def test_model_error_is_explicit_not_silent():
    class Broken(RelevanceReranker):
        async def judge(self, description, candidates):
            raise LlmError("temporary")

    from types import SimpleNamespace
    from backend.app.config import Settings
    from backend.app.models import EvidenceSearchRequest, SearchPlan
    from backend.app.services.semantic_search import SemanticSearchService
    from backend.tests.test_semantic_search import FakeProvider

    async def extract(case, provider, *, role):
        return SimpleNamespace(document=case.documents[0], text=SUPPLY_MATCH)

    from pytest import MonkeyPatch
    with MonkeyPatch.context() as patch:
        patch.setattr("backend.app.services.semantic_search.extract_case_document_text", extract)
        result = await SemanticSearchService(FakeProvider(), None, Settings(_env_file=None),
            reranker=Broken()).search_with_evidence(EvidenceSearchRequest(
                description=SUPPLY_DESCRIPTION,
                semantic_reranking=True,
                plan=SearchPlan(queries=["поставка товара", "неоплата товара"],
                                must_have=SUPPLY_TERMS),
            ))
    assert result.items[0].relevance_status == "textual"
    assert result.items[0].relevance_score >= 3
    assert any("смысловая проверка" in warning.lower() for warning in result.warnings)


@pytest.mark.asyncio
async def test_pdf_excerpts_are_not_sent_without_request_consent():
    class Spy(RelevanceReranker):
        called = False
        async def judge(self, description, candidates):
            self.called = True
            return []

    from types import SimpleNamespace
    from backend.app.config import Settings
    from backend.app.models import EvidenceSearchRequest, SearchPlan
    from backend.app.services.semantic_search import SemanticSearchService
    from backend.tests.test_semantic_search import FakeProvider

    async def extract(case, provider, *, role):
        return SimpleNamespace(document=case.documents[0], text=SUPPLY_MATCH)

    spy = Spy()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("backend.app.services.semantic_search.extract_case_document_text", extract)
        result = await SemanticSearchService(FakeProvider(), None, Settings(_env_file=None),
            reranker=spy).search_with_evidence(EvidenceSearchRequest(
                description=SUPPLY_DESCRIPTION,
                plan=SearchPlan(queries=["поставка товара", "неоплата товара"],
                                must_have=SUPPLY_TERMS),
            ))
    assert spy.called is False
    assert result.items[0].relevance_status == "textual"


def test_api_route_requires_explicit_consent_for_semantic_ranking(monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.llm.factory import get_query_planner, get_relevance_reranker
    from backend.app.models import SearchPlan
    from backend.app.providers.factory import get_court_provider
    from backend.app.services.relevance import RelevanceJudgment
    from backend.tests.test_semantic_search import FakePlanner, FakeProvider

    class Judge(RelevanceReranker):
        calls = 0
        async def judge(self, description, candidates):
            self.calls += 1
            return [RelevanceJudgment(item.key, 5, "Совпадают предмет и факты", "P1")
                    for item in candidates]

    async def extract(case, provider, *, role):
        return SimpleNamespace(document=case.documents[0], text=SUPPLY_MATCH)

    judge = Judge()
    monkeypatch.setattr("backend.app.services.semantic_search.extract_case_document_text", extract)
    app.dependency_overrides[get_court_provider] = FakeProvider
    app.dependency_overrides[get_query_planner] = FakePlanner
    app.dependency_overrides[get_relevance_reranker] = lambda: judge
    payload = {"description": SUPPLY_DESCRIPTION,
               "plan": SearchPlan(queries=["поставка товара", "неоплата товара"],
                                  must_have=SUPPLY_TERMS).model_dump(mode="json")}
    payload['plan_approval'] = {key: payload['plan'][key] for key in ('queries', 'must_have')}
    try:
        with TestClient(app) as client:
            local = client.post("/api/cases/search-with-evidence", json=payload)
            semantic = client.post("/api/cases/search-with-evidence",
                                   json={**payload, "semantic_reranking": True})
    finally:
        app.dependency_overrides.clear()
    assert local.status_code == semantic.status_code == 200
    assert local.json()["items"][0]["relevance_status"] == "textual"
    assert semantic.json()["items"][0]["relevance_status"] == "semantic"
    assert judge.calls == 1
