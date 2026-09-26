import json
import time
from uuid import UUID

import httpx
import pytest

from backend.app.config import Settings
from backend.app.llm.base import LlmError
from backend.app.llm.factory import UnavailablePlanner
from backend.app.llm.gigachat import (
    GigaChatClient, GigaChatQueryPlanner, GigaChatRelevanceReranker,
)
from backend.app.services.relevance import RelevanceCandidate


@pytest.mark.asyncio
async def test_gigachat_plan_uses_oauth_and_structured_output_and_reuses_token() -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/oauth"):
            assert request.headers["authorization"] == "Basic test-auth-key"
            assert UUID(request.headers["rquid"]).version == 4
            assert request.content == b"scope=GIGACHAT_API_PERS"
            return httpx.Response(200, json={"access_token": "test-token", "expires_at": time.time() + 1800})
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-token"
        body = json.loads(request.content)
        assert body["model"] == "GigaChat-2"
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["strict"] is True
        assert body["response_format"]["schema"]["required"] == [
            "queries", "must_have", "exclude", "filters"]
        assert body["messages"][1]["content"] == "Взыскать долг по поставке товара"
        content = {"queries": ["долг по поставке", "неоплата товара"],
                   "must_have": ["поставка", "неоплата"], "exclude": [],
                   "filters": {key: None for key in (
                       "case_number", "inn", "court", "date_from", "date_to",
                       "dispute_type", "dispute_category")}}
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(content)}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        api = GigaChatClient("test-auth-key", client=http)
        planner = GigaChatQueryPlanner(api)
        first = await planner.plan("Взыскать долг по поставке товара")
        second = await planner.plan("Взыскать долг по поставке товара")

    assert first.queries == second.queries == ["долг по поставке", "неоплата товара"]
    assert len([r for r in requests if r.url.path.endswith("/oauth")]) == 1
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_gigachat_relevance_keeps_only_sourced_valid_judgments() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth"):
            return httpx.Response(200, json={"access_token": "token", "expires_at": time.time() + 1800})
        body = json.loads(request.content)
        assert body["model"] == "GigaChat-2-Pro"
        data = json.loads(body["messages"][1]["content"])
        assert data["cases"][0]["passages"] == {"P1": "Передача товара и неоплата"}
        assert data["must_have"] == [
            {"id": "C1", "term": "передача товара"},
            {"id": "C2", "term": "неоплата товара"},
        ]
        criteria_schema = body["response_format"]["schema"]["properties"]["items"]["items"]["properties"]["criteria"]
        assert criteria_schema["required"] == ["C1", "C2"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "items": [
                {"key": "0", "score": 5, "reason": "Совпали поставка и неоплата", "passage_id": "P1",
                 "criteria": {
                     "C1": {"status": "supported", "passage_id": "P1", "quote": "Передача товара"},
                     "C2": {"status": "not_shown", "passage_id": "", "quote": ""},
                     "C3": {"status": "supported", "passage_id": "P1", "quote": "Передача товара"},
                 }},
                {"key": "0", "score": 4, "reason": "duplicate", "passage_id": "P1"},
                {"key": "1", "score": 5, "reason": "unsupported", "passage_id": "P9"},
            ]})}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        reranker = GigaChatRelevanceReranker(GigaChatClient("key", client=http))
        rows = await reranker.judge("Поставка товара не оплачена", [
            RelevanceCandidate("0", "А1", {"P1": "Передача товара и неоплата"},
                               ("передача товара", "неоплата товара")),
            RelevanceCandidate("1", "А2", {"P1": "Другой спор"}),
        ])
    assert [(row.key, row.score, row.passage_id) for row in rows] == [("0", 5, "P1")]
    assert [(term.term, term.status, term.passage_id) for term in rows[0].criteria] == [
        ("передача товара", "supported", "P1"),
        ("неоплата товара", "not_shown", None),
    ]
    assert rows[0].criteria[0].quote == "Передача товара"


@pytest.mark.asyncio
async def test_gigachat_auth_error_does_not_expose_secret_or_response() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="secret=very-sensitive")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(LlmError) as error:
            await GigaChatQueryPlanner(GigaChatClient("very-sensitive", client=http)).plan("Описание")
    assert "very-sensitive" not in str(error.value)
    assert "401" in str(error.value)


@pytest.mark.asyncio
async def test_yandex_setting_cannot_make_a_yandex_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.llm import factory

    monkeypatch.setattr(factory, "get_settings", lambda: Settings(
        _env_file=None, llm_provider="yandex", yandex_api_key="old-key",
        yandex_folder_id="old-folder"))
    planner = factory.get_query_planner()
    assert isinstance(planner, UnavailablePlanner)
    assert factory.get_relevance_reranker() is None
    with pytest.raises(LlmError, match="Unsupported LLM_PROVIDER"):
        await planner.plan("Любой запрос")
