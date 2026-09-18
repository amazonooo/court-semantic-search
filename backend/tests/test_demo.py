from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.llm.base import QueryPlanner
from backend.app.main import app
from backend.app.models import SearchPlan, EvidenceSearchRequest
from backend.app.providers.mock import MockCourtProvider
from backend.app.services.pdf import extract_pdf_text
from backend.app.services.semantic_search import SemanticSearchService


class DemoPlanner(QueryPlanner):
    async def plan(self, description: str) -> SearchPlan:
        return SearchPlan(
            queries=[
                "присоединение заем проценты налоговая выгода",
                "убытки после присоединения заем налог на прибыль",
            ],
            must_have=["заем", "присоединение", "налоговая выгода"],
            exclude=[],
        )


@pytest.mark.asyncio
async def test_demo_finds_exact_scheme_ahead_of_partial_matches() -> None:
    result = await SemanticSearchService(MockCourtProvider(), DemoPlanner()).search_with_evidence(
        EvidenceSearchRequest(
            description="Компания присоединила заемщика и учла проценты по займу и убытки",
        )
    )

    assert result.case_count == 3
    assert result.items[0].case.case_number == "DEMO-001"
    assert result.items[0].case.preferred_document_id == "demo-tax-decision"
    assert result.items[0].coverage == 1.0
    assert "присоединило" in result.items[0].excerpt
    assert all(item.coverage < 1.0 for item in result.items[1:])
    assert any("заем" in item.missing_terms for item in result.items[1:])
    assert any("присоединение" in item.missing_terms for item in result.items[1:])


def test_demo_pdf_is_downloadable_and_extractable() -> None:
    with TestClient(app) as client:
        response = client.get("/api/demo/documents/demo-tax-decision/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    extracted = extract_pdf_text(response.content)
    assert "налоговую" in extracted
    assert "выгоду" in extracted


def test_demo_page_is_served() -> None:
    with TestClient(app) as client:
        page = client.get("/demo")
    assert page.status_code == 200
    assert "Yandex LLM" in page.text
    assert "/api/cases/semantic-search" in page.text


def test_demo_status_reports_yandex_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.api.routes import demo as demo_routes

    monkeypatch.setattr(
        demo_routes,
        "get_settings",
        lambda: SimpleNamespace(
            court_provider="parser_api",
            parser_api_key="parser-key",
            llm_provider="yandex",
            yandex_api_key="yandex-key",
            yandex_folder_id="folder-1",
            yandex_model="yandexgpt-lite",
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/demo/status")

    assert response.status_code == 200
    assert response.json() == {
        "court_provider": "parser_api",
        "llm_provider": "yandex",
        "model": "yandexgpt-lite",
        "provider_ready": True,
        "llm_ready": True,
        "ready": True,
        "ollama_ready": False,
        "model_available": False,
    }
