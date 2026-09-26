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
async def test_demo_ranks_lexical_evidence_and_keeps_partial_candidates_visible() -> None:
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
    assert len(result.items) == 3
    assert result.items[0].verification_status == "terms_found"
    assert all(item.verification_status != "terms_found" for item in result.items[1:])
    assert result.items[0].missing_terms == []


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
    assert "GigaChat" in page.text
    assert "/api/cases/search-with-evidence" in page.text


def test_demo_status_reports_gigachat_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.api.routes import demo as demo_routes

    monkeypatch.setattr(
        demo_routes,
        "get_settings",
        lambda: SimpleNamespace(
            court_provider="parser_api",
            parser_api_key="parser-key",
            llm_provider="gigachat",
            gigachat_auth_key="gigachat-key",
            gigachat_plan_model="GigaChat-2",
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/demo/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload.pop("configuration_only") is True
    assert len(payload.pop("build_id")) == 12
    assert payload == {
        "court_provider": "parser_api",
        "llm_provider": "gigachat",
        "model": "GigaChat-2",
        "provider_ready": True,
        "llm_ready": True,
        "ready": True,
        "ollama_ready": False,
        "model_available": False,
    }
