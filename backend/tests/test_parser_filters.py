"""LLM output must not turn inferred legal topics into Parser API filters."""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from backend.app.llm.prompt import parse_search_plan
from backend.app.models import (
    CourtDocument, DocumentSearchParams, DocumentSearchResult, SearchFilters,
    SemanticSearchRequest,
)
from backend.app.parser_filters import DISPUTE_TYPES
from backend.app.providers.base import CourtProvider
from backend.app.services.semantic_search import SemanticSearchService


DESCRIPTION = (
    "После присоединения компании-заемщика правопреемник учел проценты по займу "
    "и убытки при расчете налога на прибыль. Налоговая оспаривает деловую цель."
)


class RecordingProvider(CourtProvider):
    def __init__(self):
        self.calls = []

    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        self.calls.append(params)
        document = CourtDocument(
            document_id="doc-1", case_id="case-1", case_number="А40-1/2023",
            registration_date=date(2023, 1, 1), file_url="https://example.org/doc.pdf",
        )
        return DocumentSearchResult(count=1, pages=1, page=params.page, items=[document])

    async def download_pdf(self, file_url: str) -> bytes | None:
        return None


def _plan(filters: dict, description: str = DESCRIPTION):
    return parse_search_plan(json.dumps({
        "queries": ["присоединение заемщика проценты", "налоговая выгода реорганизация"],
        "must_have": ["присоединение", "проценты по займу"],
        "exclude": [],
        "filters": filters,
    }, ensure_ascii=False), description)


@pytest.mark.asyncio
async def test_generated_invalid_filter_and_unstated_date_cannot_abort_search():
    plan = _plan({
        "dispute_type": "reorganization_tax_benefit",
        "dispute_category": "tax",
        "date_from": "2020-01-01",
        "court": "АС Московского округа",
    })
    assert plan.filters.dispute_type is None
    assert plan.filters.dispute_category is None
    assert plan.filters.date_from is None
    assert plan.filters.court is None

    provider = RecordingProvider()
    result = await SemanticSearchService(provider, None).search(
        SemanticSearchRequest(description=DESCRIPTION, plan=plan)
    )
    assert result.case_count == 1
    assert len(provider.calls) == 2
    assert all(call.dispute_type is None and call.date_from is None for call in provider.calls)


def test_explicit_official_filters_remain_available_for_any_topic():
    dispute_type = DISPUTE_TYPES[1]
    description = f"Найти дела вида {dispute_type} категории 7.1 после 2020 года."
    plan = _plan({
        "dispute_type": dispute_type.upper(),
        "dispute_category": "7.1",
        "date_from": "2021-01-01",
    }, description)
    assert plan.filters.dispute_type == dispute_type
    assert plan.filters.dispute_category == "7.1"
    assert plan.filters.date_from.isoformat() == "2021-01-01"


def test_explicit_structured_filters_reject_invalid_provider_values():
    with pytest.raises(ValidationError):
        SearchFilters(disputeType="reorganization_tax_benefit")
    with pytest.raises(ValidationError):
        DocumentSearchParams(text="налог", disputeType="reorganization_tax_benefit")
    with pytest.raises(ValidationError):
        SearchFilters(disputeCategory="tax")
    assert DocumentSearchParams(text="банкротство", disputeType=DISPUTE_TYPES[1]).dispute_type == DISPUTE_TYPES[1]


def test_event_year_and_article_number_are_not_search_filters():
    description = "В 2020 году произошло присоединение, применяется статья 7.1 закона. Найти похожие дела."
    plan = _plan({"date_from": "2020-01-01", "dispute_category": "7.1"}, description)
    assert plan.filters.date_from is None
    assert plan.filters.dispute_category is None
