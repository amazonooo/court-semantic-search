from datetime import date

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.main import app
from backend.app.models import (
    CaseCollectionParams,
    CourtDocument,
    DocumentSearchParams,
    DocumentSearchResult,
)
from backend.app.providers.base import CourtProvider
from backend.app.providers.factory import get_court_provider
from backend.app.services.cases import CaseAggregationService


class FakeCourtProvider(CourtProvider):
    def __init__(
        self,
        initial: DocumentSearchResult,
        *,
        case_results: dict[str, DocumentSearchResult] | None = None,
    ) -> None:
        self.initial = initial
        self.case_results = case_results or {}
        self.requests: list[DocumentSearchParams] = []

    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        self.requests.append(params)
        if params.case_number and params.case_number in self.case_results:
            return self.case_results[params.case_number]
        return self.initial

    async def download_pdf(self, file_url: str) -> bytes | None:
        return None


def make_document(
    document_id: str,
    *,
    case_id: str,
    case_number: str,
    registration_date: date | None,
    instance_level: int = 1,
) -> CourtDocument:
    return CourtDocument(
        document_id=document_id,
        case_id=case_id,
        case_number=case_number,
        case_url=f"https://kad.arbitr.ru/Card/{case_id}",
        registration_date=registration_date,
        instance_number=f"instance-{document_id}",
        instance_level=instance_level,
        court="Test court",
        document_type="Решение" if instance_level == 1 else "Постановление",
        file_name=f"{document_id}.pdf",
        file_url=f"https://kad.arbitr.ru/Document/Pdf/{document_id}.pdf",
    )


def test_document_search_rejects_inverted_date_range() -> None:
    with pytest.raises(ValidationError):
        DocumentSearchParams(
            text="налоговый спор",
            dateFrom=date(2025, 1, 2),
            dateTo=date(2025, 1, 1),
        )


@pytest.mark.asyncio
async def test_date_range_is_hard_filter_for_candidate_cases() -> None:
    in_range = make_document(
        "in-range",
        case_id="case-in",
        case_number="А40-1/2025",
        registration_date=date(2025, 6, 1),
    )
    too_old = make_document(
        "too-old",
        case_id="case-old",
        case_number="А40-2/2024",
        registration_date=date(2024, 12, 31),
    )
    unknown_date = make_document(
        "unknown-date",
        case_id="case-unknown",
        case_number="А40-3/2025",
        registration_date=None,
    )

    provider = FakeCourtProvider(
        DocumentSearchResult(
            count=3,
            pages=1,
            page=1,
            items=[in_range, too_old, unknown_date],
        )
    )

    result = await CaseAggregationService(provider).search_cases(
        DocumentSearchParams(
            text="налоговый спор",
            dateFrom=date(2025, 1, 1),
            dateTo=date(2025, 12, 31),
        )
    )

    assert result.filtered_out_by_date == 2
    assert result.candidate_unique_document_count == 1
    assert result.case_count == 1
    assert result.items[0].case_id == "case-in"


@pytest.mark.asyncio
async def test_case_expansion_keeps_full_history_after_date_match() -> None:
    matched_appeal = make_document(
        "appeal-2025",
        case_id="case-1",
        case_number="А40-1/2024",
        registration_date=date(2025, 3, 10),
        instance_level=2,
    )
    first_instance_before_period = make_document(
        "first-2024",
        case_id="case-1",
        case_number="А40-1/2024",
        registration_date=date(2024, 11, 20),
        instance_level=1,
    )

    provider = FakeCourtProvider(
        DocumentSearchResult(
            count=1,
            pages=1,
            page=1,
            items=[matched_appeal],
        ),
        case_results={
            "А40-1/2024": DocumentSearchResult(
                count=2,
                pages=1,
                page=1,
                items=[matched_appeal, first_instance_before_period],
            )
        },
    )

    result = await CaseAggregationService(provider).search_cases(
        DocumentSearchParams(
            text="обстоятельства сделки",
            dateFrom=date(2025, 1, 1),
            dateTo=date(2025, 12, 31),
        ),
        expand_cases=True,
    )

    assert result.filtered_out_by_date == 0
    assert result.case_count == 1
    assert {document.document_id for document in result.items[0].documents} == {
        "appeal-2025",
        "first-2024",
    }


def test_collection_endpoint_maps_participant_and_dates_to_search() -> None:
    document = make_document(
        "doc-1",
        case_id="case-1",
        case_number="А40-1/2025",
        registration_date=date(2025, 5, 1),
    )
    provider = FakeCourtProvider(
        DocumentSearchResult(
            count=1,
            pages=1,
            page=1,
            items=[document],
        )
    )
    app.dependency_overrides[get_court_provider] = lambda: provider

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/collections/search",
                json={
                    "participant": "7701234567",
                    "court": "Арбитражный суд города Москвы",
                    "dateFrom": "2025-01-01",
                    "dateTo": "2025-12-31",
                    "expandCases": False,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["case_count"] == 1
    assert response.json()["filtered_out_by_date"] == 0
    assert provider.requests[0].inn == "7701234567"
    assert provider.requests[0].court == "Арбитражный суд города Москвы"
    assert provider.requests[0].date_from == date(2025, 1, 1)
    assert provider.requests[0].date_to == date(2025, 12, 31)


def test_collection_model_requires_supported_search_criterion() -> None:
    with pytest.raises(ValidationError):
        CaseCollectionParams(
            dateFrom=date(2025, 1, 1),
            dateTo=date(2025, 12, 31),
        )
