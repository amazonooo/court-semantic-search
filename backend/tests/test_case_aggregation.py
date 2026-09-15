from datetime import date

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import (
    CourtDocument,
    DocumentSearchParams,
    DocumentSearchResult,
)
from backend.app.providers.base import CourtProvider
from backend.app.providers.factory import get_court_provider
from backend.app.services.cases import CaseAggregationService


class FakeCourtProvider(CourtProvider):
    def __init__(self, pages: dict[int, DocumentSearchResult]) -> None:
        self.pages = pages
        self.requested_pages: list[int] = []

    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        self.requested_pages.append(params.page)
        return self.pages[params.page]

    async def download_pdf(self, file_url: str) -> bytes | None:
        return None


def make_document(
    document_id: str,
    *,
    case_id: str,
    case_number: str,
    instance_level: int,
    registration_date: date,
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
        document_type="Test document",
        file_name=f"{document_id}.pdf",
        file_url=f"https://kad.arbitr.ru/Document/Pdf/{document_id}.pdf",
    )


@pytest.mark.asyncio
async def test_search_cases_paginates_deduplicates_and_groups_documents() -> None:
    first_instance = make_document(
        "doc-1",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=1,
        registration_date=date(2023, 1, 10),
    )
    second_case = make_document(
        "doc-2",
        case_id="case-2",
        case_number="А40-2/2023",
        instance_level=1,
        registration_date=date(2023, 2, 1),
    )
    cassation = make_document(
        "doc-3",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=3,
        registration_date=date(2023, 7, 26),
    )

    provider = FakeCourtProvider(
        {
            1: DocumentSearchResult(
                count=4,
                pages=2,
                page=1,
                items=[first_instance, second_case],
            ),
            2: DocumentSearchResult(
                count=4,
                pages=2,
                page=2,
                items=[first_instance, cassation],
            ),
        }
    )

    result = await CaseAggregationService(provider).search_cases(
        DocumentSearchParams(text="налоговый спор"),
        max_pages=3,
    )

    assert provider.requested_pages == [1, 2]
    assert result.source_document_count == 4
    assert result.pages_fetched == 2
    assert result.unique_document_count == 3
    assert result.case_count == 2

    case_one = next(case for case in result.items if case.case_id == "case-1")
    assert case_one.document_count == 2
    assert case_one.highest_instance_level == 3
    assert case_one.latest_document_date == date(2023, 7, 26)
    assert [document.document_id for document in case_one.documents] == [
        "doc-3",
        "doc-1",
    ]


@pytest.mark.asyncio
async def test_search_cases_respects_max_pages_limit() -> None:
    first_document = make_document(
        "doc-1",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=1,
        registration_date=date(2023, 1, 10),
    )

    provider = FakeCourtProvider(
        {
            1: DocumentSearchResult(
                count=10,
                pages=5,
                page=1,
                items=[first_document],
            )
        }
    )

    result = await CaseAggregationService(provider).search_cases(
        DocumentSearchParams(text="налоговый спор"),
        max_pages=1,
    )

    assert provider.requested_pages == [1]
    assert result.pages_fetched == 1
    assert result.source_pages == 5
    assert result.unique_document_count == 1
    assert result.case_count == 1


def test_cases_search_endpoint_accepts_flat_query_parameters() -> None:
    provider = FakeCourtProvider(
        {
            1: DocumentSearchResult(
                count=0,
                pages=0,
                page=1,
                items=[],
            )
        }
    )
    app.dependency_overrides[get_court_provider] = lambda: provider

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/cases/search",
                params={
                    "caseNumber": "15АП-20855/2018",
                    "page": 1,
                    "maxPages": 3,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert provider.requested_pages == [1]
    assert response.json()["case_count"] == 0
