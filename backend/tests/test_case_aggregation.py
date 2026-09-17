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
    def __init__(
        self,
        pages: dict[int, DocumentSearchResult],
        *,
        case_pages: dict[str, dict[int, DocumentSearchResult]] | None = None,
    ) -> None:
        self.pages = pages
        self.case_pages = case_pages or {}
        self.requested_pages: list[int] = []
        self.requests: list[DocumentSearchParams] = []

    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        self.requested_pages.append(params.page)
        self.requests.append(params)
        if params.case_number and params.case_number in self.case_pages:
            return self.case_pages[params.case_number][params.page]
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
    document_type: str = "Test document",
    content_types: list[str] | None = None,
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
        document_type=document_type,
        content_types=content_types or [],
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
    assert result.candidate_unique_document_count == 3
    assert result.case_expansion_pages_fetched == 0
    assert result.expanded_case_count == 0
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
    assert case_one.preferred_document_id == "doc-3"
    assert case_one.preferred_document.document_id == "doc-3"
    assert [document.document_id for document in case_one.first_instance_documents] == [
        "doc-1"
    ]
    assert case_one.factual_base_document.document_id == "doc-1"
    assert case_one.first_instance_document.document_id == "doc-1"
    assert case_one.appellate_documents == []
    assert [document.document_id for document in case_one.cassation_documents] == [
        "doc-3"
    ]
    assert case_one.procedural_documents == []
    assert case_one.latest_substantive_document is None


@pytest.mark.asyncio
async def test_search_cases_expands_candidate_case_and_filters_by_case_id() -> None:
    appellate = make_document(
        "appeal",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=2,
        registration_date=date(2023, 6, 1),
        document_type="Постановление апелляционной инстанции",
    )
    first_instance = make_document(
        "first-instance",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=1,
        registration_date=date(2023, 1, 10),
        document_type="Решение",
    )
    procedural = make_document(
        "procedural",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=2,
        registration_date=date(2023, 5, 20),
        document_type="Определение",
    )
    stray_same_number = make_document(
        "stray",
        case_id="other-case-id",
        case_number="А40-1/2023",
        instance_level=1,
        registration_date=date(2023, 2, 1),
    )

    provider = FakeCourtProvider(
        {
            1: DocumentSearchResult(
                count=1,
                pages=1,
                page=1,
                items=[appellate],
            )
        },
        case_pages={
            "А40-1/2023": {
                1: DocumentSearchResult(
                    count=4,
                    pages=1,
                    page=1,
                    items=[appellate, first_instance, procedural, stray_same_number],
                )
            }
        },
    )

    result = await CaseAggregationService(provider).search_cases(
        DocumentSearchParams(text="обстоятельства сделки"),
        expand_cases=True,
        max_cases_to_expand=10,
        max_case_pages=3,
    )

    assert result.candidate_unique_document_count == 1
    assert result.case_expansion_pages_fetched == 1
    assert result.expanded_case_count == 1
    assert result.unique_document_count == 3
    assert result.case_count == 1

    case = result.items[0]
    assert case.case_id == "case-1"
    assert {document.document_id for document in case.documents} == {
        "appeal",
        "first-instance",
        "procedural",
    }
    assert "stray" not in {document.document_id for document in case.documents}
    assert [document.document_id for document in case.first_instance_documents] == [
        "first-instance"
    ]
    assert case.factual_base_document.document_id == "first-instance"
    assert case.first_instance_document.document_id == "first-instance"
    assert [document.document_id for document in case.appellate_documents] == [
        "appeal",
        "procedural",
    ]
    assert [document.document_id for document in case.procedural_documents] == [
        "procedural"
    ]

    assert provider.requests[0].text == "обстоятельства сделки"
    assert provider.requests[1].case_number == "А40-1/2023"


@pytest.mark.asyncio
async def test_search_cases_prefers_substantive_act_over_procedural_definition() -> None:
    procedural_definition = make_document(
        "procedural-definition",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=2,
        registration_date=date(2023, 8, 1),
        document_type="Определение",
        content_types=[
            "Принять к производству апелляционную жалобу",
            "Назначить дело к судебному разбирательству",
        ],
    )
    substantive_resolution = make_document(
        "substantive-resolution",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=2,
        registration_date=date(2023, 7, 26),
        document_type="Постановление апелляционной инстанции",
        content_types=["Оставить определение без изменения, жалобу без удовлетворения"],
    )

    provider = FakeCourtProvider(
        {
            1: DocumentSearchResult(
                count=2,
                pages=1,
                page=1,
                items=[procedural_definition, substantive_resolution],
            )
        }
    )

    result = await CaseAggregationService(provider).search_cases(
        DocumentSearchParams(text="налоговый спор")
    )

    case = result.items[0]
    assert case.preferred_document_id == "substantive-resolution"
    assert case.preferred_document.document_type == "Постановление апелляционной инстанции"
    assert case.latest_substantive_document.document_id == "substantive-resolution"
    assert case.procedural_documents == [procedural_definition]


@pytest.mark.asyncio
async def test_first_instance_is_factual_base_even_when_higher_instance_is_newer() -> None:
    first_instance = make_document(
        "first-instance",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=1,
        registration_date=date(2023, 1, 10),
        document_type="Решение",
        content_types=["Взыскать задолженность"],
    )
    appellate = make_document(
        "appeal",
        case_id="case-1",
        case_number="А40-1/2023",
        instance_level=2,
        registration_date=date(2023, 7, 26),
        document_type="Постановление апелляционной инстанции",
        content_types=["Оставить решение без изменения"],
    )

    provider = FakeCourtProvider(
        {
            1: DocumentSearchResult(
                count=2,
                pages=1,
                page=1,
                items=[first_instance, appellate],
            )
        }
    )

    result = await CaseAggregationService(provider).search_cases(
        DocumentSearchParams(text="задолженность")
    )

    case = result.items[0]
    assert case.factual_base_document.document_id == "first-instance"
    assert case.first_instance_document.document_id == "first-instance"
    assert case.latest_substantive_document.document_id == "appeal"
    assert [document.document_id for document in case.appellate_documents] == [
        "appeal"
    ]
    assert [document.document_id for document in case.documents] == [
        "appeal",
        "first-instance",
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
                    "expandCases": True,
                    "maxCasesToExpand": 10,
                    "maxCasePages": 3,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert provider.requested_pages == [1]
    assert response.json()["case_count"] == 0
    assert response.json()["candidate_unique_document_count"] == 0
    assert response.json()["case_expansion_pages_fetched"] == 0
