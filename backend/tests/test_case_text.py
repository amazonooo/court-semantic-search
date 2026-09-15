import fitz
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import (
    CourtCase,
    CourtDocument,
    DocumentSearchParams,
    DocumentSearchResult,
)
from backend.app.providers.base import CourtProvider
from backend.app.providers.factory import get_court_provider


class FakeCourtProvider(CourtProvider):
    def __init__(self, pdfs: dict[str, bytes | None]) -> None:
        self.pdfs = pdfs
        self.downloaded_urls: list[str] = []

    async def search_documents(
        self,
        params: DocumentSearchParams,
    ) -> DocumentSearchResult:
        raise NotImplementedError

    async def download_pdf(self, file_url: str) -> bytes | None:
        self.downloaded_urls.append(file_url)
        return self.pdfs.get(file_url)


def make_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def make_case() -> CourtCase:
    procedural = CourtDocument(
        document_id="procedural-definition",
        case_id="case-1",
        case_number="А40-1/2023",
        registration_date="2023-08-01",
        document_type="Определение",
        file_url="https://example.test/procedural.pdf",
    )
    preferred = CourtDocument(
        document_id="substantive-resolution",
        case_id="case-1",
        case_number="А40-1/2023",
        registration_date="2023-07-26",
        document_type="Постановление апелляционной инстанции",
        file_url="https://example.test/preferred.pdf",
    )
    return CourtCase(
        case_id="case-1",
        case_number="А40-1/2023",
        case_url="https://kad.arbitr.ru/Card/case-1",
        document_count=2,
        latest_document_date="2023-08-01",
        highest_instance_level=2,
        preferred_document_id=preferred.document_id,
        preferred_document=preferred,
        documents=[procedural, preferred],
    )


def test_extract_preferred_text_downloads_only_selected_document() -> None:
    case = make_case()
    provider = FakeCourtProvider(
        {case.preferred_document.file_url: make_pdf("Substantive act text")}
    )
    app.dependency_overrides[get_court_provider] = lambda: provider

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/cases/extract-preferred-text",
                json=case.model_dump(mode="json"),
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert provider.downloaded_urls == [case.preferred_document.file_url]
    assert payload["case_id"] == "case-1"
    assert payload["document_id"] == "substantive-resolution"
    assert (
        payload["document"]["document_type"]
        == "Постановление апелляционной инстанции"
    )
    assert "Substantive act text" in payload["text"]
    assert payload["char_count"] == len(payload["text"])


def test_extract_preferred_text_returns_404_when_document_is_unavailable() -> None:
    case = make_case()
    provider = FakeCourtProvider({})
    app.dependency_overrides[get_court_provider] = lambda: provider

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/cases/extract-preferred-text",
                json=case.model_dump(mode="json"),
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "Preferred PDF document was not found by the source"
    )


def test_extract_preferred_text_returns_502_for_invalid_pdf() -> None:
    case = make_case()
    provider = FakeCourtProvider({case.preferred_document.file_url: b"not-a-pdf"})
    app.dependency_overrides[get_court_provider] = lambda: provider

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/cases/extract-preferred-text",
                json=case.model_dump(mode="json"),
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json()["detail"] == "Could not open PDF document"
