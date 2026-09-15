import base64

import httpx
import pytest

from backend.app.models import DocumentSearchParams
from backend.app.providers.parser_api import ParserApiProvider


@pytest.mark.asyncio
async def test_search_documents_normalizes_parser_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "test-key"
        assert request.url.params["caseNumber"] == "А53-30848/2015"
        return httpx.Response(
            200,
            json={
                "done": 1,
                "count": 1,
                "pages": 1,
                "page": 1,
                "items": [
                    {
                        "CaseId": "c9babcc7-e797-429c-b6e4-6287b5d7334a",
                        "CaseUrl": "https://kad.arbitr.ru/Card/case-id",
                        "RegistrationDate": "21.12.2018",
                        "InstanceNumber": "15АП-20855/2018",
                        "CaseNumber": "А53-30848/2015",
                        "FileName": "document.pdf",
                        "FileUrl": "https://kad.arbitr.ru/Document/Pdf/case/document.pdf",
                        "InstanceLevel": 2,
                        "Court": "15 арбитражный апелляционный суд",
                        "Type": "Постановление апелляционной инстанции",
                        "ContentTypes": ["Оставить без изменения"],
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ParserApiProvider(
            api_key="test-key",
            base_url="https://parser-api.com/parser/ras_arbitr_api",
            client=client,
        )
        result = await provider.search_documents(
            DocumentSearchParams(caseNumber="А53-30848/2015")
        )

    assert result.count == 1
    assert result.items[0].case_number == "А53-30848/2015"
    assert result.items[0].instance_level == 2
    assert result.items[0].registration_date.isoformat() == "2018-12-21"
    assert result.items[0].document_id


@pytest.mark.asyncio
async def test_download_pdf_decodes_base64() -> None:
    expected_pdf = b"%PDF-test"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "test-key"
        assert request.url.params["url"] == "https://kad.arbitr.ru/test.pdf"
        return httpx.Response(
            200,
            json={
                "done": 1,
                "pdfContent": base64.b64encode(expected_pdf).decode("ascii"),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ParserApiProvider(
            api_key="test-key",
            base_url="https://parser-api.com/parser/ras_arbitr_api",
            client=client,
        )
        result = await provider.download_pdf("https://kad.arbitr.ru/test.pdf")

    assert result == expected_pdf
