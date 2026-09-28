import base64

import httpx
import pytest

from backend.app.models import DocumentSearchParams, SearchDiagnostics
from backend.app.providers.parser_api import ParserApiProvider
from backend.app.services.search_runtime import active_diagnostics


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

    diagnostics = SearchDiagnostics(request_id="test")
    token = active_diagnostics.set(diagnostics)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = ParserApiProvider(
                api_key="test-key",
                base_url="https://parser-api.com/parser/ras_arbitr_api",
                client=client,
            )
            result = await provider.search_documents(
                DocumentSearchParams(caseNumber="А53-30848/2015")
            )
    finally:
        active_diagnostics.reset(token)

    assert result.count == 1
    assert result.items[0].case_number == "А53-30848/2015"
    assert result.items[0].instance_level == 2
    assert result.items[0].registration_date.isoformat() == "2018-12-21"
    assert result.items[0].document_id
    assert diagnostics.http_attempts == diagnostics.parser_successes == 1


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


@pytest.mark.asyncio
async def test_parser_redacts_keys_in_httpx_logs_and_error_text(caplog):
    from backend.app.providers.base import CourtProviderAccessError
    key = 'do-not-print-this-key'
    def handler(request):
        return httpx.Response(403, json={'error':f'Invalid key {key}', 'error_code':40301})
    caplog.set_level('INFO', logger='httpx')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ParserApiProvider(api_key=key, base_url='https://parser-api.com/test', client=client)
        with pytest.raises(CourtProviderAccessError) as exc:
            await provider.search_documents(DocumentSearchParams(text='налог'))
    assert key not in str(exc.value)
    assert key not in caplog.text
    assert '[REDACTED]' in caplog.text


@pytest.mark.asyncio
async def test_retries_share_one_wall_clock_budget():
    import asyncio
    from time import monotonic
    from backend.app.providers.base import CourtProviderTemporaryError
    calls = []
    async def handler(request):
        calls.append(1)
        return httpx.Response(503, text='<html>Unavailable</html>')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ParserApiProvider(api_key='test', base_url='https://parser-api.com/test',
                                     client=client, max_retries=3, timeout_seconds=.04)
        start = monotonic()
        with pytest.raises(CourtProviderTemporaryError):
            await provider.search_documents(DocumentSearchParams(text='налог'))
    assert monotonic() - start < .4
    assert len(calls) == 1  # Backoff is cancelled within the same total budget.


@pytest.mark.asyncio
async def test_case_id_disagreement_with_pdf_url_is_not_a_valid_search_result():
    from backend.app.providers.base import CourtProviderTemporaryError
    def handler(request):
        return httpx.Response(200, json={'done': 1, 'items': [{
            'CaseId': '11111111-1111-1111-1111-111111111111',
            'CaseNumber': 'А40-1/2025',
            'FileUrl': 'https://kad.arbitr.ru/Document/Pdf/22222222-2222-2222-2222-222222222222/doc/file.pdf',
        }]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ParserApiProvider(api_key='test', base_url='https://parser-api.com/test', client=client)
        with pytest.raises(CourtProviderTemporaryError):
            await provider.search_documents(DocumentSearchParams(text='налог'))
