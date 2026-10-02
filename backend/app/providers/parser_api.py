import asyncio
import base64
import binascii
import ssl
import logging
import re
from time import monotonic
from datetime import datetime
from typing import Any

import httpx
import truststore

from ..models import CourtDocument, DocumentSearchParams, DocumentSearchResult, SearchEvent
from ..services.search_runtime import active_diagnostics
from ..services.pdf import MAX_PDF_BYTES
from .base import (
    CourtProvider,
    CourtProviderAccessError,
    CourtProviderError,
    CourtProviderTemporaryError,
    CourtProviderValidationError,
)


class _RedactAccessKey(logging.Filter):
    def filter(self, record):
        # httpx logs the complete GET URL at INFO. Parser's API puts its key in
        # that URL, so sanitize before any configured log handler receives it.
        message = record.getMessage()
        record.msg = re.sub(r"([?&]key=)[^&\s\"']+", r"\1[REDACTED]", message)
        record.args = ()
        return True


logging.getLogger("httpx").addFilter(_RedactAccessKey())


class ParserApiProvider(CourtProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 25.0,
        max_retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max(1, max_retries)
        self._client = client

    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        payload = await self._request_json("search", self._build_search_params(params))
        try:
            raw_items = payload["items"]
            if not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items):
                raise ValueError("Invalid document items")
            items = [self._normalize_document(item) for item in raw_items]
            return DocumentSearchResult(
                count=int(payload.get("count", len(items)) or 0),
                pages=int(payload.get("pages", 0) or 0),
                page=int(payload.get("page", params.page) or params.page),
                items=items,
            )
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise CourtProviderTemporaryError("Parser API returned invalid document metadata") from exc

    async def download_pdf(self, file_url: str) -> bytes | None:
        payload = await self._request_json("pdf_download", {"url": file_url})
        encoded_pdf = payload.get("pdfContent")
        if not encoded_pdf:
            return None

        if not isinstance(encoded_pdf, str) or len(encoded_pdf) > ((MAX_PDF_BYTES + 2) // 3) * 4:
            raise CourtProviderTemporaryError("Parser API PDF exceeds the processing limit or has an invalid format")
        try:
            return base64.b64decode(encoded_pdf, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CourtProviderTemporaryError(
                "Parser API returned invalid base64 PDF content"
            ) from exc

    def _build_search_params(self, params: DocumentSearchParams) -> dict[str, str | int]:
        result: dict[str, str | int] = {"page": params.page}
        mappings: list[tuple[str, Any]] = [
            ("caseNumber", params.case_number),
            ("inn", params.inn),
            ("text", params.text),
            ("court", params.court),
            ("disputeType", params.dispute_type),
            ("disputeCategory", params.dispute_category),
        ]
        for key, value in mappings:
            if value is not None and str(value).strip():
                result[key] = str(value).strip()

        if params.date_from is not None:
            result["dateFrom"] = params.date_from.isoformat()
        if params.date_to is not None:
            result["dateTo"] = params.date_to.isoformat()
        return result

    def _safe_error(self, value: Any, fallback: str) -> str:
        message = str(value or fallback).replace(self._api_key, "[REDACTED]")
        if message.strip().casefold().startswith("test access is disabled for this account"):
            return (
                "Parser API: тестовый доступ для этой учётной записи отключён. "
                "Проверьте доступ к поиску судебных актов ras.arbitr.ru в кабинете Parser API."
            )
        return re.sub(r"([?&]key=)[^&\s\"']+", r"\1[REDACTED]", message)[:500]

    async def _request_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        # One deadline includes all attempts and backoff, including injected
        # clients; a retry must never multiply the wall-clock timeout.
        try:
            return await asyncio.wait_for(
                self._request_json_inner(endpoint, params), self._timeout
            )
        except TimeoutError as exc:
            raise CourtProviderTemporaryError("Parser API operation timed out") from exc

    async def _request_json_inner(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = {"key": self._api_key, **params}
        url = f"{self._base_url}/{endpoint}"
        for attempt in range(self._max_retries):
            diagnostics = active_diagnostics.get()
            if diagnostics:
                diagnostics.http_attempts += 1
            start = monotonic()
            outcome, http_status, error_code = "ok", None, None
            retry = False
            try:
                response = await self._send_request(url, request_params)
                http_status = response.status_code
                # Gateways may return HTML; classify status before parsing JSON.
                if http_status in (429, 502, 503, 504) or http_status >= 500:
                    raise CourtProviderTemporaryError(f"Parser API returned HTTP {http_status}")
                payload = self._parse_json(response)
                error_code = payload.get("error_code")
                if http_status == 400:
                    raise CourtProviderValidationError(
                        self._safe_error(payload.get("error"), "Parser API rejected the request"),
                        error_code=error_code,
                    )
                if http_status in (401, 403):
                    raise CourtProviderAccessError(
                        self._safe_error(payload.get("error"), "Parser API access denied"),
                        error_code=error_code,
                    )
                if http_status != 200:
                    raise CourtProviderError(f"Unexpected Parser API status: {http_status}", error_code=error_code)
                if payload.get("done") == 1:
                    if diagnostics:
                        diagnostics.parser_successes += 1
                    return payload
                if payload.get("error"):
                    raise CourtProviderError(
                        self._safe_error(payload["error"], "Parser API error"), error_code=error_code)
                raise CourtProviderTemporaryError("Parser API source did not return a stable response")
            except httpx.RequestError as exc:
                outcome = type(exc).__name__
                if attempt + 1 >= self._max_retries:
                    raise CourtProviderTemporaryError(
                        f"Could not connect to Parser API ({type(exc).__name__})"
                    ) from exc
                retry = True
            except CourtProviderTemporaryError as exc:
                outcome = type(exc).__name__
                if attempt + 1 >= self._max_retries:
                    raise
                retry = True
            except asyncio.CancelledError:
                outcome = "cancelled_or_timeout"
                raise
            except CourtProviderError as exc:
                outcome = type(exc).__name__
                raise
            finally:
                if diagnostics:
                    diagnostics.events.append(SearchEvent(
                        stage="parser_http", operation=endpoint, attempt=attempt + 1,
                        seconds=round(monotonic() - start, 3), outcome=outcome,
                        http_status=http_status, error_code=error_code,
                    ))
            if retry:
                await asyncio.sleep(2 ** attempt)
        raise CourtProviderTemporaryError("Parser API request failed")

    async def _send_request(self, url: str, params: dict[str, Any]) -> httpx.Response:
        async def fetch(client):
            chunks, size = [], 0
            # Reject oversized bodies during download, before JSON/base64 parsing.
            max_response_bytes = ((MAX_PDF_BYTES + 2) // 3) * 4 + 1_000_000
            async with client.stream("GET", url, params=params, timeout=self._timeout) as response:
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_response_bytes:
                        raise CourtProviderError("Parser API response exceeds the processing limit")
                    chunks.append(chunk)
                return httpx.Response(response.status_code, content=b"".join(chunks),
                                      headers={"content-type": response.headers.get("content-type", "")})
        if self._client is not None:
            return await fetch(self._client)
        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        async with httpx.AsyncClient(timeout=self._timeout, verify=ssl_context) as client:
            return await fetch(client)

    @staticmethod
    def _parse_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise CourtProviderTemporaryError(
                "Parser API returned a non-JSON response"
            ) from exc
        if not isinstance(payload, dict):
            raise CourtProviderTemporaryError(
                "Parser API returned an unexpected response format"
            )
        return payload

    @staticmethod
    def _normalize_document(item: dict[str, Any]) -> CourtDocument:
        registration_date = None
        raw_date = item.get("RegistrationDate")
        if raw_date:
            try:
                registration_date = datetime.strptime(raw_date, "%d.%m.%Y").date()
            except (TypeError, ValueError):
                registration_date = None

        file_url = str(item.get("FileUrl") or "")
        source_case_id = str(item.get("CaseId") or "")
        linked_case = re.search(r"/Document/Pdf/([0-9a-f-]{36})/", file_url, re.I)
        if linked_case and source_case_id and linked_case.group(1).lower() != source_case_id.lower():
            raise ValueError("PDF URL and source CaseId disagree")
        fallback_id = "|".join(
            str(item.get(key) or "")
            for key in ("CaseId", "InstanceNumber", "FileName", "RegistrationDate")
        )

        return CourtDocument(
            document_id=CourtDocument.build_document_id(file_url, fallback_id),
            case_id=str(item.get("CaseId") or ""),
            case_number=str(item.get("CaseNumber") or ""),
            case_url=item.get("CaseUrl"),
            registration_date=registration_date,
            instance_number=item.get("InstanceNumber"),
            instance_level=item.get("InstanceLevel"),
            court=item.get("Court"),
            document_type=item.get("Type"),
            content_types=list(item.get("ContentTypes") or []),
            file_name=item.get("FileName"),
            file_url=file_url,
        )
