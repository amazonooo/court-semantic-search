import asyncio
import base64
import binascii
from datetime import datetime
from typing import Any

import httpx

from ..models import CourtDocument, DocumentSearchParams, DocumentSearchResult
from .base import (
    CourtProvider,
    CourtProviderAccessError,
    CourtProviderError,
    CourtProviderTemporaryError,
    CourtProviderValidationError,
)


class ParserApiProvider(CourtProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max(1, max_retries)
        self._client = client

    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        payload = await self._request_json("search", self._build_search_params(params))
        items = [self._normalize_document(item) for item in payload.get("items", [])]

        return DocumentSearchResult(
            count=int(payload.get("count", len(items)) or 0),
            pages=int(payload.get("pages", 0) or 0),
            page=int(payload.get("page", params.page) or params.page),
            items=items,
        )

    async def download_pdf(self, file_url: str) -> bytes | None:
        payload = await self._request_json("pdf_download", {"url": file_url})
        encoded_pdf = payload.get("pdfContent")
        if not encoded_pdf:
            return None

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

    async def _request_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = {"key": self._api_key, **params}
        url = f"{self._base_url}/{endpoint}"

        for attempt in range(self._max_retries):
            try:
                response = await self._send_request(url, request_params)
            except httpx.RequestError as exc:
                if attempt + 1 >= self._max_retries:
                    raise CourtProviderTemporaryError(
                        "Could not connect to Parser API"
                    ) from exc
                await asyncio.sleep(2**attempt)
                continue

            payload = self._parse_json(response)

            if response.status_code == 400:
                raise CourtProviderValidationError(
                    payload.get("error", "Parser API rejected the request"),
                    error_code=payload.get("error_code"),
                )

            if response.status_code == 403:
                raise CourtProviderAccessError(
                    payload.get("error", "Parser API access denied"),
                    error_code=payload.get("error_code"),
                )

            if response.status_code >= 500:
                if attempt + 1 >= self._max_retries:
                    raise CourtProviderTemporaryError(
                        "Parser API is temporarily unavailable",
                        error_code=payload.get("error_code"),
                    )
                await asyncio.sleep(2**attempt)
                continue

            if response.status_code != 200:
                raise CourtProviderError(
                    f"Unexpected Parser API status: {response.status_code}",
                    error_code=payload.get("error_code"),
                )

            if payload.get("done") == 1:
                return payload

            if payload.get("error"):
                raise CourtProviderError(
                    payload["error"],
                    error_code=payload.get("error_code"),
                )

            if attempt + 1 >= self._max_retries:
                raise CourtProviderTemporaryError(
                    "Parser API source did not return a stable response"
                )
            await asyncio.sleep(2**attempt)

        raise CourtProviderTemporaryError("Parser API request failed")

    async def _send_request(
        self, url: str, params: dict[str, Any]
    ) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(url, params=params)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.get(url, params=params)

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
