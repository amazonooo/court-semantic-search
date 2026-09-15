from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...models import (
    CaseSearchParams,
    CaseSearchResult,
    CourtCase,
    PreferredDocumentTextResponse,
)
from ...providers.base import CourtProvider
from ...providers.factory import get_court_provider
from ...services.case_text import (
    PreferredDocumentNotFoundError,
    extract_preferred_document_text,
)
from ...services.cases import CaseAggregationService
from ...services.pdf import PdfExtractionError

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("/search", response_model=CaseSearchResult)
async def search_cases(
    params: Annotated[CaseSearchParams, Query()],
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> CaseSearchResult:
    service = CaseAggregationService(provider)
    return await service.search_cases(
        params.to_document_search_params(),
        max_pages=params.max_pages,
    )


@router.post(
    "/extract-preferred-text",
    response_model=PreferredDocumentTextResponse,
)
async def extract_preferred_case_text(
    case: CourtCase,
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> PreferredDocumentTextResponse:
    try:
        return await extract_preferred_document_text(case, provider)
    except PreferredDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PdfExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
