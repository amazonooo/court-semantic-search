from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...models import (
    CaseDocumentTextResponse,
    CaseSearchParams,
    CaseSearchResult,
    CourtCase,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
    PreferredDocumentTextResponse,
    SearchPlan,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from ...llm.base import LlmError, QueryPlanner
from ...llm.factory import get_query_planner
from ...providers.base import CourtProvider, CourtProviderError
from ...providers.factory import get_court_provider
from ...services.case_text import (
    PreferredDocumentNotFoundError,
    extract_case_document_text,
    extract_preferred_document_text,
)
from ...services.cases import CaseAggregationService
from ...services.pdf import PdfExtractionError
from ...services.semantic_search import SemanticSearchService

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.post("/plan", response_model=SearchPlan)
async def plan_case_search(
    request: SemanticSearchRequest,
    planner: Annotated[QueryPlanner, Depends(get_query_planner)],
) -> SearchPlan:
    try:
        return request.plan or await planner.plan(request.description)
    except LlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/search-with-evidence", response_model=EvidenceSearchResponse)
async def search_cases_with_evidence(
    request: EvidenceSearchRequest,
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
    planner: Annotated[QueryPlanner, Depends(get_query_planner)],
) -> EvidenceSearchResponse:
    try:
        return await SemanticSearchService(provider, planner).search_with_evidence(request)
    except (LlmError, CourtProviderError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search_cases(
    request: SemanticSearchRequest,
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
    planner: Annotated[QueryPlanner, Depends(get_query_planner)],
) -> SemanticSearchResponse:
    try:
        return await SemanticSearchService(provider, planner).search(request)
    except (LlmError, CourtProviderError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/search", response_model=CaseSearchResult)
async def search_cases(
    params: Annotated[CaseSearchParams, Query()],
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> CaseSearchResult:
    service = CaseAggregationService(provider)
    return await service.search_cases(
        params.to_document_search_params(),
        max_pages=params.max_pages,
        expand_cases=params.expand_cases,
        max_cases_to_expand=params.max_cases_to_expand,
        max_case_pages=params.max_case_pages,
    )


@router.post(
    "/extract-factual-base-text",
    response_model=CaseDocumentTextResponse,
)
async def extract_factual_base_case_text(
    case: CourtCase,
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> CaseDocumentTextResponse:
    try:
        return await extract_case_document_text(
            case,
            provider,
            role="factual_base",
        )
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


@router.post(
    "/extract-latest-substantive-text",
    response_model=CaseDocumentTextResponse,
)
async def extract_latest_substantive_case_text(
    case: CourtCase,
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> CaseDocumentTextResponse:
    try:
        return await extract_case_document_text(
            case,
            provider,
            role="latest_substantive",
        )
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
