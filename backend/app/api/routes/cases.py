from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ...models import CaseSearchParams, CaseSearchResult
from ...providers.base import CourtProvider
from ...providers.factory import get_court_provider
from ...services.cases import CaseAggregationService

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
