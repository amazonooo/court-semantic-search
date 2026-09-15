from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ...models import CaseSearchResult, DocumentSearchParams
from ...providers.base import CourtProvider
from ...providers.factory import get_court_provider
from ...services.cases import CaseAggregationService

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("/search", response_model=CaseSearchResult)
async def search_cases(
    params: Annotated[DocumentSearchParams, Query()],
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
    max_pages: Annotated[int, Query(alias="maxPages", ge=1, le=20)] = 3,
) -> CaseSearchResult:
    service = CaseAggregationService(provider)
    return await service.search_cases(params, max_pages=max_pages)
