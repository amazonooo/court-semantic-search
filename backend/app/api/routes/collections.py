from typing import Annotated

from fastapi import APIRouter, Depends

from ...models import CaseCollectionParams, CaseSearchResult
from ...providers.base import CourtProvider
from ...providers.factory import get_court_provider
from ...services.cases import CaseAggregationService

router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.post("/search", response_model=CaseSearchResult)
async def search_collection(
    params: CaseCollectionParams,
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> CaseSearchResult:
    case_params = params.to_case_search_params()
    service = CaseAggregationService(provider)
    return await service.search_cases(
        case_params.to_document_search_params(),
        max_pages=case_params.max_pages,
        expand_cases=case_params.expand_cases,
        max_cases_to_expand=case_params.max_cases_to_expand,
        max_case_pages=case_params.max_case_pages,
    )
