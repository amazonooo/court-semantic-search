from collections import defaultdict

from ..llm.base import QueryPlanner
from ..models import (
    CourtDocument,
    DocumentSearchParams,
    RetrievedCase,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from ..providers.base import CourtProvider
from .cases import CaseAggregationService


class SemanticSearchService:
    def __init__(self, provider: CourtProvider, planner: QueryPlanner) -> None:
        self._provider = provider
        self._planner = planner

    async def search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        plan = await self._planner.plan(request.description)
        unique: dict[str, CourtDocument] = {}
        matched: dict[str, set[str]] = defaultdict(set)
        for query in plan.queries:
            for page in range(1, request.max_pages_per_query + 1):
                result = await self._provider.search_documents(
                    DocumentSearchParams(text=query, page=page)
                )
                for document in result.items:
                    unique.setdefault(document.document_id, document)
                    matched[document.document_id].add(query)
                if page >= result.pages:
                    break

        cases = CaseAggregationService(self._provider)._group_documents_by_case(
            list(unique.values())
        )
        return SemanticSearchResponse(
            plan=plan,
            query_count=len(plan.queries),
            document_count=len(unique),
            case_count=len(cases),
            items=[
                RetrievedCase(
                    case=case,
                    matched_queries=[
                        query for query in plan.queries
                        if any(query in matched[document.document_id] for document in case.documents)
                    ],
                )
                for case in cases
            ],
        )
