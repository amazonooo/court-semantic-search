from collections import defaultdict
import re

from ..llm.base import QueryPlanner
from ..models import (
    CourtDocument,
    DocumentSearchParams,
    EvidenceCase,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
    RetrievedCase,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from ..providers.base import CourtProvider, CourtProviderError
from .case_text import PreferredDocumentNotFoundError, extract_preferred_document_text
from .cases import CaseAggregationService
from .pdf import PdfExtractionError


def _term_present(text: str, term: str) -> bool:
    normalized_text = text.casefold().replace("ё", "е")
    roots = [word[:4] for word in re.findall(r"[а-яa-z]{4,}", term.casefold().replace("ё", "е"))]
    return bool(roots) and all(root in normalized_text for root in roots)


def _excerpt(text: str, matched_terms: list[str], limit: int = 520) -> str:
    if len(text) <= limit:
        return text
    normalized = text.casefold().replace("ё", "е")
    first = next(
        (normalized.find(term.casefold().replace("ё", "е")[:4])
         for term in matched_terms
         if normalized.find(term.casefold().replace("ё", "е")[:4]) >= 0),
        0,
    )
    start = max(0, first - 100)
    end = min(len(text), start + limit)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


class SemanticSearchService:
    def __init__(self, provider: CourtProvider, planner: QueryPlanner) -> None:
        self._provider = provider
        self._planner = planner

    async def search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        plan = request.plan or await self._planner.plan(request.description)
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

    async def search_with_evidence(
        self, request: EvidenceSearchRequest
    ) -> EvidenceSearchResponse:
        retrieved = await self.search(request)
        checked: list[EvidenceCase] = []
        for item in retrieved.items[:request.max_cases]:
            try:
                extracted = await extract_preferred_document_text(item.case, self._provider)
            except (PreferredDocumentNotFoundError, PdfExtractionError, CourtProviderError) as exc:
                checked.append(EvidenceCase(
                    case=item.case,
                    matched_queries=item.matched_queries,
                    text_error=str(exc),
                ))
                continue
            matched = [term for term in retrieved.plan.must_have if _term_present(extracted.text, term)]
            missing = [term for term in retrieved.plan.must_have if term not in matched]
            excluded = [term for term in retrieved.plan.exclude if _term_present(extracted.text, term)]
            coverage = len(matched) / len(retrieved.plan.must_have) if retrieved.plan.must_have else 0.0
            checked.append(EvidenceCase(
                case=item.case,
                matched_queries=item.matched_queries,
                excerpt=_excerpt(extracted.text, matched),
                matched_terms=matched,
                missing_terms=missing,
                excluded_terms=excluded,
                coverage=coverage,
            ))
        checked.sort(key=lambda item: (item.coverage, -len(item.excluded_terms)), reverse=True)
        return EvidenceSearchResponse(
            plan=retrieved.plan,
            document_count=retrieved.document_count,
            case_count=retrieved.case_count,
            cases_checked=len(checked),
            items=checked,
        )
