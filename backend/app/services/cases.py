from collections import defaultdict
from datetime import date

from ..models import (
    CaseSearchResult,
    CourtCase,
    CourtDocument,
    DocumentSearchParams,
)
from ..providers.base import CourtProvider


class CaseAggregationService:
    def __init__(self, provider: CourtProvider) -> None:
        self._provider = provider

    async def search_cases(
        self,
        params: DocumentSearchParams,
        *,
        max_pages: int = 3,
    ) -> CaseSearchResult:
        first_result = await self._provider.search_documents(params)

        documents = list(first_result.items)
        pages_fetched = 1
        source_pages = first_result.pages
        source_document_count = first_result.count

        if source_pages > params.page and max_pages > 1:
            last_page = min(source_pages, params.page + max_pages - 1)
            for page in range(params.page + 1, last_page + 1):
                page_params = params.model_copy(update={"page": page})
                page_result = await self._provider.search_documents(page_params)
                documents.extend(page_result.items)
                pages_fetched += 1
                source_pages = max(source_pages, page_result.pages)
                source_document_count = max(source_document_count, page_result.count)

        unique_documents = self._deduplicate_documents(documents)
        cases = self._group_documents_by_case(unique_documents)

        return CaseSearchResult(
            source_document_count=source_document_count,
            source_pages=source_pages,
            pages_fetched=pages_fetched,
            unique_document_count=len(unique_documents),
            case_count=len(cases),
            items=cases,
        )

    @staticmethod
    def _deduplicate_documents(
        documents: list[CourtDocument],
    ) -> list[CourtDocument]:
        unique: dict[str, CourtDocument] = {}
        for document in documents:
            unique.setdefault(document.document_id, document)
        return list(unique.values())

    def _group_documents_by_case(
        self,
        documents: list[CourtDocument],
    ) -> list[CourtCase]:
        grouped: dict[str, list[CourtDocument]] = defaultdict(list)

        for document in documents:
            group_key = (
                document.case_id
                or document.case_number
                or f"document:{document.document_id}"
            )
            grouped[group_key].append(document)

        cases = [self._build_case(case_documents) for case_documents in grouped.values()]
        return sorted(
            cases,
            key=lambda case: (
                case.latest_document_date is not None,
                case.latest_document_date or date.min,
                case.case_number,
            ),
            reverse=True,
        )

    def _build_case(self, documents: list[CourtDocument]) -> CourtCase:
        sorted_documents = sorted(
            documents,
            key=self._document_sort_key,
            reverse=True,
        )

        first = sorted_documents[0]
        dated_documents = [
            document.registration_date
            for document in sorted_documents
            if document.registration_date is not None
        ]
        instance_levels = [
            document.instance_level
            for document in sorted_documents
            if document.instance_level is not None
        ]

        return CourtCase(
            case_id=first.case_id,
            case_number=first.case_number,
            case_url=next(
                (
                    document.case_url
                    for document in sorted_documents
                    if document.case_url
                ),
                None,
            ),
            document_count=len(sorted_documents),
            latest_document_date=max(dated_documents) if dated_documents else None,
            highest_instance_level=max(instance_levels) if instance_levels else None,
            documents=sorted_documents,
        )

    @staticmethod
    def _document_sort_key(document: CourtDocument) -> tuple[bool, int, bool, date, str]:
        return (
            document.instance_level is not None,
            document.instance_level if document.instance_level is not None else -1,
            document.registration_date is not None,
            document.registration_date or date.min,
            document.document_type or "",
        )
