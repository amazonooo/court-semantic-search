from collections import defaultdict
from datetime import date

from ..models import (
    CaseSearchResult,
    CourtCase,
    CourtDocument,
    DocumentSearchParams,
)
from ..providers.base import CourtProvider


_SUBSTANTIVE_DOCUMENT_TYPES = (
    "решение",
    "постановление",
    "судебный приказ",
)
_SUBSTANTIVE_CONTENT_MARKERS = (
    "удовлетворить",
    "отказать",
    "взыскать",
    "признать",
    "отменить",
    "изменить",
    "оставить без изменения",
    "жалобу без удовлетворения",
    "утвердить мировое соглашение",
)
_PROCEDURAL_CONTENT_MARKERS = (
    "принять к производству",
    "назначить дело",
    "назначить судебное разбирательство",
    "отложить судебное разбирательство",
    "отложение судебного разбирательства",
    "объявить перерыв",
    "оставить без движения",
    "возвратить апелляционную жалобу",
    "возвращение апелляционной жалобы",
    "выдать исполнительный лист",
    "исправить опечатку",
)
_PROCEDURAL_TYPE_MARKERS = (
    "определени",
    "приняти",
    "назначени",
    "отложени",
    "возвращени",
    "оставлени без движени",
    "выдач исполнительного листа",
    "исправлени опечат",
)


class CaseAggregationService:
    def __init__(self, provider: CourtProvider) -> None:
        self._provider = provider

    async def search_cases(
        self,
        params: DocumentSearchParams,
        *,
        max_pages: int = 3,
        expand_cases: bool = False,
        max_cases_to_expand: int = 10,
        max_case_pages: int = 3,
    ) -> CaseSearchResult:
        (
            documents,
            source_document_count,
            source_pages,
            pages_fetched,
        ) = await self._search_pages(params, max_pages=max_pages)

        raw_candidate_documents = self._deduplicate_documents(documents)
        candidate_documents = self._filter_candidate_documents_by_date(
            raw_candidate_documents,
            params,
        )
        filtered_out_by_date = len(raw_candidate_documents) - len(candidate_documents)
        candidate_cases = self._group_documents_by_case(candidate_documents)

        all_documents = list(candidate_documents)
        case_expansion_pages_fetched = 0
        expanded_case_count = 0

        if expand_cases:
            for case in candidate_cases[:max_cases_to_expand]:
                if not case.case_number:
                    continue

                case_params = DocumentSearchParams(
                    caseNumber=case.case_number,
                    page=1,
                )
                (
                    expanded_documents,
                    _expanded_source_count,
                    _expanded_source_pages,
                    expanded_pages_fetched,
                ) = await self._search_pages(
                    case_params,
                    max_pages=max_case_pages,
                )
                case_expansion_pages_fetched += expanded_pages_fetched

                # Parser API cannot search by CaseId directly. We therefore use the
                # base case number only to retrieve candidates, then keep documents
                # that belong to the exact canonical CaseId discovered initially.
                if case.case_id:
                    matching_documents = [
                        document
                        for document in expanded_documents
                        if document.case_id == case.case_id
                    ]
                else:
                    matching_documents = [
                        document
                        for document in expanded_documents
                        if document.case_number == case.case_number
                    ]

                # Date limits are a hard filter for entering the candidate set, not
                # for the procedural history of a case that already matched. Keeping
                # the full history lets later stages inspect first instance, appeal
                # and cassation even when those acts fall outside the user's period.
                if matching_documents:
                    expanded_case_count += 1
                    all_documents.extend(matching_documents)

        unique_documents = self._deduplicate_documents(all_documents)
        cases = self._group_documents_by_case(unique_documents)

        return CaseSearchResult(
            source_document_count=source_document_count,
            source_pages=source_pages,
            pages_fetched=pages_fetched,
            candidate_unique_document_count=len(candidate_documents),
            filtered_out_by_date=filtered_out_by_date,
            case_expansion_pages_fetched=case_expansion_pages_fetched,
            expanded_case_count=expanded_case_count,
            unique_document_count=len(unique_documents),
            case_count=len(cases),
            items=cases,
        )

    async def _search_pages(
        self,
        params: DocumentSearchParams,
        *,
        max_pages: int,
    ) -> tuple[list[CourtDocument], int, int, int]:
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

        return (
            documents,
            source_document_count,
            source_pages,
            pages_fetched,
        )

    @staticmethod
    def _filter_candidate_documents_by_date(
        documents: list[CourtDocument],
        params: DocumentSearchParams,
    ) -> list[CourtDocument]:
        if params.date_from is None and params.date_to is None:
            return documents

        filtered: list[CourtDocument] = []
        for document in documents:
            document_date = document.registration_date
            if document_date is None:
                # With an explicit user period, an unknown date cannot be verified
                # and therefore cannot safely enter the candidate set.
                continue
            if params.date_from is not None and document_date < params.date_from:
                continue
            if params.date_to is not None and document_date > params.date_to:
                continue
            filtered.append(document)
        return filtered

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

        first_instance_documents = [
            document for document in sorted_documents if document.instance_level == 1
        ]
        appellate_documents = [
            document for document in sorted_documents if document.instance_level == 2
        ]
        cassation_documents = [
            document
            for document in sorted_documents
            if document.instance_level is not None and document.instance_level >= 3
        ]
        procedural_documents = [
            document
            for document in sorted_documents
            if self._is_procedural(document)
        ]
        substantive_documents = [
            document
            for document in sorted_documents
            if self._is_substantive(document)
        ]
        factual_base_document = self._select_factual_base_document(
            first_instance_documents
        )
        latest_substantive_document = (
            max(substantive_documents, key=self._latest_substantive_sort_key)
            if substantive_documents
            else None
        )
        preferred_document = max(
            sorted_documents,
            key=self._preferred_document_sort_key,
        )

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
            first_instance_documents=first_instance_documents,
            first_instance_document=factual_base_document,
            factual_base_document=factual_base_document,
            appellate_documents=appellate_documents,
            cassation_documents=cassation_documents,
            procedural_documents=procedural_documents,
            latest_substantive_document=latest_substantive_document,
            preferred_document_id=preferred_document.document_id,
            preferred_document=preferred_document,
            documents=sorted_documents,
        )

    @classmethod
    def _select_factual_base_document(
        cls,
        first_instance_documents: list[CourtDocument],
    ) -> CourtDocument | None:
        if not first_instance_documents:
            return None

        non_procedural_documents = [
            document
            for document in first_instance_documents
            if not cls._is_procedural(document)
        ]
        candidates = non_procedural_documents or first_instance_documents
        return max(candidates, key=cls._factual_base_sort_key)

    @classmethod
    def _factual_base_sort_key(
        cls,
        document: CourtDocument,
    ) -> tuple[bool, bool, date, str]:
        return (
            cls._is_substantive(document),
            document.registration_date is not None,
            document.registration_date or date.min,
            document.document_id,
        )

    @staticmethod
    def _latest_substantive_sort_key(
        document: CourtDocument,
    ) -> tuple[bool, date, bool, int, str]:
        return (
            document.registration_date is not None,
            document.registration_date or date.min,
            document.instance_level is not None,
            document.instance_level if document.instance_level is not None else -1,
            document.document_id,
        )

    @staticmethod
    def _is_procedural(document: CourtDocument) -> bool:
        document_type = (document.document_type or "").casefold()
        content = " ".join(document.content_types).casefold()
        return any(marker in content for marker in _PROCEDURAL_CONTENT_MARKERS) or any(
            marker in document_type for marker in _PROCEDURAL_TYPE_MARKERS
        )

    @classmethod
    def _is_substantive(cls, document: CourtDocument) -> bool:
        if cls._is_procedural(document):
            return False

        document_type = (document.document_type or "").casefold()
        content = " ".join(document.content_types).casefold()
        return any(marker in document_type for marker in _SUBSTANTIVE_DOCUMENT_TYPES) or any(
            marker in content for marker in _SUBSTANTIVE_CONTENT_MARKERS
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

    @classmethod
    def _preferred_document_sort_key(
        cls,
        document: CourtDocument,
    ) -> tuple[int, int, bool, date, str]:
        document_type = (document.document_type or "").casefold()
        content = " ".join(document.content_types).casefold()

        type_score = 0
        for marker in _SUBSTANTIVE_DOCUMENT_TYPES:
            if marker in document_type:
                type_score = 40
                break
        if "определение" in document_type:
            type_score = max(type_score, 10)

        content_score = sum(
            25 for marker in _SUBSTANTIVE_CONTENT_MARKERS if marker in content
        )
        content_score -= sum(
            35 for marker in _PROCEDURAL_CONTENT_MARKERS if marker in content
        )

        return (
            type_score + content_score,
            document.instance_level if document.instance_level is not None else -1,
            document.registration_date is not None,
            document.registration_date or date.min,
            document.document_id,
        )
