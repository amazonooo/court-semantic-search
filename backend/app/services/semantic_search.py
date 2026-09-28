import asyncio
from collections import defaultdict
from time import monotonic

from ..config import Settings, get_settings
from ..llm.base import LlmError, QueryPlanner, RelevanceReranker
from ..models import (
    CourtDocument, CriterionAssessment, DocumentSearchParams, EvidenceCase, EvidenceSearchRequest,
    EvidenceSearchResponse, RetrievedCase, SearchPlan, SemanticSearchRequest,
    SemanticSearchResponse,
)
from ..providers.base import (
    CourtProvider, CourtProviderAccessError, CourtProviderError,
    CourtProviderValidationError,
)
from .case_text import PreferredDocumentNotFoundError, document_for_role, extract_case_document_text
from .cases import CaseAggregationService
from .evidence import find_evidence
from .pdf import PdfExtractionError
from .relevance import RelevanceCandidate, grounded_model_quote, model_passages, score_textual_relevance
from .search_runtime import (
    BoundedProvider, SearchBudgetExceeded, SearchRuntime, active_diagnostics,
)


class SemanticSearchService:
    def __init__(self, provider: CourtProvider, planner: QueryPlanner | None,
                 settings: Settings | None = None,
                 *, reranker: RelevanceReranker | None = None) -> None:
        self._provider = provider
        self._planner = planner
        self._settings = settings or get_settings()
        self._reranker = reranker

    async def _plan(self, request: SemanticSearchRequest, runtime: SearchRuntime) -> SearchPlan:
        start = monotonic()
        try:
            if request.plan is not None:
                return request.plan.model_copy(deep=True)
            if self._planner is None:
                raise LlmError("A search plan or configured LLM provider is required")
            try:
                return await asyncio.wait_for(
                    self._planner.plan(request.description),
                    min(self._settings.plan_timeout_seconds, runtime.remaining()),
                )
            except TimeoutError as exc:
                raise LlmError("Search planning exceeded its time budget") from exc
        finally:
            runtime.diagnostics.stage_seconds['planning'] = round(monotonic() - start, 3)

    async def _retrieve(self, request, runtime, provider, warnings):
        plan = await self._plan(request, runtime)
        if request.filters is not None:
            plan.filters = plan.filters.model_copy(update=request.filters.model_dump(exclude_unset=True))
            # Revalidate after combining user filters with model-proposed filters.
            try:
                plan.filters = type(plan.filters).model_validate(plan.filters.model_dump())
            except ValueError as exc:
                raise CourtProviderValidationError("User filters conflict with the search plan") from exc
        runtime.begin_retrieval()
        unique: dict[str, CourtDocument] = {}
        matched: dict[str, set[str]] = defaultdict(set)
        first_seen: dict[str, int] = {}
        max_pages = min(request.max_pages_per_query, self._settings.search_max_pages_per_query)
        queries = plan.queries[:min(request.max_queries, self._settings.search_max_queries)]
        if len(queries) < len(plan.queries):
            warnings.append(
                f'Отправлено к источнику {len(queries)} из {len(plan.queries)} формулировок плана; '
                'остальные сохранены как варианты и не расходовали лимит Parser API.'
            )
        if max_pages < request.max_pages_per_query:
            warnings.append('Просмотрены не все запрошенные страницы из-за серверного ограничения.')
        for query in queries:
            if runtime.remaining() <= 0:
                warnings.append('Лимит времени поиска исчерпан; возвращены доступные кандидаты.')
                break
            runtime.diagnostics.queries_executed.append(query)
            for page in range(1, max_pages + 1):
                params = DocumentSearchParams(text=query, page=page, **plan.filters.model_dump())
                try:
                    result = await provider.search_documents(params)
                except (CourtProviderAccessError, CourtProviderValidationError):
                    raise
                except (CourtProviderError, SearchBudgetExceeded) as exc:
                    warnings.append(f'Источник не завершил поиск по формулировке {len(runtime.diagnostics.queries_executed)}: {type(exc).__name__}.')
                    break
                candidates = CaseAggregationService._filter_candidate_documents_by_date(result.items, params)
                runtime.diagnostics.filtered_by_date += len(result.items) - len(candidates)
                for document in candidates:
                    first_seen.setdefault(document.document_id, len(first_seen))
                    unique.setdefault(document.document_id, document)
                    matched[document.document_id].add(query)
                if page >= result.pages or not result.items:
                    break
                if page == max_pages and result.pages > max_pages:
                    warnings.append('Просмотрены только первые страницы выдачи источника.')
        cases = CaseAggregationService(provider)._group_documents_by_case(list(unique.values()))
        items = [RetrievedCase(
            case=case,
            matched_queries=[query for query in plan.queries
                             if any(query in matched[d.document_id] for d in case.documents)],
        ) for case in cases]
        # Repeated acts in long cases must not boost a case merely by volume.
        items.sort(key=lambda item: (
            -len(item.matched_queries),
            min(first_seen[d.document_id] for d in item.case.documents),
        ))
        return SemanticSearchResponse(
            plan=plan, query_count=len(runtime.diagnostics.queries_executed),
            document_count=len(unique), case_count=len(cases), items=items,
        )

    async def search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        runtime = SearchRuntime(self._settings)
        token = active_diagnostics.set(runtime.diagnostics)
        warnings = []
        try:
            result = await self._retrieve(request, runtime, BoundedProvider(self._provider, runtime), warnings)
            result.warnings = list(dict.fromkeys(warnings))
            result.partial = bool(warnings)
            result.diagnostics = runtime.diagnostics
            return result
        finally:
            runtime.finish()
            active_diagnostics.reset(token)

    async def search_with_evidence(self, request: EvidenceSearchRequest) -> EvidenceSearchResponse:
        runtime = SearchRuntime(self._settings)
        token = active_diagnostics.set(runtime.diagnostics)
        provider = BoundedProvider(self._provider, runtime)
        warnings: list[str] = []
        reranker = self._reranker if request.semantic_reranking else None
        try:
            retrieved = await self._retrieve(request, runtime, provider, warnings)
            runtime.begin_evidence(reserve_seconds=(
                min(self._settings.relevance_timeout_seconds, runtime.remaining() / 3)
                if reranker else 0
            ))
            checked: list[EvidenceCase] = []
            model_inputs: dict[str, dict[str, str]] = {}
            max_cases = min(request.max_cases, self._settings.search_max_cases,
                            self._settings.search_max_pdf_downloads)
            shortlist = []
            selected = set()
            def case_key(candidate):
                return (candidate.case.case_id or candidate.case.case_number or
                        candidate.case.documents[0].document_id)

            if retrieved.items:
                shortlist.append(retrieved.items[0])
                selected.add(case_key(retrieved.items[0]))
            # Include candidates unique to each formulation when available;
            # repeated broad hits must not consume every PDF slot.
            for query in runtime.diagnostics.queries_executed:
                candidate = next((item for item in retrieved.items
                                  if item.matched_queries == [query] and case_key(item) not in selected), None)
                if candidate is None:
                    candidate = next((item for item in retrieved.items
                                      if query in item.matched_queries and case_key(item) not in selected), None)
                if candidate is not None and len(shortlist) < max_cases:
                    shortlist.append(candidate)
                    selected.add(case_key(candidate))
            for candidate in retrieved.items:
                if len(shortlist) >= max_cases:
                    break
                if case_key(candidate) not in selected:
                    shortlist.append(candidate)
                    selected.add(case_key(candidate))
            attempted = 0
            semaphore = asyncio.Semaphore(3)

            async def check_one(item):
                nonlocal attempted
                async with semaphore:
                    source_document = document_for_role(item.case, 'factual_base')
                    if runtime.remaining() <= 0:
                        return EvidenceCase(case=item.case, matched_queries=item.matched_queries,
                                            source_document=source_document,
                                            text_error='Лимит времени проверки PDF исчерпан')
                    attempted += 1
                    try:
                        extracted = await asyncio.wait_for(
                            extract_case_document_text(item.case, provider, role='factual_base'),
                            runtime.remaining(),
                        )
                        if not extracted.text.strip():
                            raise PdfExtractionError('PDF не содержит извлекаемого текста; требуется OCR')
                    except (CourtProviderAccessError, CourtProviderValidationError) as exc:
                        return exc
                    except SearchBudgetExceeded:
                        return EvidenceCase(case=item.case, matched_queries=item.matched_queries,
                                            source_document=source_document,
                                            text_error='Источник не успел загрузить PDF за отведённое время')
                    except TimeoutError:
                        return EvidenceCase(case=item.case, matched_queries=item.matched_queries,
                                            source_document=source_document,
                                            text_error='Время проверки PDF истекло')
                    except (PreferredDocumentNotFoundError, PdfExtractionError,
                            CourtProviderError) as exc:
                        return EvidenceCase(case=item.case, matched_queries=item.matched_queries,
                                            source_document=source_document,
                                            text_error=str(exc))
                    runtime.diagnostics.pdf_checked += 1
                    start = monotonic()
                    try:
                        matches = await asyncio.wait_for(asyncio.to_thread(
                            find_evidence, extracted.text,
                            list(dict.fromkeys([*retrieved.plan.must_have, *retrieved.plan.exclude])),
                        ), runtime.remaining())
                        relevance = await asyncio.wait_for(asyncio.to_thread(
                            score_textual_relevance, extracted.text, request.description,
                            retrieved.plan.must_have, retrieved.plan.exclude, matches,
                        ), runtime.remaining())
                        if reranker:
                            model_inputs[extracted.document.document_id] = await asyncio.wait_for(
                                asyncio.to_thread(model_passages, extracted.text, request.description,
                                                  retrieved.plan.must_have, matches), runtime.remaining())
                    except TimeoutError:
                        return EvidenceCase(
                            case=item.case, matched_queries=item.matched_queries,
                            source_document=source_document,
                            evidence_document=extracted.document,
                            text_error='Проверка признаков превысила общий лимит времени',
                        )
                    finally:
                        runtime.diagnostics.stage_seconds['matching'] = round(
                            runtime.diagnostics.stage_seconds.get('matching', 0) + monotonic() - start, 3)
                    evidence = [match for match in matches if match.term in retrieved.plan.must_have]
                    excluded = [match for match in matches if match.term in retrieved.plan.exclude]
                    matched_terms = [match.term for match in evidence]
                    missing = [term for term in retrieved.plan.must_have if term not in matched_terms]
                    coverage = (len(evidence) / len(retrieved.plan.must_have)
                                if retrieved.plan.must_have else None)
                    status = ('excluded' if excluded else
                              'terms_found' if evidence and not missing else
                              'partial_terms' if evidence else 'no_terms')
                    return EvidenceCase(
                        case=item.case, matched_queries=item.matched_queries,
                        source_document=source_document,
                        evidence_document=extracted.document, evidence=evidence, exclusion_evidence=excluded,
                        excerpt=evidence[0].quote if evidence else None,
                        matched_terms=matched_terms, missing_terms=missing,
                        excluded_terms=[match.term for match in excluded],
                        coverage=coverage, verification_status=status,
                        relevance_score=relevance.score, relevance_reason=relevance.reason,
                        relevance_quote=relevance.quote, relevance_status='textual',
                    )

            results = await asyncio.gather(*(check_one(item) for item in shortlist))
            for result in results:
                if isinstance(result, (CourtProviderAccessError, CourtProviderValidationError)):
                    raise result
                checked.append(result)
                if result.text_error:
                    warnings.append('Часть PDF не удалось проверить; эти дела не подтверждены.')
            if reranker:
                runtime.begin_relevance()
                candidates = []
                for index, item in enumerate(checked):
                    if item.evidence_document and item.evidence_document.document_id in model_inputs:
                        candidates.append(RelevanceCandidate(
                            key=str(index), case_number=item.case.case_number,
                            passages=model_inputs[item.evidence_document.document_id],
                            must_have=tuple(retrieved.plan.must_have),
                        ))
                if candidates and runtime.remaining() > 0:
                    start = monotonic()
                    try:
                        judgments = await asyncio.wait_for(
                            reranker.judge(request.description, candidates),
                            min(self._settings.relevance_timeout_seconds, runtime.remaining()),
                        )
                    except (LlmError, TimeoutError):
                        warnings.append('Смысловая проверка GigaChat не завершилась; показан предварительный текстовый ранг.')
                    else:
                        incomplete_criteria = False
                        for judgment in judgments:
                            item = checked[int(judgment.key)]
                            case_passages = next((candidate.passages for candidate in candidates
                                                  if candidate.key == judgment.key), {})
                            passage = case_passages.get(judgment.passage_id)
                            item.relevance_score = judgment.score
                            item.relevance_reason = judgment.reason
                            item.relevance_quote = passage
                            item.relevance_status = 'semantic' if judgment.score is not None else 'insufficient'
                            item.semantic_criteria = []
                            for criterion in judgment.criteria:
                                source = case_passages.get(criterion.passage_id) if criterion.passage_id else None
                                quote, highlights = grounded_model_quote(source, criterion.quote) if source else (None, [])
                                item.semantic_criteria.append(CriterionAssessment(
                                    term=criterion.term, status=criterion.status,
                                    quote=quote, highlights=highlights,
                                ))
                            if len(item.semantic_criteria) < len(retrieved.plan.must_have):
                                incomplete_criteria = True
                        if len(judgments) < len(candidates):
                            warnings.append('Часть дел не получила смысловую оценку; для них сохранён текстовый ранг.')
                        if incomplete_criteria:
                            warnings.append('GigaChat не оценил часть обязательных признаков; они отмечены как непроверенные моделью.')
                    finally:
                        runtime.diagnostics.stage_seconds['relevance'] = round(monotonic() - start, 3)
                elif candidates:
                    warnings.append('Лимит времени смысловой проверки исчерпан; показан предварительный текстовый ранг.')
            if attempted < retrieved.case_count:
                warnings.append(f'Проверка ограничена: рассмотрено {attempted} из {retrieved.case_count} дел.')
            if checked and all(item.relevance_score is None or item.relevance_score <= 1 for item in checked):
                warnings.append('Среди прочитанных актов не найдено убедительного текстового сходства с описанной ситуацией.')
            checked.sort(key=lambda item: (
                item.relevance_score is None, -(item.relevance_score or 0),
                item.relevance_status != 'semantic',
                -len(item.matched_queries), -(item.coverage or 0),
            ))
            return EvidenceSearchResponse(
                plan=retrieved.plan, document_count=retrieved.document_count,
                case_count=retrieved.case_count, cases_attempted=attempted,
                cases_checked=runtime.diagnostics.pdf_checked, items=checked,
                partial=bool(warnings), warnings=list(dict.fromkeys(warnings)),
                diagnostics=runtime.diagnostics,
            )
        finally:
            runtime.finish()
            active_diagnostics.reset(token)
