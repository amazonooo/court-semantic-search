import asyncio
from collections import defaultdict
import re
from time import monotonic
import unicodedata

from ..config import Settings, get_settings
from ..llm.base import LlmError, QueryPlanner, RelevanceReranker
from ..models import (
    CourtDocument, CriterionAssessment, DocumentSearchParams, EvidenceCase, EvidenceSearchRequest, QueryProgress,
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
from .recommendations import assess_recommendation
from .search_runtime import (
    BoundedProvider, SearchBudgetExceeded, SearchRuntime, active_diagnostics,
)


_QUERY_WORD = re.compile(r'[а-яa-z]{4,}', re.I)


def _select_queries(queries: list[str], limit: int, must_have: list[str] | None = None) -> list[str]:
    """Spend a small search budget on different facets of the plan."""
    if len(queries) <= limit:
        return queries[:]
    roots = [
        {word[:5] for word in _QUERY_WORD.findall(
            unicodedata.normalize('NFC', query).casefold().replace('ё', 'е'))}
        for query in queries
    ]
    if must_have:
        common = {'основание', 'подпункт', 'пункт', 'статья', 'кодекса', 'кодекс',
                  'судебная', 'практика', 'решение', 'решения', 'обжалование', 'еаэс'}
        def factual_roots(value):
            return {word[:5] for word in re.findall(r'[а-яa-z]{3,}',
                    unicodedata.normalize('NFC', value).casefold().replace('ё', 'е'))
                    if word not in common}
        facts = [factual_roots(term) for term in must_have]
        topics = [factual_roots(query) for query in queries]
        covered_facts: set[str] = set()
        selected = []
        while len(selected) < limit:
            def quality(index):
                fresh = topics[index] - covered_facts
                gain = sum(len(fresh & fact) / len(fact) for fact in facts if fact)
                coverage = sum(len(topics[index] & fact) / len(fact) for fact in facts if fact)
                return (bool(topics[index]), gain, coverage, -index)
            index = max((i for i in range(len(queries)) if i not in selected), key=quality)
            selected.append(index)
            covered_facts.update(topics[index])
        return [queries[index] for index in selected]
    selected = [0]
    covered = roots[0].copy()
    while len(selected) < limit:
        remaining = (index for index in range(len(queries)) if index not in selected)
        index = max(remaining, key=lambda candidate: (
            len(roots[candidate] - covered) / max(1, len(roots[candidate])),
            len(roots[candidate] - covered),
            -candidate,
        ))
        selected.append(index)
        covered.update(roots[index])
    return [queries[index] for index in selected]


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
        queries = _select_queries(
            plan.queries, min(request.max_queries, self._settings.search_max_queries), plan.must_have
        )
        if len(queries) < len(plan.queries):
            warnings.append(
                f'Отправлено к источнику {len(queries)} из {len(plan.queries)} формулировок плана; '
                'остальные сохранены как варианты и не расходовали лимит Parser API.'
            )
        if max_pages < request.max_pages_per_query:
            warnings.append('Просмотрены не все запрошенные страницы из-за серверного ограничения.')
        progress = {query: QueryProgress(query=query) for query in queries}
        runtime.diagnostics.query_progress = list(progress.values())
        stopped: set[str] = set()
        # Visit every selected formulation before spending calls on deeper pages.
        for page in range(1, max_pages + 1):
            for query in queries:
                if query in stopped:
                    continue
                if runtime.remaining() <= 0 or runtime.diagnostics.search_calls >= runtime.search_call_limit:
                    warnings.append('Лимит времени или операций поиска исчерпан; возвращены доступные кандидаты.')
                    stopped.update(queries)
                    break
                if query not in runtime.diagnostics.queries_executed:
                    runtime.diagnostics.queries_executed.append(query)
                params = DocumentSearchParams(text=query, page=page, **plan.filters.model_dump())
                try:
                    result = await provider.search_documents(params)
                except (CourtProviderAccessError, CourtProviderValidationError):
                    raise
                except (CourtProviderError, SearchBudgetExceeded) as exc:
                    warnings.append(f'Источник не завершил поиск по формулировке {len(runtime.diagnostics.queries_executed)}: {type(exc).__name__}.')
                    stopped.add(query)
                    continue
                state = progress[query]
                state.source_document_count = max(state.source_document_count, result.count)
                state.source_pages = max(state.source_pages, result.pages)
                state.pages_fetched.append(result.page)
                state.documents_received += len(result.items)
                candidates = CaseAggregationService._filter_candidate_documents_by_date(result.items, params)
                runtime.diagnostics.filtered_by_date += len(result.items) - len(candidates)
                for document in candidates:
                    first_seen.setdefault(document.document_id, len(first_seen))
                    unique.setdefault(document.document_id, document)
                    matched[document.document_id].add(query)
                if page >= result.pages or not result.items:
                    state.complete = page >= result.pages
                    stopped.add(query)
        for state in progress.values():
            if state.pages_fetched and not state.complete:
                warnings.append(f'Формулировка «{state.query}»: получено {len(state.pages_fetched)} '
                                f'из {state.source_pages} страниц источника.')
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

    async def _expand_substantive_cases(self, retrieved, request, runtime, provider, warnings):
        expanded = 0
        for item in retrieved.items:
            if expanded >= request.max_case_expansions:
                break
            if any(CaseAggregationService.document_status(d) == 'substantive' for d in item.case.documents):
                continue
            if not item.case.case_number or runtime.remaining() <= 0:
                continue
            if runtime.diagnostics.search_calls >= runtime.search_call_limit:
                warnings.append('Лимит поисков исчерпан; дополнительные акты дел не загружены.')
                break
            expanded += 1
            try:
                params = DocumentSearchParams(caseNumber=item.case.case_number,
                    dateFrom=retrieved.plan.filters.date_from, dateTo=retrieved.plan.filters.date_to)
                result = await provider.search_documents(params)
            except (CourtProviderAccessError, CourtProviderValidationError):
                raise
            except (CourtProviderError, SearchBudgetExceeded):
                warnings.append(f'Не удалось получить дополнительные акты дела {item.case.case_number}.')
                continue
            documents = CaseAggregationService._filter_candidate_documents_by_date(result.items, params)
            documents = [d for d in documents if d.case_id == item.case.case_id and
                         d.case_number == item.case.case_number]
            known = {d.document_id for d in item.case.documents}
            runtime.diagnostics.expansion_documents += sum(d.document_id not in known for d in documents)
            runtime.diagnostics.cases_expanded += 1
            item.case = CaseAggregationService(provider)._build_case(
                CaseAggregationService._deduplicate_documents([*item.case.documents, *documents]))
            if result.pages > 1:
                warnings.append(f'История дела {item.case.case_number} получена частично: 1 из {result.pages} страниц.')

    async def search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        runtime = SearchRuntime(self._settings, max_search_calls=request.max_search_calls)
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
        runtime = SearchRuntime(self._settings, max_search_calls=request.max_search_calls)
        token = active_diagnostics.set(runtime.diagnostics)
        provider = BoundedProvider(self._provider, runtime)
        warnings: list[str] = []
        reranker = self._reranker if request.semantic_reranking else None
        if request.semantic_reranking and reranker is None:
            warnings.append('Смысловая проверка запрошена, но модель для неё не настроена.')
        try:
            retrieved = await self._retrieve(request, runtime, provider, warnings)
            await self._expand_substantive_cases(retrieved, request, runtime, provider, warnings)
            # Known merits decisions take PDF slots ahead of procedural-only cases.
            retrieved.items.sort(key=lambda item: (
                document_for_role(item.case, 'search_evidence') is None,
                not any(CaseAggregationService.document_status(d) == 'substantive' for d in item.case.documents),
            ))
            runtime.begin_evidence(reserve_seconds=(
                min(self._settings.relevance_timeout_seconds, runtime.remaining() / 3)
                if reranker else 0
            ))
            checked: list[EvidenceCase] = []
            model_inputs: dict[str, dict[str, str]] = {}
            max_cases = min(request.max_cases, self._settings.search_max_cases,
                            self._settings.search_max_pdf_downloads)
            candidate_pool = [item for item in retrieved.items
                              if document_for_role(item.case, 'search_evidence') is not None]
            runtime.diagnostics.procedural_cases_skipped = len(retrieved.items) - len(candidate_pool)
            if runtime.diagnostics.procedural_cases_skipped:
                warnings.append(f'Дел только с процессуальными актами: {runtime.diagnostics.procedural_cases_skipped}; '
                                'их PDF не занимают места для проверки обстоятельств.')
            candidate_pool = candidate_pool or retrieved.items
            shortlist = []
            selected = set()
            def case_key(candidate):
                return (candidate.case.case_id or candidate.case.case_number or
                        candidate.case.documents[0].document_id)

            if candidate_pool:
                shortlist.append(candidate_pool[0])
                selected.add(case_key(candidate_pool[0]))
            # Include candidates unique to each formulation when available;
            # repeated broad hits must not consume every PDF slot.
            for query in runtime.diagnostics.queries_executed:
                candidate = next((item for item in candidate_pool
                                  if item.matched_queries == [query] and case_key(item) not in selected), None)
                if candidate is None:
                    candidate = next((item for item in candidate_pool
                                      if query in item.matched_queries and case_key(item) not in selected), None)
                if candidate is not None and len(shortlist) < max_cases:
                    shortlist.append(candidate)
                    selected.add(case_key(candidate))
            for candidate in candidate_pool:
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
                    source_document = document_for_role(item.case, 'search_evidence')
                    if source_document is None:
                        return EvidenceCase(case=item.case, matched_queries=item.matched_queries,
                            source_document=item.case.preferred_document, document_status='procedural')
                    if runtime.remaining() <= 0:
                        return EvidenceCase(case=item.case, matched_queries=item.matched_queries,
                                            source_document=source_document,
                                            text_error='Лимит времени проверки PDF исчерпан')
                    attempted += 1
                    try:
                        extracted = await asyncio.wait_for(
                            extract_case_document_text(item.case, provider, role='search_evidence'),
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
                        document_status=CaseAggregationService.document_status(extracted.document),
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
                        valid_keys = {candidate.key for candidate in candidates}
                        seen_keys = set()
                        for judgment in judgments:
                            if judgment.key not in valid_keys or judgment.key in seen_keys:
                                continue
                            seen_keys.add(judgment.key)
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
                                if criterion.term not in retrieved.plan.must_have:
                                    continue
                                status = criterion.status
                                if status in {'supported', 'contradicted'} and not highlights:
                                    status = 'unclear'
                                item.semantic_criteria.append(CriterionAssessment(
                                    term=criterion.term, status=status,
                                    quote=quote, highlights=highlights,
                                ))
                            if len(item.semantic_criteria) < len(retrieved.plan.must_have):
                                incomplete_criteria = True
                        if len(seen_keys) < len(candidates):
                            warnings.append('Часть дел не получила смысловую оценку; для них сохранён текстовый ранг.')
                        if incomplete_criteria:
                            warnings.append('GigaChat не оценил часть обязательных признаков; они отмечены как непроверенные моделью.')
                    finally:
                        runtime.diagnostics.stage_seconds['relevance'] = round(monotonic() - start, 3)
                elif candidates:
                    warnings.append('Лимит времени смысловой проверки исчерпан; показан предварительный текстовый ранг.')
            if attempted < retrieved.case_count:
                warnings.append(f'Проверка ограничена: начато чтение PDF для {attempted} из {retrieved.case_count} полученных дел.')
            scored = [item for item in checked if item.relevance_score is not None]
            if checked and not scored:
                warnings.append('Получены только процессуальные акты; обстоятельства по существу не проверены.'
                    if all(item.document_status == 'procedural' for item in checked) else
                    'Не удалось прочитать ни один PDF; сходство дел с запросом не оценено.'
                    if runtime.diagnostics.pdf_checked == 0 else
                    'Оценка сходства по полученным PDF не завершилась.'
                )
            elif scored and all(item.relevance_score <= 1 for item in scored):
                warnings.append('Среди прочитанных актов не найдено убедительного текстового сходства с описанной ситуацией.')
            for item in checked:
                assess_recommendation(item, retrieved.plan.must_have)
            counts = {status: sum(item.recommendation_status == status for item in checked)
                      for status in ('confirmed', 'related', 'unverified', 'not_recommended')}
            if not counts['confirmed']:
                warnings.append('Подтверждённых совпадений по совокупности обязательных условий нет в проверенной части выдачи.')
            order = {'confirmed': 0, 'related': 1, 'unverified': 2, 'not_recommended': 3}
            checked.sort(key=lambda item: (
                order[item.recommendation_status],
                item.relevance_score is None, -(item.relevance_score or 0),
                item.relevance_status != 'semantic',
                -len(item.matched_queries), -(item.coverage or 0),
            ))
            return EvidenceSearchResponse(
                plan=retrieved.plan, document_count=retrieved.document_count,
                case_count=retrieved.case_count, cases_attempted=attempted,
                confirmed_count=counts['confirmed'], related_count=counts['related'],
                unverified_count=counts['unverified'], rejected_count=counts['not_recommended'],
                cases_checked=runtime.diagnostics.pdf_checked, items=checked,
                partial=bool(warnings), warnings=list(dict.fromkeys(warnings)),
                diagnostics=runtime.diagnostics,
            )
        finally:
            runtime.finish()
            active_diagnostics.reset(token)
