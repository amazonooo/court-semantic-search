import asyncio
from collections import defaultdict
import re
from time import monotonic
import unicodedata

from ..config import Settings, get_settings
from ..llm.base import LlmError, QueryPlanner, RelevanceReranker
from ..models import (
    CourtDocument, CriterionAssessment, CriterionCitation, DocumentSearchParams, EvidenceCase, EvidenceSearchRequest, QueryProgress,
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
from .relevance import RelevanceCandidate, SourceCitation, grounded_model_quote, full_document_passages, lexical_query_overlap, score_textual_relevance
from .recommendations import assess_recommendation
from .search_runtime import (
    BoundedProvider, SearchBudgetExceeded, SearchRuntime, active_diagnostics,
)
from .search_sessions import MAX_DOCUMENTS, get_session, split_task


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
        self._session = None

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
        if request.coverage_mode == 'full':
            return await self._retrieve_full(request, plan, runtime, provider, warnings)
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
                    progress[query].error = type(exc).__name__
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
            retrieval_complete=(len(queries) == len(plan.queries) and
                                all(state.complete for state in progress.values())),
        )

    async def _retrieve_full(self, request, plan, runtime, provider, warnings):
        session = get_session(request, plan, self._provider)
        await session.lock.acquire()
        self._session = session
        runtime.diagnostics.query_progress = list(session.progress.values())
        failed = set()
        page_depth = min(request.max_pages_per_query, self._settings.search_max_pages_per_query)
        while session.pending and runtime.remaining() > 0 and runtime.diagnostics.search_calls < runtime.search_call_limit:
            if len(session.documents) >= MAX_DOCUMENTS:
                warnings.append(f'Достигнут предел хранения {MAX_DOCUMENTS} документов; обход источника не завершён. Уточните период или формулировки.')
                break
            # A failed page is retained for the next batch, not paid for again
            # repeatedly in this batch. Other formulations can still progress.
            task = next((task for task in session.pending if id(task) not in failed), None)
            if task is None:
                break
            session.pending.remove(task)
            query_state = session.progress[task.query]
            if task.query not in runtime.diagnostics.queries_executed:
                runtime.diagnostics.queries_executed.append(task.query)
            params = DocumentSearchParams(text=task.query, page=task.page,
                **{**plan.filters.model_dump(), 'date_from': task.progress.date_from,
                   'date_to': task.progress.date_to})
            try:
                result = await provider.search_documents(params)
                if result.page != task.page:
                    raise CourtProviderError('Источник вернул другую страницу')
            except asyncio.CancelledError:
                # An interrupted HTTP request has not completed this page.
                # Keep it available when the same token is resumed.
                session.pending.appendleft(task)
                raise
            except (CourtProviderAccessError, CourtProviderValidationError):
                session.pending.appendleft(task)
                raise
            except (CourtProviderError, SearchBudgetExceeded) as exc:
                task.progress.status = 'error'
                query_state.error = type(exc).__name__
                failed.add(id(task))
                session.pending.append(task)
                warnings.append(f'Страница по формулировке «{task.query}» не получена; она сохранена для продолжения.')
                continue
            task.progress.status = 'pending'
            query_state.error = None
            period = task.progress
            period.source_document_count = max(period.source_document_count, result.count)
            period.source_pages = max(period.source_pages, result.pages)
            period.pages_fetched.append(result.page)
            query_state.source_document_count = max(query_state.source_document_count, result.count)
            query_state.source_pages = max(query_state.source_pages, result.pages)
            query_state.pages_fetched = sorted(set([*query_state.pages_fetched, result.page]))
            query_state.documents_received += len(result.items)
            candidates = CaseAggregationService._filter_candidate_documents_by_date(result.items, params)
            runtime.diagnostics.filtered_by_date += len(result.items) - len(candidates)
            repeated_page = bool(result.items) and all(d.document_id in task.received for d in result.items)
            task.page_size = max(task.page_size, len(result.items))
            task.received.update(d.document_id for d in result.items)
            for document in candidates:
                session.documents.setdefault(document.document_id, document)
                session.matched.setdefault(document.document_id, set()).add(task.query)
            capped = result.count > max(1, result.pages) * max(1, task.page_size)
            if (result.pages > page_depth or capped) and period.date_from < period.date_to:
                split_task(session, task)
            elif (not result.items and result.count > 0) or repeated_page or len(candidates) != len(result.items) or (
                task.page >= result.pages and period.source_document_count > len(task.received)):
                period.status = 'blocked'
                warnings.append(f'Источник ограничил выдачу за {period.date_from}–{period.date_to}; полный охват не подтверждён. Уточните формулировку.')
            elif task.page < result.pages:
                task.page += 1
                session.pending.append(task)
            else:
                period.status = 'complete'
        for query, progress in session.progress.items():
            progress.complete = (not any(task.query == query for task in session.pending) and
                                 all(period.status in {'complete', 'split'} for period in progress.ranges))
        runtime.diagnostics.pending_ranges = len(session.pending)
        runtime.diagnostics.retrieval_complete = session.retrieval_complete
        if not session.retrieval_complete:
            warnings.append('Обход источника не завершён. Продолжение использует сохранённые страницы и периоды; отсутствие совпадений относится только к проверенной части.')
        cases = CaseAggregationService(provider)._group_documents_by_case(list(session.documents.values()))
        items = [RetrievedCase(case=case, matched_queries=[query for query in plan.queries
            if any(query in session.matched[d.document_id] for d in case.documents)]) for case in cases]
        items.sort(key=lambda item: -len(item.matched_queries))
        return SemanticSearchResponse(plan=plan, query_count=len(runtime.diagnostics.queries_executed),
            document_count=len(session.documents), case_count=len(cases), items=items,
            continuation_token=session.token, retrieval_complete=session.retrieval_complete)

    def _exclude_reviewed_cases(self, retrieved, request, runtime):
        excluded = set(request.excluded_case_ids)
        reviewed = self._session.checked if self._session else {}
        tax_only = bool(re.search(r'налогов\w*\s+(?:спор|орган|инспекц)', request.description, re.I)) and not re.search(
            r'банкрот|несостоятельн', request.description, re.I)
        bankruptcy = re.compile(r'банкрот|несостоятельн|реестр.*кредитор|требовани\w*.*реестр|ст\.?\s*60\s*ФЗ', re.I)
        candidates = []
        for item in retrieved.items:
            keys = {item.case.case_id, item.case.case_number} - {''}
            if keys & excluded:
                runtime.diagnostics.feedback_cases_skipped += 1
            elif any(key in reviewed and reviewed[key] == (
                document_for_role(item.case, 'search_evidence').document_id
                if document_for_role(item.case, 'search_evidence') else '') for key in keys):
                runtime.diagnostics.previously_checked_cases_skipped += 1
            elif tax_only and item.case.documents and all(
                bankruptcy.search(' '.join([d.document_type or '', *d.content_types]))
                for d in item.case.documents):
                # Narrowly scoped to explicit tax queries. A norm or an IFNS
                # participant does not turn a bankruptcy act into a tax case.
                runtime.diagnostics.wrong_subject_cases_skipped += 1
            else:
                candidates.append(item)
        retrieved.items = candidates

    def _release_session(self):
        if self._session:
            self._session.lock.release()
            self._session = None

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
            if self._session:
                matched_queries = set(item.matched_queries)
                for document in documents:
                    self._session.documents.setdefault(document.document_id, document)
                    self._session.matched.setdefault(document.document_id, set()).update(matched_queries)
            if result.pages > 1:
                warnings.append(f'История дела {item.case.case_number} получена частично: 1 из {result.pages} страниц.')

    async def _add_reference_case(self, retrieved, request, runtime, provider, warnings):
        if not request.reference_case_number:
            return
        if self._session and self._session.reference_loaded:
            for item in retrieved.items:
                item.is_reference = item.case.case_id in self._session.reference_case_ids
            retrieved.items.sort(key=lambda item: not item.is_reference)
            return
        try:
            result = await provider.search_documents(DocumentSearchParams(caseNumber=request.reference_case_number))
        except (CourtProviderAccessError, CourtProviderValidationError):
            raise
        except (CourtProviderError, SearchBudgetExceeded):
            warnings.append('Образец по номеру дела не получен в этой порции; он не считается найденным по описанию.')
            return
        documents = [d for d in result.items if d.case_number.upper().replace('A', 'А') ==
                     request.reference_case_number.strip().upper().replace('A', 'А')]
        if not documents:
            warnings.append('Источник не вернул документы образца по указанному номеру. Реквизиты требуют проверки.')
            return
        if self._session:
            self._session.reference_loaded = True
            for d in documents:
                self._session.documents.setdefault(d.document_id, d)
                self._session.matched.setdefault(d.document_id, set())
                self._session.reference_case_ids.add(d.case_id)
        existing = {item.case.case_id: item for item in retrieved.items}
        for case in CaseAggregationService(provider)._group_documents_by_case(documents):
            if case.case_id in existing:
                item = existing[case.case_id]
                item.case = CaseAggregationService(provider)._build_case(
                    CaseAggregationService._deduplicate_documents([*item.case.documents, *case.documents]))
                item.is_reference = True
            else:
                retrieved.items.append(RetrievedCase(case=case, matched_queries=[], is_reference=True))
        retrieved.document_count = len({d.document_id for item in retrieved.items for d in item.case.documents})
        retrieved.case_count = len(retrieved.items)
        retrieved.items.sort(key=lambda item: not item.is_reference)
        warnings.append('Образец добавлен по номеру для проверки понимания условий; это не подтверждение его нахождения по описанию.')
        if result.pages > 1:
            warnings.append(f'История образца получена частично: 1 из {result.pages} страниц.')

    async def search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        runtime = SearchRuntime(self._settings, max_search_calls=request.max_search_calls)
        token = active_diagnostics.set(runtime.diagnostics)
        warnings = []
        try:
            result = await self._retrieve(request, runtime, BoundedProvider(self._provider, runtime), warnings)
            self._exclude_reviewed_cases(result, request, runtime)
            runtime.diagnostics.retrieval_complete = result.retrieval_complete
            if self._session and not self._session.pending:
                result.continuation_token = None
            result.warnings = list(dict.fromkeys(warnings))
            result.partial = bool(warnings)
            result.diagnostics = runtime.diagnostics
            return result
        finally:
            self._release_session()
            runtime.finish()
            active_diagnostics.reset(token)

    async def search_with_evidence(self, request: EvidenceSearchRequest) -> EvidenceSearchResponse:
        runtime = SearchRuntime(self._settings, max_search_calls=request.max_search_calls)
        token = active_diagnostics.set(runtime.diagnostics)
        provider = BoundedProvider(self._provider, runtime)
        warnings: list[str] = []
        reranker = self._reranker if request.semantic_reranking else None
        runtime.diagnostics.verification_mode = 'semantic' if reranker else 'textual'
        if request.semantic_reranking and reranker is None:
            warnings.append('Смысловая проверка запрошена, но модель для неё не настроена.')
        try:
            call_limit = runtime.search_call_limit
            if request.reference_case_number and not request.continuation_token:
                runtime.search_call_limit = max(0, call_limit - 1)
            try:
                retrieved = await self._retrieve(request, runtime, provider, warnings)
            finally:
                runtime.search_call_limit = call_limit
            await self._add_reference_case(retrieved, request, runtime, provider, warnings)
            self._exclude_reviewed_cases(retrieved, request, runtime)
            if runtime.diagnostics.wrong_subject_cases_skipped:
                warnings.append(f'По метаданным исключены акты о банкротстве и реестре кредиторов: {runtime.diagnostics.wrong_subject_cases_skipped}; запрошен налоговый предмет спора.')
            runtime.diagnostics.retrieval_complete = retrieved.retrieval_complete
            await self._expand_substantive_cases(retrieved, request, runtime, provider, warnings)
            # Query-return evidence is only a proxy before downloading the PDF.
            # The actual approved-word count is computed across its complete text.
            def retrieval_word_count(item):
                return lexical_query_overlap(' '.join(item.matched_queries), retrieved.plan.queries)[0]
            # Known merits decisions take PDF slots ahead of procedural-only cases.
            retrieved.items.sort(key=lambda item: (
                document_for_role(item.case, 'search_evidence') is None,
                not any(CaseAggregationService.document_status(d) == 'substantive' for d in item.case.documents),
                not item.is_reference, -len(retrieval_word_count(item)),
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
            shortlist = candidate_pool[:max_cases]
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
                        lexical_words, lexical_total = await asyncio.wait_for(asyncio.to_thread(
                            lexical_query_overlap, extracted.text, retrieved.plan.queries), runtime.remaining())
                        if reranker:
                            model_inputs[extracted.document.document_id] = full_document_passages(extracted.text)
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
                        lexical_matched_words=lexical_words, lexical_word_count=len(lexical_words),
                        lexical_total_words=lexical_total,
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
                result.is_reference = any(item.is_reference and item.case.case_id == result.case.case_id
                                          for item in shortlist)
                if result.text_error:
                    warnings.append('Часть PDF не удалось проверить; эти дела не подтверждены.')
            # Lexical admission precedes any semantic question answering. This
            # ordering uses the full read text, not must-have phrase proximity.
            checked.sort(key=lambda item: -item.lexical_word_count)
            if reranker:
                runtime.begin_relevance()
                candidates = []
                for index, item in enumerate(checked):
                    if item.evidence_document and item.evidence_document.document_id in model_inputs:
                        candidates.append(RelevanceCandidate(
                            key=str(index), case_number=item.case.case_number,
                            passages=model_inputs[item.evidence_document.document_id],
                            must_have=tuple(retrieved.plan.must_have), full_text=True,
                        ))
                if candidates and runtime.remaining() > 0:
                    start = monotonic()
                    try:
                        model_budget = min(self._settings.relevance_timeout_seconds, runtime.remaining())
                        judgments = await asyncio.wait_for(
                            reranker.judge_with_budget(request.description, candidates, model_budget),
                            model_budget,
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
                            item.relevance_quote = passage[:700] if passage else None
                            item.relevance_status = 'semantic' if judgment.score is not None else 'insufficient'
                            item.semantic_criteria = []
                            seen_terms = set()
                            for criterion in judgment.criteria:
                                if criterion.term not in retrieved.plan.must_have or criterion.term in seen_terms:
                                    continue
                                seen_terms.add(criterion.term)
                                references = criterion.citations or (
                                    (SourceCitation(criterion.passage_id, criterion.quote),)
                                    if criterion.passage_id and criterion.quote else ())
                                citations = []
                                all_grounded = bool(references)
                                for reference in references:
                                    source = case_passages.get(reference.passage_id)
                                    quote, highlights = grounded_model_quote(source, reference.quote) if source else (None, [])
                                    if not highlights:
                                        all_grounded = False
                                    if quote and highlights:
                                        citations.append(CriterionCitation(passage_id=reference.passage_id,
                                            quote=quote, highlights=highlights))
                                status = criterion.status
                                if status in {'supported', 'contradicted'} and not all_grounded:
                                    status = 'unclear'
                                first = citations[0] if citations else None
                                item.semantic_criteria.append(CriterionAssessment(
                                    term=criterion.term, status=status, reason=criterion.reason or None,
                                    citations=citations, quote=first.quote if first else None,
                                    highlights=first.highlights if first else [],
                                ))
                            item.analysis_complete = seen_terms == set(retrieved.plan.must_have)
                            item.analysis_scope = 'full_text' if item.analysis_complete else 'incomplete'
                            if not item.analysis_complete:
                                incomplete_criteria = True
                        if len(seen_keys) < len(candidates):
                            warnings.append('Анализ полного текста части дел не завершён: лимит времени, контекста или сбой модели. Эти дела не подтверждены и доступны для повторной проверки.')
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
                if self._session and not item.text_error and (item.evidence_document or item.document_status == 'procedural') and (not request.semantic_reranking or item.analysis_complete):
                    self._session.checked[item.case.case_id or item.case.case_number] = (
                        item.evidence_document.document_id if item.evidence_document else '')
            verification_complete = len(checked) == len(candidate_pool) and all(
                not item.text_error and (not request.semantic_reranking or item.analysis_complete) for item in checked)
            counts = {status: sum(item.recommendation_status == status for item in checked)
                      for status in ('confirmed', 'related', 'unverified', 'not_recommended')}
            if not counts['confirmed']:
                warnings.append('Подтверждённых совпадений по совокупности обязательных условий нет в проверенной части выдачи.')
            order = {'confirmed': 0, 'related': 1, 'unverified': 2, 'not_recommended': 3}
            checked.sort(key=lambda item: (
                order[item.recommendation_status],
                item.relevance_score is None, -(item.relevance_score or 0),
                item.relevance_status != 'semantic',
                -item.lexical_word_count, -len(item.matched_queries), -(item.coverage or 0),
            ))
            return EvidenceSearchResponse(
                plan=retrieved.plan, document_count=retrieved.document_count,
                case_count=retrieved.case_count, cases_attempted=attempted,
                confirmed_count=counts['confirmed'], related_count=counts['related'],
                unverified_count=counts['unverified'], rejected_count=counts['not_recommended'],
                cases_checked=runtime.diagnostics.pdf_checked, items=checked,
                partial=bool(warnings), warnings=list(dict.fromkeys(warnings)),
                diagnostics=runtime.diagnostics,
                continuation_token=(retrieved.continuation_token
                    if self._session and (self._session.pending or not verification_complete) else None),
                retrieval_complete=retrieved.retrieval_complete,
                verification_complete=verification_complete,
            )
        finally:
            self._release_session()
            runtime.finish()
            active_diagnostics.reset(token)
